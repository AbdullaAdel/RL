import serial
import time
import mujoco
import mujoco.viewer
import numpy as np
import threading
import json
import cv2
from scipy.spatial.transform import Rotation as R

# --- CONFIGURATION ---
PORT = "COM3"
BAUDRATE = 1000000
XML_PATH = "scene.xml"
SIM_FPS = 60  # Balanced for Serial and Vision performance

# Camera & Object Config
CAMERA_ID = 0  
ARUCO_SIZE = 0.025      # 25mm Marker (Meters)
CUBE_SIZE_MM = 35.0     # 35mm Physical Cube
Z_MARGIN = 0.003        
FIXED_Z_HEIGHT = (CUBE_SIZE_MM / 1000.0) / 2.0 + Z_MARGIN 
SMOOTHING = 0.2         # Smooths out vision jitter

VALID_MARKER_IDS = [0, 1, 2, 3, 4, 5]

# Robot Config
ACTUATOR_MAPPING = {"Rotation": 1, "Pitch": 2, "Elbow": 3, "Wrist_Pitch": 4, "Wrist_Roll": 5, "Jaw": 6}
REVERSE_JOINTS = ["Rotation"]
MOTOR_OFFSETS = {"Rotation": 2038, "Pitch": 3076, "Elbow": 1001, "Wrist_Pitch": 2209, "Wrist_Roll": 3079, "Jaw": 2196}
SCALE = 2 * np.pi / 4096

# --- SHARED VARIABLES ---
latest_packet = None
cube_pose_target = None 
running = True
manual_mode = True # Set to True for Lead-Through

# --- LOAD CALIBRATION ---
with open("camera_calibration.json", "r") as f:
    calib = json.load(f)
    K = np.array(calib["camera_matrix"])
    D = np.array(calib["distortion_coefficients"])

with open("origin_config.json", "r") as f:
    origin = json.load(f)
    origin_rvec = np.array(origin["origin_rvec"])
    origin_tvec = np.array(origin["origin_tvec"])

# --- HELPERS ---
def get_world_pose(rvec_marker, tvec_marker, rvec_origin, tvec_origin):
    R_origin, _ = cv2.Rodrigues(rvec_origin)
    R_marker, _ = cv2.Rodrigues(rvec_marker)
    pos_camera = tvec_marker.reshape(3, 1)
    pos_origin = tvec_origin.reshape(3, 1)
    pos_world = R_origin.T @ (pos_camera - pos_origin)
    rot_world_mat = R_origin.T @ R_marker
    quat = R.from_matrix(rot_world_mat).as_quat()
    return pos_world.flatten(), [quat[3], quat[0], quat[1], quat[2]]

def set_torque(ser, id, enable):
    val = 1 if enable else 0
    checksum = (~(id + 4 + 3 + 40 + val)) & 0xFF
    packet = [0xFF, 0xFF, id, 0x04, 0x03, 40, val, checksum]
    ser.write(bytearray(packet))
    time.sleep(0.002)

# --- THREAD 1: VISION WORKER (Camera Window + Axis) ---
def vision_worker():
    global cube_pose_target, running
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened(): return

    # Setup ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    half = ARUCO_SIZE / 2.0
    obj_points = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)

    print("[VISION] Starting Camera Window...")
    while running:
        ret, frame = cap.read()
        if not ret: continue

        corners, ids, _ = detector.detectMarkers(frame)
        if ids is not None:
            ids = ids.flatten()
            for i, id_val in enumerate(ids):
                if id_val in VALID_MARKER_IDS:
                    success, rvec, tvec = cv2.solvePnP(obj_points, corners[i][0], K, D)
                    if success:
                        # 1. Update Target Position for MuJoCo
                        pos, quat = get_world_pose(rvec, tvec, origin_rvec, origin_tvec)
                        cube_pose_target = [pos[0], pos[1], FIXED_Z_HEIGHT] + quat
                        
                        # 2. Draw Visuals on Camera Frame
                        cv2.drawFrameAxes(frame, K, D, rvec, tvec, 0.03) # 3cm axis
                        cv2.putText(frame, f"ID:{id_val} X:{pos[0]:.2f} Y:{pos[1]:.2f}", 
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        break

        cv2.imshow('Calibration Verification - ArUco Axis', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()

# --- THREAD 2: ROBOT WORKER (Lead-Through Sync) ---
def robot_worker(ser, model, data):
    global running, manual_mode
    while running:
        if manual_mode:
            for name, motor_id in ACTUATOR_MAPPING.items():
                # Manually build the Read Position packet
                checksum = (~(motor_id + 4 + 2 + 0x38 + 2)) & 0xFF
                packet = [0xFF, 0xFF, motor_id, 0x04, 0x02, 0x38, 0x02, checksum]
                ser.reset_input_buffer()
                ser.write(bytearray(packet))
                res = ser.read(8)
                if len(res) == 8 and res[0] == 0xFF:
                    raw = res[5] + (res[6] << 8)
                    offset = MOTOR_OFFSETS.get(name, 2048)
                    rad = (offset - raw) * SCALE if name in REVERSE_JOINTS else (raw - offset) * SCALE
                    
                    # Update Sim qpos directly
                    a_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                    if a_id != -1:
                        data.ctrl[a_id] = rad
                        data.qpos[model.jnt_qposadr[model.actuator_trnid[a_id, 0]]] = rad
            time.sleep(0.01)
        else:
            # Handle standard simulation-to-robot control here if manual_mode=False
            time.sleep(0.01)

# --- MAIN ---
# --- MAIN ---
def main():
    global running, cube_pose_target
    try: 
        ser = serial.Serial(PORT, BAUDRATE, timeout=0.05)
    except Exception as e: 
        print(f"Serial Error: {e}")
        return

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    
    # --- CRITICAL CHANGE: DISABLE TORQUE FOR LOOSE JOINTS ---
    print("Disabling Torque for Manual Movement (Passive Mode)...")
    for i in range(1, 7): 
        # Setting enable=False (0) makes the motors 'loose'
        set_torque(ser, i, False)

    # Launch Threads
    # The robot_worker will still READ positions while torque is off
    threading.Thread(target=vision_worker, daemon=True).start()
    threading.Thread(target=robot_worker, args=(ser, model, data), daemon=True).start()

    print("\n[LOOSE MODE] You can now move the joints by hand.")
    print("The simulation will mirror your physical movements in real-time.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        cube_pose_curr = np.array([0.3, 0.0, FIXED_Z_HEIGHT, 1, 0, 0, 0])
        
        while viewer.is_running() and running:
            step_start = time.time()

            # 1. Update Cube state in Sim from Vision
            if cube_pose_target is not None:
                cube_pose_curr = (cube_pose_curr * (1 - SMOOTHING)) + (np.array(cube_pose_target) * SMOOTHING)
                c_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube")
                if c_id != -1:
                    data.qpos[model.jnt_qposadr[c_id] : model.jnt_qposadr[c_id]+7] = cube_pose_curr

            # 2. Update Physics and Sync
            mujoco.mj_forward(model, data)
            viewer.sync()
            
            elapsed = time.time() - step_start
            time.sleep(max(0, (1.0/SIM_FPS) - elapsed))

    running = False
    ser.close()

if __name__ == "__main__":
    main()

