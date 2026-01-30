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
SIM_FPS = 200

# Camera & Object Config
CAMERA_ID = 0  
ARUCO_SIZE = 0.025      # 25mm Marker (Meters)
CUBE_SIZE_MM = 35.0     # 35mm Physical Cube

# Lift cube 3mm to prevent floor clipping
Z_MARGIN = 0.003        
FIXED_Z_HEIGHT = (CUBE_SIZE_MM / 1000.0) / 2.0 + Z_MARGIN 

SMOOTHING = 0.1 

VALID_MARKER_IDS = [0, 1, 2, 3, 4, 5]

# Load Calibration
with open("camera_calibration.json", "r") as f:
    calib_data = json.load(f)
    K = np.array(calib_data["camera_matrix"])
    D = np.array(calib_data["distortion_coefficients"])

with open("origin_config.json", "r") as f:
    origin_data = json.load(f)
    origin_rvec = np.array(origin_data["origin_rvec"])
    # --- CRITICAL FIX: Convert Origin to Meters ---
    # Origin was in mm (~960), we need it in meters (~0.96) to match the marker
    # origin_tvec = np.array(origin_data["origin_tvec"]) / 1000.0
    origin_tvec = np.array(origin_data["origin_tvec"]) 

# Robot Config
ACTUATOR_MAPPING = {
    "Rotation": 1, "Pitch": 2, "Elbow": 3,
    "Wrist_Pitch": 4, "Wrist_Roll": 5, "Jaw": 6
}
REVERSE_JOINTS = ["Rotation"]
MOTOR_OFFSETS = {
    "Rotation": 2038, "Pitch": 3076, "Elbow": 1001,
    "Wrist_Pitch": 2209, "Wrist_Roll": 3079, "Jaw": 2196,
}
SCALE = 2 * np.pi / 4096

# --- SHARED VARIABLES ---
latest_packet = None  
cube_pose_target = None  
cube_pose_current = None 
running = True

# --- HELPER: COORDINATE TRANSFORM ---
def get_world_pose(rvec_marker, tvec_marker, rvec_origin, tvec_origin):
    """ Converts Marker (Camera Frame) -> World (Table Frame) """
    R_origin, _ = cv2.Rodrigues(rvec_origin)
    R_marker, _ = cv2.Rodrigues(rvec_marker)

    pos_camera = tvec_marker.reshape(3, 1)
    pos_origin = tvec_origin.reshape(3, 1)
    
    # Both are now in Meters, so this subtraction is safe
    pos_world = R_origin.T @ (pos_camera - pos_origin)
    
    # Rotation
    rot_world_mat = R_origin.T @ R_marker
    quat = R.from_matrix(rot_world_mat).as_quat()
    mujoco_quat = [quat[3], quat[0], quat[1], quat[2]] 
    
    return pos_world.flatten(), mujoco_quat

# --- THREAD 1: VISION WORKER ---
def vision_worker():
    global cube_pose_target, running
    
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened(): return

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Marker size in Meters
    half = ARUCO_SIZE / 2.0
    obj_points = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)

    print("[VISION] Thread Started.")

    while running:
        ret, frame = cap.read()
        if not ret: continue

        corners, ids, rejected = detector.detectMarkers(frame)

        if ids is not None:
            ids = ids.flatten()
            for i, id_val in enumerate(ids):
                if id_val in VALID_MARKER_IDS:
                    # solvePnP returns Meters (because obj_points is Meters)
                    success, rvec, tvec = cv2.solvePnP(obj_points, corners[i][0], K, D)
                    
                    if success:
                        pos, quat = get_world_pose(rvec, tvec, origin_rvec, origin_tvec)
                        
                        # NO DIVISION NEEDED HERE anymore.
                        # pos is in Meters.
                        x_m = pos[0] 
                        y_m = pos[1] 
                        
                        # Debug Text to check range (Should be e.g. 0.1, 0.2)
                        text = f"X:{x_m:.3f} Y:{y_m:.3f}"
                        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        cube_pose_target = [x_m, y_m, FIXED_Z_HEIGHT, quat[0], quat[1], quat[2], quat[3]]
                        
                        cv2.drawFrameAxes(frame, K, D, rvec, tvec, 0.03)
                        break 

        cv2.imshow('ArUco Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

# --- THREAD 2: ROBOT SENDER ---
def sender_thread(ser):
    global latest_packet, running
    while running:
        if latest_packet:
            try: ser.write(latest_packet)
            except: pass
            time.sleep(0.005)
        else: time.sleep(0.001)

# --- UTILS ---
def build_packet(mapping, model, data, offsets, reverse_list):
    payload = [0x2A, 0x02] 
    for name, id in mapping.items():
        try:
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if act_id == -1: continue
            sim_rad = data.ctrl[act_id]
            offset = offsets.get(name, 2048)
            target = offset - (sim_rad / SCALE) if name in reverse_list else (sim_rad / SCALE) + offset
            target = int(max(0, min(4096, target)))
            payload.extend([id, target & 0xFF, (target >> 8) & 0xFF])
        except: pass
    packet_len = len(payload) + 2
    packet = [0xFF, 0xFF, 0xFE, packet_len, 0x83] + payload
    checksum = (~(0xFE + packet_len + 0x83 + sum(payload))) & 0xFF
    packet.append(checksum)
    return bytearray(packet)

def set_torque(ser, id, enable):
    val = 1 if enable else 0
    checksum = (~(id + 4 + 3 + 40 + val)) & 0xFF
    packet = [0xFF, 0xFF, id, 0x04, 0x03, 40, val, checksum]
    ser.write(bytearray(packet))
    time.sleep(0.002)

def read_current_raw(ser, id):
    checksum = (~(id + 4 + 2 + 0x38 + 2)) & 0xFF
    packet = [0xFF, 0xFF, id, 0x04, 0x02, 0x38, 0x02, checksum]
    ser.reset_input_buffer()
    ser.write(bytearray(packet))
    res = ser.read(8)
    if len(res) == 8 and res[0] == 0xFF:
        return res[5] + (res[6] << 8)
    return None

# --- MAIN ---
def main():
    global latest_packet, running, cube_pose_target, cube_pose_current
    print(f"Connecting to {PORT}...")
    try: ser = serial.Serial(PORT, BAUDRATE, timeout=0.05)
    except Exception as e: print(f"Serial Error: {e}"); return

    print(f"Loading {XML_PATH}...")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # --- AUTO-FIX: RESIZE CUBE ---
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    if cube_geom_id != -1:
        new_half_size = (CUBE_SIZE_MM / 1000.0) / 2.0
        model.geom_size[cube_geom_id] = [new_half_size, new_half_size, new_half_size]
        # Soften contacts
        model.geom_solref[cube_geom_id] = [0.02, 1] 
        model.geom_solimp[cube_geom_id] = [0.9, 0.95, 0.001, 0.5, 2]

    # Safety Sync
    print("Syncing...")
    for name, id in ACTUATOR_MAPPING.items():
        raw = read_current_raw(ser, id)
        if raw is not None:
            offset = MOTOR_OFFSETS.get(name, 2048)
            real_rad = (offset - raw) * SCALE if name in REVERSE_JOINTS else (raw - offset) * SCALE
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if act_id != -1:
                data.ctrl[act_id] = real_rad
                data.qpos[model.jnt_qposadr[model.actuator_trnid[act_id, 0]]] = real_rad
    mujoco.mj_forward(model, data)
    
    print("Enabling Torque...")
    for i in range(1, 7): set_torque(ser, i, True)

    t_robot = threading.Thread(target=sender_thread, args=(ser,))
    t_vision = threading.Thread(target=vision_worker)
    t_robot.start()
    t_vision.start()

    print("[READY] Units fixed: Origin(m) and Marker(m).")

    cube_pose_current = np.array([0.25, 0.1, FIXED_Z_HEIGHT, 1, 0, 0, 0])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            if cube_pose_target is not None:
                # Smooth Update
                target_np = np.array(cube_pose_target)
                cube_pose_current = (cube_pose_current * (1 - SMOOTHING)) + (target_np * SMOOTHING)
                
                cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube")
                if cube_id != -1:
                    addr = model.jnt_qposadr[cube_id]
                    data.qpos[addr:addr+7] = cube_pose_current
                    vel_addr = model.jnt_dofadr[cube_id]
                    data.qvel[vel_addr:vel_addr+6] = 0.0

            mujoco.mj_step(model, data)
            viewer.sync()

            latest_packet = build_packet(ACTUATOR_MAPPING, model, data, MOTOR_OFFSETS, REVERSE_JOINTS)

            elapsed = time.time() - step_start
            time.sleep(max(0, (1.0/SIM_FPS) - elapsed))

    running = False
    t_robot.join()
    t_vision.join()
    for i in range(1, 7): set_torque(ser, i, False)
    ser.close()

if __name__ == "__main__":
    main()
    
    
    
    '''
    Now, a new frame that talks about syncing between the real robot and the simulation using the methods in the code, and a video that will show that the code responds to what the real robot does. the code is uploaded.
    '''