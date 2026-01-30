import serial
import time
import mujoco
import mujoco.viewer
import numpy as np
import cv2
import json
from scipy.spatial.transform import Rotation as R

# --- CONFIGURATION ---
PORT = "COM3"
BAUDRATE = 1000000
XML_PATH = "scene.xml"
CAMERA_ID = 0
ARUCO_SIZE = 0.025  # 25mm (Meters)

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

# Load Camera Calibration
with open("camera_calibration.json", "r") as f:
    calib_data = json.load(f)
    K = np.array(calib_data["camera_matrix"])
    D = np.array(calib_data["distortion_coefficients"])

def read_current_raw(ser, id):
    checksum = (~(id + 4 + 2 + 0x38 + 2)) & 0xFF
    packet = [0xFF, 0xFF, id, 0x04, 0x02, 0x38, 0x02, checksum]
    ser.reset_input_buffer()
    ser.write(bytearray(packet))
    res = ser.read(8)
    if len(res) == 8 and res[0] == 0xFF:
        return res[5] + (res[6] << 8)
    return None

def set_torque(ser, id, enable):
    val = 1 if enable else 0
    checksum = (~(id + 4 + 3 + 40 + val)) & 0xFF
    packet = [0xFF, 0xFF, id, 0x04, 0x03, 40, val, checksum]
    ser.write(bytearray(packet))
    time.sleep(0.002)

def main():
    print(f"Connecting to {PORT}...")
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=0.05)
    except Exception as e:
        print(f"Serial Error: {e}")
        return

    print("Loading Digital Twin...")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    print("Disabling Torque (Move robot to marker center)...")
    for i in range(1, 7):
        set_torque(ser, i, False)

    cap = cv2.VideoCapture(CAMERA_ID)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    half = ARUCO_SIZE / 2.0
    obj_points = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)

    # --- STATE VARIABLES ---
    saved_marker_tvec = None
    saved_marker_rvec = None
    marker_frozen = False

    print("\n--- INSTRUCTIONS ---")
    print("1. Place Marker. Keep robot away.")
    print("2. Press 'M' to MEMORIZE marker position.")
    print("3. Move Robot to touch Marker Center.")
    print("4. Press 'SPACE' to Calibrate.")

    while True:
        # 1. READ ROBOT (FK)
        for name, id in ACTUATOR_MAPPING.items():
            raw = read_current_raw(ser, id)
            if raw is not None:
                offset = MOTOR_OFFSETS.get(name, 2048)
                rad = (offset - raw) * SCALE if name in REVERSE_JOINTS else (raw - offset) * SCALE
                act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                if act_id != -1:
                    joint_id = model.actuator_trnid[act_id, 0]
                    data.qpos[model.jnt_qposadr[joint_id]] = rad
        
        mujoco.mj_kinematics(model, data) 
        
        # Get Gripper Tip Position
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        if ee_id != -1:
            robot_pos = data.site_xpos[ee_id]
        else:
            print("Error: 'ee_site' not found in XML!")
            break

        # 2. READ CAMERA
        ret, frame = cap.read()
        if not ret: continue
        
        corners, ids, rejected = detector.detectMarkers(frame)
        current_rvec = None
        current_tvec = None
        seeing_marker = False

        if ids is not None:
            ids = ids.flatten()
            for i, id_val in enumerate(ids):
                success, rvec, tvec = cv2.solvePnP(obj_points, corners[i][0], K, D)
                if success:
                    current_rvec = rvec
                    current_tvec = tvec
                    seeing_marker = True
                    cv2.drawFrameAxes(frame, K, D, rvec, tvec, 0.05)
                    break
        
        # 3. GUI LOGIC
        
        # Line 1: Robot Status
        cv2.putText(frame, f"Robot: [{robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f}]", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Line 2: Marker Status
        if marker_frozen:
            # Show FROZEN data
            cv2.putText(frame, "MARKER: MEMORIZED (OK to block view)", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Draw Axis of saved marker if we can calculate it relative to camera? 
            # (Hard to draw 'saved' axis if camera moves, but assuming camera is static)
            if saved_marker_tvec is not None:
                 cv2.drawFrameAxes(frame, K, D, saved_marker_rvec, saved_marker_tvec, 0.05)
        else:
            if seeing_marker:
                cv2.putText(frame, "MARKER: VISIBLE (Press 'M')", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "MARKER: NOT FOUND", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Line 3: Instructions
        if marker_frozen:
            cv2.putText(frame, "STEP 2: Touch Center & Press SPACE", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "STEP 1: Clear View & Press 'M'", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Calibration V2 (Freeze Mode)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('m'):
            if seeing_marker:
                saved_marker_tvec = current_tvec
                saved_marker_rvec = current_rvec
                marker_frozen = True
                print("Marker Position MEMORIZED!")
        elif key == 32: # SPACE
            if marker_frozen:
                print("\n--- CALIBRATING ---")
                
                # Use the SAVED marker pose + CURRENT robot pose
                origin_rvec = saved_marker_rvec.flatten()
                rmat, _ = cv2.Rodrigues(origin_rvec)
                
                pos_cam = saved_marker_tvec.flatten()
                pos_robot = robot_pos # The robot is touching the center NOW
                
                # Calculate Origin relative to Camera
                rotated_robot_pos = rmat @ pos_robot
                origin_tvec = pos_cam - rotated_robot_pos
                
                config = {
                    "origin_rvec": origin_rvec.tolist(),
                    "origin_tvec": origin_tvec.tolist(),
                    "note": "Calibrated using Freeze & Touch Method"
                }
                
                with open("origin_config.json", "w") as f:
                    json.dump(config, f, indent=4)
                
                print("SUCCESS: 'origin_config.json' updated.")
                break
            else:
                print("Error: You must press 'M' to memorize the marker first!")

    ser.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()