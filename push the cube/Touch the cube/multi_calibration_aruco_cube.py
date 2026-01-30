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
ARUCO_SIZE = 0.025  # Meters

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

# Load Camera Intrinsics
with open("camera_calibration.json", "r") as f:
    calib_data = json.load(f)
    K = np.array(calib_data["camera_matrix"])
    D = np.array(calib_data["distortion_coefficients"])

# --- HELPER FUNCTIONS ---
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

# --- MATH: KABSCH ALGORITHM ---
def rigid_transform_3D(A, B):
    """
    Finds Rotation R and Translation T such that B = R @ A + T
    A: 3xN matrix of Robot Points
    B: 3xN matrix of Camera Points
    """
    assert A.shape == B.shape
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    # 1. Centroid
    centroid_A = np.mean(A, axis=1).reshape(-1, 1)
    centroid_B = np.mean(B, axis=1).reshape(-1, 1)

    # 2. Center the points
    Am = A - centroid_A
    Bm = B - centroid_B

    # 3. Covariance Matrix
    H = Am @ Bm.T

    # 4. SVD
    U, S, Vt = np.linalg.svd(H)
    R_mat = Vt.T @ U.T

    # 5. Reflection Case
    if np.linalg.det(R_mat) < 0:
        Vt[2,:] *= -1
        R_mat = Vt.T @ U.T

    # 6. Translation
    t_vec = centroid_B - R_mat @ centroid_A

    return R_mat, t_vec

def main():
    print(f"Connecting to {PORT}...")
    try: ser = serial.Serial(PORT, BAUDRATE, timeout=0.05)
    except Exception as e: print(f"Serial Error: {e}"); return

    print("Loading Digital Twin...")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    print("Disabling Torque...")
    for i in range(1, 7): set_torque(ser, i, False)

    cap = cv2.VideoCapture(CAMERA_ID)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    half = ARUCO_SIZE / 2.0
    obj_points = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)

    # DATA STORAGE
    points_robot = []
    points_camera = []
    
    saved_marker_tvec = None
    saved_marker_rvec = None
    marker_frozen = False

    print("\n=== MULTI-POINT CALIBRATION ===")
    print("Goal: Collect 4+ points spread across the table.")
    print("1. Place Marker -> Press 'M' (Freeze Camera)")
    print("2. Touch Robot Tip to Marker -> Press 'SPACE' (Capture Point)")
    print("3. Repeat for next location.")
    print("4. Press 'ENTER' to Calculate & Save.\n")

    while True:
        # 1. READ ROBOT
        for name, id in ACTUATOR_MAPPING.items():
            raw = read_current_raw(ser, id)
            if raw is not None:
                offset = MOTOR_OFFSETS.get(name, 2048)
                rad = (offset - raw) * SCALE if name in REVERSE_JOINTS else (raw - offset) * SCALE
                act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                if act_id != -1:
                    data.qpos[model.jnt_qposadr[model.actuator_trnid[act_id, 0]]] = rad
        mujoco.mj_kinematics(model, data)
        robot_pos = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")]

        # 2. READ CAMERA
        ret, frame = cap.read()
        if not ret: continue
        corners, ids, _ = detector.detectMarkers(frame)
        curr_rvec, curr_tvec = None, None
        seeing_marker = False

        if ids is not None:
            ids = ids.flatten()
            for i, id_val in enumerate(ids):
                _, curr_rvec, curr_tvec = cv2.solvePnP(obj_points, corners[i][0], K, D)
                seeing_marker = True
                cv2.drawFrameAxes(frame, K, D, curr_rvec, curr_tvec, 0.05)
                break
        
        # UI
        cv2.putText(frame, f"Points Collected: {len(points_robot)}", (20, 30), 0, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"Robot: {robot_pos}", (20, 60), 0, 0.5, (255,255,0), 1)

        if marker_frozen:
             cv2.putText(frame, "MARKER FROZEN! Touch & Press SPACE.", (20, 90), 0, 0.6, (0,255,0), 2)
        elif seeing_marker:
             cv2.putText(frame, "Marker Visible. Press 'M' to Freeze.", (20, 90), 0, 0.6, (0,255,255), 2)
        else:
             cv2.putText(frame, "Find Marker...", (20, 90), 0, 0.6, (0,0,255), 2)
             
        if len(points_robot) >= 4:
            cv2.putText(frame, "READY TO SOLVE! Press ENTER.", (20, 130), 0, 0.7, (0, 255, 0), 2)

        cv2.imshow("Multi-Point Calib", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'): break
        elif key == ord('m') and seeing_marker:
            saved_marker_tvec = curr_tvec
            saved_marker_rvec = curr_rvec
            marker_frozen = True
            print(f"Captured Camera Pos: {curr_tvec.flatten()}")
        elif key == 32 and marker_frozen: # SPACE
            # Store point pair
            points_robot.append(robot_pos) # Robot is Ground Truth X
            points_camera.append(saved_marker_tvec.flatten()) # Camera is Measurement Y
            print(f"Captured Robot Pos: {robot_pos}")
            
            marker_frozen = False # Reset for next point
            saved_marker_tvec = None
            print(f"--> Point {len(points_robot)} Saved! Move marker to new spot.")
            
        elif key == 13 and len(points_robot) >= 4: # ENTER
            print("\nCalculating Rigid Transform...")
            
            # Prepare Matrices (3xN)
            A = np.array(points_robot).T   # Robot Points
            B = np.array(points_camera).T  # Camera Points
            
            # Solve B = R*A + T
            R_mat, T_vec = rigid_transform_3D(A, B)
            
            # Convert R matrix to Rodrigues vector for storage
            rvec_out, _ = cv2.Rodrigues(R_mat)
            tvec_out = T_vec.flatten()
            
            error_sum = 0
            for i in range(A.shape[1]):
                p_rob = A[:, i].reshape(3,1)
                p_cam = B[:, i].reshape(3,1)
                p_est = R_mat @ p_rob + T_vec
                dist = np.linalg.norm(p_cam - p_est)
                error_sum += dist
            avg_error = error_sum / A.shape[1]
            
            print(f"Rotation Matrix:\n{R_mat}")
            print(f"Translation Vector: {tvec_out}")
            print(f"Average Error: {avg_error*1000:.2f} mm")
            
            config = {
                "origin_rvec": rvec_out.flatten().tolist(),
                "origin_tvec": tvec_out.tolist(),
                "note": f"Multi-Point Calibration ({len(points_robot)} pts). Error: {avg_error:.4f}m"
            }
            
            with open("origin_config.json", "w") as f:
                json.dump(config, f, indent=4)
            
            print("SUCCESS! 'origin_config.json' updated.")
            break

    ser.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()