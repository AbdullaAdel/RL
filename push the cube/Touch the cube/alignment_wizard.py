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
CAMERA_ID = 0
ARUCO_SIZE = 0.025  
CUBE_SIZE_MM = 35.0 
FIXED_Z_HEIGHT = (CUBE_SIZE_MM / 1000.0) / 2.0 + 0.003
VALID_MARKER_IDS = [0, 1, 2, 3, 4, 5]

# --- LOAD CONFIG ---
with open("camera_calibration.json", "r") as f:
    calib_data = json.load(f)
    K = np.array(calib_data["camera_matrix"])
    D = np.array(calib_data["distortion_coefficients"])

try:
    with open("origin_config.json", "r") as f:
        data = json.load(f)
        base_tvec = np.array(data["origin_tvec"])
        base_rvec = np.array(data["origin_rvec"])
except:
    base_tvec = np.zeros(3)
    base_rvec = np.zeros(3)

# --- GLOBAL OFFSETS ---
offset_x = 0.0
offset_y = 0.0
offset_z = 0.0
rotation_z_deg = 0.0 
flip_y_axis = False # NEW: Toggle to flip Y direction

cube_pose_display = None
running = True

# --- TRANSFORM LOGIC ---
def get_adjusted_pose(rvec_marker, tvec_marker, base_rvec, base_tvec):
    # 1. Get Base Rotation Matrix
    base_rot_mat, _ = cv2.Rodrigues(base_rvec)
    
    # 2. Apply FLIP (180 deg rotation around X)
    # This keeps X same, Flips Y and Z.
    if flip_y_axis:
        flipper = R.from_euler('x', 180, degrees=True).as_matrix()
        base_rot_mat = base_rot_mat @ flipper

    # 3. Apply Z Rotation (User 'R' key)
    manual_rot = R.from_euler('z', rotation_z_deg, degrees=True).as_matrix()
    final_origin_rot = base_rot_mat @ manual_rot
    
    # 4. Apply Position Offset
    final_origin_tvec = base_tvec + np.array([offset_x, offset_y, offset_z])

    # 5. Calculate Relative Pose
    R_origin = final_origin_rot
    R_marker, _ = cv2.Rodrigues(rvec_marker)
    
    pos_camera = tvec_marker.reshape(3, 1)
    pos_origin = final_origin_tvec.reshape(3, 1)
    
    # World Position
    pos_world = R_origin.T @ (pos_camera - pos_origin)
    
    # World Rotation
    rot_world = R_origin.T @ R_marker
    quat = R.from_matrix(rot_world).as_quat()
    
    # Return computed pose + Current Origin Frame (for debug drawing)
    return pos_world.flatten(), [quat[3], quat[0], quat[1], quat[2]], final_origin_rot, final_origin_tvec

# --- THREAD: VISION ---
def vision_thread():
    global cube_pose_display, running, offset_x, offset_y, offset_z, rotation_z_deg, flip_y_axis
    
    cap = cv2.VideoCapture(CAMERA_ID)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    
    half = ARUCO_SIZE / 2.0
    obj_points = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
    
    print("[WIZARD] Controls: W/A/S/D = X/Y  |  Q/E = Z  |  R = Rotate Z  |  F = Flip Y")

    # Temp variables for smooth drawing even if marker not seen
    current_origin_rvec = base_rvec
    current_origin_tvec = base_tvec

    while running:
        ret, frame = cap.read()
        if not ret: continue
        
        # --- UPDATE ORIGIN MATH FOR DRAWING ---
        base_rot_mat, _ = cv2.Rodrigues(base_rvec)
        
        # Apply Flip
        if flip_y_axis:
            flipper = R.from_euler('x', 180, degrees=True).as_matrix()
            base_rot_mat = base_rot_mat @ flipper

        # Apply Z Rotate
        manual_rot = R.from_euler('z', rotation_z_deg, degrees=True).as_matrix()
        current_rot_mat = base_rot_mat @ manual_rot
        
        current_rvec = cv2.Rodrigues(current_rot_mat)[0]
        current_tvec = base_tvec + np.array([offset_x, offset_y, offset_z])

        # --- DRAW ROBOT BASE AXES ---
        # 
        # Red = X (Forward), Green = Y (Left), Blue = Z (Up)
        cv2.drawFrameAxes(frame, K, D, current_rvec, current_tvec, 0.15) 
        
        status_text = f"FLIP Y: {'ON' if flip_y_axis else 'OFF'}"
        cv2.putText(frame, "ORIGIN", (int(frame.shape[1]/2), 80), 0, 0.7, (0,0,255), 2)
        cv2.putText(frame, status_text, (int(frame.shape[1]/2), 110), 0, 0.7, (255,0,255), 2)

        corners, ids, rejected = detector.detectMarkers(frame)
        if ids is not None:
            ids = ids.flatten()
            for i, id_val in enumerate(ids):
                if id_val in VALID_MARKER_IDS:
                    success, rvec, tvec = cv2.solvePnP(obj_points, corners[i][0], K, D)
                    if success:
                        pos, quat, _, _ = get_adjusted_pose(rvec, tvec, base_rvec, base_tvec)
                        cube_pose_display = [pos[0], pos[1], FIXED_Z_HEIGHT, quat[0], quat[1], quat[2], quat[3]]
                        cv2.drawFrameAxes(frame, K, D, rvec, tvec, 0.03)
                        break

        # --- CONTROLS ---
        key = cv2.waitKey(1)
        if key != -1:
            speed = 0.005 # 5mm
            k = key & 0xFF
            
            if k == ord('q'): running = False
            elif k == ord('a'): offset_x -= speed 
            elif k == ord('d'): offset_x += speed 
            elif k == ord('w'): offset_y += speed 
            elif k == ord('s'): offset_y -= speed 
            elif k == ord('z'): offset_z -= speed 
            elif k == ord('c'): offset_z += speed 
            elif k == ord('r'): rotation_z_deg = (rotation_z_deg + 90) % 360
            
            # --- NEW KEY: FLIP Y ---
            elif k == ord('f'): 
                flip_y_axis = not flip_y_axis
            
            elif k == 13: # ENTER
                print("\n[SAVING...]")
                final_rvec = current_rvec.flatten()
                final_tvec = current_tvec
                
                config = {
                    "origin_rvec": final_rvec.tolist(),
                    "origin_tvec": final_tvec.tolist(),
                    "note": f"Wizard V4 (FlipY={flip_y_axis})"
                }
                with open("origin_config.json", "w") as f:
                    json.dump(config, f, indent=4)
                print("SAVED! Restart main controller.")
                running = False

        # Helper Text
        cv2.putText(frame, "F: Flip Y Axis (Fix Inversion)", (10, 30), 0, 0.6, (255, 0, 255), 2)
        cv2.putText(frame, "R: Rotate Z | W/A/S/D: Move", (10, 60), 0, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "ENTER: Save Config", (10, 90), 0, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Wizard V4 (Flip Fix)", frame)

    cap.release()
    cv2.destroyAllWindows()

# --- MAIN ---
def main():
    global running
    print("Loading Sim...")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    data.ctrl = [0, -1.57, 1.57, 1.57, 1.57, 0] 
    mujoco.mj_step(model, data)

    t = threading.Thread(target=vision_thread)
    t.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and running:
            if cube_pose_display is not None:
                cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube")
                if cube_id != -1:
                    addr = model.jnt_qposadr[cube_id]
                    data.qpos[addr:addr+7] = cube_pose_display
            
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)

    running = False
    t.join()

if __name__ == "__main__":
    main()
