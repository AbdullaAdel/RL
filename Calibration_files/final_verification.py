import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os

# --- UTILS ---
def load_json(filename):
    if not os.path.exists(filename):
        print(f"Error: '{filename}' not found.")
        return None
    with open(filename, 'r') as f:
        return json.load(f)

def get_transform_matrix(rvec, tvec):
    """Convert rvec/tvec to 4x4 Transformation Matrix"""
    mat = np.eye(4)
    mat[:3, :3], _ = cv2.Rodrigues(np.array(rvec))
    mat[:3, 3] = np.array(tvec).flatten()
    return mat

def inverse_transform(T):
    """Invert a 4x4 Transformation Matrix"""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    rvecs, tvecs = [], []
    for c in corners:
        _, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
    return rvecs, tvecs

# --- MAIN ---
def main():
    # 1. Configuration - MAKE SURE THIS MATCHES YOUR PRINT SIZE
    MARKER_SIZE = 25.0  # mm 
    
    # 2. Load Data
    calib_data = load_json("camera_calibration.json")
    origin_data = load_json("origin_config.json")
    
    if not calib_data or not origin_data:
        print("Missing configuration files. Run calibration first.")
        return

    mtx = np.array(calib_data["camera_matrix"])
    dist = np.array(calib_data["distortion_coefficients"])
    
    # 3. Calculate Global Transform (World Origin -> Camera)
    # This transforms points from the "Floor Plane" (Origin) to the Camera
    T_origin_to_cam = get_transform_matrix(origin_data["origin_rvec"], origin_data["origin_tvec"])
    
    # We need the inverse: Camera -> World Origin
    # This transforms points from the Camera back to the "Floor Plane"
    T_cam_to_origin = inverse_transform(T_origin_to_cam)

    # 4. Setup ArUco
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    # 5. Start Camera (Matching 1080p resolution)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    cv2.namedWindow('Verification', cv2.WINDOW_NORMAL)

    print("\n--- Origin Verification Tool ---")
    print(f"Loaded Origin Note: {origin_data.get('note', 'Unknown')}")
    print("Red/Green/Blue Axis = The World Origin (0,0,0)")
    print("Text = Position relative to Origin")
    print("Press [Q] to Quit")

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        # A. Draw the "World Origin" Axis
        # This projects the stored (0,0,0) point onto the current image.
        # If the camera moves, this axis moves with the scene (correct behavior).
        cv2.drawFrameAxes(frame, mtx, dist, 
                          np.array(origin_data["origin_rvec"]), 
                          np.array(origin_data["origin_tvec"]), 
                          length=60, thickness=3) # Long axis for visibility

        if ids is not None:
            rvecs, tvecs = my_estimatePoseSingleMarkers(corners, MARKER_SIZE, mtx, dist)

            for i, marker_id in enumerate(ids):
                # Draw standard detection box
                aruco.drawDetectedMarkers(frame, corners)
                
                # B. Calculate Position relative to World Origin
                
                # 1. Get Transform: Marker -> Camera
                T_marker_to_cam = get_transform_matrix(rvecs[i], tvecs[i])
                
                # 2. Chain Transforms: Marker -> Camera -> World Origin
                # T_marker_to_world = T_cam_to_origin * T_marker_to_cam
                T_marker_to_world = T_cam_to_origin @ T_marker_to_cam
                
                # Extract relative X, Y, Z
                x_rel = T_marker_to_world[0, 3]
                y_rel = T_marker_to_world[1, 3]
                z_rel = T_marker_to_world[2, 3]

                # C. Display Info
                # We offset the text slightly above the marker
                text_pos = (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 20)
                
                label = f"ID:{marker_id[0]} X:{x_rel:.1f} Y:{y_rel:.1f} Z:{z_rel:.1f}"
                
                # Logic: If Z is close to 0 (+/- 5mm), text is Green. Else Yellow.
                # Since we calibrated the floor, markers on the table SHOULD be Green.
                color = (0, 255, 0) if abs(z_rel) < 5.0 else (0, 255, 255)

                cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw small axis on the marker itself
                cv2.drawFrameAxes(frame, mtx, dist, rvecs[i], tvecs[i], 15)

        cv2.imshow('Verification', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
