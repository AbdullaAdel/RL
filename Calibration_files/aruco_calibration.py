import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os

def load_calibration(filename="camera_calibration.json"):
    if not os.path.exists(filename):
        print(f"Error: '{filename}' not found. Please run calibration first.")
        return None, None
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    mtx = np.array(data["camera_matrix"])
    dist = np.array(data["distortion_coefficients"])
    return mtx, dist

def save_origin(rvec, tvec, filename="origin_config.json"):
    data = {
        "origin_rvec": rvec.flatten().tolist(),
        "origin_tvec": tvec.flatten().tolist(),
        "note": "Transform from Camera to World Origin (Marker)"
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"\n[SAVED] New Origin saved to '{filename}'")

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

def main():
    # 1. Configuration
    MARKER_SIZE = 25.0  # mm
    calib_file = "camera_calibration.json"
    
    # Load Intrinsics
    camera_matrix, dist_coeffs = load_calibration(calib_file)
    if camera_matrix is None:
        return

    # 2. Setup ArUco
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    # 3. Start Camera
    cap = cv2.VideoCapture(0) # Change index if needed
    
    # --- CRITICAL: MATCHING RESOLUTION TO CAPTURE SCRIPT ---
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # -------------------------------------------------------

    # Allow the window to be resizable (in case 1080p is too big for your screen)
    cv2.namedWindow('ArUco Tracker', cv2.WINDOW_NORMAL)

    print("\n--- ArUco Origin Setter (1080p) ---")
    print(f"Tracking Marker Size: {MARKER_SIZE} mm")
    print("Press [SPACE] to set Origin")
    print("Press [Q] to Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        current_rvec = None
        current_tvec = None

        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = my_estimatePoseSingleMarkers(corners, MARKER_SIZE, camera_matrix, dist_coeffs)

            for i, marker_id in enumerate(ids):
                aruco.drawDetectedMarkers(frame, corners)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 15)

                if i == 0:
                    current_rvec = rvecs[i]
                    current_tvec = tvecs[i]
                    
                    x = tvecs[i][0][0]
                    y = tvecs[i][1][0]
                    z = tvecs[i][2][0]
                    
                    # Text overlay
                    text_str = f"ID: {marker_id[0]} | X:{x:.1f} Y:{y:.1f} Z:{z:.1f}"
                    cv2.putText(frame, text_str, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('ArUco Tracker', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 32: # SPACE
            if current_rvec is not None:
                save_origin(current_rvec, current_tvec)
                cv2.imshow('ArUco Tracker', 255 - frame) # Flash
                cv2.waitKey(50)
            else:
                print("[WARNING] No marker detected.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()