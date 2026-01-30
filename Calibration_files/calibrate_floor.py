import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os

def load_calibration(filename="camera_calibration.json"):
    if not os.path.exists(filename):
        print(f"Error: '{filename}' not found.")
        return None, None
    with open(filename, 'r') as f:
        data = json.load(f)
    return np.array(data["camera_matrix"]), np.array(data["distortion_coefficients"])

def save_origin(rvec, tvec, filename="origin_config.json"):
    data = {
        "origin_rvec": rvec.flatten().tolist(),
        "origin_tvec": tvec.flatten().tolist(),
        "note": "Robust Origin generated from 4-marker plane average"
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"\n[SUCCESS] Robust Origin saved to '{filename}'")

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

def compute_robust_origin(rvecs, tvecs):
    """
    Computes a new coordinate system where:
    - Origin = Centroid of all detected markers
    - Z-axis = Average surface normal of all markers (Fixes tilt/depth)
    - X-axis = Average X-axis of all markers (Keeps alignment)
    """
    
    # 1. Average the Centers (Translation)
    t_avg = np.mean(tvecs, axis=0)

    # 2. Average the Rotations
    # We cannot just average rvecs (angles). We must average vectors.
    
    z_vectors = []
    x_vectors = []
    
    for rvec in rvecs:
        # Convert rvec to Rotation Matrix
        R, _ = cv2.Rodrigues(rvec)
        # R columns are the X, Y, Z axes of the marker in Camera Space
        x_vectors.append(R[:, 0]) # X is column 0
        z_vectors.append(R[:, 2]) # Z is column 2

    # Average Z (The Surface Normal)
    z_avg = np.mean(z_vectors, axis=0)
    z_avg = z_avg / np.linalg.norm(z_avg) # Normalize

    # Average X (The Forward Direction)
    x_avg = np.mean(x_vectors, axis=0)
    x_avg = x_avg / np.linalg.norm(x_avg)

    # Re-orthogonalize coordinates (Gram-Schmidt process)
    # Ensure X is perfectly perpendicular to Z
    # New X = X - (X . Z) * Z
    x_new = x_avg - np.dot(x_avg, z_avg) * z_avg
    x_new = x_new / np.linalg.norm(x_new)
    
    # Calculate Y using Cross Product (Z cross X)
    y_new = np.cross(z_avg, x_new)
    
    # Construct new Rotation Matrix
    R_new = np.column_stack((x_new, y_new, z_avg))
    
    # Convert back to rvec
    rvec_new, _ = cv2.Rodrigues(R_new)
    
    return rvec_new, t_avg

def main():
    MARKER_SIZE = 25.0 # mm - Double check this!
    
    mtx, dist = load_calibration()
    if mtx is None: return

    # Setup ArUco
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print("\n--- Multi-Marker Floor Calibrator ---")
    print("1. Place at least 4 markers on the floor/table.")
    print("2. Ensure they are flat.")
    print("3. Press [SPACE] to capture and calculate the average plane.")
    print("   This will force the Z-depth of these markers to become 0.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)
        
        frame_display = frame.copy()

        if ids is not None and len(ids) >= 4:
            # Visualize detected markers
            aruco.drawDetectedMarkers(frame_display, corners, ids)
            
            # Show status
            cv2.putText(frame_display, f"Markers Detected: {len(ids)} (Ready for SPACE)", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame_display, f"Need 4+ Markers (Found {0 if ids is None else len(ids)})", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('Floor Calibrator', frame_display)
        key = cv2.waitKey(1) & 0xFF

        if key == 32: # SPACE
            if ids is not None and len(ids) >= 4:
                print(f"\nProcessing {len(ids)} markers...")
                
                # 1. Get Pose of all markers relative to Camera
                rvecs, tvecs = my_estimatePoseSingleMarkers(corners, MARKER_SIZE, mtx, dist)
                
                # 2. Compute the "Average Plane" (Robust Origin)
                new_rvec, new_tvec = compute_robust_origin(rvecs, tvecs)
                
                # 3. Save
                save_origin(new_rvec, new_tvec)
                
                print("Done! Run 'verify_origin.py' to see the corrected Z-values.")
                break
            else:
                print("Not enough markers found.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()