import cv2
import numpy as np
import glob
import json
import argparse
import os

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Converts Euler angles (in degrees) to a 3x3 Rotation Matrix.
    Order: X (roll) -> Y (pitch) -> Z (yaw)
    """
    # Convert to radians
    r_x = np.deg2rad(roll)
    r_y = np.deg2rad(pitch)
    r_z = np.deg2rad(yaw)

    # Rotation matrices around X, Y, Z
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(r_x), -np.sin(r_x)],
                    [0, np.sin(r_x), np.cos(r_x)]])
    
    R_y = np.array([[np.cos(r_y), 0, np.sin(r_y)],
                    [0, 1, 0],
                    [-np.sin(r_y), 0, np.cos(r_y)]])
    
    R_z = np.array([[np.cos(r_z), -np.sin(r_z), 0],
                    [np.sin(r_z), np.cos(r_z), 0],
                    [0, 0, 1]])
    
    # Combined rotation matrix R = Rz * Ry * Rx
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def run_calibration(image_dir, rows, cols, square_size, shift_config=None):
    """
    Performs calibration and optionally shifts the origin.
    """
    # 1. Setup termination criteria for sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 2. Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    # These are the "expected" coordinates of the corners on the board.
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp = objp * square_size  # Scale by square size (mm)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    found_images = []

    # 3. Load Images
    images = glob.glob(os.path.join(image_dir, '*.jpg')) + glob.glob(os.path.join(image_dir, '*.png'))
    images.sort()

    if not images:
        print(f"Error: No images found in directory '{image_dir}'")
        return

    print(f"Found {len(images)} images. Processing...")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        # Note: pattern_size must be (columns, rows) -> (9, 7)
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        if ret:
            objpoints.append(objp)
            
            # Refine corner locations
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            found_images.append(fname)
            print(f"[OK] {fname}")
        else:
            print(f"[FAIL] {fname} - Pattern not found")

    if not objpoints:
        print("Calibration failed: No patterns detected.")
        return

    # 4. Calibrate Camera
    print("\nRunning calibration optimization...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print(f"Calibration RMS Error: {ret:.4f}")

    # 5. Prepare Output Data
    output_data = {
        "rms_error": ret,
        "camera_matrix": mtx.tolist(),
        "distortion_coefficients": dist.tolist(),
        "extrinsics": {}
    }

    # 6. Process Extrinsics (and Apply Shift if requested)
    if shift_config:
        print(f"\nApplying Origin Shift:")
        print(f"  Translation (x,y,z): {shift_config['t']}")
        print(f"  Rotation (r,p,y):    {shift_config['r']} deg")
        
        # Calculate Shift Matrices
        # New Origin relative to Old Pattern: T_shift
        # P_old = R_shift * P_new + t_shift
        # We need to transform the camera pose to be relative to P_new.
        
        R_shift = euler_to_rotation_matrix(*shift_config['r'])
        t_shift = np.array(shift_config['t'], dtype=np.float32).reshape(3, 1)

    for i, fname in enumerate(found_images):
        rvec = rvecs[i]
        tvec = tvecs[i]
        
        # Convert rotation vector to rotation matrix
        R_cam, _ = cv2.Rodrigues(rvec)
        
        # Standard Extrinsics (World = Checkerboard Corner)
        # Cam_Pose = R_cam * World_Point + tvec
        
        if shift_config:
            # We want Extrinsics relative to New Origin.
            # Substituting P_old = R_shift * P_new + t_shift into Cam_Pose:
            # P_cam = R_cam * (R_shift * P_new + t_shift) + tvec
            # P_cam = (R_cam * R_shift) * P_new + (R_cam * t_shift + tvec)
            
            R_cam_new = np.dot(R_cam, R_shift)
            tvec_new = np.dot(R_cam, t_shift) + tvec
            
            # Convert back to rvec for storage
            rvec_new, _ = cv2.Rodrigues(R_cam_new)
            
            save_rvec = rvec_new.flatten().tolist()
            save_tvec = tvec_new.flatten().tolist()
        else:
            save_rvec = rvec.flatten().tolist()
            save_tvec = tvec.flatten().tolist()

        output_data["extrinsics"][os.path.basename(fname)] = {
            "rvec": save_rvec,
            "tvec": save_tvec
        }

    # 7. Save to File
    out_filename = "camera_calibration.json"
    with open(out_filename, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"\nSuccess! Calibration data saved to '{out_filename}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera Calibration with Manual Origin Shift")
    
    # Basic Config
    parser.add_argument("--image_dir", type=str, default=".", help="Directory containing calibration images")
    parser.add_argument("--rows", type=int, default=7, help="Number of internal corners in height (rows)")
    parser.add_argument("--cols", type=int, default=9, help="Number of internal corners in width (cols)")
    parser.add_argument("--square_size", type=float, default=21.0, help="Size of one square in mm")

    # Origin Shift Flags
    parser.add_argument("--shift", action="store_true", help="Enable manual origin shift")
    parser.add_argument("--sx", type=float, default=0.0, help="Shift X (mm)")
    parser.add_argument("--sy", type=float, default=0.0, help="Shift Y (mm)")
    parser.add_argument("--sz", type=float, default=0.0, help="Shift Z (mm)")
    parser.add_argument("--roll", type=float, default=0.0, help="Rotation X (degrees)")
    parser.add_argument("--pitch", type=float, default=0.0, help="Rotation Y (degrees)")
    parser.add_argument("--yaw", type=float, default=0.0, help="Rotation Z (degrees)")

    args = parser.parse_args()

    shift_config = None
    if args.shift:
        shift_config = {
            "t": [args.sx, args.sy, args.sz],
            "r": [args.roll, args.pitch, args.yaw]
        }

    run_calibration(args.image_dir, args.rows, args.cols, args.square_size, shift_config)