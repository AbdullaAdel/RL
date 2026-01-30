import cv2
import os
import argparse

def capture_images(output_dir, camera_index=0):
    # 1. Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 2. Open Camera
    # Use 0 for default webcam, 1 for external, etc.
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}")
        return

    # Set resolution (optional, try to maximize quality)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print(f"\n--- Camera Capture Tool ---")
    print(f"Storage Directory: {output_dir}")
    print(f"CONTROLS:")
    print(f"  [SPACE] or [S] : Save Image")
    print(f"  [Q] or [ESC]   : Quit")
    print(f"---------------------------\n")

    count = 0
    # Check existing files to avoid overwriting
    while os.path.exists(os.path.join(output_dir, f"img_{count:04d}.jpg")):
        count += 1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Show the live feed
        cv2.imshow('Camera Capture (Press S to Save, Q to Quit)', frame)

        key = cv2.waitKey(1) & 0xFF

        # [S] or [SPACE] to Save
        if key == ord('s') or key == 32:
            filename = os.path.join(output_dir, f"img_{count:04d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[Saved] {filename}")
            count += 1
            
            # visual flash effect
            cv2.imshow('Camera Capture (Press S to Save, Q to Quit)', 255 - frame) 
            cv2.waitKey(50)

        # [Q] or [ESC] to Quit
        elif key == ord('q') or key == 27:
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture images for calibration")
    parser.add_argument("--dir", type=str, default="calibration_images", help="Folder to save images")
    parser.add_argument("--cam", type=int, default=0, help="Camera index (default 0)")
    
    args = parser.parse_args()
    
    capture_images(args.dir, args.cam)