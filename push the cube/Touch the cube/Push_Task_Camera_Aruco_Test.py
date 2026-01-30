import time
import threading
import json
import numpy as np
import cv2
import serial
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from scipy.spatial.transform import Rotation as R
from so_arm_push_env import SoArmPushEnv

# --- CONFIGURATION ---
MODEL_PATH = "ppo_soarm_push"  
PORT = "COM3"
BAUDRATE = 1000000
XML_PATH = "scene.xml"
SIM_FPS = 60 

# --- SPEED & SAFETY CONTROL ---
ACTION_SCALE = 0.02         # AI Strength (Lower = Smoother)
MAX_SPEED_RAD_PER_STEP = 0.01  # Max speed limit per frame
HOMING_DURATION_SEC = 5.0   # Seconds to move to start pose

# --- CRITICAL: MANUAL START POSE ---
# Instead of trusting Keyframe 0 (which might be straight out),
# we define a safe "Ready" pose here (in Radians).
# [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw]
MANUAL_START_POSE = [0.0, -1.57, 1.57, 1.57, 1.57, 0.0] 

# Vision Config
CAMERA_ID = 0
ARUCO_SIZE = 0.025      
CUBE_SIZE_MM = 35.0     
Z_MARGIN = 0.003        
FIXED_Z_HEIGHT = (CUBE_SIZE_MM / 1000.0) / 2.0 + Z_MARGIN 
SMOOTHING = 0.1 
VALID_MARKER_IDS = [0, 1, 2, 3, 4, 5]

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

# --- GLOBAL STATE ---
latest_packet = None  
cube_pose_target = None  
cube_pose_current = None 
running = True
ai_active = False 
homing_complete = False 

# --- LOAD CALIBRATION ---
with open("camera_calibration.json", "r") as f:
    calib_data = json.load(f)
    K = np.array(calib_data["camera_matrix"])
    D = np.array(calib_data["distortion_coefficients"])

with open("origin_config.json", "r") as f:
    origin_data = json.load(f)
    origin_rvec = np.array(origin_data["origin_rvec"])
    origin_tvec = np.array(origin_data["origin_tvec"]) 

# --- HELPER FUNCTIONS ---
def get_world_pose(rvec_marker, tvec_marker, rvec_origin, tvec_origin):
    R_origin, _ = cv2.Rodrigues(rvec_origin)
    R_marker, _ = cv2.Rodrigues(rvec_marker)
    pos_camera = tvec_marker.reshape(3, 1)
    pos_origin = tvec_origin.reshape(3, 1)
    pos_world = R_origin.T @ (pos_camera - pos_origin)
    rot_world_mat = R_origin.T @ R_marker
    quat = R.from_matrix(rot_world_mat).as_quat()
    return pos_world.flatten(), [quat[3], quat[0], quat[1], quat[2]]

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

# --- THREAD 1: VISION WORKER ---
def vision_worker():
    global cube_pose_target, running, ai_active, homing_complete
    cap = cv2.VideoCapture(CAMERA_ID)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    half = ARUCO_SIZE / 2.0
    obj_points = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)

    print("[SYSTEM] Vision Active.")

    while running:
        ret, frame = cap.read()
        if not ret: continue
        
        cv2.drawFrameAxes(frame, K, D, origin_rvec, origin_tvec, 0.1)

        corners, ids, rejected = detector.detectMarkers(frame)
        if ids is not None:
            ids = ids.flatten()
            for i, id_val in enumerate(ids):
                if id_val in VALID_MARKER_IDS:
                    success, rvec, tvec = cv2.solvePnP(obj_points, corners[i][0], K, D)
                    if success:
                        pos, quat = get_world_pose(rvec, tvec, origin_rvec, origin_tvec)
                        cube_pose_target = [pos[0], pos[1], FIXED_Z_HEIGHT, quat[0], quat[1], quat[2], quat[3]]
                        cv2.drawFrameAxes(frame, K, D, rvec, tvec, 0.03)
                        break 

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            running = False
            break
        elif key == ord('s'): 
            if homing_complete:
                if not ai_active: print(">>> AI STARTED <<<")
                ai_active = True
            else:
                print("[WAIT] Homing to Ready Pose...")
        elif key == ord('x'): 
            if ai_active: print(">>> AI PAUSED <<<")
            ai_active = False

        if not homing_complete:
            status_text = "STATUS: HOMING..."
            color = (0, 255, 255)
        else:
            status_text = "AI: ACTIVE (X)" if ai_active else "AI: READY (S)"
            color = (0, 255, 0) if ai_active else (0, 0, 255)
            
        cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('Robot Vision', frame)

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

# --- HOMING: MOVE TO MANUAL POSE ---
def perform_homing(model, data):
    print(f"\n[HOMING] Moving to Manual Ready Pose: {MANUAL_START_POSE}")
    
    # Target is our manual list
    target_qpos = np.array(MANUAL_START_POSE)
    
    # Get Current Position
    start_qpos = data.qpos[:6].copy()
    
    # Interpolate
    steps = int(HOMING_DURATION_SEC * SIM_FPS)
    
    for i in range(steps):
        alpha = i / steps
        smooth_alpha = alpha * alpha * (3 - 2 * alpha) 
        
        current_target = (1 - smooth_alpha) * start_qpos + smooth_alpha * target_qpos
        
        data.ctrl[:6] = current_target
        data.qpos[:6] = current_target
        
        mujoco.mj_step(model, data)
        time.sleep(1.0 / SIM_FPS)
        
    print("[HOMING] Complete.\n")

# --- MAIN ---
def main():
    global latest_packet, running, cube_pose_target, cube_pose_current, ai_active, homing_complete
    
    print(f"Connecting to {PORT}...")
    try: ser = serial.Serial(PORT, BAUDRATE, timeout=0.05)
    except Exception as e: print(f"Serial Error: {e}"); return

    print(f"Loading {XML_PATH}...")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    if cube_geom_id != -1:
        new_half_size = (CUBE_SIZE_MM / 1000.0) / 2.0
        model.geom_size[cube_geom_id] = [new_half_size, new_half_size, new_half_size]

    # --- INITIAL SYNC ---
    print("Reading Initial Robot State...")
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

    print(f"Loading AI: {MODEL_PATH}...")
    ai_model = PPO.load(MODEL_PATH)
    expected_dim = ai_model.policy.observation_space.shape[0]
    print(f"[SYSTEM] Model expects {expected_dim} inputs.")

    t_robot = threading.Thread(target=sender_thread, args=(ser,))
    t_vision = threading.Thread(target=vision_worker)
    t_robot.start()
    t_vision.start()
    
    # --- START HOMING SEQUENCE ---
    perform_homing(model, data)
    homing_complete = True
    
    print("[READY] Press 'S' in Video Window to Start Task.")

    cube_pose_current = np.array([0.25, 0.1, FIXED_Z_HEIGHT, 1, 0, 0, 0])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and running:
            step_start = time.time()
            
            # --- VISION ---
            if cube_pose_target is not None:
                target_np = np.array(cube_pose_target)
                cube_pose_current = (cube_pose_current * (1 - SMOOTHING)) + (target_np * SMOOTHING)
                cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube")
                if cube_id != -1:
                    addr = model.jnt_qposadr[cube_id]
                    data.qpos[addr:addr+7] = cube_pose_current
                    vel_addr = model.jnt_dofadr[cube_id]
                    data.qvel[vel_addr:vel_addr+6] = 0.0

            # --- AI ---
            if ai_active:
                arm_qpos = data.qpos[:6] 
                cube_pos = data.qpos[6:9] 
                goal_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "push_goal")
                goal_pos = data.site_xpos[goal_id]
                
                # Dynamic Obs
                if expected_dim == 12:
                    obs = np.concatenate([arm_qpos, cube_pos, goal_pos])
                elif expected_dim == 18:
                    arm_qvel = data.qvel[:6]
                    obs = np.concatenate([arm_qpos, arm_qvel, cube_pos, goal_pos])
                elif expected_dim == 21:
                    arm_qvel = data.qvel[:6]
                    cube_vel = np.array([0.0, 0.0, 0.0]) 
                    obs = np.concatenate([arm_qpos, arm_qvel, cube_pos, cube_vel, goal_pos])
                else:
                    ai_active = False
                    continue

                obs = obs.astype(np.float32)
                action, _ = ai_model.predict(obs, deterministic=True)
                
                # Speed Limiter
                current_ctrl = data.ctrl[:6].copy()
                desired_delta = action * ACTION_SCALE 
                safe_delta = np.clip(desired_delta, -MAX_SPEED_RAD_PER_STEP, MAX_SPEED_RAD_PER_STEP)
                
                data.ctrl[:6] = current_ctrl + safe_delta
                data.ctrl[:] = np.clip(data.ctrl, model.actuator_ctrlrange[:,0], model.actuator_ctrlrange[:,1])

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
