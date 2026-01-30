import serial
import time
import mujoco
import mujoco.viewer
import numpy as np

# --- CONFIGURATION ---
PORT = "COM3"
BAUDRATE = 1000000
XML_PATH = "scene.xml"
LOOP_RATE = 60  # Hz (60Hz Sync Write is 4x faster per motor than 100Hz Round Robin)

# Map XML ACTUATOR names to Motor IDs
ACTUATOR_MAPPING = {
    "Rotation": 1,
    "Pitch": 2,
    "Elbow": 3,
    "Wrist_Pitch": 4,
    "Wrist_Roll": 5,
    "Jaw": 6
}

# List of joints that move in the OPPOSITE direction
REVERSE_JOINTS = ["Rotation"]

# --- YOUR CALIBRATED OFFSETS ---
MOTOR_OFFSETS = {
    "Rotation": 2038,
    "Pitch": 3076,
    "Elbow": 1001,
    "Wrist_Pitch": 2209,
    "Wrist_Roll": 3079,
    "Jaw": 2196,
}
# -------------------------------

# STS Specs
STEPS_PER_REV = 4096
SCALE = 2 * np.pi / STEPS_PER_REV

def set_torque(ser, id, enable):
    """ Enable/Disable Torque """
    val = 1 if enable else 0
    checksum = (~(id + 4 + 3 + 40 + val)) & 0xFF
    packet = [0xFF, 0xFF, id, 0x04, 0x03, 40, val, checksum]
    ser.write(bytearray(packet))
    time.sleep(0.002)

def sync_write_positions(ser, mapping, model, data, offsets, reverse_list):
    """
    Sends ONE high-speed packet to update ALL motors simultaneously.
    This fixes the jitter and lag.
    Protocol: [Header, ID(FE), Len, Instr(83), Addr(2A), LenData(02), ID1, Pos1, ID2, Pos2..., Checksum]
    """
    
    # 1. Start Payload: Address 42 (0x2A) + Data Length (2 bytes per motor)
    payload = [0x2A, 0x02] 
    
    # 2. Add data for every motor
    for name, id in mapping.items():
        try:
            # Get Simulation Value
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if act_id == -1: continue
            
            sim_rad = data.ctrl[act_id]
            offset = offsets.get(name, 2048)
            
            # Calculate Target Steps
            if name in reverse_list:
                target = offset - (sim_rad / SCALE)
            else:
                target = (sim_rad / SCALE) + offset
                
            # Clamp to valid range
            target = int(max(0, min(4096, target)))
            
            # Split into Low and High bytes
            val_L = target & 0xFF
            val_H = (target >> 8) & 0xFF
            
            # Add [ID, Low, High] to payload
            payload.extend([id, val_L, val_H])
            
        except:
            pass

    # 3. Construct Final Packet
    # ID 0xFE = Broadcast (All listen)
    # Instruction 0x83 = Sync Write
    packet_len = len(payload) + 2
    packet = [0xFF, 0xFF, 0xFE, packet_len, 0x83] + payload
    
    # 4. Calculate Checksum
    checksum_sum = 0xFE + packet_len + 0x83 + sum(payload)
    checksum = (~checksum_sum) & 0xFF
    packet.append(checksum)
    
    # 5. Send Packet
    ser.write(bytearray(packet))

def read_current_raw(ser, id):
    """ Read current position (used only once at startup) """
    checksum = (~(id + 4 + 2 + 0x38 + 2)) & 0xFF
    packet = [0xFF, 0xFF, id, 0x04, 0x02, 0x38, 0x02, checksum]
    ser.reset_input_buffer()
    ser.write(bytearray(packet))
    res = ser.read(8)
    if len(res) == 8 and res[0] == 0xFF:
        return res[5] + (res[6] << 8)
    return None

def main():
    print(f"Connecting to {PORT}...")
    try:
        # Timeout is 0 because Sync Write is "Fire and Forget"
        ser = serial.Serial(PORT, BAUDRATE, timeout=0) 
    except Exception as e:
        print(f"Serial Error: {e}")
        return

    print(f"Loading {XML_PATH}...")
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"XML Error: {e}")
        return

    # --- SAFETY SYNC ---
    print("\n--- SAFETY SYNC ---")
    print("Reading Real Robot positions...")
    
    for name, id in ACTUATOR_MAPPING.items():
        raw = read_current_raw(ser, id)
        if raw is not None:
            offset = MOTOR_OFFSETS.get(name, 2048)
            is_reversed = (name in REVERSE_JOINTS)

            if is_reversed:
                real_rad = (offset - raw) * SCALE
            else:
                real_rad = (raw - offset) * SCALE
            
            try:
                act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                if act_id != -1:
                    data.ctrl[act_id] = real_rad
                    # Force visual update too
                    joint_id = model.actuator_trnid[act_id, 0]
                    data.qpos[model.jnt_qposadr[joint_id]] = real_rad
            except:
                pass

    mujoco.mj_forward(model, data)
    print("Sim Sliders snapped to Real Robot.")
    
    print("Enabling Torque...")
    for i in range(1, 7):
        set_torque(ser, i, True)
    print("Ready! Control the robot via Sim Sliders.")

    # --- CONTROL LOOP ---
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # DEBUG: Watch Rotation Slider
            debug_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Rotation")
            if debug_act_id != -1:
                print(f"Slider 'Rotation': {data.ctrl[debug_act_id]:.2f} rad   ", end="\r")

            # ---------------------------------------------------------
            # SYNC WRITE (Update ALL motors in ONE packet)
            # ---------------------------------------------------------
            sync_write_positions(ser, ACTUATOR_MAPPING, model, data, MOTOR_OFFSETS, REVERSE_JOINTS)
            # ---------------------------------------------------------

            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Rate Limit (60Hz is optimal for smoothness)
            elapsed = time.time() - step_start
            time.sleep(max(0, (1.0/LOOP_RATE) - elapsed))

    print("\nStopping...")
    for i in range(1, 7):
        set_torque(ser, i, False)
    ser.close()

if __name__ == "__main__":
    main()