import serial
import time
import mujoco
import mujoco.viewer
import numpy as np

# --- CONFIGURATION ---
PORT = "COM3"
BAUDRATE = 1000000
XML_PATH = "scene.xml"
KEYFRAME_ID = 1  # The pose you want to match
CAPTURE_DELAY = 15

# Map XML names to Motor IDs
JOINT_MAPPING = {
    "Rotation": 1,
    "Pitch": 2,
    "Elbow": 3,
    "Wrist_Pitch": 4,
    "Wrist_Roll": 5,
    "Jaw": 6
}

# STS3215 Specs
STEPS_PER_REV = 4096
SCALE = 2 * np.pi / STEPS_PER_REV  # Radians per step

def set_torque(ser, id, enable):
    """
    FIXED: Packet Length changed from 5 to 4.
    Now the motors will actually accept the command to go limp.
    """
    val = 1 if enable else 0
    # Checksum: ~(ID + Length(4) + Instr(3) + Addr(40) + Val)
    checksum = (~(id + 4 + 3 + 40 + val)) & 0xFF
    
    # Packet: [FF, FF, ID, LEN=4, INSTR=3, ADDR=40, VAL, CHK]
    packet = [0xFF, 0xFF, id, 0x04, 0x03, 40, val, checksum]
    
    ser.write(bytearray(packet))
    time.sleep(0.005)

def read_raw(ser, id):
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
        ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
    except Exception as e:
        print(f"Error: {e}")
        return

    # 1. DISABLE TORQUE (This makes the robot limp)
    print("Disabling Torque... (You should be able to move the robot now)")
    for i in range(1, 7):
        set_torque(ser, i, False)

    # 2. Load Model
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # 3. Get Keyframe Angles (The "Target" Angles)
    if model.nkey > 0:
        if KEYFRAME_ID < model.nkey:
            key_qpos = model.key_qpos[KEYFRAME_ID]
            print(f"Loaded Keyframe #{KEYFRAME_ID}")
        else:
            print(f"Error: KEYFRAME_ID {KEYFRAME_ID} does not exist (Max is {model.nkey-1})")
            return
    else:
        print("Error: No Keyframes found in XML!")
        return

    print("\n" + "="*50)
    print("      SMART CALIBRATION WIZARD")
    print("="*50)
    print(f"1. Match the REAL robot to the SIM robot.")
    print(f"2. You have {CAPTURE_DELAY} seconds.")
    print("="*50 + "\n")

    start_time = time.time()
    captured = False

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Force Sim to Keyframe
            data.qpos[:] = key_qpos
            mujoco.mj_forward(model, data)
            viewer.sync()

            elapsed = time.time() - start_time
            remaining = CAPTURE_DELAY - elapsed

            if remaining > 0:
                print(f"Capturing in {int(remaining)}... ", end='\r')
                time.sleep(0.1)
            elif not captured:
                print("\n\n[CALCULATING TRUE ZEROS]...\n")
                
                print("MOTOR_OFFSETS = {")
                for name, id in JOINT_MAPPING.items():
                    # 1. Get the Target Angle from MuJoCo (Radians)
                    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    qpos_adr = model.jnt_qposadr[joint_id]
                    target_angle = key_qpos[qpos_adr]

                    # 2. Get the Actual Raw Step from Robot (Steps)
                    raw_current = read_raw(ser, id)
                    if raw_current is None: raw_current = 2048

                    # 3. Calculate "True Zero" Offset
                    # Formula: Angle = (Raw - Offset) * Scale
                    # Therefore: Offset = Raw - (Angle / Scale)
                    
                    true_zero = raw_current - (target_angle / SCALE)
                    
                    # Round to nearest integer step
                    print(f'    "{name}": {int(true_zero)},  # ID {id} (Target Angle: {target_angle:.2f})')

                print("}")
                captured = True
                print("\n[DONE] Paste these values into your sync script.")
                print("Note: Values might look weird (e.g. -500 or 3000). That is OK!")
                print("They represent where the motor WOULD be if it were fully straight.")

    ser.close()

if __name__ == "__main__":
    main()