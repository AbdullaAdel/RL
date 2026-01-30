import serial
import time
import mujoco
import mujoco.viewer
import numpy as np

# --- CONFIGURATION ---
XML_PATH = "scene.xml"  # Ensure this matches your file name
PORT = "COM3"              # Your port
BAUDRATE = 1000000

# ---------------------------------------------------------
# SETUP SERIAL
# ---------------------------------------------------------
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=0.05)
    print(f"[SUCCESS] Opened {PORT}")
except Exception as e:
    print(f"[ERROR] Could not open port: {e}")
    exit()

def read_servo(id):
    """ Reads position from STS servo. Returns Raw Steps (0-4096). """
    # Checksum = (~(ID + Length + Instruction + Param + LenRead)) & 0xFF
    checksum = (~(id + 4 + 2 + 0x38 + 2)) & 0xFF
    packet = [0xFF, 0xFF, id, 0x04, 0x02, 0x38, 0x02, checksum]
    
    ser.reset_input_buffer()
    ser.write(bytearray(packet))
    res = ser.read(8)
    
    if len(res) == 8 and res[0] == 0xFF:
        val = res[5] + (res[6] << 8)
        return val
    return None

# ---------------------------------------------------------
# MAIN DEBUG LOOP
# ---------------------------------------------------------
def main():
    print(f"Loading MuJoCo model: {XML_PATH}...")
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"[FATAL] Cannot load XML: {e}")
        return

    # 1. PRINT DETECTED JOINTS (Check this output carefully!)
    print("\n--- CHECKING XML JOINTS ---")
    sim_joint_names = []
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        sim_joint_names.append(name)
        print(f"  > Found Joint ID {i}: '{name}'")
    print("---------------------------\n")

    # 2. DEFINE MAPPING (Edit this if your names above are different)
    # Format: "XML_Joint_Name": Real_Servo_ID
    # If your XML names are different (e.g., 'base', 'shoulder'), CHANGE THEM HERE.
    mapping = {
        "joint1": 1,
        "joint2": 2,
        "joint3": 3,
        "joint4": 4,
        "joint5": 5,
        "gripper": 6
    }

    # 3. VERIFY MAPPING
    print("--- VERIFYING MAPPING ---")
    valid_map = {}
    for name, id in mapping.items():
        if name in sim_joint_names:
            print(f"  [OK] Mapping '{name}' -> Servo {id}")
            valid_map[name] = id
        else:
            print(f"  [FAIL] '{name}' is in your Python script but NOT in the XML file!")
            print(f"         (The robot will NOT move this joint)")
    print("-------------------------\n")

    if not valid_map:
        print("[ERROR] No valid joints found. Check your XML names!")
        return

    print("Starting Loop. Press Ctrl+C to stop.")
    
    # Disable torque so you can move it by hand
    for i in range(1, 7):
        # Torque Disable Packet
        chk = (~(i + 5 + 3 + 40 + 0)) & 0xFF
        ser.write(bytearray([0xFF, 0xFF, i, 0x05, 0x03, 40, 0, chk]))
        time.sleep(0.01)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # Read and Update
            debug_str = ""
            for j_name, s_id in valid_map.items():
                raw = read_servo(s_id)
                
                if raw is not None:
                    # Convert 0-4096 -> -3.14 to +3.14 radians
                    # Adjust '2048' if your robot's center is different
                    rad = (raw - 2048) * (2 * np.pi / 4096)
                    
                    # Update Sim
                    try:
                        joint_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
                        addr = model.jnt_qposadr[joint_idx]
                        data.qpos[addr] = rad
                        
                        # Add to debug print (only show first 3 for brevity)
                        if s_id <= 3:
                            debug_str += f"{j_name}:{rad:.2f} "
                    except:
                        pass
            
            # Print status to verify data is flowing
            print(f"\r{debug_str}      ", end="")

            mujoco.mj_forward(model, data)
            viewer.sync()
            
            # 30Hz Limit
            time.sleep(0.033)

if __name__ == "__main__":
    main()