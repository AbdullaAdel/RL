import serial
import time
import numpy as np

# CONFIG
PORT = "COM3"
BAUDRATE = 1000000 # Try 1000000 first. If that fails, try 500000 or 115200.

def read_pos(ser, id):
    # STS Protocol Checksum
    # (~(ID + Length + Instruction + Param + LenRead)) & 0xFF
    checksum = (~(id + 4 + 2 + 0x38 + 2)) & 0xFF
    packet = [0xFF, 0xFF, id, 0x04, 0x02, 0x38, 0x02, checksum]
    
    ser.reset_input_buffer()
    ser.write(bytearray(packet))
    res = ser.read(8) # Expecting 8 bytes back
    
    if len(res) == 8 and res[0] == 0xFF and res[1] == 0xFF:
        val = res[5] + (res[6] << 8)
        return val
    return None

try:
    print(f"Opening {PORT} at {BAUDRATE}...")
    ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
    print("Port Open. Scanning IDs 1-6...")
    
    while True:
        output = ""
        for i in range(1, 7):
            pos = read_pos(ser, i)
            if pos is not None:
                output += f"[ID {i}: {pos}] "
            else:
                output += f"[ID {i}: --] "
        
        print(f"\r{output}", end="")
        time.sleep(0.1)

except Exception as e:
    print(f"\nError: {e}")