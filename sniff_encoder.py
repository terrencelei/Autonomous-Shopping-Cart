"""
Quick serial sniffer for the ESP32. Prints every line from /dev/ttyUSB0
for 20 seconds and flags whenever the encoder counts change.

Use to confirm cart_motor.ino is alive and that pushing each wheel by
hand makes its counter move.

    python3 sniff_encoder.py
"""

import serial
import time

PORT = "/dev/ttyUSB0"
BAUD = 115200
DURATION_S = 20

s = serial.Serial(PORT, BAUD, timeout=1)
print(f"Watching {PORT} for {DURATION_S}s.")
print("Push LEFT wheel hard for 5s, then RIGHT wheel for 5s.\n")

end = time.time() + DURATION_S
last_l = last_r = None
line_count = 0
while time.time() < end:
    line = s.readline().decode(errors="ignore").strip()
    if not line:
        continue
    line_count += 1
    if line.startswith("E,"):
        try:
            _, l_s, r_s = line.split(",", 2)
            l, r = int(l_s), int(r_s)
        except ValueError:
            print(f"  bad packet: {line!r}")
            continue
        flag = ""
        if last_l is not None and l != last_l:
            flag += " *L*"
        if last_r is not None and r != last_r:
            flag += " *R*"
        last_l, last_r = l, r
        print(f"  E,L={l:6d}  R={r:6d}{flag}")
    else:
        print(f"  msg: {line}")

print(f"\nDone. Saw {line_count} lines.")
print(f"Final counts:  L={last_l}  R={last_r}")
