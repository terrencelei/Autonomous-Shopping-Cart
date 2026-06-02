"""
Quick serial sniffer for the ESP32 cart_motor firmware. Prints every
"E,<right>,<left>" packet that arrives on /dev/ttyUSB0 alongside the
per-update delta, mirroring the format of the latest dual_motor_test.ino:

    L=     0 (d   +0)   R=     7 (d   +1)  *R*

The *L* / *R* flags fire whenever that side's count changed since the
previous packet. Use to confirm cart_motor.ino is alive, that the deltas
during motor-driven phases are non-zero, and that pushing each wheel by
hand also produces deltas (the truth-test for real encoder signal).

    python3 sniff_encoder.py
"""

import serial
import time

PORT = "/dev/ttyUSB0"
BAUD = 115200
DURATION_S = 20

# Flip if a wheel's encoder count goes negative under forward motion.
RIGHT_SIGN  = -1
LEFT_SIGN = +1

s = serial.Serial(PORT, BAUD, timeout=1)
print(f"Watching {PORT} for {DURATION_S}s.")
print("Push RIGHT wheel hard for 5s, then LEFT wheel for 5s.\n")

end = time.time() + DURATION_S
prev_l = prev_r = None
line_count = 0
while time.time() < end:
    line = s.readline().decode(errors="ignore").strip()
    if not line:
        continue
    line_count += 1
    if line.startswith("E,"):
        try:
            _, l_s, r_s = line.split(",", 2)
            l = RIGHT_SIGN  * int(l_s)
            r = LEFT_SIGN * int(r_s)
        except ValueError:
            print(f"  bad packet: {line!r}")
            continue
        d_l = 0 if prev_l is None else l - prev_l
        d_r = 0 if prev_r is None else r - prev_r
        prev_l, prev_r = l, r
        flag = ""
        if d_l: flag += " *L*"
        if d_r: flag += " *R*"
        print(f"  L={l:7d} (d {d_l:+4d})   R={r:7d} (d {d_r:+4d}){flag}")
    else:
        print(f"  msg: {line}")

print(f"\nDone. Saw {line_count} lines.")
print(f"Final counts:  L={prev_l}  R={prev_r}")
