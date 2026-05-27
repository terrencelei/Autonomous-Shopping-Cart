"""
Quick serial sniffer for the ESP32. Prints every changed line from
/dev/ttyUSB0 for 15 seconds. Use to confirm cart_motor.ino is sending
"E,<left>,<right>" packets and that pushing each wheel by hand makes
the counts move.

    python3 sniff_encoder.py
"""

import serial
import time

PORT = "/dev/ttyUSB0"
BAUD = 115200
DURATION_S = 15

s = serial.Serial(PORT, BAUD, timeout=1)
print(f"Watching {PORT} for {DURATION_S}s. Push each wheel by hand.")
end = time.time() + DURATION_S
last = ""
while time.time() < end:
    line = s.readline().decode(errors="ignore").strip()
    if line and line != last:
        print(line)
        last = line
