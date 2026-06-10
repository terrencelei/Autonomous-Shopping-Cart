# Autonomous Shopping Cart

> **🛒 New here? Take the friendly tour first → [terrencelei.github.io/Autonomous-Shopping-Cart](https://terrencelei.github.io/Autonomous-Shopping-Cart/)** — a visual, non-technical introduction to the project.

A self-following shopping cart that tracks a designated shopper and treats all other people as obstacles. The main program is **`follow.py`** — a single-process reactive follower on the Raspberry Pi.

| Component | Role | Technology |
|-----------|------|------------|
| **Vision** | Sensing | YOLO11n on IMX500 NPU + ByteTrack (`vision/yolo_detect.py`) |
| **Follower** | Control | `follow.py` — continuous damped-P steering + distance follow |
| **ESP32** | Motor control | Dual TB9051FTG drivers with quadrature encoders |

### Runtime pipeline

```
IMX500 camera ──► follow.py ───────────────────► ESP32 ──► motors
 (NPU detections)  imports vision/yolo_detect    USB "L<rpm> R<rpm>"
                   for target/obstacle tracking,
                   runs the control loop
       ▲                                              │
       └───────────── "E,<l>,<r>" encoder feedback ◄──┘
```

`follow.py` is a single process: it imports `vision/yolo_detect.py` for the calibrated target distance/angle (and the obstacle list), runs its control loop, and drives the ESP32 directly over USB serial. There is no A\* planner and no store map.

### Running the cart

```bash
ssh pi
cd Autonomous-Shopping-Cart
python3 follow.py                 # live camera + drive
python3 follow.py --no-display    # headless (SSH)
python3 follow.py --no-drive      # vision only, no serial output
python3 follow.py --trace         # echo serial traffic + control-flow calls
```

Stop with **Ctrl-C** (or **Q** in the Cart View window). Headless mode is auto-enabled if no display is detected. `follow_straight.py` is a distance-only (no steering) variant with the same flags.

> **Startup sequence:** `follow.py` first runs a **15 s vision warmup** (`VISION_WARMUP_S`) so ByteTrack and the color tracking settle on the shopper — stand in view during this. After the `--countdown` delay it starts following immediately; there is no calibration run — both inertia constants are learned on the fly (see *Inertia learning*). **Safety:** keep the cart's path clear when it starts.

### Autostart on boot (optional)

Install the systemd unit to launch `follow.py` headless on every boot:

```bash
sudo cp systemd/cart-follow.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now cart-follow     # start now + on every boot
```

Manage it with `sudo systemctl {status,stop,disable} cart-follow` and `journalctl -fu cart-follow`. **Safety:** with the unit enabled the cart starts following the moment the Pi boots — keep it clear, or leave the unit disabled until you're ready.

---

## Vision System

A detection pipeline that runs entirely on the Raspberry Pi AI Camera (IMX500). The YOLO11n object detector executes on the IMX500's on-chip neural processor — no inference on the Pi CPU — and the result is streamed back over CSI alongside each frame. The host uses OpenCV (`cv2`) for image conversion, overlays, and display windows, then runs ByteTrack to assign stable track IDs, locks onto the closest centred shopper, and maps everyone else as an obstacle.

The default model is Raspberry Pi's shipped YOLO11n post-processed RPK:

```text
/usr/share/imx500-models/imx500_network_yolo11n_pp.rpk
```

`yolo_detect.py` applies the matching Picamera2 demo settings: bounding-box normalization enabled and `bbox_order = xy`. YOLO11n is a COCO detector, so `PERSON_CLASS_ID = 0` still selects people. The Raspberry Pi model zoo lists the YOLO11n RPK under AGPL-3.0.

### Setup (Raspberry Pi)

```bash
sudo apt install imx500-models python3-picamera2
cd vision
pip install -r requirements.txt
```

`vision/requirements.txt` includes `opencv-python`, `supervision`, and `pyserial`. If OpenCV import fails, reinstall the Python dependencies from the `vision/` directory.

### Usage

```bash
python3 yolo_detect.py                # vision-only preview, GUI window
python3 yolo_detect.py --no-display   # headless (SSH)
```

Run directly, `yolo_detect.py` is a **vision-only preview** (no motor control) — it shows the annotated feed and prints the per-detection table. The actual driving lives in `follow.py`, which imports this module for its tracking. Press **Q** to quit; headless mode is auto-enabled if neither `DISPLAY` nor `WAYLAND_DISPLAY` is set.

### Target Locking

Each frame, every detected person is scored by `distance_m + 0.3 × |angle_deg|`. The person with the lowest score is locked as **TARGET** (green box) — favouring whoever is closest and most centred. All others are labelled **OBSTACLE** (red box). The lock updates every frame.

### Distance Calibration

Raw depth is estimated from the bounding-box height using a pinhole model, then corrected:

```
reported_dist = (raw_depth - DISTANCE_OFFSET_M) * DISTANCE_SCALE
```

| Constant | File | Calibrated value |
|----------|------|-----------------|
| `DISTANCE_OFFSET_M` | `vision/yolo_detect.py` | `0` |
| `DISTANCE_SCALE` | `vision/yolo_detect.py` | `0.7` |

To recalibrate: stand at known distances (e.g. 1 m, 2 m, 3 m), record the reported values, and fit new constants so `(raw - DISTANCE_OFFSET_M) * DISTANCE_SCALE` equals the true distance at each point. Recheck this after changing detector models because YOLO11n bounding boxes may be tighter or taller than SSD-MobileNetV2 boxes.

### Output Windows

| Window | Contents |
|--------|----------|
| **Cart View** | OpenCV window with annotated camera feed — bounding boxes, per-object distance/angle, FPS/latency overlay |

---

## ESP32 Motor Controller

`firmware/cart_motor/cart_motor.ino` is the Arduino sketch that runs on the ESP32. It receives `L<rpm> R<rpm>` commands from the Pi over USB and drives two TB9051FTG-controlled motors open-loop, while reporting encoder counts back.

The firmware currently treats RPM commands as an open-loop speed scale: `MAX_RPM = 100` maps to full PWM. For example, `L31 R31` is roughly the same output as `firmware/constant_motor_speed/constant_motor_speed.ino` with `MOTOR_DUTY = 80`.

### Wiring

| Side | PWM A | PWM B | Encoder A | Encoder B |
|------|-------|-------|-----------|-----------|
| Right (M1)  | GPIO 18 | GPIO 19 | GPIO 32 | GPIO 33 |
| Left (M2) | GPIO 22 | GPIO 23 | GPIO 25 | GPIO 26 |

Connect the ESP32 to the Pi via the CP2102 USB-UART bridge; it will enumerate as `/dev/ttyUSB0`.

### Build

Open `firmware/cart_motor/cart_motor.ino`, select your ESP32 board, and Upload.

### Protocol

| Direction | Format | Meaning |
|-----------|--------|---------|
| Pi → ESP32 | `L<rpm> R<rpm>\n` | Open-loop speed command, each side (e.g. `L42.5 R-30.0\n`) |
| ESP32 → Pi | `E,<l>,<r>\n` | Cumulative signed encoder counts, sent every 50 ms |

### Constant-speed motor test

After flashing `firmware/cart_motor/cart_motor.ino`, run both motors at a constant open-loop speed from the Pi:

```bash
python3 -c 'import serial,time
ser=serial.Serial("/dev/ttyUSB0",115200,timeout=0.1)
print("running L31 R31, Ctrl-C to stop")
try:
    while True:
        ser.write(b"L31 R31\n")
        while ser.in_waiting:
            print(ser.readline().decode(errors="replace").rstrip())
        time.sleep(0.1)
except KeyboardInterrupt:
    ser.write(b"L0 R0\n")
    ser.close()
    print("stopped")'
```

Use `L10 R10` for a slower test, `L50 R50` for a faster one, and `L0 R0` to stop. Keep sending the command at least every 500 ms, or the ESP32 watchdog will stop the motors.

There is also a standalone Arduino test sketch at `firmware/constant_motor_speed/constant_motor_speed.ino`. It does not require serial commands; it starts both motors at `MOTOR_DUTY` after upload and prints encoder counts.

### Safety

A 500 ms watchdog stops the motors if no `L… R…` line arrives — so a host crash, USB unplug, or Pi reboot cannot leave the cart driving away.

---

## Standalone Follower (`follow.py`)

A single-process reactive follower — no A\* and no store map. It imports `vision/yolo_detect.py` for the calibrated target distance/angle and drives the ESP32 over the RPM protocol directly. All hardware/calibration constants live at the top of `follow.py`; both control axes (distance and steering) are continuous damped-P laws with online-learned coast compensation.

```bash
python3 follow.py                 # live camera + drive
python3 follow.py --no-display    # headless (SSH)
python3 follow.py --no-drive      # vision only, no serial output
python3 follow.py --duration 30   # stop after 30 s (0 = until Ctrl-C / Q)
python3 follow.py --trace         # echo serial traffic + control-flow calls
```

`--trace` turns the terminal into a live monitor: each tick it prints the controller state (`step: dist… ang…`), kick arming/release, inertia-coast measurements, the raw encoder packets received (`RX  E,<l>,<r>`) with the net delta, and the wheel commands sent (`TX  L<rpm> R<rpm>`).

### Control loop

Each frame it reads the locked shopper's `(distance, angle)` and then:

1. **Centering — continuous damped-P steering.** Every tick the steering difference is `rpm_diff = KP_ANGLE·(θ − ω·ANGULAR_INERTIA)`: the target angle minus the yaw the cart will still coast through at its measured yaw rate ω, so turns ease off early instead of overshooting. A small `ANGLE_DEADBAND_DEG` ignores vision angle noise and the command is slew-limited (`DIFF_SLEW_PER_S`). There is no spin threshold and no discrete spin manoeuvre — the shopper is steered back to centre continuously while driving (the old `Spinner` pulsed a point turn every time the angle crossed 18°, the angular twin of the forward stop/kick limit cycle). A *standing* re-centre that can't break static friction is kicked free at `SPIN_KICK_RPM` by the stall watch — armed only when at least `TURN_STALL_MIN_RPM` of steering is demanded, so noise inside the deadband can never trigger a pulse.
2. **Distance — damped P.** Inside the centre band it holds `THRESH_M` (= `HOLD_DIST`, 2 m) with the same law `follow_straight.py` uses: `S = KP_DIST·(error − deadband) + KD_DIST·dx`, slew-limited (`RPM_SLEW_UP_PER_S` / `RPM_SLEW_DOWN_PER_S`) and clipped to `MAX_RPM` (= 100 rpm, the firmware's full-PWM point). The error is **coast-compensated**: `error = (x − thresh) − v·LINEAR_INERTIA`, where `LINEAR_INERTIA` is learned online from every commanded stop (coast distance ÷ speed, by the same `InertiaLearner` that tunes the yaw axis), so the cart brakes early and rolls into the ring instead of through it. A `DIST_DEADBAND_M` no-drive band plus `RESUME_HYST_M` restart hysteresis keeps vision noise at the 2 m boundary from toggling stop/kick/stop. On departure from rest a separate `Kick` floors the forward speed at `KICK_RPM` (= 50 rpm) until `KICK_TICKS` of encoder movement confirm breakaway; if still stalled after `KICK_TIMEOUT_S` the kick ramps up by `KICK_RAMP_STEP` (a stall watch also re-arms it if a low-RPM crawl stalls later). A momentary target dropout no longer brake-slams: during `SEARCH_GRACE_S` the cart coasts down on the slew instead.
3. **Mixing.** The forward speed and steering difference are mixed onto the wheels (`R = S + rpm_diff`, `L = S − rpm_diff`) with peak scaling so neither wheel exceeds `MAX_RPM`, then sent as `L<rpm> R<rpm>`.
4. **Lost → reacquire.** If the shopper leaves the frame, the cart first coasts down through a `SEARCH_GRACE_S` grace window (a one-frame dropout doesn't jerk it), then the `Searcher` **spins in place toward the side the shopper was last seen**. It spins **forever** (never gives up), reversing direction each full 360° and ramping its RPM if stalled, until a target reappears (then normal follow resumes) — `Mode: SEARCH`.
5. **Obstacle hold.** Each tick, if any `OBSTACLE` detection is closer than the target (by at least `OBSTACLE_MARGIN_M`) and within `±OBSTACLE_BLOCK_DEG` of straight ahead, forward motion is paused (`S = 0`, `Mode: BLOCKED`) — the cart holds (steering still keeps it aimed) rather than driving into the obstacle. To ride through the noisy per-frame detections, the block is held for `OBSTACLE_HOLD_S` after the last in-path sighting (hysteresis), so a one-frame dropout doesn't make it lurch forward. It resumes once the obstacle stays clear. Intentionally minimal: it only *stops*, it does not route around.

### Return-home (spun-around recovery)

`follow.py` dead-reckons its pose from the encoders the whole time (`Odometry`, origin = the start pose). If the cart is **physically rotated ~180° while it was commanding a full stop** — i.e. an *uncommanded* spin, someone turned it around — it abandons following and drives **back to the start position and heading** along a direct line (`ReturnHome`, `Mode: RETURN`, then `HOME` on arrival):

1. Turn to face the start position.
2. Drive straight to it (steering on the odometry bearing to stay on the line).
3. Turn to the original heading.

Detection: each tick it accumulates the **measured yaw minus the commanded yaw** (a manual spin adds to the measured rotation but not the commanded, so it registers whether the cart is following, stopped, or searching; a normal commanded spin tracks and doesn't drift). Crossing `HOME_ROT_TRIGGER_DEG ± HOME_ROT_RANGE_DEG` (180° ± 30°) triggers the return. Tunables: `HOME_TURN_RPM`, `HOME_FWD_RPM`, `HOME_DRIVE_KP`, `HOME_ANGLE_TOL_DEG`, `HOME_DIST_TOL_M`. Because it's dead-reckoning-only, accuracy degrades with how far/long the cart has driven (encoder drift).

**Slip rejection.** The odometry runs on slip-corrected encoder deltas (`deslip()`). A free-spinning wheel reads more ticks than the cart actually moved; if the two wheels diverge beyond the *commanded* differential by more than `SLIP_DIFF_TICKS_PER_S`, the faster wheel is slipping, so the odometry trusts the slower (gripping) wheel and rebuilds the faster one from it plus the commanded turn. Real commanded turns are preserved; only the un-commanded excess is dropped. (Encoder-only, so it handles *one* wheel slipping, not both — an IMU would be the robust fix.)

### Inertia learning (no startup calibration)

Both coast constants are learned online by a shared `InertiaLearner` (one instance per axis) — there is no calibration run at launch:

- **`ANGULAR_INERTIA`** (yaw coast, seconds) seeds at 0.15 s and is refined every time a turn is commanded to a stop: the learner integrates how much yaw the encoders coast through and divides by the yaw rate at the stop. Search-spin exits, standing re-centres and return-home turns all feed it, and the steering law uses it immediately (`θ − ω·ANGULAR_INERTIA`).
- **`LINEAR_INERTIA`** (forward coast) starts at 0 and learns the same way from every commanded forward stop, shrinking the standoff error by the predicted coast distance.

Rates are taken from the encoders (the serial link is open-loop PWM, so commanded RPM is not a real speed). Each measurement is EMA-blended (`ALPHA = 0.3`, capped at 1 s); current values are shown in the periodic status line (`AI=… LI=…`). The gains at the top of `follow.py` marked `[calibrate]` are hand-tuned values.

The Cart View overlay and terminal monitor show `follow.py`'s own state each tick — `FOLLOW`, `KICK`, `TURN`, `STOP`, `BLOCKED`, `SEARCH`, `RETURN`, `HOME`, or `LOST`.

---

## Straight Follower (`follow_straight.py`)

`follow_straight.py` is the distance-only follower. It never steers on the camera angle (that input is dropped from its controller entirely) — it only drives forward to hold `HOLD_DIST` from the locked shopper. It is **forward-only**: when the shopper is at or closer than the hold distance it stops, never reverses. A damped proportional speed (`KP_DIST·error + KD_DIST·dx`, slew-limited by `RPM_SLEW_PER_S`) avoids the overshoot/stop cycles of an integrating controller.

```bash
python3 follow_straight.py                 # live camera + drive
python3 follow_straight.py --no-display    # headless (SSH)
python3 follow_straight.py --no-drive      # vision only, no serial output
python3 follow_straight.py --duration 30   # stop after 30 s
```

**Straight-line heading hold.** Because the open-loop ESP32 firmware turns equal RPM commands into equal *PWM* — not equal *speed* — the cart drifts on uneven friction. To counter that without steering on the target, `follow_straight` adds a small **encoder-driven** trim: it accumulates the left-vs-right encoder imbalance and nudges the lagging wheel up (`STRAIGHT_KP`, `STRAIGHT_KD`, capped at `STRAIGHT_TRIM_MAX`) to keep the two wheels' travel equal. It resets at each stop/new segment and never reverses a wheel. This holds the heading it set off with — it is *not* camera/angle steering.

Current key constants:

| Constant | Value | Meaning |
|----------|-------|---------|
| `MAX_RPM` | `75.0` | Maximum open-loop command sent to each wheel |
| `KICK_RPM` / `KICK_TICKS` | `18.0` / `45` | Breakaway floor when starting from rest, released after this many ticks |
| `DIST_DEADBAND_M` | `0.20` | No-drive band beyond the 2 m hold distance |
| `KP_DIST` / `KD_DIST` | `30.0` / `8.0` | Forward speed per metre of standoff error / per (m/s) closing rate |
| `RPM_SLEW_PER_S` | `35.0` | Max command change per second (smooths the approach) |
| `STRAIGHT_KP` / `STRAIGHT_KD` / `STRAIGHT_TRIM_MAX` | `0.20` / `2.0` / `15.0` | Encoder heading-hold trim gains and cap |

The terminal monitor prints distance, angle, smoothed distance rate, commanded RPM, and encoder-derived actual RPM, for example:

```text
cmd L+18.0 R+18.0rpm  actual L+14.2 R+13.9rpm
```

---

## Project Structure

```
autonomous-shopping-cart/
├── follow.py                       # MAIN PROGRAM — reactive follower (continuous damped-P
│                                   #   steering + distance, obstacle hold, search, return-home)
├── follow_straight.py              # Distance-only (no steering) follower variant
├── sniff_encoder.py                # Serial encoder sniffer (debug)
├── vision/                         # Camera tracking, imported by follow.py
│   ├── yolo_detect.py              # IMX500 + ByteTrack target/obstacle tracking (vision-only)
│   └── requirements.txt
├── firmware/
│   ├── cart_motor/
│   │   └── cart_motor.ino          # ESP32 open-loop motor controller
│   └── constant_motor_speed/
│       └── constant_motor_speed.ino # Standalone constant-speed motor test
├── systemd/
│   └── cart-follow.service         # Optional boot autostart for follow.py
└── README.md
```

---

## Troubleshooting

**Camera busy error on Pi:**
A previous process is still holding the camera. Kill it:
```bash
sudo pkill -f follow.py
```

**NPU model not found:**
Install pre-built models: `sudo apt install imx500-models`

**`qt.qpa.xcb: could not connect to display` over SSH:**
Run with `--no-display`, or run from the Pi's own desktop terminal.

**`imx500_transition_to_network: unable to apply register writes from firmware` (in dmesg):**
The `.rpk` is incompatible with the current `imx500-firmware`. Reinstall the matching model package: `sudo apt install --reinstall imx500-models`.

**Cart doesn't move even though detections look fine:**
Check that the ESP32 is connected and `/dev/ttyUSB0` is accessible. Confirm `follow.py` is not running with `--no-drive`, and watch its console for `ESP32 connected on …` (not the `WARNING: ESP32 unavailable` line).

**`could not open /dev/ttyUSB0`:**
Check the ESP32 is plugged in: `ls /dev/ttyUSB*`. Add the user to the `dialout` group if it's a permission error: `sudo usermod -aG dialout $USER` and re-login.

**Motors run but speed doesn't match commands:**
Calibrate `MAX_RPM` in `firmware/cart_motor/cart_motor.ino`. Run the cart at full output for a fixed time, read the final `E,<l>,<r>` line, and compute `rpm = (ticks / ENCODER_PPR) * (60 / seconds)`.

**Cart drifts off-track after a few metres:**
Check `RIGHT_ENC_SIGN` / `LEFT_ENC_SIGN` — if either encoder counts the wrong direction the odometry will diverge quickly. Verify with: push the cart forward by hand and check that both encoder counts in the `E,<l>,<r>` stream increase.

**Obstacle avoidance not triggering:**
The obstacle hold only runs while a TARGET is visible (`FOLLOW`/`KICK`/`TURN` ticks). In `SEARCH` and `RETURN` the cart does not check obstacles. This is by design — avoidance is only relevant while actively following.
