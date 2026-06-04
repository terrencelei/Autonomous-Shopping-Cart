# Autonomous Shopping Cart

A self-following shopping cart that tracks a designated shopper and treats all other people as obstacles.

| Component | Role | Technology |
|-----------|------|------------|
| **Vision** | Sensing (primary) | SSD-MobileNetV2 on IMX500 NPU + ByteTrack |
| **UWB** | Sensing (backup) | Apple Ultra-Wideband, iPhone-to-iPhone ranging |
| **Pathfinder** | Planning | A\* + state machine on a known store map |
| **ESP32** | Motor control | Dual TB9051FTG drivers with quadrature encoders |

### Runtime pipeline

```
IMX500 camera
     │ detections (in-process)
     ▼
yolo_detect.py ── P.tick(target, obstacles) ──► Pathfinding_algorithm.py ── USB "L<rpm> R<rpm>" ──► ESP32 ── PWM ──► motors
                                                          ▲
                                                   "E,<l>,<r>" ── USB ──┘   (encoder feedback)
```

`yolo_detect.py` imports `Pathfinding_algorithm` directly. Each control tick it passes the locked shopper's distance/angle plus a list of all obstacle detections. The pathfinder integrates encoder odometry, runs A\* around the map's shelving grid, and sends wheel-velocity commands to the ESP32 over USB serial.

### Running the cart (one command)

`start_cart.sh` launches both Python processes detached from your SSH session, so they keep running after you log out:

```bash
ssh pi
cd Autonomous-Shopping-Cart
./start_cart.sh             # place cart at the start pose FIRST
exit                        # safe to disconnect — cart keeps going
```

To stop:

```bash
ssh pi
./stop_cart.sh
```

Logs are written to `logs/<timestamp>-pathfinder.log` and `logs/<timestamp>-vision.log`. Tail them while connected: `tail -f logs/*pathfinder.log`. Pass `--display` to `start_cart.sh` if you're sitting at the Pi's own desktop and want the OpenCV preview windows.

> **Important:** the cart starts driving the moment `start_cart.sh` finishes — place it at `START_POS` (default `[0.0, 0.0]`, heading `0`) before running.

### Autostart on boot (optional)

Three boot modes — pick one. `install.sh` is idempotent, so re-run it any time to switch.

```bash
cd Autonomous-Shopping-Cart
sudo ./systemd/install.sh --test   # run pathfinding_arc_test.py once at boot
sudo ./systemd/install.sh --live   # run the full cart stack at boot
sudo ./systemd/install.sh --none   # install units, autostart nothing
```

| Mode | What runs on boot | Use it for |
|---|---|---|
| `--test` | `pathfinding_arc_test.py` (oneshot, writes PNGs and exits) | Iterating on the controller without driving the cart |
| `--live` | `cart-pathfinder` + `cart-vision` services | Real operation |
| `--none` | nothing — use `./start_cart.sh` manually | When you want full control |

Useful operations once installed:

| Command | Effect |
|---|---|
| `systemctl status cart-pathfinder cart-vision cart-arc-test` | Health check |
| `journalctl -fu cart-pathfinder -fu cart-vision` | Live log stream (live mode) |
| `journalctl -u cart-arc-test -b` | Read this-boot's test run output (test mode) |
| `sudo systemctl start cart-pathfinder cart-vision` | Trigger live mode now (no reboot needed) |
| `sudo systemctl start cart-arc-test` | Trigger one arc-test run now |
| `sudo systemctl stop cart-vision cart-pathfinder` | Stop the live cart |

> **Safety (live mode only):** the cart drives the moment the Pi finishes booting. Either always place it on the start coordinate before powering on, or stay in `--test` / `--none` mode until you're ready.

In `--live` mode, **don't also run `./start_cart.sh`** — both copies would fight for `/dev/ttyUSB0` and the IMX500 camera. Pick one launch method per boot.

---

## Vision System (Primary)

A detection pipeline that runs entirely on the Raspberry Pi AI Camera (IMX500). The SSD-MobileNetV2 object detector executes on the IMX500's on-chip neural processor — no inference on the Pi CPU — and the result is streamed back over CSI alongside each frame. The host then runs ByteTrack to assign stable track IDs, locks onto the closest centred shopper, and maps everyone else as an obstacle.

### Setup (Raspberry Pi)

```bash
sudo apt install imx500-models python3-picamera2
cd vision
pip install -r requirements.txt
```

### Usage

```bash
python3 yolo_detect.py                # live camera, GUI windows
python3 yolo_detect.py --no-display   # headless (SSH)
python3 yolo_detect.py --no-drive     # camera and detection only, no motor commands
```

Press **Q** to quit. Headless mode is auto-enabled if neither `DISPLAY` nor `WAYLAND_DISPLAY` is set.

### Target Locking

Each frame, every detected person is scored by `distance_m + 0.3 × |angle_deg|`. The person with the lowest score is locked as **TARGET** (green box) — favouring whoever is closest and most centred. All others are labelled **OBSTACLE** (red box). The lock updates every frame.

### Distance Calibration

Raw depth is estimated from the bounding-box height using a pinhole model, then corrected:

```
reported_dist = (raw_depth - DISTANCE_OFFSET_M) * DISTANCE_SCALE
```

| Constant | File | Calibrated value |
|----------|------|-----------------|
| `DISTANCE_OFFSET_M` | `vision/yolo_detect.py` | `0.89` |
| `DISTANCE_SCALE` | `vision/yolo_detect.py` | `0.95` |

To recalibrate: stand at known distances (e.g. 1 m, 2 m, 3 m), record the reported values, and fit new constants so `(raw - DISTANCE_OFFSET_M) * DISTANCE_SCALE` equals the true distance at each point.

### Output Windows

| Window | Contents |
|--------|----------|
| **Cart View** | Annotated camera feed — bounding boxes, per-object distance/angle, FPS/latency overlay, mini relative map (top-right corner) |
| **World Map** | Bird's-eye absolute map: CHUNK_MAP grid, cart position + heading arrow (white), target (green dot), obstacles (red dots), current pathfinder mode |

---

## Pathfinder

`Pathfinding_algorithm.py` is the live planner. It is called directly by `yolo_detect.py` on each control tick (every 20 ms). It maintains the cart's pose via encoder odometry, runs the chase state machine on the store's grid map, and writes wheel-velocity commands to the ESP32 over USB.

### Setup

```bash
pip install pyserial numpy
```

### How It Works

1. **Odometry** — encoder deltas (`E,<l>,<r>` lines from the ESP32) are integrated each tick to maintain an absolute `(x, y, heading)` pose.
2. **Sighting conversion** — the vision system's `(dist_m, angle_deg)` is projected into map coordinates using the current pose.
3. **State machine** — `tick()` runs the priority-ordered state machine (see below).
4. **Drive** — `tick()` outputs `(v_right, v_left)` in m/s, converted to RPM and written as `L<rpm> R<rpm>\n` to the ESP32.

### State Machine

#### Follow states

| State | Behaviour |
|-------|-----------|
| **IN_VIEW** | Shopper visible. Cart drives parallel to the aisle, maintaining the aisle centre line. Centering correction only fires when the shopper drifts past `FOV_ENGAGE_DEG = 25°`. Forward speed is a PD controller on distance to the shopper. |
| **FOLLOW_GOAL** | Shopper just left view. Cart navigates via A\* to the shopper's last known map position. |

#### Obstacle avoidance states

Triggered from `IN_VIEW`, `FOLLOW_GOAL`, or `RETURN_CENTER` the moment any obstacle detection is received. The obstacle's world Y-coordinate is computed from its camera angle + current heading to determine which aisle edge to hug.

| State | Behaviour |
|-------|-----------|
| **AVOID_EDGE** | A\* route to the opposite aisle edge (`aisle_cy ± (AISLE_W/2 - AISLE_EDGE_MARGIN)`). Full speed. |
| **EDGE_FOLLOW** | At the edge, continues forward at full speed while maintaining lateral position at the edge. Stays here while any obstacle is visible. After all obstacles clear, travels a further `EDGE_CLEAR_DIST_M = 1.0 m` before exiting. Exits to **RETURN_CENTER** if shopper is visible, or **SPIN** if not. |
| **RETURN_CENTER** | A\* route back to the aisle centre line. Transitions to **IN_VIEW** on arrival if shopper is visible, otherwise falls into search. |

#### Search states

Entered when the shopper cannot be found after exhausting `FOLLOW_GOAL` or `RETURN_CENTER`, or when `EDGE_FOLLOW` clears with no shopper in view.

| State | Behaviour |
|-------|-----------|
| **SPIN** | Full 360° spin. Transitions to `EDGE_PATROL` if at a map edge, or `AISLE_TRAVERSE` if in the middle. |
| **EDGE_PATROL** | Moves along the map edge, pausing at each aisle mouth to peek in. |
| **EDGE_PEEK** | Sweeps heading across the aisle mouth FOV. Marks the aisle as checked, then back to `EDGE_PATROL`. |
| **AISLE_TRAVERSE** | Crosses the store floor to the opposite edge, marking traversed aisles as checked. |

### Calibration

| Constant | File | What to set |
|----------|------|-------------|
| `WHEEL_DIAMETER_M` | `Pathfinding_algorithm.py` | Drive wheel outer diameter |
| `TRACK_M` | `Pathfinding_algorithm.py` | Centre-to-centre wheel spacing |
| `ENCODER_PPR` | `Pathfinding_algorithm.py` | Pulses per revolution after gearbox + 4× quadrature |
| `GEAR_RATIO` | `Pathfinding_algorithm.py` | Motor gearbox ratio |
| `RIGHT_ENC_SIGN` / `LEFT_ENC_SIGN` | `Pathfinding_algorithm.py` | Flip to `+1` if that encoder counts backwards |
| `RIGHT_MOTOR_SIGN` / `LEFT_MOTOR_SIGN` | `Pathfinding_algorithm.py` | Flip to `+1` if that motor drives the wrong direction |

---

## ESP32 Motor Controller

`firmware/cart_motor/cart_motor.ino` is the Arduino sketch that runs on the ESP32. It receives wheel-velocity commands from the Pi over USB and drives two TB9051FTG-controlled motors, while reporting encoder counts back.

### Wiring

| Side | PWM A | PWM B | Encoder A | Encoder B |
|------|-------|-------|-----------|-----------|
| Right (M1)  | GPIO 18 | GPIO 19 | GPIO 32 | GPIO 33 |
| Left (M2) | GPIO 22 | GPIO 23 | GPIO 25 | GPIO 26 |

Connect the ESP32 to the Pi via the CP2102 USB-UART bridge; it will enumerate as `/dev/ttyUSB0`.

### Build

1. Install the [TB9051FTGMotorCarrier](https://github.com/pololu/tb9051ftg-motor-driver-carrier) library in Arduino IDE (Library Manager → search "TB9051FTG").
2. Open `firmware/cart_motor/cart_motor.ino`, select your ESP32 board, and Upload.

### Protocol

| Direction | Format | Meaning |
|-----------|--------|---------|
| Pi → ESP32 | `L<rpm> R<rpm>\n` | Target wheel RPM, each side (e.g. `L42.5 R-30.0\n`) |
| ESP32 → Pi | `E,<l>,<r>\n` | Cumulative signed encoder counts, sent every 50 ms |

### Safety

A 500 ms watchdog stops the motors if no `L… R…` line arrives — so a host crash, USB unplug, or Pi reboot cannot leave the cart driving away.

---

## Pathfinding Simulation (offline)

`pathfinding_sim.py` is a standalone 2D simulator of the cart's chase behaviour — no camera, no motors. Useful for tuning the state machine without hardware.

### What It Does

- Generates a store map with configurable aisles
- Simulates a **moving target** (shopper) navigating the aisles
- The cart chases using **A\* pathfinding** around shelving obstacles
- Outputs an MP4 animation (`pathfinding_sim.mp4`)

### Usage

```bash
pip install matplotlib numpy
python3 pathfinding_sim.py
```

---

## Hardware Tests (`pathfinding_arc_test.py`)

Standalone test script for verifying motor control and pathfinder logic without running the full cart stack. Each mode exercises a different part of the drive system in isolation.

All modes share these flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--no-drive` | off | Skip serial — simulate only, writes PNGs |
| `--port PORT` | `/dev/ttyUSB0` | Override ESP32 serial port |
| `--countdown N` | `3` | Seconds before motors start |
| `--duration S` | `0` | Run time in seconds; `0` = run until Ctrl-C |

### Modes

#### `--mode spin` — full 360° spin

Commands the cart to rotate one complete revolution at `MAX_TURN` and stops when the encoder tick count matches the expected 360° arc. Use it to verify encoder wiring, signs, and the PPR calibration.

```bash
python3 pathfinding_arc_test.py --mode spin
python3 pathfinding_arc_test.py --mode spin --no-drive   # simulated
```

#### `--mode slow-spin` — stall characterisation

Ramps wheel RPM up from `STALL_RAMP_UP_START_RPM` until encoders report movement, then ramps back down until the motors stall again. Reports the minimum RPM needed to overcome static friction. Run this once after assembly or after changing motors/gearboxes.

```bash
python3 pathfinding_arc_test.py --mode slow-spin
python3 pathfinding_arc_test.py --mode slow-spin --direction right --duration 60
```

#### `--mode center` — angle-only centering

Rotates the cart to keep the shopper centred in the FOV; no forward motion. Reads from the live IMX500 camera by default, or from UDP with `--source udp`, or from a fixed simulated angle with `--sim-angle`.

```bash
python3 pathfinding_arc_test.py --mode center                  # live camera
python3 pathfinding_arc_test.py --mode center --source udp     # UDP feed
python3 pathfinding_arc_test.py --mode center --sim-angle 15   # fixed +15° target
python3 pathfinding_arc_test.py --mode center --no-drive --sim-angle 15
```

#### `--mode follow` — distance-only following

Drives the cart forward and backward to maintain `HOLD_DIST` (2.0 m) from the target. **Omega is forced to zero** — no angle correction at all. Use this to isolate and tune the distance PD controller (`DIST_KP`, `DIST_KD`) independently of the steering loop.

Anti-stall kick (same parameters as center mode: `FOLLOW_KICK_RPM = 8`, `FOLLOW_KICK_RELEASE_TICKS = 30`) fires on startup and on every forward↔reverse direction change, releasing once either encoder accumulates 30 ticks.

```bash
python3 pathfinding_arc_test.py --mode follow                       # live UDP
python3 pathfinding_arc_test.py --mode follow --sim-dist 3.5        # fixed 3.5 m target
python3 pathfinding_arc_test.py --mode follow --no-drive --sim-dist 3.5 --duration 10
```

`--sim-dist` sets a constant target distance. The cart will drive toward `HOLD_DIST` and hold; pass a value above or below 2.0 m to test approach or reverse respectively. Values within `FOLLOW_DIST_DEADBAND_M` (0.05 m) of `HOLD_DIST` produce no output.

### Output

Each run saves one or two PNG plots to the working directory (`pathfinding_arc_test.png`, `pathfinding_arc_test_encoders.png`). When run via the `--test` systemd unit the PNGs land in `/home/terrencelei/Autonomous-Shopping-Cart/`.

---

## UWB System (Backup / Redundancy)

Two iPhones use Apple's Ultra-Wideband chip to maintain a precise distance and angle measurement between the shopper and the cart, independent of the camera.

### Apps

#### Shopper (Tag App) — `uwb/UWBCart/`
Runs on the **shopper's iPhone**. Advertises over MultipeerConnectivity and acts as a UWB beacon.

#### CartView (Viewer App) — `uwb/ViewerApp/`
Runs on the **cart's iPhone**. Displays:
- Top-down radar with the shopper's position
- Smoothed distance in metres
- Smoothed horizontal angle
- Auto-scaling range rings

### How It Works

1. Shopper app advertises via MultipeerConnectivity (`_uwb-cart._tcp`)
2. CartView discovers and connects to the Shopper
3. Both devices exchange NearbyInteraction discovery tokens
4. UWB ranging begins — distance and angle update continuously
5. Sessions auto-restart if the peer goes out of range

### Smoothing & Calibration

Raw UWB readings are filtered through an EMA (α = 0.2). Both distance and angle support zeroing:
- **Zero Dist** — samples 20 readings and averages for the offset
- **Zero Angle** — captures current heading as the zero reference
- **Reset All** — clears both offsets

Offsets persist in UserDefaults across launches.

### Requirements

- Two iPhones with U2 chip (iPhone 14 Pro or later for angle support)
- iOS 16.0+
- Both devices on the same local network

---

## Project Structure

```
autonomous-shopping-cart/
├── vision/                         # Camera-based detection
│   ├── yolo_detect.py              # Detection + tracking + pathfinder integration
│   ├── throttle_watch.sh           # CPU/thermal throttle monitor
│   └── requirements.txt
├── Pathfinding_algorithm.py        # A* pathfinder + state machine
├── pathfinding_arc_test.py         # Hardware test: spin / slow-spin / center / follow modes
├── pathfinding_sim.py              # 2D simulator used to design the planner
├── start_cart.sh / stop_cart.sh    # Launch / stop the full stack detached from SSH
├── firmware/
│   └── cart_motor/
│       └── cart_motor.ino          # ESP32 motor controller (TB9051FTG)
├── uwb/                            # Backup: UWB positioning
│   ├── UWBCart/                    # Shopper app source
│   ├── ViewerApp/                  # CartView app source
│   └── UWBCart.xcodeproj
└── README.md
```

---

## Troubleshooting

**Camera busy error on Pi:**
A previous process is still holding the camera. Kill it:
```bash
sudo pkill -f yolo_detect.py
```

**NPU model not found:**
Install pre-built models: `sudo apt install imx500-models`

**`qt.qpa.xcb: could not connect to display` over SSH:**
Run with `--no-display`, or run from the Pi's own desktop terminal.

**`imx500_transition_to_network: unable to apply register writes from firmware` (in dmesg):**
The `.rpk` is incompatible with the current `imx500-firmware`. Reinstall the matching model package: `sudo apt install --reinstall imx500-models`.

**Cart doesn't move even though detections look fine:**
Check that the ESP32 is connected and `/dev/ttyUSB0` is accessible. Confirm `yolo_detect.py` is not running with `--no-drive`.

**`could not open /dev/ttyUSB0`:**
Check the ESP32 is plugged in: `ls /dev/ttyUSB*`. Add the user to the `dialout` group if it's a permission error: `sudo usermod -aG dialout $USER` and re-login.

**Motors run but speed doesn't match commands:**
Calibrate `MAX_RPM` in `firmware/cart_motor/cart_motor.ino`. Run the cart at full output for a fixed time, read the final `E,<l>,<r>` line, and compute `rpm = (ticks / ENCODER_PPR) * (60 / seconds)`.

**Cart drifts off-track after a few metres:**
Check `RIGHT_ENC_SIGN` / `LEFT_ENC_SIGN` — if either encoder counts the wrong direction the odometry will diverge quickly. Verify with: push the cart forward by hand and check that both encoder counts in the `E,<l>,<r>` stream increase.

**Obstacle avoidance not triggering:**
The cart only avoids from `IN_VIEW`, `FOLLOW_GOAL`, and `RETURN_CENTER`. If the cart is in a search state (`SPIN` etc.) when an obstacle appears, avoidance does not interrupt it. This is by design — avoidance is only relevant while actively following.

**Xcode "Executable is not codesigned":**
1. **Product → Clean Build Folder** (⇧⌘K)
2. **Settings → General → VPN & Device Management → Trust** on the iPhone
3. Reconnect and run again

**UWB angle shows nil:**
- Grant camera permission to CartView
- Move the cart phone slightly to initialise ARKit world tracking

**UWB disconnects frequently:**
Keep both phones within ~10m with clear line of sight. Metal shelving attenuates the UWB signal. Sessions auto-restart on reconnect.
