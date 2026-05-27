# Autonomous Shopping Cart

A self-following shopping cart that tracks a designated shopper and treats all other people as obstacles.

| Component | Role | Technology |
|-----------|------|------------|
| **Vision** | Sensing (primary) | SSD-MobileNetV2 on IMX500 NPU + ByteTrack |
| **UWB** | Sensing (backup) | Apple Ultra-Wideband, iPhone-to-iPhone ranging |
| **Pathfinder** | Planning | A\* + state machine on a known 20 × 20 m store map |
| **ESP32** | Motor control | Dual TB9051FTG drivers with quadrature encoders |

### Runtime pipeline

```
yolo_detect.py ─UDP "dist,ang"─► Pathfinding_algorithm.py ─USB "L<rpm> R<rpm>"─► ESP32 ─PWM─► motors
                                                                ▲
                                                         "E,<l>,<r>" ─USB─┘   (encoder feedback)
```

The vision system handles detection and target locking; it streams the locked shopper's distance and bearing over UDP to the pathfinder. The pathfinder converts that to map coordinates using the cart's known starting pose plus odometry, runs A\* around shelving obstacles, and sends wheel-velocity commands over USB serial to the ESP32, which drives the motors. The UWB system provides a precise distance/angle fallback when the camera view is obstructed.

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

Logs are written to `logs/<timestamp>-pathfinder.log` and `logs/<timestamp>-vision.log`. Tail them while connected: `tail -f logs/*pathfinder.log`. Pass `--display` to `start_cart.sh` if you're sitting at the Pi's own desktop and want the OpenCV preview window.

> **Important:** the cart starts driving the moment `start_cart.sh` finishes — place it at `START_POS` (default `[0.0, 0.0]`, heading `0`) before running.

### Autostart on boot (optional)

To have the cart run automatically every time the Pi powers up, install the systemd services:

```bash
cd Autonomous-Shopping-Cart
sudo ./systemd/install.sh                # install + enable autostart
sudo ./systemd/install.sh --no-enable    # install but don't autostart yet
```

Then either reboot (`sudo reboot`) or start the services now:

```bash
sudo systemctl start cart-pathfinder cart-vision
```

Useful operations:

| Command | Effect |
|---|---|
| `systemctl status cart-pathfinder cart-vision` | One-shot health check |
| `journalctl -fu cart-pathfinder -fu cart-vision` | Live merged log stream |
| `sudo systemctl stop cart-vision cart-pathfinder` | Stop now (boot autostart unchanged) |
| `sudo systemctl disable cart-pathfinder cart-vision` | Remove from boot — manual launch only |
| `sudo systemctl enable cart-pathfinder cart-vision` | Restore autostart |

> **Safety:** autostart means the cart drives the moment the Pi finishes booting. Either always place it on the start coordinate before powering on, or leave autostart disabled and use `./start_cart.sh` manually.

When autostart is enabled, **don't also run `./start_cart.sh`** — both copies would fight for `/dev/ttyUSB0` and the IMX500 camera. Pick one launch method per boot.

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
python3 yolo_detect.py                              # live camera, GUI windows
python3 yolo_detect.py --no-display                 # headless (SSH)
python3 yolo_detect.py --no-udp                     # don't stream to pathfinder
python3 yolo_detect.py --pathfinder-host 10.0.0.5   # pathfinder on a different host
```

Press **Q** to quit. Headless mode is auto-enabled if neither `DISPLAY` nor `WAYLAND_DISPLAY` is set.

By default, every frame's locked TARGET is published to `127.0.0.1:5005` as a UDP packet `"<distance_m>,<angle_deg>\n"`. The pathfinder (running on the same Pi) listens on that port.

### Target Locking

Each frame, every detected person is scored by `distance_m + 0.3 × |angle_deg|`. The person with the lowest score is locked as **TARGET** (green box) — favouring whoever is closest and most centred. All others are labelled **OBSTACLE** (red box). The lock updates every frame as people move.

### Output

A single **Cart View** window shows:
- Annotated camera feed with bounding boxes, per-object distance and angle
- Mini overhead map (picture-in-picture, top-right corner) showing all detections relative to the cart
- FPS and latency overlay

### Model

| Model | Purpose |
|-------|---------|
| `imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk` | COCO-pretrained SSD-MobileNetV2 detector, runs on the IMX500 NPU |

Shipped by the `imx500-models` apt package and loaded from `/usr/share/imx500-models/`. The `.rpk` is uploaded over CSI to the sensor at startup (~3 s).

---

## Pathfinder (Hardware Runtime)

`Pathfinding_algorithm.py` is the live planner that runs on the cart. It listens on UDP port 5005 for target sightings from the vision system, runs the chase state machine on a 20 × 20 m store map, and writes wheel-velocity commands to the ESP32 over USB.

### Setup

```bash
pip install pyserial numpy
```

### Usage

Place the cart at the configured start pose (`START_POS = [0.5, 0.5]`, `START_HEADING = 0` rad — facing +x) and run:

```bash
python3 Pathfinding_algorithm.py
```

It will keep running until interrupted. On exit it sends a zero-velocity stop command.

### How It Works

1. **Receive** — a background thread reads UDP packets `"<dist_m>,<angle_deg>\n"` from `yolo_detect.py`. Readings older than 0.5 s are treated as "no target".
2. **Convert** — the relative sighting is transformed into an absolute `(x, y)` on the map using the cart's current pose (start pose + integrated odometry).
3. **Plan** — `tick()` runs the state machine: **IN_VIEW** (chase) → **FOLLOW_GOAL** (head to last-known) → **SPIN** → **EDGE_PATROL** / **EDGE_PEEK** → **AISLE_TRAVERSE**. A\* re-routes around shelving when the direct path is blocked.
4. **Drive** — `tick()` outputs `(v_forward, omega)`, which is split into left/right wheel surface speeds, converted to RPM, and written to the ESP32 as `"L<rpm> R<rpm>\n"`.

### Calibration

These constants must match your hardware before driving:

| Constant | Where | What to set |
|----------|-------|-------------|
| `WHEEL_DIAMETER_M` | `Pathfinding_algorithm.py` | Drive wheel diameter |
| `WHEEL_TRACK_M`    | `Pathfinding_algorithm.py` | Centre-to-centre between wheels |
| `ENCODER_PPR`      | `Pathfinding_algorithm.py` | Pulses per revolution after gearbox |
| `START_POS`, `START_HEADING` | `Pathfinding_algorithm.py` | Cart's physical start spot on the map |
| `MAX_RPM`          | `firmware/cart_motor/cart_motor.ino` | Wheel RPM at full motor output |

> **Note:** the on-Pi `Odometry` class is currently a stub (`_read_encoder_deltas` returns 0, 0). The ESP32 already streams encoder counts back at 20 Hz on the same serial port (`"E,<left>,<right>\n"`); wire that into `Odometry._read_encoder_deltas` to enable closed-loop pose tracking. Until then, IN_VIEW following still works (the relative→absolute conversion cancels out the stale pose), but the patrol states will misbehave.

---

## ESP32 Motor Controller

`firmware/cart_motor/cart_motor.ino` is the Arduino sketch that runs on the ESP32. It receives wheel-velocity commands from the Pi over USB and drives two TB9051FTG-controlled motors, while reporting encoder counts back.

### Wiring

| Side | PWM A | PWM B | Encoder A | Encoder B |
|------|-------|-------|-----------|-----------|
| Left (M1)  | GPIO 18 | GPIO 19 | GPIO 32 | GPIO 33 |
| Right (M2) | GPIO 22 | GPIO 23 | GPIO 25 | GPIO 26 |

Connect the ESP32 to the Pi via the micro-USB port; it will enumerate as `/dev/ttyACM0`.

### Build

1. Install the [TB9051FTGMotorCarrier](https://github.com/pololu/tb9051ftg-motor-driver-carrier) library in Arduino IDE (Library Manager → search "TB9051FTG").
2. Open `firmware/cart_motor/cart_motor.ino`, select your ESP32 board, and Upload.

### Protocol

| Direction | Format | Meaning |
|-----------|--------|---------|
| Pi → ESP32 | `L<rpm> R<rpm>\n` | Target wheel RPM, each side (e.g. `L42.5 R-30.0\n`) |
| ESP32 → Pi | `E,<l>,<r>\n` | Cumulative encoder counts, sent every 50 ms |

### Safety

A 500 ms watchdog stops the motors if no `L… R…` line arrives — so a host crash, USB unplug, or Pi reboot cannot leave the cart driving away.

---

## Pathfinding Simulation (offline)

`pathfinding_sim.py` is a standalone 2D simulator of the cart's chase behaviour — no camera, no motors. Useful for tuning the state machine without hardware. The runtime pathfinder above (`Pathfinding_algorithm.py`) is derived from this sim.

### What It Does

- Generates a 20 × 20 m store map with 10 aisles
- Simulates a **moving target** (shopper) navigating the aisles at 1.8 m/s
- The cart chases using **A\* pathfinding** around shelving obstacles, replanning every 10 frames
- Outputs an MP4 animation (`pathfinding_sim.mp4`)

### Usage

```bash
pip install matplotlib numpy
python3 pathfinding_sim.py
```

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
│   ├── yolo_detect.py              # Detection + tracking + UDP target feed
│   ├── throttle_watch.sh           # CPU/thermal throttle monitor
│   └── requirements.txt
├── Pathfinding_algorithm.py        # Live A* pathfinder (UDP in → USB out)
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
Check that `Pathfinding_algorithm.py` is actually running and listening on UDP 5005. From another terminal: `sudo lsof -iUDP:5005` should show the python process. If vision and pathfinder are on different hosts, pass `--pathfinder-host <ip>` to `yolo_detect.py`.

**`could not open /dev/ttyACM0` from Pathfinding_algorithm.py:**
Check the ESP32 is plugged in: `ls /dev/ttyACM*`. Add the user to the `dialout` group if it's a permission error: `sudo usermod -aG dialout $USER` and re-login.

**Motors run but speed doesn't match commands:**
Calibrate `MAX_RPM` in `firmware/cart_motor/cart_motor.ino`. Run the cart at full output for a fixed time, read the final `E,<l>,<r>` line, and compute `rpm = (ticks / ENCODER_PPR) * (60 / seconds)`.

**Cart drifts off-track after a few metres:**
Odometry is stubbed — the pathfinder thinks the cart never moves. Wire the ESP32's `E,<l>,<r>` stream into `Odometry._read_encoder_deltas` for closed-loop pose tracking.

**Xcode "Executable is not codesigned":**
1. **Product → Clean Build Folder** (⇧⌘K)
2. **Settings → General → VPN & Device Management → Trust** on the iPhone
3. Reconnect and run again

**UWB angle shows nil:**
- Grant camera permission to CartView
- Move the cart phone slightly to initialise ARKit world tracking

**UWB disconnects frequently:**
Keep both phones within ~10m with clear line of sight. Metal shelving attenuates the UWB signal. Sessions auto-restart on reconnect.
