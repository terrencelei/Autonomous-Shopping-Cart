# Autonomous Shopping Cart

A self-following shopping cart that tracks a designated shopper and treats all other people as obstacles. Two independent sensing systems provide redundancy:

| System | Role | Technology |
|--------|------|------------|
| **Vision** | Primary | SSD-MobileNetV2 on IMX500 NPU + ByteTrack |
| **UWB** | Backup / redundancy | Apple Ultra-Wideband, iPhone-to-iPhone ranging |

The vision system handles detection and scene understanding — identifying the target shopper and classifying other people as obstacles. The UWB system provides a precise distance and angle fallback when the camera view is obstructed or the target is temporarily lost.

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
```

Press **Q** to quit.

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

## Pathfinding Simulation

`pathfinding_sim.py` is a standalone 2D simulation of the cart's chase behaviour on a store map. It does not use the camera — it simulates what the motor controller should do given a known target position.

### What It Does

- Generates a 20×20 metre store map with 10 aisles
- Simulates a **moving target** (shopper) navigating the aisles at 1.8 m/s
- The **robot** (cart) chases using a **bubble chase** algorithm: it always aims for a point 0.5 m behind the target rather than the target itself, avoiding collisions
- Uses **A\* pathfinding** to navigate around shelving obstacles
- Replans every 10 frames to adapt to target movement
- Outputs a 300-frame MP4 animation (`pathfinding_sim.mp4`)

### Algorithm

```
score = distance + 0.3 × |angle|   ← target selection (vision)
chase_point = target + 0.5m bubble ← motor goal
path = A*(robot, chase_point)       ← obstacle-aware routing
```

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
├── vision/                         # Primary: camera-based detection
│   ├── yolo_detect.py              # Detection + tracking + overhead map
│   ├── throttle_watch.sh           # CPU/thermal throttle monitor
│   └── requirements.txt
├── uwb/                            # Backup: UWB positioning
│   ├── UWBCart/                    # Shopper app source
│   ├── ViewerApp/                  # CartView app source
│   └── UWBCart.xcodeproj
├── pathfinding_sim.py              # 2D pathfinding simulation (A* + bubble chase)
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

**Xcode "Executable is not codesigned":**
1. **Product → Clean Build Folder** (⇧⌘K)
2. **Settings → General → VPN & Device Management → Trust** on the iPhone
3. Reconnect and run again

**UWB angle shows nil:**
- Grant camera permission to CartView
- Move the cart phone slightly to initialise ARKit world tracking

**UWB disconnects frequently:**
Keep both phones within ~10m with clear line of sight. Metal shelving attenuates the UWB signal. Sessions auto-restart on reconnect.
