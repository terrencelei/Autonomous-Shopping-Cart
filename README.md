# Autonomous Shopping Cart

A self-following shopping cart that tracks a designated shopper and treats all other people and carts as obstacles. Two independent sensing systems provide redundancy:

| System | Role | Technology |
|--------|------|------------|
| **Vision** | Primary | YOLO26n + ByteTrack, camera-based detection |
| **UWB** | Backup / redundancy | Apple Ultra-Wideband, iPhone-to-iPhone ranging |

The vision system handles detection and scene understanding — identifying the target shopper and classifying obstacles. The UWB system provides a precise distance and angle fallback when the camera view is obstructed or the target is temporarily lost.

---

## Vision System (Primary)

A YOLO26n-based pipeline that detects people and shopping carts from a live camera feed, locks onto the closest centred shopper, and maps everything else as an obstacle.

### Setup

```bash
cd vision
pip install -r requirements.txt
```

### Usage

```bash
python3 yolo_detect.py image.jpg        # image file
python3 yolo_detect.py video.mp4        # video file
```

Press **Q** to quit.

### Target Locking

Each frame, every detected person is scored by `distance_m + 0.3 × |angle_deg|`. The person with the lowest score is locked as **TARGET** (green box) — favouring whoever is closest and most centred. All other detections are labelled **OBSTACLE** (red box). The lock updates every frame as people move.

### Output

Two windows run simultaneously:
- **ByteTrack** — annotated camera feed with bounding boxes, per-object distance and angle
- **Overhead Map** — live top-down 2D map of all detections relative to the camera, with distance rings and FOV cone

### Models

| Model | Purpose |
|-------|---------|
| `yolo26n.pt` | COCO pretrained — person detection only |
| `cart.pt` | Fine-tuned on shopping cart dataset — cart detection |

Both pre-trained models are included. To retrain the cart model:

```bash
python3 train_cart.py --api-key <YOUR_ROBOFLOW_KEY>
```

Dataset: [Shopping Cart — Roboflow](https://universe.roboflow.com/furkan-bakkal/shopping-cart-1r48s) (215 train / 61 val images)

### Cart Model Performance

| Metric | Value |
|--------|-------|
| mAP50 | 0.793 |
| Precision | 88.5% |
| Recall | 70.6% |

---

## UWB System (Backup / Redundancy)

Two iPhones use Apple's Ultra-Wideband chip to maintain a precise distance and angle measurement between the shopper and the cart, independent of the camera. This serves as a fallback when the vision system loses the target.

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

### Angle Measurement

Uses `NINearbyObject.horizontalAngle` with `isCameraAssistanceEnabled = true` on the cart phone. This activates an ARKit world-tracking session that compensates for cart movement — **the camera does not need to point at the shopper.**

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
│   ├── train_cart.py               # Fine-tune cart detector
│   ├── cart.pt                     # Trained cart detection model
│   ├── yolo26n.pt                  # Base COCO model (person detection)
│   └── requirements.txt
├── uwb/                            # Backup: UWB positioning
│   ├── UWBCart/                    # Shopper app source
│   ├── ViewerApp/                  # CartView app source
│   ├── UWBCart.xcodeproj
│   ├── UWBCartTests/
│   ├── UWBCartUITests/
│   ├── ViewerAppTests/
│   └── ViewerAppUITests/
└── README.md
```

---

## Troubleshooting

**Camera permission denied (vision system):**
Open **System Settings → Privacy & Security → Camera** and enable access for your terminal app.

**Xcode "Executable is not codesigned":**
1. **Product → Clean Build Folder** (⇧⌘K)
2. **Settings → General → VPN & Device Management → Trust** on the iPhone
3. Reconnect and run again

**UWB angle shows nil:**
- Grant camera permission to CartView
- Move the cart phone slightly to initialise ARKit world tracking

**UWB disconnects frequently:**
Keep both phones within ~10m with clear line of sight. Metal shelving attenuates the UWB signal. Sessions auto-restart on reconnect.
