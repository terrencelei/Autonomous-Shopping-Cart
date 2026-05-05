# UWBCart — Autonomous Shopping Cart System

A two-part system for tracking and following a shopper with a smart cart:
- **UWB positioning** — two iPhones use Apple's Ultra-Wideband chip to measure precise distance and angle between shopper and cart in real time
- **Machine vision** — a YOLO-based computer vision pipeline detects and tracks people and shopping carts from a camera feed, with center-frame target locking and a live overhead map

Together these provide the sensing layer for an autonomous following cart: UWB gives precise shopper location, and vision identifies the target and classifies all other people and carts as obstacles.

---

## UWB Positioning (iOS Apps)

Uses two iPhones — one carried by the shopper (**Shopper**) and one mounted on the cart (**CartView**). The devices discover each other over MultipeerConnectivity, exchange NearbyInteraction discovery tokens, then use the U2 UWB chip combined with ARKit camera assistance to measure precise distance and horizontal angle.

CartView displays the shopper's position on a top-down radar in real time, with smoothed distance and angle readouts and persistent zeroing for both.

### Apps

#### Shopper (Tag App)
Runs on the **shopper's iPhone**. Advertises itself over the local network and acts as a UWB beacon. No interaction required — the shopper just keeps the app open.

#### CartView (Viewer App)
Runs on the **cart's iPhone**. Discovers the Shopper, connects, and displays:
- Top-down radar with the shopper's relative position (orange dot)
- Smoothed distance in metres
- Smoothed horizontal angle in radians and degrees
- Auto-scaling range rings

### Connection Flow

1. Shopper app starts advertising via MultipeerConnectivity (`_uwb-cart._tcp`)
2. CartView browses the local network and invites the Shopper
3. Encrypted MC session is established
4. Both devices exchange NearbyInteraction discovery tokens over MC (reliable delivery)
5. Both sides call `NISession.run()` with the peer's token — UWB ranging begins
6. CartView receives distance and angle updates via `NISessionDelegate`
7. Sessions automatically restart if the peer goes out of range or the app is suspended

### Angle Measurement

Horizontal angle uses `NINearbyObject.horizontalAngle` — the azimuthal bearing to the peer in radians. On U2 devices (iPhone 14 Pro and later), `supportsDirectionMeasurement` is false for peer-to-peer sessions, so camera assistance is required to activate the angle algorithm.

CartView enables `isCameraAssistanceEnabled = true` on its `NINearbyPeerConfiguration`. This activates an internal ARKit world-tracking session that fuses IMU and camera data to produce a world-space angle compensated for the cart phone's own movement and rotation.

**The cart camera does not need to point at the shopper.** It tracks the environment (walls, floor, shelves) for ego-motion estimation only. The shopper is located purely by UWB signal.

### Smoothing

Raw UWB readings are noisy. Both distance and horizontal angle pass through an exponential moving average (EMA) filter before display:

```
output = α × new_sample + (1 − α) × previous_output   (α = 0.2)
```

### Calibration

#### Distance
1. Hold both phones at your chosen zero reference point
2. Tap **Zero Dist** — CartView samples 20 readings (~2 s) and averages them
3. All subsequent distance readings subtract the offset (clamped to ≥ 0)

#### Angle
1. Point the cart phone in the direction you want to treat as 0°
2. Tap **Zero Angle** — the current smoothed angle is captured as the reference
3. All subsequent angle readings are relative to that direction, normalised to [−180°, +180°]

### Requirements

- Two iPhones with a U2 chip (iPhone 14 Pro or later recommended for angle support)
- iOS 16.0+
- Both devices on the same local network (for MultipeerConnectivity discovery)
- Camera permission granted on the CartView phone

---

## Machine Vision (Python)

A YOLO11-based detection and tracking pipeline that runs on a video feed (webcam, video file, or image). Detects people and shopping carts, tracks them with ByteTrack, estimates distance and angle, and locks onto the shopper closest to the center of the frame.

### Setup

```bash
cd machine_vision
pip install -r requirements.txt
```

### Usage

```bash
python3 yolo_detect.py                  # test image (Inside_Costco_Perth.jpg)
python3 yolo_detect.py video.mp4        # video file
python3 yolo_detect.py 0                # webcam
```

Press **Q** to quit the live view.

### How It Works

Two models run in parallel on each frame:
- `yolo11n.pt` — COCO pretrained model, used only for person detection (class 0)
- `cart.pt` — fine-tuned on a shopping cart dataset (Roboflow), detects `shoppingcart`

ByteTrack assigns persistent IDs across frames. The person whose bounding box center is closest to the frame center is locked as **TARGET** (cyan). Everyone else — people and carts — is labelled **OBSTACLE** (red).

Two windows display simultaneously:
- **ByteTrack** — annotated camera feed with bounding boxes, labels, distance, and angle
- **Overhead Map** — live top-down 2D map showing all detections relative to the camera, with distance rings and FOV cone

### Training the Cart Model

Download the dataset from Roboflow and fine-tune:

```bash
python3 train_cart.py --api-key <YOUR_ROBOFLOW_KEY>
```

The trained model is saved to `cart.pt`. The pre-trained model is included in this repo.

### Cart Model Performance

Trained on the [Shopping Cart dataset](https://universe.roboflow.com/furkan-bakkal/shopping-cart-1r48s) (215 train / 61 val images, 1 class):

| Metric | Value |
|--------|-------|
| mAP50  | 0.793 |
| Precision | 88.5% |
| Recall | 70.6% |

---

## Project Structure

```
UWBCart/
├── UWBCart/                        # Shopper app (shopper's phone)
│   ├── UWBCartApp.swift
│   ├── ContentView.swift
│   ├── TagSessionManager.swift
│   └── Info.plist
├── ViewerApp/                      # CartView app (cart phone)
│   ├── ViewerAppApp.swift
│   ├── ContentView.swift
│   ├── ViewerSessionManager.swift
│   └── Info.plist
├── machine_vision/                 # Vision pipeline (Python)
│   ├── yolo_detect.py              # Detection + tracking + overhead map
│   ├── train_cart.py               # Fine-tune cart detector
│   ├── cart.pt                     # Trained cart detection model
│   ├── yolo11n.pt                  # Base COCO model (person detection)
│   ├── requirements.txt
│   └── Inside_Costco_Perth.jpg     # Test image
├── UWBCartTests/
├── UWBCartUITests/
├── ViewerAppTests/
└── ViewerAppUITests/
```

---

## Troubleshooting

**"Executable is not codesigned" when running ViewerApp:**
1. In Xcode: **Product → Clean Build Folder** (⇧⌘K)
2. On the iPhone: **Settings → General → VPN & Device Management → [your Apple ID] → Trust**
3. Reconnect the iPhone and run again from Xcode

**Angle shows "nil" and never appears:**
- Check **Settings → Privacy & Security → Camera → CartView** is set to Allow
- Move the cart phone slightly — ARKit needs a moment of motion to initialise world tracking

**UWB session disconnects frequently:**
Keep both phones within ~10m with clear line of sight. Metal shelving and walls attenuate the UWB signal. The session auto-restarts on reconnect.

**Camera permission denied for webcam (Python):**
Open **System Settings → Privacy & Security → Camera** and enable access for your terminal app.
