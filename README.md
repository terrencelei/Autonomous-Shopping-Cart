# UWBCart

A two-device iOS system that tracks a shopper's position relative to a shopping cart in real time using Apple's Ultra-Wideband (UWB) technology.

## How It Works

UWBCart uses two iPhones — one carried by the shopper (**Shopper**) and one mounted on the cart (**CartView**). The devices discover each other over MultipeerConnectivity, exchange NearbyInteraction discovery tokens, then use the U2 UWB chip combined with ARKit camera assistance to measure precise distance and horizontal angle.

CartView displays the shopper's position on a top-down radar in real time, with smoothed distance and angle readouts and persistent zeroing for both.

## Apps

### Shopper (Tag App)
Runs on the **shopper's iPhone**. Advertises itself over the local network and acts as a UWB beacon. No interaction required — the shopper just keeps the app open.

### CartView (Viewer App)
Runs on the **cart's iPhone**. Discovers the Shopper, connects, and displays:
- Top-down radar with the shopper's relative position (orange dot)
- Smoothed distance in metres
- Smoothed horizontal angle in radians and degrees
- Auto-scaling range rings

## Connection Flow

1. Shopper app starts advertising via MultipeerConnectivity (`_uwb-cart._tcp`)
2. CartView browses the local network and invites the Shopper
3. Encrypted MC session is established
4. Both devices exchange NearbyInteraction discovery tokens over MC (reliable delivery)
5. Both sides call `NISession.run()` with the peer's token — UWB ranging begins
6. CartView receives distance and angle updates via `NISessionDelegate`
7. Sessions automatically restart if the peer goes out of range or the app is suspended

## Angle Measurement

Horizontal angle uses `NINearbyObject.horizontalAngle` — the azimuthal bearing to the peer in radians. On U2 devices (iPhone 14 Pro and later), `supportsDirectionMeasurement` is false for peer-to-peer sessions, so camera assistance is required to activate the angle algorithm.

CartView enables `isCameraAssistanceEnabled = true` on its `NINearbyPeerConfiguration`. This activates an internal ARKit world-tracking session that fuses IMU and camera data to produce a world-space angle that is compensated for the cart phone's own movement and rotation.

**The cart camera does not need to point at the shopper.** It tracks the environment (walls, floor, shelves) for ego-motion estimation only. The shopper is located purely by UWB signal.

The status bar shows **"Move phone to establish angle…"** while ARKit is converging (typically a few seconds of movement) and **"Angle locked"** once stable. A stationary cart phone converges via IMU alone.

Both phones moving is fully supported — world-space angle compensation handles cart movement automatically.

## Smoothing

Raw UWB readings are noisy. Both distance and horizontal angle pass through an exponential moving average (EMA) filter before display:

```
output = α × new_sample + (1 − α) × previous_output   (α = 0.2)
```

A low alpha responds instantly to large movements while damping high-frequency jitter. Buffers reset on disconnect so stale data never bleeds into a new session.

## Calibration

### Distance
UWB time-of-flight includes a fixed hardware offset — adjacent phones never read zero.

1. Hold both phones at your chosen zero reference point
2. Tap **Zero Dist** — CartView samples 20 readings (~2 s) and averages them
3. All subsequent distance readings subtract the offset (clamped to ≥ 0)

### Angle
Point the cart phone in the direction you want to treat as 0°.

1. Tap **Zero Angle** — the current smoothed angle is captured as the reference
2. All subsequent angle readings are relative to that direction, normalised to [−180°, +180°]

Both offsets persist in UserDefaults across app launches. **Reset All** clears both at once.

## Requirements

- Two iPhones with a U2 chip (iPhone 14 Pro or later recommended for angle support)
- iOS 16.0+
- Both devices on the same local network (for MultipeerConnectivity discovery)
- Camera permission granted on the CartView phone

## Permissions

| App | Key | Purpose |
|-----|-----|---------|
| Both | `NSLocalNetworkUsageDescription` | MultipeerConnectivity device discovery |
| Both | `NSBonjourServices` | Bonjour service registration (`_uwb-cart._tcp`) |
| Both | `NSNearbyInteractionUsageDescription` | UWB distance and angle measurement |
| CartView | `NSCameraUsageDescription` | ARKit camera assistance for angle computation |

## Project Structure

```
UWBCart/
├── UWBCart/                       # Shopper app (shopper's phone)
│   ├── UWBCartApp.swift           # App entry point
│   ├── ContentView.swift          # Pulse animation UI
│   ├── TagSessionManager.swift    # MC advertiser + NI session
│   └── Info.plist
├── ViewerApp/                     # CartView app (cart phone)
│   ├── ViewerAppApp.swift         # App entry point
│   ├── ContentView.swift          # Radar, readout, calibration buttons
│   ├── ViewerSessionManager.swift # MC browser + NI session + smoothing + calibration
│   └── Info.plist
├── UWBCartTests/
├── UWBCartUITests/
├── ViewerAppTests/
└── ViewerAppUITests/
```

## Troubleshooting

**"Executable is not codesigned" when running ViewerApp:**
1. In Xcode: **Product → Clean Build Folder** (⇧⌘K)
2. On the iPhone: **Settings → General → VPN & Device Management → [your Apple ID] → Trust**
3. Reconnect the iPhone and run again from Xcode

**Angle shows "nil" and never appears:**
- Check **Settings → Privacy & Security → Camera → CartView** is set to Allow
- Move the cart phone slightly — ARKit needs a moment of motion to initialise world tracking
- A stationary phone on a flat surface converges via IMU and also works

**Angle is unstable or drifting:**
- Ensure the cart phone is rigidly mounted; vibration confuses the IMU
- In very featureless environments ARKit relies more on IMU and may drift slightly over time — re-zero the angle periodically

**UWB session disconnects frequently:**
Keep both phones within ~10m with clear line of sight. Metal shelving and walls attenuate the UWB signal. The session auto-restarts on reconnect.

**"Invalid peerIDs" error in logs:**
This was a race condition between the MC connection callback and `connectedPeers` being populated. Fixed by capturing the `MCPeerID` directly from the delegate parameter.
