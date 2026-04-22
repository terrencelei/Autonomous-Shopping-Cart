# UWBCart

A two-device iOS system that tracks a shopper's position relative to a shopping cart in real time using Apple's Ultra-Wideband (UWB) technology.

## How It Works

UWBCart uses two iPhones — one carried by the shopper (**Shopper**) and one mounted on the cart (**CartView**). The devices discover each other over MultipeerConnectivity, exchange NearbyInteraction discovery tokens, then use the U1/U2 UWB chip to measure precise distance and direction.

The CartView app displays the shopper's position on a top-down radar in real time, with live distance and angle readouts.

## Apps

### Shopper (Tag App)
Runs on the **shopper's iPhone**. Advertises itself over the local network and acts as a UWB beacon. No interaction required — the shopper just keeps the app open.

### CartView (Viewer App)
Runs on the **cart's iPhone**. Discovers the Shopper, connects, and displays a live radar showing:
- Shopper's relative position (orange dot on top-down radar)
- Distance in metres (calibrated)
- Angle in degrees (when direction is available from UWB AoA)
- Auto-scaling range rings

## Connection Flow

1. Shopper app starts advertising via MultipeerConnectivity (`_uwb-cart._tcp`)
2. CartView browses the local network and invites the Shopper
3. Encrypted MC session is established
4. Both devices exchange NearbyInteraction discovery tokens over MC (reliable delivery)
5. Both sides call `NISession.run()` with the peer's token — UWB ranging begins
6. CartView receives distance and direction updates via `NISessionDelegate`
7. Sessions automatically restart if the peer goes out of range or the app is suspended

## Direction

Direction (`NINearbyObject.direction`) is optionally populated by the U1/U2 chip's angle-of-arrival phased antenna array — no camera assistance required. When the peer is within the chip's field of view, the direction vector is provided as a `simd_float3`. When unavailable (peer outside AoA coverage or signal geometry insufficient), the angle readout shows `—` and the radar shows the shopper directly ahead at the measured distance.

## Distance Calibration

UWB time-of-flight measurements include a fixed hardware offset from antenna positions and firmware delays, meaning adjacent phones never read zero. CartView provides a persistent calibration offset:

1. Hold both phones together at your chosen zero reference point
2. Tap **Set Zero** in the CartView UI
3. CartView samples 20 readings over ~2 seconds and averages them as the offset
4. All subsequent distance readings subtract the offset (clamped to 0)
5. Tap **Reset** to clear the calibration

The offset is stored in UserDefaults and persists across app launches.

## Requirements

- Two iPhones with a U1 or U2 chip (iPhone 11 or later)
- iOS 16.0+
- Both devices on the same local network (for MultipeerConnectivity discovery)

## Permissions

Both apps declare the following in their Info.plist:

| Key | Purpose |
|-----|---------|
| `NSLocalNetworkUsageDescription` | MultipeerConnectivity device discovery |
| `NSBonjourServices` | Bonjour service registration (`_uwb-cart._tcp`) |
| `NSNearbyInteractionUsageDescription` | UWB distance and direction measurement |

## Project Structure

```
UWBCart/
├── UWBCart/                      # Shopper app (shopper's phone)
│   ├── UWBCartApp.swift          # App entry point
│   ├── ContentView.swift         # Pulse animation UI
│   ├── TagSessionManager.swift   # MC advertiser + NI session
│   └── Info.plist
├── ViewerApp/                    # CartView app (cart phone)
│   ├── ViewerAppApp.swift        # App entry point
│   ├── ContentView.swift         # Radar UI, distance/angle readouts, calibration buttons
│   ├── ViewerSessionManager.swift # MC browser + NI session + calibration
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

NearbyInteraction does not require any special entitlement — Xcode's automatic signing handles it.

**Direction never appears:**
Direction requires the peer to be within the UWB chip's AoA field of view (roughly in front of the device). Hold the cart phone facing the shopper with a clear line of sight.

**UWB session disconnects frequently:**
Keep both phones within ~10m. Walls and metal carts can attenuate the UWB signal. The session auto-restarts on reconnect.
