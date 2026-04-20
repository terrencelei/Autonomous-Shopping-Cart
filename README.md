# UWBCart

A two-device iOS system that tracks a shopper's position relative to a shopping cart in real time using Apple's Ultra-Wideband (UWB) technology.

## How It Works

UWBCart uses two iPhones — one carried by the shopper (**Tag**) and one mounted on the cart (**Viewer**). The devices discover each other over MultipeerConnectivity, then use Nearby Interaction to measure precise distance and direction via the U1/U2 UWB chip.

The Viewer app displays the shopper's position on a top-down radar in real time.

## Apps

### UWBCart (Tag App)
Runs on the **shopper's iPhone**. Advertises itself over the local network and acts as a UWB beacon. The shopper just needs to keep the app open — no interaction required.

### ViewerApp
Runs on the **cart's iPhone**. Discovers the Tag, connects, and displays a live radar showing:
- The shopper's relative position (orange dot)
- Distance in metres
- Angle in degrees
- Range rings from 1m to 4m

## Connection Flow

1. Tag app starts advertising via MultipeerConnectivity (`_uwb-cart._tcp`)
2. Viewer app browses the local network and finds the Tag
3. Encrypted MC session is established
4. Both devices exchange NearbyInteraction discovery tokens over MC
5. UWB ranging begins — Viewer receives distance and direction updates
6. Sessions automatically restart if the peer goes out of range or the app is suspended

## Requirements

- Two iPhones with a U1 or U2 chip (iPhone 11 or later)
- iOS 16.0+
- Both devices on the same local network (for initial MultipeerConnectivity discovery)

## Project Structure

```
UWBCart/
├── UWBCart/                  # Tag app (shopper's phone)
│   ├── UWBCartApp.swift      # App entry point
│   ├── ContentView.swift     # Tag UI with pulse animation
│   ├── TagSessionManager.swift # MC advertiser + NI session
│   └── Info.plist            # Network & NI permissions
├── ViewerApp/                # Viewer app (cart phone)
│   ├── ViewerAppApp.swift    # App entry point
│   ├── ContentView.swift     # Radar UI with range rings
│   ├── ViewerSessionManager.swift # MC browser + NI session + position tracking
│   └── Info.plist            # Network & NI permissions
├── UWBCartTests/
├── UWBCartUITests/
├── ViewerAppTests/
└── ViewerAppUITests/
```

## Permissions

Both apps request the following permissions (declared in their Info.plist files):

| Key | Purpose |
|-----|---------|
| `NSLocalNetworkUsageDescription` | MultipeerConnectivity device discovery |
| `NSBonjourServices` | Bonjour service registration (`_uwb-cart._tcp`) |
| `NSNearbyInteractionUsageDescription` | UWB distance and direction measurement |
