//
//  ViewerSessionManager.swift
//  ViewerApp
//
//  Created by Terrence Lei on 4/19/26.
//

import Foundation
import Combine
import NearbyInteraction
import MultipeerConnectivity
import simd

/// Position data from a UWB reading.
struct TagReading {
    var distance: Float     // metres
    var direction: simd_float3? // raw direction vector from NI
    var x: Float            // derived: metres right (+) or left (-)
    var y: Float            // derived: metres forward (+)
}

/// Manages the UWB + MultipeerConnectivity session for the Viewer (cart) phone.
/// This device browses for the Tag phone and displays its position on a radar.
class ViewerSessionManager: NSObject, ObservableObject {

    // MARK: - Published State

    @Published var reading: TagReading?
    @Published var status = "Searching for Shopper..."
    @Published var isConnected = false
    @Published var isRanging = false
    @Published var convergenceHint: String? = nil
    @Published private(set) var calibrationOffset: Float
    @Published var isCalibrating = false
    @Published var calibrationProgress: Double = 0

    // MARK: - NearbyInteraction

    private var niSession: NISession?
    private var peerDiscoveryToken: NIDiscoveryToken?

    // MARK: - MultipeerConnectivity

    private var mcSession: MCSession!
    private var browser: MCNearbyServiceBrowser!
    private let myPeerID = MCPeerID(displayName: "CartView-\(UIDevice.current.name)")
    private let serviceType = "uwb-cart"

    // MARK: - Init

    override init() {
        calibrationOffset = UserDefaults.standard.float(forKey: "uwb_calibration_offset")
        super.init()

        mcSession = MCSession(peer: myPeerID, securityIdentity: nil, encryptionPreference: .required)
        mcSession.delegate = self

        browser = MCNearbyServiceBrowser(peer: myPeerID, serviceType: serviceType)
        browser.delegate = self
        browser.startBrowsingForPeers()
        print("[Viewer] Started browsing for service: \(serviceType)")
    }

    deinit {
        browser.stopBrowsingForPeers()
        niSession?.invalidate()
    }

    // MARK: - NI Session Setup

    private func startNISession() {
        guard NISession.deviceCapabilities.supportsPreciseDistanceMeasurement else {
            status = "Connected (UWB not available on this device)"
            return
        }

        niSession?.invalidate()
        niSession = NISession()
        niSession?.delegate = self

        // Share our discovery token with the tag phone
        guard let myToken = niSession?.discoveryToken else {
            status = "Error: could not get discovery token"
            return
        }

        sendDiscoveryToken(myToken)
        status = "Sent token — waiting for Shopper's token..."
    }

    private func sendDiscoveryToken(_ token: NIDiscoveryToken) {
        let data: Data
        do {
            data = try NSKeyedArchiver.archivedData(withRootObject: token, requiringSecureCoding: true)
        } catch {
            print("[Viewer] ERROR: Failed to archive discovery token: \(error.localizedDescription)")
            DispatchQueue.main.async {
                self.status = "Token error — restart app"
            }
            return
        }

        do {
            try mcSession.send(data, toPeers: mcSession.connectedPeers, with: .reliable)
            print("[Viewer] Sent discovery token to \(mcSession.connectedPeers.count) peer(s)")
        } catch {
            print("[Viewer] ERROR: Failed to send discovery token: \(error.localizedDescription)")
            DispatchQueue.main.async {
                self.status = "Send failed — retrying..."
            }
        }
    }

    // MARK: - Distance Calibration

    private var calibrationSamples: [Float] = []
    private let calibrationSampleCount = 20

    /// Hold phones together then call this. Collects samples over ~2 s and averages them.
    func beginCalibration() {
        guard isRanging, !isCalibrating else { return }
        calibrationSamples = []
        calibrationProgress = 0
        isCalibrating = true
    }

    func resetCalibration() {
        isCalibrating = false
        calibrationSamples = []
        calibrationProgress = 0
        calibrationOffset = 0
        UserDefaults.standard.removeObject(forKey: "uwb_calibration_offset")
    }

    private func configureAndRun(with peerToken: NIDiscoveryToken) {
        guard let niSession else { return }
        let config = NINearbyPeerConfiguration(peerToken: peerToken)
        if #available(iOS 16.0, *) {
            let supported = NISession.deviceCapabilities.supportsCameraAssistance
            print("[Viewer] supportsCameraAssistance: \(supported)")
            if supported {
                config.isCameraAssistanceEnabled = true
                print("[Viewer] Camera assistance enabled on session config")
            } else {
                print("[Viewer] Camera assistance not supported — direction may be unavailable")
            }
        }
        niSession.run(config)
        DispatchQueue.main.async {
            self.isRanging = true
            self.status = "UWB ranging active"
        }
    }
}

// MARK: - NISessionDelegate

extension ViewerSessionManager: NISessionDelegate {

    func session(_ session: NISession, didUpdate nearbyObjects: [NINearbyObject]) {
        guard let obj = nearbyObjects.first else { return }

        // Skip update when distance is nil (signal too weak).
        // The previous valid reading remains on screen.
        guard let rawDist = obj.distance else { return }

        // Apply calibration offset and clamp so display never goes negative.
        let dist = max(0, rawDist - calibrationOffset)

        // Derive 2D position from the direction vector
        // direction.x: positive = right, direction.z: negative = forward (away from screen)
        var screenX: Float = 0
        var screenY: Float = dist  // default: straight ahead at measured distance

        if let dir = obj.direction {
            screenX = dist * dir.x
            screenY = dist * (-dir.z)  // flip z so forward is positive
        }

        DispatchQueue.main.async {
            // Accumulate raw samples during calibration
            if self.isCalibrating {
                self.calibrationSamples.append(rawDist)
                self.calibrationProgress = Double(self.calibrationSamples.count) / Double(self.calibrationSampleCount)
                if self.calibrationSamples.count >= self.calibrationSampleCount {
                    let avg = self.calibrationSamples.reduce(0, +) / Float(self.calibrationSamples.count)
                    self.calibrationOffset = avg
                    UserDefaults.standard.set(avg, forKey: "uwb_calibration_offset")
                    self.isCalibrating = false
                    self.calibrationSamples = []
                    self.calibrationProgress = 0
                    self.status = "Distance zeroed"
                }
                return
            }

            self.reading = TagReading(
                distance: dist,
                direction: obj.direction,
                x: screenX,
                y: screenY
            )
            if let dir = obj.direction {
                self.convergenceHint = nil
                let angleDeg = atan2(dir.x, -dir.z) * 180 / .pi
                self.status = String(format: "%.2fm  %+.0f°", dist, angleDeg)
            } else {
                self.status = String(format: "%.2fm", dist)
            }
        }
    }

    @available(iOS 16.0, *)
    func session(_ session: NISession, didUpdateAlgorithmConvergence convergence: NIAlgorithmConvergence, for object: NINearbyObject?) {
        guard object != nil, reading?.direction == nil else { return }
        if case .converged = convergence.status {
            convergenceHint = nil
            status = "Direction locked"
        } else {
            convergenceHint = "Slowly sweep phone left/right to calibrate direction"
        }
    }

    func session(_ session: NISession, didRemove nearbyObjects: [NINearbyObject], reason: NINearbyObject.RemovalReason) {
        DispatchQueue.main.async {
            self.isRanging = false
            self.reading = nil
            if reason == .timeout {
                self.status = "Peer out of range — waiting..."
                if !self.mcSession.connectedPeers.isEmpty {
                    self.startNISession()
                }
            }
        }
    }

    func session(_ session: NISession, didInvalidateWith error: Error) {
        DispatchQueue.main.async {
            self.isRanging = false
            self.reading = nil
            self.status = "Session error — reconnecting..."
            if !self.mcSession.connectedPeers.isEmpty {
                self.startNISession()
            }
        }
    }

    func sessionWasSuspended(_ session: NISession) {
        DispatchQueue.main.async {
            self.status = "Suspended — bring app to foreground"
        }
    }

    func sessionSuspensionEnded(_ session: NISession) {
        DispatchQueue.main.async {
            self.status = "Resumed — restarting..."
            self.startNISession()
        }
    }
}

// MARK: - MCNearbyServiceBrowserDelegate

extension ViewerSessionManager: MCNearbyServiceBrowserDelegate {

    func browser(_ browser: MCNearbyServiceBrowser, foundPeer peerID: MCPeerID, withDiscoveryInfo info: [String: String]?) {
        print("[Viewer] Found peer: \(peerID.displayName)")
        browser.invitePeer(peerID, to: mcSession, withContext: nil, timeout: 10)
        DispatchQueue.main.async {
            self.status = "Found \(peerID.displayName), connecting..."
        }
    }

    func browser(_ browser: MCNearbyServiceBrowser, lostPeer peerID: MCPeerID) {
        print("[Viewer] Lost peer: \(peerID.displayName)")
    }

    func browser(_ browser: MCNearbyServiceBrowser, didNotStartBrowsingForPeers error: Error) {
        print("[Viewer] ERROR: Failed to start browsing: \(error.localizedDescription)")
        DispatchQueue.main.async {
            self.status = "Browsing failed: \(error.localizedDescription)"
        }
    }
}

// MARK: - MCSessionDelegate

extension ViewerSessionManager: MCSessionDelegate {

    func session(_ session: MCSession, peer peerID: MCPeerID, didChange state: MCSessionState) {
        print("[Viewer] Peer \(peerID.displayName) state: \(state.rawValue) (0=notConnected, 1=connecting, 2=connected)")
        DispatchQueue.main.async {
            switch state {
            case .connected:
                self.isConnected = true
                self.status = "Connected — starting UWB"
                self.startNISession()
            case .notConnected:
                self.isConnected = false
                self.isRanging = false
                self.reading = nil
                self.convergenceHint = nil
                self.niSession?.invalidate()
                self.niSession = nil
                self.status = "Disconnected — searching..."
            case .connecting:
                self.status = "Connecting..."
            @unknown default:
                break
            }
        }
    }

    func session(_ session: MCSession, didReceive data: Data, fromPeer peerID: MCPeerID) {
        // Received tag phone's discovery token
        guard let token = try? NSKeyedUnarchiver.unarchivedObject(ofClass: NIDiscoveryToken.self, from: data) else {
            print("[Viewer] ERROR: Failed to unarchive discovery token from \(peerID.displayName)")
            return
        }
        print("[Viewer] Received discovery token from \(peerID.displayName)")

        DispatchQueue.main.async {
            self.peerDiscoveryToken = token
            self.configureAndRun(with: token)
        }
    }

    func session(_ session: MCSession, didReceive stream: InputStream, withName streamName: String, fromPeer peerID: MCPeerID) {}
    func session(_ session: MCSession, didStartReceivingResourceWithName resourceName: String, fromPeer peerID: MCPeerID, with progress: Progress) {}
    func session(_ session: MCSession, didFinishReceivingResourceWithName resourceName: String, fromPeer peerID: MCPeerID, at localURL: URL?, withError error: Error?) {}
}
