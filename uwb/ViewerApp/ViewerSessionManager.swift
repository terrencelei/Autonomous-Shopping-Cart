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
import AVFoundation
import simd

/// Position data from a UWB reading.
struct TagReading {
    var distance: Float          // metres (smoothed)
    var direction: simd_float3?  // full 3D direction vector
    var horizontalAngle: Float?  // azimuthal angle in radians (smoothed)
    var verticalEstimate: Int    // NINearbyObject.VerticalDirectionEstimate rawValue
    var x: Float                 // derived: metres right (+) or left (-)
    var y: Float                 // derived: metres forward (+)

    var hasAngle: Bool { direction != nil || horizontalAngle != nil }

    var angleDegrees: Float? {
        if let dir = direction { return atan2(dir.x, -dir.z) * 180 / .pi }
        if let h = horizontalAngle { return h * 180 / .pi }
        return nil
    }

    var verticalLabel: String {
        switch verticalEstimate {
        case 1: return "same"
        case 2: return "above"
        case 3: return "below"
        case 4: return "above/below"
        default: return "unknown"
        }
    }
}

/// Manages the UWB + MultipeerConnectivity session for the Viewer (cart) phone.
class ViewerSessionManager: NSObject, ObservableObject {

    // MARK: - Published State

    @Published var reading: TagReading?
    @Published var status = "Searching for Shopper..."
    @Published var isConnected = false
    @Published var isRanging = false
    @Published private(set) var calibrationOffset: Float
    @Published var isCalibrating = false
    @Published var calibrationProgress: Double = 0
    @Published private(set) var angleOffset: Float

    // MARK: - NearbyInteraction

    private var niSession: NISession?
    private var peerDiscoveryToken: NIDiscoveryToken?
    private var connectedPeerID: MCPeerID?

    // MARK: - MultipeerConnectivity

    private var mcSession: MCSession!
    private var browser: MCNearbyServiceBrowser!
    private let myPeerID = MCPeerID(displayName: "CartView-\(UIDevice.current.name)")
    private let serviceType = "uwb-cart"

    // MARK: - Init

    override init() {
        calibrationOffset = UserDefaults.standard.float(forKey: "uwb_calibration_offset")
        angleOffset = UserDefaults.standard.float(forKey: "uwb_angle_offset")
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
        let caps = NISession.deviceCapabilities
        let camAuth = AVCaptureDevice.authorizationStatus(for: .video)
        print("[Viewer] Capabilities — preciseDist=\(caps.supportsPreciseDistanceMeasurement)  direction=\(caps.supportsDirectionMeasurement)  cameraAssistance=\(caps.supportsCameraAssistance)  extendedDist=\(caps.supportsExtendedDistanceMeasurement)")
        print("[Viewer] Camera auth status: \(camAuth.rawValue) (0=notDetermined 1=restricted 2=denied 3=authorized)")

        guard caps.supportsPreciseDistanceMeasurement else {
            status = "Connected (UWB not available on this device)"
            return
        }

        niSession?.invalidate()
        niSession = NISession()
        niSession?.delegate = self

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
            print("[Viewer] ERROR: Failed to archive token: \(error.localizedDescription)")
            DispatchQueue.main.async { self.status = "Token error — restart app" }
            return
        }
        guard let peer = connectedPeerID else {
            print("[Viewer] ERROR: No connected peer to send token to")
            return
        }
        do {
            try mcSession.send(data, toPeers: [peer], with: .reliable)
            print("[Viewer] Sent discovery token to \(peer.displayName)")
        } catch {
            print("[Viewer] ERROR: Failed to send token: \(error.localizedDescription)")
            DispatchQueue.main.async { self.status = "Send failed — retrying..." }
        }
    }

    // MARK: - Smoothing (exponential moving average)
    // alpha: 0=no smoothing/instant, 1=heavily smoothed/slow. 0.2 keeps noise low with minimal lag.

    private let emaAlpha: Float = 0.2
    private var emaDist: Float? = nil
    private var emaAngle: Float? = nil

    private func smoothedDistance(_ raw: Float) -> Float {
        let out = emaDist.map { emaAlpha * raw + (1 - emaAlpha) * $0 } ?? raw
        emaDist = out
        return out
    }

    private func smoothedHAngle(_ raw: Float?) -> Float? {
        guard let raw else { return emaAngle }   // hold last known on dropout
        let out = emaAngle.map { emaAlpha * raw + (1 - emaAlpha) * $0 } ?? raw
        emaAngle = out
        return out
    }

    private func clearSmoothingBuffers() {
        emaDist = nil
        emaAngle = nil
    }

    // MARK: - Distance Calibration

    private var calibrationSamples: [Float] = []
    private let calibrationSampleCount = 20

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

    // MARK: - Angle Calibration

    func zeroAngle() {
        guard let current = emaAngle else { return }
        angleOffset = current
        UserDefaults.standard.set(current, forKey: "uwb_angle_offset")
    }

    func resetAngleCalibration() {
        angleOffset = 0
        UserDefaults.standard.removeObject(forKey: "uwb_angle_offset")
    }

    private func configureAndRun(with peerToken: NIDiscoveryToken) {
        guard let niSession else {
            print("[Viewer] configureAndRun: niSession is nil")
            return
        }
        let config = NINearbyPeerConfiguration(peerToken: peerToken)
        // Let NI manage its own ARSession internally — only set flag, don't call setARSession.
        // setARSession is for apps that already have ARKit running for other purposes.
        if NISession.deviceCapabilities.supportsCameraAssistance {
            config.isCameraAssistanceEnabled = true
        }
        print("[Viewer] Running NISession — cameraAssistance=\(config.isCameraAssistanceEnabled)")
        niSession.run(config)
        isRanging = true
        status = "UWB ranging active"
    }
}

// MARK: - NISessionDelegate

extension ViewerSessionManager: NISessionDelegate {

    // Called only when isCameraAssistanceEnabled = true.
    // If this never prints, camera assistance is not activating.
    func session(_ session: NISession, didUpdateAlgorithmConvergence convergence: NIAlgorithmConvergence, for peer: NINearbyObject?) {
        if case .converged = convergence.status {
            print("[Viewer] Convergence: CONVERGED ✓")
            if isRanging { status = "Angle locked" }
        } else {
            print("[Viewer] Convergence: not yet — \(convergence.status)")
            if isRanging { status = "Move phone to establish angle…" }
        }
    }

    func session(_ session: NISession, didUpdate nearbyObjects: [NINearbyObject]) {
        guard let obj = nearbyObjects.first else {
            print("[Viewer] didUpdate: empty")
            return
        }
        guard let rawDist = obj.distance else {
            print("[Viewer] didUpdate: distance=nil")
            return
        }

        let dirStr    = obj.direction.map { "(\(String(format: "%.2f", $0.x)), \(String(format: "%.2f", $0.y)), \(String(format: "%.2f", $0.z)))" } ?? "nil"
        let hAngleStr = obj.horizontalAngle.map { String(format: "%.3f rad (%.1f°)", $0, $0 * 180 / .pi) } ?? "nil"
        print("[Viewer] dist=\(String(format: "%.2f", rawDist))m  dir=\(dirStr)  hAngle=\(hAngleStr)  vert=\(obj.verticalDirectionEstimate.rawValue)")

        let smoothDist  = max(0, smoothedDistance(rawDist) - calibrationOffset)
        let rawAngle    = smoothedHAngle(obj.horizontalAngle)
        // Apply angle offset and normalise to [-π, π]
        let smoothAngle: Float? = rawAngle.map {
            var a = $0 - angleOffset
            while a >  .pi { a -= 2 * .pi }
            while a < -.pi { a += 2 * .pi }
            return a
        }

        // NISession delegates are called on the main queue — update reading synchronously.
        let direction = obj.direction
        let vertEst   = obj.verticalDirectionEstimate.rawValue

        var screenX: Float = 0
        var screenY: Float = smoothDist

        if let dir = direction {
            screenX = smoothDist * dir.x
            screenY = smoothDist * (-dir.z)
        } else if let h = smoothAngle {
            screenX = smoothDist * sin(h)
            screenY = smoothDist * cos(h)
        }

        if isCalibrating {
            calibrationSamples.append(rawDist)
            calibrationProgress = Double(calibrationSamples.count) / Double(calibrationSampleCount)
            if calibrationSamples.count >= calibrationSampleCount {
                let avg = calibrationSamples.reduce(0, +) / Float(calibrationSamples.count)
                calibrationOffset = avg
                UserDefaults.standard.set(avg, forKey: "uwb_calibration_offset")
                isCalibrating = false
                calibrationSamples = []
                calibrationProgress = 0
                status = "Distance zeroed"
            }
            return
        }

        reading = TagReading(
            distance: smoothDist,
            direction: direction,
            horizontalAngle: smoothAngle,
            verticalEstimate: vertEst,
            x: screenX,
            y: screenY
        )

        if let deg = reading?.angleDegrees {
            status = String(format: "%.2fm  %+.0f°", smoothDist, deg)
        } else {
            status = String(format: "%.2fm  (no angle)", smoothDist)
        }
    }

    func session(_ session: NISession, didRemove nearbyObjects: [NINearbyObject], reason: NINearbyObject.RemovalReason) {
        print("[Viewer] didRemove: reason=\(reason.rawValue)")
        isRanging = false
        reading = nil
        clearSmoothingBuffers()
        if reason == .timeout {
            status = "Peer out of range — waiting..."
            if connectedPeerID != nil { startNISession() }
        }
    }

    func session(_ session: NISession, didInvalidateWith error: Error) {
        print("[Viewer] didInvalidateWith: \(error.localizedDescription)")
        isRanging = false
        reading = nil
        clearSmoothingBuffers()
        status = "Session error: \(error.localizedDescription)"
        if connectedPeerID != nil { startNISession() }
    }

    func sessionWasSuspended(_ session: NISession) {
        status = "Suspended — bring app to foreground"
    }

    func sessionSuspensionEnded(_ session: NISession) {
        status = "Resumed — restarting..."
        startNISession()
    }
}

// MARK: - MCNearbyServiceBrowserDelegate

extension ViewerSessionManager: MCNearbyServiceBrowserDelegate {

    func browser(_ browser: MCNearbyServiceBrowser, foundPeer peerID: MCPeerID, withDiscoveryInfo info: [String: String]?) {
        print("[Viewer] Found peer: \(peerID.displayName)")
        browser.invitePeer(peerID, to: mcSession, withContext: nil, timeout: 10)
        DispatchQueue.main.async { self.status = "Found \(peerID.displayName), connecting..." }
    }

    func browser(_ browser: MCNearbyServiceBrowser, lostPeer peerID: MCPeerID) {
        print("[Viewer] Lost peer: \(peerID.displayName)")
    }

    func browser(_ browser: MCNearbyServiceBrowser, didNotStartBrowsingForPeers error: Error) {
        print("[Viewer] ERROR: browsing failed: \(error.localizedDescription)")
        DispatchQueue.main.async { self.status = "Browsing failed: \(error.localizedDescription)" }
    }
}

// MARK: - MCSessionDelegate

extension ViewerSessionManager: MCSessionDelegate {

    func session(_ session: MCSession, peer peerID: MCPeerID, didChange state: MCSessionState) {
        print("[Viewer] Peer \(peerID.displayName) state: \(state.rawValue)")
        DispatchQueue.main.async {
            switch state {
            case .connected:
                self.connectedPeerID = peerID
                self.isConnected = true
                self.status = "Connected — starting UWB"
                self.startNISession()
            case .notConnected:
                self.connectedPeerID = nil
                self.isConnected = false
                self.isRanging = false
                self.reading = nil
                self.clearSmoothingBuffers()
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
        guard let token = try? NSKeyedUnarchiver.unarchivedObject(ofClass: NIDiscoveryToken.self, from: data) else {
            print("[Viewer] ERROR: Failed to unarchive token from \(peerID.displayName)")
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
