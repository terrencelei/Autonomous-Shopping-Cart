//
//  TagSessionManager.swift
//  UWBCart
//
//  Created by Terrence Lei on 4/19/26.
//

import Foundation
import Combine
import NearbyInteraction
import MultipeerConnectivity

/// Manages the UWB + MultipeerConnectivity session for the Tag (shopper's) phone.
/// This device advertises itself and waits for the Viewer (cart) phone to connect.
class TagSessionManager: NSObject, ObservableObject {

    // MARK: - Published State

    @Published var status = "Starting..."
    @Published var connectedPeer: String?
    @Published var isRanging = false

    // MARK: - NearbyInteraction

    private var niSession: NISession?
    private var peerDiscoveryToken: NIDiscoveryToken?

    // MARK: - MultipeerConnectivity

    private var mcSession: MCSession!
    private var advertiser: MCNearbyServiceAdvertiser!
    private let myPeerID = MCPeerID(displayName: "Shopper-\(UIDevice.current.name)")
    private let serviceType = "uwb-cart"

    // MARK: - Init

    override init() {
        super.init()

        mcSession = MCSession(peer: myPeerID, securityIdentity: nil, encryptionPreference: .required)
        mcSession.delegate = self

        advertiser = MCNearbyServiceAdvertiser(peer: myPeerID, discoveryInfo: nil, serviceType: serviceType)
        advertiser.delegate = self
        advertiser.startAdvertisingPeer()
        print("[Tag] Started advertising as \(myPeerID.displayName) on service: \(serviceType)")

        status = "Advertising — open CartView app on cart iPhone"
    }

    deinit {
        advertiser.stopAdvertisingPeer()
        niSession?.invalidate()
    }

    // MARK: - NI Session Setup

    private func startNISession() {
        let caps = NISession.deviceCapabilities
        print("[Tag] Device capabilities — preciseDist=\(caps.supportsPreciseDistanceMeasurement)  cameraAssistance=\(caps.supportsCameraAssistance)")

        guard caps.supportsPreciseDistanceMeasurement else {
            status = "Connected (UWB not available on this device)"
            return
        }

        niSession?.invalidate()
        niSession = NISession()
        niSession?.delegate = self

        // Share our discovery token with the viewer
        guard let myToken = niSession?.discoveryToken else {
            status = "Error: could not get discovery token"
            return
        }

        sendDiscoveryToken(myToken)
        status = "Sent token — waiting for CartView's token..."
    }

    private func sendDiscoveryToken(_ token: NIDiscoveryToken) {
        let data: Data
        do {
            data = try NSKeyedArchiver.archivedData(withRootObject: token, requiringSecureCoding: true)
        } catch {
            print("[Tag] ERROR: Failed to archive discovery token: \(error.localizedDescription)")
            DispatchQueue.main.async {
                self.status = "Token error — restart app"
            }
            return
        }

        do {
            try mcSession.send(data, toPeers: mcSession.connectedPeers, with: .reliable)
            print("[Tag] Sent discovery token to \(mcSession.connectedPeers.count) peer(s)")
        } catch {
            print("[Tag] ERROR: Failed to send discovery token: \(error.localizedDescription)")
            DispatchQueue.main.async {
                self.status = "Send failed — retrying..."
            }
        }
    }

    private func configureAndRun(with peerToken: NIDiscoveryToken) {
        guard let niSession else {
            print("[Tag] configureAndRun: niSession is nil — cannot run")
            return
        }
        let config = NINearbyPeerConfiguration(peerToken: peerToken)
        // Pure UWB — no camera assistance
        print("[Tag] Running NISession — pure UWB, isCameraAssistanceEnabled=\(config.isCameraAssistanceEnabled)")
        niSession.run(config)
        DispatchQueue.main.async {
            self.isRanging = true
            self.status = "UWB ranging active"
        }
    }
}

// MARK: - NISessionDelegate

extension TagSessionManager: NISessionDelegate {

    func session(_ session: NISession, didUpdate nearbyObjects: [NINearbyObject]) {
        // Tag app doesn't need to process position updates — just keep ranging
        if let obj = nearbyObjects.first {
            print("[Tag] didUpdate: distance=\(obj.distance.map { "\($0)m" } ?? "nil")  direction=\(obj.direction != nil ? "available" : "nil")")
        } else {
            print("[Tag] didUpdate: nearbyObjects is empty")
        }
    }

    func session(_ session: NISession, didRemove nearbyObjects: [NINearbyObject], reason: NINearbyObject.RemovalReason) {
        DispatchQueue.main.async {
            self.isRanging = false
            if reason == .timeout {
                self.status = "Peer out of range — waiting..."
                // Restart the NI session to resume ranging when peer comes back
                if !self.mcSession.connectedPeers.isEmpty {
                    self.startNISession()
                }
            }
        }
    }

    func session(_ session: NISession, didInvalidateWith error: Error) {
        print("[Tag] didInvalidateWith: \(error.localizedDescription)")
        DispatchQueue.main.async {
            self.isRanging = false
            self.status = "Session error: \(error.localizedDescription)"
            // Restart if still connected via MC
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

// MARK: - MCNearbyServiceAdvertiserDelegate

extension TagSessionManager: MCNearbyServiceAdvertiserDelegate {

    func advertiser(_ advertiser: MCNearbyServiceAdvertiser,
                    didReceiveInvitationFromPeer peer: MCPeerID,
                    withContext context: Data?,
                    invitationHandler: @escaping (Bool, MCSession?) -> Void) {
        print("[Tag] Received invitation from \(peer.displayName)")
        invitationHandler(true, mcSession)
    }

    func advertiser(_ advertiser: MCNearbyServiceAdvertiser, didNotStartAdvertisingPeer error: Error) {
        print("[Tag] ERROR: Failed to start advertising: \(error.localizedDescription)")
        DispatchQueue.main.async {
            self.status = "Advertising failed: \(error.localizedDescription)"
        }
    }
}

// MARK: - MCSessionDelegate

extension TagSessionManager: MCSessionDelegate {

    func session(_ session: MCSession, peer peerID: MCPeerID, didChange state: MCSessionState) {
        print("[Tag] Peer \(peerID.displayName) state: \(state.rawValue) (0=notConnected, 1=connecting, 2=connected)")
        DispatchQueue.main.async {
            switch state {
            case .connected:
                self.connectedPeer = peerID.displayName
                self.status = "Connected to \(peerID.displayName)"
                self.startNISession()
            case .notConnected:
                self.connectedPeer = nil
                self.isRanging = false
                self.niSession?.invalidate()
                self.niSession = nil
                self.status = "Disconnected — advertising..."
            case .connecting:
                self.status = "Connecting to \(peerID.displayName)..."
            @unknown default:
                break
            }
        }
    }

    func session(_ session: MCSession, didReceive data: Data, fromPeer peerID: MCPeerID) {
        // Received viewer's discovery token
        guard let token = try? NSKeyedUnarchiver.unarchivedObject(ofClass: NIDiscoveryToken.self, from: data) else {
            print("[Tag] ERROR: Failed to unarchive discovery token from \(peerID.displayName)")
            return
        }
        print("[Tag] Received discovery token from \(peerID.displayName)")

        DispatchQueue.main.async {
            self.peerDiscoveryToken = token
            self.configureAndRun(with: token)
        }
    }

    func session(_ session: MCSession, didReceive stream: InputStream, withName streamName: String, fromPeer peerID: MCPeerID) {}
    func session(_ session: MCSession, didStartReceivingResourceWithName resourceName: String, fromPeer peerID: MCPeerID, with progress: Progress) {}
    func session(_ session: MCSession, didFinishReceivingResourceWithName resourceName: String, fromPeer peerID: MCPeerID, at localURL: URL?, withError error: Error?) {}
}
