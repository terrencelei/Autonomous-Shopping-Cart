//
//  TagView.swift
//  UWBCart
//
//  Created by Terrence Lei on 4/19/26.
//

import SwiftUI

/// The main UI for the Tag (shopper's) phone.
/// Shows connection status and a pulsing indicator when ranging is active.
struct TagView: View {
    @StateObject private var manager = TagSessionManager()

    var body: some View {
        VStack(spacing: 32) {
            Spacer()

            ZStack {
                // Pulse rings when ranging
                if manager.isRanging {
                    ForEach(0..<3, id: \.self) { i in
                        PulseRing(delay: Double(i) * 0.6)
                    }
                }

                Image(systemName: manager.isRanging ? "wave.3.forward.circle.fill" : "antenna.radiowaves.left.and.right")
                    .font(.system(size: 80))
                    .foregroundStyle(manager.isRanging ? .green : .orange)
                    .symbolEffect(.pulse, isActive: manager.isRanging)
            }
            .frame(width: 160, height: 160)

            Text("Shopper")
                .font(.largeTitle.bold())

            Text(manager.status)
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)

            if let peer = manager.connectedPeer {
                Label(peer, systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                    .font(.headline)
            }

            Spacer()

            Text("Keep this app open on the shopper's iPhone")
                .font(.caption)
                .foregroundStyle(.tertiary)
                .padding(.bottom, 24)
        }
        .padding()
    }
}

// MARK: - Pulse Ring Animation

struct PulseRing: View {
    let delay: Double
    @State private var isAnimating = false

    var body: some View {
        Circle()
            .stroke(Color.green.opacity(isAnimating ? 0 : 0.4), lineWidth: 2)
            .frame(width: isAnimating ? 160 : 80, height: isAnimating ? 160 : 80)
            .onAppear {
                withAnimation(
                    .easeOut(duration: 1.8)
                    .repeatForever(autoreverses: false)
                    .delay(delay)
                ) {
                    isAnimating = true
                }
            }
    }
}

#Preview {
    TagView()
}
