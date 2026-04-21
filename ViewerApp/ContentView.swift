//
//  ViewerRootView.swift
//  ViewerApp
//
//  Created by Terrence Lei on 4/19/26.
//

import SwiftUI

/// The main UI for the Viewer (cart) phone.
/// Displays a top-down radar showing the shopper's relative position.
struct ViewerRootView: View {
    @StateObject private var manager = ViewerSessionManager()

    var body: some View {
        VStack(spacing: 0) {
            // Status bar
            HStack {
                Circle()
                    .fill(statusColor)
                    .frame(width: 10, height: 10)
                Text(manager.status)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.vertical, 8)

            // Radar view
            RadarView(reading: manager.reading)
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            // Numeric readout
            if let r = manager.reading {
                HStack(spacing: 32) {
                    VStack {
                        Text(String(format: "%.2f m", r.distance))
                            .font(.system(.title, design: .monospaced).bold())
                        Text("distance")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    if r.direction != nil {
                        VStack {
                            let angleDeg = atan2(r.x, r.y) * 180 / .pi
                            Text(String(format: "%+.0f°", angleDeg))
                                .font(.system(.title, design: .monospaced).bold())
                            Text("angle")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .padding(.bottom, 32)
            } else {
                Text(manager.isConnected ? "Waiting for UWB signal..." : "Waiting for connection...")
                    .foregroundStyle(.secondary)
                    .padding(.bottom, 32)
            }
        }
        .background(Color(UIColor.systemBackground))
    }

    private var statusColor: Color {
        if manager.isRanging { return .green }
        if manager.isConnected { return .yellow }
        return .orange
    }
}

// MARK: - Radar View

struct RadarView: View {
    let reading: TagReading?

    private let ringCount = 4

    /// Nice round scale tiers in metres.
    private let scaleTiers: [Float] = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]

    /// Default range when no reading is available.
    private let defaultRange: Float = 4.0

    /// Compute the ideal maxRange for the current distance.
    private var targetMaxRange: Float {
        guard let distance = reading?.distance else { return defaultRange }

        // Add 40% padding so the dot is not at the very edge
        let padded = distance * 1.4

        // Find the smallest tier that accommodates the padded distance
        for tier in scaleTiers {
            if tier >= padded {
                return tier
            }
        }

        // Beyond the largest tier: round up to the nearest 5
        return ceil(padded / 5) * 5
    }

    var body: some View {
        GeometryReader { geo in
            let size = min(geo.size.width, geo.size.height)
            let center = CGPoint(x: geo.size.width / 2, y: geo.size.height * 0.75)
            let maxRange = targetMaxRange
            let scale = CGFloat(size) * 0.42 / CGFloat(maxRange)

            ZStack {
                // Range rings
                ForEach(1...ringCount, id: \.self) { i in
                    let radius = CGFloat(i) / CGFloat(ringCount) * scale * CGFloat(maxRange)
                    Circle()
                        .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
                        .frame(width: radius * 2, height: radius * 2)
                        .position(center)

                    // Range label
                    let ringDistance = Float(i) * maxRange / Float(ringCount)
                    Text(ringLabelText(ringDistance))
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary.opacity(0.6))
                        .position(x: center.x + radius + 14, y: center.y)
                }

                // Angle sweep lines
                ForEach([-60, -30, 0, 30, 60], id: \.self) { deg in
                    let rad = Double(deg) * .pi / 180
                    let lineLen = scale * CGFloat(maxRange)
                    Path { path in
                        path.move(to: center)
                        path.addLine(to: CGPoint(
                            x: center.x + lineLen * CGFloat(sin(rad)),
                            y: center.y - lineLen * CGFloat(cos(rad))
                        ))
                    }
                    .stroke(Color.primary.opacity(0.06), lineWidth: 1)
                }

                // Cart marker (self)
                CartIcon()
                    .fill(Color.blue)
                    .frame(width: 28, height: 28)
                    .position(center)

                // Tag dot (shopper)
                if let r = reading {
                    let px = center.x + CGFloat(r.x) * scale
                    let py = center.y - CGFloat(r.y) * scale
                    let clampedX = clamp(px, min: 20, max: geo.size.width - 20)
                    let clampedY = clamp(py, min: 20, max: geo.size.height - 20)

                    // Dashed line from cart to tag
                    Path { path in
                        path.move(to: center)
                        path.addLine(to: CGPoint(x: clampedX, y: clampedY))
                    }
                    .stroke(Color.orange.opacity(0.25), style: StrokeStyle(lineWidth: 1, dash: [4, 4]))

                    // Pulse ring
                    Circle()
                        .strokeBorder(Color.orange.opacity(0.3), lineWidth: 2)
                        .frame(width: 36, height: 36)
                        .position(x: clampedX, y: clampedY)

                    // Solid dot
                    Circle()
                        .fill(Color.orange)
                        .frame(width: 18, height: 18)
                        .position(x: clampedX, y: clampedY)
                }
            }
            .animation(.easeInOut(duration: 0.5), value: maxRange)
        }
    }

    /// Format ring label: integer for whole numbers, one decimal otherwise.
    private func ringLabelText(_ distance: Float) -> String {
        if distance == distance.rounded() {
            return String(format: "%.0fm", distance)
        }
        return String(format: "%.1fm", distance)
    }

    private func clamp(_ value: CGFloat, min minVal: CGFloat, max maxVal: CGFloat) -> CGFloat {
        Swift.min(Swift.max(value, minVal), maxVal)
    }
}

// MARK: - Cart Icon Shape

/// A simple upward-pointing arrow representing the cart's position and heading.
struct CartIcon: Shape {
    func path(in rect: CGRect) -> Path {
        var p = Path()
        let w = rect.width, h = rect.height
        p.move(to:    CGPoint(x: w * 0.5,  y: 0))
        p.addLine(to: CGPoint(x: w,        y: h * 0.7))
        p.addLine(to: CGPoint(x: w * 0.65, y: h * 0.7))
        p.addLine(to: CGPoint(x: w * 0.65, y: h))
        p.addLine(to: CGPoint(x: w * 0.35, y: h))
        p.addLine(to: CGPoint(x: w * 0.35, y: h * 0.7))
        p.addLine(to: CGPoint(x: 0,        y: h * 0.7))
        p.closeSubpath()
        return p
    }
}

#Preview {
    ViewerRootView()
}
