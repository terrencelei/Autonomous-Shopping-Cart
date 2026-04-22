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

            // Bottom section — mutually exclusive states
            Group {
                if manager.isCalibrating {
                    DistanceCalibrationView(progress: manager.calibrationProgress)
                } else if let r = manager.reading {
                    ReadoutView(reading: r, manager: manager)
                } else {
                    Text(manager.isConnected ? "Waiting for UWB signal…" : "Waiting for connection…")
                        .foregroundStyle(.secondary)
                        .padding(.bottom, 32)
                }
            }
            .padding(.horizontal)
            .padding(.bottom, 24)
        }
        .background(Color(UIColor.systemBackground))
    }

    private var statusColor: Color {
        if manager.isRanging { return .green }
        if manager.isConnected { return .yellow }
        return .orange
    }
}

// MARK: - Readout (console-style raw figures)

private struct ReadoutView: View {
    let reading: TagReading
    let manager: ViewerSessionManager

    private var dirString: String {
        guard let d = reading.direction else { return "nil" }
        return String(format: "(%.3f, %.3f, %.3f)", d.x, d.y, d.z)
    }

    private var hAngleString: String {
        guard let h = reading.horizontalAngle else { return "nil" }
        return String(format: "%.3f rad  %+.1f°", h, h * 180 / .pi)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            row("dist",   String(format: "%.3f m", reading.distance))
            row("dir",    dirString)
            row("hAngle", hAngleString)
            row("vert",   "\(reading.verticalEstimate)  (\(reading.verticalLabel))")

            HStack(spacing: 12) {
                Button { manager.beginCalibration() } label: {
                    Label("Set Zero", systemImage: "scope")
                        .font(.caption.bold())
                        .padding(.horizontal, 14)
                        .padding(.vertical, 7)
                        .background(Color.blue.opacity(0.15))
                        .foregroundStyle(.blue)
                        .clipShape(Capsule())
                }
                if manager.calibrationOffset != 0 {
                    Button { manager.resetCalibration() } label: {
                        Label("Reset", systemImage: "arrow.counterclockwise")
                            .font(.caption)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 7)
                            .background(Color.secondary.opacity(0.12))
                            .foregroundStyle(.secondary)
                            .clipShape(Capsule())
                    }
                }
            }
            .padding(.top, 4)
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 12)
        .background(Color(UIColor.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack(alignment: .top, spacing: 0) {
            Text(label.padding(toLength: 8, withPad: " ", startingAt: 0))
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.primary)
        }
    }
}

// MARK: - Distance Calibration Progress

struct DistanceCalibrationView: View {
    let progress: Double

    var body: some View {
        VStack(spacing: 8) {
            ZStack {
                Circle()
                    .stroke(Color.secondary.opacity(0.2), lineWidth: 5)
                Circle()
                    .trim(from: 0, to: progress)
                    .stroke(Color.blue, style: StrokeStyle(lineWidth: 5, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                    .animation(.linear(duration: 0.1), value: progress)
            }
            .frame(width: 44, height: 44)
            Text("Hold phones together…")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
    }
}

// MARK: - Radar View

struct RadarView: View {
    let reading: TagReading?

    private let ringCount = 4
    private let scaleTiers: [Float] = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
    private let defaultRange: Float = 4.0

    private var targetMaxRange: Float {
        guard let distance = reading?.distance else { return defaultRange }
        let padded = distance * 1.4
        for tier in scaleTiers { if tier >= padded { return tier } }
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

                    Path { path in
                        path.move(to: center)
                        path.addLine(to: CGPoint(x: clampedX, y: clampedY))
                    }
                    .stroke(Color.orange.opacity(0.25), style: StrokeStyle(lineWidth: 1, dash: [4, 4]))

                    Circle()
                        .strokeBorder(Color.orange.opacity(0.3), lineWidth: 2)
                        .frame(width: 36, height: 36)
                        .position(x: clampedX, y: clampedY)

                    Circle()
                        .fill(Color.orange)
                        .frame(width: 18, height: 18)
                        .position(x: clampedX, y: clampedY)
                }
            }
            .animation(.easeInOut(duration: 0.5), value: maxRange)
        }
    }

    private func ringLabelText(_ distance: Float) -> String {
        distance == distance.rounded()
            ? String(format: "%.0fm", distance)
            : String(format: "%.1fm", distance)
    }

    private func clamp(_ value: CGFloat, min minVal: CGFloat, max maxVal: CGFloat) -> CGFloat {
        Swift.min(Swift.max(value, minVal), maxVal)
    }
}

// MARK: - Cart Icon Shape

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
