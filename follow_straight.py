#!/usr/bin/env python3
"""
follow_straight.py — standalone straight-line shopper-following controller,
built from the displacement pseudocode.

  * Vision: reuses vision/yolo_detect.py as-is — its calibrated distance and
    angle estimates, IMX500 detector, and ByteTrack target lock.
  * Drive: speaks the ESP32 RPM protocol directly ("L<rpm> R<rpm>\\n") and
    reads "E,<l>,<r>\\n" encoder feedback.  The proven breakaway kick
    (KICK_RPM = 18 released after KICK_TICKS = 45 encoder ticks) is reused.
  * Control: implemented fresh from the pseudocode (distance follow only).
    Self-contained — the hardware/dimensional specs it needs are defined
    below (keep in sync if the cart is re-measured).

Usage:
    python3 follow.py                 # live camera + drive
    python3 follow.py --no-display    # headless (SSH)
    python3 follow.py --no-drive      # vision only, no serial output
    python3 follow.py --duration 30   # stop after 30 s (0 = until Ctrl-C / Q)
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# HARDWARE / DIMENSIONAL SPECS  (measured cart geometry —
# keep in sync if the cart is re-measured)
# =============================================================================

WHEEL_DIAMETER_M = 0.06778        # outer drive-wheel diameter (m)
ENCODER_PPR      = 298            # encoder pulses per wheel rev (post-gearbox, 4x quadrature)
GEAR_RATIO       = 5              # motor gearbox ratio

WHEEL_CIRC  = math.pi * WHEEL_DIAMETER_M   # metres per wheel revolution

# Drive direction / encoder polarity (flip if a wheel counts or drives backwards)
RIGHT_ENC_SIGN   = -1
LEFT_ENC_SIGN    = +1
RIGHT_MOTOR_SIGN = +1             # +1 = forward
LEFT_MOTOR_SIGN  = +1

# Serial link to the ESP32
MOTOR_PORT = "/dev/ttyUSB0"       # CP2102 USB-UART bridge
MOTOR_BAUD = 115200

# Motion limits / loop rate
MAX_RPM   = 75.0                  # open-loop cap; cart_motor.ino maps 100 RPM to full PWM
DT        = 0.02                  # control loop period (s)
HOLD_DIST = 2.0                   # target hold distance (m)

# =============================================================================
# CALIBRATION  (speeds are WHEEL RPM unless noted; angles in radians, +CCW/left)
# =============================================================================

# Vision capture
CAM_W, CAM_H, CAM_FPS = 640, 480, 30

# Proven breakaway kick (see memory: cart-drive-calibration)
KICK_RPM      = 18.0  # wheel RPM burst to overcome static friction
KICK_TICKS    = 45    # release the kick once this many encoder ticks accumulate
KICK_TIMEOUT_S = 1.0  # give up holding the kick after this long if no movement is seen
TICKS_PER_SEC = 1.0 / DT   # control ticks per second (=50); used for open-loop fallback

# Distance controller. Use a damped proportional target speed instead of
# accumulating speed every frame; that avoids overshoot/stop/overshoot cycles.
THRESH_M       = HOLD_DIST   # standoff distance to hold (m)
DIST_DEADBAND_M = 0.20       # no drive inside this band around hold distance
KP_DIST        = 30.0        # RPM per metre beyond the deadband
KD_DIST        = 8.0         # RPM per (m/s) target-distance closing rate
DX_ALPHA       = 0.2         # EMA factor for smoothed dx/dt
RPM_SLEW_PER_S = 35.0        # max command change per second

# Straight-line heading hold (encoder-driven). Trims the two wheels to keep
# their travel equal so the cart doesn't curve on uneven friction. NOT camera /
# angle steering — it only cancels drift to hold the heading it started with.
STRAIGHT_KP       = 0.20     # RPM trim per tick of accumulated L-R imbalance   [calibrate]
STRAIGHT_KD       = 2.0      # RPM trim per tick of this-tick L-R imbalance      [calibrate]
STRAIGHT_TRIM_MAX = 15.0     # cap on the heading-hold trim (RPM)

# =============================================================================
# HELPERS
# =============================================================================

def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _ticks_to_rpm(ticks, dt):
    if dt <= 0.0:
        return 0.0
    return (ticks / ENCODER_PPR) * (60.0 / dt)


# =============================================================================
# MOTOR I/O  (RPM-native ESP32 link)
# =============================================================================

class MotorIO:
    """Direct ESP32 serial link.

    Outbound: "L<rpm> R<rpm>\\n"   (wheel RPM, motor signs applied)
    Inbound:  "E,<left>,<right>\\n" (cumulative signed encoder counts)
    """

    def __init__(self, port=None, drive=True):
        self._ser = None
        self._last_l = self._last_r = None
        if not drive:
            return
        try:
            import serial
            self._ser = serial.Serial(port or MOTOR_PORT, MOTOR_BAUD, timeout=0.01)
            print(f"ESP32 connected on {port or MOTOR_PORT}")
        except Exception as e:
            print(f"WARNING: ESP32 unavailable ({e}) — running without drive")

    @property
    def has_serial(self):
        return self._ser is not None and self._ser.is_open

    def send_rpm(self, l_rpm, r_rpm):
        if not self.has_serial:
            return
        cmd = f"L{LEFT_MOTOR_SIGN * l_rpm:.1f} R{RIGHT_MOTOR_SIGN * r_rpm:.1f}\n"
        self._ser.write(cmd.encode())

    def stop(self):
        self.send_rpm(0.0, 0.0)

    def read_deltas(self):
        """Drain the serial buffer; return net (d_left, d_right) tick deltas
        since the last call (signs applied)."""
        if not self.has_serial:
            return 0, 0
        dl = dr = 0
        while self._ser.in_waiting:
            try:
                line = self._ser.readline().decode(errors="ignore").strip()
            except Exception:
                break
            if not line.startswith("E,"):
                continue
            try:
                _, ls, rs = line.split(",", 2)
                lc = LEFT_ENC_SIGN * int(ls)
                rc = RIGHT_ENC_SIGN * int(rs)
            except ValueError:
                continue
            if self._last_l is not None:
                dl += lc - self._last_l
                dr += rc - self._last_r
            self._last_l, self._last_r = lc, rc
        return dl, dr

    def flush(self):
        if self.has_serial:
            self._ser.reset_input_buffer()
        self._last_l = self._last_r = None


# =============================================================================
# FOLLOW CONTROLLER  (per-tick distance control)
# =============================================================================

class Kick:
    """Forward breakaway helper.

    Armed when the cart departs from rest; stays active until KICK_TICKS of
    encoder movement accumulate or KICK_TIMEOUT_S elapses. While active the
    caller floors the forward-speed magnitude at KICK_RPM.
    """

    def __init__(self):
        self._active = False
        self._ticks = 0
        self._t0 = 0.0

    @property
    def active(self):
        return self._active

    def arm(self):
        self._active = True
        self._ticks = 0
        self._t0 = time.monotonic()

    def cancel(self):
        self._active = False
        self._ticks = 0

    def update(self, d_l, d_r):
        """Feed this tick's encoder deltas; release once movement is confirmed
        (KICK_TICKS) or the timeout expires."""
        if not self._active:
            return
        self._ticks += abs(d_l) + abs(d_r)
        if self._ticks >= KICK_TICKS or (time.monotonic() - self._t0) >= KICK_TIMEOUT_S:
            self._active = False


class FollowController:
    """Live state for the follow loop.  ``step`` returns a forward wheel RPM.
    The breakaway kick lives outside, in the ``Kick`` helper."""

    def __init__(self):
        self.S = 0.0           # current forward wheel RPM
        self.dx = 0.0          # smoothed d(distance)/dt  (m/s)
        self._prev_x = None

    def _derivatives(self, x, dt):
        raw_dx = 0.0 if self._prev_x is None else (x - self._prev_x) / dt
        self.dx = (1.0 - DX_ALPHA) * self.dx + DX_ALPHA * raw_dx
        self._prev_x = x

    def target_lost(self, motors=None):
        """Coast to a stop and reset derivative history when no target."""
        self.S = 0.0
        self._prev_x = None
        if motors is not None:
            motors.send_rpm(0.0, 0.0)

    def step(self, x, dt, motors=None):
        # Distance is the ONLY input — no angle/steering term by design.
        self._derivatives(x, dt)

        error = x - THRESH_M
        if error <= DIST_DEADBAND_M:
            # At the hold distance or closer → stop. Forward-only: this
            # follower never reverses, it only drives when the shopper is too far.
            target_s = 0.0
        else:
            # Shopper is too far → drive forward.
            # Positive dx (target moving away) adds speed; negative dx
            # (closing in) eases off early, but clamped to 0 so it never reverses.
            drive_error = error - DIST_DEADBAND_M
            target_s = KP_DIST * drive_error + KD_DIST * self.dx
            target_s = _clip(target_s, 0.0, MAX_RPM)

        max_delta = RPM_SLEW_PER_S * dt
        self.S = _clip(target_s, self.S - max_delta, self.S + max_delta)
        return self.S


# =============================================================================
# RUN  — wire vision + drive together
# =============================================================================

def run(drive=True, no_display=False, countdown=3, duration=0.0):
    try:
        from vision import yolo_detect as Y
    except Exception as e:
        raise SystemExit(f"ERROR: could not import vision/yolo_detect.py: {e}")

    if not Y.RPK_MODEL_PATH.exists():
        raise SystemExit(
            f"ERROR: {Y.RPK_MODEL_PATH} not found. Install with: sudo apt install imx500-models"
        )
    try:
        cap = Y.IMX500Capture(model_path=Y.RPK_MODEL_PATH, width=CAM_W, height=CAM_H, fps=CAM_FPS)
    except RuntimeError as e:
        raise SystemExit(f"ERROR: could not open IMX500 camera: {e}")

    motors = MotorIO(drive=drive)
    tracker = Y.sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=60,
        minimum_matching_threshold=0.8,
        frame_rate=CAM_FPS,
    )
    smooth_state = {}
    target_lock = Y.TargetLock()
    ctrl = FollowController()

    print("follow_straight.py — straight PI distance follow")
    print(f"Hold {THRESH_M:.2f} m  kick {KICK_RPM} rpm/{KICK_TICKS} ticks  max {MAX_RPM:.0f} rpm")

    if drive and motors.has_serial and countdown > 0:
        print(f"\n*** Cart will start following in {countdown}s ***")
        for k in range(countdown, 0, -1):
            print(f"  {k}...")
            time.sleep(1)
        print("  GO!\n")

    motors.flush()
    kick = Kick()
    prev_S = 0.0
    heading_err = 0.0          # accumulated L-R encoder imbalance for the current segment
    start = time.monotonic()
    prev_t = start
    last_log = 0.0
    cmd_l_rpm = cmd_r_rpm = 0.0
    actual_l_rpm = actual_r_rpm = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            now = time.monotonic()
            t = now - start
            if duration > 0 and t >= duration:
                break
            dt = now - prev_t
            prev_t = now
            if dt <= 0.0:
                dt = DT

            dets = cap.get_detections()
            tracked = tracker.update_with_detections(dets)
            out, rows = Y.annotate_frame(frame, tracked, smooth_state, target_lock, now)
            target_row = next((r for r in rows if r[0] == "TARGET"), None)

            if target_row is None:
                ctrl.target_lost(motors)
                kick.cancel()
                prev_S = 0.0
                heading_err = 0.0
                cmd_l_rpm = cmd_r_rpm = 0.0
                actual_l_rpm = actual_r_rpm = 0.0
                x = angle_deg = None
            else:
                x = target_row[3]          # calibrated distance (m)
                angle_deg = target_row[4]  # read for the status readout only — never steered on
                S = ctrl.step(x, dt, motors=motors)
                d_l, d_r = motors.read_deltas()
                actual_l_rpm = _ticks_to_rpm(d_l, dt)
                actual_r_rpm = _ticks_to_rpm(d_r, dt)
                if S == 0.0:               # holding station (too close) — stop; reset heading hold
                    kick.cancel()
                    heading_err = 0.0
                    l_rpm = r_rpm = 0.0
                else:
                    if prev_S == 0.0:      # departing from rest: start a fresh straight segment
                        kick.arm()
                        heading_err = 0.0
                    kick.update(d_l, d_r)
                    if kick.active:        # floor the speed until breakaway is confirmed
                        S = math.copysign(max(abs(S), KICK_RPM), S)
                        ctrl.S = S         # keep the integrator consistent
                    # Straight-line heading hold (encoder-driven, NOT camera steering):
                    # trim the lagging wheel up so left/right travel stays equal.
                    heading_err += (d_l - d_r)
                    trim = _clip(STRAIGHT_KP * heading_err + STRAIGHT_KD * (d_l - d_r),
                                 -STRAIGHT_TRIM_MAX, STRAIGHT_TRIM_MAX)
                    l_rpm = _clip(S - trim, 0.0, MAX_RPM)
                    r_rpm = _clip(S + trim, 0.0, MAX_RPM)
                prev_S = S
                cmd_l_rpm, cmd_r_rpm = l_rpm, r_rpm
                motors.send_rpm(l_rpm, r_rpm)

            if t - last_log >= 0.5:
                last_log = t
                if target_row is None:
                    print(f"t={t:5.1f}s  no target")
                else:
                    tag = ("KICK " if kick.active else "") + f"S={ctrl.S:+6.1f}rpm"
                    print(f"t={t:5.1f}s  dist={x:.2f}m  angle={angle_deg:+5.1f}°  "
                          f"dx={ctrl.dx:+.2f}m/s  {tag}  "
                          f"cmd L{cmd_l_rpm:+5.1f} R{cmd_r_rpm:+5.1f}rpm  "
                          f"actual L{actual_l_rpm:+5.1f} R{actual_r_rpm:+5.1f}rpm")

            if not no_display:
                dist_str = f"dist={x:.2f}m ang={angle_deg:+.0f}°" if target_row else "no target"
                Y.cv2.putText(out, f"FOLLOW  {dist_str}", (10, 25),
                              Y.cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                Y.cv2.imshow("Cart View", out)
                if Y.cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        motors.stop()
        cap.release()
        if not no_display:
            Y.cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--no-drive", action="store_true",
                        help="don't open the ESP32 serial port (vision only)")
    parser.add_argument("--no-display", action="store_true",
                        help="suppress OpenCV windows (headless / SSH)")
    parser.add_argument("--countdown", type=int, default=3,
                        help="seconds before the cart starts moving (default: 3)")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="run time in seconds; 0 runs until Ctrl-C / Q (default: 0)")
    args = parser.parse_args()

    no_display = args.no_display
    if not no_display and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        no_display = True
        print("No display detected — running headless (use --no-display to silence).")

    run(drive=not args.no_drive, no_display=no_display,
        countdown=args.countdown, duration=args.duration)
