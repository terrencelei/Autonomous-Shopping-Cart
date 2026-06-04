#!/usr/bin/env python3
"""
follow.py — standalone shopper-following controller, built directly from the
spin + displacement pseudocode.

  * Vision: reuses vision/yolo_detect.py as-is — its calibrated distance and
    angle estimates, IMX500 detector, and ByteTrack target lock.
  * Drive: speaks the ESP32 RPM protocol directly ("L<rpm> R<rpm>\\n") and
    reads "E,<l>,<r>\\n" encoder feedback.  The proven breakaway kick
    (KICK_RPM = 18 released after KICK_TICKS = 45 encoder ticks) is reused.
  * Control: implemented fresh from the pseudocode (drift / spin / follow).
    Fully independent of Pathfinding_algorithm.py — the hardware/dimensional
    specs it needs are cloned below (keep in sync if the cart is re-measured).

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
# HARDWARE / DIMENSIONAL SPECS  (cloned from Pathfinding_algorithm.py so this
# follower stands alone — keep in sync if the cart is re-measured)
# =============================================================================

WHEEL_DIAMETER_M = 0.06778        # outer drive-wheel diameter (m)
TRACK_M          = 0.333          # wheel spacing, centre-to-centre (m)
ENCODER_PPR      = 298            # encoder pulses per wheel rev (post-gearbox, 4x quadrature)
GEAR_RATIO       = 5              # motor gearbox ratio

WHEEL_CIRC  = math.pi * WHEEL_DIAMETER_M   # metres per wheel revolution
M_PER_PULSE = WHEEL_CIRC / ENCODER_PPR     # metres of wheel travel per encoder tick

# Drive direction / encoder polarity (flip if a wheel counts or drives backwards)
RIGHT_ENC_SIGN   = -1
LEFT_ENC_SIGN    = +1
RIGHT_MOTOR_SIGN = +1             # +1 = forward
LEFT_MOTOR_SIGN  = +1

# Serial link to the ESP32
MOTOR_PORT = "/dev/ttyUSB0"       # CP2102 USB-UART bridge
MOTOR_BAUD = 115200

# Motion limits / loop rate
MAX_SPEED = 0.5                   # m/s forward
DT        = 0.02                  # control loop period (s)
HOLD_DIST = 2.0                   # target hold distance (m)

# =============================================================================
# CALIBRATION  (speeds are WHEEL RPM unless noted; angles in radians, +CCW/left)
# =============================================================================

# Vision capture
CAM_W, CAM_H, CAM_FPS = 640, 480, 30

# Proven breakaway kick (see memory: cart-drive-calibration)
KICK_RPM      = 20.0  # wheel RPM burst to overcome static friction
KICK_TICKS    = 20    # release the kick once this many encoder ticks accumulate
KICK_TIMEOUT_S = 1.0  # give up holding the kick after this long if no movement is seen
TICKS_PER_SEC = 1.0 / DT   # control ticks per second (=50); used for open-loop fallback

# Spin (point-turn) controller
SPIN_KICK_RPM     = 12.0    # breakaway burst for a point turn (half the forward KICK_RPM)
TURN_RPM          = 2.0    # steady wheel RPM during the turn after the kick
ANGULAR_INERTIA   = 0.0    # s — yaw coast factor; drift = (w0-w1)*inertia   [calibrate]
THETA_THRESH_DEG  = 4.0    # follow loop: re-centre with spin() once |angle| exceeds this
SPIN_DEADBAND_DEG = 0.5    # spin: ignore turn requests smaller than this (no-op)
SPIN_TURN_TIMEOUT_S = 2.0  # max extra time for the steady-turn phase (exit a stuck/loaded turn)

# Distance (displacement) controller
THRESH_M       = HOLD_DIST   # standoff distance to hold (m)
LINEAR_INERTIA = 0.0    # s — forward coast factor, reserved for linear drift  [calibrate]
KP_DIST        = 15.0   # RPM speed change per (m/s) of approach rate dx        [calibrate]
KI_DIST        = 0.0    # RPM speed change per metre of standoff error (x-thresh) [calibrate]
DX_ALPHA       = 0.3    # EMA factor for the smoothed dx/dt

# Steering (angle) controller
KP_ANGLE = 40.0   # RPM of wheel-difference per radian of angle error          [calibrate]
KD_ANGLE = 0.0    # RPM of wheel-difference per (rad/s) of angular rate         [calibrate]

# Limits
MAX_RPM = 75.0   # wheel RPM cap (cart_motor.ino maps ~100 RPM to full PWM)


# =============================================================================
# HELPERS
# =============================================================================

TRACE = False   # set by run(); when True, echo serial traffic + control-flow trace


def _trace(msg):
    if TRACE:
        print(msg)


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _rpm_to_yaw(wheel_rpm):
    """Point-turn wheel RPM -> cart yaw rate (rad/s), magnitude."""
    v = wheel_rpm * WHEEL_CIRC / 60.0      # wheel linear speed (m/s)
    return 2.0 * v / TRACK_M               # differential yaw rate


def _ticks_to_yaw(d_left, d_right):
    """Encoder tick deltas -> swept yaw (rad), +CCW (right wheel ahead of left)."""
    left_m  = d_left  * M_PER_PULSE
    right_m = d_right * M_PER_PULSE
    return (right_m - left_m) / TRACK_M


def _ticks_to_rpm(ticks, dt):
    if dt <= 0.0:
        return 0.0
    return (ticks / ENCODER_PPR) * (60.0 / dt)


def drift(initial_rpm, final_rpm, inertia):
    """Extra travel coasted through on a commanded-rate step from initial to
    final, modelled as proportional to the change: ``(initial-final)*inertia``.
    Unit-agnostic; with inertia=0 it is a no-op."""
    return (initial_rpm - final_rpm) * inertia


def mix_wheels(S, rpm_diff):
    """Differential mix of forward speed S and steering rpm_diff into
    (left, right) wheel RPM, peak-scaled so neither wheel exceeds MAX_RPM."""
    if max(abs(S + rpm_diff), abs(S - rpm_diff)) > MAX_RPM:
        rpm_diff = math.copysign(
            min(abs(MAX_RPM - abs(S)), abs(MAX_RPM + abs(S))), rpm_diff)
    return S - rpm_diff, S + rpm_diff   # (left, right)


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
        cmd = f"L{LEFT_MOTOR_SIGN * l_rpm:.1f} R{RIGHT_MOTOR_SIGN * r_rpm:.1f}\n"
        _trace(f"  TX  {cmd.strip()}")
        if not self.has_serial:
            return
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
            _trace(f"  RX  {line}")
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
        if dl or dr:
            _trace(f"  enc Δ  L{dl:+d} R{dr:+d}")
        return dl, dr

    def flush(self):
        if self.has_serial:
            self._ser.reset_input_buffer()
        self._last_l = self._last_r = None


# =============================================================================
# SPIN  — point-turn through theta_rad (+CCW / left)
# =============================================================================

class Spinner:
    """Non-blocking drift-compensated point turn.

    ``start(theta)`` captures the turn; ``update(d_l, d_r)`` is called once per
    control tick with that tick's encoder deltas and returns the wheel commands
    ``(l_rpm, r_rpm)`` for this tick, or ``None`` when the turn is complete.

    Unlike the old blocking spin(), this advances ONE tick per call, so the
    camera/control loop keeps running at full rate during a re-centre — no more
    multi-second freeze while loaded wheels grind through a turn.

    Profile (from the pseudocode): breakaway kick at KICK_RPM until KICK_TICKS
    of encoder movement (or KICK_TIMEOUT_S), then a steady turn at TURN_RPM for
    the remaining angle (kick angle phi + inertial coast subtracted), bounded by
    SPIN_TURN_TIMEOUT_S so a stuck/loaded turn always exits.
    """

    def __init__(self):
        self._active = False

    @property
    def active(self):
        return self._active

    def cancel(self):
        self._active = False

    def start(self, theta_rad):
        if abs(theta_rad) < math.radians(SPIN_DEADBAND_DEG):
            self._active = False
            return
        self._active = True
        self._dir    = 1.0 if theta_rad > 0 else -1.0   # +CCW / left
        self._target = abs(theta_rad)
        self._phase  = "kick"
        self._kick_ticks = 0
        self._phi    = 0.0     # angle swept during the kick
        self._swept  = 0.0     # angle swept during the steady turn
        self._theta2 = 0.0     # remaining angle for the steady turn
        self._t0     = time.monotonic()
        _trace(f"spin start {math.degrees(theta_rad):+.1f}°  kick…")

    def update(self, d_l, d_r):
        """Advance one tick; return (l_rpm, r_rpm) or None when the turn ends."""
        if not self._active:
            return None
        now = time.monotonic()
        swept_inc = abs(_ticks_to_yaw(d_l, d_r))

        if self._phase == "kick":
            self._kick_ticks += abs(d_l) + abs(d_r)
            self._phi += swept_inc
            cmd_rpm = SPIN_KICK_RPM
            if self._kick_ticks >= KICK_TICKS or (now - self._t0) >= KICK_TIMEOUT_S:
                omega_turn = _rpm_to_yaw(TURN_RPM)
                kick_drift = drift(_rpm_to_yaw(SPIN_KICK_RPM), omega_turn, ANGULAR_INERTIA)
                stop_drift = drift(omega_turn, 0.0, ANGULAR_INERTIA)
                self._theta2 = self._target - self._phi - kick_drift - stop_drift
                self._phase = "turn"
                self._t0 = now
                _trace(f"  spin: phi={math.degrees(self._phi):.1f}° → "
                       f"turn {math.degrees(max(self._theta2, 0.0)):.1f}° @ {TURN_RPM}rpm")
                if self._theta2 <= 0.0:
                    self._active = False
                    return None
        else:  # steady-turn phase
            self._swept += swept_inc
            cmd_rpm = TURN_RPM
            omega_turn = _rpm_to_yaw(TURN_RPM)
            timeout = (self._theta2 / omega_turn if omega_turn > 0 else 0.0) + SPIN_TURN_TIMEOUT_S
            if self._swept >= self._theta2 or (now - self._t0) >= timeout:
                self._active = False
                return None

        return (-self._dir * cmd_rpm, +self._dir * cmd_rpm)


# =============================================================================
# FOLLOW CONTROLLER  (per-tick distance + steering, from the pseudocode)
# =============================================================================

class Kick:
    """Encoder-confirmed forward breakaway.

    Armed when the cart departs from rest; stays active until KICK_TICKS of
    encoder movement accumulate (proving it broke free) or KICK_TIMEOUT_S
    elapses.  While active the caller floors the forward-speed magnitude at
    KICK_RPM.  Kept separate from the control law so FollowController stays a
    pure PI + steering controller.
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
        _trace(f"kick.arm()  floor forward speed to {KICK_RPM}rpm until {KICK_TICKS} ticks")

    def cancel(self):
        self._active = False
        self._ticks = 0

    def update(self, d_l, d_r):
        """Feed this tick's encoder deltas; release once movement is confirmed
        (KICK_TICKS) or the timeout expires."""
        if not self._active:
            return
        self._ticks += abs(d_l) + abs(d_r)
        if self._ticks >= KICK_TICKS:
            self._active = False
            _trace(f"kick released: {self._ticks} ticks (breakaway confirmed)")
        elif (time.monotonic() - self._t0) >= KICK_TIMEOUT_S:
            self._active = False
            _trace(f"kick released: timeout after {KICK_TIMEOUT_S}s ({self._ticks} ticks)")


class FollowController:
    """Live state for the follow loop.  ``step`` returns (S, rpm_diff) — forward
    wheel RPM and steering RPM-difference — or None on a tick that ran a
    blocking re-centre spin (caller skips its send).  The breakaway kick lives
    outside, in the ``Kick`` helper."""

    def __init__(self):
        self.S = 0.0           # current forward wheel RPM
        self.dx = 0.0          # smoothed d(distance)/dt  (m/s)
        self._prev_x = None
        self._prev_theta = 0.0

    def after_spin(self):
        """Reset forward state after a re-centre spin: re-kick from rest and
        discard the stale distance derivative (the cart turned in place)."""
        self.S = 0.0
        self._prev_x = None

    def _derivatives(self, x, theta, dt):
        raw_dx = 0.0 if self._prev_x is None else (x - self._prev_x) / dt
        self.dx = (1.0 - DX_ALPHA) * self.dx + DX_ALPHA * raw_dx
        dtheta = (theta - self._prev_theta) / dt
        self._prev_x = x
        self._prev_theta = theta
        return dtheta

    def target_lost(self, motors=None):
        """Coast to a stop and reset derivative history when no target."""
        self.S = 0.0
        self._prev_x = None
        if motors is not None:
            motors.send_rpm(0.0, 0.0)

    def step(self, x, angle_deg, dt):
        """Distance + in-band steering. Returns (S, rpm_diff). Big re-centre
        turns are handled by the non-blocking Spinner in the run loop, so this
        only does small steering corrections while following."""
        theta = math.radians(-angle_deg)   # +CCW/left: positive gains steer toward target
        dtheta = self._derivatives(x, theta, dt)
        rpm_diff = KP_ANGLE * theta + KD_ANGLE * dtheta

        # Distance control (PI). Breakaway kick is applied by the caller's Kick.
        # Forward-only: stop at the hold distance or closer, never reverse.
        if x - THRESH_M < 0.0:
            _trace(f"step: dist {x:.2f} < hold {THRESH_M:.2f} → stop")
            self.S = 0.0
        else:
            delta_S = KP_DIST * self.dx + KI_DIST * (x - THRESH_M)
            self.S = _clip(self.S + delta_S, 0.0, MAX_RPM)   # clamp >= 0: no reverse
            _trace(f"step: PI dist={x:.2f} dx={self.dx:+.2f} → S={self.S:+.1f} diff={rpm_diff:+.1f}")
        return self.S, rpm_diff


# =============================================================================
# RUN  — wire vision + drive together
# =============================================================================

def run(drive=True, no_display=False, countdown=3, duration=0.0, trace=False):
    global TRACE
    TRACE = trace
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

    print("follow.py — spin-to-centre + PI distance follow")
    print(f"Hold {THRESH_M:.2f} m  centre band ±{THETA_THRESH_DEG:.0f}°  "
          f"turn {TURN_RPM} rpm  kick {KICK_RPM} rpm/{KICK_TICKS} ticks  max {MAX_RPM:.0f} rpm")

    if drive and motors.has_serial and countdown > 0:
        print(f"\n*** Cart will start following in {countdown}s ***")
        for k in range(countdown, 0, -1):
            print(f"  {k}...")
            time.sleep(1)
        print("  GO!\n")

    motors.flush()
    kick = Kick()
    spinner = Spinner()
    prev_S = 0.0
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
                _trace("follow: target lost → stop")
                ctrl.target_lost(motors)
                kick.cancel()
                spinner.cancel()
                prev_S = 0.0
                cmd_l_rpm = cmd_r_rpm = 0.0
                actual_l_rpm = actual_r_rpm = 0.0
                x = angle_deg = None
                mode_str = "LOST"
            else:
                x = target_row[3]          # calibrated distance (m)
                angle_deg = target_row[4]  # calibrated angle (deg, +right)
                d_l, d_r = motors.read_deltas()
                actual_l_rpm = _ticks_to_rpm(d_l, dt)
                actual_r_rpm = _ticks_to_rpm(d_r, dt)

                # Engage a non-blocking re-centre spin when too far off-centre.
                if not spinner.active and abs(angle_deg) > THETA_THRESH_DEG:
                    spinner.start(math.radians(-angle_deg))
                    kick.cancel()
                    ctrl.after_spin()
                    prev_S = 0.0

                if spinner.active:
                    cmd = spinner.update(d_l, d_r)   # one turn tick — never blocks
                    if cmd is None:                  # turn finished this tick
                        l_rpm = r_rpm = 0.0
                        ctrl.after_spin()
                    else:
                        l_rpm, r_rpm = cmd
                    mode_str = "SPIN"
                else:
                    S, rpm_diff = ctrl.step(x, angle_deg, dt)
                    if S == 0.0:           # holding station (too close) — no kick
                        kick.cancel()
                        mode_str = "STOP"
                    else:
                        if prev_S == 0.0:  # departing from rest
                            kick.arm()
                        kick.update(d_l, d_r)
                        if kick.active:    # floor the speed until breakaway is confirmed
                            S = math.copysign(max(abs(S), KICK_RPM), S)
                            ctrl.S = S      # keep the PI integrator consistent
                        mode_str = "KICK" if kick.active else "FOLLOW"
                    prev_S = S
                    l_rpm, r_rpm = mix_wheels(S, rpm_diff)

                cmd_l_rpm, cmd_r_rpm = l_rpm, r_rpm
                motors.send_rpm(l_rpm, r_rpm)

            Y.P.S.mode = mode_str

            if t - last_log >= 0.5:
                last_log = t
                if target_row is None:
                    print(f"t={t:5.1f}s  no target")
                else:
                    print(f"t={t:5.1f}s  dist={x:.2f}m  angle={angle_deg:+5.1f}°  "
                          f"dx={ctrl.dx:+.2f}m/s  {mode_str:6s}  "
                          f"cmd L{cmd_l_rpm:+5.1f} R{cmd_r_rpm:+5.1f}rpm  "
                          f"actual L{actual_l_rpm:+5.1f} R{actual_r_rpm:+5.1f}rpm")

            if not no_display:
                dist_str = f"dist={x:.2f}m ang={angle_deg:+.0f}°" if target_row else "no target"
                Y.cv2.putText(out, f"{mode_str}  {dist_str}", (10, 25),
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
    parser.add_argument("--trace", action="store_true",
                        help="echo serial traffic (TX/RX) and control-flow calls each tick")
    args = parser.parse_args()

    no_display = args.no_display
    if not no_display and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        no_display = True
        print("No display detected — running headless (use --no-display to silence).")

    run(drive=not args.no_drive, no_display=no_display,
        countdown=args.countdown, duration=args.duration, trace=args.trace)
