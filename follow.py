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
KICK_RPM      = 18.0  # wheel RPM burst to overcome static friction
KICK_TICKS    = 45    # release the kick once this many encoder ticks accumulate
KICK_TIMEOUT_S = 1.0  # give up holding the kick after this long if no movement is seen
TICKS_PER_SEC = 1.0 / DT   # control ticks per second (=50); used for open-loop fallback

# Spin (point-turn) controller
TURN_RPM          = 6.0    # steady wheel RPM during the turn after the kick
ANGULAR_INERTIA   = 0.0    # s — yaw coast factor; drift = (w0-w1)*inertia   [calibrate]
THETA_THRESH_DEG  = 8.0    # follow loop: re-centre with spin() once |angle| exceeds this
SPIN_DEADBAND_DEG = 0.5    # spin(): ignore turn requests smaller than this (no-op)

# Distance (displacement) controller
THRESH_M       = HOLD_DIST   # standoff distance to hold (m)
LINEAR_INERTIA = 0.0    # s — forward coast factor, reserved for linear drift  [calibrate]
KP_DIST        = 15.0   # RPM speed change per (m/s) of approach rate dx        [calibrate]
KI_DIST        = 15.0   # RPM speed change per metre of standoff error (x-thresh) [calibrate]
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

def _turn_wheels(motors, wheel_rpm, direction):
    """Command a point turn at wheel_rpm * direction (+1 = CCW/left)."""
    motors.send_rpm(-direction * wheel_rpm, +direction * wheel_rpm)


def spin(theta_rad, motors):
    """Re-square on the target by turning through ``theta_rad`` (+CCW/left).

    Profile (from the pseudocode): a breakaway kick at KICK_RPM, then a steady
    turn at TURN_RPM for the remaining angle.  The kick's swept angle (phi) and
    the inertial coast on each speed change (kick->turn, turn->0) are subtracted
    so the *total* swept angle lands on theta.

        phi        = angle swept during the kick
        kick_drift = drift(kick, turn, inertia)
        stop_drift = drift(turn, 0,    inertia)
        theta_2    = |theta| - phi - kick_drift - stop_drift   # steady-turn part

    With encoders present each phase is closed-loop on the measured swept angle
    (robust); without them it falls back to open-loop timing.
    """
    if abs(theta_rad) < math.radians(SPIN_DEADBAND_DEG):
        return
    direction  = 1.0 if theta_rad > 0 else -1.0
    target     = abs(theta_rad)
    omega_kick = _rpm_to_yaw(KICK_RPM)
    omega_turn = _rpm_to_yaw(TURN_RPM)
    closed_loop = motors is not None and motors.has_serial
    _trace(f"spin({math.degrees(theta_rad):+.1f}°, "
           f"{'closed-loop' if closed_loop else 'open-loop'})  kick…")

    # --- breakaway kick: hold until KICK_TICKS ticks; measure swept angle ----
    if closed_loop:
        motors.flush()
        _turn_wheels(motors, KICK_RPM, direction)
        acc_ticks = 0
        phi = 0.0
        t_timeout = time.monotonic() + 1.5
        while acc_ticks < KICK_TICKS and time.monotonic() < t_timeout:
            time.sleep(DT)
            dl, dr = motors.read_deltas()
            acc_ticks += abs(dl) + abs(dr)
            phi += abs(_ticks_to_yaw(dl, dr))
    else:
        kick_time = KICK_TICKS / TICKS_PER_SEC
        if motors is not None:
            _turn_wheels(motors, KICK_RPM, direction)
            time.sleep(kick_time)
        phi = omega_kick * kick_time

    kick_drift = drift(omega_kick, omega_turn, ANGULAR_INERTIA)
    stop_drift = drift(omega_turn, 0.0, ANGULAR_INERTIA)
    theta_2 = target - phi - kick_drift - stop_drift
    _trace(f"  spin: phi={math.degrees(phi):.1f}° → turn {math.degrees(max(theta_2,0)):.1f}° @ {TURN_RPM}rpm")

    # --- steady turn for the remaining angle ---------------------------------
    if theta_2 > 0.0 and omega_turn > 0.0:
        if closed_loop:
            _turn_wheels(motors, TURN_RPM, direction)
            swept = 0.0
            t_timeout = time.monotonic() + theta_2 / omega_turn + 1.5
            while swept < theta_2 and time.monotonic() < t_timeout:
                time.sleep(DT)
                dl, dr = motors.read_deltas()
                swept += abs(_ticks_to_yaw(dl, dr))
        else:
            if motors is not None:
                _turn_wheels(motors, TURN_RPM, direction)
            time.sleep(theta_2 / omega_turn)

    if motors is not None:
        motors.stop()


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
        self.spun = False
        self._prev_x = None
        self._prev_theta = 0.0

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

    def step(self, x, angle_deg, dt, motors=None):
        self.spun = False
        theta = math.radians(-angle_deg)   # +CCW/left: positive gains steer toward target
        dtheta = self._derivatives(x, theta, dt)

        # Centering — always parallel, independent of distance
        if abs(theta) > math.radians(THETA_THRESH_DEG):
            _trace(f"step: |angle|={abs(angle_deg):.1f}° > {THETA_THRESH_DEG}° → spin()")
            spin(theta, motors)
            self.S = 0.0
            self._prev_x = None            # distance derivative invalid after the turn
            self.spun = True
            return None                    # skip distance control this tick

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
                prev_S = 0.0
                cmd_l_rpm = cmd_r_rpm = 0.0
                actual_l_rpm = actual_r_rpm = 0.0
                x = angle_deg = None
                mode_str = "LOST"
            else:
                x = target_row[3]          # calibrated distance (m)
                angle_deg = target_row[4]  # calibrated angle (deg, +right)
                res = ctrl.step(x, angle_deg, dt, motors=motors)
                if res is None:            # a blocking re-centre spin ran this tick
                    kick.cancel()
                    prev_S = 0.0
                    mode_str = "SPIN"
                else:
                    S, rpm_diff = res
                    d_l, d_r = motors.read_deltas()
                    actual_l_rpm = _ticks_to_rpm(d_l, dt)
                    actual_r_rpm = _ticks_to_rpm(d_r, dt)
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
                    if ctrl.spun:
                        tag = "SPIN"
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
    parser.add_argument("--trace", action="store_true",
                        help="echo serial traffic (TX/RX) and control-flow calls each tick")
    args = parser.parse_args()

    no_display = args.no_display
    if not no_display and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        no_display = True
        print("No display detected — running headless (use --no-display to silence).")

    run(drive=not args.no_drive, no_display=no_display,
        countdown=args.countdown, duration=args.duration, trace=args.trace)
