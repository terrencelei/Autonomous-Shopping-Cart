#!/usr/bin/env python3
"""
follow.py — standalone shopper-following controller.

  * Vision: reuses vision/yolo_detect.py as-is — its calibrated distance and
    angle estimates, IMX500 detector, and ByteTrack target lock.
  * Drive: speaks the ESP32 RPM protocol directly ("L<rpm> R<rpm>\\n") and
    reads "E,<l>,<r>\\n" encoder feedback.
  * Control: continuous damped-P loops on both axes — distance (deadband +
    hysteresis + slew) and steering (angle deadband + slew) — each
    coast-compensated by an inertia constant learned online from every
    commanded stop of that axis (no startup calibration). Encoder-confirmed
    breakaway kicks beat static friction on departure from rest and on
    standing point turns.

Usage:
    python3 follow.py                 # live camera + drive
    python3 follow.py --no-display    # headless (SSH)
    python3 follow.py --no-drive     # vision only, no serial output
    python3 follow.py --duration 30   # stop after 30 s (0 = until Ctrl-C / Q)
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_RPK_MODEL_PATH = Path(
    "/usr/share/imx500-models/imx500_network_yolo11n_pp.rpk"
)

# =============================================================================
# HARDWARE / DIMENSIONAL SPECS  (measured cart geometry —
# keep in sync if the cart is re-measured)
# =============================================================================

WHEEL_DIAMETER_M = 0.06778        # outer drive-wheel diameter (m)
TRACK_M          = 0.333          # wheel spacing, center-to-center (m)
ENCODER_PPR      = 298            # encoder pulses per wheel rev (post-gearbox, 4x quadrature)

WHEEL_CIRC  = math.pi * WHEEL_DIAMETER_M   # meters per wheel revolution
M_PER_PULSE = WHEEL_CIRC / ENCODER_PPR     # meters of wheel travel per encoder tick

# Drive direction / encoder polarity (flip if a wheel counts or drives backwards)
RIGHT_ENC_SIGN   = -1
LEFT_ENC_SIGN    = +1
RIGHT_MOTOR_SIGN = +1             # +1 = forward
LEFT_MOTOR_SIGN  = +1

# Serial link to the ESP32
MOTOR_PORT = "/dev/ttyUSB0"       # CP2102 USB-UART bridge
MOTOR_BAUD = 115200

# Motion limits / loop rate
DT        = 0.02                  # nominal control period (s); real dt is measured
HOLD_DIST = 2.0                   # target hold distance (m)

# =============================================================================
# CALIBRATION  (speeds are WHEEL RPM unless noted; angles in radians, +CCW/left)
# =============================================================================

# Vision capture
CAM_W, CAM_H, CAM_FPS = 640, 480, 30
VISION_WARMUP_S = 15.0   # run vision (no drive) this long at startup so color tracking settles
SEARCH_PERSON_CONFIDENCE = 0.15  # lower YOLO person threshold while spin-searching
SEARCH_TRACK_ACTIVATION = 0.05   # lower ByteTrack activation threshold while spin-searching
SEARCH_TRACK_MATCHING = 0.5      # looser ByteTrack matching while spin-searching

# Breakaway kicks (encoder-confirmed bursts that beat static friction)
KICK_RPM       = 50.0  # forward kick: floor on departure from rest
SPIN_KICK_RPM  = 18.0  # point-turn kick: floor for a standing re-center / search spin
KICK_TICKS     = 30    # release a kick once this many encoder ticks accumulate
KICK_TIMEOUT_S = 0.5   # if still stalled after this long, ramp the kick up (don't give up)
KICK_RAMP_STEP = 10.0  # wheel RPM added to a kick each KICK_TIMEOUT_S it stays stalled

# Steering (angle) controller — continuous damped P, the angular twin of the
# distance law: no discrete spin maneuver, the shopper is steered back to
# center every tick while driving (the old Spinner pulsed a point turn each
# time the angle crossed a threshold — same limit cycle the forward
# controller had at the hold ring).
KP_ANGLE           = 120.0  # steering RPM-difference per radian of angle error  [calibrate]
ANGLE_DEADBAND_DEG = 2.0    # ignore target angles smaller than this (vision noise)
DIFF_SLEW_PER_S    = 300.0  # max steering-difference change per second
TURN_STALL_MIN_RPM = 8.0    # standing-turn demand below this never arms a spin kick
STEER_SIGN = +1   # angle→turn polarity. Flip to -1 if the cart corrects the WRONG way.

# Yaw coast compensation: steering eases off by the angle the cart will coast
# through at its measured yaw rate (theta_eff = theta - omega*ANGULAR_INERTIA).
# ANGULAR_INERTIA is learned online from every commanded turn-stop (search
# exits, standing re-centers, return-home turns) — no startup calibration.
ANGULAR_INERTIA = 0.15  # s — initial guess; refined online by the InertiaLearner
OMEGA_ALPHA     = 0.3   # EMA for the encoder-measured yaw rate (rad/s)

# Reacquire when the target leaves the frame: after a grace window (so a
# one-frame dropout doesn't react) spin-search toward the last-seen side.
SEARCH_GRACE_S = 0.3   # coast (don't search) this long after losing the target

# Distance (displacement) controller — damped proportional with a deadband and
# slew limiting (the law follow_straight.py proved out; an integrating law
# wound up and lurched at the hold ring).
THRESH_M        = HOLD_DIST  # standoff distance to hold (m)
DIST_DEADBAND_M = 0.20       # no-drive band beyond the hold distance
RESUME_HYST_M   = 0.10       # once stopped, error must exceed deadband+this to restart
KP_DIST         = 30.0       # RPM per meter beyond the deadband                 [calibrate]
KD_DIST         = 8.0        # RPM per (m/s) of opening/closing rate dx          [calibrate]
DX_ALPHA        = 0.2        # EMA factor for the smoothed dx/dt
MIN_DRIVE_RPM   = 12.0       # smallest nonzero forward command (lower duty just stalls) [calibrate]
RPM_SLEW_UP_PER_S   = 50.0   # max command increase per second (gentle accel)
RPM_SLEW_DOWN_PER_S = 150.0  # max command decrease per second (firm, but no brake-slam)

# Forward coast compensation: brake early by the distance the cart will coast
# (error -= v_fwd*LINEAR_INERTIA). Learned online from every commanded stop.
LINEAR_INERTIA = 0.0    # s — forward coast factor; learned by the InertiaLearner
V_FWD_ALPHA    = 0.3    # EMA for the encoder-measured forward speed (m/s)

# Basic obstacle avoidance: pause forward motion when an obstacle is closer than
# the target and roughly in the path; resume normal follow once it clears.
OBSTACLE_BLOCK_DEG = 20.0   # obstacle within this of straight-ahead counts as "in the way"
OBSTACLE_MARGIN_M  = 0.3    # obstacle must be at least this much closer than the target to block
OBSTACLE_HOLD_S    = 0.6    # stay blocked this long after the last obstacle-in-path sighting

# Return-home: if the cart is spun ~180° while stopped (i.e. an uncommanded
# rotation), dead-reckon back to the start pose (0,0,0) along a direct line.
HOME_ROT_TRIGGER_DEG = 180.0   # uncommanded rotation (while stopped) that triggers a return
HOME_ROT_RANGE_DEG   = 30.0    # ± tolerance band around the trigger angle
HOME_TURN_RPM        = 40.0    # wheel RPM for return-home point turns               [calibrate]
HOME_FWD_RPM         = 50.0    # wheel RPM for the drive-home leg                     [calibrate]
HOME_DRIVE_KP        = 60.0    # steering RPM per rad of bearing error while driving  [calibrate]
HOME_ANGLE_TOL_DEG   = 5.0     # a turn is "done" within this
HOME_DIST_TOL_M      = 0.15    # home reached within this

# Slip rejection (odometry): a free-spinning wheel reads more ticks than the
# cart actually moved. If the wheels diverge beyond the commanded differential
# by more than this rate, the faster wheel is slipping — trust the slower
# (gripping) wheel and reconstruct the faster one from it + the commanded turn.
SLIP_DIFF_TICKS_PER_S = 120.0  # ticks/s of unexplained wheel divergence = slip [calibrate]

# Limits — cart_motor.ino clamps commands to ±100 and maps 100 → full PWM, so
# any cap above 100 is pure windup headroom that saturates the firmware.
MAX_RPM = 100.0  # wheel RPM cap = cart_motor.ino's full-PWM point


# =============================================================================
# HELPERS
# =============================================================================

TRACE = False   # set by run(); when True, echo serial traffic + control-flow trace


def _trace(msg):
    if TRACE:
        print(msg)


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _wrap_pi(a):
    """Wrap an angle (rad) to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _ticks_to_yaw(d_left, d_right):
    """Encoder tick deltas -> swept yaw (rad), +CCW (right wheel ahead of left)."""
    left_m  = d_left  * M_PER_PULSE
    right_m = d_right * M_PER_PULSE
    return (right_m - left_m) / TRACK_M


def _ticks_to_rpm(ticks, dt):
    if dt <= 0.0:
        return 0.0
    return (ticks / ENCODER_PPR) * (60.0 / dt)


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
# SEARCH  — spin in place until the target reappears
# =============================================================================

class Searcher:
    """Spin in place toward the last-seen side until the target reappears.

    ``start(direction)`` picks the spin direction; ``update`` is called once per
    camera frame and returns the point-turn wheel command. It spins **forever**
    (never gives up) — the caller cancels it the instant a target reappears.
    Starts at SPIN_KICK_RPM and ramps up by KICK_RAMP_STEP each KICK_TIMEOUT_S
    it's stalled (up to MAX_RPM) so it reliably overcomes stall.
    """

    def __init__(self):
        self._active = False

    @property
    def active(self):
        return self._active

    def cancel(self):
        self._active = False

    def start(self, direction):
        self._active = True
        self._dir = 1.0 if direction >= 0 else -1.0
        self._rpm = SPIN_KICK_RPM
        self._yaw = 0.0
        self._yaw_at_ramp = 0.0
        self._t_ramp = time.monotonic()
        self._slowed_once = False
        _trace(f"search start dir={'+L' if self._dir > 0 else '-R'} — spinning")

    def update(self, d_l, d_r):
        """Spin-search. Every 360° with no target, reverse direction.
        Only slow down on the first 360° reversal. Stall recovery always runs."""
        if not self._active:
            return None

        self._yaw += abs(_ticks_to_yaw(d_l, d_r))
        now = time.monotonic()

        # Every 360°, reverse direction.
        if self._yaw >= math.radians(360.0):
            self._dir *= -1.0
            self._yaw = 0.0
            self._yaw_at_ramp = 0.0
            self._t_ramp = now

            # Only slow down the first time.
            if not self._slowed_once:
                self._rpm = max(self._rpm * 0.5, 1.0)
                self._slowed_once = True
                _trace(f"search completed 360° — reversing and slowing to {self._rpm:.1f} rpm")
            else:
                _trace(f"search completed 360° — reversing at {self._rpm:.1f} rpm")

        # Stall recovery ALWAYS runs.
        if now - self._t_ramp >= KICK_TIMEOUT_S:
            if self._yaw - self._yaw_at_ramp < math.radians(5.0):
                self._rpm = min(self._rpm + KICK_RAMP_STEP, MAX_RPM)
                _trace(f"search stalled — ramping spin to {self._rpm:.0f} rpm")
            self._yaw_at_ramp = self._yaw
            self._t_ramp = now

        return (-self._dir * self._rpm, +self._dir * self._rpm)


# =============================================================================
# DEAD RECKONING + RETURN-HOME
# =============================================================================

def deslip(d_l, d_r, cmd_l_rpm, cmd_r_rpm, dt):
    """Reject single-wheel free-spin slip for odometry.

    A slipping drive wheel spins faster than the cart moves, so its encoder
    over-reports.  We know the *commanded* wheel RPMs, hence the intended L/R
    tick difference.  If the measured difference exceeds that by more than the
    tolerance, the wheel on the high side is free-spinning: trust the slower
    (gripping) wheel and rebuild the faster wheel from it plus the commanded
    differential, so odometry sees only the motion the cart actually made.
    Returns the corrected (d_l, d_r).
    """
    exp_diff  = (cmd_r_rpm - cmd_l_rpm) / 60.0 * dt * ENCODER_PPR   # intended d_r - d_l
    excess    = (d_r - d_l) - exp_diff                              # divergence beyond command
    if abs(excess) <= SLIP_DIFF_TICKS_PER_S * dt:
        return d_l, d_r                       # within tolerance — no slip
    if excess > 0:                            # right moved too much → right slipping
        return d_l, d_l + exp_diff            # trust left, rebuild right
    return d_r - exp_diff, d_r                # left slipping → trust right, rebuild left


class Odometry:
    """Dead-reckoned cart pose from encoder deltas.

    Origin (0, 0, heading 0) at construction = the start pose ('home'). x/y in
    meters, heading in radians (+CCW, 0 = the start facing direction). Uses the
    same encoder sign convention as ``_ticks_to_yaw`` so it stays consistent
    with the turn commands.
    """

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0

    def update(self, d_l, d_r):
        d_left   = d_l * M_PER_PULSE
        d_right  = d_r * M_PER_PULSE
        d_center = (d_left + d_right) / 2.0
        d_theta  = (d_right - d_left) / TRACK_M   # +CCW (matches _ticks_to_yaw)
        mid = self.heading + d_theta / 2.0        # midpoint integration
        self.x += d_center * math.cos(mid)
        self.y += d_center * math.sin(mid)
        self.heading += d_theta
        return d_theta


class ReturnHome:
    """Non-blocking dead-reckoned return to the start pose (0, 0, 0).

    Drives a *direct line* home in three phases: turn to face home → drive to
    home (steering to hold the bearing) → turn to the original heading.
    ``update(odom, …)`` returns wheel commands once per tick, or None when the
    start pose is reached. Closed-loop on the odometry, so it self-corrects.
    """

    def __init__(self):
        self._active = False

    @property
    def active(self):
        return self._active

    def cancel(self):
        self._active = False

    def start(self, odom):
        self._active = True
        self._phase = "turn_to_home"
        _trace(f"return-home from ({odom.x:.2f},{odom.y:.2f}) "
               f"hdg {math.degrees(odom.heading):.0f}°")

    def _point_turn(self, err):
        direction = 1.0 if err > 0 else -1.0       # +CCW
        return (-direction * HOME_TURN_RPM, +direction * HOME_TURN_RPM)

    def update(self, odom, d_l, d_r):
        if not self._active:
            return None
        dist = math.hypot(odom.x, odom.y)          # distance to home (0,0)
        bearing = math.atan2(-odom.y, -odom.x)     # direction toward home

        if self._phase == "turn_to_home":
            if dist < HOME_DIST_TOL_M:
                self._phase = "turn_to_orig"       # already at home position
            else:
                err = _wrap_pi(bearing - odom.heading)
                if abs(err) < math.radians(HOME_ANGLE_TOL_DEG):
                    self._phase = "drive"
                else:
                    return self._point_turn(err)

        if self._phase == "drive":
            if dist < HOME_DIST_TOL_M:
                self._phase = "turn_to_orig"
            else:
                err = _wrap_pi(bearing - odom.heading)
                steer = _clip(HOME_DRIVE_KP * err, -HOME_TURN_RPM, HOME_TURN_RPM)
                return (HOME_FWD_RPM - steer, HOME_FWD_RPM + steer)

        if self._phase == "turn_to_orig":
            err = _wrap_pi(0.0 - odom.heading)     # back to the start heading
            if abs(err) < math.radians(HOME_ANGLE_TOL_DEG):
                self._active = False
                _trace("return-home: arrived at start pose")
                return None
            return self._point_turn(err)

        return None


# =============================================================================
# FOLLOW CONTROLLER  (per-tick distance + steering)
# =============================================================================

class Kick:
    """Encoder-confirmed breakaway burst.

    Armed when an axis departs from rest against static friction; stays active
    until KICK_TICKS of encoder movement accumulate (proving it broke free).
    While active the caller floors the command magnitude at ``self.rpm``.  If
    still stalled after KICK_TIMEOUT_S, the kick RAMPS UP by KICK_RAMP_STEP
    (instead of giving up) and keeps trying, up to MAX_RPM.  One instance per
    axis: Kick(KICK_RPM) for forward departures, Kick(SPIN_KICK_RPM) for
    standing point turns.
    """

    def __init__(self, rpm=KICK_RPM):
        self._rpm0 = rpm
        self._rpm = rpm
        self._active = False
        self._ticks = 0
        self._t0 = 0.0

    @property
    def active(self):
        return self._active

    @property
    def rpm(self):
        """Current kick floor — starts at the configured RPM, ramps while stalled."""
        return self._rpm

    def arm(self):
        self._active = True
        self._ticks = 0
        self._t0 = time.monotonic()
        self._rpm = self._rpm0
        _trace(f"kick.arm()  floor at {self._rpm0:.0f}rpm until {KICK_TICKS} ticks")

    def cancel(self):
        self._active = False
        self._ticks = 0

    def update(self, d_l, d_r):
        """Feed this tick's encoder deltas; release once movement is confirmed
        (KICK_TICKS). If still stalled past KICK_TIMEOUT_S, ramp the kick up by
        KICK_RAMP_STEP (capped at MAX_RPM) and keep trying instead of giving up."""
        if not self._active:
            return
        self._ticks += abs(d_l) + abs(d_r)
        if self._ticks >= KICK_TICKS:
            self._active = False
            _trace(f"kick released: {self._ticks} ticks (breakaway confirmed)")
        elif (time.monotonic() - self._t0) >= KICK_TIMEOUT_S:
            self._rpm = min(self._rpm + KICK_RAMP_STEP, MAX_RPM)
            self._ticks = 0
            self._t0 = time.monotonic()
            _trace(f"kick ramp → {self._rpm:.0f}rpm (still stalled)")


class InertiaLearner:
    """Online coast-inertia estimator, one instance per axis (forward /
    yaw) — the generalisation of "calibrate by spin-and-coast" to every
    commanded stop, so no startup calibration run is needed.

    Feed it the axis's commanded effort, measured rate and per-tick measured
    displacement every tick.  When a sustained command drops to ~zero while
    the axis still has speed, it integrates how far the encoders coast until
    the rate settles; each event yields coast/rate = seconds of inertia,
    EMA-blended into ``self.value``.  The controllers use the value to ease
    off early (error -= rate * inertia).
    """

    ON_RPM    = 8.0      # demand at/above this counts as "driving the axis"
    OFF_RPM   = 2.0      # demand below this counts as "commanded stop"
    SETTLE_S  = 0.20     # rate must stay low this long = coast finished
    TIMEOUT_S = 1.5      # abandon the measurement (pushed by hand / re-commanded)
    ALPHA     = 0.3      # EMA weight of each new measurement
    CAP_S     = 1.0      # sanity cap (s)

    def __init__(self, name, min_rate, seed=0.0):
        self.value = seed
        self._name = name
        self._min_rate = min_rate   # ignore stops slower than this (noise dominates)
        self._was_on = False
        self._coasting = False

    def feed(self, demand, rate, d_disp, now):
        """Returns True when ``self.value`` was updated this tick."""
        updated = False
        if self._coasting:
            if demand >= self.OFF_RPM or (now - self._t0) > self.TIMEOUT_S:
                self._coasting = False            # re-commanded / timed out — discard
            else:
                self._disp += d_disp
                if abs(rate) >= 0.4 * self._min_rate:
                    self._calm_since = None
                elif self._calm_since is None:
                    self._calm_since = now
                elif now - self._calm_since >= self.SETTLE_S:
                    self._coasting = False        # settled — fold in the measurement
                    est = abs(self._disp) / self._rate0
                    if self.value <= 0.0:         # first sample seeds directly
                        self.value = min(est, self.CAP_S)
                    else:
                        self.value = min(self.CAP_S,
                                         (1.0 - self.ALPHA) * self.value
                                         + self.ALPHA * est)
                    _trace(f"{self._name}: coast {abs(self._disp):.4f} @ "
                           f"{self._rate0:.2f}/s → {self.value:.3f}s")
                    updated = True
        elif demand < self.OFF_RPM and self._was_on and abs(rate) >= self._min_rate:
            self._coasting = True                 # sustained drive just commanded to stop
            self._rate0 = abs(rate)
            self._disp = 0.0
            self._t0 = now
            self._calm_since = None
        self._was_on = demand >= self.ON_RPM
        return updated


class FollowController:
    """Live state for the follow loop.  ``step`` returns (S, rpm_diff) —
    forward wheel RPM and steering RPM-difference — from two continuous
    damped-P laws:

      * Distance: deadband + stop/restart hysteresis + slew limiting, with the
        standoff error shrunk by the predicted forward coast
        (v_fwd * LINEAR_INERTIA) so the cart rolls into the hold ring.
      * Steering: angle deadband + slew limiting, with the angle error shrunk
        by the predicted yaw coast (omega * ANGULAR_INERTIA) so turns ease off
        early instead of overshooting — centering is continuous, not a pulsed
        spin maneuver.

    The breakaway kicks live outside, in the ``Kick`` helpers."""

    def __init__(self):
        self.S = 0.0           # current forward wheel RPM (slew-limited state)
        self.diff = 0.0        # current steering RPM-difference (slew-limited state)
        self.dx = 0.0          # smoothed d(distance)/dt  (m/s)
        self._prev_x = None
        self._holding = False  # stopped at the ring; restart needs deadband+hysteresis

    def _smooth_dx(self, x, dt):
        raw_dx = 0.0 if self._prev_x is None else (x - self._prev_x) / dt
        self.dx = (1.0 - DX_ALPHA) * self.dx + DX_ALPHA * raw_dx
        self._prev_x = x

    def target_lost(self, motors=None):
        """Stop and reset state when no target."""
        self.S = 0.0
        self.diff = 0.0
        self.dx = 0.0
        self._prev_x = None
        self._holding = False
        if motors is not None:
            motors.send_rpm(0.0, 0.0)

    def coast_step(self, dt):
        """One tick of coasting down during the lost-target grace window: slew
        both axes toward 0 instead of brake-slamming on a one-frame flicker."""
        self.S = max(0.0, self.S - RPM_SLEW_DOWN_PER_S * dt)
        self.diff -= math.copysign(min(abs(self.diff), DIFF_SLEW_PER_S * dt), self.diff)
        self._prev_x = None
        return self.S

    def step(self, x, angle_deg, dt, v_fwd=0.0, omega=0.0):
        """One control tick. ``omega`` is the measured yaw rate (rad/s, +CCW),
        ``v_fwd`` the measured forward speed (m/s). Returns (S, rpm_diff)."""
        self._smooth_dx(x, dt)

        # ── steering: continuous damped P with yaw-coast compensation ────────
        theta = math.radians(STEER_SIGN * angle_deg)   # signed turn-toward-target error
        theta_eff = theta - omega * ANGULAR_INERTIA    # ease off by the predicted coast
        if abs(theta_eff) < math.radians(ANGLE_DEADBAND_DEG):
            target_diff = 0.0
        else:
            target_diff = _clip(KP_ANGLE * theta_eff, -MAX_RPM, MAX_RPM)
        self.diff = _clip(target_diff,
                          self.diff - DIFF_SLEW_PER_S * dt,
                          self.diff + DIFF_SLEW_PER_S * dt)

        # ── distance: damped P with deadband, hysteresis and coast comp ──────
        error = (x - THRESH_M) - max(v_fwd, 0.0) * LINEAR_INERTIA
        resume_at = DIST_DEADBAND_M + (RESUME_HYST_M if self._holding else 0.0)
        if error <= resume_at:
            self._holding = True
            target_s = 0.0
        else:
            self._holding = False
            target_s = KP_DIST * (error - DIST_DEADBAND_M) + KD_DIST * self.dx
            target_s = _clip(target_s, 0.0, MAX_RPM)   # forward-only: never reverse
            if 0.0 < target_s < MIN_DRIVE_RPM:
                target_s = MIN_DRIVE_RPM               # below this duty the cart stalls
        self.S = _clip(target_s,
                       self.S - RPM_SLEW_DOWN_PER_S * dt,
                       self.S + RPM_SLEW_UP_PER_S * dt)

        _trace(f"step: dist={x:.2f} err={error:+.2f} dx={self.dx:+.2f} → S={self.S:5.1f}   "
               f"ang={angle_deg:+5.1f}° eff={math.degrees(theta_eff):+5.1f}° → diff={self.diff:+6.1f}")
        return self.S, self.diff


# =============================================================================
# RUN  — wire vision + drive together
# =============================================================================

def _warmup_vision(cap, tracker, smooth_state, target_lock, Y, seconds, no_display):
    """Run the vision pipeline (no drive) for ``seconds`` so ByteTrack and the
    TargetLock color profile settle on the shopper before the cart moves."""
    print(f"Warming up camera + color tracking for {seconds:.0f}s — stand in view…")
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        ret, frame = cap.read()
        if not ret:
            break
        dets = cap.get_detections()
        tracked = tracker.update_with_detections(dets)
        out, _rows = Y.annotate_frame(frame, tracked, smooth_state, target_lock, time.monotonic())
        if not no_display:
            remaining = seconds - (time.monotonic() - t0)
            Y.cv2.putText(out, f"CALIBRATING color… {remaining:.0f}s", (10, 25),
                          Y.cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            Y.cv2.imshow("Cart View", out)
            if Y.cv2.waitKey(1) & 0xFF == ord("q"):
                break


def run(drive=True, no_display=False, countdown=3, duration=0.0, trace=False):
    global TRACE, ANGULAR_INERTIA, LINEAR_INERTIA
    TRACE = trace
    try:
        from vision import yolo_detect as Y
    except Exception as e:
        raise SystemExit(f"ERROR: could not import vision/yolo_detect.py: {e}")

    model_path = getattr(Y, "RPK_MODEL_PATH", DEFAULT_RPK_MODEL_PATH)
    if not model_path.exists():
        raise SystemExit(
            f"ERROR: {model_path} not found. Install imx500-models or copy the YOLO11n RPK there"
        )
    try:
        cap = Y.IMX500Capture(model_path=model_path, width=CAM_W, height=CAM_H, fps=CAM_FPS)
    except RuntimeError as e:
        raise SystemExit(f"ERROR: could not open IMX500 camera: {e}")

    motors = MotorIO(drive=drive)
    tracker = Y.sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=60,
        minimum_matching_threshold=0.8,
        frame_rate=CAM_FPS,
    )
    search_tracker = Y.sv.ByteTrack(
        track_activation_threshold=SEARCH_TRACK_ACTIVATION,
        lost_track_buffer=90,
        minimum_matching_threshold=SEARCH_TRACK_MATCHING,
        frame_rate=CAM_FPS,
    )
    smooth_state = {}
    target_lock = Y.TargetLock()
    ctrl = FollowController()
    odom = Odometry()
    returnhome = ReturnHome()
    searcher = Searcher()
    kick = Kick(KICK_RPM)
    spin_kick = Kick(SPIN_KICK_RPM)
    li_learner = InertiaLearner("linear inertia", min_rate=0.05, seed=LINEAR_INERTIA)
    ai_learner = InertiaLearner("angular inertia", min_rate=0.15, seed=ANGULAR_INERTIA)

    prev_S = 0.0
    lost_since = None
    last_angle_deg = 0.0
    unexpected_yaw = 0.0
    obstacle_blocked_until = 0.0
    v_fwd = 0.0                  # encoder-measured forward speed (m/s, EMA)
    omega_meas = 0.0             # encoder-measured yaw rate (rad/s, +CCW, EMA)
    stall_since = None           # commanded-but-not-moving watchdog timer

    print("follow.py — continuous damped-P distance + steering follow")
    print(f"Hold {THRESH_M:.2f}m ±{DIST_DEADBAND_M:.2f}  angle deadband ±{ANGLE_DEADBAND_DEG:.0f}°  "
          f"kick {KICK_RPM:.0f}/{SPIN_KICK_RPM:.0f} rpm  max {MAX_RPM:.0f} rpm")

    # Camera + color-tracking warmup before any motion.
    _warmup_vision(cap, tracker, smooth_state, target_lock, Y,
                   seconds=VISION_WARMUP_S, no_display=no_display)

    if motors.has_serial and countdown > 0:
        print(f"*** Following starts in {countdown}s — stand at the hold distance ***")
        for k in range(countdown, 0, -1):
            print(f"  {k}...")
            time.sleep(1.0)

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

            # --- encoders + dead reckoning + uncommanded-rotation detection ----
            d_l, d_r = motors.read_deltas()
            actual_l_rpm = _ticks_to_rpm(d_l, dt)
            actual_r_rpm = _ticks_to_rpm(d_r, dt)
            # Dead reckoning runs on slip-rejected deltas (cmd_*_rpm = last tick's command).
            od_l, od_r = deslip(d_l, d_r, cmd_l_rpm, cmd_r_rpm, dt)
            meas_yaw = odom.update(od_l, od_r)
            v_inst = ((od_l + od_r) / 2.0) * M_PER_PULSE / dt   # forward speed (m/s)
            v_fwd = (1.0 - V_FWD_ALPHA) * v_fwd + V_FWD_ALPHA * v_inst
            omega_inst = meas_yaw / dt                          # yaw rate (rad/s, +CCW)
            omega_meas = (1.0 - OMEGA_ALPHA) * omega_meas + OMEGA_ALPHA * omega_inst
            # Uncommanded rotation = measured yaw minus the yaw we actually commanded
            # last tick. A manual spin adds to the measured yaw but not the commanded,
            # so it accumulates whether the cart is following, stopped, or searching.
            cmd_yaw = (cmd_r_rpm - cmd_l_rpm) * WHEEL_CIRC / 60.0 / TRACK_M * dt
            unexpected_yaw += meas_yaw - cmd_yaw
            spun_around = (abs(abs(unexpected_yaw) - math.radians(HOME_ROT_TRIGGER_DEG))
                           <= math.radians(HOME_ROT_RANGE_DEG))

            search_vision = searcher.active
            confidence_threshold = (
                getattr(Y, "SEARCH_PERSON_CONFIDENCE", SEARCH_PERSON_CONFIDENCE)
                if search_vision else None
            )
            dets = cap.get_detections(confidence_threshold=confidence_threshold)
            active_tracker = search_tracker if search_vision else tracker
            tracked = active_tracker.update_with_detections(dets)
            out, rows = Y.annotate_frame(
                frame, tracked, smooth_state, target_lock, now, draw=not no_display
            )
            target_row = next((r for r in rows if r[0] == "TARGET"), None)

            if returnhome.active or spun_around:
                # Highest priority: an uncommanded ~180° spin → dead-reckon home.
                if not returnhome.active:
                    returnhome.start(odom)
                    unexpected_yaw = 0.0
                ctrl.target_lost()
                kick.cancel(); spin_kick.cancel()
                searcher.cancel(); prev_S = 0.0
                x = angle_deg = None
                cmd = returnhome.update(odom, d_l, d_r)
                if cmd is None:             # back at the start pose
                    returnhome.cancel()
                    unexpected_yaw = 0.0    # clear so it doesn't immediately re-trigger
                    l_rpm = r_rpm = 0.0
                    mode_str = "HOME"
                else:
                    l_rpm, r_rpm = cmd
                    mode_str = "RETURN"
            elif target_row is None:
                # Target lost. During the grace window coast down (slewed)
                # instead of brake-slamming — a one-frame lock flicker no
                # longer jerks the cart. After the grace, spin-search toward
                # the side it was last seen; the Searcher never gives up and
                # is cancelled the instant a target reappears.
                kick.cancel(); spin_kick.cancel()
                x = angle_deg = None
                if lost_since is None:
                    lost_since = now

                if (now - lost_since) < SEARCH_GRACE_S and not searcher.active:
                    S = ctrl.coast_step(dt)   # decay toward 0, hold the line
                    prev_S = S
                    l_rpm, r_rpm = mix_wheels(S, ctrl.diff)
                    mode_str = "LOST"
                else:
                    ctrl.target_lost()        # reset forward state (search drives)
                    prev_S = 0.0
                    if not searcher.active:
                        searcher.start(STEER_SIGN * last_angle_deg)
                    l_rpm, r_rpm = searcher.update(d_l, d_r)
                    mode_str = "SEARCH"
            else:
                x = target_row[3]          # calibrated distance (m)
                angle_deg = target_row[4]  # calibrated angle (deg)
                last_angle_deg = angle_deg # remember which way it went, for search
                lost_since = None          # reacquired
                searcher.cancel()

                # Basic obstacle avoidance: an obstacle closer than the target and
                # roughly in our forward path blocks advancing toward the target.
                obstacle_in_path = any(
                    obs[3] < x - OBSTACLE_MARGIN_M and abs(obs[4]) <= OBSTACLE_BLOCK_DEG
                    for obs in rows if obs[0] == "OBSTACLE"
                )
                if obstacle_in_path:                       # refresh the hold each sighting
                    obstacle_blocked_until = now + OBSTACLE_HOLD_S
                blocked = now < obstacle_blocked_until     # hysteresis: ride through dropouts

                S, rpm_diff = ctrl.step(x, angle_deg, dt, v_fwd, omega_meas)

                if blocked:                # obstacle in the way — hold, keep aiming
                    S = 0.0
                    ctrl.S = 0.0
                    kick.cancel()
                    mode_str = "BLOCKED"
                elif S == 0.0:             # at the hold ring — steering may still aim
                    kick.cancel()
                    mode_str = "TURN" if abs(rpm_diff) >= 1.0 else "STOP"
                else:
                    if prev_S == 0.0:      # departing from rest
                        kick.arm()
                    kick.update(d_l, d_r)
                    if kick.active:        # floor the speed at the (ramping) kick level
                        S = math.copysign(max(abs(S), kick.rpm), S)
                        ctrl.S = S
                    mode_str = "KICK" if kick.active else "FOLLOW"

                # Point-turn breakaway: a standing re-center fights static
                # friction the forward kick never sees. The stall watch below
                # arms spin_kick; while active, floor the steering difference
                # until the encoders confirm rotation.
                spin_kick.update(d_l, d_r)
                if S != 0.0 or rpm_diff == 0.0:
                    spin_kick.cancel()
                elif spin_kick.active:
                    rpm_diff = math.copysign(max(abs(rpm_diff), spin_kick.rpm), rpm_diff)

                # Stall watch: effort commanded but the encoders are silent →
                # arm the matching breakaway kick. Steering-only stalls need
                # TURN_STALL_MIN_RPM of demand, so angle noise inside the
                # deadband can never trigger a re-center pulse.
                moving = (abs(d_l) + abs(d_r)) >= 2
                demanded = S > 0.0 or abs(rpm_diff) >= TURN_STALL_MIN_RPM
                if moving or not demanded or kick.active or spin_kick.active:
                    stall_since = None
                elif stall_since is None:
                    stall_since = now
                elif now - stall_since >= KICK_TIMEOUT_S:
                    (kick if S > 0.0 else spin_kick).arm()
                    stall_since = None

                prev_S = S
                l_rpm, r_rpm = mix_wheels(S, rpm_diff)

            cmd_l_rpm, cmd_r_rpm = l_rpm, r_rpm
            motors.send_rpm(l_rpm, r_rpm)
            if x is None:
                stall_since = None

            # Online inertia learning: every commanded stop of an axis measures
            # its coast (displacement ÷ rate at the stop) and refines that
            # axis's inertia constant — forward and yaw alike. The linear axis
            # also counts steering as demand, so a forward→point-turn handoff
            # (near-zero net displacement) can't fake a tiny coast sample; a
            # yaw coast while rolling straight ahead is still a valid sample.
            fwd_demand = max(0.0, (cmd_l_rpm + cmd_r_rpm) / 2.0)
            yaw_demand = abs(cmd_r_rpm - cmd_l_rpm) / 2.0
            if li_learner.feed(max(fwd_demand, yaw_demand), v_fwd,
                               ((od_l + od_r) / 2.0) * M_PER_PULSE, now):
                LINEAR_INERTIA = li_learner.value
            if ai_learner.feed(yaw_demand, omega_meas, meas_yaw, now):
                ANGULAR_INERTIA = ai_learner.value

            if t - last_log >= 0.5:
                last_log = t
                if x is None:    # lost / search / return-home (no live distance)
                    extra = (f"  (toward last-seen {'+L' if STEER_SIGN*last_angle_deg >= 0 else '-R'})"
                             if mode_str == "SEARCH" else "")
                    print(f"t={t:5.1f}s  {mode_str:6s}  cmd L{cmd_l_rpm:+5.1f} R{cmd_r_rpm:+5.1f}rpm{extra}")
                else:
                    print(f"t={t:5.1f}s  dist={x:.2f}m  angle={angle_deg:+5.1f}°  "
                          f"dx={ctrl.dx:+.2f}m/s  {mode_str:6s}  "
                          f"cmd L{cmd_l_rpm:+5.1f} R{cmd_r_rpm:+5.1f}rpm  "
                          f"actual L{actual_l_rpm:+5.1f} R{actual_r_rpm:+5.1f}rpm  "
                          f"AI={ANGULAR_INERTIA:.2f} LI={LINEAR_INERTIA:.2f}")

            if not no_display:
                dist_str = f"dist={x:.2f}m ang={angle_deg:+.0f}°" if x is not None else "no target"
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
