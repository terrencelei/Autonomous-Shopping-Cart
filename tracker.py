"""
tracker.py — Single-target following controller
------------------------------------------------
Runs on a Raspberry Pi connected to an ESP32 over USB serial.

ESP32 → Pi  (inbound):  "E,<left_ticks>,<right_ticks>\\n"
Pi → ESP32  (outbound): "L<left_rpm> R<right_rpm>\\n"
Vision → Pi (UDP):      "<distance_m>,<angle_deg>\\n"
  angle_deg: 0 = centred, positive = target to the right

Behaviour:
  - Target not visible  → stop
  - Target visible      → centre on target, hold at BUBBLE_RAD using
                          delta-speed PD control
"""

import math
import logging
import socket
import threading
import time

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("tracker")

# =============================================================================
# CONFIGURE
# =============================================================================

# Hardware
WHEEL_DIAMETER_M  = 0.06778
TRACK_M           = 0.333
GEAR_RATIO        = 5
ENCODER_PPR       = 447
LEFT_ENC_SIGN     = +1
RIGHT_ENC_SIGN    = -1
LEFT_MOTOR_SIGN   = +1
RIGHT_MOTOR_SIGN  = -1

MOTOR_PORT  = "/dev/ttyUSB0"
MOTOR_BAUD  = 115200
UDP_PORT    = 5005
UDP_STALE_S = 0.5

DT = 0.02    # control loop period (seconds)

# Tracking
BUBBLE_RAD   = 2.0    # desired hold distance (m)
SPEED_THRESH = 0.05   # target speed below which we stop if inside bubble (m/s)
ANGLE_THRESH = 25.0   # degrees — suppress forward motion if off-centre beyond this
MAX_SPEED    = 0.5    # m/s
MAX_TURN     = math.radians(90)   # rad/s

CENTER_KP    = 0.006  # rad/s per degree of angle error
DIST_KP      = 0.4    # (m/s)/m   — proportional gain on distance error
DIST_KD      = 0.3    # (m/s)/(m/s) — derivative gain on relative speed

# =============================================================================
# DERIVED
# =============================================================================

WHEEL_CIRC  = math.pi * WHEEL_DIAMETER_M
M_PER_PULSE = WHEEL_CIRC / ENCODER_PPR

# =============================================================================
# HARDWARE
# =============================================================================

class MotorDriver:
    """
    Bidirectional ESP32 link.
    Outbound: "L<rpm> R<rpm>\\n"
    Inbound:  "E,<left_ticks>,<right_ticks>\\n"  (signed, cumulative)
    """

    def __init__(self):
        self._ser    = None
        self._last_l = self._last_r = None
        self._dl = self._dr = 0
        try:
            import serial
            self._ser = serial.Serial(MOTOR_PORT, MOTOR_BAUD, timeout=0.01)
            log.info(f"ESP32 on {MOTOR_PORT}")
        except Exception as e:
            log.warning(f"ESP32 unavailable ({e}) — logging only")

    def read_encoder_deltas(self):
        """Return signed pulse deltas (left, right) since last call."""
        if not (self._ser and self._ser.is_open):
            return 0, 0
        while self._ser.in_waiting:
            try:
                line = self._ser.readline().decode(errors="ignore").strip()
            except Exception:
                break
            if not line.startswith("E,"):
                continue
            try:
                _, ls, rs = line.split(",", 2)
                lc = LEFT_ENC_SIGN  * int(ls)
                rc = RIGHT_ENC_SIGN * int(rs)
                if self._last_l is not None:
                    self._dl += lc - self._last_l
                    self._dr += rc - self._last_r
                self._last_l, self._last_r = lc, rc
            except ValueError:
                pass
        dl, dr       = self._dl, self._dr
        self._dl = self._dr = 0
        return dl, dr

    def send(self, vl, vr):
        cmd = f"L{LEFT_MOTOR_SIGN  * _rpm(vl):.1f} " \
              f"R{RIGHT_MOTOR_SIGN * _rpm(vr):.1f}\n".encode()
        if self._ser and self._ser.is_open:
            self._ser.write(cmd)
        else:
            log.debug(f"MOTOR: {cmd.decode().strip()}")

    def stop(self):
        self.send(0.0, 0.0)


class Odometry:
    """Midpoint-heading dead-reckoning from encoder deltas."""

    def __init__(self, encoder_reader):
        self._read   = encoder_reader
        self.heading = 0.0

    def update(self):
        dl, dr   = self._read()
        d_left   = dl * M_PER_PULSE
        d_right  = dr * M_PER_PULSE
        d_theta  = (d_right - d_left) / TRACK_M
        self.heading = (self.heading + d_theta) % (2 * math.pi)
        return self.heading


class TargetReceiver:
    """
    UDP listener for vision packets: "<distance_m>,<angle_deg>\\n"
    Returns (dist_m, angle_deg) or None when stale / never received.
    angle_deg: 0 = centred, positive = target to the right.
    """

    def __init__(self):
        self._data = None
        self._ts   = 0.0
        self._lock = threading.Lock()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("", UDP_PORT))
        sock.settimeout(0.02)
        threading.Thread(target=self._loop, args=(sock,), daemon=True).start()
        log.info(f"Vision on UDP:{UDP_PORT}")

    def get(self):
        with self._lock:
            if self._data and time.monotonic() - self._ts < UDP_STALE_S:
                return self._data
        return None

    def _loop(self, sock):
        while True:
            try:
                d, _ = sock.recvfrom(64)
                dist, ang = map(float, d.decode().strip().split(","))
                with self._lock:
                    self._data = (dist, ang)
                    self._ts   = time.monotonic()
            except socket.timeout:
                pass
            except Exception as e:
                log.warning(f"Vision: {e}")

# =============================================================================
# UTILITIES
# =============================================================================

def _rpm(v):
    return v / WHEEL_CIRC * 60.0

def _wheel_commands(v_fwd, omega):
    """
    Differential drive mixing → (v_left, v_right) in m/s.
    Scales both down proportionally if either exceeds MAX_SPEED,
    preserving the turn radius.
    """
    half = TRACK_M / 2.0
    vl   = v_fwd - omega * half
    vr   = v_fwd + omega * half
    peak = max(abs(vl), abs(vr))
    if peak > MAX_SPEED:
        vl *= MAX_SPEED / peak
        vr *= MAX_SPEED / peak
    return vl, vr

# =============================================================================
# CONTROLLER STATE
# =============================================================================

cart_speed = 0.0   # persistent forward speed (m/s), updated by delta-speed PD
prev_dist  = None  # distance from previous tick, used for derivative

# =============================================================================
# TICK
# =============================================================================

def tick(reading):
    """
    One control cycle.

    reading : (dist_m, angle_deg) or None
    Returns : (v_left, v_right) in m/s

    Logic
    -----
    Not in view
        → stop immediately, reset speed state

    In view
        Centering (always running):
            omega = -CENTER_KP * angle_deg
            clamped to ±MAX_TURN

        Target speed estimate:
            rel_speed   = (dist_m - prev_dist) / DT   (+ = opening gap)
            target_speed = cart_speed + rel_speed       (absolute target speed)

        Stop condition — inside bubble AND target stationary:
            if dist_m < BUBBLE_RAD and abs(target_speed) < SPEED_THRESH
                cart_speed = 0

        Drive condition — everything else:
            delta_speed = DIST_KP * (dist_m - BUBBLE_RAD) + DIST_KD * rel_speed
            cart_speed  = clip(cart_speed + delta_speed, -MAX_SPEED, +MAX_SPEED)

        Angle gate — too far off-centre to drive safely:
            if abs(angle_deg) > ANGLE_THRESH
                cart_speed = 0     (omega still runs, robot centres first)

        Wheel mix:
            vl, vr = cart_speed ± omega * (track/2)
            scale down if peak > MAX_SPEED
    """
    global cart_speed, prev_dist

    # ------------------------------------------------------------------
    # Not in view — full stop, reset state
    # ------------------------------------------------------------------
    if reading is None:
        cart_speed = 0.0
        prev_dist  = None
        return _wheel_commands(0.0, 0.0)

    dist_m, angle_deg = reading

    # ------------------------------------------------------------------
    # Centering — always active when target is visible
    # ------------------------------------------------------------------
    omega = -CENTER_KP * angle_deg
    omega = math.copysign(min(abs(omega), MAX_TURN), omega)

    # ------------------------------------------------------------------
    # Relative and target speed estimates
    # ------------------------------------------------------------------
    rel_speed    = (dist_m - prev_dist) / DT if prev_dist is not None else 0.0
    target_speed = cart_speed + rel_speed
    prev_dist    = dist_m

    # ------------------------------------------------------------------
    # Stop condition: inside bubble, target not moving
    # ------------------------------------------------------------------
    if dist_m < BUBBLE_RAD and abs(target_speed) < SPEED_THRESH:
        cart_speed = 0.0

    # ------------------------------------------------------------------
    # Drive condition: delta-speed PD
    # ------------------------------------------------------------------
    else:
        delta_speed = DIST_KP * (dist_m - BUBBLE_RAD) + DIST_KD * rel_speed
        cart_speed  = cart_speed + delta_speed
        cart_speed  = max(-MAX_SPEED, min(MAX_SPEED, cart_speed))

    # ------------------------------------------------------------------
    # Angle gate: suppress forward motion if too far off-centre
    # Centering (omega) keeps running so robot re-centres before moving
    # ------------------------------------------------------------------
    if abs(angle_deg) > ANGLE_THRESH:
        cart_speed = 0.0

    # ------------------------------------------------------------------
    # Wheel mixing
    # ------------------------------------------------------------------
    vl, vr = _wheel_commands(cart_speed, omega)

    log.debug(
        f"dist={dist_m:.2f}m  ang={angle_deg:+.1f}°  "
        f"rel={rel_speed:+.3f}  tgt={target_speed:+.3f}  "
        f"cart={cart_speed:+.3f}  vL={vl:+.3f}  vR={vr:+.3f}"
    )

    return vl, vr

# =============================================================================
# MAIN
# =============================================================================

def main():
    motors   = MotorDriver()
    odom     = Odometry(motors.read_encoder_deltas)
    receiver = TargetReceiver()

    log.info(f"Wheel {WHEEL_DIAMETER_M*1000:.1f}mm  track {TRACK_M*1000:.1f}mm  "
             f"PPR={ENCODER_PPR}  m/pulse={M_PER_PULSE*1000:.3f}mm")
    log.info(f"Bubble {BUBBLE_RAD}m  speed_thresh {SPEED_THRESH}m/s  "
             f"angle_thresh {ANGLE_THRESH}°  Kp={DIST_KP}  Kd={DIST_KD}")

    try:
        while True:
            t0      = time.monotonic()
            odom.update()                  # keeps heading fresh for future use
            reading = receiver.get()
            vl, vr  = tick(reading)
            motors.send(vl, vr)
            spare = DT - (time.monotonic() - t0)
            if spare > 0:
                time.sleep(spare)
            elif spare < -0.005:
                log.warning(f"Overrun {-spare*1000:.1f}ms")
    except KeyboardInterrupt:
        log.info("Stopped.")
    finally:
        motors.stop()


if __name__ == "__main__":
    main()
