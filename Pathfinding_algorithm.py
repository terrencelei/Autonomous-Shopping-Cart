import time
import math
import heapq
import threading
import socket
import logging

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("robot")

# =============================================================================
# PHYSICAL PARAMETERS
# =============================================================================

INCH_TO_M         = 0.0254

WHEEL_DIAMETER_IN  = 3.0
WHEEL_TRACK_IN     = 13.0
WHEEL_DIAMETER_M   = WHEEL_DIAMETER_IN * INCH_TO_M
WHEEL_TRACK_M      = WHEEL_TRACK_IN    * INCH_TO_M   # inside-to-inside; replace
                                                       # with centre-to-centre when
                                                       # wheel width is measured

GEAR_RATIO         = 5         # ← UPDATE from gearbox sticker
ENCODER_PPR        = 11 * 4 * GEAR_RATIO

LEFT_ENC_SIGN      = -1
RIGHT_ENC_SIGN     = +1

MOTOR_UART_PORT    = "/dev/ttyACM0"
MOTOR_UART_BAUD    = 115200
TARGET_UDP_PORT    = 5005

START_POS          = [0.0, 0.0]
START_HEADING      = 0.0

# =============================================================================
# DERIVED WHEEL CONSTANTS
# =============================================================================

WHEEL_CIRCUMFERENCE_M            = math.pi * WHEEL_DIAMETER_M
METRES_PER_PULSE                 = WHEEL_CIRCUMFERENCE_M / ENCODER_PPR
WHEEL_ROTATIONS_PER_ROBOT_DEGREE = WHEEL_TRACK_IN / (360.0 * WHEEL_DIAMETER_IN)
WHEEL_ROTATIONS_PER_90_DEGREES   = WHEEL_ROTATIONS_PER_ROBOT_DEGREE * 90.0
PULSES_PER_ROBOT_DEGREE          = WHEEL_ROTATIONS_PER_ROBOT_DEGREE * ENCODER_PPR
RADIANS_PER_COUNTERROTATION_PULSE = (2.0 * METRES_PER_PULSE) / WHEEL_TRACK_M

# =============================================================================
# MAP SETTINGS
# =============================================================================

chunk_size   = 0.1
map_size     = [5, 5]
FREE         = 1
BLOCKED      = 0
aisle_width  = 1.0
aisle_amount = 3
PEEK_SWEEP_RAD = math.radians(90.0)

cols = int(map_size[0] / chunk_size)
rows = int(map_size[1] / chunk_size)
gap_width = (map_size[1] - aisle_width * aisle_amount) / (aisle_amount - 1)

chunk_map = np.zeros((rows, cols), dtype=np.int8)
for _r in range(rows):
    for _c in range(cols):
        _x = _c * chunk_size
        _y = _r * chunk_size
        if _x < aisle_width:
            chunk_map[_r, _c] = FREE
        elif _x > map_size[0] - aisle_width:
            chunk_map[_r, _c] = FREE
        elif _y < aisle_width:
            chunk_map[_r, _c] = FREE
        elif _y > map_size[1] - aisle_width:
            chunk_map[_r, _c] = FREE
        elif (_y % (aisle_width + gap_width)) < aisle_width:
            chunk_map[_r, _c] = FREE

_period = aisle_width + gap_width
AISLE_Y_CENTRES = []
_probe = 0.0
while _probe <= map_size[1]:
    if _probe < aisle_width:
        AISLE_Y_CENTRES.append(round(aisle_width / 2.0, 6))
    elif _probe > map_size[1] - aisle_width:
        cy = map_size[1] - aisle_width / 2.0
        if round(cy, 6) not in AISLE_Y_CENTRES:
            AISLE_Y_CENTRES.append(round(cy, 6))
    else:
        phase = _probe % _period
        if phase < aisle_width:
            cy = _probe - phase + aisle_width / 2.0
            if round(cy, 6) not in AISLE_Y_CENTRES:
                AISLE_Y_CENTRES.append(round(cy, 6))
    _probe += chunk_size
AISLE_Y_CENTRES = sorted(set(AISLE_Y_CENTRES))

# =============================================================================
# CONTROLLER PARAMETERS
# =============================================================================

ROBOT_SPEED_MPS  = 0.5               # approach speed (full speed phase)
TURN_SPEED_RAD   = math.radians(90.0)
DT               = 0.1
ARRIVE_THRESH    = 1.5
ALIGN_THRESH_RAD = math.radians(5.0)

# -----------------------------------------------------------------------------
# TARGET TRACKING PARAMETERS
# -----------------------------------------------------------------------------

# Distance at which the robot switches from full-speed approach to PD hold
HOLD_DIST_M      = 2.0               # metres

# Centering: proportional gain on camera angle error (rad/s per degree of error)
# Tune this first. Start low (~0.01) and increase until centering is snappy
# without oscillation. At 0.02: a 10° offset produces 0.2 rad/s turn.
CENTER_KP        = 0.02              # rad/s per degree of angle error

# PD gains for the hold-distance controller.
# The error signal is:  e = dist_measured - HOLD_DIST_M   (metres)
# Positive e → too far  → move forward
# Negative e → too close → move backward
#
# Kp: proportional gain (m/s per metre of error)
#   At Kp=0.4, a 0.5 m error produces 0.2 m/s command.
#   Increase if the robot is sluggish to correct; decrease if it oscillates.
#
# Kd: derivative gain (m/s per m/s of range rate)
#   Kd dampens overshoot. de/dt = (dist_now - dist_prev) / DT is the
#   measured range rate (positive = target moving away).
#   At Kd=0.3, approaching at 0.5 m/s produces a 0.15 m/s braking correction.
#   Increase if robot overshoots and oscillates; decrease if response is sluggish.
#
# Max hold speed caps the PD output so the robot never rushes at the target.
HOLD_KP          = 0.4               # m/s per metre of distance error
HOLD_KD          = 0.3               # m/s per m/s of range rate
HOLD_MAX_SPEED   = ROBOT_SPEED_MPS   # clip PD output to this magnitude

# =============================================================================
# STATE LABELS
# =============================================================================

S_IN_VIEW        = "IN_VIEW"
S_FOLLOW_GOAL    = "FOLLOW_GOAL"
S_SPIN           = "SPIN"
S_EDGE_PATROL    = "EDGE_PATROL"
S_EDGE_PEEK      = "EDGE_PEEK"
S_AISLE_TRAVERSE = "AISLE_TRAVERSE"

# Tracking sub-states (only active inside S_IN_VIEW)
TRACK_APPROACH   = "APPROACH"        # dist > HOLD_DIST → full speed
TRACK_HOLD       = "HOLD"            # dist <= HOLD_DIST → PD control

# =============================================================================
# GEOMETRY & MAP HELPERS
# =============================================================================

def world_to_chunk(pos):
    col = int(pos[0] / chunk_size)
    row = int(pos[1] / chunk_size)
    return row, col

def chunk_to_world(chunk):
    row, col = chunk
    return [col * chunk_size + chunk_size / 2,
            row * chunk_size + chunk_size / 2]

def is_valid_chunk(chunk):
    row, col = chunk
    if row < 0 or row >= rows: return False
    if col < 0 or col >= cols: return False
    return chunk_map[row, col] == FREE

def distance_between(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])

def angle_diff(a, b):
    return ((a - b) + math.pi) % (2 * math.pi) - math.pi

def bearing_to(frm, to):
    return math.atan2(to[1] - frm[1], to[0] - frm[0])

def is_on_edge(pos):
    return pos[0] < aisle_width or pos[0] > map_size[0] - aisle_width

def is_in_aisle(pos):
    if is_on_edge(pos):
        return False
    y = pos[1]
    if y < aisle_width or y > map_size[1] - aisle_width:
        return True
    return (y % (aisle_width + gap_width)) < aisle_width

def aisle_index_of(pos):
    y = pos[1]
    for i, cy in enumerate(AISLE_Y_CENTRES):
        if abs(y - cy) <= aisle_width / 2.0 + 0.05:
            return i
    return -1

def rotate_step(heading, direction, dt):
    return (heading + direction * TURN_SPEED_RAD * dt) % (2 * math.pi)

def rotate_toward(heading, target_h, dt):
    diff = angle_diff(target_h, heading)
    step = math.copysign(min(abs(diff), TURN_SPEED_RAD * dt), diff)
    return (heading + step) % (2 * math.pi)

def inward_heading(pos):
    return 0.0 if pos[0] < map_size[0] / 2.0 else math.pi

def edge_x_centre(pos):
    return aisle_width / 2.0 if pos[0] < map_size[0] / 2.0 \
           else map_size[0] - aisle_width / 2.0

def peek_sweep_bounds(edge_pos):
    perp       = inward_heading(edge_pos)
    half_sweep = math.atan2(aisle_width / 2.0, aisle_width) + PEEK_SWEEP_RAD / 2.0
    half_sweep = min(half_sweep, math.pi / 2.0)
    return perp - half_sweep, perp + half_sweep

def next_unchecked_aisle(robot_y, patrol_dir, aisles_checked):
    candidates = []
    for i, cy in enumerate(AISLE_Y_CENTRES):
        if aisles_checked[i]:
            continue
        if patrol_dir > 0 and cy >= robot_y - 0.05:
            candidates.append((cy, i))
        elif patrol_dir < 0 and cy <= robot_y + 0.05:
            candidates.append((cy, i))
    if not candidates:
        return None, None
    return (min if patrol_dir > 0 else max)(candidates, key=lambda t: t[0])

# =============================================================================
# A* PATHFINDING
# =============================================================================

def direct_path_clear(start_pos, end_pos):
    sx, sy = start_pos
    ex, ey = end_pos
    dist  = math.hypot(ex - sx, ey - sy)
    steps = max(int(dist / chunk_size), 1)
    for i in range(steps + 1):
        t = i / steps
        pt = [sx + t * (ex - sx), sy + t * (ey - sy)]
        if not is_valid_chunk(world_to_chunk(pt)):
            return False
    return True

def astar(start_chunk, goal_chunk):
    open_set = []
    heapq.heappush(open_set, (0, start_chunk))
    came_from = {}
    g = {start_chunk: 0}
    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal_chunk:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nb = (cur[0]+dr, cur[1]+dc)
            if not is_valid_chunk(nb):
                continue
            ng = g[cur] + 1
            if nb not in g or ng < g[nb]:
                came_from[nb] = cur
                g[nb] = ng
                heapq.heappush(open_set,
                    (ng + abs(nb[0]-goal_chunk[0]) + abs(nb[1]-goal_chunk[1]), nb))
    return None

def build_route(start_pos, goal_pos):
    sc = world_to_chunk(start_pos)
    gc = world_to_chunk(goal_pos)
    if not is_valid_chunk(sc) or not is_valid_chunk(gc):
        return [start_pos]
    if direct_path_clear(start_pos, goal_pos):
        return [start_pos, goal_pos]
    path = astar(sc, gc)
    if path is None:
        return [start_pos]
    return [chunk_to_world(c) for c in path]

def move_along_route(pos, route, speed, dt):
    if len(route) < 2:
        return pos, route
    cx, cy = pos
    budget = speed * dt
    route  = list(route)
    while len(route) >= 2 and budget > 0:
        nx, ny = route[1]
        dx, dy = nx - cx, ny - cy
        d = math.hypot(dx, dy)
        if d < 1e-4:
            route.pop(0); continue
        if budget >= d:
            cx, cy = nx, ny; budget -= d; route.pop(0)
        else:
            cx += (dx / d) * budget
            cy += (dy / d) * budget
            budget = 0
    return [cx, cy], route

# =============================================================================
# MOTOR COMMAND CALCULATION
# =============================================================================

def velocities_to_wheel_commands(v_forward, omega):
    """
    Body-frame → wheel surface speeds (m/s).
      v_left  = v_forward - omega * (track / 2)
      v_right = v_forward + omega * (track / 2)
    """
    half_track = WHEEL_TRACK_M / 2.0
    v_left  = v_forward - omega * half_track
    v_right = v_forward + omega * half_track
    return v_left, v_right

def wheel_speed_to_rpm(v_wheel):
    return (v_wheel / WHEEL_CIRCUMFERENCE_M) * 60.0

# =============================================================================
# ODOMETRY
# =============================================================================

class Odometry:
    """
    Dead-reckoning from signed wheel encoder counts streamed by the ESP32.
    Uses midpoint-heading integration for better arc accuracy.
    """

    def __init__(self, start_pos, start_heading, encoder_reader=None):
        self.x       = start_pos[0]
        self.y       = start_pos[1]
        self.heading = start_heading
        self._encoder_reader = encoder_reader

    def update(self, dt):
        delta_left, delta_right = self._read_encoder_deltas()

        d_left   = delta_left  * METRES_PER_PULSE
        d_right  = delta_right * METRES_PER_PULSE
        d_centre = (d_left + d_right) / 2.0
        d_theta  = (d_right - d_left) / WHEEL_TRACK_M

        mid_heading  = self.heading + d_theta / 2.0
        self.x      += d_centre * math.cos(mid_heading)
        self.y      += d_centre * math.sin(mid_heading)
        self.heading = (self.heading + d_theta) % (2 * math.pi)

        return [self.x, self.y], self.heading

    def _read_encoder_deltas(self):
        if self._encoder_reader is None:
            return 0, 0
        return self._encoder_reader()


ENCODER_COUNTS_ARE_CUMULATIVE = True

# =============================================================================
# MOTOR DRIVER
# =============================================================================

class MotorDriver:
    """
    Shares one serial port with the ESP32 for both motor commands (out) and
    encoder counts (in).

    Outbound: "L<rpm> R<rpm>\\n"
    Inbound:  "E,<left_count>,<right_count>\\n"
    """

    def __init__(self, port, baud):
        self._port = port
        self._baud = baud
        self._ser  = None
        self._last_left_count  = None
        self._last_right_count = None
        self._pending_left_delta  = 0
        self._pending_right_delta = 0
        self._connect()

    def _connect(self):
        try:
            import serial
            self._ser = serial.Serial(self._port, self._baud, timeout=0.01)
            log.info(f"ESP32 link on {self._port} @ {self._baud}")
        except Exception as e:
            log.warning(f"ESP32 not available ({e}) — logging only")
            self._ser = None

    def _poll_serial(self):
        if not (self._ser and self._ser.is_open):
            return
        while self._ser.in_waiting:
            try:
                line = self._ser.readline().decode(errors="ignore").strip()
            except Exception as e:
                log.warning(f"Serial read error: {e}")
                return
            if not line or not line.startswith("E,"):
                log.debug(f"ESP32: {line}")
                continue
            try:
                _, left_s, right_s = line.split(",", 2)
                left_count  = LEFT_ENC_SIGN  * int(left_s)
                right_count = RIGHT_ENC_SIGN * int(right_s)
            except ValueError:
                log.warning(f"Bad encoder packet: {line}")
                continue

            if ENCODER_COUNTS_ARE_CUMULATIVE:
                if self._last_left_count is None:
                    delta_left = delta_right = 0
                else:
                    delta_left  = left_count  - self._last_left_count
                    delta_right = right_count - self._last_right_count
                self._last_left_count  = left_count
                self._last_right_count = right_count
            else:
                delta_left  = left_count
                delta_right = right_count

            self._pending_left_delta  += delta_left
            self._pending_right_delta += delta_right

    def read_encoder_deltas(self):
        self._poll_serial()
        dl = self._pending_left_delta
        dr = self._pending_right_delta
        self._pending_left_delta  = 0
        self._pending_right_delta = 0
        return dl, dr

    def send_velocities(self, v_left_ms, v_right_ms):
        rpm_l = wheel_speed_to_rpm(v_left_ms)
        rpm_r = wheel_speed_to_rpm(v_right_ms)
        cmd   = f"L{rpm_l:.1f} R{rpm_r:.1f}\n".encode()
        if self._ser and self._ser.is_open:
            self._ser.write(cmd)
        else:
            log.debug(f"MOTOR: {cmd.decode().strip()}")

    def stop(self):
        self.send_velocities(0.0, 0.0)

# =============================================================================
# TARGET RECEIVER
# =============================================================================

class TargetReceiver:
    """
    Listens on UDP for "<distance_m>,<angle_deg>\\n" packets.
    angle_deg: 0 = centred, positive = target to the right.
    Returns None when no fresh packet is available (target lost).
    """

    STALE_S = 0.5

    def __init__(self):
        self._reading = None
        self._ts      = 0.0
        self._lock    = threading.Lock()
        self._sock    = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("", TARGET_UDP_PORT))
        self._sock.settimeout(0.05)
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        log.info(f"Target receiver on UDP:{TARGET_UDP_PORT}")

    def get(self):
        with self._lock:
            if self._reading is None:
                return None
            if time.monotonic() - self._ts > self.STALE_S:
                return None
            return self._reading   # (dist_m, angle_deg)

    def _recv_loop(self):
        while True:
            try:
                data, _ = self._sock.recvfrom(64)
                dist, ang = map(float, data.decode().strip().split(","))
                with self._lock:
                    self._reading = (dist, ang)
                    self._ts      = time.monotonic()
            except socket.timeout:
                pass
            except Exception as e:
                log.warning(f"Target receiver: {e}")


def relative_to_absolute(reading, robot_pos, robot_heading):
    """Camera (dist, angle_deg) → absolute map [x, y]."""
    if reading is None:
        return None
    dist, angle_deg = reading
    bearing_world = robot_heading - math.radians(angle_deg)
    return [robot_pos[0] + dist * math.cos(bearing_world),
            robot_pos[1] + dist * math.sin(bearing_world)]

# =============================================================================
# ROBOT STATE
# =============================================================================

robot_pos     = list(START_POS)
robot_heading = START_HEADING

target_visible         = False
last_target_pos        = list(robot_pos)
last_seen_bearing      = 0.0
last_seen_angle_in_fov = 0.0

goal_coords      = []
robot_route      = [list(robot_pos)]
lost             = True
robot_state      = S_SPIN
spin_turned      = 0.0
spin_dir         = 1.0
patrol_dir       = 1.0
aisles_checked   = [False] * len(AISLE_Y_CENTRES)
peek_start       = 0.0
peek_end         = 0.0
peek_aisle_i     = -1
aisle_travel_dir = 1.0

# Tracking sub-state and PD memory
track_phase      = TRACK_APPROACH    # APPROACH or HOLD
prev_target_dist = None              # distance measurement from the previous tick
                                     # used for the derivative term: de/dt ≈ Δdist/DT

# =============================================================================
# TICK
# =============================================================================

def tick(reading, sensor_pos, sensor_heading):
    """
    One control cycle.

    Parameters
    ----------
    reading        : (dist_m, angle_deg) from TargetReceiver.get(), or None.
                     dist_m    — range to target in metres
                     angle_deg — camera-frame bearing: 0 = centred,
                                 positive = target to the right
    sensor_pos     : [x, y] from Odometry.update()
    sensor_heading : float radians from Odometry.update()

    Returns
    -------
    (v_left_ms, v_right_ms)
    """
    global robot_pos, robot_heading, robot_route, robot_state
    global target_visible, last_target_pos, last_seen_bearing, last_seen_angle_in_fov
    global lost, goal_coords
    global spin_turned, spin_dir
    global patrol_dir, aisles_checked
    global peek_start, peek_end, peek_aisle_i
    global aisle_travel_dir
    global track_phase, prev_target_dist

    # -------------------------------------------------------------------------
    # 1. POSE FROM ODOMETRY
    # -------------------------------------------------------------------------
    robot_pos     = list(sensor_pos)
    robot_heading = sensor_heading

    # -------------------------------------------------------------------------
    # 2. SENSE
    #    reading is not None  ←→  camera sees the target.
    #    No FOV geometry recheck — camera visibility is the sole gate.
    # -------------------------------------------------------------------------
    target_visible = (reading is not None)

    if target_visible:
        dist_m, angle_deg = reading
        target_xy = relative_to_absolute(reading, robot_pos, robot_heading)
        last_target_pos        = list(target_xy)
        last_seen_bearing      = bearing_to(robot_pos, target_xy)
        last_seen_angle_in_fov = angle_diff(last_seen_bearing, robot_heading)
    else:
        dist_m    = None
        angle_deg = None
        target_xy = None

    # -------------------------------------------------------------------------
    # 3. ZONE
    # -------------------------------------------------------------------------
    on_edge  = is_on_edge(robot_pos)
    in_aisle = is_in_aisle(robot_pos)

    # -------------------------------------------------------------------------
    # 4. OBSTACLE CHECK
    # -------------------------------------------------------------------------
    if len(robot_route) >= 2:
        if not direct_path_clear(robot_route[0], robot_route[1]):
            robot_route = build_route(robot_pos, robot_route[-1])

    # -------------------------------------------------------------------------
    # 5. STATE TRANSITIONS
    # -------------------------------------------------------------------------
    if target_visible:
        if robot_state != S_IN_VIEW:
            lost           = False
            aisles_checked = [False] * len(AISLE_Y_CENTRES)
            goal_coords    = [list(target_xy)]
            robot_state    = S_IN_VIEW
            # Reset PD state on fresh acquisition
            track_phase      = TRACK_APPROACH
            prev_target_dist = dist_m

    elif robot_state == S_IN_VIEW:
        # Camera just lost the target
        goal_coords      = [list(last_target_pos)]
        robot_state      = S_FOLLOW_GOAL
        prev_target_dist = None

    elif robot_state == S_FOLLOW_GOAL and len(goal_coords) == 0:
        lost        = True
        spin_dir    = math.copysign(1.0, last_seen_angle_in_fov) \
                      if last_seen_angle_in_fov != 0 else 1.0
        spin_turned = 0.0
        robot_state = S_SPIN

    # -------------------------------------------------------------------------
    # 6. STATE ACTIONS
    # -------------------------------------------------------------------------
    v_forward = 0.0
    omega     = 0.0

    if robot_state == S_IN_VIEW:
        # ----------------------------------------------------------------
        # CENTERING
        # Always drive omega from the raw camera angle error so that the
        # target is kept at the centre of frame regardless of phase.
        # angle_deg > 0 → target is to the RIGHT → we need to turn RIGHT
        # (clockwise) → negative omega in CCW-positive convention.
        # ----------------------------------------------------------------
        omega = -CENTER_KP * angle_deg     # rad/s; negative = clockwise turn

        # Clamp centering to the maximum turn rate
        omega = math.copysign(min(abs(omega), TURN_SPEED_RAD), omega)

        # ----------------------------------------------------------------
        # PHASE DECISION
        # ----------------------------------------------------------------
        if dist_m is not None:
            if dist_m > HOLD_DIST_M:
                track_phase = TRACK_APPROACH
            else:
                track_phase = TRACK_HOLD

        # ----------------------------------------------------------------
        # APPROACH PHASE  — full speed forward until within HOLD_DIST_M
        # ----------------------------------------------------------------
        if track_phase == TRACK_APPROACH:
            v_forward = ROBOT_SPEED_MPS

        # ----------------------------------------------------------------
        # HOLD PHASE  — PD controller on distance error
        #
        # Error:      e  = dist_measured - HOLD_DIST_M
        #               positive → too far  → move forward
        #               negative → too close → move backward
        #
        # Derivative: de/dt ≈ (dist_now - dist_prev) / DT
        #               positive → target moving away → add forward speed
        #               negative → target approaching → reduce speed / back up
        #
        # Output:     v = Kp * e + Kd * de_dt
        #               clipped to ±HOLD_MAX_SPEED
        # ----------------------------------------------------------------
        elif track_phase == TRACK_HOLD:
            if dist_m is not None:
                e = dist_m - HOLD_DIST_M

                # Derivative of distance w.r.t. time
                if prev_target_dist is not None:
                    de_dt = (dist_m - prev_target_dist) / DT
                else:
                    de_dt = 0.0

                v_pd = HOLD_KP * e + HOLD_KD * de_dt

                # Clip to safe speed range (allows reversing if too close)
                v_forward = math.copysign(
                    min(abs(v_pd), HOLD_MAX_SPEED), v_pd
                )

        # Store current distance for next tick's derivative
        prev_target_dist = dist_m

        # Keep goal_coords up to date for use when we lose the target
        goal_coords = [list(target_xy)] if target_xy else goal_coords

        log.debug(
            f"[IN_VIEW/{track_phase}]  "
            f"dist={dist_m:.2f}m  angle={angle_deg:+.1f}°  "
            f"vFwd={v_forward:+.3f}  omega={math.degrees(omega):+.1f}°/s"
        )

    elif robot_state == S_FOLLOW_GOAL:
        if len(goal_coords) > 0:
            wp          = goal_coords[0]
            robot_route = build_route(robot_pos, wp)
            if len(robot_route) >= 2:
                want  = bearing_to(robot_route[0], robot_route[1])
                new_h = rotate_toward(robot_heading, want, DT)
                omega = angle_diff(new_h, robot_heading) / DT
                if abs(angle_diff(robot_heading, want)) < ALIGN_THRESH_RAD:
                    v_forward = ROBOT_SPEED_MPS
            if distance_between(robot_pos, wp) < ARRIVE_THRESH:
                goal_coords.pop(0)

    elif robot_state == S_SPIN:
        omega        = spin_dir * TURN_SPEED_RAD
        spin_turned += TURN_SPEED_RAD * DT
        v_forward    = 0.0
        if spin_turned >= 2 * math.pi:
            spin_turned = 0.0
            if on_edge:
                robot_state = S_EDGE_PATROL
                patrol_dir  = math.copysign(
                    1.0, last_target_pos[1] - robot_pos[1]) or 1.0
            else:
                robot_state      = S_AISLE_TRAVERSE
                aisle_travel_dir = math.copysign(
                    1.0, last_target_pos[0] - robot_pos[0]) or 1.0

    elif robot_state == S_EDGE_PATROL:
        if not on_edge:
            robot_state = S_AISLE_TRAVERSE
        else:
            ex = edge_x_centre(robot_pos)
            cy, idx = next_unchecked_aisle(robot_pos[1], patrol_dir, aisles_checked)
            if cy is None:
                patrol_dir *= -1
                cy, idx = next_unchecked_aisle(robot_pos[1], patrol_dir, aisles_checked)
            if cy is None:
                robot_state      = S_AISLE_TRAVERSE
                aisle_travel_dir = -1.0 if ex > map_size[0] / 2 else 1.0
            else:
                dist_to_mouth = abs(robot_pos[1] - cy)
                if dist_to_mouth < ARRIVE_THRESH:
                    peek_start, peek_end = peek_sweep_bounds([ex, cy])
                    new_h        = peek_start
                    omega        = angle_diff(new_h, robot_heading) / DT
                    v_forward    = 0.0
                    peek_aisle_i = idx
                    robot_state  = S_EDGE_PEEK
                else:
                    goal  = [ex, cy]
                    want  = bearing_to(robot_pos, goal)
                    new_h = rotate_toward(robot_heading, want, DT)
                    omega = angle_diff(new_h, robot_heading) / DT
                    if abs(angle_diff(robot_heading, want)) < ALIGN_THRESH_RAD:
                        robot_route = build_route(robot_pos, goal)
                        v_forward   = ROBOT_SPEED_MPS
                    if (robot_pos[1] <= aisle_width / 2.0 + 0.05 or
                            robot_pos[1] >= map_size[1] - aisle_width / 2.0 - 0.05):
                        patrol_dir *= -1

    elif robot_state == S_EDGE_PEEK:
        new_h = rotate_toward(robot_heading, peek_end, DT)
        omega = angle_diff(new_h, robot_heading) / DT
        v_forward = 0.0
        if abs(angle_diff(peek_end, robot_heading)) < math.radians(1.5):
            if peek_aisle_i >= 0:
                aisles_checked[peek_aisle_i] = True
            robot_state = S_EDGE_PATROL

    elif robot_state == S_AISLE_TRAVERSE:
        goal_x = (map_size[0] - aisle_width / 2.0) if aisle_travel_dir > 0 \
                 else aisle_width / 2.0
        goal   = [goal_x, robot_pos[1]]
        want   = 0.0 if aisle_travel_dir > 0 else math.pi
        new_h  = rotate_toward(robot_heading, want, DT)
        omega  = angle_diff(new_h, robot_heading) / DT
        if abs(angle_diff(robot_heading, want)) < ALIGN_THRESH_RAD:
            robot_route = build_route(robot_pos, goal)
            v_forward   = ROBOT_SPEED_MPS
        ai = aisle_index_of(robot_pos)
        if ai >= 0:
            aisles_checked[ai] = True
        if distance_between(robot_pos, goal) < ARRIVE_THRESH or is_on_edge(robot_pos):
            robot_state      = S_SPIN
            spin_turned      = 0.0
            spin_dir         = 1.0
            aisle_travel_dir *= -1
            v_forward        = 0.0

    # -------------------------------------------------------------------------
    # 7. WHEEL VELOCITIES
    # -------------------------------------------------------------------------
    v_left, v_right = velocities_to_wheel_commands(v_forward, omega)

    log.debug(
        f"[{robot_state:16s}]  pos=({robot_pos[0]:.2f},{robot_pos[1]:.2f})  "
        f"hdg={math.degrees(robot_heading):6.1f}°  "
        f"vL={v_left:+.3f} vR={v_right:+.3f} m/s  lost={lost}"
    )

    return v_left, v_right

# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    motors   = MotorDriver(MOTOR_UART_PORT, MOTOR_UART_BAUD)
    odometry = Odometry(
        start_pos=robot_pos,
        start_heading=robot_heading,
        encoder_reader=motors.read_encoder_deltas,
    )
    receiver = TargetReceiver()

    log.info("=" * 60)
    log.info("Robot controller starting")
    log.info(f"  Wheel diameter   : {WHEEL_DIAMETER_IN} in ({WHEEL_DIAMETER_M*1000:.1f} mm)")
    log.info(f"  Effective track  : {WHEEL_TRACK_IN} in ({WHEEL_TRACK_M*1000:.1f} mm)")
    log.info(f"  Encoder PPR      : {ENCODER_PPR}")
    log.info(f"  m / pulse        : {METRES_PER_PULSE*1000:.4f} mm")
    log.info(f"  rev / robot deg  : {WHEEL_ROTATIONS_PER_ROBOT_DEGREE:.6f}")
    log.info(f"  Hold distance    : {HOLD_DIST_M} m")
    log.info(f"  Center Kp        : {CENTER_KP} rad/s per deg")
    log.info(f"  Hold Kp / Kd     : {HOLD_KP} / {HOLD_KD}")
    log.info("=" * 60)

    try:
        while True:
            t0 = time.monotonic()

            pos, heading = odometry.update(DT)
            reading      = receiver.get()          # (dist_m, angle_deg) or None

            v_left, v_right = tick(reading, pos, heading)
            motors.send_velocities(v_left, v_right)

            elapsed = time.monotonic() - t0
            sleep_t = DT - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
            elif sleep_t < -0.01:
                log.warning(f"Loop overrun by {-sleep_t*1000:.1f} ms")

    except KeyboardInterrupt:
        log.info("Stopping.")
    finally:
        motors.stop()


if __name__ == "__main__":
    main()
