import time
import math
import heapq
import threading
import socket
import struct
import logging

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("robot")

# =============================================================================
# ~~HARDWARE~~  PHYSICAL PARAMETERS  — measure and set these for your robot
# =============================================================================

WHEEL_DIAMETER_M   = 0.10      # metres  (e.g. 100 mm wheel)
WHEEL_TRACK_M      = 0.35      # metres  centre-to-centre between drive wheels
ENCODER_PPR        = 360       # pulses per full wheel revolution (after gearbox)

# Encoders are wired to the ESP32, not the Pi GPIO. The ESP32 streams
# encoder counts back over USB serial (line format: "E,<left>,<right>\n").
# Pin numbers below are for the ESP32 sketch in firmware/cart_motor/ — kept
# here as documentation only; the Pi does not read GPIO directly.
ESP32_ENC_LEFT_A   = 32   # ENC_A on dual_motor_test.ino (M1)
ESP32_ENC_LEFT_B   = 33   # ENC_B
ESP32_ENC_RIGHT_A  = 25   # ENC_C on dual_motor_test.ino (M2)
ESP32_ENC_RIGHT_B  = 26   # ENC_D

# Serial port for the ESP32 motor controller (USB CDC over micro-USB).
# On the Pi the ESP32 enumerates as /dev/ttyACM0 (or /dev/ttyUSB0 on some
# adapters). Matching firmware lives in firmware/cart_motor/.
MOTOR_UART_PORT    = "/dev/ttyACM0"
MOTOR_UART_BAUD    = 115200

# UDP port that vision streams target sightings on, formatted as
# "<distance_m>,<angle_deg>\n" — bearing is relative to the cart's camera
# (positive = target to the right of centre). The receiver converts to
# absolute map coordinates using the cart's current pose.
TARGET_UDP_PORT    = 5005

# Cart's fixed starting pose on the store map. The cart is physically
# placed here before each run; odometry integrates motion from this point.
START_POS          = [0.5, 0.5]   # metres
START_HEADING      = 0.0          # radians (0 = +x axis, CCW positive)

# =============================================================================
# DERIVED WHEEL CONSTANTS
# =============================================================================

WHEEL_CIRCUMFERENCE_M = math.pi * WHEEL_DIAMETER_M   # metres per revolution
# Metres of travel per encoder pulse
METRES_PER_PULSE      = WHEEL_CIRCUMFERENCE_M / ENCODER_PPR
# For a point turn: robot rotates (arc_length / track) radians
# arc_length per pulse on one wheel = METRES_PER_PULSE
RADIANS_PER_PULSE     = METRES_PER_PULSE / (WHEEL_TRACK_M / 2.0)

# =============================================================================
# MAP SETTINGS  (must match your physical warehouse layout)
# =============================================================================

chunk_size   = 0.1          # metres per chunk cell
map_size     = [20, 20]     # [width, height] in metres
FREE         = 1
BLOCKED      = 0
aisle_width  = 1.0          # metres
aisle_amount = 10

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

# Pre-compute aisle y-centres
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

ROBOT_SPEED_MPS  = 1.5              # maximum forward speed  m/s
TURN_SPEED_RAD   = math.radians(180.0)   # maximum turn rate  rad/s
FOV_DEG          = 60.0
FOV_RAD          = math.radians(FOV_DEG)
DT               = 0.1              # control loop period  seconds
ARRIVE_THRESH    = 0.18             # waypoint arrival radius  metres
ALIGN_THRESH_RAD = math.radians(5.0)  # must be within this to start moving

# =============================================================================
# STATE LABELS
# =============================================================================

S_IN_VIEW        = "IN_VIEW"
S_FOLLOW_GOAL    = "FOLLOW_GOAL"
S_SPIN           = "SPIN"
S_EDGE_PATROL    = "EDGE_PATROL"
S_EDGE_PEEK      = "EDGE_PEEK"
S_AISLE_TRAVERSE = "AISLE_TRAVERSE"

# =============================================================================
# GEOMETRY & MAP HELPERS  (unchanged from simulation)
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

def line_of_sight_clear(from_pos, to_pos):
    fx, fy = from_pos
    tx, ty = to_pos
    dist = math.hypot(tx - fx, ty - fy)
    if dist < 1e-6:
        return True
    steps = max(int(dist / (chunk_size * 0.5)), 2)
    for i in range(steps + 1):
        t  = i / steps
        pt = [fx + t * (tx - fx), fy + t * (ty - fy)]
        if not is_valid_chunk(world_to_chunk(pt)):
            return False
    return True

def target_in_fov(heading_rad, robot_pos, tgt_pos):
    b = bearing_to(robot_pos, tgt_pos)
    if abs(angle_diff(b, heading_rad)) > FOV_RAD / 2.0:
        return False
    return line_of_sight_clear(robot_pos, tgt_pos)

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
    half_sweep = math.atan2(aisle_width / 2.0, aisle_width) + FOV_RAD / 2.0
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
# A* PATHFINDING  (unchanged from simulation)
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
    Convert body-frame velocity commands to individual wheel speeds.

      v_forward  : desired forward speed  (m/s, positive = forward)
      omega      : desired angular velocity  (rad/s, positive = CCW)

    Returns (v_left, v_right) in m/s.

    Derivation:
      v_right = v_forward + omega * (track / 2)
      v_left  = v_forward - omega * (track / 2)

    The inner wheel of a turn travels a shorter arc:
      r_inner_contact = R - track/2 - wheel_width/2
    but for velocity commands to the motor controller, only the wheel
    centre radius matters:
      r_left  = R - track/2
      r_right = R + track/2
    which gives the formula above when combined with v = R * omega.
    """
    half_track = WHEEL_TRACK_M / 2.0
    v_left  = v_forward - omega * half_track
    v_right = v_forward + omega * half_track
    return v_left, v_right

def wheel_speed_to_rpm(v_wheel):
    """Convert wheel surface speed (m/s) to motor shaft RPM."""
    return (v_wheel / WHEEL_CIRCUMFERENCE_M) * 60.0

# =============================================================================
# ~~HARDWARE~~  ODOMETRY  — replace with your encoder library
# =============================================================================

class Odometry:
    """
    Dead-reckoning odometry from wheel encoder pulse counts.
    On a real Pi, read encoder ticks from GPIO interrupts or a
    dedicated encoder library (e.g. pigpio, RPi.GPIO).

    This class is a stub — replace _read_encoder_deltas() with
    real GPIO reads.
    """

    def __init__(self, start_pos, start_heading):
        self.x       = start_pos[0]
        self.y       = start_pos[1]
        self.heading = start_heading   # radians

    def update(self, dt):
        """
        Call once per tick.  Reads encoder deltas and integrates position.
        Returns (pos, heading).
        """
        delta_left, delta_right = self._read_encoder_deltas()

        # Distance each wheel travelled this tick
        d_left  = delta_left  * METRES_PER_PULSE
        d_right = delta_right * METRES_PER_PULSE

        # Robot-frame displacement
        d_centre = (d_left + d_right) / 2.0
        d_theta  = (d_right - d_left) / WHEEL_TRACK_M

        # Integrate pose
        self.heading = (self.heading + d_theta) % (2 * math.pi)
        self.x      += d_centre * math.cos(self.heading)
        self.y      += d_centre * math.sin(self.heading)

        return [self.x, self.y], self.heading

    def _read_encoder_deltas(self):
        """
        ~~HARDWARE~~
        Return (left_pulses, right_pulses) since the last call.
        Replace this stub with real GPIO encoder reads, e.g.:

            # Using pigpio:
            left  = encoder_left.get_and_reset()
            right = encoder_right.get_and_reset()
            return left, right
        """
        # Stub: returns zero movement
        return 0, 0


# =============================================================================
# ~~HARDWARE~~  MOTOR DRIVER  — replace with your UART/I2C protocol
# =============================================================================

class MotorDriver:
    """
    Sends left/right velocity commands to a motor driver over UART.

    Default protocol: "L<v_left_rpm> R<v_right_rpm>\n"
    Change send_velocities() to match your driver's protocol.
    """

    def __init__(self, port, baud):
        self._port = port
        self._baud = baud
        self._ser  = None
        self._connect()

    def _connect(self):
        try:
            import serial
            self._ser = serial.Serial(self._port, self._baud, timeout=0.05)
            log.info(f"Motor driver connected on {self._port} at {self._baud} baud")
        except Exception as e:
            log.warning(f"Motor driver not available ({e}) — commands will be logged only")
            self._ser = None

    def send_velocities(self, v_left_ms, v_right_ms):
        """
        ~~HARDWARE~~
        Convert m/s to whatever your driver expects and send it.

        Example for a Roboclaw (sends RPM targets):
            rpm_l = wheel_speed_to_rpm(v_left_ms)
            rpm_r = wheel_speed_to_rpm(v_right_ms)
            cmd = f"L{rpm_l:.1f} R{rpm_r:.1f}\n"

        Example for PWM-only driver (you add a PID loop here):
            duty_l = self._pid_left.update(v_left_ms, measured_left_ms)
            duty_r = self._pid_right.update(v_right_ms, measured_right_ms)
            GPIO.output(LEFT_PWM_PIN, duty_l)
            ...
        """
        rpm_l = wheel_speed_to_rpm(v_left_ms)
        rpm_r = wheel_speed_to_rpm(v_right_ms)
        cmd   = f"L{rpm_l:.1f} R{rpm_r:.1f}\n".encode()

        if self._ser and self._ser.is_open:
            self._ser.write(cmd)
        else:
            log.debug(f"MOTOR CMD: {cmd.decode().strip()}")

    def stop(self):
        self.send_velocities(0.0, 0.0)


# =============================================================================
# ~~HARDWARE~~  TARGET POSITION RECEIVER  — replace with your localisation feed
# =============================================================================

class TargetReceiver:
    """
    Listens for UDP packets of the form  "<distance_m>,<angle_deg>\\n"
    on TARGET_UDP_PORT.  Sender is vision/yolo_detect.py — distance is the
    metric range to the locked target and angle is its bearing in the
    camera frame (positive = right of centre).

    A packet is considered stale after STALE_S seconds; get() returns None
    after that so the state machine treats the target as lost.

    Runs in a background thread so the control loop never blocks on I/O.
    """

    STALE_S = 0.5

    def __init__(self):
        self._reading = None       # (dist_m, angle_deg) or None
        self._ts      = 0.0
        self._lock    = threading.Lock()
        self._sock    = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("", TARGET_UDP_PORT))
        self._sock.settimeout(0.05)
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        log.info(f"Target receiver listening on UDP port {TARGET_UDP_PORT}")

    def get(self):
        """Return latest (distance_m, angle_deg) or None if stale/never received."""
        with self._lock:
            if self._reading is None:
                return None
            if time.monotonic() - self._ts > self.STALE_S:
                return None
            return self._reading

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
                log.warning(f"Target receiver error: {e}")


def relative_to_absolute(reading, robot_pos, robot_heading):
    """
    Convert a (distance_m, angle_deg) sighting from the camera frame into
    an absolute (x, y) on the store map.

    angle_deg convention (matches yolo_detect.py): 0 = straight ahead,
    positive = target right of camera centre (clockwise from above).
    """
    if reading is None:
        return None
    dist, angle_deg = reading
    bearing_world = robot_heading - math.radians(angle_deg)
    return [robot_pos[0] + dist * math.cos(bearing_world),
            robot_pos[1] + dist * math.sin(bearing_world)]


# =============================================================================
# ROBOT STATE  (identical to simulation, initialised here as plain variables)
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


# =============================================================================
# TICK  — one control cycle (replaces animation update())
# =============================================================================

def tick(target_xy, sensor_pos, sensor_heading):
    """
    Run one control cycle.

    Parameters
    ----------
    target_xy      : [x, y] from TargetReceiver, or None if not yet seen
    sensor_pos     : [x, y] from Odometry.update()
    sensor_heading : float (radians) from Odometry.update()

    Returns
    -------
    (v_left, v_right)  wheel surface speeds in m/s
    """
    global robot_pos, robot_heading, robot_route, robot_state
    global target_visible, last_target_pos, last_seen_bearing, last_seen_angle_in_fov
    global lost, goal_coords
    global spin_turned, spin_dir
    global patrol_dir, aisles_checked
    global peek_start, peek_end, peek_aisle_i
    global aisle_travel_dir

    # -------------------------------------------------------------------------
    # 1. UPDATE POSE FROM SENSORS
    #    The simulation integrated position internally; here we trust the
    #    odometry/localisation system instead.
    # -------------------------------------------------------------------------
    robot_pos     = list(sensor_pos)
    robot_heading = sensor_heading

    # -------------------------------------------------------------------------
    # 2. SENSE — check FOV against received target position
    # -------------------------------------------------------------------------
    if target_xy is not None:
        in_fov = target_in_fov(robot_heading, robot_pos, target_xy)
    else:
        in_fov = False

    target_visible = in_fov

    if in_fov:
        last_target_pos        = list(target_xy)
        last_seen_bearing      = bearing_to(robot_pos, target_xy)
        last_seen_angle_in_fov = angle_diff(last_seen_bearing, robot_heading)

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
    if in_fov:
        if robot_state != S_IN_VIEW:
            lost           = False
            aisles_checked = [False] * len(AISLE_Y_CENTRES)
            goal_coords    = [list(target_xy)]
            robot_state    = S_IN_VIEW

    elif robot_state == S_IN_VIEW:
        goal_coords = [list(last_target_pos)]
        robot_state = S_FOLLOW_GOAL

    elif robot_state == S_FOLLOW_GOAL and len(goal_coords) == 0:
        lost        = True
        spin_dir    = math.copysign(1.0, last_seen_angle_in_fov) if last_seen_angle_in_fov != 0 else 1.0
        spin_turned = 0.0
        robot_state = S_SPIN

    # -------------------------------------------------------------------------
    # 6. STATE ACTIONS — compute desired v_forward and omega
    #    Instead of directly mutating robot_pos/heading (simulation style),
    #    we compute the *desired* velocities and let the motor driver +
    #    odometry close the loop on actual motion.
    # -------------------------------------------------------------------------

    v_forward = 0.0    # m/s forward
    omega     = 0.0    # rad/s CCW positive

    new_heading = robot_heading   # heading the state machine wants next tick

    if robot_state == S_IN_VIEW:
        want        = bearing_to(robot_pos, target_xy or last_target_pos)
        new_heading = rotate_toward(robot_heading, want, DT)
        omega       = angle_diff(new_heading, robot_heading) / DT

        if abs(angle_diff(robot_heading, want)) < FOV_RAD / 2.0:
            robot_route = build_route(robot_pos, target_xy or last_target_pos)
            v_forward   = ROBOT_SPEED_MPS
        goal_coords = [list(target_xy)] if target_xy else goal_coords

    elif robot_state == S_FOLLOW_GOAL:
        if len(goal_coords) > 0:
            wp          = goal_coords[0]
            robot_route = build_route(robot_pos, wp)
            if len(robot_route) >= 2:
                want        = bearing_to(robot_route[0], robot_route[1])
                new_heading = rotate_toward(robot_heading, want, DT)
                omega       = angle_diff(new_heading, robot_heading) / DT
                if abs(angle_diff(robot_heading, want)) < ALIGN_THRESH_RAD:
                    v_forward = ROBOT_SPEED_MPS
            if distance_between(robot_pos, wp) < ARRIVE_THRESH:
                goal_coords.pop(0)

    elif robot_state == S_SPIN:
        new_heading  = rotate_step(robot_heading, spin_dir, DT)
        omega        = spin_dir * TURN_SPEED_RAD
        spin_turned += TURN_SPEED_RAD * DT
        v_forward    = 0.0

        if spin_turned >= 2 * math.pi:
            spin_turned = 0.0
            if on_edge:
                robot_state = S_EDGE_PATROL
                patrol_dir  = math.copysign(1.0, last_target_pos[1] - robot_pos[1]) or 1.0
            else:
                robot_state      = S_AISLE_TRAVERSE
                aisle_travel_dir = math.copysign(1.0, last_target_pos[0] - robot_pos[0]) or 1.0

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
                    new_heading  = peek_start
                    omega        = angle_diff(new_heading, robot_heading) / DT
                    v_forward    = 0.0
                    peek_aisle_i = idx
                    robot_state  = S_EDGE_PEEK
                else:
                    goal        = [ex, cy]
                    want        = bearing_to(robot_pos, goal)
                    new_heading = rotate_toward(robot_heading, want, DT)
                    omega       = angle_diff(new_heading, robot_heading) / DT
                    if abs(angle_diff(robot_heading, want)) < ALIGN_THRESH_RAD:
                        robot_route = build_route(robot_pos, goal)
                        v_forward   = ROBOT_SPEED_MPS
                    if (robot_pos[1] <= aisle_width / 2.0 + 0.05 or
                            robot_pos[1] >= map_size[1] - aisle_width / 2.0 - 0.05):
                        patrol_dir *= -1

    elif robot_state == S_EDGE_PEEK:
        new_heading = rotate_toward(robot_heading, peek_end, DT)
        omega       = angle_diff(new_heading, robot_heading) / DT
        v_forward   = 0.0

        if abs(angle_diff(peek_end, robot_heading)) < math.radians(1.5):
            if peek_aisle_i >= 0:
                aisles_checked[peek_aisle_i] = True
            robot_state = S_EDGE_PATROL

    elif robot_state == S_AISLE_TRAVERSE:
        goal_x = (map_size[0] - aisle_width / 2.0) if aisle_travel_dir > 0 \
                 else aisle_width / 2.0
        goal   = [goal_x, robot_pos[1]]
        want   = 0.0 if aisle_travel_dir > 0 else math.pi

        new_heading = rotate_toward(robot_heading, want, DT)
        omega       = angle_diff(new_heading, robot_heading) / DT

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
    # 7. CONVERT TO WHEEL VELOCITIES
    #    v_left, v_right are surface speeds in m/s.
    #    Pass these to MotorDriver.send_velocities().
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
    odometry = Odometry(start_pos=robot_pos, start_heading=robot_heading)
    motors   = MotorDriver(MOTOR_UART_PORT, MOTOR_UART_BAUD)
    receiver = TargetReceiver()

    log.info("Control loop starting.  Press Ctrl-C to stop.")
    log.info(f"Wheel diameter : {WHEEL_DIAMETER_M*100:.0f} mm")
    log.info(f"Wheel track    : {WHEEL_TRACK_M*100:.0f} mm  (centre to centre)")
    log.info(f"Encoder PPR    : {ENCODER_PPR}")
    log.info(f"m/pulse        : {METRES_PER_PULSE*1000:.3f} mm")

    try:
        while True:
            t0 = time.monotonic()

            # Read sensors
            pos, heading = odometry.update(DT)
            reading      = receiver.get()              # (dist_m, angle_deg) or None
            target_xy    = relative_to_absolute(reading, pos, heading)

            # Run one control tick
            v_left, v_right = tick(target_xy, pos, heading)

            # Send to motors
            motors.send_velocities(v_left, v_right)

            # Hold loop period precisely
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
