"""
robot_pi.py — Warehouse cart tracking controller
-------------------------------------------------
Runs on a Raspberry Pi connected to an ESP32 over USB serial.

ESP32 → Pi  (inbound):  "E,<left_ticks>,<right_ticks>\\n"
Pi → ESP32  (outbound): "L<left_rpm> R<right_rpm>\\n"
Vision → Pi (UDP):      "<distance_m>,<angle_deg>\\n"
  angle_deg: 0 = centred, positive = target to the right

Three-behaviour priority cascade:
  1. IN_VIEW     — PD on distance + P centering, simultaneous fwd+turn
  2. FOLLOW_GOAL — A* to last confirmed sighting coordinates
  3. LOST        — systematic edge/aisle patrol (SPIN→EDGE_PATROL→
                   EDGE_PEEK→AISLE_TRAVERSE)
"""

import math
import heapq
import logging
import socket
import threading
import time
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("robot")

# =============================================================================
# CONFIGURE — edit these before running
# =============================================================================

# Hardware: wheel width is 26.1mm, platform to wheel is 14.1mm, platform width is 30.48cm, wheel diameter is 67.78mm
WHEEL_DIAMETER_M = 0.06778          # outer wheel diameter (meters)
TRACK_M    = 0.333         # inside-to-inside wheel spacing (meters)

GEAR_RATIO        = 5            # motor gearbox ratio (check sticker)
LEFT_ENC_SIGN     = -1           # flip to +1 if left wheel counts backwards
RIGHT_ENC_SIGN    = +1
LEFT_MOTOR_SIGN   = -1           # flip to +1 if left motor drives wrong direction
RIGHT_MOTOR_SIGN  = +1

MOTOR_PORT  = "/dev/ttyUSB0"     # CP2102 USB-UART bridge
MOTOR_BAUD  = 115200
UDP_PORT    = 5005               # vision node sends here
UDP_STALE_S = 0.5                # seconds before a sighting is considered lost

# Map  (metres)
MAP_W        = 5.0
MAP_H        = 5.0
CHUNK        = 0.1               # grid resolution
AISLE_W      = 1.0               # corridor width
AISLE_COUNT  = 3                 # number of horizontal aisles

# Motion
MAX_SPEED    = 0.5               # m/s forward
MAX_TURN     = math.radians(90)  # rad/s
DT           = 0.02              # control loop period (seconds)
ARRIVE       = 1.5               # waypoint arrival radius (metres)
ALIGN        = math.radians(5)   # heading tolerance before moving (search states)

# Tracking  (IN_VIEW state)
HOLD_DIST    = 2.0               # target hold distance (metres)
MIN_APPROACH = 0.15              # floor speed when approaching (ensures progress)
CENTER_KP    = 0.02              # centering gain  (rad/s per degree of angle error)
DIST_KP      = 0.4               # distance PD proportional gain  (m/s per metre)
DIST_KD      = 0.3               # distance PD derivative gain    (m/s per m/s)
FOV_ENGAGE_DEG    = 25.0   # only rotate to keep shopper centred when angle exceeds this
LATERAL_KP        = 0.5    # cross-track correction weight for aisle-centre steering
AISLE_EDGE_MARGIN = 0.15   # clearance from aisle wall when edge-hugging (metres)
EDGE_ARRIVE       = 0.25   # tighter arrival radius for edge / centre goals (metres)

# Encoder counts per wheel revolution reported by the ESP32 after 4x
# quadrature decoding and gearbox reduction.
ENCODER_PPR  = 503

# =============================================================================
# DERIVED CONSTANTS  (do not edit)
# =============================================================================

WHEEL_CIRC   = math.pi * WHEEL_DIAMETER_M   # metres per revolution
M_PER_PULSE  = WHEEL_CIRC / ENCODER_PPR

# Wheel rotations for a robot-body degree of rotation (point turn geometry)
ROT_PER_DEG  = TRACK_M / (360.0 * WHEEL_DIAMETER_M)

# =============================================================================
# MAP CONSTRUCTION
# =============================================================================

_COLS     = int(MAP_W / CHUNK)
_ROWS     = int(MAP_H / CHUNK)
_GAP      = (MAP_H - AISLE_W * AISLE_COUNT) / (AISLE_COUNT - 1)
_PERIOD   = AISLE_W + _GAP

def _is_free(x, y):
    on_border = x < AISLE_W or x > MAP_W - AISLE_W \
             or y < AISLE_W or y > MAP_H - AISLE_W
    in_aisle  = (y % _PERIOD) < AISLE_W
    return on_border or in_aisle

CHUNK_MAP = [
    [1 if _is_free(_c * CHUNK, _r * CHUNK) else 0
     for _c in range(_COLS)]
    for _r in range(_ROWS)
]

def _aisle_centres():
    """Y-coordinates of all horizontal aisle centre lines."""
    seen, centres = set(), []
    for _r in range(_ROWS):
        y = _r * CHUNK
        if y < AISLE_W:
            cy = AISLE_W / 2
        elif y > MAP_H - AISLE_W:
            cy = MAP_H - AISLE_W / 2
        else:
            phase = y % _PERIOD
            if phase >= AISLE_W:
                continue
            cy = y - phase + AISLE_W / 2
        cy = round(cy, 6)
        if cy not in seen:
            seen.add(cy)
            centres.append(cy)
    return sorted(centres)

AISLE_CY = _aisle_centres()

# =============================================================================
# MAP HELPERS
# =============================================================================

def _chunk(pos):
    return int(pos[1] / CHUNK), int(pos[0] / CHUNK)

def _centre(rc):
    r, c = rc
    return [c * CHUNK + CHUNK / 2, r * CHUNK + CHUNK / 2]

def _free(rc):
    r, c = rc
    return 0 <= r < _ROWS and 0 <= c < _COLS and CHUNK_MAP[r][c] == 1

def _path_clear(a, b):
    ax, ay = a;  bx, by = b
    d = math.hypot(bx - ax, by - ay)
    n = max(int(d / (CHUNK * 0.5)), 2)
    return all(_free(_chunk([ax + t/n*(bx-ax), ay + t/n*(by-ay)]))
               for t in range(n + 1))

def _on_edge(pos):
    return pos[0] < AISLE_W or pos[0] > MAP_W - AISLE_W

def _edge_x(pos):
    return AISLE_W / 2 if pos[0] < MAP_W / 2 else MAP_W - AISLE_W / 2

def _inward(pos):
    return 0.0 if pos[0] < MAP_W / 2 else math.pi

# =============================================================================
# A* PATHFINDING
# =============================================================================

def _astar(start, goal):
    open_set = [(0, start)]
    came, g  = {}, {start: 0}
    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal:
            path = []
            while cur in came:
                path.append(cur); cur = came[cur]
            return list(reversed(path)) + [] or [goal]
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nb = (cur[0]+dr, cur[1]+dc)
            if not _free(nb): continue
            ng = g[cur] + 1
            if ng < g.get(nb, 1e9):
                came[nb] = cur; g[nb] = ng
                heapq.heappush(open_set,
                    (ng + abs(nb[0]-goal[0]) + abs(nb[1]-goal[1]), nb))
    return None

def build_route(start, goal):
    """Return a list of [x,y] waypoints from start to goal."""
    sc, gc = _chunk(start), _chunk(goal)
    if not _free(sc) or not _free(gc): return [start]
    if _path_clear(start, goal):       return [start, goal]
    path = _astar(sc, gc)
    if path is None:                   return [start]
    # Reconstruct: _astar returns chunks; convert and prepend start
    # (rebuild properly — _astar above has a bug in path reconstruction; fix below)
    return [start] + [_centre(rc) for rc in path]

# Fix _astar to return the full chunk path correctly
def _astar(start_rc, goal_rc):
    open_set = [(0, start_rc)]
    came, g  = {}, {start_rc: 0}
    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal_rc:
            path = [cur]
            while cur in came:
                cur = came[cur]; path.append(cur)
            path.reverse()
            return path
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nb = (cur[0]+dr, cur[1]+dc)
            if not _free(nb): continue
            ng = g[cur] + 1
            if ng < g.get(nb, 1e9):
                came[nb] = cur; g[nb] = ng
                heapq.heappush(open_set,
                    (ng + abs(nb[0]-goal_rc[0]) + abs(nb[1]-goal_rc[1]), nb))
    return None

def build_route(start, goal):
    sc, gc = _chunk(start), _chunk(goal)
    if not _free(sc) or not _free(gc): return [list(start)]
    if _path_clear(start, goal):       return [list(start), list(goal)]
    path = _astar(sc, gc)
    if path is None:                   return [list(start)]
    return [_centre(rc) for rc in path]

# =============================================================================
# GEOMETRY UTILITIES
# =============================================================================

def _dist(a, b):   return math.hypot(b[0]-a[0], b[1]-a[1])
def _bearing(f,t): return math.atan2(t[1]-f[1], t[0]-f[0])
def _adiff(a, b):  return ((a - b) + math.pi) % (2*math.pi) - math.pi

def _rotate_toward(h, target, dt):
    d = _adiff(target, h)
    return (h + math.copysign(min(abs(d), MAX_TURN * dt), d)) % (2*math.pi)

def _peek_bounds(edge_pos):
    """Heading sweep range for a full FOV pass across an aisle mouth."""
    perp = _inward(edge_pos)
    half = min(math.atan2(AISLE_W / 2, AISLE_W) + math.radians(45), math.pi/2)
    return perp - half, perp + half

def _nearest_aisle_cy(y):
    return min(AISLE_CY, key=lambda cy: abs(cy - y))

def _next_aisle(robot_y, direction, checked):
    """Nearest unchecked aisle y-centre in the given direction."""
    pool = [(cy, i) for i, cy in enumerate(AISLE_CY)
            if not checked[i]
            and (cy >= robot_y - 0.05 if direction > 0 else cy <= robot_y + 0.05)]
    if not pool: return None, None
    return (min if direction > 0 else max)(pool, key=lambda t: t[0])

def _wheel_commands(v_fwd, omega):
    """Differential drive mixing. Returns (v_left, v_right) in m/s."""
    half = TRACK_M / 2
    vl   = v_fwd - omega * half
    vr   = v_fwd + omega * half
    # Scale down if either wheel exceeds MAX_SPEED, preserving turn radius
    peak = max(abs(vl), abs(vr))
    if peak > MAX_SPEED:
        vl *= MAX_SPEED / peak
        vr *= MAX_SPEED / peak
    return vl, vr

def _rpm(v): return v / WHEEL_CIRC * 60

# =============================================================================
# STATE
# =============================================================================

@dataclass
class RobotState:
    # Pose (written from odometry every tick)
    pos:     list  = field(default_factory=lambda: [0.0, 0.0])
    heading: float = 0.0

    # Target knowledge
    target_visible:  bool  = False
    last_target_pos: list  = field(default_factory=lambda: [0.0, 0.0])
    last_angle:      float = 0.0   # signed offset in FOV at last sighting
    prev_dist:       float = None  # previous tick's distance (for PD derivative)

    # Navigation
    mode:       str  = "SPIN"
    goal_queue: list = field(default_factory=list)
    route:      list = field(default_factory=list)
    lost:       bool = True

    # Search state
    spin_turned:     float = 0.0
    spin_dir:        float = 1.0
    patrol_dir:      float = 1.0
    aisles_checked:  list  = field(default_factory=list)
    peek_start:      float = 0.0
    peek_end:        float = 0.0
    peek_aisle_i:    int   = -1
    traverse_dir:    float = 1.0
    avoid_edge_y:    float = 0.0   # y-target when hugging an aisle edge
    avoid_dir:       int   = 0     # +1 = positive-y edge, -1 = negative-y edge

    def __post_init__(self):
        self.aisles_checked = [False] * len(AISLE_CY)
        self.route          = [list(self.pos)]

S = RobotState()

# =============================================================================
# HARDWARE — Motor driver + encoder reader (shared serial port)
# =============================================================================

class MotorDriver:
    """
    Bidirectional ESP32 link over USB serial.
    Outbound: "L<rpm> R<rpm>\\n"
    Inbound:  "E,<left_ticks>,<right_ticks>\\n"  (signed, cumulative)
    """

    def __init__(self):
        self._ser = None
        self._last_l = self._last_r = None
        self._dl = self._dr = 0
        try:
            import serial
            self._ser = serial.Serial(MOTOR_PORT, MOTOR_BAUD, timeout=0.01)
            log.info(f"ESP32 connected on {MOTOR_PORT}")
        except Exception as e:
            log.warning(f"ESP32 unavailable ({e}) — logging only")

    def read_encoder_deltas(self):
        """Drain serial buffer, return signed pulse deltas since last call."""
        if not (self._ser and self._ser.is_open):
            return 0, 0
        while self._ser.in_waiting:
            try:
                line = self._ser.readline().decode(errors="ignore").strip()
            except Exception:
                break
            if not line.startswith("E,"): continue
            try:
                _, ls, rs  = line.split(",", 2)
                lc = LEFT_ENC_SIGN  * int(ls)
                rc = RIGHT_ENC_SIGN * int(rs)
                if self._last_l is not None:
                    self._dl += lc - self._last_l
                    self._dr += rc - self._last_r
                self._last_l, self._last_r = lc, rc
            except ValueError:
                pass
        dl, dr        = self._dl, self._dr
        self._dl = self._dr = 0
        return dl, dr

    def send(self, vl, vr):
        cmd = f"L{LEFT_MOTOR_SIGN * _rpm(vl):.1f} R{RIGHT_MOTOR_SIGN * _rpm(vr):.1f}\n".encode()
        if self._ser and self._ser.is_open:
            self._ser.write(cmd)
        else:
            log.debug(f"MOTOR: {cmd.decode().strip()}")

    def stop(self): self.send(0, 0)


class Odometry:
    """Dead-reckoning from encoder deltas using midpoint-heading integration."""

    def __init__(self, encoder_reader):
        self._read = encoder_reader
        self.x = self.y = 0.0
        self.heading = 0.0

    def update(self):
        dl, dr   = self._read()
        d_left   = dl * M_PER_PULSE
        d_right  = dr * M_PER_PULSE
        d_centre = (d_left + d_right) / 2
        d_theta  = (d_right - d_left) / TRACK_M
        mid      = self.heading + d_theta / 2
        self.x      += d_centre * math.cos(mid)
        self.y      += d_centre * math.sin(mid)
        self.heading = (self.heading + d_theta) % (2 * math.pi)
        return [self.x, self.y], self.heading


class TargetReceiver:
    """
    Background UDP listener. Returns (dist_m, angle_deg) or None if stale.
    angle_deg: 0 = centred, positive = target to the right of frame centre.
    """

    def __init__(self):
        self._data = None
        self._ts   = 0.0
        self._lock = threading.Lock()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("", UDP_PORT))
        sock.settimeout(0.02)
        threading.Thread(target=self._loop, args=(sock,), daemon=True).start()
        log.info(f"Vision receiver on UDP:{UDP_PORT}")

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
# TICK — one control cycle
# =============================================================================

def tick(reading, pos, heading, obstacles=None):
    """
    reading   : (dist_m, angle_deg) from TargetReceiver, or None
    pos       : [x, y] from Odometry
    heading   : float radians from Odometry
    obstacles : list of (dist_m, angle_deg) for non-target detections
    Returns   : (v_left, v_right) in m/s
    """
    obstacles = obstacles or []

    # 1. Update pose from sensors
    S.pos     = list(pos)
    S.heading = heading

    # 2. Target visibility — a UDP packet means the camera sees the target
    S.target_visible = reading is not None
    if S.target_visible:
        dist_m, angle_deg = reading
        # Convert camera sighting to map coordinates
        bearing   = S.heading - math.radians(angle_deg)
        target_xy = [S.pos[0] + dist_m * math.cos(bearing),
                     S.pos[1] + dist_m * math.sin(bearing)]
        S.last_target_pos = target_xy
        S.last_angle      = angle_deg

    # 3. Obstacle check — rebuild route if next segment is now blocked
    if len(S.route) >= 2 and not _path_clear(S.route[0], S.route[1]):
        S.route = build_route(S.pos, S.route[-1])

    # 4. State transitions (priority order)
    on_edge = _on_edge(S.pos)

    # Obstacle avoidance — highest priority, interrupts tracking/return modes
    if obstacles and S.mode in ("IN_VIEW", "FOLLOW_GOAL", "RETURN_CENTER"):
        avg_dist  = sum(d for d, _ in obstacles) / len(obstacles)
        avg_ang   = sum(a for _, a in obstacles) / len(obstacles)
        obs_bear  = S.heading - math.radians(avg_ang)
        obs_y     = S.pos[1] + avg_dist * math.sin(obs_bear)
        cy        = _nearest_aisle_cy(S.pos[1])
        S.avoid_dir    = -1 if obs_y >= cy else +1
        S.avoid_edge_y = cy + S.avoid_dir * (AISLE_W / 2 - AISLE_EDGE_MARGIN)
        S.goal_queue   = [[S.pos[0], S.avoid_edge_y]]
        S.route        = build_route(S.pos, S.goal_queue[0])
        S.mode         = "AVOID_EDGE"

    elif S.mode == "AVOID_EDGE" and not S.goal_queue:
        S.mode = "EDGE_FOLLOW"

    elif S.mode == "EDGE_FOLLOW" and not obstacles and S.target_visible:
        cy           = _nearest_aisle_cy(S.pos[1])
        S.goal_queue = [[S.pos[0], cy]]
        S.route      = build_route(S.pos, S.goal_queue[0])
        S.mode       = "RETURN_CENTER"
        S.lost       = False

    elif S.mode == "RETURN_CENTER" and not S.goal_queue:
        if S.target_visible:
            S.mode           = "IN_VIEW"
            S.lost           = False
            S.aisles_checked = [False] * len(AISLE_CY)
        else:
            S.mode        = "SPIN"
            S.lost        = True
            S.spin_turned = 0.0
            S.spin_dir    = math.copysign(1.0, S.last_angle) or 1.0

    elif S.target_visible and S.mode not in ("IN_VIEW", "AVOID_EDGE", "EDGE_FOLLOW", "RETURN_CENTER"):
        S.mode           = "IN_VIEW"
        S.lost           = False
        S.aisles_checked = [False] * len(AISLE_CY)
        S.goal_queue     = [list(target_xy)]

    elif not S.target_visible and S.mode == "IN_VIEW":
        S.mode       = "FOLLOW_GOAL"
        S.goal_queue = [list(S.last_target_pos)]
        S.prev_dist  = None

    elif S.mode == "FOLLOW_GOAL" and not S.goal_queue:
        S.mode        = "SPIN"
        S.lost        = True
        S.spin_turned = 0.0
        S.spin_dir    = math.copysign(1.0, S.last_angle) or 1.0

    # 5. State actions → compute v_forward and omega
    v_forward = omega = 0.0

    if S.mode == "IN_VIEW":
        aisle_cy  = _nearest_aisle_cy(S.pos[1])
        aisle_hdg = 0.0 if target_xy[0] >= S.pos[0] else math.pi

        # Cross-track correction: blend a small heading offset to return to aisle centre
        lat_err   = aisle_cy - S.pos[1]
        cross_ang = math.atan2(lat_err * LATERAL_KP, 1.0)
        want_hdg  = (aisle_hdg + cross_ang) % (2 * math.pi)
        omega     = _adiff(want_hdg, S.heading) / DT
        omega     = math.copysign(min(abs(omega), MAX_TURN), omega)

        # FOV-edge gate: only add centering correction when shopper nears FOV edge
        if abs(angle_deg) > FOV_ENGAGE_DEG:
            fov_err = angle_deg - math.copysign(FOV_ENGAGE_DEG, angle_deg)
            omega  += -CENTER_KP * fov_err
            omega   = math.copysign(min(abs(omega), MAX_TURN), omega)

        # Distance PD
        e     = dist_m - HOLD_DIST
        de_dt = (dist_m - S.prev_dist) / DT if S.prev_dist is not None else 0.0
        v_pd  = DIST_KP * e + DIST_KD * de_dt
        if e > 0:
            v_forward = min(max(v_pd, MIN_APPROACH), MAX_SPEED)
        else:
            v_forward = max(v_pd, -MAX_SPEED)

        S.prev_dist  = dist_m
        S.goal_queue = [list(target_xy)]

    elif S.mode == "FOLLOW_GOAL":
        if S.goal_queue:
            wp      = S.goal_queue[0]
            S.route = build_route(S.pos, wp)
            if len(S.route) >= 2:
                want  = _bearing(S.route[0], S.route[1])
                new_h = _rotate_toward(S.heading, want, DT)
                omega = _adiff(new_h, S.heading) / DT
                if abs(_adiff(S.heading, want)) < ALIGN:
                    v_forward = MAX_SPEED
            if _dist(S.pos, wp) < ARRIVE:
                S.goal_queue.pop(0)

    elif S.mode == "AVOID_EDGE":
        if S.goal_queue:
            wp      = S.goal_queue[0]
            S.route = build_route(S.pos, wp)
            if len(S.route) >= 2:
                want  = _bearing(S.route[0], S.route[1])
                new_h = _rotate_toward(S.heading, want, DT)
                omega = _adiff(new_h, S.heading) / DT
                if abs(_adiff(S.heading, want)) < ALIGN:
                    v_forward = MAX_SPEED
            if _dist(S.pos, wp) < EDGE_ARRIVE:
                S.goal_queue.pop(0)

    elif S.mode == "EDGE_FOLLOW":
        # Move along the aisle at the edge while waiting for shopper to reappear
        aisle_hdg = 0.0 if S.last_target_pos[0] >= S.pos[0] else math.pi
        lat_err   = S.avoid_edge_y - S.pos[1]
        cross_ang = math.atan2(lat_err * LATERAL_KP, 1.0)
        want_hdg  = (aisle_hdg + cross_ang) % (2 * math.pi)
        omega     = _adiff(want_hdg, S.heading) / DT
        omega     = math.copysign(min(abs(omega), MAX_TURN), omega)
        v_forward = MAX_SPEED

    elif S.mode == "RETURN_CENTER":
        if S.goal_queue:
            wp      = S.goal_queue[0]
            S.route = build_route(S.pos, wp)
            if len(S.route) >= 2:
                want  = _bearing(S.route[0], S.route[1])
                new_h = _rotate_toward(S.heading, want, DT)
                omega = _adiff(new_h, S.heading) / DT
                if abs(_adiff(S.heading, want)) < ALIGN:
                    v_forward = MAX_SPEED
            if _dist(S.pos, wp) < ARRIVE:
                S.goal_queue.pop(0)

    elif S.mode == "SPIN":
        omega         = S.spin_dir * MAX_TURN
        S.spin_turned += MAX_TURN * DT
        if S.spin_turned >= 2 * math.pi:
            S.spin_turned = 0.0
            if on_edge:
                S.mode       = "EDGE_PATROL"
                S.patrol_dir = math.copysign(
                    1.0, S.last_target_pos[1] - S.pos[1]) or 1.0
            else:
                S.mode         = "AISLE_TRAVERSE"
                S.traverse_dir = math.copysign(
                    1.0, S.last_target_pos[0] - S.pos[0]) or 1.0

    elif S.mode == "EDGE_PATROL":
        if not on_edge:
            S.mode = "AISLE_TRAVERSE"
        else:
            ex = _edge_x(S.pos)
            cy, idx = _next_aisle(S.pos[1], S.patrol_dir, S.aisles_checked)
            if cy is None:
                S.patrol_dir *= -1
                cy, idx = _next_aisle(S.pos[1], S.patrol_dir, S.aisles_checked)
            if cy is None:
                S.mode         = "AISLE_TRAVERSE"
                S.traverse_dir = -1.0 if ex > MAP_W / 2 else 1.0
            elif abs(S.pos[1] - cy) < ARRIVE:
                S.peek_start, S.peek_end = _peek_bounds([ex, cy])
                S.peek_aisle_i = idx
                S.mode         = "EDGE_PEEK"
                omega = _adiff(S.peek_start, S.heading) / DT
            else:
                goal  = [ex, cy]
                new_h = _rotate_toward(S.heading, _bearing(S.pos, goal), DT)
                omega = _adiff(new_h, S.heading) / DT
                if abs(_adiff(S.heading, _bearing(S.pos, goal))) < ALIGN:
                    S.route   = build_route(S.pos, goal)
                    v_forward = MAX_SPEED
                if S.pos[1] <= AISLE_W/2 + 0.05 or S.pos[1] >= MAP_H - AISLE_W/2 - 0.05:
                    S.patrol_dir *= -1

    elif S.mode == "EDGE_PEEK":
        new_h = _rotate_toward(S.heading, S.peek_end, DT)
        omega = _adiff(new_h, S.heading) / DT
        if abs(_adiff(S.peek_end, S.heading)) < math.radians(1.5):
            if S.peek_aisle_i >= 0:
                S.aisles_checked[S.peek_aisle_i] = True
            S.mode = "EDGE_PATROL"

    elif S.mode == "AISLE_TRAVERSE":
        goal_x = MAP_W - AISLE_W/2 if S.traverse_dir > 0 else AISLE_W/2
        goal   = [goal_x, S.pos[1]]
        want   = 0.0 if S.traverse_dir > 0 else math.pi
        new_h  = _rotate_toward(S.heading, want, DT)
        omega  = _adiff(new_h, S.heading) / DT
        if abs(_adiff(S.heading, want)) < ALIGN:
            S.route   = build_route(S.pos, goal)
            v_forward = MAX_SPEED
        ai = next((i for i, cy in enumerate(AISLE_CY)
                   if abs(S.pos[1] - cy) <= AISLE_W/2 + 0.05), -1)
        if ai >= 0:
            S.aisles_checked[ai] = True
        if _dist(S.pos, goal) < ARRIVE or on_edge:
            S.mode          = "SPIN"
            S.spin_turned   = 0.0
            S.spin_dir      = 1.0
            S.traverse_dir *= -1
            v_forward       = 0.0

    # 6. Mix and send
    vl, vr = _wheel_commands(v_forward, omega)

    log.debug(f"[{S.mode:14s}] pos=({S.pos[0]:.2f},{S.pos[1]:.2f}) "
              f"hdg={math.degrees(S.heading):6.1f}° "
              f"vL={vl:+.3f} vR={vr:+.3f}  lost={S.lost}")

    return vl, vr

# =============================================================================
# MAIN
# =============================================================================

def main():
    motors   = MotorDriver()
    odom     = Odometry(motors.read_encoder_deltas)
    receiver = TargetReceiver()

    log.info(f"Wheel: {WHEEL_DIAMETER_M*1000:.1f}mm dia, {TRACK_M*1000:.1f}mm track  "
             f"| PPR={ENCODER_PPR}  m/pulse={M_PER_PULSE*1000:.3f}mm  "
             f"| rev/deg={ROT_PER_DEG:.5f}")
    log.info(f"Hold {HOLD_DIST}m  |  Kp={DIST_KP}  Kd={DIST_KD}  "
             f"center_Kp={CENTER_KP}")

    try:
        while True:
            t0           = time.monotonic()
            pos, heading = odom.update()
            reading      = receiver.get()
            vl, vr       = tick(reading, pos, heading)
            motors.send(vl, vr)
            spare = DT - (time.monotonic() - t0)
            if spare > 0:
                time.sleep(spare)
            elif spare < -0.01:
                log.warning(f"Loop overrun {-spare*1000:.1f}ms")
    except KeyboardInterrupt:
        log.info("Stopped.")
    finally:
        motors.stop()


if __name__ == "__main__":
    main()
