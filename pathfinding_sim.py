import numpy as np
import heapq
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from IPython.display import Video

# =============================================================================
# MAP SETTINGS
# =============================================================================

chunk_size   = 0.1
map_size     = [20, 20]
FREE         = 1
BLOCKED      = 0
aisle_width  = 1
aisle_amount = 10

cols = int(map_size[0] / chunk_size)
rows = int(map_size[1] / chunk_size)

gap_width = (map_size[1] - aisle_width * aisle_amount) / (aisle_amount - 1)

chunk_map = np.zeros((rows, cols), dtype=int)

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

# Pre-compute y-centres of every horizontal aisle corridor
_period = aisle_width + gap_width
AISLE_Y_CENTRES = []
_probe_y = 0.0
while _probe_y <= map_size[1]:
    if _probe_y < aisle_width:
        AISLE_Y_CENTRES.append(round(aisle_width / 2.0, 6))
    elif _probe_y > map_size[1] - aisle_width:
        cy = map_size[1] - aisle_width / 2.0
        if round(cy, 6) not in AISLE_Y_CENTRES:
            AISLE_Y_CENTRES.append(round(cy, 6))
    else:
        phase = _probe_y % _period
        if phase < aisle_width:
            cy = _probe_y - phase + aisle_width / 2.0
            if round(cy, 6) not in AISLE_Y_CENTRES:
                AISLE_Y_CENTRES.append(round(cy, 6))
    _probe_y += chunk_size

AISLE_Y_CENTRES = sorted(set(AISLE_Y_CENTRES))

# =============================================================================
# ROBOT & TARGET PARAMETERS
# =============================================================================

ROBOT_SPEED_MPS  = 1.5          # cart speed m/s
TARGET_SPEED_MPS = 1.7          # target is faster than cart
TURN_SPEED_RAD   = np.deg2rad(180.0)   # 180 deg/s turn rate
FOV_DEG          = 60.0
FOV_RAD          = np.deg2rad(FOV_DEG)
DT               = 0.1          # seconds per tick
ARRIVE_THRESH    = 0.18         # metres — "close enough" to a waypoint

# =============================================================================
# STATE LABELS
# =============================================================================
S_IN_VIEW        = "IN_VIEW"       # target visible → go to it
S_FOLLOW_GOAL    = "FOLLOW_GOAL"   # following last-known pos, target not visible
S_SPIN           = "SPIN"          # full 360° spin to search
S_EDGE_PATROL    = "EDGE_PATROL"   # walk along edge, stop at each aisle mouth
S_EDGE_PEEK      = "EDGE_PEEK"     # sweep FOV across aisle opening
S_AISLE_TRAVERSE = "AISLE_TRAVERSE"# traverse aisle to far edge

# =============================================================================
# HELPERS
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
    return np.hypot(b[0] - a[0], b[1] - a[1])

def angle_diff(a, b):
    """Signed shortest arc a-b, in (-pi, pi]."""
    return ((a - b) + np.pi) % (2 * np.pi) - np.pi

def bearing_to(frm, to):
    return np.arctan2(to[1] - frm[1], to[0] - frm[0])

def line_of_sight_clear(from_pos, to_pos):
    """
    Raycast from from_pos to to_pos through chunk_map.
    Returns True only if every chunk along the ray is FREE.
    Blocked chunks are opaque — they break line of sight.
    """
    start = np.array(from_pos, dtype=float)
    end   = np.array(to_pos,   dtype=float)
    dist  = np.linalg.norm(end - start)
    if dist < 1e-6:
        return True
    # Sample at half-chunk resolution so no chunk is skipped
    steps = max(int(dist / (chunk_size * 0.5)), 2)
    for i in range(steps + 1):
        t  = i / steps
        pt = start + t * (end - start)
        if not is_valid_chunk(world_to_chunk(pt)):
            return False
    return True

def target_in_fov(heading_rad, robot_pos, tgt_pos):
    """
    Target is visible only if:
      1. It lies within the FOV angular cone, AND
      2. The line of sight is unobstructed (blocked chunks are opaque).
    Cart has zero knowledge of target position when either condition fails.
    """
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
    """Rotate heading by TURN_SPEED_RAD*dt in direction (+1 CCW, -1 CW)."""
    return (heading + direction * TURN_SPEED_RAD * dt) % (2 * np.pi)

def rotate_toward(heading, target_h, dt):
    """Rotate toward target_h at TURN_SPEED_RAD, return new heading."""
    diff = angle_diff(target_h, heading)
    step = np.sign(diff) * min(abs(diff), TURN_SPEED_RAD * dt)
    return (heading + step) % (2 * np.pi)

def inward_heading(pos):
    """Heading pointing straight inward (toward map centre in x) from an edge."""
    return 0.0 if pos[0] < map_size[0] / 2.0 else np.pi

def edge_x_centre(pos):
    return aisle_width / 2.0 if pos[0] < map_size[0] / 2.0 \
           else map_size[0] - aisle_width / 2.0

def peek_sweep_bounds(edge_pos):
    """
    Start and end headings for a full FOV sweep across an aisle opening from
    the edge.  The sweep spans ±(arctan(aisle_width/2 / aisle_width) + FOV/2)
    around the inward-perpendicular heading, capped at ±90°.
    """
    perp       = inward_heading(edge_pos)
    half_sweep = np.arctan2(aisle_width / 2.0, aisle_width) + FOV_RAD / 2.0
    half_sweep = min(half_sweep, np.pi / 2.0)
    return perp - half_sweep, perp + half_sweep

def next_unchecked_aisle(robot_y, patrol_dir, aisles_checked):
    """
    Return (y_centre, index) of the nearest unchecked aisle in patrol_dir.
    Returns (None, None) if all checked.
    """
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
    if patrol_dir > 0:
        return min(candidates, key=lambda t: t[0])
    else:
        return max(candidates, key=lambda t: t[0])

# =============================================================================
# A* PATHFINDING
# =============================================================================

def direct_path_clear(start_pos, end_pos):
    start = np.array(start_pos)
    end   = np.array(end_pos)
    dist  = distance_between(start_pos, end_pos)
    steps = max(int(dist / chunk_size), 1)
    for i in range(steps + 1):
        t  = i / steps
        pt = start + t * (end - start)
        if not is_valid_chunk(world_to_chunk(pt)):
            return False
    return True

def get_neighbors(chunk):
    r, c = chunk
    return [(r + dr, c + dc)
            for dr, dc in ((1,0),(-1,0),(0,1),(0,-1))
            if is_valid_chunk((r + dr, c + dc))]

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
        for nb in get_neighbors(cur):
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

# =============================================================================
# MOVEMENT
# =============================================================================

def move_along_route(pos, route, speed, dt):
    if len(route) < 2:
        return pos, route
    cur    = np.array(pos, dtype=float)
    budget = speed * dt
    route  = list(route)
    while len(route) >= 2 and budget > 0:
        nxt  = np.array(route[1], dtype=float)
        diff = nxt - cur
        d    = np.linalg.norm(diff)
        if d < 1e-4:
            route.pop(0); continue
        if budget >= d:
            cur = nxt; budget -= d; route.pop(0)
        else:
            cur += (diff / d) * budget; budget = 0
    return [float(cur[0]), float(cur[1])], route

def random_valid_position():
    while True:
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        if is_valid_chunk((r, c)):
            return chunk_to_world((r, c))

# =============================================================================
# INITIALISE TARGET (independent controller)
# =============================================================================

target_pos   = random_valid_position()
target_goal  = random_valid_position()
target_route = build_route(target_pos, target_goal)

# =============================================================================
# ROBOT STATE
# =============================================================================

robot_pos     = [0.5, 0.5]
robot_heading = 0.0                     # radians

if not is_valid_chunk(world_to_chunk(robot_pos)):
    raise ValueError("Robot starts in blocked space.")

# ---- Sensing variables (spec) ----
# last_target_pos: only updated when target_visible is True.
# Initialised to robot_pos so the first spin/patrol direction is neutral.
target_visible         = False
last_target_pos        = list(robot_pos)   # unknown until first sighting
last_seen_bearing      = 0.0
last_seen_angle_in_fov = 0.0

# ---- Goal queue ----
goal_coords = []          # list of [x,y] waypoints; robot works front-to-back
robot_route = [list(robot_pos)]

# ---- Lost flag ----
lost = True

# ---- State ----
robot_state = S_SPIN

# ---- SPIN ----
spin_turned   = 0.0       # accumulated radians this spin
spin_dir      = 1.0       # +1 CCW (toward last-seen side)

# ---- EDGE_PATROL ----
patrol_dir     = 1.0      # +1 up in y, -1 down
aisles_checked = [False] * len(AISLE_Y_CENTRES)

# ---- EDGE_PEEK ----
peek_start   = 0.0
peek_end     = 0.0
peek_aisle_i = -1

# ---- AISLE_TRAVERSE ----
aisle_travel_dir = 1.0    # +1 right (+x), -1 left (-x)

# =============================================================================
# PLOT SETUP
# =============================================================================

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(chunk_map, origin="lower",
          extent=[0, map_size[0], 0, map_size[1]],
          interpolation="nearest", cmap="gray", vmin=0, vmax=1)

robot_dot,       = ax.plot([], [], "o",  ms=9,  color="royalblue",  label="Robot")
target_dot,      = ax.plot([], [], "x",  ms=9,  color="crimson",    label="Target")
target_goal_dot, = ax.plot([], [], "*",  ms=8,  color="orange",     label="Target goal")
robot_wp_dot,    = ax.plot([], [], "D",  ms=6,  color="cyan",       label="Robot waypoint")

robot_route_line,  = ax.plot([], [], lw=1.5, color="royalblue", label="Robot route")
target_route_line, = ax.plot([], [], lw=1,   color="crimson", ls="--", label="Target route")

fov_wedge = mpatches.Wedge([0, 0], r=4.0, theta1=0, theta2=0,
                            alpha=0.18, color="royalblue", label="FOV")
ax.add_patch(fov_wedge)

heading_arrow = ax.annotate("", xy=[0, 0], xytext=[0, 0],
                             arrowprops=dict(arrowstyle="->", color="royalblue", lw=1.8))

state_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                     va="top", fontsize=7.5, color="white",
                     bbox=dict(boxstyle="round", fc="black", alpha=0.55))

ax.set_xlim(0, map_size[0])
ax.set_ylim(0, map_size[1])
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title(f"FOV patrol  |  FOV={FOV_DEG:.0f}°  "
             f"turn=180°/s  robot={ROBOT_SPEED_MPS}m/s  target={TARGET_SPEED_MPS}m/s")
ax.legend(loc="upper right", fontsize=7)

# =============================================================================
# UPDATE — one tick = DT seconds
# =============================================================================

def update(frame):
    global robot_pos, robot_heading, robot_route, robot_state
    global target_pos, target_goal, target_route
    global target_visible, last_target_pos, last_seen_bearing, last_seen_angle_in_fov
    global lost, goal_coords
    global spin_turned, spin_dir
    global patrol_dir, aisles_checked
    global peek_start, peek_end, peek_aisle_i
    global aisle_travel_dir

    # =========================================================================
    # 1. MOVE TARGET  (independent controller — faster than robot)
    # =========================================================================
    if distance_between(target_pos, target_goal) < 0.15 or len(target_route) < 2:
        target_goal  = random_valid_position()
        target_route = build_route(target_pos, target_goal)
    if frame % 10 == 0:
        target_route = build_route(target_pos, target_goal)
    target_pos, target_route = move_along_route(
        target_pos, target_route, TARGET_SPEED_MPS, DT)

    # =========================================================================
    # 2. SENSE
    # =========================================================================
    in_fov = target_in_fov(robot_heading, robot_pos, target_pos)
    target_visible = in_fov

    if in_fov:
        last_target_pos        = list(target_pos)
        last_seen_bearing      = bearing_to(robot_pos, target_pos)
        last_seen_angle_in_fov = angle_diff(last_seen_bearing, robot_heading)

    # =========================================================================
    # 3. ZONE
    # =========================================================================
    on_edge  = is_on_edge(robot_pos)
    in_aisle = is_in_aisle(robot_pos)

    # =========================================================================
    # 4. OBSTACLE CHECK — interrupt any state if route becomes blocked
    #    (In this map obstacles are static, so we just check the current route.)
    # =========================================================================
    if len(robot_route) >= 2:
        if not direct_path_clear(robot_route[0], robot_route[1]):
            robot_route = build_route(robot_pos, robot_route[-1])

    # =========================================================================
    # 5. STATE TRANSITIONS  (priority: IN_VIEW > FOLLOW_GOAL > lost sequence)
    # =========================================================================

    if in_fov:
        # --- Target acquired / re-acquired ---
        if robot_state != S_IN_VIEW:
            lost           = False
            aisles_checked = [False] * len(AISLE_Y_CENTRES)
            goal_coords    = [list(target_pos)]
            robot_state    = S_IN_VIEW

    elif robot_state == S_IN_VIEW:
        # --- Just lost the target ---
        # Keep heading toward last known position
        goal_coords = [list(last_target_pos)]
        robot_state = S_FOLLOW_GOAL

    elif robot_state == S_FOLLOW_GOAL and len(goal_coords) == 0:
        # --- Reached last known position, still no target → go lost ---
        lost        = True
        # Spin in the direction the target was last seen (left/right of FOV)
        spin_dir    = np.sign(last_seen_angle_in_fov) if last_seen_angle_in_fov != 0 else 1.0
        spin_turned = 0.0
        robot_state = S_SPIN

    # =========================================================================
    # 6. STATE ACTIONS
    # =========================================================================

    # ------------------------------------------------------------------
    if robot_state == S_IN_VIEW:
    # Face the target, then move toward it.
    # Heading = direction of travel at all times.
    # ------------------------------------------------------------------
        goal_coords = [list(target_pos)]
        want        = bearing_to(robot_pos, target_pos)
        robot_heading = rotate_toward(robot_heading, want, DT)

        # Move only once heading is aligned within half the FOV cone
        if abs(angle_diff(robot_heading, want)) < FOV_RAD / 2.0:
            robot_route = build_route(robot_pos, target_pos)
            robot_pos, robot_route = move_along_route(
                robot_pos, robot_route, ROBOT_SPEED_MPS, DT)
            # After moving, update heading to face next waypoint
            if len(robot_route) >= 2:
                robot_heading = rotate_toward(
                    robot_heading, bearing_to(robot_route[0], robot_route[1]), DT)

    # ------------------------------------------------------------------
    elif robot_state == S_FOLLOW_GOAL:
    # Walk toward last known target position.
    # Heading always equals direction of travel — turn first, then move.
    # ------------------------------------------------------------------
        if len(goal_coords) > 0:
            wp          = goal_coords[0]
            robot_route = build_route(robot_pos, wp)
            # Determine the immediate travel direction from the route
            if len(robot_route) >= 2:
                want          = bearing_to(robot_route[0], robot_route[1])
                robot_heading = rotate_toward(robot_heading, want, DT)
                # Move only once heading is close to travel direction
                if abs(angle_diff(robot_heading, want)) < np.deg2rad(5.0):
                    robot_pos, robot_route = move_along_route(
                        robot_pos, robot_route, ROBOT_SPEED_MPS, DT)
            if distance_between(robot_pos, wp) < ARRIVE_THRESH:
                goal_coords.pop(0)
        # (emptied queue handled in transition block above)

    # ------------------------------------------------------------------
    elif robot_state == S_SPIN:
    # Rotate at full turn speed in spin_dir for a full 360°.
    # On completion, enter edge or aisle patrol depending on zone.
    # ------------------------------------------------------------------
        robot_heading = rotate_step(robot_heading, spin_dir, DT)
        spin_turned  += TURN_SPEED_RAD * DT

        if spin_turned >= 2 * np.pi:
            spin_turned = 0.0
            if on_edge:
                robot_state = S_EDGE_PATROL
                patrol_dir  = np.sign(last_target_pos[1] - robot_pos[1]) or 1.0
            else:
                robot_state      = S_AISLE_TRAVERSE
                aisle_travel_dir = np.sign(last_target_pos[0] - robot_pos[0]) or 1.0

    # ------------------------------------------------------------------
    elif robot_state == S_EDGE_PATROL:
    # Walk along the vertical edge.  At each unchecked aisle mouth,
    # trigger a full FOV sweep (EDGE_PEEK).  Flip direction at map
    # boundary.  When all aisles checked, cut across via AISLE_TRAVERSE.
    # ------------------------------------------------------------------
        if not on_edge:
            # Drifted off edge → traverse current aisle to far edge
            robot_state = S_AISLE_TRAVERSE
        else:
            ex = edge_x_centre(robot_pos)
            cy, idx = next_unchecked_aisle(robot_pos[1], patrol_dir, aisles_checked)

            if cy is None:
                # This direction exhausted — try the other
                patrol_dir *= -1
                cy, idx = next_unchecked_aisle(robot_pos[1], patrol_dir, aisles_checked)

            if cy is None:
                # All aisles checked — enter the map via aisle traverse
                robot_state      = S_AISLE_TRAVERSE
                aisle_travel_dir = -1.0 if ex > map_size[0] / 2 else 1.0
            else:
                dist_to_mouth = abs(robot_pos[1] - cy)
                if dist_to_mouth < ARRIVE_THRESH:
                    # Snap to mouth centre and begin sweep
                    robot_pos     = [ex, cy]
                    peek_start, peek_end = peek_sweep_bounds(robot_pos)
                    robot_heading = peek_start      # jump to sweep start
                    peek_aisle_i  = idx
                    robot_state   = S_EDGE_PEEK
                else:
                    # Walk toward the next aisle mouth.
                    # Heading = direction of travel (along the edge in y).
                    # The cart looks where it's going — it will spot the target
                    # if it walks past an open aisle mouth within the FOV cone.
                    goal = [ex, cy]
                    want = bearing_to(robot_pos, goal)
                    robot_heading = rotate_toward(robot_heading, want, DT)

                    if abs(angle_diff(robot_heading, want)) < np.deg2rad(5.0):
                        robot_route = build_route(robot_pos, goal)
                        robot_pos, robot_route = move_along_route(
                            robot_pos, robot_route, ROBOT_SPEED_MPS, DT)

                    # Hit the map boundary → flip direction
                    if (robot_pos[1] <= aisle_width / 2.0 + 0.05 or
                            robot_pos[1] >= map_size[1] - aisle_width / 2.0 - 0.05):
                        patrol_dir *= -1

    # ------------------------------------------------------------------
    elif robot_state == S_EDGE_PEEK:
    # Sweep heading from peek_start to peek_end at full turn speed.
    # Mark aisle as checked when sweep is complete, then resume patrol.
    # ------------------------------------------------------------------
        robot_heading = rotate_toward(robot_heading, peek_end, DT)

        if abs(angle_diff(peek_end, robot_heading)) < np.deg2rad(1.5):
            # Sweep done
            if peek_aisle_i >= 0:
                aisles_checked[peek_aisle_i] = True
            robot_state = S_EDGE_PATROL

    # ------------------------------------------------------------------
    elif robot_state == S_AISLE_TRAVERSE:
    # Travel along the current aisle to the far vertical edge.
    # Heading always equals the direction of travel (left or right).
    # Turn first, then move — the cart looks only where it's going.
    # On reaching the far edge → SPIN → EDGE_PATROL on that edge.
    # ------------------------------------------------------------------
        goal_x = (map_size[0] - aisle_width / 2.0) if aisle_travel_dir > 0 \
                 else aisle_width / 2.0
        goal   = [goal_x, robot_pos[1]]
        want   = 0.0 if aisle_travel_dir > 0 else np.pi

        robot_heading = rotate_toward(robot_heading, want, DT)

        # Move only once heading matches travel direction
        if abs(angle_diff(robot_heading, want)) < np.deg2rad(5.0):
            robot_route = build_route(robot_pos, goal)
            robot_pos, robot_route = move_along_route(
                robot_pos, robot_route, ROBOT_SPEED_MPS, DT)

        # Mark the aisle we are traversing as checked
        ai = aisle_index_of(robot_pos)
        if ai >= 0:
            aisles_checked[ai] = True

        if distance_between(robot_pos, goal) < ARRIVE_THRESH or is_on_edge(robot_pos):
            robot_state      = S_SPIN
            spin_turned      = 0.0
            spin_dir         = 1.0
            aisle_travel_dir *= -1

    # =========================================================================
    # 7. UPDATE VISUALS
    # =========================================================================
    hd = np.rad2deg(robot_heading)
    fov_wedge.set_center(robot_pos)
    fov_wedge.theta1 = hd - FOV_DEG / 2.0
    fov_wedge.theta2 = hd + FOV_DEG / 2.0
    fov_wedge.set_color("limegreen" if in_fov else "royalblue")

    arr_len = 1.2
    heading_arrow.xy = (robot_pos[0], robot_pos[1])
    heading_arrow.set_position((robot_pos[0] + arr_len * np.cos(robot_heading),
                                robot_pos[1] + arr_len * np.sin(robot_heading)))

    rrx = [p[0] for p in robot_route]
    rry = [p[1] for p in robot_route]
    trx = [p[0] for p in target_route]
    try_ = [p[1] for p in target_route]

    robot_dot.set_data([robot_pos[0]], [robot_pos[1]])
    target_dot.set_data([target_pos[0]], [target_pos[1]])
    target_goal_dot.set_data([target_goal[0]], [target_goal[1]])

    robot_wp_dot.set_data(
        [goal_coords[0][0]] if goal_coords else [],
        [goal_coords[0][1]] if goal_coords else [])

    robot_route_line.set_data(rrx, rry)
    target_route_line.set_data(trx, try_)

    zone = "edge" if on_edge else ("aisle" if in_aisle else "gap")
    state_text.set_text(
        f"State  : {robot_state}\n"
        f"Zone   : {zone}\n"
        f"Heading: {hd:.1f}°\n"
        f"Lost   : {lost}   visible: {target_visible}\n"
        f"FOV angle: {np.rad2deg(last_seen_angle_in_fov):.1f}°\n"
        f"Aisles checked: {sum(aisles_checked)}/{len(AISLE_Y_CENTRES)}"
    )

    return (robot_dot, target_dot, target_goal_dot, robot_wp_dot,
            robot_route_line, target_route_line,
            fov_wedge, state_text)

# =============================================================================
# RUN
# =============================================================================

ani = FuncAnimation(fig, update, frames=500, interval=int(DT * 1000), blit=True)
ani.save("pathfinding_sim.mp4", writer="ffmpeg", fps=10, dpi=70)
Video("pathfinding_sim.mp4", embed=True)
