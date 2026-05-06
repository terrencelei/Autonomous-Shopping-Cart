import numpy as np
import heapq
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import Video

# ================= Map settings =================

chunk_size = 0.1
map_size = [20, 20]

FREE = 1
BLOCKED = 0

aisle_width = 1
aisle_amount = 10

cols = int(map_size[0] / chunk_size)
rows = int(map_size[1] / chunk_size)

gap_width = (map_size[1] - aisle_width * aisle_amount) / (aisle_amount - 1)

chunk_map = np.zeros((rows, cols), dtype=int)

for row in range(rows):
    for col in range(cols):
        x = col * chunk_size
        y = row * chunk_size

        if x < aisle_width:
            chunk_map[row, col] = FREE
        elif x > map_size[0] - aisle_width:
            chunk_map[row, col] = FREE
        elif y < aisle_width:
            chunk_map[row, col] = FREE
        elif y > map_size[1] - aisle_width:
            chunk_map[row, col] = FREE
        elif (y % (aisle_width + gap_width)) < aisle_width:
            chunk_map[row, col] = FREE


# ================= Robot / target settings =================

robot_pos = [0.5, 0.5]

robot_speed_mps = 2.0
TARGET_SPEED_MPS = 1.8

TARGET_STEP_TIME = 0.1
BUBBLE_RADIUS = 0.5


# ================= Helper functions =================

def world_to_chunk(pos):
    x, y = pos
    col = int(x / chunk_size)
    row = int(y / chunk_size)
    return row, col


def chunk_to_world(chunk):
    row, col = chunk
    x = col * chunk_size + chunk_size / 2
    y = row * chunk_size + chunk_size / 2
    return [x, y]


def is_valid_chunk(chunk):
    row, col = chunk

    if row < 0 or row >= rows:
        return False
    if col < 0 or col >= cols:
        return False

    return chunk_map[row, col] == FREE


def distance_between(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def direct_path_clear(start_pos, end_pos):
    start = np.array(start_pos)
    end = np.array(end_pos)

    path_length = distance_between(start_pos, end_pos)
    steps = int(path_length / chunk_size)

    for i in range(steps + 1):
        t = i / max(steps, 1)
        point = start + t * (end - start)

        if not is_valid_chunk(world_to_chunk(point)):
            return False

    return True


def get_bubble_chase_point(robot_pos, target_pos, bubble_radius=BUBBLE_RADIUS):
    robot = np.array(robot_pos, dtype=float)
    target = np.array(target_pos, dtype=float)

    direction = robot - target
    distance = np.linalg.norm(direction)

    if distance <= bubble_radius:
        return robot_pos

    direction = direction / distance

    chase_point = target + direction * bubble_radius
    chase_point = [chase_point[0], chase_point[1]]

    if is_valid_chunk(world_to_chunk(chase_point)):
        return chase_point

    best_point = target_pos
    best_dist = float("inf")

    for angle in np.linspace(0, 2 * np.pi, 72):
        candidate = [
            target[0] + bubble_radius * np.cos(angle),
            target[1] + bubble_radius * np.sin(angle),
        ]

        if is_valid_chunk(world_to_chunk(candidate)):
            d = distance_between(robot_pos, candidate)

            if d < best_dist:
                best_dist = d
                best_point = candidate

    return best_point


# ================= A* pathfinding =================

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(chunk):
    row, col = chunk

    possible = [
        (row + 1, col),
        (row - 1, col),
        (row, col + 1),
        (row, col - 1),
    ]

    return [c for c in possible if is_valid_chunk(c)]


def astar(start_chunk, goal_chunk):
    open_set = []
    heapq.heappush(open_set, (0, start_chunk))

    came_from = {}
    g_score = {start_chunk: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal_chunk:
            path = [current]

            while current in came_from:
                current = came_from[current]
                path.append(current)

            path.reverse()
            return path

        for neighbor in get_neighbors(current):
            new_score = g_score[current] + 1

            if neighbor not in g_score or new_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = new_score

                f_score = new_score + heuristic(neighbor, goal_chunk)
                heapq.heappush(open_set, (f_score, neighbor))

    return None


def build_route(start_pos, goal_pos):
    start_chunk = world_to_chunk(start_pos)
    goal_chunk = world_to_chunk(goal_pos)

    if not is_valid_chunk(start_chunk):
        return [start_pos]

    if not is_valid_chunk(goal_chunk):
        return [start_pos]

    if direct_path_clear(start_pos, goal_pos):
        return [start_pos, goal_pos]

    chunk_path = astar(start_chunk, goal_chunk)

    if chunk_path is None:
        return [start_pos]

    return [chunk_to_world(c) for c in chunk_path]


# ================= Movement functions =================

def move_along_route(current_pos, route, speed_mps, dt):
    if len(route) < 2:
        return current_pos, route

    current = np.array(current_pos, dtype=float)
    max_step = speed_mps * dt

    while len(route) >= 2 and max_step > 0:
        next_point = np.array(route[1], dtype=float)
        direction = next_point - current
        distance = np.linalg.norm(direction)

        if distance < 0.001:
            route = route[1:]
            continue

        if max_step >= distance:
            current = next_point
            max_step -= distance
            route = route[1:]
        else:
            direction = direction / distance
            current = current + direction * max_step
            max_step = 0

    return [current[0], current[1]], route


def random_valid_position():
    while True:
        row = random.randint(0, rows - 1)
        col = random.randint(0, cols - 1)

        if is_valid_chunk((row, col)):
            return chunk_to_world((row, col))


# ================= Initialize target =================

target_pos = random_valid_position()
target_goal = random_valid_position()
target_route = build_route(target_pos, target_goal)

if not is_valid_chunk(world_to_chunk(robot_pos)):
    raise ValueError("Robot starts in blocked space.")


# ================= Plot setup =================

fig, ax = plt.subplots(figsize=(6, 6))

ax.imshow(
    chunk_map,
    origin="lower",
    extent=[0, map_size[0], 0, map_size[1]],
    interpolation="nearest",
)

robot_dot, = ax.plot([], [], "o", markersize=9, label="Robot")
target_dot, = ax.plot([], [], "x", markersize=9, label="Moving Target")
target_goal_dot, = ax.plot([], [], "*", markersize=8, label="Target Goal")
robot_goal_dot, = ax.plot([], [], ".", markersize=10, label="Robot Bubble Goal")

robot_route_line, = ax.plot([], [], linewidth=2, label="Robot Route")
target_route_line, = ax.plot([], [], linewidth=1, linestyle="--", label="Target Route")

bubble_circle = plt.Circle(
    target_pos,
    BUBBLE_RADIUS,
    fill=False,
    linestyle=":",
    linewidth=1.5,
)

ax.add_patch(bubble_circle)

ax.set_xlim(0, map_size[0])
ax.set_ylim(0, map_size[1])
ax.set_xlabel("x position, meters")
ax.set_ylabel("y position, meters")
ax.set_title("Robot Chasing 0.5 m Bubble Around Moving Target")
ax.legend(loc="upper right")


# ================= Animation update =================

def update(frame):
    global robot_pos, target_pos, target_goal, target_route

    if distance_between(target_pos, target_goal) < 0.15 or len(target_route) < 2:
        target_goal = random_valid_position()
        target_route = build_route(target_pos, target_goal)

    if frame % 10 == 0:
        target_route = build_route(target_pos, target_goal)

    target_pos, target_route = move_along_route(
        target_pos,
        target_route,
        TARGET_SPEED_MPS,
        TARGET_STEP_TIME,
    )

    robot_goal = get_bubble_chase_point(
        robot_pos,
        target_pos,
        bubble_radius=BUBBLE_RADIUS,
    )

    robot_route = build_route(robot_pos, robot_goal)

    robot_pos, robot_route = move_along_route(
        robot_pos,
        robot_route,
        robot_speed_mps,
        TARGET_STEP_TIME,
    )

    robot_goal = get_bubble_chase_point(
        robot_pos,
        target_pos,
        bubble_radius=BUBBLE_RADIUS,
    )

    robot_route = build_route(robot_pos, robot_goal)

    robot_route_x = [p[0] for p in robot_route]
    robot_route_y = [p[1] for p in robot_route]

    target_route_x = [p[0] for p in target_route]
    target_route_y = [p[1] for p in target_route]

    robot_dot.set_data([robot_pos[0]], [robot_pos[1]])
    target_dot.set_data([target_pos[0]], [target_pos[1]])
    target_goal_dot.set_data([target_goal[0]], [target_goal[1]])
    robot_goal_dot.set_data([robot_goal[0]], [robot_goal[1]])

    robot_route_line.set_data(robot_route_x, robot_route_y)
    target_route_line.set_data(target_route_x, target_route_y)

    bubble_circle.center = target_pos

    return (
        robot_dot,
        target_dot,
        target_goal_dot,
        robot_goal_dot,
        robot_route_line,
        target_route_line,
        bubble_circle,
    )


# ================= Save animation =================

ani = FuncAnimation(
    fig,
    update,
    frames=300,
    interval=int(TARGET_STEP_TIME * 1000),
    blit=True,
)

ani.save(
    "pathfinding_sim.mp4",
    writer="ffmpeg",
    fps=10,
    dpi=60,
)

Video("pathfinding_sim.mp4", embed=True)
