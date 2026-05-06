import matplotlib
matplotlib.use("Agg")

import numpy as np
import heapq
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ================= Robot / map settings =================

chunk_size = 0.1
map_size = [20, 20]

robot_pos = [0.5, 0.5]

FREE = 1
BLOCKED = 0

# ================= Moving target settings =================

MAX_SPEED_MPH = 4
MAX_SPEED_MPS = MAX_SPEED_MPH * 0.44704

TARGET_STEP_TIME = 0.1
MAX_TARGET_STEP = MAX_SPEED_MPS * TARGET_STEP_TIME

target_pos = [3.0, 1.5]

# ================= Generate chunk map =================

aisle_width = 1
aisle_amount = 10
gap_width = (map_size[1] - aisle_width * aisle_amount) / (aisle_amount - 1)

cols = int(map_size[0] / chunk_size)
rows = int(map_size[1] / chunk_size)

chunk_map = np.zeros((rows, cols), dtype=int)

for row in range(rows):
    for col in range(cols):
        y = row * chunk_size

        if y < aisle_width:
            chunk_map[row, col] = FREE
        elif y > map_size[1] - aisle_width:
            chunk_map[row, col] = FREE
        elif (y % (aisle_width + gap_width)) < aisle_width:
            chunk_map[row, col] = FREE

# ================= Helpers =================

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
    if row < 0 or row >= rows or col < 0 or col >= cols:
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

# ================= Movement simulation =================

def move_along_route(current_pos, route, speed_mps, dt):
    if len(route) < 2:
        return current_pos

    target = np.array(route[1])
    current = np.array(current_pos)

    direction = target - current
    distance = np.linalg.norm(direction)

    if distance < 0.001:
        return route[1]

    max_step = speed_mps * dt

    if max_step >= distance:
        return route[1]

    direction = direction / distance
    new_pos = current + direction * max_step

    return [new_pos[0], new_pos[1]]


def random_valid_target_move(current_pos):
    for _ in range(100):
        angle = random.uniform(0, 2 * np.pi)
        distance = random.uniform(0, MAX_TARGET_STEP)

        new_pos = [
            current_pos[0] + distance * np.cos(angle),
            current_pos[1] + distance * np.sin(angle),
        ]

        if is_valid_chunk(world_to_chunk(new_pos)):
            return new_pos

    return current_pos

# ================= Validate starting positions =================

if not is_valid_chunk(world_to_chunk(robot_pos)):
    raise ValueError("Robot starts in blocked space.")

if not is_valid_chunk(world_to_chunk(target_pos)):
    raise ValueError("Target starts in blocked space.")

# ================= Plot setup =================

fig, ax = plt.subplots(figsize=(8, 8))

ax.imshow(
    chunk_map,
    origin="lower",
    extent=[0, map_size[0], 0, map_size[1]],
    interpolation="nearest",
)

robot_dot, = ax.plot([], [], "o", markersize=10, label="Robot")
target_dot, = ax.plot([], [], "x", markersize=10, label="Target")
route_line, = ax.plot([], [], linewidth=2, label="Route")

ax.set_xlim(0, map_size[0])
ax.set_ylim(0, map_size[1])
ax.set_xlabel("x position, meters")
ax.set_ylabel("y position, meters")
ax.set_title("Robot Pathfinding Simulation")
ax.legend()

robot_speed_mps = 1.0
route = build_route(robot_pos, target_pos)

# ================= Animation update =================

def update(frame):
    global robot_pos, target_pos, route

    target_pos = random_valid_target_move(target_pos)

    route = build_route(robot_pos, target_pos)

    robot_pos = move_along_route(
        robot_pos,
        route,
        robot_speed_mps,
        TARGET_STEP_TIME,
    )

    route = build_route(robot_pos, target_pos)

    route_x = [p[0] for p in route]
    route_y = [p[1] for p in route]

    robot_dot.set_data([robot_pos[0]], [robot_pos[1]])
    target_dot.set_data([target_pos[0]], [target_pos[1]])
    route_line.set_data(route_x, route_y)

    return robot_dot, target_dot, route_line

# ================= Save animation =================

ani = FuncAnimation(
    fig,
    update,
    frames=1000,
    interval=int(TARGET_STEP_TIME * 1000),
    blit=True,
)

print("Saving animation...")

ani.save(
    "pathfinding_sim.mp4",
    writer="ffmpeg",
    fps=int(1 / TARGET_STEP_TIME),
    dpi=150,
)

print("Saved as pathfinding_sim.mp4")
