"""
Closed-loop test of Pathfinding_algorithm.tick() chasing a target that
moves along a circular arc.

For each simulated tick we:
  1. Place the target at its scripted position on the arc
  2. Call tick() with the current robot pose — same call the live main
     loop makes after converting a UDP packet to absolute coords
  3. Take the returned (v_left, v_right), integrate differential-drive
     kinematics, and update the robot's pose
  4. Log everything for the plot

Output: pathfinding_arc_test.png

To tune the test, edit the CONFIG block below.

Usage:
    pip install matplotlib numpy
    python3 pathfinding_arc_test.py
"""

import math
import numpy as np
import matplotlib.pyplot as plt

import Pathfinding_algorithm as P


# =============================================================================
# CONFIG  — tweak these to explore different test cases
# =============================================================================

ARC_CENTRE   = (2.5, 3.0)
ARC_RADIUS   = 0.8
ARC_OMEGA    = 0.3           # rad/s — how fast the target circles (0.3 ≈ 17°/s)
TOTAL_S      = 25.0

# Cart starting pose — picked to be in free space and roughly facing the arc
START_POS     = [2.5, 0.5]
START_HEADING = math.radians(90)   # facing +y toward the arc

# Make the whole 20x20 map free space so the FOV / line-of-sight check
# never spuriously blocks the chase.  We're testing the controller, not
# obstacle avoidance.
P.chunk_map = np.ones_like(P.chunk_map)

# =============================================================================


def target_at(t):
    """Where the target is at time t (anti-clockwise from +x of ARC_CENTRE)."""
    theta = ARC_OMEGA * t
    return [ARC_CENTRE[0] + ARC_RADIUS * math.cos(theta),
            ARC_CENTRE[1] + ARC_RADIUS * math.sin(theta)]


def integrate_kinematics(pos, heading, v_left, v_right, dt):
    """Differential-drive forward integration (Euler at start-of-step heading)."""
    v_fwd = (v_left + v_right) / 2.0
    omega = (v_right - v_left) / P.WHEEL_TRACK_M
    new_heading = (heading + omega * dt) % (2 * math.pi)
    new_x = pos[0] + v_fwd * math.cos(heading) * dt
    new_y = pos[1] + v_fwd * math.sin(heading) * dt
    return [new_x, new_y], new_heading, v_fwd, omega


def run():
    P.robot_pos     = list(START_POS)
    P.robot_heading = START_HEADING

    times          = []
    target_xs, target_ys = [], []
    robot_xs, robot_ys, robot_thetas = [], [], []
    v_fwds, omegas, v_lefts, v_rights = [], [], [], []
    states         = []
    in_fovs        = []

    n_steps = int(TOTAL_S / P.DT)
    for i in range(n_steps):
        t   = i * P.DT
        tgt = target_at(t)

        # tick() is the same call the live main loop makes after converting
        # a UDP "<dist>,<angle>" packet to absolute coords.
        v_left, v_right = P.tick(tgt, P.robot_pos, P.robot_heading)

        # Record what was commanded for this state
        times.append(t)
        target_xs.append(tgt[0]); target_ys.append(tgt[1])
        robot_xs.append(P.robot_pos[0]); robot_ys.append(P.robot_pos[1])
        robot_thetas.append(P.robot_heading)
        v_lefts.append(v_left); v_rights.append(v_right)
        states.append(P.robot_state)
        in_fovs.append(P.target_visible)

        # Integrate kinematics
        new_pos, new_heading, v_fwd, omega = integrate_kinematics(
            P.robot_pos, P.robot_heading, v_left, v_right, P.DT)
        P.robot_pos     = new_pos
        P.robot_heading = new_heading
        v_fwds.append(v_fwd); omegas.append(omega)

    return dict(
        t=times,
        tx=target_xs, ty=target_ys,
        rx=robot_xs, ry=robot_ys, rt=robot_thetas,
        v_fwd=v_fwds, omega=omegas,
        v_left=v_lefts, v_right=v_rights,
        state=states, in_fov=in_fovs,
    )


def plot(data, out_path="pathfinding_arc_test.png"):
    fig = plt.figure(figsize=(13, 9))
    gs  = fig.add_gridspec(3, 2, width_ratios=[1.1, 1])

    # ── Trajectory ─────────────────────────────────────────
    ax = fig.add_subplot(gs[:, 0])
    arc_circle = plt.Circle(ARC_CENTRE, ARC_RADIUS,
                             color='crimson', fill=False, ls=':', alpha=0.4,
                             label='target arc')
    ax.add_patch(arc_circle)
    ax.plot(data['tx'], data['ty'], 'r--', lw=1, label='target path')
    ax.plot(data['rx'], data['ry'], 'b-',  lw=1.5, label='robot path')
    ax.scatter([data['rx'][0]], [data['ry'][0]], c='blue',  marker='o', s=70,
               label='robot start', zorder=5)
    ax.scatter([data['tx'][0]], [data['ty'][0]], c='red',   marker='x', s=70,
               label='target start', zorder=5)
    ax.scatter([data['rx'][-1]], [data['ry'][-1]], c='blue', marker='s', s=70,
               label='robot end',   zorder=5)

    # Heading arrows every ~2s
    every = max(1, int(2.0 / P.DT))
    for i in range(0, len(data['t']), every):
        x, y, th = data['rx'][i], data['ry'][i], data['rt'][i]
        ax.annotate('', xy=(x + 0.4*math.cos(th), y + 0.4*math.sin(th)),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color='royalblue',
                                    lw=1.2, alpha=0.5))

    ax.set_aspect('equal')
    pad = 2
    xs = data['rx'] + data['tx']; ys = data['ry'] + data['ty']
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title(f'Trajectory  (arc ω={ARC_OMEGA} rad/s, r={ARC_RADIUS}m)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Forward speed ──────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(data['t'], data['v_fwd'], 'b-')
    ax.axhline(P.ROBOT_SPEED_MPS, color='gray', ls=':', alpha=0.5,
               label=f'cap {P.ROBOT_SPEED_MPS} m/s')
    ax.set_ylabel('v_forward (m/s)')
    ax.set_title('Forward speed command')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ── Turn rate ──────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(data['t'], np.degrees(data['omega']), 'b-')
    ax.axhline( math.degrees(P.TURN_SPEED_RAD), color='gray', ls=':', alpha=0.5,
                label=f'cap ±{math.degrees(P.TURN_SPEED_RAD):.0f}°/s')
    ax.axhline(-math.degrees(P.TURN_SPEED_RAD), color='gray', ls=':', alpha=0.5)
    ax.set_ylabel('omega (deg/s)')
    ax.set_title('Turn rate command')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ── Distance to target ─────────────────────────────────
    distances = [math.hypot(tx - rx, ty - ry)
                 for tx, ty, rx, ry in zip(data['tx'], data['ty'],
                                            data['rx'], data['ry'])]
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(data['t'], distances, 'b-')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('distance to target (m)')
    ax.set_title('Tracking error')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    print(f"Saved: {out_path}")

    # Numerical summary
    in_view_pct = 100.0 * sum(data['in_fov']) / len(data['in_fov'])
    print(f"\nSummary over {TOTAL_S:.1f} s, {len(data['t'])} ticks:")
    print(f"  target in FOV         : {in_view_pct:.1f}% of ticks")
    print(f"  mean distance to tgt  : {np.mean(distances):.2f} m")
    print(f"  max  distance to tgt  : {np.max(distances):.2f} m")
    print(f"  mean v_forward (when commanded) : "
          f"{np.mean([v for v in data['v_fwd'] if v > 0.01]):.2f} m/s")
    print(f"  max  |omega|          : {math.degrees(max(abs(w) for w in data['omega'])):.1f} deg/s")


if __name__ == "__main__":
    data = run()
    plot(data)
