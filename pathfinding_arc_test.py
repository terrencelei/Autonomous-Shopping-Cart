"""
Closed-loop test of Pathfinding_algorithm.tick() following a scripted
subject that walks one lap of a 0.5 m square box.

The subject's absolute position is converted to a camera-frame
(dist_m, angle_deg) reading each tick — matching exactly what
yolo_detect.py streams from the real camera — so tick() exercises the
same code path it uses in production.

Usage:
    pip install matplotlib numpy pyserial
    python3 pathfinding_arc_test.py            # drives motors
    python3 pathfinding_arc_test.py --no-drive # pure sim, no serial
    python3 pathfinding_arc_test.py --port /dev/ttyUSB0
"""

import argparse
import math
import os
import time

import numpy as np
import matplotlib.pyplot as plt

import Pathfinding_algorithm as P


# =============================================================================
# CONFIG  — tweak these to explore different test cases
# =============================================================================

BOX_CENTRE  = (2.5, 3.0)
BOX_SIDE    = 0.5    # metres — length of each side
BOX_SPEED   = 0.15   # m/s — subject walking speed along the perimeter
TOTAL_S     = 4 * BOX_SIDE / BOX_SPEED   # exactly one full lap

# Cart starting pose — far enough from the box that the cart must approach
START_POS     = [2.5, 0.5]
START_HEADING = math.radians(90)   # facing +y toward the box

# Make the whole 20x20 map free space so the FOV / line-of-sight check
# never spuriously blocks the chase.  We're testing the controller, not
# obstacle avoidance.
P.chunk_map = np.ones_like(P.chunk_map)

# =============================================================================


def box_corners():
    """Counter-clockwise corners of the box starting from bottom-left."""
    half = BOX_SIDE / 2.0
    cx, cy = BOX_CENTRE
    return [
        [cx - half, cy - half],   # C0 bottom-left
        [cx + half, cy - half],   # C1 bottom-right
        [cx + half, cy + half],   # C2 top-right
        [cx - half, cy + half],   # C3 top-left
    ]


def target_at(t):
    """Subject's absolute position at time t, walking the box CCW at BOX_SPEED."""
    perimeter = 4 * BOX_SIDE
    dist = (BOX_SPEED * t) % perimeter
    half = BOX_SIDE / 2.0
    cx, cy = BOX_CENTRE
    if dist < BOX_SIDE:
        return [cx - half + dist, cy - half]
    elif dist < 2 * BOX_SIDE:
        return [cx + half, cy - half + (dist - BOX_SIDE)]
    elif dist < 3 * BOX_SIDE:
        return [cx + half - (dist - 2 * BOX_SIDE), cy + half]
    else:
        return [cx - half, cy + half - (dist - 3 * BOX_SIDE)]


def absolute_to_reading(target_pos, robot_pos, robot_heading):
    """Convert absolute target position to (dist_m, angle_deg) camera reading.

    Inverse of Pathfinding_algorithm.relative_to_absolute().
    angle_deg: 0 = straight ahead, positive = target to the right.
    """
    dx = target_pos[0] - robot_pos[0]
    dy = target_pos[1] - robot_pos[1]
    dist_m = math.hypot(dx, dy)
    if dist_m < 1e-6:
        return 0.0, 0.0
    bearing_world = math.atan2(dy, dx)
    angle_deg = math.degrees(robot_heading - bearing_world)
    return dist_m, angle_deg


def integrate_kinematics(pos, heading, v_left, v_right, dt):
    """Differential-drive forward integration (Euler at start-of-step heading)."""
    v_fwd = (v_left + v_right) / 2.0
    omega = (v_right - v_left) / P.WHEEL_TRACK_M
    new_heading = (heading + omega * dt) % (2 * math.pi)
    new_x = pos[0] + v_fwd * math.cos(heading) * dt
    new_y = pos[1] + v_fwd * math.sin(heading) * dt
    return [new_x, new_y], new_heading, v_fwd, omega


def find_serial_port(preferred):
    """Return the first existing /dev/tty* candidate, preferring the named one."""
    candidates = [preferred, '/dev/ttyACM0', '/dev/ttyUSB0',
                  '/dev/ttyACM1', '/dev/ttyUSB1']
    seen = set()
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        if os.path.exists(c):
            return c
    return preferred


def run(drive=True, port=None, countdown=3):
    P.robot_pos     = list(START_POS)
    P.robot_heading = START_HEADING

    motors = None
    if drive:
        chosen = port or find_serial_port(P.MOTOR_UART_PORT)
        if chosen != P.MOTOR_UART_PORT:
            print(f"Note: using {chosen} "
                  f"(Pathfinding_algorithm.MOTOR_UART_PORT = {P.MOTOR_UART_PORT})")
        motors = P.MotorDriver(chosen, P.MOTOR_UART_BAUD)
        if countdown > 0:
            print(f"\n*** Cart will start driving in {countdown}s ***")
            for k in range(countdown, 0, -1):
                print(f"  {k}...")
                time.sleep(1)
            print("  GO!\n")

    times          = []
    target_xs, target_ys = [], []
    robot_xs, robot_ys, robot_thetas = [], [], []
    v_fwds, omegas, v_lefts, v_rights = [], [], [], []
    dists          = []
    enc_left_cum, enc_right_cum = [], []
    enc_left_delta, enc_right_delta = [], []
    cum_l, cum_r = 0, 0

    n_steps = int(TOTAL_S / P.DT)
    try:
      for i in range(n_steps):
        t_loop0 = time.monotonic()
        t   = i * P.DT

        # Subject walks the box; convert to camera-frame reading for tick()
        tgt     = target_at(t)
        reading = absolute_to_reading(tgt, P.robot_pos, P.robot_heading)

        v_left, v_right = P.tick(reading, P.robot_pos, P.robot_heading)

        if motors is not None:
            motors.send_velocities(v_left, v_right)
            d_l, d_r = motors.read_encoder_deltas()
        else:
            d_l, d_r = 0, 0
        cum_l += d_l
        cum_r += d_r
        enc_left_delta.append(d_l)
        enc_right_delta.append(d_r)
        enc_left_cum.append(cum_l)
        enc_right_cum.append(cum_r)

        times.append(t)
        target_xs.append(tgt[0]); target_ys.append(tgt[1])
        robot_xs.append(P.robot_pos[0]); robot_ys.append(P.robot_pos[1])
        robot_thetas.append(P.robot_heading)
        v_lefts.append(v_left); v_rights.append(v_right)
        dists.append(reading[0])

        new_pos, new_heading, v_fwd, omega = integrate_kinematics(
            P.robot_pos, P.robot_heading, v_left, v_right, P.DT)
        P.robot_pos     = new_pos
        P.robot_heading = new_heading
        v_fwds.append(v_fwd); omegas.append(omega)

        if motors is not None:
            elapsed = time.monotonic() - t_loop0
            if elapsed < P.DT:
                time.sleep(P.DT - elapsed)
    finally:
        if motors is not None:
            print("\nStopping motors.")
            motors.stop()

    return dict(
        t=times,
        tx=target_xs, ty=target_ys,
        rx=robot_xs, ry=robot_ys, rt=robot_thetas,
        v_fwd=v_fwds, omega=omegas,
        v_left=v_lefts, v_right=v_rights,
        dist=dists,
        enc_l_cum=enc_left_cum, enc_r_cum=enc_right_cum,
        enc_l_delta=enc_left_delta, enc_r_delta=enc_right_delta,
        drove=(motors is not None),
    )


def plot(data, out_path="pathfinding_arc_test.png"):
    fig = plt.figure(figsize=(13, 9))
    gs  = fig.add_gridspec(3, 2, width_ratios=[1.1, 1])

    # ── Trajectory ─────────────────────────────────────────
    ax = fig.add_subplot(gs[:, 0])
    corners = box_corners()
    box_xs = [c[0] for c in corners] + [corners[0][0]]
    box_ys = [c[1] for c in corners] + [corners[0][1]]
    ax.plot(box_xs, box_ys, 'r:', lw=1.5, alpha=0.6, label='subject path')
    ax.plot(data['tx'], data['ty'], 'r-', lw=1.0, alpha=0.35)
    ax.plot(data['rx'], data['ry'], 'b-',  lw=1.5, label='cart path')
    ax.scatter([data['rx'][0]], [data['ry'][0]], c='blue',  marker='o', s=70,
               label='cart start', zorder=5)
    ax.scatter([data['tx'][0]], [data['ty'][0]], c='crimson', marker='x', s=70,
               label='subject start', zorder=5)
    ax.scatter([data['rx'][-1]], [data['ry'][-1]], c='blue', marker='s', s=70,
               label='cart end',   zorder=5)

    # Heading arrows every ~2s
    every = max(1, int(2.0 / P.DT))
    for i in range(0, len(data['t']), every):
        x, y, th = data['rx'][i], data['ry'][i], data['rt'][i]
        ax.annotate('', xy=(x + 0.4*math.cos(th), y + 0.4*math.sin(th)),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color='royalblue',
                                    lw=1.2, alpha=0.5))

    ax.set_aspect('equal')
    pad = 1.0
    xs = data['rx'] + data['tx']; ys = data['ry'] + data['ty']
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title(f'Trajectory  (subject {BOX_SPEED} m/s, hold {P.HOLD_DIST_M} m)')
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

    # ── Distance to subject ────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(data['t'], data['dist'], 'b-')
    ax.axhline(P.HOLD_DIST_M, color='gray', ls=':', alpha=0.6,
               label=f'hold dist {P.HOLD_DIST_M} m')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('distance to subject (m)')
    ax.set_title('Cart–subject distance')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    print(f"Saved: {out_path}")

    # Standalone trajectory plot ────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    corners = box_corners()
    box_xs = [c[0] for c in corners] + [corners[0][0]]
    box_ys = [c[1] for c in corners] + [corners[0][1]]
    ax2.plot(box_xs, box_ys, 'r:', lw=2.0, alpha=0.6, label='subject path')
    ax2.plot(data['tx'], data['ty'], 'r-', lw=1.0, alpha=0.4)
    for idx, (cx, cy) in enumerate(corners):
        ax2.scatter([cx], [cy], c='crimson', marker='D', s=60, zorder=4)
        ax2.annotate(f'C{idx}', (cx, cy), textcoords='offset points',
                     xytext=(5, 5), fontsize=8, color='crimson')
    ax2.plot(data['rx'], data['ry'], 'b-',  lw=2.0, label='cart path')
    ax2.scatter([data['rx'][0]], [data['ry'][0]], c='blue',  marker='o', s=90,
                label='cart start', zorder=5)
    ax2.scatter([data['rx'][-1]], [data['ry'][-1]], c='blue', marker='s', s=90,
                label='cart end',   zorder=5)

    every = max(1, int(1.0 / P.DT))   # 1 s heading samples
    for i in range(0, len(data['t']), every):
        x, y, th = data['rx'][i], data['ry'][i], data['rt'][i]
        ax2.annotate('', xy=(x + 0.3*math.cos(th), y + 0.3*math.sin(th)),
                     xytext=(x, y),
                     arrowprops=dict(arrowstyle='->', color='royalblue',
                                     lw=1.2, alpha=0.55))

    ax2.set_aspect('equal')
    pad = 1.0
    xs = data['rx'] + data['tx']; ys = data['ry'] + data['ty']
    ax2.set_xlim(min(xs) - pad, max(xs) + pad)
    ax2.set_ylim(min(ys) - pad, max(ys) + pad)
    ax2.set_xlabel('x (m)'); ax2.set_ylabel('y (m)')
    ax2.set_title(
        f'Subject walks {BOX_SIDE}m box @ {BOX_SPEED} m/s  —  '
        f'cart follows (hold {P.HOLD_DIST_M} m)'
    )
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    traj_path = out_path.replace('.png', '_trajectory.png')
    plt.tight_layout()
    plt.savefig(traj_path, dpi=110, bbox_inches='tight')
    print(f"Saved: {traj_path}")

    # ── Motor / encoder plot ───────────────────────────────
    if data.get('drove'):
        cmd_rpm_l = [P.wheel_speed_to_rpm(v) for v in data['v_left']]
        cmd_rpm_r = [P.wheel_speed_to_rpm(v) for v in data['v_right']]
        # Measured RPM from this tick's encoder delta
        meas_rpm_l = [d / P.DT / P.ENCODER_PPR * 60.0
                      for d in data['enc_l_delta']]
        meas_rpm_r = [d / P.DT / P.ENCODER_PPR * 60.0
                      for d in data['enc_r_delta']]

        fig3, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax_cum, ax_rpm = axes

        ax_cum.plot(data['t'], data['enc_l_cum'], 'b-', label='left encoder')
        ax_cum.plot(data['t'], data['enc_r_cum'], 'r-', label='right encoder')
        ax_cum.set_ylabel('cumulative ticks')
        ax_cum.set_title('Encoder counts')
        ax_cum.grid(True, alpha=0.3)
        ax_cum.legend(fontsize=9)

        ax_rpm.plot(data['t'], cmd_rpm_l,  'b--', lw=1.2, label='left commanded')
        ax_rpm.plot(data['t'], meas_rpm_l, 'b-',  lw=1.2, label='left measured')
        ax_rpm.plot(data['t'], cmd_rpm_r,  'r--', lw=1.2, label='right commanded')
        ax_rpm.plot(data['t'], meas_rpm_r, 'r-',  lw=1.2, label='right measured')
        ax_rpm.set_xlabel('time (s)')
        ax_rpm.set_ylabel('RPM')
        ax_rpm.set_title('Commanded vs measured wheel RPM')
        ax_rpm.grid(True, alpha=0.3)
        ax_rpm.legend(fontsize=8, ncol=2)

        motors_path = out_path.replace('.png', '_motors.png')
        plt.tight_layout()
        plt.savefig(motors_path, dpi=110, bbox_inches='tight')
        print(f"Saved: {motors_path}")

    # Numerical summary
    print(f"\nSummary over {TOTAL_S:.1f} s, {len(data['t'])} ticks "
          f"(subject: one {BOX_SIDE}m lap @ {BOX_SPEED} m/s):")
    print(f"  mean cart–subject distance   : {np.mean(data['dist']):.2f} m")
    print(f"  min  cart–subject distance   : {np.min(data['dist']):.2f} m")
    print(f"  hold target distance         : {P.HOLD_DIST_M:.2f} m")
    print(f"  mean v_forward (when moving) : "
          f"{np.mean([v for v in data['v_fwd'] if abs(v) > 0.01]):.2f} m/s")
    print(f"  max  |omega|                 : "
          f"{math.degrees(max(abs(w) for w in data['omega'])):.1f} deg/s")
    if data.get('drove'):
        rev_l = data['enc_l_cum'][-1] / P.ENCODER_PPR
        rev_r = data['enc_r_cum'][-1] / P.ENCODER_PPR
        print(f"  left wheel revolutions : {rev_l:+.2f}  ({data['enc_l_cum'][-1]:+d} ticks)")
        print(f"  right wheel revolutions: {rev_r:+.2f}  ({data['enc_r_cum'][-1]:+d} ticks)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-drive", action="store_true",
                        help="pure simulation — don't open the ESP32 serial port")
    parser.add_argument("--port", default=None,
                        help=f"serial port override "
                             f"(default: {P.MOTOR_UART_PORT}, auto-falls back to /dev/ttyUSB0)")
    parser.add_argument("--countdown", type=int, default=3,
                        help="seconds to wait before commanding motors (default: 3)")
    args = parser.parse_args()

    data = run(drive=not args.no_drive,
               port=args.port,
               countdown=args.countdown)
    plot(data)
