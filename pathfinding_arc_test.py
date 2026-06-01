"""
Closed-loop test of Pathfinding_algorithm.tick() chasing a target that
moves along a circular arc. Drives the real motors via the ESP32 by
default; pass --no-drive for pure simulation (e.g. on the Mac).

For each tick:
  1. Place the target at its scripted position on the arc
  2. Call tick() with the cart's current simulated pose
  3. Send (v_left, v_right) to the ESP32 over USB serial
  4. Integrate differential-drive kinematics in software to advance the
     simulated pose for the next tick (the cart has no closed-loop
     odometry yet, so the simulated pose is what tick() sees)

Outputs (always written, even when motors are driven):
  pathfinding_arc_test.png             — 4-panel diagnostic figure
  pathfinding_arc_test_trajectory.png  — trajectory only

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

ARC_CENTRE   = (2.5, 3.0)
ARC_RADIUS   = 0.25
ARC_OMEGA    = 0.3           # rad/s — how fast the target circles (0.3 ≈ 17°/s)
TOTAL_S      = 25.0

# Cart starting pose — picked to be in free space and roughly facing the arc
START_POS     = [2.5, 1.5]
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
    states         = []
    in_fovs        = []
    enc_left_cum, enc_right_cum = [], []
    enc_left_delta, enc_right_delta = [], []
    cum_l, cum_r = 0, 0

    n_steps = int(TOTAL_S / P.DT)
    try:
      for i in range(n_steps):
        t_loop0 = time.monotonic()
        t   = i * P.DT
        tgt = target_at(t)

        # tick() is the same call the live main loop makes after converting
        # a UDP "<dist>,<angle>" packet to absolute coords.
        v_left, v_right = P.tick(tgt, P.robot_pos, P.robot_heading)

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

        # Real-time pacing when driving real motors — otherwise the cart
        # would receive 250 commands in a fraction of a second.
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
        state=states, in_fov=in_fovs,
        enc_l_cum=enc_left_cum, enc_r_cum=enc_right_cum,
        enc_l_delta=enc_left_delta, enc_r_delta=enc_right_delta,
        drove=(motors is not None),
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

    # Standalone trajectory plot ────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    arc_circle = plt.Circle(ARC_CENTRE, ARC_RADIUS,
                             color='crimson', fill=False, ls=':', alpha=0.4,
                             label='target arc')
    ax2.add_patch(arc_circle)
    ax2.plot(data['tx'], data['ty'], 'r--', lw=1.2, label='target path')
    ax2.plot(data['rx'], data['ry'], 'b-',  lw=2.0, label='cart path')
    ax2.scatter([data['rx'][0]], [data['ry'][0]], c='blue',  marker='o', s=90,
                label='cart start', zorder=5)
    ax2.scatter([data['rx'][-1]], [data['ry'][-1]], c='blue', marker='s', s=90,
                label='cart end',   zorder=5)
    ax2.scatter([data['tx'][0]], [data['ty'][0]], c='red',   marker='x', s=90,
                label='target start', zorder=5)

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
        f'Cart trajectory  (cart cap {P.ROBOT_SPEED_MPS} m/s,  '
        f'target {ARC_OMEGA*ARC_RADIUS:.2f} m/s on arc r={ARC_RADIUS}m)'
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
    in_view_pct = 100.0 * sum(data['in_fov']) / len(data['in_fov'])
    print(f"\nSummary over {TOTAL_S:.1f} s, {len(data['t'])} ticks:")
    print(f"  target in FOV         : {in_view_pct:.1f}% of ticks")
    print(f"  mean distance to tgt  : {np.mean(distances):.2f} m")
    print(f"  max  distance to tgt  : {np.max(distances):.2f} m")
    print(f"  mean v_forward (when commanded) : "
          f"{np.mean([v for v in data['v_fwd'] if v > 0.01]):.2f} m/s")
    print(f"  max  |omega|          : {math.degrees(max(abs(w) for w in data['omega'])):.1f} deg/s")
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
