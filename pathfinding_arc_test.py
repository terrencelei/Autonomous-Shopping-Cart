"""
Spin test — drives the cart one full 360° rotation in place, then stops.

Usage:
    python3 pathfinding_arc_test.py            # drives motors
    python3 pathfinding_arc_test.py --no-drive # pure sim, no serial
    python3 pathfinding_arc_test.py --port /dev/ttyUSB0
"""

import argparse
import math
import os
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — required on headless Pi
import matplotlib.pyplot as plt

import Pathfinding_algorithm as P

# Handle renamed constants between versions of Pathfinding_algorithm
_MOTOR_PORT = getattr(P, 'MOTOR_PORT', getattr(P, 'MOTOR_UART_PORT', '/dev/ttyACM0'))

# =============================================================================
# CONFIG
# =============================================================================

START_POS     = [0.0, 0.0]
START_HEADING = 0.0

# =============================================================================


def integrate_kinematics(pos, heading, v_left, v_right, dt):
    v_fwd = (v_left + v_right) / 2.0
    omega = (v_right - v_left) / P.TRACK_M
    new_heading = (heading + omega * dt) % (2 * math.pi)
    new_x = pos[0] + v_fwd * math.cos(heading) * dt
    new_y = pos[1] + v_fwd * math.sin(heading) * dt
    return [new_x, new_y], new_heading, v_fwd, omega


def find_serial_port(preferred):
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
    P.S.pos     = list(START_POS)
    P.S.heading = START_HEADING

    motors = None
    if drive:
        chosen = port or find_serial_port(_MOTOR_PORT)
        if chosen != _MOTOR_PORT:
            P.MOTOR_PORT = chosen
        motors = P.MotorDriver()
        if countdown > 0:
            print(f"\n*** Cart will start spinning in {countdown}s ***")
            for k in range(countdown, 0, -1):
                print(f"  {k}...")
                time.sleep(1)
            print("  GO!\n")

    times         = []
    robot_xs, robot_ys, robot_thetas = [], [], []
    v_fwds, omegas, v_lefts, v_rights = [], [], [], []
    enc_left_cum, enc_right_cum = [], []
    enc_left_delta, enc_right_delta = [], []
    cum_l, cum_r = 0, 0

    # Ticks one wheel travels during a 360° point turn:
    #   arc = π * TRACK_M  (half-circumference of the turn circle)
    #   ticks = arc / WHEEL_CIRC * ENCODER_PPR
    TICKS_360 = math.pi * P.TRACK_M / P.WHEEL_CIRC * P.ENCODER_PPR
    print(f"Target: {TICKS_360:.0f} left-encoder ticks for 360°  "
          f"(ENCODER_PPR={P.ENCODER_PPR})")

    abs_ticks_l = 0   # accumulated absolute left encoder ticks (real hardware)
    turned      = 0.0 # simulated radians (--no-drive fallback)

    # Hard-flush the serial receive buffer and reset encoder tracking
    # so packets that built up during the countdown don't count
    if motors is not None and motors._ser and motors._ser.is_open:
        motors._ser.reset_input_buffer()
        motors._last_l = motors._last_r = None
        motors._dl = motors._dr = 0

    i = 0
    try:
        while True:
            t_loop0 = time.monotonic()
            t = i * P.DT

            v_left, v_right = P._wheel_commands(0.0, P.MAX_TURN)

            if motors is not None:
                motors.send(v_left, v_right)
                d_l, d_r = motors.read_encoder_deltas()
            else:
                d_l, d_r = 0, 0
            cum_l += d_l; cum_r += d_r
            abs_ticks_l += abs(d_l)
            enc_left_delta.append(d_l);  enc_right_delta.append(d_r)
            enc_left_cum.append(cum_l);  enc_right_cum.append(cum_r)

            times.append(t)
            robot_xs.append(P.S.pos[0]); robot_ys.append(P.S.pos[1])
            robot_thetas.append(P.S.heading)

            new_pos, new_heading, v_fwd, omega = integrate_kinematics(
                P.S.pos, P.S.heading, v_left, v_right, P.DT)
            P.S.pos     = new_pos
            P.S.heading = new_heading
            v_fwds.append(v_fwd); omegas.append(omega)
            v_lefts.append(v_left); v_rights.append(v_right)

            turned += abs(omega) * P.DT

            # Stop condition: real encoder ticks when driving, simulated
            # radians when running --no-drive
            if motors is not None:
                if abs_ticks_l >= TICKS_360:
                    break
            else:
                if turned >= 2 * math.pi:
                    break

            i += 1

            if motors is not None:
                elapsed = time.monotonic() - t_loop0
                if elapsed < P.DT:
                    time.sleep(P.DT - elapsed)
    finally:
        if motors is not None:
            print("\nStopping motors.")
            motors.stop()

    elapsed_s = times[-1] if times else 0.0
    print(f"\nSpin complete: {math.degrees(turned):.1f}° sim  |  "
          f"{abs_ticks_l} left encoder ticks  |  "
          f"{elapsed_s:.1f} s  ({len(times)} ticks)")

    return dict(
        t=times,
        rx=robot_xs, ry=robot_ys, rt=robot_thetas,
        v_fwd=v_fwds, omega=omegas,
        v_left=v_lefts, v_right=v_rights,
        enc_l_cum=enc_left_cum, enc_r_cum=enc_right_cum,
        enc_l_delta=enc_left_delta, enc_r_delta=enc_right_delta,
        drove=(motors is not None),
    )


def plot(data, out_path="pathfinding_arc_test.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # ── Heading over time ──────────────────────────────────
    ax = axes[0]
    headings_deg = [math.degrees(h) for h in data['rt']]
    ax.plot(data['t'], headings_deg, 'b-')
    ax.axhline(360, color='gray', ls=':', alpha=0.6, label='360°')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('heading (deg)')
    ax.set_title('Heading over time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Turn rate ──────────────────────────────────────────
    ax = axes[1]
    ax.plot(data['t'], np.degrees(data['omega']), 'b-')
    ax.axhline( math.degrees(P.MAX_TURN), color='gray', ls=':', alpha=0.5,
                label=f'cap {math.degrees(P.MAX_TURN):.0f}°/s')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('omega (deg/s)')
    ax.set_title('Turn rate command')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    print(f"Saved: {out_path}")

    if data.get('drove'):
        fig2, axes2 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax_cum, ax_rpm = axes2

        ax_cum.plot(data['t'], data['enc_l_cum'], 'b-', label='left')
        ax_cum.plot(data['t'], data['enc_r_cum'], 'r-', label='right')
        ax_cum.set_ylabel('cumulative ticks')
        ax_cum.set_title('Encoder counts')
        ax_cum.legend(fontsize=9)
        ax_cum.grid(True, alpha=0.3)

        rpm_l = [v / P.WHEEL_CIRC * 60 for v in data['v_left']]
        rpm_r = [v / P.WHEEL_CIRC * 60 for v in data['v_right']]
        ax_rpm.plot(data['t'], rpm_l, 'b-', label='left commanded RPM')
        ax_rpm.plot(data['t'], rpm_r, 'r-', label='right commanded RPM')
        ax_rpm.set_xlabel('time (s)')
        ax_rpm.set_ylabel('RPM')
        ax_rpm.set_title('Wheel speed commands')
        ax_rpm.legend(fontsize=8)
        ax_rpm.grid(True, alpha=0.3)

        enc_path = out_path.replace('.png', '_encoders.png')
        plt.tight_layout()
        plt.savefig(enc_path, dpi=110, bbox_inches='tight')
        print(f"Saved: {enc_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-drive", action="store_true",
                        help="simulation only — don't open the ESP32 serial port")
    parser.add_argument("--port", default=None,
                        help=f"serial port override (default: {_MOTOR_PORT})")
    parser.add_argument("--countdown", type=int, default=3,
                        help="seconds before motors start (default: 3)")
    args = parser.parse_args()

    data = run(drive=not args.no_drive, port=args.port, countdown=args.countdown)
    plot(data)
