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


def integrate_kinematics(pos, heading, v_right, v_left, dt):
    v_fwd = (v_right + v_left) / 2.0
    omega = (v_left - v_right) / P.TRACK_M
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
    pos     = list(START_POS)
    heading = START_HEADING

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
    encoder_degrees = []
    v_fwds, omegas, v_rights, v_lefts = [], [], [], []
    enc_right_cum, enc_left_cum = [], []
    enc_right_delta, enc_left_delta = [], []
    cum_l, cum_r = 0, 0

    # Ticks one wheel travels during a 360° point turn:
    #   arc = π * TRACK_M  (half-circumference of the turn circle)
    #   ticks = arc / WHEEL_CIRC * ENCODER_PPR
    TICKS_360 = math.pi * P.TRACK_M / P.WHEEL_CIRC * P.ENCODER_PPR
    print(f"Target: {TICKS_360:.0f} active-wheel encoder ticks for 360°  "
          f"(ENCODER_PPR={P.ENCODER_PPR})")

    abs_ticks_l = 0   # accumulated absolute right encoder ticks (real hardware)
    abs_ticks_r = 0   # accumulated absolute left encoder ticks (real hardware)
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

            v_right, v_left = P._wheel_commands(0.0, P.MAX_TURN)

            if motors is not None:
                motors.send(v_right, v_left)
                d_l, d_r = motors.read_encoder_deltas()
            else:
                d_l, d_r = 0, 0
            cum_l += d_l; cum_r += d_r
            abs_ticks_l += abs(d_l)
            abs_ticks_r += abs(d_r)
            active_ticks = [
                ticks for ticks in (abs_ticks_l, abs_ticks_r)
                if ticks > 0
            ]
            measured_ticks = (
                sum(active_ticks) / len(active_ticks)
                if active_ticks else 0.0
            )
            encoder_degrees.append(measured_ticks / TICKS_360 * 360.0)
            enc_right_delta.append(d_l);  enc_left_delta.append(d_r)
            enc_right_cum.append(cum_l);  enc_left_cum.append(cum_r)

            times.append(t)
            robot_xs.append(pos[0]); robot_ys.append(pos[1])
            robot_thetas.append(heading)

            new_pos, new_heading, v_fwd, omega = integrate_kinematics(
                pos, heading, v_right, v_left, P.DT)
            pos     = new_pos
            heading = new_heading
            v_fwds.append(v_fwd); omegas.append(omega)
            v_rights.append(v_right); v_lefts.append(v_left)

            turned += abs(omega) * P.DT

            # Stop condition: real encoder ticks when driving, simulated
            # radians when running --no-drive
            if motors is not None:
                if measured_ticks >= TICKS_360:
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

    elapsed_s = len(times) * P.DT
    active_ticks = [ticks for ticks in (abs_ticks_l, abs_ticks_r) if ticks > 0]
    measured_ticks = sum(active_ticks) / len(active_ticks) if active_ticks else 0.0
    measured_degrees = measured_ticks / TICKS_360 * 360.0
    print(f"\nSpin complete: {measured_degrees:.1f}° encoder  |  "
          f"L={abs_ticks_l} R={abs_ticks_r} measured={measured_ticks:.0f} encoder ticks  |  "
          f"{elapsed_s:.1f} s  ({len(times)} ticks)")
    if motors is not None and (abs_ticks_l == 0 or abs_ticks_r == 0):
        print("WARNING: one encoder reported 0 ticks during the spin. "
              "Check encoder wiring/signs before trusting odometry.")

    return dict(
        t=times,
        rx=robot_xs, ry=robot_ys, rt=robot_thetas,
        encoder_degrees=encoder_degrees,
        v_fwd=v_fwds, omega=omegas,
        v_right=v_rights, v_left=v_lefts,
        enc_l_cum=enc_right_cum, enc_r_cum=enc_left_cum,
        enc_l_delta=enc_right_delta, enc_r_delta=enc_left_delta,
        drove=(motors is not None),
    )


def plot(data, out_path="pathfinding_arc_test.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # ── Angle over time ────────────────────────────────────
    ax = axes[0]
    if data.get('drove'):
        angles_deg = data['encoder_degrees']
        title = 'Encoder angle over time'
        ylabel = 'encoder angle (deg)'
    else:
        angles_deg = [math.degrees(h) for h in data['rt']]
        title = 'Sim heading over time'
        ylabel = 'sim heading (deg)'
    ax.plot(data['t'], angles_deg, 'b-')
    if angles_deg and data['t']:
        final_angle = angles_deg[-1]
        ax.plot(data['t'][-1], final_angle, 'ko', ms=4)
        ax.annotate(f'{final_angle:.1f}°',
                    xy=(data['t'][-1], final_angle),
                    xytext=(-55, 14),
                    textcoords='offset points',
                    fontsize=9,
                    arrowprops=dict(arrowstyle='->', lw=0.8))
    ax.axhline(360, color='gray', ls=':', alpha=0.6, label='360°')
    ax.set_xlabel('time (s)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
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

        ax_cum.plot(data['t'], data['enc_l_cum'], 'b-', label='right')
        ax_cum.plot(data['t'], data['enc_r_cum'], 'r-', label='left')
        ax_cum.set_ylabel('cumulative ticks')
        ax_cum.set_title('Encoder counts')
        ax_cum.legend(fontsize=9)
        ax_cum.grid(True, alpha=0.3)

        rpm_l = [v / P.WHEEL_CIRC * 60 for v in data['v_right']]
        rpm_r = [v / P.WHEEL_CIRC * 60 for v in data['v_left']]
        ax_rpm.plot(data['t'], rpm_l, 'b-', label='right commanded RPM')
        ax_rpm.plot(data['t'], rpm_r, 'r-', label='left commanded RPM')
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
