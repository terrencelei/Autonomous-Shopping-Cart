"""
Drive the cart straight forward for one metre, then stop.

Usage:
    python3 move_forward_one_meter.py
    python3 move_forward_one_meter.py --rpm 0.5 --port /dev/ttyUSB0
"""

import argparse
import time

import Pathfinding_algorithm as P


TARGET_DISTANCE_M = 1.0
DEFAULT_RPM = 0.5
DEFAULT_KICK_RPM = 8.0
DEFAULT_KICK_MAX_RPM = 15.0
DEFAULT_KICK_RAMP_STEP_RPM = 1.0
DEFAULT_KICK_RAMP_HOLD_S = 0.1
DEFAULT_KICK_RELEASE_TICKS = 30
DEFAULT_TIMEOUT_S = 20.0


def encoder_delta_to_wheel_distance(delta_ticks):
    ticks_per_wheel_rev = P.ENCODER_PPR * getattr(P, "GEAR_RATIO", 1)
    return delta_ticks / ticks_per_wheel_rev * P.WHEEL_CIRC


def wheel_rpm_to_speed(rpm):
    return rpm * P.WHEEL_CIRC / 60.0


def forward_wheel_commands(speed_m_s):
    # Use the same straight-drive direction for startup kick and normal running.
    return speed_m_s, speed_m_s


def flush_motors(motors):
    if motors._ser and motors._ser.is_open:
        motors._ser.reset_input_buffer()
        motors._last_l = motors._last_r = None
        motors._dl = motors._dr = 0


def run(distance_m=TARGET_DISTANCE_M, rpm=DEFAULT_RPM,
        kick_rpm=DEFAULT_KICK_RPM,
        kick_max_rpm=DEFAULT_KICK_MAX_RPM,
        kick_ramp_step_rpm=DEFAULT_KICK_RAMP_STEP_RPM,
        kick_ramp_hold_s=DEFAULT_KICK_RAMP_HOLD_S,
        kick_release_ticks=DEFAULT_KICK_RELEASE_TICKS,
        port=None, countdown=3, timeout_s=DEFAULT_TIMEOUT_S):
    if port:
        P.MOTOR_PORT = port

    motors = P.MotorDriver()
    normal_speed = wheel_rpm_to_speed(abs(rpm))
    current_kick_rpm = abs(kick_rpm)
    kick_speed = wheel_rpm_to_speed(current_kick_rpm)
    v_left, v_right = forward_wheel_commands(kick_speed)

    print(f"\nDrive forward target: {distance_m:.2f} m")
    print(f"Command: {rpm:.2f} wheel RPM  |  timeout: {timeout_s:.1f}s")
    print(f"Kick: {current_kick_rpm:.2f}-{kick_max_rpm:.2f} wheel RPM "
          f"until {kick_release_ticks} encoder ticks")
    if countdown > 0:
        print(f"Starting in {countdown}s...")
        for remaining in range(countdown, 0, -1):
            print(f"  {remaining}...")
            time.sleep(1)

    flush_motors(motors)

    travelled_m = 0.0
    cum_l = 0
    cum_r = 0
    kick_active = kick_rpm > rpm and kick_release_ticks > 0
    kick_l_ticks = 0
    kick_r_ticks = 0
    kick_window_ticks = 0
    start = time.monotonic()
    next_kick_ramp_t = start + kick_ramp_hold_s
    last_print = start

    try:
        while travelled_m < distance_m:
            loop_start = time.monotonic()
            elapsed = loop_start - start
            if timeout_s > 0 and elapsed >= timeout_s:
                print(f"Timeout at {elapsed:.1f}s before reaching target.")
                break

            motors.send(v_left, v_right)
            d_l, d_r = motors.read_encoder_deltas()
            cum_l += d_l
            cum_r += d_r
            if kick_active:
                delta_ticks = abs(d_l) + abs(d_r)
                kick_l_ticks += abs(d_l)
                kick_r_ticks += abs(d_r)
                kick_window_ticks += delta_ticks
                if kick_l_ticks >= kick_release_ticks or kick_r_ticks >= kick_release_ticks:
                    kick_active = False
                    v_left, v_right = forward_wheel_commands(normal_speed)
                    print(f"Kick released at ticks=({kick_l_ticks},{kick_r_ticks})")
                elif time.monotonic() >= next_kick_ramp_t:
                    if kick_window_ticks == 0 and current_kick_rpm < kick_max_rpm:
                        current_kick_rpm = min(
                            kick_max_rpm,
                            current_kick_rpm + kick_ramp_step_rpm,
                        )
                        v_left, v_right = forward_wheel_commands(
                            wheel_rpm_to_speed(current_kick_rpm))
                        print(f"No encoder ticks; increasing kick to {current_kick_rpm:.2f} RPM")
                    kick_window_ticks = 0
                    next_kick_ramp_t = time.monotonic() + kick_ramp_hold_s

            d_left_m = abs(encoder_delta_to_wheel_distance(d_l))
            d_right_m = abs(encoder_delta_to_wheel_distance(d_r))
            if d_l or d_r:
                travelled_m += (d_left_m + d_right_m) / 2.0

            now = time.monotonic()
            if now - last_print >= 0.5:
                phase = "kick" if kick_active else "run"
                print(f"distance={travelled_m:.3f} m  {phase}  "
                      f"cmd={current_kick_rpm if kick_active else rpm:.2f}rpm  "
                      f"ticks=({cum_l:+d},{cum_r:+d})")
                last_print = now

            sleep_s = P.DT - (time.monotonic() - loop_start)
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        print("\nStopping motors.")
        motors.stop()

    print(f"Final distance: {travelled_m:.3f} m  ticks=({cum_l:+d},{cum_r:+d})")
    return travelled_m


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drive forward one metre.")
    parser.add_argument("--distance", type=float, default=TARGET_DISTANCE_M,
                        help="target distance in metres (default: 1.0)")
    parser.add_argument("--rpm", type=float, default=DEFAULT_RPM,
                        help="commanded wheel RPM (default: 0.4)")
    parser.add_argument("--kick-rpm", type=float, default=DEFAULT_KICK_RPM,
                        help="startup kick wheel RPM (default: 15.0)")
    parser.add_argument("--kick-max-rpm", type=float, default=DEFAULT_KICK_MAX_RPM,
                        help="max kick wheel RPM if encoders do not move (default: 30.0)")
    parser.add_argument("--kick-ramp-step-rpm", type=float,
                        default=DEFAULT_KICK_RAMP_STEP_RPM,
                        help="kick RPM increase after each no-tick hold (default: 2.0)")
    parser.add_argument("--kick-ramp-hold", type=float,
                        default=DEFAULT_KICK_RAMP_HOLD_S,
                        help="seconds to wait before increasing kick RPM (default: 0.5)")
    parser.add_argument("--kick-release-ticks", type=int,
                        default=DEFAULT_KICK_RELEASE_TICKS,
                        help="encoder ticks before dropping to normal RPM (default: 100)")
    parser.add_argument("--port", default=None,
                        help=f"serial port override (default: {P.MOTOR_PORT})")
    parser.add_argument("--countdown", type=int, default=3,
                        help="seconds before motors start (default: 3)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                        help="max run time in seconds; 0 disables timeout (default: 20)")
    args = parser.parse_args()

    run(distance_m=args.distance, rpm=args.rpm,
        kick_rpm=args.kick_rpm,
        kick_max_rpm=args.kick_max_rpm,
        kick_ramp_step_rpm=args.kick_ramp_step_rpm,
        kick_ramp_hold_s=args.kick_ramp_hold,
        kick_release_ticks=args.kick_release_ticks,
        port=args.port,
        countdown=args.countdown, timeout_s=args.timeout)
