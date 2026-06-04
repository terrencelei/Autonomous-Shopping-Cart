#!/usr/bin/env python3
"""
calibrate_angular_inertia.py — measure ANGULAR_INERTIA for follow.py

Method
------
1. Spin in place at SPIN_KICK_RPM until the cart reaches steady-state yaw rate
   (we drive for STEADY_TICKS encoder ticks to let it settle).
2. Command 0 RPM (hard stop).
3. Drain the encoder for COAST_WINDOW_S seconds, accumulating every tick that
   arrives after the stop command — this is the inertial coast.
4. Convert coast ticks → radians using _ticks_to_yaw().
5. Convert SPIN_KICK_RPM → rad/s using _rpm_to_yaw().
6. ANGULAR_INERTIA (s) = overshoot_rad / omega_kick_rad_s

Run on a flat surface with the cart at normal operating weight.
Repeat 3–5 times and average; discard outliers.

Usage:
    python3 calibrate_angular_inertia.py
    python3 calibrate_angular_inertia.py --runs 5 --port /dev/ttyUSB1
"""

import argparse
import time
import math
import statistics

# ── reuse every constant and helper from follow.py ────────────────────────────
from follow import (
    SPIN_KICK_RPM,
    KICK_TICKS,
    KICK_TIMEOUT_S,
    ENCODER_PPR,
    LEFT_ENC_SIGN,
    RIGHT_ENC_SIGN,
    LEFT_MOTOR_SIGN,
    RIGHT_MOTOR_SIGN,
    MOTOR_PORT,
    MOTOR_BAUD,
    _rpm_to_yaw,
    _ticks_to_yaw,
)

# ── calibration-specific knobs ─────────────────────────────────────────────────
STEADY_TICKS    = 80      # encoder ticks to drive before stopping (let cart reach steady spin)
COAST_WINDOW_S  = 1.5     # seconds to keep reading encoders after the stop command
INTER_RUN_SLEEP = 2.0     # seconds between runs (let the cart fully settle)


def _open_serial(port):
    import serial
    ser = serial.Serial(port, MOTOR_BAUD, timeout=0.02)
    time.sleep(0.1)
    ser.reset_input_buffer()
    print(f"Connected on {port}")
    return ser


def _send(ser, l_rpm, r_rpm):
    cmd = f"L{LEFT_MOTOR_SIGN * l_rpm:.1f} R{RIGHT_MOTOR_SIGN * r_rpm:.1f}\n"
    ser.write(cmd.encode())


def _read_delta(ser, last):
    """Drain one pass of the serial buffer; return (d_left, d_right, new_last)."""
    dl = dr = 0
    while ser.in_waiting:
        try:
            line = ser.readline().decode(errors="ignore").strip()
        except Exception:
            break
        if not line.startswith("E,"):
            continue
        try:
            _, ls, rs = line.split(",", 2)
            lc = LEFT_ENC_SIGN  * int(ls)
            rc = RIGHT_ENC_SIGN * int(rs)
        except ValueError:
            continue
        if last is not None:
            dl += lc - last[0]
            dr += rc - last[1]
        last = (lc, rc)
    return dl, dr, last


def single_run(ser, run_index):
    """Execute one spin-and-coast measurement; return ANGULAR_INERTIA in seconds."""
    print(f"\n── run {run_index} ──")
    ser.reset_input_buffer()
    last = None

    # ── phase 1: spin at SPIN_KICK_RPM until STEADY_TICKS accumulate ──────────
    # Direction: CCW (left wheel back, right wheel forward) — matches Spinner convention
    _send(ser, -SPIN_KICK_RPM, +SPIN_KICK_RPM)
    drive_ticks = 0
    t0 = time.monotonic()
    while drive_ticks < STEADY_TICKS:
        time.sleep(0.005)
        dl, dr, last = _read_delta(ser, last)
        drive_ticks += abs(dl) + abs(dr)
        if time.monotonic() - t0 > 5.0:          # safety: bail if stuck
            print("  WARNING: cart did not reach STEADY_TICKS — is it moving?")
            break
    print(f"  drive phase: {drive_ticks} ticks accumulated @ {SPIN_KICK_RPM} RPM")

    # ── phase 2: hard stop ─────────────────────────────────────────────────────
    _send(ser, 0.0, 0.0)
    t_stop = time.monotonic()

    # ── phase 3: collect coast ticks ───────────────────────────────────────────
    coast_dl = coast_dr = 0
    while time.monotonic() - t_stop < COAST_WINDOW_S:
        time.sleep(0.005)
        dl, dr, last = _read_delta(ser, last)
        coast_dl += dl
        coast_dr += dr

    coast_ticks = abs(coast_dl) + abs(coast_dr)
    print(f"  coast ticks: L{coast_dl:+d}  R{coast_dr:+d}  (total {coast_ticks})")

    # ── phase 4: unit conversions ──────────────────────────────────────────────
    # _ticks_to_yaw(d_left, d_right) → swept yaw in radians, +CCW
    # For a CCW spin: left ticks are negative (wheel moving backward),
    # right ticks are positive (wheel moving forward).
    # coast_dl/coast_dr already carry the correct signs from LEFT/RIGHT_ENC_SIGN.
    overshoot_rad = abs(_ticks_to_yaw(coast_dl, coast_dr))

    # _rpm_to_yaw(wheel_rpm) → yaw rate magnitude in rad/s
    omega_kick = _rpm_to_yaw(SPIN_KICK_RPM)     # rad/s at SPIN_KICK_RPM

    # ── phase 5: inertia ──────────────────────────────────────────────────────
    # drift model: extra_angle = (omega_before - omega_after) * ANGULAR_INERTIA
    # At a hard stop: omega_after = 0  →  extra_angle = omega_kick * ANGULAR_INERTIA
    inertia_s = overshoot_rad / omega_kick if omega_kick > 0 else 0.0

    print(f"  overshoot : {math.degrees(overshoot_rad):.3f}°  ({overshoot_rad:.5f} rad)")
    print(f"  omega_kick: {omega_kick:.4f} rad/s  ({SPIN_KICK_RPM:.1f} RPM)")
    print(f"  → ANGULAR_INERTIA = {inertia_s:.5f} s")
    return inertia_s


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--runs", type=int, default=3,
                        help="number of measurement runs to average (default: 3)")
    parser.add_argument("--port", default=MOTOR_PORT,
                        help=f"ESP32 serial port (default: {MOTOR_PORT})")
    args = parser.parse_args()

    ser = _open_serial(args.port)
    results = []
    try:
        for i in range(1, args.runs + 1):
            val = single_run(ser, i)
            results.append(val)
            if i < args.runs:
                print(f"  (waiting {INTER_RUN_SLEEP}s for cart to settle…)")
                time.sleep(INTER_RUN_SLEEP)
    finally:
        _send(ser, 0.0, 0.0)
        ser.close()

    print("\n══ RESULTS ══")
    for i, v in enumerate(results, 1):
        print(f"  run {i}: {v:.5f} s")
    if len(results) > 1:
        mean = statistics.mean(results)
        stdev = statistics.stdev(results)
        print(f"  mean  : {mean:.5f} s")
        print(f"  stdev : {stdev:.5f} s")
        print(f"\nSet in follow.py:\n  ANGULAR_INERTIA = {mean:.5f}  # seconds (calibrated)")
    else:
        print(f"\nSet in follow.py:\n  ANGULAR_INERTIA = {results[0]:.5f}  # seconds (calibrated)")


if __name__ == "__main__":
    main()
