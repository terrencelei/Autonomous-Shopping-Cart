#!/usr/bin/env python3
"""
rpm_log.py — record commanded vs measured wheel RPM during a forward drive
and graph them, so left/right tracking and drive direction can be analysed.

It mirrors move_forward_one_meter.py's kick-then-run command profile but logs
every control tick to a CSV, and (where matplotlib is available) plots it.

On the Pi (cart on /dev/ttyUSB0) — wheels up on a box for a first look:
    python3 rpm_log.py                       # drive, log to rpm_log.csv, plot rpm_log.png
    python3 rpm_log.py --distance 0.3 --timeout 6

Plot an existing log anywhere (e.g. copy the CSV to a laptop with matplotlib):
    python3 rpm_log.py --plot rpm_log.csv

Reading the graph
-----------------
* Commanded RPM is exactly what is sent to the ESP32 (the PID target).
* Measured RPM is derived from encoder deltas in the SAME sign convention,
  smoothed over a short window (at low RPM each 20 ms tick is coarse).
* From the hand-push sniff test, FORWARD motion shows up as NEGATIVE measured
  RPM on both wheels. So: measured sign => direction; |left| vs |right| => turn;
  measured vs commanded => whether the loop is actually tracking.
"""

import argparse
import csv
import time

CSV_FIELDS = ["t", "cmd_l", "cmd_r", "d_l", "d_r", "cum_l", "cum_r", "phase"]


def drive_and_log(args):
    import Pathfinding_algorithm as P

    def rpm_to_speed(r):
        return r * P.WHEEL_CIRC / 60.0

    def fwd(speed):
        return P._wheel_commands(speed, 0.0)

    if args.port:
        P.MOTOR_PORT = args.port
    motors = P.MotorDriver()

    normal_speed = rpm_to_speed(abs(args.rpm))
    kick_rpm = abs(args.kick_rpm)
    v_left, v_right = fwd(rpm_to_speed(kick_rpm))

    if motors._ser and motors._ser.is_open:
        motors._ser.reset_input_buffer()
        motors._last_l = motors._last_r = None
        motors._dl = motors._dr = 0

    print(f"Logging forward drive: {args.distance:.2f} m, kick "
          f"{kick_rpm:.1f}->{args.kick_max_rpm:.1f} rpm until "
          f"{args.kick_release_ticks} ticks, then {args.rpm:.1f} rpm")
    for n in range(args.countdown, 0, -1):
        print(f"  {n}...")
        time.sleep(1)

    rows = []
    travelled = 0.0
    cum_l = cum_r = 0
    kick_active = args.kick_rpm > args.rpm and args.kick_release_ticks > 0
    kick_l = kick_r = kick_window = 0
    start = time.monotonic()
    next_ramp = start + args.kick_ramp_hold
    tpr = P.ENCODER_PPR * getattr(P, "GEAR_RATIO", 1)

    try:
        while travelled < args.distance:
            loop_start = time.monotonic()
            elapsed = loop_start - start
            if args.timeout > 0 and elapsed >= args.timeout:
                print(f"Timeout at {elapsed:.1f}s")
                break

            motors.send(v_left, v_right)
            d_l, d_r = motors.read_encoder_deltas()
            cum_l += d_l
            cum_r += d_r

            rows.append([
                round(elapsed, 4),
                round(P.LEFT_MOTOR_SIGN * P._rpm(v_left), 3),
                round(P.RIGHT_MOTOR_SIGN * P._rpm(v_right), 3),
                d_l, d_r, cum_l, cum_r,
                "kick" if kick_active else "run",
            ])

            if kick_active:
                kick_l += abs(d_l)
                kick_r += abs(d_r)
                kick_window += abs(d_l) + abs(d_r)
                if kick_l >= args.kick_release_ticks or kick_r >= args.kick_release_ticks:
                    kick_active = False
                    v_left, v_right = fwd(normal_speed)
                    print(f"kick released at ticks=({kick_l},{kick_r}) t={elapsed:.2f}s")
                elif time.monotonic() >= next_ramp:
                    if kick_window == 0 and kick_rpm < args.kick_max_rpm:
                        kick_rpm = min(args.kick_max_rpm, kick_rpm + args.kick_ramp_step_rpm)
                        v_left, v_right = fwd(rpm_to_speed(kick_rpm))
                    kick_window = 0
                    next_ramp = time.monotonic() + args.kick_ramp_hold

            if d_l or d_r:
                travelled += (abs(d_l) + abs(d_r)) / 2.0 / tpr * P.WHEEL_CIRC

            sleep_s = P.DT - (time.monotonic() - loop_start)
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        motors.stop()
        print("Motors stopped.")

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_FIELDS)
        w.writerows(rows)
    print(f"Wrote {len(rows)} samples -> {args.csv}")
    return args.csv


def _windowed_rpm(t, cum, ppr, window_s):
    import numpy as np
    rpm = np.zeros(len(t))
    for i in range(len(t)):
        lo, hi = i, i
        while lo > 0 and t[i] - t[lo] < window_s / 2:
            lo -= 1
        while hi < len(t) - 1 and t[hi] - t[i] < window_s / 2:
            hi += 1
        dt = t[hi] - t[lo]
        if dt > 0:
            rpm[i] = (cum[hi] - cum[lo]) / ppr * 60.0 / dt
    return rpm


def plot(csv_path, png_path, ppr, window_s):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = {k: [] for k in CSV_FIELDS}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            for k in CSV_FIELDS:
                cols[k].append(row[k])
    t = np.array(cols["t"], float)
    cmd_l = np.array(cols["cmd_l"], float)
    cmd_r = np.array(cols["cmd_r"], float)
    cum_l = np.array(cols["cum_l"], float)
    cum_r = np.array(cols["cum_r"], float)
    phase = cols["phase"]

    meas_l = _windowed_rpm(t, cum_l, ppr, window_s)
    meas_r = _windowed_rpm(t, cum_r, ppr, window_s)

    kick_end = next((t[i] for i in range(1, len(phase))
                     if phase[i - 1] == "kick" and phase[i] == "run"), None)

    fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for a, cmd, meas, name in ((ax[0], cmd_l, meas_l, "LEFT"),
                               (ax[1], cmd_r, meas_r, "RIGHT")):
        a.plot(t, cmd, color="tab:blue", lw=2, label="commanded")
        a.plot(t, meas, color="tab:red", lw=1.3, label="measured")
        a.axhline(0, color="k", lw=0.6, alpha=0.4)
        if kick_end is not None:
            a.axvline(kick_end, color="gray", ls="--", lw=1, label="kick release")
        a.set_ylabel(f"{name} wheel RPM")
        a.grid(alpha=0.3)
        a.legend(loc="upper right", fontsize=8)
    ax[2].plot(t, meas_l, color="tab:green", label="left measured")
    ax[2].plot(t, meas_r, color="tab:orange", label="right measured")
    ax[2].axhline(0, color="k", lw=0.6, alpha=0.4)
    if kick_end is not None:
        ax[2].axvline(kick_end, color="gray", ls="--", lw=1)
    ax[2].set_ylabel("measured L vs R")
    ax[2].set_xlabel("time (s)")
    ax[2].grid(alpha=0.3)
    ax[2].legend(loc="upper right", fontsize=8)
    fig.suptitle("Commanded vs measured wheel RPM   "
                 f"(measured smoothed {window_s * 1000:.0f} ms; "
                 "forward = negative RPM)")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    print(f"Wrote {png_path}")


def main():
    ap = argparse.ArgumentParser(description="Log/plot commanded vs measured wheel RPM.")
    ap.add_argument("--plot", metavar="CSV", help="skip driving; just plot this CSV")
    ap.add_argument("--csv", default="rpm_log.csv")
    ap.add_argument("--png", default="rpm_log.png")
    ap.add_argument("--window", type=float, default=0.15, help="measured-RPM smoothing window (s)")
    ap.add_argument("--distance", type=float, default=1.0)
    ap.add_argument("--rpm", type=float, default=0.5)
    ap.add_argument("--kick-rpm", type=float, default=8.0)
    ap.add_argument("--kick-max-rpm", type=float, default=15.0)
    ap.add_argument("--kick-ramp-step-rpm", type=float, default=1.0)
    ap.add_argument("--kick-ramp-hold", type=float, default=0.1)
    ap.add_argument("--kick-release-ticks", type=int, default=30)
    ap.add_argument("--port", default=None)
    ap.add_argument("--countdown", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args()

    csv_path = args.plot or drive_and_log(args)

    try:
        import Pathfinding_algorithm as P
        ppr = P.ENCODER_PPR
    except Exception:
        ppr = 298.0

    try:
        plot(csv_path, args.png, ppr, args.window)
    except ImportError as e:
        print(f"Plot skipped ({e}). CSV is at {csv_path}; copy it to a machine "
              f"with matplotlib and run:  python3 rpm_log.py --plot {csv_path}")


if __name__ == "__main__":
    main()
