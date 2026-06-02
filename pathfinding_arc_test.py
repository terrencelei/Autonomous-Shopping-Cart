"""
Rotation test — either spin one full 360° turn or rotate in place to keep
the shopper centred in the camera FOV.

Usage:
    python3 pathfinding_arc_test.py                    # centre target from live camera
    python3 pathfinding_arc_test.py --source udp       # centre target from UDP
    python3 pathfinding_arc_test.py --mode spin         # one 360° spin
    python3 pathfinding_arc_test.py --mode slow-spin    # slow in-place spin
    python3 pathfinding_arc_test.py --no-drive --sim-angle 15
    python3 pathfinding_arc_test.py --port /dev/ttyUSB0
"""

import argparse
import math
import os
import time

import Pathfinding_algorithm as P

# Handle renamed constants between versions of Pathfinding_algorithm
_MOTOR_PORT = getattr(P, 'MOTOR_PORT', getattr(P, 'MOTOR_UART_PORT', '/dev/ttyACM0'))

# =============================================================================
# CONFIG
# =============================================================================

START_POS     = [0.0, 0.0]
START_HEADING = 0.0
CENTER_DEADBAND_DEG = 4.0
CENTER_REACQUIRE_S  = 0.25
CENTER_MIN_TURN_DEG = 2.0
CENTER_MAX_TURN_DEG = 8.0
CENTER_KP           = 0.006  # rad/s per degree of angle error

# Slow-spin / stall-test parameters
STALL_BOOST_OMEGA_DEG = 90.0   # speed during initial burst to break static friction (deg/s)
STALL_BOOST_SECS      = 0.5    # duration of boost phase
STALL_RAMP_START_DEG  = 10.0   # ramp begins at this speed after boost
STALL_DETECT_TICKS    = 5      # consecutive zero-delta ticks before declaring stall

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


def open_motors(drive=True, port=None, countdown=3, action="moving"):
    if not drive:
        return None

    chosen = port or find_serial_port(_MOTOR_PORT)
    if chosen != _MOTOR_PORT:
        P.MOTOR_PORT = chosen
    motors = P.MotorDriver()
    if countdown > 0:
        print(f"\n*** Cart will start {action} in {countdown}s ***")
        for k in range(countdown, 0, -1):
            print(f"  {k}...")
            time.sleep(1)
        print("  GO!\n")
    return motors


def flush_motors(motors):
    if motors is not None and motors._ser and motors._ser.is_open:
        motors._ser.reset_input_buffer()
        motors._last_l = motors._last_r = None
        motors._dl = motors._dr = 0


def run_spin(drive=True, port=None, countdown=3):
    pos     = list(START_POS)
    heading = START_HEADING

    motors = open_motors(drive=drive, port=port, countdown=countdown, action="spinning")

    times         = []
    robot_xs, robot_ys, robot_thetas = [], [], []
    encoder_degrees = []
    v_fwds, omegas, v_lefts, v_rights = [], [], [], []
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
    flush_motors(motors)

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
                pos, heading, v_left, v_right, P.DT)
            pos     = new_pos
            heading = new_heading
            v_fwds.append(v_fwd); omegas.append(omega)
            v_lefts.append(v_left); v_rights.append(v_right)

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
        mode="spin",
        t=times,
        rx=robot_xs, ry=robot_ys, rt=robot_thetas,
        encoder_degrees=encoder_degrees,
        v_fwd=v_fwds, omega=omegas,
        v_left=v_lefts, v_right=v_rights,
        enc_l_cum=enc_right_cum, enc_r_cum=enc_left_cum,
        enc_l_delta=enc_right_delta, enc_r_delta=enc_left_delta,
        drove=(motors is not None),
    )


def center_turn_command(angle_deg):
    if abs(angle_deg) <= CENTER_DEADBAND_DEG:
        return 0.0
    omega = -CENTER_KP * angle_deg
    min_turn = math.radians(CENTER_MIN_TURN_DEG)
    max_turn = math.radians(CENTER_MAX_TURN_DEG)
    omega_mag = min(max(abs(omega), min_turn), max_turn)
    return math.copysign(omega_mag, omega)


def run_slow_spin(drive=True, port=None, countdown=3, duration=30.0, direction=1.0):
    pos     = list(START_POS)
    heading = START_HEADING
    motors  = open_motors(drive=drive, port=port, countdown=countdown, action="slow-spinning")

    ramp_duration = max(duration - STALL_BOOST_SECS, 1.0)
    print(f"Slow spin: {STALL_BOOST_SECS:.1f}s boost at {STALL_BOOST_OMEGA_DEG:.0f}°/s, "
          f"then ramp {STALL_RAMP_START_DEG:.0f}→0°/s over {ramp_duration:.1f}s.")

    times, robot_xs, robot_ys, robot_thetas = [], [], [], []
    omegas_cmd, omegas_enc, v_lefts, v_rights = [], [], [], []
    enc_right_cum, enc_left_cum = [], []
    enc_right_delta, enc_left_delta = [], []
    cum_l, cum_r = 0, 0
    zero_tick_streak = 0
    stall_omega_deg  = None
    ramp_start_t     = None
    start = time.monotonic()
    i = 0

    flush_motors(motors)
    try:
        while True:
            t_loop0 = time.monotonic()
            t = t_loop0 - start

            # Phase 1: boost to overcome static friction
            if t < STALL_BOOST_SECS:
                omega_cmd = math.copysign(math.radians(STALL_BOOST_OMEGA_DEG), direction)
            else:
                # Phase 2: linear ramp from STALL_RAMP_START_DEG down to 0
                if ramp_start_t is None:
                    ramp_start_t = t
                frac = min(1.0, (t - ramp_start_t) / ramp_duration)
                omega_mag = math.radians(STALL_RAMP_START_DEG) * (1.0 - frac)
                omega_cmd = math.copysign(omega_mag, direction)

            v_left, v_right = P._wheel_commands(0.0, omega_cmd)
            if motors is not None:
                motors.send(v_left, v_right)
                d_l, d_r = motors.read_encoder_deltas()
            else:
                d_l, d_r = 0, 0

            # Encoder-derived angular velocity
            vl_enc   = d_l * P.M_PER_PULSE / P.DT
            vr_enc   = d_r * P.M_PER_PULSE / P.DT
            omega_enc = (vr_enc - vl_enc) / P.TRACK_M

            cum_l += d_l; cum_r += d_r
            enc_right_delta.append(d_l); enc_left_delta.append(d_r)
            enc_right_cum.append(cum_l); enc_left_cum.append(cum_r)

            times.append(t)
            robot_xs.append(pos[0]); robot_ys.append(pos[1])
            robot_thetas.append(heading)

            pos, heading, _v_fwd, _ = integrate_kinematics(
                pos, heading, v_left, v_right, P.DT)
            omegas_cmd.append(omega_cmd)
            omegas_enc.append(omega_enc)
            v_lefts.append(v_left); v_rights.append(v_right)

            # Stall detection during ramp phase only
            if ramp_start_t is not None and motors is not None:
                if abs(d_l) + abs(d_r) == 0:
                    zero_tick_streak += 1
                    if zero_tick_streak >= STALL_DETECT_TICKS and stall_omega_deg is None:
                        stall_omega_deg = math.degrees(abs(omega_cmd))
                        print(f"Stall at {stall_omega_deg:.1f}°/s  (t={t:.2f}s)")
                        break
                else:
                    zero_tick_streak = 0

            if duration > 0 and t >= duration:
                break

            if i % max(1, int(1.0 / P.DT)) == 0:
                print(f"t={t:5.1f}s  cmd={math.degrees(abs(omega_cmd)):5.1f}°/s  "
                      f"enc={math.degrees(abs(omega_enc)):5.1f}°/s")
            i += 1

            elapsed = time.monotonic() - t_loop0
            if elapsed < P.DT:
                time.sleep(P.DT - elapsed)
    except KeyboardInterrupt:
        print("\nSlow spin stopped by user.")
    finally:
        if motors is not None:
            print("\nStopping motors.")
            motors.stop()

    if stall_omega_deg is not None:
        print(f"\nStall speed: {stall_omega_deg:.1f}°/s")
    else:
        print("\nNo stall detected within duration.")

    return dict(
        mode="slow-spin",
        t=times,
        rx=robot_xs, ry=robot_ys, rt=robot_thetas,
        omega=omegas_cmd,
        omega_enc=omegas_enc,
        stall_omega_deg=stall_omega_deg,
        v_left=v_lefts, v_right=v_rights,
        enc_l_cum=enc_right_cum, enc_r_cum=enc_left_cum,
        enc_l_delta=enc_right_delta, enc_r_delta=enc_left_delta,
        drove=(motors is not None),
    )


def import_yolo_detect():
    try:
        from vision import yolo_detect as Y
    except Exception as e:
        raise SystemExit(f"ERROR: could not import vision/yolo_detect.py: {e}")
    return Y


def run_center_camera(drive=True, port=None, countdown=3, duration=30.0, no_display=False):
    Y = import_yolo_detect()
    if not Y.RPK_MODEL_PATH.exists():
        raise SystemExit(
            f"ERROR: {Y.RPK_MODEL_PATH} not found. Install with: sudo apt install imx500-models"
        )

    motors = open_motors(drive=drive, port=port, countdown=countdown, action="centring")
    odometry = P.Odometry(motors.read_encoder_deltas if motors else lambda: (0, 0))
    cap = Y.IMX500Capture(model_path=Y.RPK_MODEL_PATH, width=640, height=480, fps=30)
    tracker = Y.sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=60,
        minimum_matching_threshold=0.8,
        frame_rate=30,
    )
    smooth_state = {}
    target_lock = Y.TargetLock()

    print("Camera centering mode: yolo_detect boxes/radar, rotate only, no forward motion.")
    print("Press Q in the Cart View window to stop." if not no_display else "Display disabled.")

    times, robot_xs, robot_ys, robot_thetas = [], [], [], []
    target_angles, omegas, v_lefts, v_rights = [], [], [], []
    enc_right_cum, enc_left_cum = [], []
    enc_right_delta, enc_left_delta = [], []
    cum_l, cum_r = 0, 0
    start = time.monotonic()
    last_tick = time.monotonic()
    frame_times = []
    target_visible_since = None
    i = 0

    flush_motors(motors)
    try:
        while True:
            loop_start = time.monotonic()
            t = loop_start - start
            if duration > 0 and t >= duration:
                break

            ret, frame = cap.read()
            if not ret:
                break

            dets = cap.get_detections()
            tracked = tracker.update_with_detections(dets)
            out, rows = Y.annotate_frame(frame, tracked, smooth_state, target_lock, loop_start)

            target_row = next((r for r in rows if r[0] == "TARGET"), None)
            angle_deg = target_row[4] if target_row is not None else None
            if angle_deg is None:
                target_visible_since = None
                command_angle = None
            else:
                if target_visible_since is None:
                    target_visible_since = loop_start
                command_angle = (
                    angle_deg
                    if loop_start - target_visible_since >= CENTER_REACQUIRE_S
                    else None
                )
            omega = center_turn_command(command_angle) if command_angle is not None else 0.0
            v_left, v_right = P._wheel_commands(0.0, omega)

            now = time.monotonic()
            if now - last_tick >= P.DT:
                pos, heading = odometry.update()
                P.S.pos = list(pos)
                P.S.heading = heading

                if motors is not None:
                    motors.send(v_left, v_right)
                last_tick = now

                times.append(t)
                robot_xs.append(P.S.pos[0]); robot_ys.append(P.S.pos[1])
                robot_thetas.append(P.S.heading)
                target_angles.append(float("nan") if angle_deg is None else angle_deg)
                omegas.append(omega)
                v_lefts.append(v_left); v_rights.append(v_right)
                enc_right_delta.append(0); enc_left_delta.append(0)
                enc_right_cum.append(cum_l); enc_left_cum.append(cum_r)

                if i % max(1, int(0.5 / P.DT)) == 0:
                    if angle_deg is None:
                        label = "no target"
                    elif command_angle is None:
                        label = f"reacquire {angle_deg:+.1f}°"
                    else:
                        label = f"angle={angle_deg:+.1f}°"
                    print(f"t={t:5.1f}s  {label:<16} omega={math.degrees(omega):+6.1f}°/s")
                i += 1

            frame_times.append(time.time())
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps_live = (
                (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                if len(frame_times) > 1 else 0.0
            )
            Y.cv2.putText(out, f"CENTER  FPS: {fps_live:.1f}  omega={math.degrees(omega):+.1f}deg/s",
                          (10, 25), Y.cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            if not no_display:
                Y.overlay_map(out, rows)
                Y.cv2.imshow("Cart View", out)
                Y.cv2.imshow("World Map", Y.draw_world_map(rows))
                if Y.cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\nCamera centering stopped by user.")
    finally:
        if motors is not None:
            print("\nStopping motors.")
            motors.stop()
        cap.release()
        if not no_display:
            Y.cv2.destroyAllWindows()

    return dict(
        mode="center",
        t=times,
        rx=robot_xs, ry=robot_ys, rt=robot_thetas,
        target_angle=target_angles,
        omega=omegas,
        v_left=v_lefts, v_right=v_rights,
        enc_l_cum=enc_right_cum, enc_r_cum=enc_left_cum,
        enc_l_delta=enc_right_delta, enc_r_delta=enc_left_delta,
        drove=(motors is not None),
    )


def run_center(drive=True, port=None, countdown=3, duration=30.0, sim_angle=None):
    pos     = list(START_POS)
    heading = START_HEADING
    motors  = open_motors(drive=drive, port=port, countdown=countdown, action="centring")
    receiver = None if sim_angle is not None else P.TargetReceiver()

    print("Centering mode: rotate only, no forward motion.")
    if sim_angle is None:
        print(f"Listening for UDP target readings on port {P.UDP_PORT}.")
    else:
        print(f"Using simulated target angle {sim_angle:+.1f}°.")

    times, robot_xs, robot_ys, robot_thetas = [], [], [], []
    target_angles, omegas, v_lefts, v_rights = [], [], [], []
    enc_right_cum, enc_left_cum = [], []
    enc_right_delta, enc_left_delta = [], []
    cum_l, cum_r = 0, 0
    start = time.monotonic()
    target_visible_since = None
    i = 0

    flush_motors(motors)
    try:
        while True:
            t_loop0 = time.monotonic()
            t = t_loop0 - start
            if duration > 0 and t >= duration:
                break

            reading = (0.0, sim_angle) if sim_angle is not None else receiver.get()
            angle_deg = reading[1] if reading is not None else None
            if sim_angle is not None:
                command_angle = angle_deg
            elif angle_deg is None:
                target_visible_since = None
                command_angle = None
            else:
                if target_visible_since is None:
                    target_visible_since = t_loop0
                command_angle = (
                    angle_deg
                    if t_loop0 - target_visible_since >= CENTER_REACQUIRE_S
                    else None
                )
            omega = center_turn_command(command_angle) if command_angle is not None else 0.0
            v_left, v_right = P._wheel_commands(0.0, omega)

            if motors is not None:
                motors.send(v_left, v_right)
                d_l, d_r = motors.read_encoder_deltas()
            else:
                d_l, d_r = 0, 0
            cum_l += d_l; cum_r += d_r
            enc_right_delta.append(d_l); enc_left_delta.append(d_r)
            enc_right_cum.append(cum_l); enc_left_cum.append(cum_r)

            times.append(t)
            robot_xs.append(pos[0]); robot_ys.append(pos[1])
            robot_thetas.append(heading)
            target_angles.append(float("nan") if angle_deg is None else angle_deg)

            new_pos, new_heading, v_fwd, sim_omega = integrate_kinematics(
                pos, heading, v_left, v_right, P.DT)
            pos = new_pos
            heading = new_heading
            omegas.append(omega)
            v_lefts.append(v_left); v_rights.append(v_right)

            if i % max(1, int(0.5 / P.DT)) == 0:
                if angle_deg is None:
                    label = "no target"
                elif command_angle is None:
                    label = f"reacquire {angle_deg:+.1f}°"
                else:
                    label = f"angle={angle_deg:+.1f}°"
                print(f"t={t:5.1f}s  {label:<16} omega={math.degrees(omega):+6.1f}°/s")
            i += 1

            elapsed = time.monotonic() - t_loop0
            if elapsed < P.DT:
                time.sleep(P.DT - elapsed)
    except KeyboardInterrupt:
        print("\nCentering stopped by user.")
    finally:
        if motors is not None:
            print("\nStopping motors.")
            motors.stop()

    return dict(
        mode="center",
        t=times,
        rx=robot_xs, ry=robot_ys, rt=robot_thetas,
        target_angle=target_angles,
        omega=omegas,
        v_left=v_lefts, v_right=v_rights,
        enc_l_cum=enc_right_cum, enc_r_cum=enc_left_cum,
        enc_l_delta=enc_right_delta, enc_r_delta=enc_left_delta,
        drove=(motors is not None),
    )


def _plot_slow_spin(data, out_path):
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Skipping plots: {e}")
        return

    t            = data['t']
    cmd_deg_s    = [math.degrees(o) for o in data['omega']]
    enc_deg_s    = [math.degrees(o) for o in data.get('omega_enc', [0.0] * len(t))]
    rpm_cmd_l    = [v / P.WHEEL_CIRC * 60 for v in data['v_left']]
    rpm_cmd_r    = [v / P.WHEEL_CIRC * 60 for v in data['v_right']]
    rpm_enc_l    = [d * P.M_PER_PULSE / P.DT / P.WHEEL_CIRC * 60
                    for d in data['enc_l_delta']]
    rpm_enc_r    = [d * P.M_PER_PULSE / P.DT / P.WHEEL_CIRC * 60
                    for d in data['enc_r_delta']]

    fig, (ax_omega, ax_rpm) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # ── Commanded vs encoder angular speed ────────────────
    ax_omega.plot(t, cmd_deg_s, 'b-',  label='commanded', lw=1.5)
    ax_omega.plot(t, enc_deg_s, 'r-',  label='encoder',   lw=1.2, alpha=0.85)
    stall = data.get('stall_omega_deg')
    if stall is not None:
        ax_omega.axhline(stall, color='orange', ls='--', lw=1.2,
                         label=f'stall ≈ {stall:.1f}°/s')
    ax_omega.axvline(STALL_BOOST_SECS, color='gray', ls=':', alpha=0.5, label='ramp start')
    ax_omega.set_ylabel('angular speed (deg/s)')
    ax_omega.set_title('Commanded vs encoder angular speed')
    ax_omega.legend(fontsize=8)
    ax_omega.grid(True, alpha=0.3)

    # ── Per-wheel commanded vs encoder RPM ────────────────
    ax_rpm.plot(t, rpm_cmd_l, 'b-',  label='left cmd',  lw=1.5)
    ax_rpm.plot(t, rpm_cmd_r, 'r-',  label='right cmd', lw=1.5)
    ax_rpm.plot(t, rpm_enc_l, 'b--', label='left enc',  lw=1.2, alpha=0.8)
    ax_rpm.plot(t, rpm_enc_r, 'r--', label='right enc', lw=1.2, alpha=0.8)
    ax_rpm.axvline(STALL_BOOST_SECS, color='gray', ls=':', alpha=0.5)
    ax_rpm.set_xlabel('time (s)')
    ax_rpm.set_ylabel('RPM')
    ax_rpm.set_title('Per-wheel commanded vs encoder RPM')
    ax_rpm.legend(fontsize=8)
    ax_rpm.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    print(f"Saved: {out_path}")


def plot(data, out_path="pathfinding_arc_test.png"):
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')   # non-interactive backend — required on headless Pi
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Skipping plots: {e}")
        return

    mode = data.get("mode", "spin")

    if mode == "slow-spin":
        _plot_slow_spin(data, out_path)
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # ── Angle over time ────────────────────────────────────
    ax = axes[0]
    if mode == "center":
        angles_deg = data.get("target_angle", [])
        title = 'Target angle in FOV'
        ylabel = 'target angle (deg)'
        ax.axhline(0, color='gray', ls=':', alpha=0.6, label='center')
        ax.axhspan(-CENTER_DEADBAND_DEG, CENTER_DEADBAND_DEG,
                   color='green', alpha=0.12, label='deadband')
    elif data.get('drove'):
        angles_deg = data['encoder_degrees']
        title = 'Encoder angle over time'
        ylabel = 'encoder angle (deg)'
        ax.axhline(360, color='gray', ls=':', alpha=0.6, label='360°')
    else:
        angles_deg = [math.degrees(h) for h in data['rt']]
        title = 'Sim heading over time'
        ylabel = 'sim heading (deg)'
        ax.axhline(360, color='gray', ls=':', alpha=0.6, label='360°')
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
    ax.set_xlabel('time (s)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Turn rate ──────────────────────────────────────────
    ax = axes[1]
    ax.plot(data['t'], np.degrees(data['omega']), 'b-')
    cap = CENTER_MAX_TURN_DEG if mode == "center" else math.degrees(P.MAX_TURN)
    ax.axhline(cap, color='gray', ls=':', alpha=0.5, label=f'cap {cap:.0f}°/s')
    ax.axhline(-cap, color='gray', ls=':', alpha=0.5)
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
    parser.add_argument("--mode", choices=("center", "spin", "slow-spin"), default="center",
                        help="center keeps shopper centred; spin runs 360°; slow-spin turns in place slowly")
    parser.add_argument("--source", choices=("camera", "udp"), default="camera",
                        help="target angle source for center mode (default: camera)")
    parser.add_argument("--no-drive", action="store_true",
                        help="simulation only — don't open the ESP32 serial port")
    parser.add_argument("--no-display", action="store_true",
                        help="suppress OpenCV windows in camera center mode")
    parser.add_argument("--port", default=None,
                        help=f"serial port override (default: {_MOTOR_PORT})")
    parser.add_argument("--countdown", type=int, default=3,
                        help="seconds before motors start (default: 3)")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="run time in seconds for center/slow-spin; 0 runs until Ctrl-C (default: 30)")
    parser.add_argument("--direction", choices=("left", "right"), default="left",
                        help="slow-spin direction (default: left)")
    parser.add_argument("--sim-angle", type=float, default=None,
                        help="simulate a fixed target angle instead of listening for UDP")
    args = parser.parse_args()

    if args.mode == "spin":
        data = run_spin(drive=not args.no_drive, port=args.port, countdown=args.countdown)
    elif args.mode == "slow-spin":
        spin_dir = 1.0 if args.direction == "left" else -1.0
        data = run_slow_spin(drive=not args.no_drive, port=args.port,
                             countdown=args.countdown, duration=args.duration,
                             direction=spin_dir)
    elif args.sim_angle is not None or args.source == "udp":
        if args.source == "udp" and args.sim_angle is None:
            print(f"UDP centering mode: listening on port {P.UDP_PORT}.")
        data = run_center(drive=not args.no_drive, port=args.port,
                          countdown=args.countdown, duration=args.duration,
                          sim_angle=args.sim_angle)
    else:
        no_display = args.no_display
        if not no_display and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            no_display = True
            print("No display detected — running headless (use --no-display to silence this message).")
        data = run_center_camera(drive=not args.no_drive, port=args.port,
                                 countdown=args.countdown, duration=args.duration,
                                 no_display=no_display)
    plot(data)
