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
    python3 pathfinding_arc_test.py --mode follow                  # forward/backward, live camera
    python3 pathfinding_arc_test.py --mode follow --no-drive --sim-dist 3.5  # simulation


    logic sequence for general turning:
    inertia = something (via calibration)

    Function drift (initial_rpm, final_rpm, inertia)
	    Some calculation, such as (initial-final)*inertia)
	    Return drift

    Function turn (theta)

	    Send kick_rpm for kick_ticks
	    Kick_drift = drift(kick_rpm, turn_rpm, inertia)
	    phi = kick_rpm * kick_ticks / ticks_per_sec
	
        turn_Drift = drift(turn_rpm, 0, inertia)
        theta_2 = theta - phi - turn_drift - kick_drift

	    Send turn_rpm for t = theta_2/turn_rpm 

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
CENTER_DEADBAND_DEG = 3.0
CENTER_REACQUIRE_S  = 0.3
CENTER_START_RPM    = 0.3
CENTER_MIN_RUN_RPM  = 0.3
CENTER_KICK_RPM  =8
CENTER_KICK_RELEASE_TICKS = 30
CENTER_SEARCH_KICK_RPM = 8
CENTER_SEARCH_KICK_RELEASE_TICKS = 30
CENTER_SEARCH_DISTANCE_SPLIT_M = 3.0
CENTER_SEARCH_NEAR_MAX_RPM = 0.35
CENTER_SEARCH_FAR_MAX_RPM = 0.3
CENTER_SEARCH_RAMP_STEP_RPM = 0.1
CENTER_SEARCH_RAMP_HOLD_S = 1
CENTER_SEARCH_STALL_TICKS = 5

FOLLOW_DIST_DEADBAND_M    = 0.05   # stop commanding when within this of HOLD_DIST

# Slow-spin / stall-test parameters.  The test commands wheel RPMs, not raw
# PWM, because the ESP32 firmware exposes an RPM serial protocol.
STALL_RAMP_UP_START_RPM = 0.05   # first commanded wheel RPM during stall search
STALL_RAMP_UP_STEP_RPM  = 0.05   # RPM added after each no-movement hold
STALL_RAMP_DOWN_STEP_RPM = 0.05  # RPM removed after each moving hold
STALL_STEP_HOLD_SECS    = 0.5  # time to hold each RPM before changing it
STALL_MAX_RPM           = 2.0  # safety cap for the ramp-up search
STALL_OVERCOME_CUM_RPM  = 100.0 # cumulative encoder RPM in a hold before stall is overcome

# =============================================================================


def integrate_kinematics(pos, heading, v_left, v_right, dt):
    v_fwd = (v_left + v_right) / 2.0
    omega = (v_right - v_left) / P.TRACK_M
    new_heading = (heading + omega * dt) % (2 * math.pi)
    new_x = pos[0] + v_fwd * math.cos(heading) * dt
    new_y = pos[1] + v_fwd * math.sin(heading) * dt
    return [new_x, new_y], new_heading, v_fwd, omega


def update_odometry_from_deltas(odom, d_l, d_r):
    d_right = d_l * P.M_PER_PULSE
    d_left = d_r * P.M_PER_PULSE
    d_centre = (d_right + d_left) / 2
    d_theta = (d_left - d_right) / P.TRACK_M
    mid = odom.heading + d_theta / 2
    odom.x += d_centre * math.cos(mid)
    odom.y += d_centre * math.sin(mid)
    odom.heading = (odom.heading + d_theta) % (2 * math.pi)
    return [odom.x, odom.y], odom.heading


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


def wheel_rpm_to_spin_omega(rpm, direction=1.0):
    wheel_speed = abs(rpm) * P.WHEEL_CIRC / 60.0
    return math.copysign(2.0 * wheel_speed / P.TRACK_M, direction)


def spin_omega_to_wheel_rpm(omega):
    wheel_speed = abs(omega) * P.TRACK_M / 2.0
    return wheel_speed / P.WHEEL_CIRC * 60.0


def encoder_delta_to_wheel_rpm(delta_ticks):
    ticks_per_wheel_rev = P.ENCODER_PPR * getattr(P, 'GEAR_RATIO', 1)
    wheel_revs = delta_ticks / ticks_per_wheel_rev
    return wheel_revs / P.DT * 60.0


def search_max_rpm_for_distance(dist_m):
    if dist_m is not None and dist_m < CENTER_SEARCH_DISTANCE_SPLIT_M:
        return CENTER_SEARCH_NEAR_MAX_RPM
    return CENTER_SEARCH_FAR_MAX_RPM


def clamp_center_rpm(rpm, max_rpm=CENTER_SEARCH_FAR_MAX_RPM):
    if rpm <= 0.0:
        return 0.0
    return min(max(rpm, CENTER_MIN_RUN_RPM), max_rpm)


class CenterRpmCommand:
    def __init__(self):
        self._moving = False
        self._last_sign = 0.0
        self._kick_active = False
        self._kick_l_ticks = 0
        self._kick_r_ticks = 0

    def reset(self):
        self._moving = False
        self._last_sign = 0.0
        self._kick_active = False
        self._kick_l_ticks = 0
        self._kick_r_ticks = 0

    @property
    def kick_active(self):
        return self._kick_active

    def force_kick(self):
        self._kick_active = True
        self._kick_l_ticks = 0
        self._kick_r_ticks = 0

    def command(
            self, rpm, direction, encoder_delta=(0, 0),
            kick_rpm=CENTER_KICK_RPM,
            kick_release_ticks=CENTER_KICK_RELEASE_TICKS,
            max_rpm=CENTER_SEARCH_FAR_MAX_RPM):
        rpm = clamp_center_rpm(rpm, max_rpm)
        if rpm <= 0.0:
            self.reset()
            return 0.0

        sign = math.copysign(1.0, direction)
        if not self._moving or sign != self._last_sign:
            self._kick_active = True
            self._kick_l_ticks = 0
            self._kick_r_ticks = 0
        self._moving = True
        self._last_sign = sign

        d_l, d_r = encoder_delta
        self._kick_l_ticks += abs(d_l)
        self._kick_r_ticks += abs(d_r)
        if (self._kick_l_ticks >= kick_release_ticks or
                self._kick_r_ticks >= kick_release_ticks):
            self._kick_active = False

        if self._kick_active:
            return wheel_rpm_to_spin_omega(kick_rpm, sign)
        return wheel_rpm_to_spin_omega(rpm, sign)


def center_turn_request(angle_deg):
    if angle_deg is None or abs(angle_deg) <= CENTER_DEADBAND_DEG:
        return 0.0, 0.0
    return CENTER_START_RPM, -angle_deg


class FollowRpmCommand:
    """Kick-on-start / kick-on-direction-change stall combat for linear motion.

    Mirrors CenterRpmCommand but drives forward/backward instead of rotating.
    Returns v_forward in m/s (positive = forward, negative = reverse).
    """

    def __init__(self):
        self._moving = False
        self._last_sign = 0.0
        self._kick_active = False
        self._kick_l_ticks = 0
        self._kick_r_ticks = 0

    def reset(self):
        self._moving = False
        self._last_sign = 0.0
        self._kick_active = False
        self._kick_l_ticks = 0
        self._kick_r_ticks = 0

    @property
    def kick_active(self):
        return self._kick_active

    def command(self, rpm, direction, encoder_delta=(0, 0),
                kick_rpm=CENTER_KICK_RPM,
                kick_release_ticks=CENTER_KICK_RELEASE_TICKS):
        """rpm: desired wheel RPM (>= 0); direction: +1 forward, -1 reverse.
        Returns v_forward in m/s with the correct sign."""
        if rpm <= 0.0:
            self.reset()
            return 0.0

        sign = math.copysign(1.0, direction)
        if not self._moving or sign != self._last_sign:
            self._kick_active = True
            self._kick_l_ticks = 0
            self._kick_r_ticks = 0
        self._moving = True
        self._last_sign = sign

        d_l, d_r = encoder_delta
        self._kick_l_ticks += abs(d_l)
        self._kick_r_ticks += abs(d_r)
        if (self._kick_l_ticks >= kick_release_ticks or
                self._kick_r_ticks >= kick_release_ticks):
            self._kick_active = False

        actual_rpm = kick_rpm if self._kick_active else rpm
        v = actual_rpm * P.WHEEL_CIRC / 60.0
        return math.copysign(v, sign)


def run_slow_spin(drive=True, port=None, countdown=3, duration=30.0, direction=1.0):
    pos     = list(START_POS)
    heading = START_HEADING
    motors  = open_motors(drive=drive, port=port, countdown=countdown, action="slow-spinning")

    print("Slow spin stall test: ramp wheel RPM up until encoder ticks appear, "
          "then ramp down until ticks stop.")
    print(f"Ramp up: {STALL_RAMP_UP_START_RPM:.1f} RPM +{STALL_RAMP_UP_STEP_RPM:.2f} "
          f"every {STALL_STEP_HOLD_SECS:.2f}s, max {STALL_MAX_RPM:.1f} RPM.")
    print(f"Ramp down: -{STALL_RAMP_DOWN_STEP_RPM:.2f} RPM every "
          f"{STALL_STEP_HOLD_SECS:.2f}s.")

    times, robot_xs, robot_ys, robot_thetas = [], [], [], []
    omegas_cmd, omegas_enc, command_rpms, enc_rpm_mags = [], [], [], []
    v_lefts, v_rights = [], []
    enc_right_cum, enc_left_cum = [], []
    enc_right_delta, enc_left_delta = [], []
    cum_l, cum_r = 0, 0
    hold_ticks = 0
    hold_enc_rpm_total = 0.0
    hold_enc_rpm_peak = 0.0
    phase = "ramp_up"
    command_rpm = STALL_RAMP_UP_START_RPM
    next_step_t = STALL_STEP_HOLD_SECS
    initial_move_rpm = None
    initial_move_enc_rpm = None
    lowest_move_rpm = None
    lowest_move_enc_rpm = None
    stall_rpm = None
    stall_omega_deg  = None
    start = time.monotonic()
    i = 0

    flush_motors(motors)
    try:
        while True:
            t_loop0 = time.monotonic()
            t = t_loop0 - start

            wheel_speed = command_rpm * P.WHEEL_CIRC / 60.0
            omega_cmd = math.copysign(2.0 * wheel_speed / P.TRACK_M, direction)

            v_left, v_right = P._wheel_commands(0.0, omega_cmd)
            if motors is not None:
                motors.send(v_left, v_right)
                d_l, d_r = motors.read_encoder_deltas()
            else:
                d_l, d_r = 0, 0

            # Encoder-derived angular velocity
            enc_rpm_l = encoder_delta_to_wheel_rpm(d_l)
            enc_rpm_r = encoder_delta_to_wheel_rpm(d_r)
            vl_enc = enc_rpm_l * P.WHEEL_CIRC / 60.0
            vr_enc = enc_rpm_r * P.WHEEL_CIRC / 60.0
            omega_enc = (vr_enc - vl_enc) / P.TRACK_M
            enc_rpm_mag = max(abs(enc_rpm_l), abs(enc_rpm_r))
            tick_count = abs(d_l) + abs(d_r)
            hold_ticks += tick_count
            hold_enc_rpm_total += enc_rpm_mag
            hold_enc_rpm_peak = max(hold_enc_rpm_peak, enc_rpm_mag)

            cum_l += d_l; cum_r += d_r
            enc_left_delta.append(d_l); enc_right_delta.append(d_r)
            enc_left_cum.append(cum_l); enc_right_cum.append(cum_r)

            times.append(t)
            robot_xs.append(pos[0]); robot_ys.append(pos[1])
            robot_thetas.append(heading)

            pos, heading, _v_fwd, _ = integrate_kinematics(
                pos, heading, v_left, v_right, P.DT)
            omegas_cmd.append(omega_cmd)
            omegas_enc.append(omega_enc)
            command_rpms.append(command_rpm)
            enc_rpm_mags.append(enc_rpm_mag)
            v_lefts.append(v_left); v_rights.append(v_right)

            if motors is None:
                if t >= next_step_t:
                    command_rpm += STALL_RAMP_UP_STEP_RPM
                    next_step_t = t + STALL_STEP_HOLD_SECS
            elif t >= next_step_t:
                stall_overcome = hold_enc_rpm_total >= STALL_OVERCOME_CUM_RPM
                hold_moved = hold_ticks > 0
                if phase == "ramp_up" and stall_overcome:
                    initial_move_rpm = command_rpm
                    initial_move_enc_rpm = hold_enc_rpm_peak
                    lowest_move_rpm = command_rpm
                    lowest_move_enc_rpm = hold_enc_rpm_peak
                    phase = "ramp_down"
                    print(f"Initial movement: cmd={initial_move_rpm:.2f} wheel RPM, "
                          f"enc≈{initial_move_enc_rpm:.2f} RPM  (t={t:.2f}s)")
                elif phase == "ramp_up":
                    if command_rpm >= STALL_MAX_RPM:
                        print(f"Reached {STALL_MAX_RPM:.1f} RPM without encoder movement.")
                        break
                    command_rpm = min(command_rpm + STALL_RAMP_UP_STEP_RPM, STALL_MAX_RPM)
                elif phase == "ramp_down" and hold_moved:
                    lowest_move_rpm = command_rpm
                    lowest_move_enc_rpm = hold_enc_rpm_peak
                    command_rpm = max(0.0, command_rpm - STALL_RAMP_DOWN_STEP_RPM)
                elif phase == "ramp_down":
                    stall_rpm = command_rpm
                    stall_omega_deg = math.degrees(abs(omega_cmd))
                    print(f"Stopped moving: lowest moving cmd={lowest_move_rpm:.2f} "
                          f"wheel RPM, first stopped cmd={stall_rpm:.2f} wheel RPM  "
                          f"(t={t:.2f}s)")
                    break

                hold_ticks = 0
                hold_enc_rpm_total = 0.0
                hold_enc_rpm_peak = 0.0
                next_step_t = t + STALL_STEP_HOLD_SECS

            if duration > 0 and t >= duration:
                break

            if i % max(1, int(1.0 / P.DT)) == 0:
                print(f"t={t:5.1f}s  {phase:9s}  cmd={command_rpm:5.2f} wheel RPM  "
                      f"enc={enc_rpm_mag:5.2f} RPM  "
                      f"hold_rpm={hold_enc_rpm_total:6.1f}/{STALL_OVERCOME_CUM_RPM:.0f}  "
                      f"hold_ticks={hold_ticks:4d}  "
                      f"ticks=({d_l:+d},{d_r:+d})")
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

    if initial_move_rpm is not None:
        print(f"\nInitial RPM to overcome stall: {initial_move_rpm:.2f} wheel RPM "
              f"(encoder≈{initial_move_enc_rpm:.2f} RPM)")
    else:
        print("\nInitial movement was not detected.")

    if lowest_move_rpm is not None and stall_rpm is not None:
        print(f"Lowest RPM before stalling again: {lowest_move_rpm:.2f} wheel RPM "
              f"(encoder≈{lowest_move_enc_rpm:.2f} RPM)")
        print(f"First no-tick command after stall: {stall_rpm:.2f} wheel RPM")
    elif stall_omega_deg is not None:
        print(f"Stall speed: {stall_omega_deg:.1f}°/s")
    else:
        print("No second stall detected within duration.")

    return dict(
        mode="slow-spin",
        t=times,
        rx=robot_xs, ry=robot_ys, rt=robot_thetas,
        omega=omegas_cmd,
        omega_enc=omegas_enc,
        command_rpm=command_rpms,
        encoder_rpm=enc_rpm_mags,
        stall_omega_deg=stall_omega_deg,
        initial_move_rpm=initial_move_rpm,
        initial_move_enc_rpm=initial_move_enc_rpm,
        lowest_move_rpm=lowest_move_rpm,
        lowest_move_enc_rpm=lowest_move_enc_rpm,
        stall_rpm=stall_rpm,
        v_left=v_lefts, v_right=v_rights,
        enc_l_cum=enc_left_cum, enc_r_cum=enc_right_cum,
        enc_l_delta=enc_left_delta, enc_r_delta=enc_right_delta,
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

    try:
        cap = Y.IMX500Capture(model_path=Y.RPK_MODEL_PATH, width=640, height=480, fps=30)
    except RuntimeError as e:
        raise SystemExit(
            f"ERROR: could not open IMX500 camera: {e}\n"
            "Check that the Raspberry Pi AI Camera is connected, enabled, and not already in use."
        )

    motors = open_motors(drive=drive, port=port, countdown=countdown, action="centring")
    odometry = P.Odometry(lambda: (0, 0))
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
    last_target_dist_m = None
    last_search_omega_sign = 1.0
    search_rpm = CENTER_START_RPM
    next_search_step_t = 0.0
    search_stall_ticks = 0
    rpm_command = CenterRpmCommand()
    omega = 0.0
    v_left = v_right = 0.0
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
            if target_row is not None:
                last_target_dist_m = target_row[3]
            desired_rpm = 0.0
            desired_direction = 0.0
            search_mode = False
            search_max_rpm = search_max_rpm_for_distance(last_target_dist_m)
            if angle_deg is None:
                target_visible_since = None
                command_angle = None
            else:
                search_rpm = CENTER_MIN_RUN_RPM
                next_search_step_t = t + CENTER_SEARCH_RAMP_HOLD_S
                if abs(angle_deg) > CENTER_DEADBAND_DEG:
                    last_search_omega_sign = math.copysign(1.0, -angle_deg)
                if target_visible_since is None:
                    target_visible_since = loop_start
                command_angle = (
                    angle_deg
                    if loop_start - target_visible_since >= CENTER_REACQUIRE_S
                    else None
                )
            if command_angle is not None:
                desired_rpm, desired_direction = center_turn_request(command_angle)
            elif angle_deg is None:
                search_mode = True
                desired_rpm = min(search_rpm, search_max_rpm)
                desired_direction = last_search_omega_sign
                if t >= next_search_step_t:
                    search_rpm = min(
                        search_max_rpm,
                        search_rpm + CENTER_SEARCH_RAMP_STEP_RPM,
                    )
                    next_search_step_t = t + CENTER_SEARCH_RAMP_HOLD_S

            now = time.monotonic()
            if now - last_tick >= P.DT:
                if motors is not None:
                    d_l, d_r = motors.read_encoder_deltas()
                else:
                    d_l, d_r = 0, 0
                search_at_max = (
                    search_mode and desired_rpm >= search_max_rpm
                )
                search_moved = abs(d_l) > 0 or abs(d_r) > 0
                if not search_at_max or search_moved or rpm_command.kick_active:
                    search_stall_ticks = 0
                else:
                    search_stall_ticks += 1
                    if search_stall_ticks >= CENTER_SEARCH_STALL_TICKS:
                        rpm_command.force_kick()
                        search_stall_ticks = 0
                kick_rpm = CENTER_SEARCH_KICK_RPM if search_mode else CENTER_KICK_RPM
                kick_release_ticks = (
                    CENTER_SEARCH_KICK_RELEASE_TICKS
                    if search_mode else CENTER_KICK_RELEASE_TICKS
                )
                omega = rpm_command.command(
                    desired_rpm, desired_direction, encoder_delta=(d_l, d_r),
                    kick_rpm=kick_rpm,
                    kick_release_ticks=kick_release_ticks,
                    max_rpm=search_max_rpm)
                v_left, v_right = P._wheel_commands(0.0, omega)
                pos, heading = update_odometry_from_deltas(odometry, d_l, d_r)
                P.S.pos = list(pos)
                P.S.heading = heading
                cum_l += d_l; cum_r += d_r

                if motors is not None:
                    motors.send(v_left, v_right)
                last_tick = now

                times.append(t)
                robot_xs.append(P.S.pos[0]); robot_ys.append(P.S.pos[1])
                robot_thetas.append(P.S.heading)
                target_angles.append(float("nan") if angle_deg is None else angle_deg)
                omegas.append(omega)
                v_lefts.append(v_left); v_rights.append(v_right)
                enc_right_delta.append(d_l); enc_left_delta.append(d_r)
                enc_right_cum.append(cum_l); enc_left_cum.append(cum_r)

                if i % max(1, int(0.5 / P.DT)) == 0:
                    cmd_rpm = spin_omega_to_wheel_rpm(omega)
                    if angle_deg is None:
                        label = f"search {cmd_rpm:.2f}rpm"
                    elif command_angle is None:
                        label = f"reacquire {angle_deg:+.1f}°"
                    else:
                        label = f"angle={angle_deg:+.1f}°"
                    print(f"t={t:5.1f}s  {label:<16} "
                          f"cmd={cmd_rpm:4.2f}rpm  "
                          f"omega={math.degrees(omega):+6.1f}°/s")
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
    rpm_command = CenterRpmCommand()
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
            desired_rpm = 0.0
            desired_direction = 0.0
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
            if command_angle is not None:
                desired_rpm, desired_direction = center_turn_request(command_angle)

            if motors is not None:
                d_l, d_r = motors.read_encoder_deltas()
            else:
                d_l, d_r = 0, 0
            omega = rpm_command.command(
                desired_rpm, desired_direction, encoder_delta=(d_l, d_r))
            v_left, v_right = P._wheel_commands(0.0, omega)

            if motors is not None:
                motors.send(v_left, v_right)
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
                cmd_rpm = spin_omega_to_wheel_rpm(omega)
                print(f"t={t:5.1f}s  {label:<16} "
                      f"cmd={cmd_rpm:4.2f}rpm  "
                      f"omega={math.degrees(omega):+6.1f}°/s")
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


def run_follow(drive=True, port=None, countdown=3, duration=30.0, sim_dist=None):
    """Distance-only follow test: PD forward/backward, omega forced to zero.

    Uses the same kick-on-start stall combat as the center spin tests
    (CENTER_KICK_RPM / CENTER_KICK_RELEASE_TICKS).  Reads distance from UDP
    unless --sim-dist is given.
    """
    motors  = open_motors(drive=drive, port=port, countdown=countdown, action="following")
    receiver = None if sim_dist is not None else P.TargetReceiver()

    print("Follow mode: distance PD control only, omega=0, no angle correction.")
    print(f"Hold distance: {P.HOLD_DIST:.2f} m  "
          f"Kp={P.DIST_KP}  Kd={P.DIST_KD}  "
          f"kick={CENTER_KICK_RPM} RPM / {CENTER_KICK_RELEASE_TICKS} ticks")
    if sim_dist is not None:
        print(f"Simulated target distance: {sim_dist:.2f} m")
    else:
        print(f"Listening for UDP target readings on port {P.UDP_PORT}.")

    times, v_fwds, v_lefts, v_rights = [], [], [], []
    target_dists, dist_errors = [], []
    enc_right_cum, enc_left_cum = [], []
    enc_right_delta, enc_left_delta = [], []
    cum_l, cum_r = 0, 0
    prev_dist = None
    start = time.monotonic()
    rpm_command = FollowRpmCommand()
    i = 0

    flush_motors(motors)
    try:
        while True:
            t_loop0 = time.monotonic()
            t = t_loop0 - start
            if duration > 0 and t >= duration:
                break

            reading = (sim_dist, 0.0) if sim_dist is not None else receiver.get()
            dist_m = reading[0] if reading is not None else None

            if dist_m is not None:
                e = dist_m - P.HOLD_DIST
                de_dt = (dist_m - prev_dist) / P.DT if prev_dist is not None else 0.0
                v_pd = P.DIST_KP * e + P.DIST_KD * de_dt
                prev_dist = dist_m
                if abs(e) > FOLLOW_DIST_DEADBAND_M:
                    desired_rpm = max(abs(v_pd) / P.WHEEL_CIRC * 60.0, CENTER_MIN_RUN_RPM)
                    desired_dir = math.copysign(1.0, v_pd)
                else:
                    desired_rpm = 0.0
                    desired_dir = 0.0
            else:
                desired_rpm = 0.0
                desired_dir = 0.0
                prev_dist = None

            if motors is not None:
                d_l, d_r = motors.read_encoder_deltas()
            else:
                d_l, d_r = 0, 0

            v_forward = rpm_command.command(
                desired_rpm, desired_dir, encoder_delta=(d_l, d_r))
            v_left, v_right = P._wheel_commands(v_forward, 0.0)

            if motors is not None:
                motors.send(v_left, v_right)

            cum_l += d_l; cum_r += d_r
            enc_right_delta.append(d_l); enc_left_delta.append(d_r)
            enc_right_cum.append(cum_l); enc_left_cum.append(cum_r)

            times.append(t)
            v_fwds.append(v_forward)
            v_lefts.append(v_left); v_rights.append(v_right)
            target_dists.append(float("nan") if dist_m is None else dist_m)
            dist_errors.append(float("nan") if dist_m is None else e)

            if i % max(1, int(0.5 / P.DT)) == 0:
                if dist_m is None:
                    label = "no target          "
                else:
                    label = f"dist={dist_m:.2f}m e={e:+.2f}m"
                fwd_rpm = v_forward / P.WHEEL_CIRC * 60.0
                kick_str = " KICK" if rpm_command.kick_active else "     "
                print(f"t={t:5.1f}s  {label:<22}  cmd={fwd_rpm:+6.2f}rpm{kick_str}")
            i += 1

            elapsed = time.monotonic() - t_loop0
            if elapsed < P.DT:
                time.sleep(P.DT - elapsed)
    except KeyboardInterrupt:
        print("\nFollow stopped by user.")
    finally:
        if motors is not None:
            print("\nStopping motors.")
            motors.stop()

    return dict(
        mode="follow",
        t=times,
        v_fwd=v_fwds,
        v_left=v_lefts, v_right=v_rights,
        target_dist=target_dists,
        dist_error=dist_errors,
        enc_l_cum=enc_right_cum, enc_r_cum=enc_left_cum,
        enc_l_delta=enc_right_delta, enc_r_delta=enc_left_delta,
        drove=(motors is not None),
    )


def run_follow_camera(drive=True, port=None, countdown=3, duration=30.0, no_display=False):
    """Camera-driven distance-only follow test. Uses IMX500 for distance readings.

    Identical control logic to run_follow but reads distance from the live
    camera instead of UDP/sim. Omega is forced to zero — no angle correction.
    """
    Y = import_yolo_detect()
    if not Y.RPK_MODEL_PATH.exists():
        raise SystemExit(
            f"ERROR: {Y.RPK_MODEL_PATH} not found. Install with: sudo apt install imx500-models"
        )

    try:
        cap = Y.IMX500Capture(model_path=Y.RPK_MODEL_PATH, width=640, height=480, fps=30)
    except RuntimeError as e:
        raise SystemExit(
            f"ERROR: could not open IMX500 camera: {e}\n"
            "Check that the Raspberry Pi AI Camera is connected, enabled, and not already in use."
        )

    motors = open_motors(drive=drive, port=port, countdown=countdown, action="following")
    odometry = P.Odometry(lambda: (0, 0))
    tracker = Y.sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=60,
        minimum_matching_threshold=0.8,
        frame_rate=30,
    )
    smooth_state = {}
    target_lock = Y.TargetLock()

    print("Camera follow mode: distance PD control only, omega=0, no angle correction.")
    print(f"Hold distance: {P.HOLD_DIST:.2f} m  Kp={P.DIST_KP}  Kd={P.DIST_KD}  "
          f"kick={CENTER_KICK_RPM} RPM / {CENTER_KICK_RELEASE_TICKS} ticks")
    print("Press Q in the Cart View window to stop." if not no_display else "Display disabled.")

    times, v_fwds, v_lefts, v_rights = [], [], [], []
    target_dists, dist_errors = [], []
    enc_right_cum, enc_left_cum = [], []
    enc_right_delta, enc_left_delta = [], []
    cum_l, cum_r = 0, 0
    prev_dist = None
    start = time.monotonic()
    last_tick = time.monotonic()
    frame_times = []
    target_visible_since = None
    rpm_command = FollowRpmCommand()
    v_forward = 0.0
    v_left = v_right = 0.0
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
            dist_m = target_row[3] if target_row is not None else None

            now = time.monotonic()
            if now - last_tick >= P.DT:
                if motors is not None:
                    d_l, d_r = motors.read_encoder_deltas()
                else:
                    d_l, d_r = 0, 0

                if dist_m is not None:
                    if target_visible_since is None:
                        target_visible_since = now
                    e = dist_m - P.HOLD_DIST
                    de_dt = (dist_m - prev_dist) / P.DT if prev_dist is not None else 0.0
                    v_pd = P.DIST_KP * e + P.DIST_KD * de_dt
                    prev_dist = dist_m
                    if abs(e) > FOLLOW_DIST_DEADBAND_M:
                        desired_rpm = max(abs(v_pd) / P.WHEEL_CIRC * 60.0, CENTER_MIN_RUN_RPM)
                        desired_dir = math.copysign(1.0, v_pd)
                    else:
                        desired_rpm = 0.0
                        desired_dir = 0.0
                else:
                    target_visible_since = None
                    desired_rpm = 0.0
                    desired_dir = 0.0
                    prev_dist = None

                v_forward = rpm_command.command(
                    desired_rpm, desired_dir, encoder_delta=(d_l, d_r))
                v_left, v_right = P._wheel_commands(v_forward, 0.0)

                pos, heading = update_odometry_from_deltas(odometry, d_l, d_r)
                P.S.pos = list(pos)
                P.S.heading = heading
                cum_l += d_l; cum_r += d_r

                if motors is not None:
                    motors.send(v_left, v_right)
                last_tick = now

                times.append(t)
                v_fwds.append(v_forward)
                v_lefts.append(v_left); v_rights.append(v_right)
                target_dists.append(float("nan") if dist_m is None else dist_m)
                dist_errors.append(float("nan") if dist_m is None else e)
                enc_right_delta.append(d_l); enc_left_delta.append(d_r)
                enc_right_cum.append(cum_l); enc_left_cum.append(cum_r)

                if i % max(1, int(0.5 / P.DT)) == 0:
                    if dist_m is None:
                        label = "no target          "
                    else:
                        fwd_rpm = v_forward / P.WHEEL_CIRC * 60.0
                        kick_str = " KICK" if rpm_command.kick_active else "     "
                        label = f"dist={dist_m:.2f}m e={e:+.2f}m"
                        print(f"t={t:5.1f}s  {label:<22}  cmd={fwd_rpm:+6.2f}rpm{kick_str}")
                i += 1

            frame_times.append(time.time())
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps_live = (
                (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                if len(frame_times) > 1 else 0.0
            )
            fwd_rpm = v_forward / P.WHEEL_CIRC * 60.0
            dist_str = f"dist={dist_m:.2f}m" if dist_m is not None else "no target"
            Y.cv2.putText(out, f"FOLLOW  FPS:{fps_live:.1f}  {dist_str}  cmd={fwd_rpm:+.2f}rpm",
                          (10, 25), Y.cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            if not no_display:
                Y.overlay_map(out, rows)
                Y.cv2.imshow("Cart View", out)
                Y.cv2.imshow("World Map", Y.draw_world_map(rows))
                if Y.cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\nCamera follow stopped by user.")
    finally:
        if motors is not None:
            print("\nStopping motors.")
            motors.stop()
        cap.release()
        if not no_display:
            Y.cv2.destroyAllWindows()

    return dict(
        mode="follow",
        t=times,
        v_fwd=v_fwds,
        v_left=v_lefts, v_right=v_rights,
        target_dist=target_dists,
        dist_error=dist_errors,
        enc_l_cum=enc_right_cum, enc_r_cum=enc_left_cum,
        enc_l_delta=enc_right_delta, enc_r_delta=enc_left_delta,
        drove=(motors is not None),
    )


def _plot_follow(data, out_path):
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Skipping plots: {e}")
        return

    t          = data['t']
    dist       = data['target_dist']
    dist_err   = data['dist_error']
    rpm_cmd_l  = [v / P.WHEEL_CIRC * 60 for v in data['v_left']]
    rpm_cmd_r  = [v / P.WHEEL_CIRC * 60 for v in data['v_right']]
    rpm_enc_l  = [encoder_delta_to_wheel_rpm(d) for d in data['enc_l_delta']]
    rpm_enc_r  = [encoder_delta_to_wheel_rpm(d) for d in data['enc_r_delta']]

    fig, (ax_dist, ax_rpm) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # ── Distance over time ─────────────────────────────────
    ax_dist.plot(t, dist, 'b-', label='measured distance', lw=1.5)
    ax_dist.axhline(P.HOLD_DIST, color='green', ls='--', lw=1.2,
                    label=f'hold dist {P.HOLD_DIST:.2f} m')
    ax_dist.axhspan(P.HOLD_DIST - FOLLOW_DIST_DEADBAND_M,
                    P.HOLD_DIST + FOLLOW_DIST_DEADBAND_M,
                    color='green', alpha=0.10, label='deadband')
    ax_dist.set_ylabel('distance (m)')
    ax_dist.set_title('Target distance vs hold distance')
    ax_dist.legend(fontsize=8)
    ax_dist.grid(True, alpha=0.3)

    # ── Per-wheel commanded vs encoder RPM ────────────────
    ax_rpm.plot(t, rpm_cmd_l, 'b-',  label='left cmd',  lw=1.5)
    ax_rpm.plot(t, rpm_cmd_r, 'r-',  label='right cmd', lw=1.5)
    ax_rpm.plot(t, rpm_enc_l, 'b--', label='left enc',  lw=1.2, alpha=0.8)
    ax_rpm.plot(t, rpm_enc_r, 'r--', label='right enc', lw=1.2, alpha=0.8)
    ax_rpm.axhline(0, color='gray', ls=':', alpha=0.5)
    ax_rpm.set_xlabel('time (s)')
    ax_rpm.set_ylabel('RPM')
    ax_rpm.set_title('Per-wheel commanded vs encoder RPM')
    ax_rpm.legend(fontsize=8)
    ax_rpm.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    print(f"Saved: {out_path}")


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
    cmd_wheel_rpm = data.get(
        'command_rpm',
        [spin_omega_to_wheel_rpm(o) for o in data['omega']],
    )
    enc_wheel_rpm = data.get(
        'encoder_rpm',
        [spin_omega_to_wheel_rpm(o) for o in data.get('omega_enc', [0.0] * len(t))],
    )
    rpm_cmd_l    = [v / P.WHEEL_CIRC * 60 for v in data['v_left']]
    rpm_cmd_r    = [v / P.WHEEL_CIRC * 60 for v in data['v_right']]
    rpm_enc_l    = [encoder_delta_to_wheel_rpm(d) for d in data['enc_l_delta']]
    rpm_enc_r    = [encoder_delta_to_wheel_rpm(d) for d in data['enc_r_delta']]

    fig, (ax_omega, ax_rpm) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # ── Commanded vs encoder equivalent wheel RPM ─────────
    ax_omega.plot(t, cmd_wheel_rpm, 'b-',  label='commanded wheel RPM', lw=1.5)
    ax_omega.plot(t, enc_wheel_rpm, 'r-',  label='encoder wheel RPM',   lw=1.2, alpha=0.85)
    stall = data.get('stall_rpm')
    if stall is not None:
        ax_omega.axhline(stall, color='orange', ls='--', lw=1.2,
                         label=f'stall {stall:.2f} RPM')
    ax_omega.set_ylabel('wheel RPM')
    ax_omega.set_title('Commanded vs encoder wheel RPM')
    ax_omega.legend(fontsize=8)
    ax_omega.grid(True, alpha=0.3)

    # ── Per-wheel commanded vs encoder RPM ────────────────
    ax_rpm.plot(t, rpm_cmd_l, 'b-',  label='left cmd',  lw=1.5)
    ax_rpm.plot(t, rpm_cmd_r, 'r-',  label='right cmd', lw=1.5)
    ax_rpm.plot(t, rpm_enc_l, 'b--', label='left enc',  lw=1.2, alpha=0.8)
    ax_rpm.plot(t, rpm_enc_r, 'r--', label='right enc', lw=1.2, alpha=0.8)
    initial = data.get('initial_move_rpm')
    if initial is not None:
        ax_rpm.axhline(initial, color='green', ls='--', lw=1.2,
                       label=f'initial move {initial:.2f} RPM')
        ax_rpm.axhline(-initial, color='green', ls='--', lw=1.0, alpha=0.45)
    lowest = data.get('lowest_move_rpm')
    if lowest is not None:
        ax_rpm.axhline(lowest, color='orange', ls=':', lw=1.2,
                       label=f'lowest moving {lowest:.2f} RPM')
        ax_rpm.axhline(-lowest, color='orange', ls=':', lw=1.0, alpha=0.45)
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

    if mode == "follow":
        _plot_follow(data, out_path)
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
    center_search_plot_cap_rpm = max(
        CENTER_SEARCH_NEAR_MAX_RPM,
        CENTER_SEARCH_FAR_MAX_RPM,
    )
    cap = (
        math.degrees(wheel_rpm_to_spin_omega(center_search_plot_cap_rpm))
        if mode == "center" else math.degrees(P.MAX_TURN)
    )
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
        rpm_enc_l = [encoder_delta_to_wheel_rpm(d) for d in data['enc_l_delta']]
        rpm_enc_r = [encoder_delta_to_wheel_rpm(d) for d in data['enc_r_delta']]
        ax_rpm.plot(data['t'], rpm_l, 'b-', label='left commanded RPM')
        ax_rpm.plot(data['t'], rpm_r, 'r-', label='right commanded RPM')
        ax_rpm.plot(data['t'], rpm_enc_l, 'b--', label='left encoder RPM', alpha=0.8)
        ax_rpm.plot(data['t'], rpm_enc_r, 'r--', label='right encoder RPM', alpha=0.8)
        ax_rpm.set_xlabel('time (s)')
        ax_rpm.set_ylabel('RPM')
        ax_rpm.set_title('Commanded vs encoder wheel RPM')
        ax_rpm.legend(fontsize=8)
        ax_rpm.grid(True, alpha=0.3)

        enc_path = out_path.replace('.png', '_encoders.png')
        plt.tight_layout()
        plt.savefig(enc_path, dpi=110, bbox_inches='tight')
        print(f"Saved: {enc_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("center", "spin", "slow-spin", "follow"), default="center",
                        help="center keeps shopper centred; spin runs 360°; slow-spin turns in place slowly; follow drives forward/backward only")
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
    parser.add_argument("--duration", type=float, default=0.0,
                        help="run time in seconds for center/slow-spin; 0 runs until Ctrl-C (default: 0)")
    parser.add_argument("--direction", choices=("left", "right"), default="left",
                        help="slow-spin direction (default: left)")
    parser.add_argument("--sim-angle", type=float, default=None,
                        help="simulate a fixed target angle instead of listening for UDP")
    parser.add_argument("--sim-dist", type=float, default=None,
                        help="simulate a fixed target distance (m) for follow mode instead of UDP")
    args = parser.parse_args()

    if args.mode == "follow":
        if args.sim_dist is not None or args.source == "udp":
            if args.source == "udp" and args.sim_dist is None:
                print(f"UDP follow mode: listening on port {P.UDP_PORT}.")
            data = run_follow(drive=not args.no_drive, port=args.port,
                              countdown=args.countdown, duration=args.duration,
                              sim_dist=args.sim_dist)
        else:
            no_display = args.no_display
            if not no_display and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                no_display = True
                print("No display detected — running headless (use --no-display to silence this message).")
            data = run_follow_camera(drive=not args.no_drive, port=args.port,
                                     countdown=args.countdown, duration=args.duration,
                                     no_display=no_display)
    elif args.mode == "spin":
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
