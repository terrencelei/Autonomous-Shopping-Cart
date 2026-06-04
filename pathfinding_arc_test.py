"""
Rotation test — either spin one full 360° turn or rotate in place to keep
the shopper centred in the camera FOV.

Usage:
    python3 pathfinding_arc_test.py                    # centre target from live camera
    python3 pathfinding_arc_test.py --source udp       # centre target from UDP
    python3 pathfinding_arc_test.py --mode spin         # one 360° spin
    python3 pathfinding_arc_test.py --no-drive --sim-angle 15
    python3 pathfinding_arc_test.py --port /dev/ttyUSB0
    python3 pathfinding_arc_test.py --mode follow                  # forward/backward, live camera
    python3 pathfinding_arc_test.py --mode follow --no-drive --sim-dist 3.5  # simulation
    python3 pathfinding_arc_test.py --mode track                   # centre + follow, live camera
    python3 pathfinding_arc_test.py --mode track --source udp      # centre + follow from UDP
    python3 pathfinding_arc_test.py --mode track --no-drive --sim-dist 3 --sim-angle 20


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

# --- Combined tracking controller (spin-to-centre + distance follow) --------
# Speeds below are WHEEL RPM.  Gains/inertia marked "calibrate" are placeholders;
# inertia defaults to 0 so an un-tuned controller simply skips drift comp.
TRACK_TICKS_PER_SEC    = 1.0 / P.DT                # control ticks per second (=50)
TRACK_TURN_RPM         = 6.0                       # steady wheel RPM during a spin turn
TRACK_KICK_RPM         = CENTER_KICK_RPM           # 8  — breakaway burst (see memory)
TRACK_KICK_TICKS       = CENTER_KICK_RELEASE_TICKS # 30 — kick duration in control ticks
ANGULAR_INERTIA        = 0.0    # s — yaw coast factor; drift = (w0-w1)*inertia  [calibrate]
LINEAR_INERTIA         = 0.0    # s — fwd coast factor, reserved for linear drift  [calibrate]
TRACK_THETA_THRESH_DEG = 8.0    # |angle| beyond which we stop and re-centre via spin()
TRACK_KP_ANGLE         = 30.0   # wheel RPM of steering per rad of angle error   [calibrate]
TRACK_KD_ANGLE         = 0.0    # wheel RPM of steering per (rad/s)              [calibrate]
TRACK_KP_DIST          = 20.0   # RPM speed change per (m/s) approach rate (dx)  [calibrate]
TRACK_KI_DIST          = 10.0   # RPM speed change per metre of standoff error   [calibrate]
TRACK_DIST_THRESH_M    = P.HOLD_DIST               # standoff distance to hold (m)
TRACK_MAX_RPM          = P._rpm(P.MAX_SPEED)       # wheel RPM cap (= MAX_SPEED)
TRACK_DX_ALPHA         = 0.3                        # EMA factor for the smoothed dx/dt

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
                    motors.stop()
                    print("Target ticks reached — draining coast ticks...")
                    # Keep reading until the cart has fully stopped (no new
                    # ticks for COAST_IDLE_LOOPS consecutive DT periods)
                    COAST_IDLE_LOOPS = 10
                    idle = 0
                    while idle < COAST_IDLE_LOOPS:
                        time.sleep(P.DT)
                        d_l, d_r = motors.read_encoder_deltas()
                        if abs(d_l) + abs(d_r) == 0:
                            idle += 1
                        else:
                            idle = 0
                        cum_l += d_l; cum_r += d_r
                        abs_ticks_l += abs(d_l)
                        abs_ticks_r += abs(d_r)
                        enc_right_delta.append(d_l); enc_left_delta.append(d_r)
                        enc_right_cum.append(cum_l); enc_left_cum.append(cum_r)
                        t_coast = times[-1] + P.DT * (len(enc_right_delta) - len(times))
                        times.append(t_coast)
                        encoder_degrees.append(
                            (sum(t for t in (abs_ticks_l, abs_ticks_r) if t > 0)
                             / max(sum(1 for t in (abs_ticks_l, abs_ticks_r) if t > 0), 1))
                            / TICKS_360 * 360.0)
                        robot_xs.append(pos[0]); robot_ys.append(pos[1])
                        robot_thetas.append(heading)
                        v_fwds.append(0.0); omegas.append(0.0)
                        v_lefts.append(0.0); v_rights.append(0.0)
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


def follow_wheel_commands(v_forward):
    return P._wheel_commands(v_forward, 0.0)


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
            v_left, v_right = follow_wheel_commands(v_forward)

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
                v_left, v_right = follow_wheel_commands(v_forward)

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


# =============================================================================
# COMBINED TRACKING CONTROLLER (spin-to-centre + distance follow)
# =============================================================================


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def drift(initial_rpm, final_rpm, inertia):
    """Extra travel the cart coasts through when the commanded rate steps from
    ``initial_rpm`` to ``final_rpm``, modelled as proportional to the change.

    Unit-agnostic: callers using cart yaw (rad/s) pass ``inertia`` in seconds
    and get radians back.  Returns 0 while ``inertia`` is 0, so an un-tuned
    controller simply skips drift compensation.
    """
    return (initial_rpm - final_rpm) * inertia


def _drive_spin(motors, read_deltas, wheel_rpm, direction, seconds):
    """Hold a constant point turn at ``wheel_rpm`` * ``direction`` for
    ``seconds``, paced at the control period and draining encoders so the
    ESP32 watchdog stays fed."""
    if seconds <= 0.0 or motors is None:
        return
    omega = wheel_rpm_to_spin_omega(wheel_rpm, direction)
    v_left, v_right = P._wheel_commands(0.0, omega)
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        loop0 = time.monotonic()
        motors.send(v_left, v_right)
        if read_deltas is not None:
            read_deltas()
        rest = P.DT - (time.monotonic() - loop0)
        if rest > 0:
            time.sleep(rest)


def spin(theta_rad, motors, read_deltas=None,
         turn_rpm=TRACK_TURN_RPM, kick_rpm=TRACK_KICK_RPM,
         kick_ticks=TRACK_KICK_TICKS, inertia=ANGULAR_INERTIA):
    """Open-loop point turn through ``theta_rad`` (rad, +CCW / left).

    Profile: a short breakaway kick at ``kick_rpm`` to beat static friction,
    then a steady turn at ``turn_rpm`` for the remaining angle.  The angle the
    kick already sweeps (phi) and the inertial coast on each speed change
    (kick->turn, turn->0) are subtracted up front so the *total* swept angle
    lands on theta.
    """
    if theta_rad == 0.0 or motors is None:
        return
    direction = math.copysign(1.0, theta_rad)
    target    = abs(theta_rad)

    omega_kick = wheel_rpm_to_spin_omega(kick_rpm)   # cart yaw rate, rad/s
    omega_turn = wheel_rpm_to_spin_omega(turn_rpm)
    kick_time  = kick_ticks / TRACK_TICKS_PER_SEC    # seconds

    phi        = omega_kick * kick_time              # angle swept by the kick
    kick_drift = drift(omega_kick, omega_turn, inertia)
    stop_drift = drift(omega_turn, 0.0, inertia)
    theta_2    = target - phi - kick_drift - stop_drift   # angle for steady turn

    flush_motors(motors)
    _drive_spin(motors, read_deltas, kick_rpm, direction, kick_time)   # breakaway kick
    if theta_2 > 0.0 and omega_turn > 0.0:                             # steady turn
        _drive_spin(motors, read_deltas, turn_rpm, direction, theta_2 / omega_turn)
    motors.stop()


class TrackController:
    """Reactive follow controller: spin to keep the shopper centred, then a
    PI-on-distance forward speed mixed onto the wheels with peak scaling.

    Distances in metres, angles in degrees (camera convention: +angle = target
    to the right).  Speeds are wheel RPM internally.  ``step`` returns
    (left_rpm, right_rpm), or None on a tick that ran a blocking re-centre spin
    (the caller should not send its own command that tick).
    """

    def __init__(self):
        self.S = 0.0            # current forward wheel RPM
        self.dx = 0.0           # smoothed d(dist)/dt  (m/s)
        self.dx2 = 0.0          # finite difference of dx (m/s^2)
        self.spun = False       # True on a tick that re-centred
        self._prev_x = None
        self._prev_theta = 0.0

    def reset(self):
        self.__init__()

    def _update_derivatives(self, x, theta_rad):
        raw_dx = 0.0 if self._prev_x is None else (x - self._prev_x) / P.DT
        smooth_dx = (1.0 - TRACK_DX_ALPHA) * self.dx + TRACK_DX_ALPHA * raw_dx
        self.dx2 = (smooth_dx - self.dx) / P.DT
        self.dx = smooth_dx
        dtheta = (theta_rad - self._prev_theta) / P.DT
        self._prev_x = x
        self._prev_theta = theta_rad
        return dtheta

    def step(self, x, angle_deg, motors=None, read_deltas=None):
        """One control tick.  x = distance to target (m), angle_deg = its FOV
        angle (deg, +right), or None/None when the target is lost."""
        self.spun = False
        if x is None or angle_deg is None:
            self.S = 0.0               # target lost — coast to a stop
            return 0.0, 0.0

        theta = math.radians(-angle_deg)   # +CCW (left); sign turns toward target
        dtheta = self._update_derivatives(x, theta)

        # Centering — always parallel, independent of distance
        if abs(theta) > math.radians(TRACK_THETA_THRESH_DEG):
            spin(theta, motors, read_deltas)
            self.S = 0.0                   # motion stopped during the turn
            self.spun = True
            return None                    # skip distance control this tick

        rpm_diff = TRACK_KP_ANGLE * theta + TRACK_KD_ANGLE * dtheta

        # Distance control
        if x - TRACK_DIST_THRESH_M < 0.0:
            self.S = 0.0
        else:
            if self.S == 0.0:
                self.S = TRACK_KICK_RPM    # breakaway kick on departure from rest
            delta_S = TRACK_KP_DIST * self.dx + TRACK_KI_DIST * (x - TRACK_DIST_THRESH_M)
            self.S = _clip(self.S + delta_S, -TRACK_MAX_RPM, TRACK_MAX_RPM)
        S = self.S

        # Wheel mixing with peak scaling
        if max(abs(S + rpm_diff), abs(S - rpm_diff)) > TRACK_MAX_RPM:
            rpm_diff = math.copysign(
                min(abs(TRACK_MAX_RPM - abs(S)), abs(TRACK_MAX_RPM + abs(S))),
                rpm_diff)
        r_motor = S + rpm_diff
        l_motor = S - rpm_diff
        return l_motor, r_motor


def _rpm_to_v(wheel_rpm):
    """Wheel RPM -> linear wheel velocity (m/s) for MotorDriver.send()."""
    return wheel_rpm * P.WHEEL_CIRC / 60.0


def run_track(drive=True, port=None, countdown=3, duration=30.0,
              sim_dist=None, sim_angle=None):
    """Combined tracking test: spin-to-centre + PI distance follow.

    Reads (dist, angle) from the vision UDP feed, or holds fixed --sim-dist /
    --sim-angle values when either is given.
    """
    use_sim  = sim_dist is not None or sim_angle is not None
    motors   = open_motors(drive=drive, port=port, countdown=countdown, action="tracking")
    receiver = None if use_sim else P.TargetReceiver()
    ctrl     = TrackController()

    print("Track mode: spin-to-centre + PI distance follow.")
    print(f"Hold {TRACK_DIST_THRESH_M:.2f} m  centre band ±{TRACK_THETA_THRESH_DEG:.0f}°  "
          f"turn {TRACK_TURN_RPM} rpm  kick {TRACK_KICK_RPM} rpm/{TRACK_KICK_TICKS} ticks")
    if use_sim:
        print(f"Simulated target: dist={sim_dist} m, angle={sim_angle}°")
    else:
        print(f"Listening for UDP target readings on port {P.UDP_PORT}.")

    times, v_lefts, v_rights = [], [], []
    target_dists, dist_errors, target_angles = [], [], []
    enc_l_cum, enc_r_cum, enc_l_delta, enc_r_delta = [], [], [], []
    cum_l = cum_r = 0
    start = time.monotonic()
    i = 0

    flush_motors(motors)
    read = motors.read_encoder_deltas if motors is not None else (lambda: (0, 0))
    try:
        while True:
            loop0 = time.monotonic()
            t = loop0 - start
            if duration > 0 and t >= duration:
                break

            if use_sim:
                reading = (sim_dist, sim_angle if sim_angle is not None else 0.0)
            else:
                reading = receiver.get()
            dist_m  = reading[0] if reading is not None else None
            angle_d = reading[1] if reading is not None else None

            d_l, d_r = read()
            out = ctrl.step(dist_m, angle_d, motors=motors, read_deltas=read)

            if out is None:                 # a blocking re-centre spin ran this tick
                v_left = v_right = 0.0
            else:
                l_rpm, r_rpm = out
                v_left, v_right = _rpm_to_v(l_rpm), _rpm_to_v(r_rpm)
                if motors is not None:
                    motors.send(v_left, v_right)

            cum_l += d_l; cum_r += d_r
            times.append(t)
            v_lefts.append(v_left); v_rights.append(v_right)
            target_dists.append(float("nan") if dist_m is None else dist_m)
            dist_errors.append(float("nan") if dist_m is None else dist_m - TRACK_DIST_THRESH_M)
            target_angles.append(float("nan") if angle_d is None else angle_d)
            enc_l_delta.append(d_l); enc_r_delta.append(d_r)
            enc_l_cum.append(cum_l); enc_r_cum.append(cum_r)

            if i % max(1, int(0.5 / P.DT)) == 0:
                if dist_m is None:
                    print(f"t={t:5.1f}s  no target")
                else:
                    tag = "SPIN          " if ctrl.spun else f"S={ctrl.S:+7.1f}rpm"
                    print(f"t={t:5.1f}s  dist={dist_m:.2f}m  angle={angle_d:+5.1f}°  "
                          f"dx={ctrl.dx:+.2f}m/s  {tag}")
            i += 1

            rest = P.DT - (time.monotonic() - loop0)
            if rest > 0:
                time.sleep(rest)
    except KeyboardInterrupt:
        print("\nTracking stopped by user.")
    finally:
        if motors is not None:
            print("\nStopping motors.")
            motors.stop()

    return dict(
        mode="track",
        t=times,
        v_left=v_lefts, v_right=v_rights,
        target_dist=target_dists, dist_error=dist_errors, target_angle=target_angles,
        enc_l_cum=enc_l_cum, enc_r_cum=enc_r_cum,
        enc_l_delta=enc_l_delta, enc_r_delta=enc_r_delta,
        drove=(motors is not None),
    )


def run_track_camera(drive=True, port=None, countdown=3, duration=30.0, no_display=False):
    """Camera-driven combined tracking: spin-to-centre + PI distance follow,
    reading distance and angle from the live IMX500 target lock."""
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

    motors = open_motors(drive=drive, port=port, countdown=countdown, action="tracking")
    tracker = Y.sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=60,
        minimum_matching_threshold=0.8,
        frame_rate=30,
    )
    smooth_state = {}
    target_lock = Y.TargetLock()
    ctrl = TrackController()

    print("Camera track mode: spin-to-centre + PI distance follow.")
    print(f"Hold {TRACK_DIST_THRESH_M:.2f} m  centre band ±{TRACK_THETA_THRESH_DEG:.0f}°  "
          f"turn {TRACK_TURN_RPM} rpm  kick {TRACK_KICK_RPM} rpm/{TRACK_KICK_TICKS} ticks")
    print("Press Q in the Cart View window to stop." if not no_display else "Display disabled.")

    times, v_lefts, v_rights = [], [], []
    target_dists, dist_errors, target_angles = [], [], []
    enc_l_cum, enc_r_cum, enc_l_delta, enc_r_delta = [], [], [], []
    cum_l = cum_r = 0
    start = time.monotonic()
    last_tick = time.monotonic()
    frame_times = []
    v_left = v_right = 0.0
    i = 0

    flush_motors(motors)
    read = motors.read_encoder_deltas if motors is not None else (lambda: (0, 0))
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
            dist_m  = target_row[3] if target_row is not None else None
            angle_d = target_row[4] if target_row is not None else None

            now = time.monotonic()
            if now - last_tick >= P.DT:
                d_l, d_r = read()
                cmd = ctrl.step(dist_m, angle_d, motors=motors, read_deltas=read)
                if cmd is None:                 # blocking re-centre spin ran
                    v_left = v_right = 0.0
                else:
                    l_rpm, r_rpm = cmd
                    v_left, v_right = _rpm_to_v(l_rpm), _rpm_to_v(r_rpm)
                    if motors is not None:
                        motors.send(v_left, v_right)
                last_tick = now

                cum_l += d_l; cum_r += d_r
                times.append(t)
                v_lefts.append(v_left); v_rights.append(v_right)
                target_dists.append(float("nan") if dist_m is None else dist_m)
                dist_errors.append(float("nan") if dist_m is None else dist_m - TRACK_DIST_THRESH_M)
                target_angles.append(float("nan") if angle_d is None else angle_d)
                enc_l_delta.append(d_l); enc_r_delta.append(d_r)
                enc_l_cum.append(cum_l); enc_r_cum.append(cum_r)

                if i % max(1, int(0.5 / P.DT)) == 0 and dist_m is not None:
                    tag = "SPIN" if ctrl.spun else f"S={ctrl.S:+7.1f}rpm"
                    print(f"t={t:5.1f}s  dist={dist_m:.2f}m  angle={angle_d:+5.1f}°  {tag}")
                i += 1

            frame_times.append(time.time())
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps_live = ((len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                        if len(frame_times) > 1 else 0.0)
            dist_str = f"dist={dist_m:.2f}m" if dist_m is not None else "no target"
            ang_str = f"ang={angle_d:+.1f}°" if angle_d is not None else ""
            Y.cv2.putText(out, f"TRACK  FPS:{fps_live:.1f}  {dist_str} {ang_str}",
                          (10, 25), Y.cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            if not no_display:
                Y.overlay_map(out, rows)
                Y.cv2.imshow("Cart View", out)
                Y.cv2.imshow("World Map", Y.draw_world_map(rows))
                if Y.cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\nCamera tracking stopped by user.")
    finally:
        if motors is not None:
            print("\nStopping motors.")
            motors.stop()
        cap.release()
        if not no_display:
            Y.cv2.destroyAllWindows()

    return dict(
        mode="track",
        t=times,
        v_left=v_lefts, v_right=v_rights,
        target_dist=target_dists, dist_error=dist_errors, target_angle=target_angles,
        enc_l_cum=enc_l_cum, enc_r_cum=enc_r_cum,
        enc_l_delta=enc_l_delta, enc_r_delta=enc_r_delta,
        drove=(motors is not None),
    )


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

    if mode in ("follow", "track"):
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
    parser.add_argument("--mode", choices=("center", "spin", "follow", "track"), default="center",
                        help="center keeps shopper centred; spin runs 360°; follow drives "
                             "forward/backward only; track combines spin-to-centre + distance follow")
    parser.add_argument("--source", choices=("camera", "udp"), default="camera",
                        help="target source for center/track modes (default: camera)")
    parser.add_argument("--no-drive", action="store_true",
                        help="simulation only — don't open the ESP32 serial port")
    parser.add_argument("--no-display", action="store_true",
                        help="suppress OpenCV windows in camera center mode")
    parser.add_argument("--port", default=None,
                        help=f"serial port override (default: {_MOTOR_PORT})")
    parser.add_argument("--countdown", type=int, default=3,
                        help="seconds before motors start (default: 3)")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="run time in seconds for center; 0 runs until Ctrl-C (default: 0)")
    parser.add_argument("--sim-angle", type=float, default=None,
                        help="simulate a fixed target angle instead of listening for UDP")
    parser.add_argument("--sim-dist", type=float, default=None,
                        help="simulate a fixed target distance (m) for follow mode instead of UDP")
    args = parser.parse_args()

    if args.mode == "track":
        if args.sim_dist is not None or args.sim_angle is not None or args.source == "udp":
            if args.source == "udp" and args.sim_dist is None and args.sim_angle is None:
                print(f"UDP track mode: listening on port {P.UDP_PORT}.")
            data = run_track(drive=not args.no_drive, port=args.port,
                             countdown=args.countdown, duration=args.duration,
                             sim_dist=args.sim_dist, sim_angle=args.sim_angle)
        else:
            no_display = args.no_display
            if not no_display and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                no_display = True
                print("No display detected — running headless (use --no-display to silence this message).")
            data = run_track_camera(drive=not args.no_drive, port=args.port,
                                    countdown=args.countdown, duration=args.duration,
                                    no_display=no_display)
    elif args.mode == "follow":
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
