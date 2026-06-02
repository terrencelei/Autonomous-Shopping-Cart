"""
Detect and track people with SSD-MobileNetV2 on the IMX500 NPU + ByteTrack,
then drive the cart directly via Pathfinding_algorithm.tick().

Usage:
  python3 yolo_detect.py                # live camera + motors
  python3 yolo_detect.py --no-display   # headless (SSH)
  python3 yolo_detect.py --no-drive     # camera only, no motors

Inference runs entirely on the IMX500's on-chip neural processor; no CPU
inference path. Requires the imx500-models apt package.
"""

import math
import os
import sys
import time
import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics

# Pathfinding_algorithm.py lives one directory up from vision/
sys.path.insert(0, str(Path(__file__).parent.parent))
import Pathfinding_algorithm as P

warnings.filterwarnings("ignore", category=FutureWarning)

RPK_MODEL_PATH = Path(
    "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
)

PERSON_CONFIDENCE = 0.6
PERSON_CLASS_ID   = 0   # COCO "person" in the labels shipped with imx500-models

PERSON_HEIGHT_M   = 1.8
H_FOV_DEG         = 66.0   # Pi AI Camera horizontal field of view
DISTANCE_OFFSET_M = 0.89
DISTANCE_SCALE    = 0.95
ANGLE_SCALE       = 1.0

COLOR_TARGET   = (0, 255, 0)
COLOR_OBSTACLE = (0, 0, 255)

MAP_SIZE    = 200
MAP_RANGE_M = 10.0

TARGET_DIST_WEIGHT  = 1.0
TARGET_ANGLE_WEIGHT = 0.3

DIST_EMA_ALPHA = 0.4

# World-map window
WORLD_MAP_PX = 500


class IMX500Capture:
    """Pi AI Camera with detection running entirely on the IMX500 NPU.

    Loads an .rpk model onto the sensor at startup. Each call to read() returns
    the current BGR frame; call get_detections() immediately after to get
    the corresponding inference results from the same request's metadata.
    """

    def __init__(self, model_path, width=640, height=480, fps=30):
        self._imx500 = IMX500(str(model_path))
        intrinsics = self._imx500.network_intrinsics or NetworkIntrinsics()
        intrinsics.task = "object detection"
        self._intrinsics = intrinsics

        self._picam2 = Picamera2(self._imx500.camera_num)
        self._cfg = self._picam2.create_preview_configuration(
            main={"format": "BGR888", "size": (width, height)},
            controls={"FrameRate": fps},
            buffer_count=12,
        )
        self._imx500.show_network_fw_progress_bar()
        self._picam2.start(self._cfg)
        self._meta      = None
        self._last_dets = sv.Detections.empty()
        self._w, self._h, self._fps = width, height, fps

    def isOpened(self):
        return True

    def read(self):
        req = self._picam2.capture_request()
        frame = req.make_array("main")
        self._meta = req.get_metadata()
        req.release()
        return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def get_detections(self):
        if self._meta is None:
            return self._last_dets
        np_outputs = self._imx500.get_outputs(self._meta, add_batch=True)
        if np_outputs is None:
            return self._last_dets

        boxes_raw = np_outputs[0][0]
        scores    = np_outputs[1][0]
        classes   = np_outputs[2][0].astype(int)

        keep = (scores >= PERSON_CONFIDENCE) & (classes == PERSON_CLASS_ID)
        if not keep.any():
            self._last_dets = sv.Detections.empty()
            return self._last_dets
        boxes_raw, scores, classes = boxes_raw[keep], scores[keep], classes[keep]

        xyxy = []
        for box in boxes_raw:
            x, y, w, h = self._imx500.convert_inference_coords(box, self._meta, self._picam2)
            xyxy.append([x, y, x + w, y + h])

        self._last_dets = sv.Detections(
            xyxy=np.array(xyxy, dtype=float),
            confidence=scores,
            class_id=classes,
        )
        return self._last_dets

    def get(self, prop):
        return {
            cv2.CAP_PROP_FPS:          self._fps,
            cv2.CAP_PROP_FRAME_WIDTH:  self._w,
            cv2.CAP_PROP_FRAME_HEIGHT: self._h,
        }.get(prop, 0)

    def release(self):
        self._picam2.stop()
        self._picam2.close()


def focal_length_px(dim, fov_deg):
    return (dim / 2) / np.tan(np.radians(fov_deg / 2))


def estimate_distance(bbox_h, bbox_cx, img_h, img_w):
    v_fov         = H_FOV_DEG * (img_h / img_w)
    fl_v          = focal_length_px(img_h, v_fov)
    raw_depth     = (PERSON_HEIGHT_M * fl_v) / bbox_h
    raw_angle_rad = np.arctan((bbox_cx - img_w / 2) / focal_length_px(img_w, H_FOV_DEG))
    slant         = raw_depth / np.cos(raw_angle_rad)
    return max(0.0, (slant - DISTANCE_OFFSET_M) * DISTANCE_SCALE)


def estimate_angle(bbox_cx, img_w):
    fl = focal_length_px(img_w, H_FOV_DEG)
    return np.degrees(np.arctan((bbox_cx - img_w / 2) / fl)) * ANGLE_SCALE


def find_target_idx(detections, img_w, img_h):
    best_idx, best_score = None, float("inf")
    for i, (x1, y1, x2, y2) in enumerate(detections.xyxy):
        bbox_h  = y2 - y1
        bbox_cx = (x1 + x2) / 2
        dist    = estimate_distance(bbox_h, bbox_cx, img_h, img_w) if bbox_h > 0 else float("inf")
        angle   = abs(estimate_angle(bbox_cx, img_w))
        score   = TARGET_DIST_WEIGHT * dist + TARGET_ANGLE_WEIGHT * angle
        if score < best_score:
            best_score, best_idx = score, i
    return best_idx


def annotate_frame(frame, detections: sv.Detections, smooth_state: dict):
    img_h, img_w = frame.shape[:2]
    out = frame.copy()

    target_idx = find_target_idx(detections, img_w, img_h)
    ids = detections.tracker_id if detections.tracker_id is not None else [None] * len(detections)

    rows = []
    for i, ((x1, y1, x2, y2), conf, tid) in enumerate(zip(
        detections.xyxy, detections.confidence, ids
    )):
        is_target = (i == target_idx)
        color     = COLOR_TARGET if is_target else COLOR_OBSTACLE
        role      = "TARGET" if is_target else "OBSTACLE"
        label_id  = f"ID{tid}" if tid is not None else "?"

        bbox_h  = y2 - y1
        bbox_cx = (x1 + x2) / 2

        raw_dist = estimate_distance(bbox_h, bbox_cx, img_h, img_w) if bbox_h > 0 else 0
        if tid is not None:
            prev = smooth_state.get(tid, raw_dist)
            dist = DIST_EMA_ALPHA * raw_dist + (1 - DIST_EMA_ALPHA) * prev
            smooth_state[tid] = dist
        else:
            dist = raw_dist
        angle = estimate_angle(bbox_cx, img_w)

        thickness = 3 if is_target else 2
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        label = f"{role} {label_id} {dist:.1f}m {angle:+.1f}deg"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        top = max(int(y1) - 10, th + 4)
        cv2.rectangle(out, (int(x1), top - th - 4), (int(x1) + tw, top), color, -1)
        cv2.putText(out, label, (int(x1), top - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        rows.append((role, label_id, conf, dist, angle))

    return out, rows


def draw_map(rows):
    s      = MAP_SIZE
    img    = np.zeros((s, s, 3), dtype=np.uint8)
    cam_px = s // 2
    cam_py = s - 20
    scale  = (s - 30) / MAP_RANGE_M

    def to_px(dist, angle_deg):
        rad = np.radians(angle_deg)
        return (int(cam_px + dist * np.sin(rad) * scale),
                int(cam_py - dist * np.cos(rad) * scale))

    cv2.circle(img, (cam_px, cam_py), int(5 * scale), (50, 50, 50), 1)
    cv2.putText(img, "5m", (cam_px + int(5 * scale) + 2, cam_py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)

    for side in (-1, 1):
        cv2.line(img, (cam_px, cam_py), to_px(MAP_RANGE_M, side * H_FOV_DEG / 2), (40, 40, 40), 1)

    for role, label_id, conf, dist, angle in rows:
        color = COLOR_TARGET if role == "TARGET" else COLOR_OBSTACLE
        px, py = to_px(dist, angle)
        r = 7 if role == "TARGET" else 5
        cv2.circle(img, (px, py), r, color, -1)

    pts = np.array([[cam_px, cam_py - 8], [cam_px - 6, cam_py + 4],
                    [cam_px + 6, cam_py + 4]], np.int32)
    cv2.fillPoly(img, [pts], (200, 200, 200))
    return img


def overlay_map(frame, rows):
    m = draw_map(rows)
    h, w = frame.shape[:2]
    pad = 8
    y1, y2 = pad, pad + MAP_SIZE
    x1, x2 = w - MAP_SIZE - pad, w - pad
    frame[y1:y2, x1:x2] = (frame[y1:y2, x1:x2] * 0.3 + m * 0.7).astype(np.uint8)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)
    return frame


def draw_world_map(rows):
    """Bird's-eye world-coordinate map: cart, target, obstacles on the CHUNK_MAP grid."""
    img = np.zeros((WORLD_MAP_PX, WORLD_MAP_PX, 3), dtype=np.uint8)
    sx  = WORLD_MAP_PX / P.MAP_W
    sy  = WORLD_MAP_PX / P.MAP_H
    cpw = max(1, int(P.CHUNK * sx))
    cph = max(1, int(P.CHUNK * sy))

    for r in range(P._ROWS):
        for c in range(P._COLS):
            px = int(c * P.CHUNK * sx)
            py = int((P.MAP_H - (r + 1) * P.CHUNK) * sy)
            col = (50, 50, 50) if P.CHUNK_MAP[r][c] else (18, 18, 18)
            cv2.rectangle(img, (px, py), (px + cpw, py + cph), col, -1)

    def w2p(wx, wy):
        return int(wx * sx), int((P.MAP_H - wy) * sy)

    # Cart body + heading arrow
    cx, cy = w2p(*P.S.pos)
    cv2.circle(img, (cx, cy), 8, (200, 200, 200), -1)
    ax = cx + int(18 * math.cos(P.S.heading))
    ay = cy - int(18 * math.sin(P.S.heading))
    cv2.arrowedLine(img, (cx, cy), (ax, ay), (255, 255, 255), 2, tipLength=0.4)

    # Detections projected from cart pose into world coords
    for role, label_id, conf, dist, angle in rows:
        bearing = P.S.heading - math.radians(angle)
        wx = P.S.pos[0] + dist * math.cos(bearing)
        wy = P.S.pos[1] + dist * math.sin(bearing)
        px, py = w2p(wx, wy)
        color = COLOR_TARGET if role == "TARGET" else COLOR_OBSTACLE
        cv2.circle(img, (px, py), 7 if role == "TARGET" else 5, color, -1)
        cv2.putText(img, label_id, (px + 8, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    cv2.putText(img, f"Mode: {P.S.mode}", (5, WORLD_MAP_PX - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1)
    return img


def print_header():
    print(f"\n{'Role':<10} {'ID':<8} {'Conf':>6}  {'Distance':>10}  {'Angle':>8}")
    print("-" * 52)


def run(no_display=False, no_drive=False):
    if not RPK_MODEL_PATH.exists():
        raise SystemExit(
            f"ERROR: {RPK_MODEL_PATH} not found. Install with: sudo apt install imx500-models"
        )

    cap = IMX500Capture(model_path=RPK_MODEL_PATH, width=640, height=480, fps=30)

    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=60,
        minimum_matching_threshold=0.8,
        frame_rate=30,
    )
    smooth_state = {}

    # Motor driver and odometry from Pathfinding_algorithm
    motors   = None if no_drive else P.MotorDriver()
    odometry = P.Odometry(motors.read_encoder_deltas if motors else lambda: (0, 0))

    frame_idx   = 0
    frame_times = []
    t_last_tick = time.monotonic()
    latest_target = None   # (dist_m, angle_deg) from the most recent frame

    quit_msg = "" if no_display else " — press Q to quit"
    drive_msg = " (motors disabled)" if no_drive else ""
    print(f"\nTracking{quit_msg}{drive_msg}\n")
    print_header()

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            dets    = cap.get_detections()
            tracked = tracker.update_with_detections(dets)
            out, rows = annotate_frame(frame, tracked, smooth_state)

            # Keep the freshest target sighting for the next control tick
            target_row = next((r for r in rows if r[0] == "TARGET"), None)
            if target_row is not None:
                _role, _id, _conf, t_dist, t_angle = target_row
                latest_target = (t_dist, t_angle)
            else:
                latest_target = None

            # Collect obstacles for pathfinding
            obstacle_rows = [r for r in rows if r[0] == "OBSTACLE"]

            # Control tick at P.DT rate (independent of camera fps)
            now = time.monotonic()
            if now - t_last_tick >= P.DT:
                pos, heading = odometry.update()
                obs = [(r[3], r[4]) for r in obstacle_rows]
                v_right, v_left = P.tick(latest_target, pos, heading, obstacles=obs)
                if motors is not None:
                    motors.send(v_right, v_left)
                t_last_tick = now

            t1 = time.time()
            latency_ms = (t1 - t0) * 1000
            frame_times.append(t1)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps_live = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0.0
            cv2.putText(out, f"NPU  FPS: {fps_live:.1f}  {latency_ms:.0f}ms  [{P.S.mode}]",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            for role, label_id, conf, dist, angle in rows:
                print(f"{role:<10} {label_id:<8} {conf:>6.0%}  {dist:>8.1f}m  {angle:>+7.1f}°  [f{frame_idx}]")

            if not no_display:
                overlay_map(out, rows)
                cv2.imshow("Cart View", out)
                cv2.imshow("World Map", draw_world_map(rows))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    finally:
        if motors is not None:
            motors.stop()
        cap.release()
        if not no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSD person tracker on IMX500 NPU + ByteTrack → Pathfinding_algorithm")
    parser.add_argument("--no-display", dest="no_display", action="store_true",
                        help="suppress cv2 windows (headless / SSH use) — auto-enabled if no DISPLAY")
    parser.add_argument("--no-drive", dest="no_drive", action="store_true",
                        help="run camera and detection but do not command motors")
    args = parser.parse_args()

    # Auto-detect headless environments (e.g. SSH without X forwarding)
    if not args.no_display and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        args.no_display = True
        print("No display detected — running headless (use --no-display to silence this message).")

    run(no_display=args.no_display, no_drive=args.no_drive)
