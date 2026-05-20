"""
Detect and track people with ByteTrack.

Usage:
  python3 yolo_detect.py image.jpg                   # image file
  python3 yolo_detect.py video.mp4                   # video file
  python3 yolo_detect.py 0                           # webcam index
  python3 yolo_detect.py --picamera                  # Pi AI Camera, YOLO on Pi CPU
  python3 yolo_detect.py --npu                       # Pi AI Camera, YOLO on IMX500 NPU
  python3 yolo_detect.py --npu --npu-model my.rpk    # NPU with custom .rpk model
  python3 yolo_detect.py --npu --no-display          # headless NPU (SSH)

NPU notes:
  --npu uses the IMX500's on-chip neural processor — no YOLO inference on the Pi CPU.
  Requires a .rpk model. Default: /usr/share/imx500-models/imx500_network_yolov8n_pp.rpk
  Install pre-built models: sudo apt install imx500-models
  To convert yolo26n.pt → RPK: export to ONNX then run imx500-convert (Sony MCT toolchain).
"""

import sys
import time
import argparse
import warnings
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from pathlib import Path

try:
    from picamera2 import Picamera2
    _PICAMERA2_AVAILABLE = True
except ImportError:
    _PICAMERA2_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)

_DIR = Path(__file__).parent

MODEL_PATH        = _DIR / "yolo11n.pt"
NPU_MODEL_PATH    = _DIR / "yolo11n.rpk"
PERSON_CONFIDENCE     = 0.35
PERSON_CONFIDENCE_NPU = 0.25   # INT8 quantization lowers raw confidence scores

PERSON_HEIGHT_M   = 1.7
WEBCAM_H_FOV_DEG  = 54.0
PICAM_H_FOV_DEG   = 66.0
H_FOV_DEG         = WEBCAM_H_FOV_DEG   # overridden at runtime for --picamera
DISTANCE_OFFSET_M = 3.0   # calibration offset — subtract from computed distance
DISTANCE_SCALE    = 1.0   # multiply final distance estimate
ANGLE_SCALE       = 1.0   # multiply final angle estimate

PERSON_CLASS_ID = 0

COLOR_TARGET   = (0, 255, 0)
COLOR_OBSTACLE = (0, 0, 255)

MAP_SIZE    = 200
MAP_RANGE_M = 10.0

TARGET_DIST_WEIGHT  = 1.0
TARGET_ANGLE_WEIGHT = 0.3

DIST_EMA_ALPHA = 0.4


class PiCameraCapture:
    """picamera2 wrapper with a cv2.VideoCapture-compatible interface."""

    def __init__(self, width=640, height=480, fps=30):
        if not _PICAMERA2_AVAILABLE:
            raise RuntimeError("picamera2 is not installed — run: pip install picamera2")
        self._cam = Picamera2(camera_num=0)  # CAM/DISP 0 port
        cfg = self._cam.create_preview_configuration(
            main={"format": "BGR888", "size": (width, height)},
            controls={"FrameRate": fps},
        )
        self._cam.configure(cfg)
        self._cam.start()
        self._w, self._h, self._fps = width, height, fps

    def isOpened(self):
        return True

    def read(self):
        frame = self._cam.capture_array("main")
        return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def get(self, prop):
        return {
            cv2.CAP_PROP_FPS:          self._fps,
            cv2.CAP_PROP_FRAME_WIDTH:  self._w,
            cv2.CAP_PROP_FRAME_HEIGHT: self._h,
        }.get(prop, 0)

    def release(self):
        self._cam.stop()
        self._cam.close()


class IMX500Capture:
    """Pi AI Camera with YOLO inference running entirely on the IMX500 NPU.

    Loads an .rpk model onto the sensor at startup. Each call to read() returns
    the current BGR frame; call get_npu_detections() immediately after to get
    the corresponding inference results from the same request's metadata.
    """

    def __init__(self, model_path, width=640, height=480, fps=30):
        if not _PICAMERA2_AVAILABLE:
            raise RuntimeError("picamera2 is not installed — run: pip install picamera2")
        from picamera2.devices.imx500 import IMX500, NetworkIntrinsics

        self._imx500 = IMX500(str(model_path))
        intrinsics = self._imx500.network_intrinsics or NetworkIntrinsics()
        intrinsics.task = "object detection"
        self._intrinsics = intrinsics

        self._picam2 = Picamera2(self._imx500.camera_num)  # camera_num=0 → CAM/DISP 0
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

    def get_npu_detections(self):
        """Parse IMX500 output tensors from the last read() into sv.Detections."""
        if self._meta is None:
            return self._last_dets
        np_outputs = self._imx500.get_outputs(self._meta, add_batch=True)
        if np_outputs is None:
            return self._last_dets  # reuse last result between NPU inference cycles

        # YOLOv8n_pp output layout: [boxes (N,4), scores (N,), classes (N,)]
        # boxes are normalized (x, y, w, h) in inference-input space
        boxes_raw = np_outputs[0][0]
        scores    = np_outputs[1][0]
        classes   = np_outputs[2][0].astype(int)

        keep = (scores >= PERSON_CONFIDENCE_NPU) & (classes == PERSON_CLASS_ID)
        if not keep.any():
            self._last_dets = sv.Detections.empty()
            return self._last_dets
        boxes_raw, scores, classes = boxes_raw[keep], scores[keep], classes[keep]

        # convert_inference_coords maps normalized inference xywh → display-frame pixels xywh
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


def infer_frame(model, frame):
    results = model(frame, verbose=False, conf=PERSON_CONFIDENCE)[0]
    if not len(results.boxes):
        return sv.Detections.empty()

    xyxy    = results.boxes.xyxy.cpu().numpy()
    conf    = results.boxes.conf.cpu().numpy()
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)

    keep = cls_ids == PERSON_CLASS_ID
    if not keep.any():
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=xyxy[keep],
        confidence=conf[keep],
        class_id=cls_ids[keep],
    )


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

    # single distance ring at 5m
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
    """Draw the mini map as a picture-in-picture in the top-right corner."""
    m = draw_map(rows)
    h, w = frame.shape[:2]
    pad = 8
    y1, y2 = pad, pad + MAP_SIZE
    x1, x2 = w - MAP_SIZE - pad, w - pad
    # dim the background slightly
    frame[y1:y2, x1:x2] = (frame[y1:y2, x1:x2] * 0.3 + m * 0.7).astype(np.uint8)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)
    return frame


def print_header():
    print(f"\n{'Role':<10} {'ID':<8} {'Conf':>6}  {'Distance':>10}  {'Angle':>8}")
    print("-" * 52)


def run_image(source, model):
    frame = cv2.imread(source)
    if frame is None:
        print(f"Could not load {source}")
        return

    tracker = sv.ByteTrack()
    dets    = infer_frame(model, frame)
    tracked = tracker.update_with_detections(dets)
    out, rows = annotate_frame(frame, tracked, {})

    print(f"\n{source} — {len(rows)} object(s) detected")
    print_header()
    for role, label_id, conf, dist, angle in rows:
        print(f"{role:<10} {label_id:<8} {conf:>6.0%}  {dist:>8.1f}m  {angle:>+7.1f}°")

    out_path = source.rsplit(".", 1)[0] + "_tracked.jpg"
    cv2.imwrite(out_path, out)
    print(f"\nSaved: {out_path}")


def run_video(source, model, use_picamera=False, use_npu=False, npu_model=None, no_display=False):
    global H_FOV_DEG
    H_FOV_DEG = PICAM_H_FOV_DEG if (use_picamera or use_npu) else WEBCAM_H_FOV_DEG

    if use_npu:
        cap = IMX500Capture(model_path=npu_model or NPU_MODEL_PATH, width=640, height=480, fps=30)
    elif use_picamera:
        cap = PiCameraCapture(width=640, height=480, fps=30)
    else:
        try:
            source = int(source)
        except (ValueError, TypeError):
            pass
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Could not open: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=60,   # remember lost tracks for 2s at 30fps
        minimum_matching_threshold=0.8,
        frame_rate=30,
    )
    smooth_state = {}

    writer = None

    frame_idx   = 0
    frame_times = []
    quit_msg = "" if no_display else " — press Q to quit"
    print(f"\nTracking{quit_msg}\n")
    print_header()

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        dets    = cap.get_npu_detections() if use_npu else infer_frame(model, frame)
        tracked = tracker.update_with_detections(dets)
        out, rows = annotate_frame(frame, tracked, smooth_state)

        t1 = time.time()
        latency_ms = (t1 - t0) * 1000
        frame_times.append(t1)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps_live = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0.0
        mode_str = "NPU" if use_npu else ("PiCam" if use_picamera else "CPU")
        cv2.putText(out, f"{mode_str}  FPS: {fps_live:.1f}  {latency_ms:.0f}ms",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        for role, label_id, conf, dist, angle in rows:
            print(f"{role:<10} {label_id:<8} {conf:>6.0%}  {dist:>8.1f}m  {angle:>+7.1f}°  [f{frame_idx}]")

        if not no_display:
            overlay_map(out, rows)
            cv2.imshow("Cart View", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if not no_display:
        cv2.destroyAllWindows()


def run():
    parser = argparse.ArgumentParser(description="YOLO person tracker with ByteTrack")
    parser.add_argument("source", nargs="?", default=None,
                        help="image, video file, or webcam index (default: 0)")
    parser.add_argument("--picamera", action="store_true",
                        help="use Pi AI Camera via picamera2 (YOLO runs on Pi CPU)")
    parser.add_argument("--npu", action="store_true",
                        help="use Pi AI Camera with YOLO inference on IMX500 NPU")
    parser.add_argument("--npu-model", dest="npu_model", default=None,
                        help=f"path to .rpk model for NPU (default: {NPU_MODEL_PATH})")
    parser.add_argument("--no-display", dest="no_display", action="store_true",
                        help="suppress cv2 windows (headless / SSH use)")
    args = parser.parse_args()

    _no_explicit_source = not args.picamera and args.source is None
    use_npu      = args.npu or (_no_explicit_source and _PICAMERA2_AVAILABLE)
    use_picamera = args.picamera

    if not use_npu and not use_picamera and args.source is None:
        parser.error("provide a source (image, video, or webcam index) or use --picamera / --npu")

    if use_npu:
        npu_path = Path(args.npu_model) if args.npu_model else NPU_MODEL_PATH
        if not npu_path.exists():
            print(f"ERROR: NPU model not found: {npu_path}")
            print("Convert your model first: python3 -c \"from ultralytics import YOLO; YOLO('yolo26n.pt').export(format='imx500', imgsz=640)\"")
            sys.exit(1)
        run_video(args.source or "0", model=None, use_npu=True,
                  npu_model=npu_path, no_display=args.no_display)
        return

    if not MODEL_PATH.exists():
        print(f"ERROR: {MODEL_PATH} not found.")
        sys.exit(1)

    model  = YOLO(MODEL_PATH)
    source = args.source or "0"

    is_image = (not use_picamera and isinstance(source, str) and
                source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")))
    if is_image:
        run_image(source, model)
    else:
        run_video(source, model, use_picamera=use_picamera, no_display=args.no_display)


if __name__ == "__main__":
    run()
