"""
Detect people and shopping carts, track each with ByteTrack.
Uses the Darknet C++ backend (hank-ai/darknet) for lighter inference.

Models (darknet format):
  PERSON_CFG / PERSON_WEIGHTS  — any YOLO darknet model trained on COCO
                                  e.g. yolov4-tiny.cfg + yolov4-tiny.weights
  PERSON_NAMES                 — COCO class names file (coco.names)
  CART_CFG / CART_WEIGHTS      — optional darknet cart model (skip if absent)

Setup:
  1. Build hank-ai/darknet: https://github.com/hank-ai/darknet
  2. Add <darknet>/src-python to PYTHONPATH:
       export PYTHONPATH=/path/to/darknet/src-python:$PYTHONPATH
  3. macOS: ensure libdarknet.dylib is accessible:
       export DYLD_LIBRARY_PATH=/path/to/darknet/build:$DYLD_LIBRARY_PATH
     Linux:
       export LD_LIBRARY_PATH=/path/to/darknet/build:$LD_LIBRARY_PATH
  4. Download weights (if not already present):
       wget https://github.com/hank-ai/darknet/releases/download/v3.0/yolov4-tiny.weights
       wget https://raw.githubusercontent.com/hank-ai/darknet/master/cfg/yolov4-tiny.cfg
       wget https://raw.githubusercontent.com/hank-ai/darknet/master/cfg/coco.names

Usage:
  python3 darknet_detect.py image.jpg
  python3 darknet_detect.py video.mp4
"""

import sys
import os
import ctypes
import warnings
import cv2
import numpy as np
import supervision as sv
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

# Resolve darknet paths relative to where it was built on this machine
_DARKNET_ROOT = Path.home() / "Desktop" / "darknet"
_DARKNET_LIB  = _DARKNET_ROOT / "build" / "src-lib" / "libdarknet.dylib"
_DARKNET_PY   = _DARKNET_ROOT / "src-python"

# Pre-load the shared library so the Python module can find it
if _DARKNET_LIB.exists():
    ctypes.CDLL(str(_DARKNET_LIB))
if str(_DARKNET_PY) not in sys.path:
    sys.path.insert(0, str(_DARKNET_PY))

try:
    import darknet
except ImportError:
    print(f"ERROR: darknet module not found at {_DARKNET_PY}")
    print("Build hank-ai/darknet: https://github.com/hank-ai/darknet")
    sys.exit(1)

_DIR = Path(__file__).parent

PERSON_CFG     = str(_DIR / "yolov4-tiny.cfg")
PERSON_WEIGHTS = str(_DIR / "yolov4-tiny.weights")
PERSON_NAMES   = str(_DIR / "coco.names")
CART_CFG       = str(_DIR / "cart_darknet.cfg")
CART_WEIGHTS   = str(_DIR / "cart_darknet.weights")
CART_NAMES     = str(_DIR / "cart.names")

PERSON_CONFIDENCE = 0.45
CART_CONFIDENCE   = 0.30
NMS_THRESHOLD     = 0.45

PERSON_HEIGHT_M   = 1.7
H_FOV_DEG         = 54.0
DISTANCE_OFFSET_M = 3.0
DISTANCE_SCALE    = 0.5
ANGLE_SCALE       = 2.0

CLASS_ID   = {"person": 0, "cart": 1}
CLASS_NAME = {0: "person", 1: "cart"}

COLOR_TARGET   = (0, 255, 0)
COLOR_OBSTACLE = (0, 0, 255)

MAP_SIZE    = 500
MAP_RANGE_M = 10.0

TARGET_DIST_WEIGHT  = 1.0
TARGET_ANGLE_WEIGHT = 0.3

DIST_EMA_ALPHA = 0.4


# ---------------------------------------------------------------------------
# Geometry helpers (identical to yolo_detect.py)
# ---------------------------------------------------------------------------

def focal_length_px(dim, fov_deg):
    return (dim / 2) / np.tan(np.radians(fov_deg / 2))


def estimate_distance(bbox_h, bbox_cx, img_h, img_w):
    v_fov     = H_FOV_DEG * (img_h / img_w)
    fl_v      = focal_length_px(img_h, v_fov)
    raw_depth = (PERSON_HEIGHT_M * fl_v) / bbox_h
    raw_angle_rad = np.arctan((bbox_cx - img_w / 2) / focal_length_px(img_w, H_FOV_DEG))
    slant = raw_depth / np.cos(raw_angle_rad)
    return max(0.0, (slant - DISTANCE_OFFSET_M) * DISTANCE_SCALE)


def estimate_angle(bbox_cx, img_w):
    fl = focal_length_px(img_w, H_FOV_DEG)
    return np.degrees(np.arctan((bbox_cx - img_w / 2) / fl)) * ANGLE_SCALE


def find_target_idx(detections, img_w, img_h):
    best_idx, best_score = None, float("inf")
    for i, ((x1, y1, x2, y2), cls_id) in enumerate(
        zip(detections.xyxy, detections.class_id)
    ):
        if cls_id != CLASS_ID["person"]:
            continue
        bbox_h  = y2 - y1
        bbox_cx = (x1 + x2) / 2
        dist    = estimate_distance(bbox_h, bbox_cx, img_h, img_w) if bbox_h > 0 else float("inf")
        angle   = abs(estimate_angle(bbox_cx, img_w))
        score   = TARGET_DIST_WEIGHT * dist + TARGET_ANGLE_WEIGHT * angle
        if score < best_score:
            best_score, best_idx = score, i
    return best_idx


# ---------------------------------------------------------------------------
# Darknet model wrapper
# ---------------------------------------------------------------------------

class DarknetModel:
    def __init__(self, cfg, weights, names_file, confidence, cls_id):
        self.net        = darknet.load_network(cfg, names_file, weights)
        self.net_w      = darknet.network_width(self.net)
        self.net_h      = darknet.network_height(self.net)
        self.dk_image   = darknet.make_image(self.net_w, self.net_h, 3)
        self.names      = open(names_file).read().strip().splitlines()
        self.confidence = confidence
        self.cls_id     = cls_id  # our internal class id for all detections from this model

    def infer(self, frame_rgb, orig_w, orig_h):
        """Run inference on an RGB frame; return (xyxy, conf, cls_id) arrays."""
        resized = cv2.resize(frame_rgb, (self.net_w, self.net_h),
                             interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(self.dk_image, resized.tobytes())
        detections = darknet.detect_image(
            self.net, self.names, self.dk_image,
            thresh=self.confidence, nms=NMS_THRESHOLD
        )

        xyxy_list, conf_list = [], []
        scale_x = orig_w / self.net_w
        scale_y = orig_h / self.net_h

        for class_name, confidence, bbox in detections:
            if class_name != "person" and self.cls_id == CLASS_ID["person"]:
                continue  # person model — only keep person class
            cx, cy, bw, bh = bbox
            x1 = (cx - bw / 2) * scale_x
            y1 = (cy - bh / 2) * scale_y
            x2 = (cx + bw / 2) * scale_x
            y2 = (cy + bh / 2) * scale_y
            xyxy_list.append([x1, y1, x2, y2])
            conf_list.append(float(confidence) / 100.0)

        return xyxy_list, conf_list

    def free(self):
        darknet.free_image(self.dk_image)


# ---------------------------------------------------------------------------
# Frame inference
# ---------------------------------------------------------------------------

def infer_frame(person_model, cart_model, frame):
    img_h, img_w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    all_xyxy, all_conf, all_cls = [], [], []

    p_xyxy, p_conf = person_model.infer(frame_rgb, img_w, img_h)
    if p_xyxy:
        all_xyxy.extend(p_xyxy)
        all_conf.extend(p_conf)
        all_cls.extend([CLASS_ID["person"]] * len(p_xyxy))

    if cart_model is not None:
        c_xyxy, c_conf = cart_model.infer(frame_rgb, img_w, img_h)
        if c_xyxy:
            all_xyxy.extend(c_xyxy)
            all_conf.extend(c_conf)
            all_cls.extend([CLASS_ID["cart"]] * len(c_xyxy))

    if not all_xyxy:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=np.array(all_xyxy, dtype=float),
        confidence=np.array(all_conf, dtype=float),
        class_id=np.array(all_cls, dtype=int),
    )


# ---------------------------------------------------------------------------
# Annotation (identical logic to yolo_detect.py)
# ---------------------------------------------------------------------------

def annotate_frame(frame, detections: sv.Detections, smooth_state: dict):
    img_h, img_w = frame.shape[:2]
    out = frame.copy()

    target_idx = find_target_idx(detections, img_w, img_h)
    ids = detections.tracker_id if detections.tracker_id is not None else [None] * len(detections)

    rows = []
    for i, ((x1, y1, x2, y2), cls_id, conf, tid) in enumerate(zip(
        detections.xyxy, detections.class_id, detections.confidence, ids
    )):
        is_target = (i == target_idx)
        color     = COLOR_TARGET if is_target else COLOR_OBSTACLE
        role      = "TARGET" if is_target else "OBSTACLE"
        label_id  = f"ID{tid}" if tid is not None else "?"
        class_tag = CLASS_NAME[cls_id]

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

        display_role = role if is_target else class_tag.upper()
        label = f"{display_role} {label_id} {dist:.1f}m {angle:+.1f}deg"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        top = max(int(y1) - 10, th + 4)
        cv2.rectangle(out, (int(x1), top - th - 4), (int(x1) + tw, top), color, -1)
        cv2.putText(out, label, (int(x1), top - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        rows.append((role, label_id, class_tag, conf, dist, angle))

    return out, rows


def draw_map(rows):
    s = MAP_SIZE
    img = np.zeros((s, s, 3), dtype=np.uint8)
    cam_px = s // 2
    cam_py = s - 40
    scale  = (s - 60) / MAP_RANGE_M

    def to_px(dist, angle_deg):
        rad = np.radians(angle_deg)
        px = int(cam_px + dist * np.sin(rad) * scale)
        py = int(cam_py - dist * np.cos(rad) * scale)
        return px, py

    for r_m in range(2, int(MAP_RANGE_M) + 1, 2):
        r_px = int(r_m * scale)
        cv2.circle(img, (cam_px, cam_py), r_px, (50, 50, 50), 1)
        cv2.putText(img, f"{r_m}m", (cam_px + r_px + 3, cam_py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)

    for side in (-1, 1):
        end = to_px(MAP_RANGE_M, side * H_FOV_DEG / 2)
        cv2.line(img, (cam_px, cam_py), end, (40, 40, 40), 1)

    cv2.line(img, (cam_px, cam_py), (cam_px, cam_py - int(MAP_RANGE_M * scale)),
             (40, 40, 40), 1)

    for role, label_id, class_tag, conf, dist, angle in rows:
        color = COLOR_TARGET if role == "TARGET" else COLOR_OBSTACLE
        px, py = to_px(dist, angle)
        if role == "TARGET":
            radius = 10
            cv2.circle(img, (px, py), radius, color, -1)
        elif class_tag == "cart":
            half = 7
            cv2.rectangle(img, (px - half, py - half), (px + half, py + half), color, -1)
            radius = half
        else:
            radius = 7
            cv2.circle(img, (px, py), radius, color, -1)
        display_role = role if role == "TARGET" else class_tag.upper()
        cv2.putText(img, f"{display_role} {label_id}", (px + radius + 2, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    pts = np.array([[cam_px, cam_py - 12], [cam_px - 10, cam_py + 6],
                    [cam_px + 10, cam_py + 6]], np.int32)
    cv2.fillPoly(img, [pts], (200, 200, 200))
    cv2.putText(img, "OVERHEAD MAP", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    return img


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models():
    import os
    for path in [PERSON_CFG, PERSON_WEIGHTS, PERSON_NAMES]:
        if not os.path.exists(path):
            print(f"ERROR: missing {path}")
            print("See setup instructions at the top of this file.")
            sys.exit(1)

    person_model = DarknetModel(PERSON_CFG, PERSON_WEIGHTS, PERSON_NAMES,
                                PERSON_CONFIDENCE, CLASS_ID["person"])
    print(f"Loaded person model: {PERSON_WEIGHTS}  ({person_model.net_w}x{person_model.net_h})")

    cart_model = None
    if os.path.exists(CART_CFG) and os.path.exists(CART_WEIGHTS) and os.path.exists(CART_NAMES):
        cart_model = DarknetModel(CART_CFG, CART_WEIGHTS, CART_NAMES,
                                  CART_CONFIDENCE, CLASS_ID["cart"])
        print(f"Loaded cart model: {CART_WEIGHTS}")
    else:
        print("[warn] No darknet cart model found — person detection only.")
        print("       To add cart detection, provide cart_darknet.cfg / .weights / .names")

    return person_model, cart_model


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def print_header():
    print(f"\n{'Role':<10} {'ID':<8} {'Class':<10} {'Conf':>6}  {'Distance':>10}  {'Angle':>8}")
    print("-" * 60)


def run_image(source, person_model, cart_model):
    frame = cv2.imread(source)
    if frame is None:
        print(f"Could not load {source}")
        return

    tracker = sv.ByteTrack()
    dets    = infer_frame(person_model, cart_model, frame)
    tracked = tracker.update_with_detections(dets)
    out, rows = annotate_frame(frame, tracked, {})

    print(f"\n{source} — {len(rows)} object(s) detected")
    print_header()
    for role, label_id, class_tag, conf, dist, angle in rows:
        print(f"{role:<10} {label_id:<8} {class_tag:<10} {conf:>6.0%}  {dist:>8.1f}m  {angle:>+7.1f}°")

    out_path = source.rsplit(".", 1)[0] + "_tracked.jpg"
    cv2.imwrite(out_path, out)
    print(f"\nSaved: {out_path}")


def run_video(source, person_model, cart_model):
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

    tracker      = sv.ByteTrack()
    smooth_state = {}

    writer = None
    if isinstance(source, str):
        out_path = source.rsplit(".", 1)[0] + "_tracked.mp4"
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    print(f"\nTracking — press Q to quit\n")
    print_header()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets    = infer_frame(person_model, cart_model, frame)
        tracked = tracker.update_with_detections(dets)
        out, rows = annotate_frame(frame, tracked, smooth_state)

        for role, label_id, class_tag, conf, dist, angle in rows:
            print(f"{role:<10} {label_id:<8} {class_tag:<10} {conf:>6.0%}  {dist:>8.1f}m  {angle:>+7.1f}°  [f{frame_idx}]")

        if writer:
            writer.write(out)
        cv2.imshow("Darknet Track", out)
        cv2.imshow("Overhead Map", draw_map(rows))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
        print(f"\nSaved: {out_path}")
    cv2.destroyAllWindows()


def run():
    if len(sys.argv) < 2:
        print("Usage: python3 darknet_detect.py <image|video|webcam_id>")
        sys.exit(1)
    source = sys.argv[1]
    person_model, cart_model = load_models()

    is_image = isinstance(source, str) and source.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    )
    try:
        if is_image:
            run_image(source, person_model, cart_model)
        else:
            run_video(source, person_model, cart_model)
    finally:
        person_model.free()
        if cart_model:
            cart_model.free()


if __name__ == "__main__":
    run()
