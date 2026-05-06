"""
Detect people (and optionally shopping carts), track each with ByteTrack.

Models:
  yolo26n.pt  — COCO pretrained, person-only mode (default)
  combined.pt — YOLOv11n trained on COCO persons + Roboflow carts
                Switch MODEL_PATH and set PERSON_ONLY = False to enable cart detection.
                Train with: python3 train_combined.py --api-key <KEY>

Usage:
  python3 yolo_detect.py image.jpg           # image file
  python3 yolo_detect.py video.mp4           # video file
  python3 yolo_detect.py 0                   # webcam
"""

import sys
import warnings
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

_DIR = Path(__file__).parent

MODEL_PATH        = _DIR / "yolo26n.pt"   # switch to combined.pt for cart detection
PERSON_ONLY       = True                  # set False when using combined.pt
PERSON_CONFIDENCE = 0.35
CART_CONFIDENCE   = 0.30

PERSON_HEIGHT_M   = 1.7
H_FOV_DEG         = 54.0
DISTANCE_OFFSET_M = 3.0   # calibration offset — subtract from computed distance
DISTANCE_SCALE    = 1.0   # multiply final distance estimate
ANGLE_SCALE       = 1.0   # multiply final angle estimate

CLASS_ID   = {"person": 0, "cart": 1}
CLASS_NAME = {0: "person", 1: "cart"}

COLOR_TARGET   = (0, 255, 0)
COLOR_OBSTACLE = (0, 0, 255)

MAP_SIZE    = 500
MAP_RANGE_M = 10.0

TARGET_DIST_WEIGHT  = 1.0
TARGET_ANGLE_WEIGHT = 0.3

DIST_EMA_ALPHA = 0.4


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


def infer_frame(model, frame):
    results = model(frame, verbose=False, conf=0.1)[0]
    if not len(results.boxes):
        return sv.Detections.empty()

    xyxy     = results.boxes.xyxy.cpu().numpy()
    conf     = results.boxes.conf.cpu().numpy()
    cls_ids  = results.boxes.cls.cpu().numpy().astype(int)

    # Apply per-class confidence thresholds
    thresholds = np.where(cls_ids == CLASS_ID["person"], PERSON_CONFIDENCE, CART_CONFIDENCE)
    keep = conf >= thresholds
    if PERSON_ONLY:
        keep &= cls_ids == CLASS_ID["person"]
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
    for i, ((x1, y1, x2, y2), cls_id, conf, tid) in enumerate(zip(
        detections.xyxy, detections.class_id, detections.confidence, ids
    )):
        is_target = (i == target_idx)
        color     = COLOR_TARGET if is_target else COLOR_OBSTACLE
        role      = "TARGET" if is_target else "OBSTACLE"
        label_id  = f"ID{tid}" if tid is not None else "?"
        class_tag = CLASS_NAME.get(cls_id, str(cls_id))

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
    s      = MAP_SIZE
    img    = np.zeros((s, s, 3), dtype=np.uint8)
    cam_px = s // 2
    cam_py = s - 40
    scale  = (s - 60) / MAP_RANGE_M

    def to_px(dist, angle_deg):
        rad = np.radians(angle_deg)
        return (int(cam_px + dist * np.sin(rad) * scale),
                int(cam_py - dist * np.cos(rad) * scale))

    for r_m in range(2, int(MAP_RANGE_M) + 1, 2):
        r_px = int(r_m * scale)
        cv2.circle(img, (cam_px, cam_py), r_px, (50, 50, 50), 1)
        cv2.putText(img, f"{r_m}m", (cam_px + r_px + 3, cam_py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)

    for side in (-1, 1):
        cv2.line(img, (cam_px, cam_py), to_px(MAP_RANGE_M, side * H_FOV_DEG / 2), (40, 40, 40), 1)
    cv2.line(img, (cam_px, cam_py), (cam_px, cam_py - int(MAP_RANGE_M * scale)), (40, 40, 40), 1)

    for role, label_id, class_tag, conf, dist, angle in rows:
        color = COLOR_TARGET if role == "TARGET" else COLOR_OBSTACLE
        px, py = to_px(dist, angle)
        if class_tag == "cart":
            half = 7
            cv2.rectangle(img, (px - half, py - half), (px + half, py + half), color, -1)
            r = half
        else:
            r = 10 if role == "TARGET" else 7
            cv2.circle(img, (px, py), r, color, -1)
        display = role if role == "TARGET" else class_tag.upper()
        cv2.putText(img, f"{display} {label_id}", (px + r + 2, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    pts = np.array([[cam_px, cam_py - 12], [cam_px - 10, cam_py + 6],
                    [cam_px + 10, cam_py + 6]], np.int32)
    cv2.fillPoly(img, [pts], (200, 200, 200))
    cv2.putText(img, "OVERHEAD MAP", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    return img


def print_header():
    print(f"\n{'Role':<10} {'ID':<8} {'Class':<10} {'Conf':>6}  {'Distance':>10}  {'Angle':>8}")
    print("-" * 60)


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
    for role, label_id, class_tag, conf, dist, angle in rows:
        print(f"{role:<10} {label_id:<8} {class_tag:<10} {conf:>6.0%}  {dist:>8.1f}m  {angle:>+7.1f}°")

    out_path = source.rsplit(".", 1)[0] + "_tracked.jpg"
    cv2.imwrite(out_path, out)
    print(f"\nSaved: {out_path}")


def run_video(source, model):
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
    print("\nTracking — press Q to quit\n")
    print_header()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets    = infer_frame(model, frame)
        tracked = tracker.update_with_detections(dets)
        out, rows = annotate_frame(frame, tracked, smooth_state)

        for role, label_id, class_tag, conf, dist, angle in rows:
            print(f"{role:<10} {label_id:<8} {class_tag:<10} {conf:>6.0%}  {dist:>8.1f}m  {angle:>+7.1f}°  [f{frame_idx}]")

        if writer:
            writer.write(out)
        cv2.imshow("ByteTrack", out)
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
        print("Usage: python3 yolo_detect.py <image|video|webcam_id>")
        sys.exit(1)

    if not MODEL_PATH.exists():
        print(f"ERROR: {MODEL_PATH} not found.")
        if not PERSON_ONLY:
            print("Train the combined model first:")
            print("  python3 train_combined.py --api-key <YOUR_ROBOFLOW_KEY>")
        sys.exit(1)

    model = YOLO(MODEL_PATH)
    source = sys.argv[1]

    is_image = isinstance(source, str) and source.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    )
    if is_image:
        run_image(source, model)
    else:
        run_video(source, model)


if __name__ == "__main__":
    run()
