"""
Detect people and shopping carts, track each with ByteTrack.

Models:
  yolo11n.pt  — COCO model, used only for 'person' (class 0)
  cart.pt     — fine-tuned model from train_cart.py, detects 'shopping-cart'

Usage:
  python3 yolo_detect.py                     # default image
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

PERSON_MODEL_PATH = _DIR / "yolo26n.pt"
CART_MODEL_PATH   = _DIR / "cart.pt"
PERSON_CONFIDENCE = 0.45
CART_CONFIDENCE   = 0.30
DEFAULT_IMAGE     = str(_DIR / "Inside_Costco_Perth.jpg")

PERSON_HEIGHT_M  = 1.7
H_FOV_DEG        = 60.0
DISTANCE_OFFSET_M = 3.0   # calibration offset — subtract from computed distance

CLASS_ID   = {"person": 0, "cart": 1}
CLASS_NAME = {0: "person", 1: "cart"}

COLOR_TARGET   = (0, 255, 0)     # green — locked shopper
COLOR_OBSTACLE = (0, 0, 255)     # red   — everyone/everything else

MAP_SIZE    = 500    # pixels per side for the overhead map
MAP_RANGE_M = 10.0   # meters shown front-to-back


def focal_length_px(dim, fov_deg):
    return (dim / 2) / np.tan(np.radians(fov_deg / 2))


def estimate_distance(bbox_h, img_h, img_w):
    v_fov = H_FOV_DEG * (img_h / img_w)
    fl = focal_length_px(img_h, v_fov)
    return max(0.0, (PERSON_HEIGHT_M * fl) / bbox_h - DISTANCE_OFFSET_M)


def find_target_idx(detections, img_w, img_h):
    """Return index of the person detection whose center is closest to the frame center."""
    best_idx, best_dist = None, float("inf")
    for i, ((x1, y1, x2, y2), cls_id) in enumerate(
        zip(detections.xyxy, detections.class_id)
    ):
        if cls_id != CLASS_ID["person"]:
            continue
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        d = (cx - img_w / 2) ** 2 + (cy - img_h / 2) ** 2
        if d < best_dist:
            best_dist, best_idx = d, i
    return best_idx


def estimate_angle(bbox_cx, img_w):
    fl = focal_length_px(img_w, H_FOV_DEG)
    return np.degrees(np.arctan((bbox_cx - img_w / 2) / fl))


def infer_frame(person_model, cart_model, frame):
    """Run both models and return a combined sv.Detections."""
    img_h, img_w = frame.shape[:2]

    all_xyxy, all_conf, all_cls = [], [], []

    # --- person model (COCO class 0 only) ---
    p_res = person_model(frame, conf=PERSON_CONFIDENCE, classes=[0], verbose=False)[0]
    if len(p_res.boxes):
        all_xyxy.append(p_res.boxes.xyxy.cpu().numpy())
        all_conf.append(p_res.boxes.conf.cpu().numpy())
        all_cls.append(np.full(len(p_res.boxes), CLASS_ID["person"], dtype=int))

    # --- cart model (all its classes map to our 'cart' id) ---
    if cart_model is not None:
        c_res = cart_model(frame, conf=CART_CONFIDENCE, augment=True, verbose=False)[0]
        if len(c_res.boxes):
            all_xyxy.append(c_res.boxes.xyxy.cpu().numpy())
            all_conf.append(c_res.boxes.conf.cpu().numpy())
            all_cls.append(np.full(len(c_res.boxes), CLASS_ID["cart"], dtype=int))

    if not all_xyxy:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=np.concatenate(all_xyxy),
        confidence=np.concatenate(all_conf),
        class_id=np.concatenate(all_cls),
    )


def annotate_frame(frame, detections: sv.Detections):
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

        dist  = estimate_distance(bbox_h, img_h, img_w) if bbox_h > 0 else 0
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
    """Render a top-down 2D map of detections relative to the camera."""
    s = MAP_SIZE
    img = np.zeros((s, s, 3), dtype=np.uint8)

    # Camera sits at the bottom-center of the map
    cam_px = s // 2
    cam_py = s - 40
    scale  = (s - 60) / MAP_RANGE_M   # pixels per metre

    def to_px(dist, angle_deg):
        rad = np.radians(angle_deg)
        px = int(cam_px + dist * np.sin(rad) * scale)
        py = int(cam_py - dist * np.cos(rad) * scale)
        return px, py

    # Grid rings every 2 m
    for r_m in range(2, int(MAP_RANGE_M) + 1, 2):
        r_px = int(r_m * scale)
        cv2.circle(img, (cam_px, cam_py), r_px, (50, 50, 50), 1)
        cv2.putText(img, f"{r_m}m", (cam_px + r_px + 3, cam_py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)

    # FOV cone (H_FOV_DEG wide)
    for side in (-1, 1):
        end = to_px(MAP_RANGE_M, side * H_FOV_DEG / 2)
        cv2.line(img, (cam_px, cam_py), end, (40, 40, 40), 1)

    # Centre forward line
    cv2.line(img, (cam_px, cam_py), (cam_px, cam_py - int(MAP_RANGE_M * scale)),
             (40, 40, 40), 1)

    # Detections
    for role, label_id, class_tag, conf, dist, angle in rows:
        color = COLOR_TARGET if role == "TARGET" else COLOR_OBSTACLE
        px, py = to_px(dist, angle)
        if role == "TARGET":
            radius = 10
            cv2.circle(img, (px, py), radius, color, -1)
        elif class_tag == "cart":
            # draw a square for carts
            half = 7
            cv2.rectangle(img, (px - half, py - half), (px + half, py + half), color, -1)
            radius = half
        else:
            radius = 7
            cv2.circle(img, (px, py), radius, color, -1)
        display_role = role if role == "TARGET" else class_tag.upper()
        tag = f"{display_role} {label_id}"
        cv2.putText(img, tag, (px + radius + 2, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    # Camera icon
    pts = np.array([
        [cam_px,      cam_py - 12],
        [cam_px - 10, cam_py + 6],
        [cam_px + 10, cam_py + 6],
    ], np.int32)
    cv2.fillPoly(img, [pts], (200, 200, 200))

    cv2.putText(img, "OVERHEAD MAP", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    return img


def load_models():
    person_model = YOLO(PERSON_MODEL_PATH)
    cart_model = None
    import os
    if os.path.exists(CART_MODEL_PATH):
        cart_model = YOLO(CART_MODEL_PATH)
        print(f"Loaded cart model: {CART_MODEL_PATH}")
    else:
        print(f"[warn] {CART_MODEL_PATH} not found — run train_cart.py first. "
              "Continuing with person-only detection.")
    return person_model, cart_model


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
    out, rows = annotate_frame(frame, tracked)

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

    tracker = sv.ByteTrack()

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
        out, rows = annotate_frame(frame, tracked)

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
    source = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    person_model, cart_model = load_models()

    is_image = isinstance(source, str) and source.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    )
    if is_image:
        run_image(source, person_model, cart_model)
    else:
        run_video(source, person_model, cart_model)


if __name__ == "__main__":
    run()