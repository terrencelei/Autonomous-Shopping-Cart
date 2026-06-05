"""
Color vector data collector for weight training.

Runs the IMX500 detector live. The largest bounding box is treated as the
subject. Press a number key (0-9) to save the current vector as that person
label. Press Q to quit.

Vectors are appended to vectors.csv in the format:
  person_id, dist_m, frame, H0,S0,V0, H1,S1,V1, ..., H11,S11,V11

Usage:
  python3 collect_vectors.py
  python3 collect_vectors.py --out my_data.csv
  python3 collect_vectors.py --no-display   # headless — press Ctrl+C and
                                             # it will prompt for a label
"""

import os
import sys
import csv
import time
import argparse
import warnings
from pathlib import Path

def configure_qt_fonts():
    if os.environ.get("QT_QPA_FONTDIR"):
        return
    for font_dir in (
            "/usr/share/fonts/truetype/dejavu",
            "/usr/share/fonts/truetype/liberation2",
            "/usr/share/fonts/truetype/freefont"):
        if os.path.isdir(font_dir):
            os.environ["QT_QPA_FONTDIR"] = font_dir
            return

configure_qt_fonts()

import cv2
import numpy as np
import supervision as sv

from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics

sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore", category=FutureWarning)

RPK_MODEL_PATH = Path(
    "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
)

PERSON_CONFIDENCE = 0.5
PERSON_CLASS_ID   = 0

H_FOV_DEG         = 66.0
PERSON_HEIGHT_M   = 1.8
DISTANCE_SCALE    = 0.7
DISTANCE_OFFSET_M = 0

N_PROFILE_CHUNKS  = 12
BOX_TRIM_X        = 0.12
BOX_TRIM_Y        = 0.08
MIN_ROW_PX        = 4

COLOR_SUBJECT = (0, 255, 0)
COLOR_OTHER   = (128, 128, 128)


# =========================================================================== #
# Camera
# =========================================================================== #

class IMX500Capture:
    def __init__(self, model_path, width=640, height=480, fps=30):
        self._imx500 = IMX500(str(model_path))
        intrinsics = self._imx500.network_intrinsics or NetworkIntrinsics()
        intrinsics.task = "object detection"
        self._picam2 = Picamera2(self._imx500.camera_num)
        cfg = self._picam2.create_preview_configuration(
            main={"format": "BGR888", "size": (width, height)},
            controls={"FrameRate": fps},
            buffer_count=12,
        )
        self._imx500.show_network_fw_progress_bar()
        self._picam2.start(cfg)
        self._meta = None
        self._last_dets = sv.Detections.empty()
        self._w, self._h = width, height

    def read(self):
        req = self._picam2.capture_request()
        frame = req.make_array("main")
        self._meta = req.get_metadata()
        req.release()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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

    def release(self):
        self._picam2.stop()
        self._picam2.close()


# =========================================================================== #
# Geometry
# =========================================================================== #

def focal_length_px(dim, fov_deg):
    return (dim / 2) / np.tan(np.radians(fov_deg / 2))

def estimate_distance(bbox_h, img_h, img_w):
    v_fov     = H_FOV_DEG * (img_h / img_w)
    fl_v      = focal_length_px(img_h, v_fov)
    raw_depth = (PERSON_HEIGHT_M * fl_v) / bbox_h
    return max(0.0, (raw_depth - DISTANCE_OFFSET_M) * DISTANCE_SCALE)


# =========================================================================== #
# Color profile  [H, S, V] — all three channels stored for dataset use
# =========================================================================== #

def clothing_color_profile(hsv_frame, xyxy):
    """
    Returns (N_PROFILE_CHUNKS, 3) float32 [H, S, V].
    All three channels are stored so the dataset can be used to evaluate
    whether V adds discriminating power after collection.
    """
    img_h, img_w = hsv_frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(img_w - 1, int(x1)))
    x2 = max(0, min(img_w,     int(x2)))
    y1 = max(0, min(img_h - 1, int(y1)))
    y2 = max(0, min(img_h,     int(y2)))

    box_w = x2 - x1
    box_h = y2 - y1
    if box_w < 8 or box_h < N_PROFILE_CHUNKS:
        return None

    tx = int(box_w * BOX_TRIM_X)
    ty = int(box_h * BOX_TRIM_Y)
    ix1, ix2 = x1 + tx, x2 - tx
    iy1, iy2 = y1 + ty, y2 - ty
    if ix2 <= ix1 or iy2 <= iy1:
        return None

    inner  = hsv_frame[iy1:iy2, ix1:ix2].astype(np.float32)
    n_rows = inner.shape[0]

    row_avgs = np.full((n_rows, 3), np.nan, dtype=np.float32)
    for r in range(n_rows):
        row = inner[r]
        if len(row) >= MIN_ROW_PX:
            row_avgs[r] = row.mean(axis=0)   # [H, S, V]

    chunk_size = n_rows / N_PROFILE_CHUNKS
    profile    = np.full((N_PROFILE_CHUNKS, 3), np.nan, dtype=np.float32)
    for c in range(N_PROFILE_CHUNKS):
        r_start = int(round(c * chunk_size))
        r_end   = int(round((c + 1) * chunk_size))
        valid   = row_avgs[r_start:r_end]
        valid   = valid[~np.isnan(valid[:, 0])]
        if len(valid) > 0:
            profile[c] = valid.mean(axis=0)

    if np.all(np.isnan(profile[:, 0])):
        return None

    for c in range(N_PROFILE_CHUNKS):
        if np.isnan(profile[c, 0]):
            prev_c = next((i for i in range(c - 1, -1, -1) if not np.isnan(profile[i, 0])), None)
            next_c = next((i for i in range(c + 1, N_PROFILE_CHUNKS) if not np.isnan(profile[i, 0])), None)
            if prev_c is not None and next_c is not None:
                profile[c] = (profile[prev_c] + profile[next_c]) / 2
            elif prev_c is not None:
                profile[c] = profile[prev_c]
            elif next_c is not None:
                profile[c] = profile[next_c]

    return profile


# =========================================================================== #
# CSV helpers
# =========================================================================== #

def csv_header():
    cols = ["person_id", "dist_m", "frame"]
    for c in range(N_PROFILE_CHUNKS):
        cols += [f"H{c}", f"S{c}", f"V{c}"]
    return cols

def profile_to_row(person_id, dist_m, frame_idx, profile):
    row = [person_id, f"{dist_m:.2f}", frame_idx]
    for c in range(N_PROFILE_CHUNKS):
        row += [f"{profile[c, 0]:.2f}", f"{profile[c, 1]:.2f}", f"{profile[c, 2]:.2f}"]
    return row

def save_vector(writer, person_id, dist_m, frame_idx, profile):
    writer.writerow(profile_to_row(person_id, dist_m, frame_idx, profile))


# =========================================================================== #
# Annotation
# =========================================================================== #

def draw_frame(frame, detections, subject_idx, current_profile, saved_count, pending_label):
    out      = frame.copy()
    img_h, img_w = frame.shape[:2]

    for i, (x1, y1, x2, y2) in enumerate(detections.xyxy):
        is_subject = (i == subject_idx)
        color      = COLOR_SUBJECT if is_subject else COLOR_OTHER
        thickness  = 3 if is_subject else 1
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        if is_subject:
            cv2.putText(out, "SUBJECT", (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Swatch strip for subject
    if subject_idx is not None and current_profile is not None and len(detections) > 0:
        x1, y1, x2, y2 = detections.xyxy[subject_idx]
        swatch_w = max(8, int((x2 - x1) * 0.08))
        swatch_h = max(2, int((y2 - y1) / N_PROFILE_CHUNKS))
        strip_x  = int(x1) + 4
        for ci in range(N_PROFILE_CHUNKS):
            sy1    = int(y1) + ci * swatch_h
            sy2    = sy1 + swatch_h
            h_val  = float(np.clip(current_profile[ci, 0], 0, 179))
            s_val  = float(np.clip(current_profile[ci, 1], 0, 255))
            v_val  = float(np.clip(current_profile[ci, 2], 0, 255))
            hsv_px = np.uint8([[[h_val, s_val, v_val]]])
            bgr_px = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0, 0].tolist()
            cv2.rectangle(out, (strip_x, sy1), (strip_x + swatch_w, sy2), bgr_px, -1)
        cv2.rectangle(out, (strip_x, int(y1) + 4),
                      (strip_x + swatch_w, int(y1) + 4 + N_PROFILE_CHUNKS * swatch_h),
                      (255, 255, 255), 1)

    # HUD
    lines = [
        f"Saved: {saved_count} vectors",
        "Press 0-9 to label subject",
        "Press Q to quit",
    ]
    if pending_label is not None:
        lines.insert(0, f"*** SAVED as person {pending_label} ***")
    for i, line in enumerate(lines):
        cv2.putText(out, line, (10, 28 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return out


# =========================================================================== #
# Main
# =========================================================================== #

def run(out_path, no_display):
    if not RPK_MODEL_PATH.exists():
        raise SystemExit(f"ERROR: model not found at {RPK_MODEL_PATH}")

    cap = IMX500Capture(RPK_MODEL_PATH)
    out_path = Path(out_path)
    write_header = not out_path.exists()

    print(f"\nSaving vectors to: {out_path}")
    print("Press 0-9 to label the green (largest) bounding box as that person.")
    print("Press Q to quit.\n")

    frame_idx     = 0
    saved_count   = 0
    pending_label = None   # shown in HUD for a few frames after save
    pending_clear = 0

    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(csv_header())

        try:
            while True:
                frame = cap.read()
                dets  = cap.get_detections()
                img_h, img_w = frame.shape[:2]

                # Subject = largest bounding box
                subject_idx     = None
                subject_profile = None
                subject_dist    = None

                if len(dets) > 0:
                    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in dets.xyxy]
                    subject_idx = int(np.argmax(areas))
                    xyxy        = dets.xyxy[subject_idx]
                    bbox_h      = xyxy[3] - xyxy[1]
                    subject_dist = estimate_distance(bbox_h, img_h, img_w) if bbox_h > 0 else 0.0

                    hsv_frame       = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    subject_profile = clothing_color_profile(hsv_frame, xyxy)

                # Clear pending label after 30 frames
                if pending_clear > 0:
                    pending_clear -= 1
                    if pending_clear == 0:
                        pending_label = None

                if not no_display:
                    out = draw_frame(frame, dets, subject_idx,
                                     subject_profile, saved_count, pending_label)
                    cv2.imshow("Vector Collector", out)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        break
                    elif ord("0") <= key <= ord("9") and subject_profile is not None:
                        person_id = key - ord("0")
                        save_vector(writer, person_id, subject_dist, frame_idx, subject_profile)
                        f.flush()
                        saved_count  += 1
                        pending_label = person_id
                        pending_clear = 30
                        print(f"  Saved frame {frame_idx}  person={person_id}  "
                              f"dist={subject_dist:.1f}m  ({saved_count} total)")
                else:
                    # Headless: print current vector summary, Ctrl+C to label
                    if subject_profile is not None and frame_idx % 30 == 0:
                        h_mean = subject_profile[:, 0].mean()
                        s_mean = subject_profile[:, 1].mean()
                        v_mean = subject_profile[:, 2].mean()
                        print(f"f{frame_idx:5d}  dist={subject_dist:.1f}m  "
                              f"H={h_mean:.1f} S={s_mean:.1f} V={v_mean:.1f}  "
                              f"saved={saved_count}")

                frame_idx += 1

        except KeyboardInterrupt:
            if no_display and subject_profile is not None:
                label = input("\nEnter person label (0-9): ").strip()
                if label.isdigit():
                    person_id = int(label)
                    save_vector(writer, person_id, subject_dist, frame_idx, subject_profile)
                    f.flush()
                    saved_count += 1
                    print(f"Saved as person {person_id}  ({saved_count} total)")
        finally:
            cap.release()
            if not no_display:
                cv2.destroyAllWindows()

    print(f"\nDone. {saved_count} vectors saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect HSV color vectors for weight training")
    parser.add_argument("--out",        default="vectors.csv", help="Output CSV file")
    parser.add_argument("--no-display", dest="no_display", action="store_true")
    args = parser.parse_args()

    if not args.no_display and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        args.no_display = True
        print("No display — running headless. Ctrl+C to label current frame.")

    run(args.out, args.no_display)
