"""
Detect and track people with YOLO11n on the IMX500 NPU + ByteTrack.
Provides the target/obstacle tracking that follow.py (the main program) drives
on. Run directly for a vision-only preview — no motor control lives here.

Usage:
  python3 yolo_detect.py                # vision preview
  python3 yolo_detect.py --no-display   # headless (SSH)

Color signature pipeline (clothing_color_profile):
  1. Convert full frame BGR -> LAB once per frame
  2. Tighten bounding box inward to reduce background bleed
  3. Sample a band outside the box; use its average L to normalise lighting
  4. For each pixel row, reject pixels whose a* and b* are both close to the
     surrounding band average (background bleed rejection)
  5. Average the remaining pixels per row
  6. Compress into N_PROFILE_CHUNKS chunks
  7. Return (N_PROFILE_CHUNKS x 2) float32 array [a*, b*] — L discarded
     since lighting is already normalised out and not compared
"""

import os
import sys
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

warnings.filterwarnings("ignore", category=FutureWarning)

RPK_MODEL_PATH = Path(
    "/usr/share/imx500-models/imx500_network_yolo11n_pp.rpk"
)

PERSON_CONFIDENCE = 0.5
PERSON_CLASS_ID   = 0

# YOLO11n post-processed model settings from raspberrypi/imx500-models:
# imx500_object_detection_demo.py --bbox-normalization --bbox-order xy
MODEL_BBOX_NORMALIZATION = True
MODEL_BBOX_ORDER = "xy"

PERSON_HEIGHT_M   = 1.8
H_FOV_DEG         = 66.0
DISTANCE_OFFSET_M = 0
DISTANCE_SCALE    = 0.7
ANGLE_SCALE       = 0.99
ANGLE_OFFSET_DEG  = 2.0

COLOR_TARGET   = (0, 255, 0)
COLOR_OBSTACLE = (0, 0, 255)

TARGET_DIST_WEIGHT  = 1.0
TARGET_ANGLE_WEIGHT = 0.3

DIST_EMA_ALPHA  = 0.4
ANGLE_EMA_ALPHA = 0.4

TARGET_LOST_TIMEOUT_S           = 0.75
TARGET_SWITCH_MARGIN            = 2.0
TARGET_SWITCH_FRAMES            = 12
TARGET_REID_MAX_DIST_DELTA_M    = 1.0
TARGET_REID_MAX_ANGLE_DELTA_DEG = 12.0
TARGET_COLOR_EMA_ALPHA          = 0.25
TARGET_COLOR_REID_MIN_SCORE     = 0.70
TARGET_COLOR_SWITCH_MIN_SCORE   = 0.70
TARGET_COLOR_SWITCH_PENALTY     = 1.5

SIM_SMOOTH_WINDOW  = 10
ACCUM_MAX_DURATION_S = 10.0

# --------------------------------------------------------------------------- #
# Color profile tuning
# --------------------------------------------------------------------------- #
N_PROFILE_CHUNKS = 12
BOX_TRIM_X       = 0.12
BOX_TRIM_Y       = 0.08
SURROUND_PX      = 12
MIN_ROW_PX       = 4

# Background rejection: pixel excluded if a* and b* are both within these
# deltas of the surrounding band average (LAB units, 0-255 OpenCV scale).
BG_A_DELTA = 10
BG_B_DELTA = 10


# =========================================================================== #
# Camera capture
# =========================================================================== #

class IMX500Capture:
    def __init__(self, model_path, width=640, height=480, fps=30):
        self._imx500 = IMX500(str(model_path))
        intrinsics = self._imx500.network_intrinsics or NetworkIntrinsics()
        intrinsics.task = "object detection"
        intrinsics.bbox_normalization = MODEL_BBOX_NORMALIZATION
        intrinsics.bbox_order = MODEL_BBOX_ORDER
        intrinsics.update_with_defaults()
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
        _, input_h = self._imx500.get_input_size()

        boxes = boxes_raw.astype(float)
        if self._intrinsics.bbox_normalization:
            boxes = boxes / input_h
        if self._intrinsics.bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]

        keep = (scores >= PERSON_CONFIDENCE) & (classes == PERSON_CLASS_ID)
        if not keep.any():
            self._last_dets = sv.Detections.empty()
            return self._last_dets
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

        xyxy = []
        for box in boxes:
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


# =========================================================================== #
# Geometry helpers
# =========================================================================== #

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
    return np.degrees(np.arctan((bbox_cx - img_w / 2) / fl)) * ANGLE_SCALE - ANGLE_OFFSET_DEG


def detection_score(dist, angle):
    return TARGET_DIST_WEIGHT * dist + TARGET_ANGLE_WEIGHT * abs(angle)


def normalize_track_id(tid):
    if tid is None:
        return None
    try:
        if np.isnan(tid):
            return None
    except TypeError:
        pass
    return int(tid)


# =========================================================================== #
# Color profile  (LAB — a* and b* only)
# =========================================================================== #

def clothing_color_profile(lab_frame, xyxy):
    """
    Build a 1-D vertical color profile in LAB space.

    Returns (N_PROFILE_CHUNKS, 2) float32 [a*, b*], or None.
    L is used only for background rejection and is not stored.

    LAB in OpenCV: L in [0,255], a* and b* in [0,255] centred at 128.
    """
    img_h, img_w = lab_frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(img_w - 1, int(x1)))
    x2 = max(0, min(img_w,     int(x2)))
    y1 = max(0, min(img_h - 1, int(y1)))
    y2 = max(0, min(img_h,     int(y2)))

    box_w = x2 - x1
    box_h = y2 - y1
    if box_w < 8 or box_h < N_PROFILE_CHUNKS:
        return None

    # Tighten box
    tx = int(box_w * BOX_TRIM_X)
    ty = int(box_h * BOX_TRIM_Y)
    ix1, ix2 = x1 + tx, x2 - tx
    iy1, iy2 = y1 + ty, y2 - ty
    if ix2 <= ix1 or iy2 <= iy1:
        return None

    # Sample surrounding band for background reference
    surround_pixels = []
    for sx1, sx2, sy1, sy2 in [
        (max(0, ix1 - SURROUND_PX), ix1,                  iy1, iy2),
        (ix2,                        min(img_w, ix2 + SURROUND_PX), iy1, iy2),
        (ix1, ix2, max(0, iy1 - SURROUND_PX), iy1),
        (ix1, ix2, iy2,                        min(img_h, iy2 + SURROUND_PX)),
    ]:
        if sx2 > sx1 and sy2 > sy1:
            surround_pixels.append(lab_frame[sy1:sy2, sx1:sx2].reshape(-1, 3))

    if not surround_pixels:
        return None

    surround_all = np.concatenate(surround_pixels, axis=0).astype(np.float32)
    surround_avg = surround_all.mean(axis=0)   # [L, a*, b*]

    # Per-row averages with background rejection on a* and b*
    inner  = lab_frame[iy1:iy2, ix1:ix2].astype(np.float32)
    n_rows = inner.shape[0]

    row_avgs = np.full((n_rows, 2), np.nan, dtype=np.float32)
    for r in range(n_rows):
        row    = inner[r]                              # (cols, 3)
        a_diff = np.abs(row[:, 1] - surround_avg[1])
        b_diff = np.abs(row[:, 2] - surround_avg[2])
        fg     = ~((a_diff < BG_A_DELTA) & (b_diff < BG_B_DELTA))
        if fg.sum() >= MIN_ROW_PX:
            row_avgs[r] = row[fg, 1:3].mean(axis=0)   # keep only a*, b*

    # Compress into N_PROFILE_CHUNKS
    chunk_size = n_rows / N_PROFILE_CHUNKS
    profile    = np.full((N_PROFILE_CHUNKS, 2), np.nan, dtype=np.float32)
    for c in range(N_PROFILE_CHUNKS):
        r_start = int(round(c * chunk_size))
        r_end   = int(round((c + 1) * chunk_size))
        valid   = row_avgs[r_start:r_end]
        valid   = valid[~np.isnan(valid[:, 0])]
        if len(valid) > 0:
            profile[c] = valid.mean(axis=0)

    if np.all(np.isnan(profile[:, 0])):
        return None

    # Interpolate isolated NaN chunks
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


def profile_similarity(a, b):
    """
    Weighted similarity between two (N_PROFILE_CHUNKS, 2) LAB profiles [a*, b*].

    Weights calibrated from bounding box measurements at 1.5-3.9m.
    Max meaningful delta per channel: ~127 (half of 0-255 range).
    Returns float in [0, 1], or None if either profile is None.
    """
    if a is None or b is None:
        return None

    # Empirically calibrated weights (see landmark fractions in comments)
    # chunks 0-1: head, 2: neck/shoulder, 3-5: upper/mid torso,
    # 6: waist, 7-9: upper legs, 10: lower legs, 11: feet
    weights = np.array(
        [0.05, 0.05, 0.30, 3.50, 3.50, 3.50, 2.00, 3.00, 3.00, 3.00, 0.80, 0.02],
        dtype=np.float32,
    )
    if len(weights) != N_PROFILE_CHUNKS:
        weights = np.interp(
            np.linspace(0, 1, N_PROFILE_CHUNKS),
            np.linspace(0, 1, 12),
            weights,
        ).astype(np.float32)

    # a* and b* each span ~127 units of meaningful range
    diff_a = np.abs(a[:, 0] - b[:, 0])
    diff_b = np.abs(a[:, 1] - b[:, 1])

    err_a = np.clip(diff_a / 127.0, 0.0, 1.0) ** 1.5
    err_b = np.clip(diff_b / 127.0, 0.0, 1.0) ** 1.5
    err   = (err_a + err_b) / 2.0

    weighted_err = np.dot(err, weights) / weights.sum()
    return float(np.clip(1.0 - weighted_err, 0.0, 1.0))


# =========================================================================== #
# Target lock
# =========================================================================== #

class TargetLock:
    """
    Locks onto the first person detected. Their averaged color profile over
    the first unbroken bbox run (up to ACCUM_MAX_DURATION_S) becomes the
    fixed reference_profile. All subsequent comparisons use this reference —
    it is never modified after commitment.

    color_profile is an EMA-updated live profile used only for swatch display.
    """

    def __init__(self):
        self.locked_id               = None
        self.last_seen               = 0.0
        self.last_dist               = None
        self.last_angle              = None
        self.reference_profile       = None
        self.color_profile           = None
        self.switch_candidate_id     = None
        self.switch_candidate_frames = 0
        self._accum_id               = None
        self._accum_profiles         = []
        self._accum_start            = None

    def choose(self, measurements, now):
        if not measurements:
            self.switch_candidate_id     = None
            self.switch_candidate_frames = 0
            if self.locked_id is not None and now - self.last_seen > TARGET_LOST_TIMEOUT_S:
                self.locked_id = None
            return None

        # Accumulation phase
        if self.reference_profile is None:
            best = min(measurements, key=lambda m: m["score"])
            tid  = best["track_id"]

            if self._accum_id is None:
                self._accum_id    = tid
                self._accum_start = now
                if best["color_profile"] is not None:
                    self._accum_profiles.append(best["color_profile"].copy())
                self.locked_id  = tid
                self.last_seen  = now
                self.last_dist  = best["dist"]
                self.last_angle = best["angle"]
                return best["index"]

            elapsed       = now - self._accum_start
            timed_out     = elapsed >= ACCUM_MAX_DURATION_S
            accum_present = any(m["track_id"] == self._accum_id for m in measurements)

            if accum_present and not timed_out:
                m_acc = next(m for m in measurements if m["track_id"] == self._accum_id)
                if m_acc["color_profile"] is not None:
                    self._accum_profiles.append(m_acc["color_profile"].copy())
                self.last_seen  = now
                self.last_dist  = m_acc["dist"]
                self.last_angle = m_acc["angle"]
                return next(i for i, m in enumerate(measurements)
                            if m["track_id"] == self._accum_id)

            self._commit_reference()
            # Fall through to normal tracking

        tracked = [m for m in measurements if m["track_id"] is not None]

        if not tracked:
            best_color = max(measurements, key=lambda m: m["ref_similarity"] or 0.0)
            sim = best_color["ref_similarity"]
            if sim is not None and sim >= TARGET_COLOR_REID_MIN_SCORE:
                self._remember(best_color, now)
                return best_color["index"]
            if now - self.last_seen > TARGET_LOST_TIMEOUT_S:
                self.locked_id = None
            return None

        current = next((m for m in tracked if m["track_id"] == self.locked_id), None)

        if current is None:
            elapsed = now - self.last_seen
            if elapsed <= TARGET_LOST_TIMEOUT_S:
                reid = self._reidentify(tracked)
                if reid is not None:
                    self._lock(reid["track_id"], now, reid)
                    return reid["index"]
                return None
            best_match = max(tracked, key=lambda m: m["ref_similarity"] or 0.0)
            sim = best_match["ref_similarity"]
            if sim is not None and sim >= TARGET_COLOR_REID_MIN_SCORE:
                self._lock(best_match["track_id"], now, best_match)
                return best_match["index"]
            self.locked_id = None
            return None

        self._remember(current, now)

        for candidate in tracked:
            if candidate["track_id"] == self.locked_id:
                continue
            c_sim = candidate["ref_similarity"] or 0.0
            t_sim = current["ref_similarity"]   or 0.0
            if (c_sim - t_sim > 0.15 and
                    c_sim >= TARGET_COLOR_SWITCH_MIN_SCORE and
                    candidate["score"] + TARGET_SWITCH_MARGIN < current["score"]):
                if candidate["track_id"] == self.switch_candidate_id:
                    self.switch_candidate_frames += 1
                else:
                    self.switch_candidate_id     = candidate["track_id"]
                    self.switch_candidate_frames = 1
                if self.switch_candidate_frames >= TARGET_SWITCH_FRAMES:
                    self._lock(candidate["track_id"], now, candidate)
                    return candidate["index"]
                break
        else:
            self.switch_candidate_id     = None
            self.switch_candidate_frames = 0

        return current["index"]

    def _commit_reference(self):
        if self._accum_profiles:
            stacked = np.stack(self._accum_profiles, axis=0)
            avg     = stacked.mean(axis=0).astype(np.float32)
            self.reference_profile = avg
            self.color_profile     = avg.copy()
        self.locked_id               = self._accum_id
        self.switch_candidate_id     = None
        self.switch_candidate_frames = 0

    def _remember(self, measurement, now):
        self.last_seen  = now
        self.last_dist  = measurement["dist"]
        self.last_angle = measurement["angle"]
        self._update_profile(measurement)

    def _reidentify(self, measurements):
        if self.last_dist is None or self.last_angle is None:
            return None
        candidates = []
        for m in measurements:
            dist_delta  = abs(m["dist"]  - self.last_dist)
            angle_delta = abs(m["angle"] - self.last_angle)
            color_score = m.get("ref_similarity")
            color_ok = (
                self.reference_profile is None or
                color_score is None or
                color_score >= TARGET_COLOR_REID_MIN_SCORE
            )
            if (dist_delta  <= TARGET_REID_MAX_DIST_DELTA_M and
                    angle_delta <= TARGET_REID_MAX_ANGLE_DELTA_DEG and color_ok):
                color_bonus = color_score if color_score is not None else 0.0
                candidates.append((dist_delta + 0.1 * angle_delta - color_bonus, m))
        return min(candidates, key=lambda x: x[0])[1] if candidates else None

    def _lock(self, track_id, now, measurement=None):
        self.locked_id = track_id
        self.last_seen = now
        if measurement is not None:
            self.last_dist  = measurement["dist"]
            self.last_angle = measurement["angle"]
            self._update_profile(measurement)
        self.switch_candidate_id     = None
        self.switch_candidate_frames = 0

    def _target_score(self, measurement):
        score       = measurement["score"]
        color_score = measurement.get("ref_similarity")
        if self.reference_profile is not None and color_score is not None:
            score += TARGET_COLOR_SWITCH_PENALTY * (1.0 - color_score)
        return score

    def _update_profile(self, measurement):
        new_prof = measurement.get("color_profile")
        if new_prof is None:
            return
        if self.color_profile is None:
            self.color_profile = new_prof.copy()
            return
        self.color_profile = (
            TARGET_COLOR_EMA_ALPHA         * new_prof +
            (1.0 - TARGET_COLOR_EMA_ALPHA) * self.color_profile
        )


# =========================================================================== #
# Annotation
# =========================================================================== #

def annotate_frame(frame, detections: sv.Detections, smooth_state: dict,
                   target_lock: TargetLock, now: float):
    img_h, img_w = frame.shape[:2]
    out = frame.copy()

    # Convert once per frame — passed into clothing_color_profile
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    ids = detections.tracker_id if detections.tracker_id is not None else [None] * len(detections)

    measurements = []
    for i, ((x1, y1, x2, y2), conf, tid_raw) in enumerate(zip(
        detections.xyxy, detections.confidence, ids
    )):
        tid     = normalize_track_id(tid_raw)
        bbox_h  = y2 - y1
        bbox_cx = (x1 + x2) / 2
        raw_dist  = estimate_distance(bbox_h, bbox_cx, img_h, img_w) if bbox_h > 0 else 0
        raw_angle = estimate_angle(bbox_cx, img_w)

        if tid is not None:
            prev  = smooth_state.get(tid, {"dist": raw_dist, "angle": raw_angle})
            dist  = DIST_EMA_ALPHA  * raw_dist  + (1 - DIST_EMA_ALPHA)  * prev["dist"]
            angle = ANGLE_EMA_ALPHA * raw_angle + (1 - ANGLE_EMA_ALPHA) * prev["angle"]
            smooth_state[tid] = {"dist": dist, "angle": angle, "last_seen": now}
        else:
            dist  = raw_dist
            angle = raw_angle

        color_profile = clothing_color_profile(lab_frame, (x1, y1, x2, y2))

        raw_sim = profile_similarity(target_lock.reference_profile, color_profile) \
                  if target_lock.reference_profile is not None else None

        if tid is not None and raw_sim is not None:
            buf = smooth_state.setdefault(tid, {}).get("sim_buf", [])
            buf = (buf + [raw_sim])[-SIM_SMOOTH_WINDOW:]
            smooth_state[tid]["sim_buf"] = buf
            smoothed_sim = float(np.mean(buf))
        else:
            smoothed_sim = raw_sim

        measurements.append({
            "index":          i,
            "track_id":       tid,
            "xyxy":           (x1, y1, x2, y2),
            "confidence":     conf,
            "dist":           dist,
            "angle":          angle,
            "score":          detection_score(dist, angle),
            "color_profile":  color_profile,
            "ref_similarity": smoothed_sim,
        })

    for tid, state in list(smooth_state.items()):
        if now - state.get("last_seen", now) > TARGET_LOST_TIMEOUT_S * 4:
            smooth_state.pop(tid, None)

    target_idx = target_lock.choose(measurements, now)

    rows = []
    for m in measurements:
        i               = m["index"]
        x1, y1, x2, y2 = m["xyxy"]
        conf            = m["confidence"]
        tid             = m["track_id"]
        is_target       = (i == target_idx)
        color           = COLOR_TARGET if is_target else COLOR_OBSTACLE
        role            = "TARGET"     if is_target else "OBSTACLE"
        label_id        = f"ID{tid}"   if tid is not None else "?"
        dist            = m["dist"]
        angle           = m["angle"]
        profile         = m["color_profile"]
        ref_sim         = m.get("ref_similarity")

        thickness = 3 if is_target else 2
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        sim_str = f" sim:{ref_sim:.2f}" if ref_sim is not None else ""
        label   = f"{role} {label_id} {dist:.1f}m {angle:+.1f}deg{sim_str}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        top = max(int(y1) - 10, th + 4)
        cv2.rectangle(out, (int(x1), top - th - 4), (int(x1) + tw, top), color, -1)
        cv2.putText(out, label, (int(x1), top - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        # Swatch strip: reconstruct displayable BGR from a*, b* with fixed L=128
        if profile is not None:
            swatch_w = max(6, int((x2 - x1) * 0.06))
            swatch_h = max(2, int((y2 - y1) / N_PROFILE_CHUNKS))
            strip_x  = int(x1) + thickness + 1
            for ci in range(N_PROFILE_CHUNKS):
                sy1    = int(y1) + ci * swatch_h
                sy2    = sy1 + swatch_h
                a_val  = float(np.clip(profile[ci, 0], 0, 255))
                b_val  = float(np.clip(profile[ci, 1], 0, 255))
                lab_px = np.uint8([[[128, a_val, b_val]]])
                bgr_px = cv2.cvtColor(lab_px, cv2.COLOR_LAB2BGR)[0, 0].tolist()
                cv2.rectangle(out, (strip_x, sy1), (strip_x + swatch_w, sy2), bgr_px, -1)
            cv2.rectangle(out,
                          (strip_x, int(y1) + thickness + 1),
                          (strip_x + swatch_w,
                           int(y1) + thickness + 1 + N_PROFILE_CHUNKS * swatch_h),
                          (255, 255, 255), 1)

        rows.append((role, label_id, conf, dist, angle, ref_sim))

    return out, rows


# =========================================================================== #
# Main loop
# =========================================================================== #

def run(no_display=False):
    """Vision-only preview: run the IMX500 tracker and show the annotated frame
    plus a per-detection table. No motor control — follow.py is the driver."""
    if not RPK_MODEL_PATH.exists():
        raise SystemExit(
            f"ERROR: {RPK_MODEL_PATH} not found. "
            f"Install with: sudo apt install imx500-models"
        )

    cap = IMX500Capture(model_path=RPK_MODEL_PATH, width=640, height=480, fps=30)

    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=60,
        minimum_matching_threshold=0.8,
        frame_rate=30,
    )
    smooth_state = {}
    target_lock  = TargetLock()
    frame_idx    = 0
    frame_times  = []

    print("\nTracking — press Q to quit (vision only, no drive)\n")
    print(f"{'Role':<10} {'ID':<8} {'Conf':>6}  {'Dist':>8}  {'Angle':>8}  {'Sim':>6}")
    print("-" * 58)

    try:
        while True:
            t0      = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            dets    = cap.get_detections()
            tracked = tracker.update_with_detections(dets)
            out, rows = annotate_frame(frame, tracked, smooth_state,
                                       target_lock, time.monotonic())

            t1 = time.time()
            frame_times.append(t1)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps_live = (
                (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                if len(frame_times) > 1 else 0.0
            )
            cv2.putText(out,
                        f"NPU  FPS:{fps_live:.1f}  {(t1-t0)*1000:.0f}ms",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            for role, label_id, conf, dist, angle, ref_sim in rows:
                sim_str = f"{ref_sim:.2f}" if ref_sim is not None else " N/A"
                print(f"{role:<10} {label_id:<8} {conf:>6.0%}  "
                      f"{dist:>6.1f}m  {angle:>+7.1f}°  {sim_str}  [f{frame_idx}]")

            if not no_display:
                cv2.imshow("Cart View", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    finally:
        cap.release()
        if not no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IMX500 person tracker (vision-only preview; follow.py is the driver)"
    )
    parser.add_argument("--no-display", dest="no_display", action="store_true")
    args = parser.parse_args()

    if not args.no_display and not (os.environ.get("DISPLAY") or
                                     os.environ.get("WAYLAND_DISPLAY")):
        args.no_display = True
        print("No display detected — running headless.")

    run(no_display=args.no_display)
