"""
Detect and track people with SSD-MobileNetV2 on the IMX500 NPU + ByteTrack,
then drive the cart directly via Pathfinding_algorithm.tick().

Usage:
  python3 yolo_detect.py                # live camera + motors
  python3 yolo_detect.py --no-display   # headless (SSH)
  python3 yolo_detect.py --no-drive     # camera only, no motors

Inference runs entirely on the IMX500's on-chip neural processor; no CPU
inference path. Requires the imx500-models apt package.

Color signature pipeline (clothing_color_profile):
  1. Convert full frame BGR -> HSV once per frame
  2. Tighten bounding box inward to reduce background bleed
  3. For each row in the tightened box, average HSV of non-background pixels
     (background defined as pixels too close in hue/sat to the surrounding band)
  4. Compress the per-row profile down to N_PROFILE_CHUNKS evenly-spaced chunks
  5. Sample a band of SURROUND_PX pixels outside each side of the box,
     average their V channel, and divide the profile's V channel by that value
     to normalise for local lighting
  6. Return the (N_PROFILE_CHUNKS x 3) float32 array [H, S, V_normalised]
"""

import math
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
DISTANCE_OFFSET_M = 0
DISTANCE_SCALE    = 0.7
ANGLE_SCALE       = 0.99
ANGLE_OFFSET_DEG  = 2.0

COLOR_TARGET   = (0, 255, 0)
COLOR_OBSTACLE = (0, 0, 255)

MAP_SIZE    = 200
MAP_RANGE_M = 10.0

TARGET_DIST_WEIGHT  = 1.0
TARGET_ANGLE_WEIGHT = 0.3

DIST_EMA_ALPHA = 0.4
ANGLE_EMA_ALPHA = 0.4
TARGET_LOST_TIMEOUT_S = 0.75
TARGET_SWITCH_MARGIN = 2.0
TARGET_SWITCH_FRAMES = 12
TARGET_REID_MAX_DIST_DELTA_M = 1.0
TARGET_REID_MAX_ANGLE_DELTA_DEG = 12.0
TARGET_COLOR_EMA_ALPHA = 0.25
TARGET_COLOR_REID_MIN_SCORE = 0.45
TARGET_COLOR_SWITCH_MIN_SCORE = 0.35
TARGET_COLOR_SWITCH_PENALTY = 1.5

# --------------------------------------------------------------------------- #
# Color profile tuning
# --------------------------------------------------------------------------- #
# Number of vertical chunks in the 1-D color profile
N_PROFILE_CHUNKS = 12

# Fraction to trim from each edge of the bounding box before profiling.
# Reduces background bleed; 0.12 horizontal, 0.08 vertical is a good start.
BOX_TRIM_X = 0.12
BOX_TRIM_Y = 0.08

# Width of the surrounding band (pixels each side) used for V normalisation.
# A wider band is more stable but may sample other objects.
SURROUND_PX = 12

# Minimum number of foreground pixels in a row for the row to be included.
# Rows below this threshold (e.g. at small distances) are skipped.
MIN_ROW_PX = 4

# Background-rejection thresholds: a pixel in the box is called "background"
# and excluded if its H and S are both within these deltas of the surrounding
# band average.  Set large to disable (effectively no per-pixel rejection).
BG_HUE_DELTA = 18   # degrees (0-179 OpenCV scale)
BG_SAT_DELTA = 30   # 0-255

# World-map window
WORLD_MAP_PX = 500


# =========================================================================== #
# Camera capture
# =========================================================================== #

class IMX500Capture:
    """Pi AI Camera with detection running entirely on the IMX500 NPU."""

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
# Color profile  (replaces clothing_color_hist)
# =========================================================================== #

def clothing_color_profile(hsv_frame, xyxy):
    """
    Build a 1-D vertical color profile for the person in *xyxy*.

    Pipeline
    --------
    1. Convert full frame to HSV *before* calling this function (done once in
       annotate_frame and passed in as hsv_frame).
    2. Tighten the bounding box inward by BOX_TRIM_X / BOX_TRIM_Y.
    3. Sample the surrounding band (SURROUND_PX wide) to get the local
       background average H, S, V.
    4. For each pixel row inside the tightened box:
         - reject pixels whose H and S are both within BG_HUE_DELTA /
           BG_SAT_DELTA of the surrounding band average (background bleed).
         - average the remaining HSV values.
         - rows with fewer than MIN_ROW_PX valid pixels are skipped (NaN).
    5. Compress the per-row averages into N_PROFILE_CHUNKS chunks by averaging
       contiguous groups of rows.
    6. Divide each chunk's V by the surrounding band's average V to normalise
       for local lighting.  H and S are left unchanged (already
       lighting-independent in HSV).

    Returns
    -------
    np.ndarray of shape (N_PROFILE_CHUNKS, 3) dtype float32, channels [H, S, V],
    or None if the box is too small to be useful.
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

    # --- Step 2: tighten box ---
    tx = int(box_w * BOX_TRIM_X)
    ty = int(box_h * BOX_TRIM_Y)
    ix1, ix2 = x1 + tx, x2 - tx
    iy1, iy2 = y1 + ty, y2 - ty
    if ix2 <= ix1 or iy2 <= iy1:
        return None

    # --- Step 3: sample surrounding band for background reference ---
    # Collect pixels from a SURROUND_PX-wide strip just outside each side,
    # clamped to image boundaries.  We only sample the rows that overlap the
    # tightened box vertically so the reference is spatially consistent.
    surround_pixels = []

    left_x1  = max(0, ix1 - SURROUND_PX)
    left_x2  = ix1
    right_x1 = ix2
    right_x2 = min(img_w, ix2 + SURROUND_PX)
    top_y1   = max(0, iy1 - SURROUND_PX)
    top_y2   = iy1
    bot_y1   = iy2
    bot_y2   = min(img_h, iy2 + SURROUND_PX)

    for sx1, sx2, sy1, sy2 in [
        (left_x1,  left_x2,  iy1, iy2),
        (right_x1, right_x2, iy1, iy2),
        (ix1,      ix2,      top_y1, top_y2),
        (ix1,      ix2,      bot_y1, bot_y2),
    ]:
        if sx2 > sx1 and sy2 > sy1:
            patch = hsv_frame[sy1:sy2, sx1:sx2].reshape(-1, 3)
            surround_pixels.append(patch)

    if not surround_pixels:
        return None

    surround_all = np.concatenate(surround_pixels, axis=0).astype(np.float32)
    surround_avg = surround_all.mean(axis=0)   # [H_bg, S_bg, V_bg]
    surround_v   = max(surround_avg[2], 1.0)   # avoid divide-by-zero

    # --- Step 4: per-row averages with background pixel rejection ---
    inner = hsv_frame[iy1:iy2, ix1:ix2].astype(np.float32)  # (rows, cols, 3)
    n_rows = inner.shape[0]

    row_avgs = np.full((n_rows, 3), np.nan, dtype=np.float32)
    for r in range(n_rows):
        row = inner[r]                           # (cols, 3)
        h_diff = np.abs(row[:, 0] - surround_avg[0])
        # Hue wraps at 180 in OpenCV; handle the wrap
        h_diff = np.minimum(h_diff, 180.0 - h_diff)
        s_diff = np.abs(row[:, 1] - surround_avg[1])
        fg_mask = ~((h_diff < BG_HUE_DELTA) & (s_diff < BG_SAT_DELTA))

        if fg_mask.sum() >= MIN_ROW_PX:
            row_avgs[r] = row[fg_mask].mean(axis=0)

    # --- Step 5: compress into N_PROFILE_CHUNKS chunks ---
    chunk_size = n_rows / N_PROFILE_CHUNKS
    profile = np.full((N_PROFILE_CHUNKS, 3), np.nan, dtype=np.float32)
    for c in range(N_PROFILE_CHUNKS):
        r_start = int(round(c * chunk_size))
        r_end   = int(round((c + 1) * chunk_size))
        chunk_rows = row_avgs[r_start:r_end]
        valid = chunk_rows[~np.isnan(chunk_rows[:, 0])]
        if len(valid) > 0:
            profile[c] = valid.mean(axis=0)

    # If every chunk is NaN the box is useless
    if np.all(np.isnan(profile[:, 0])):
        return None

    # Fill isolated NaN chunks by interpolating from neighbours so that the
    # profile vector has no holes (makes distance calculation simpler).
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

    # --- Step 6: V-channel lighting normalisation ---
    profile[:, 2] = np.clip(profile[:, 2] / surround_v, 0.0, 4.0)

    return profile


def profile_similarity(a, b):
    """
    Weighted similarity between two (N_PROFILE_CHUNKS, 3) profiles.

    Weights are calibrated from measured bounding boxes at 1.5–3.9 m.
    Landmark fractions (mean ± std across all distances):
      chin        0.142 ± 0.014
      shoulder    0.222 ± 0.023
      chest       0.319 ± 0.027
      waist       0.512 ± 0.019
      hip         0.614 ± 0.017
      knee        0.817 ± 0.012
      ankle       0.953 ± 0.008

    The detector box proportions are highly stable across this range, so a
    fixed weight vector (no distance adjustment) is appropriate.

    Chunk mapping for N_PROFILE_CHUNKS=12 (each chunk = 1/12 ≈ 8.3% of box):
      0  [0.000-0.083]  head          0.15
      1  [0.083-0.167]  head          0.15
      2  [0.167-0.250]  neck/shoulder 0.40
      3  [0.250-0.333]  upper torso   2.00  ← hoodie chest
      4  [0.333-0.417]  mid torso     2.00  ← hoodie body
      5  [0.417-0.500]  mid torso     2.00
      6  [0.500-0.583]  waist/hip     1.60
      7  [0.583-0.667]  upper legs    1.80  ← jeans
      8  [0.667-0.750]  upper legs    1.80
      9  [0.750-0.833]  upper legs    1.80
     10  [0.833-0.917]  lower legs    1.00
     11  [0.917-1.000]  feet          0.05

    Only H and S channels are compared; V is normalised out.
    Returns a float in [0, 1], or None if either profile is None.
    """
    if a is None or b is None:
        return None

    # Empirically calibrated from 5 screenshots at 1.5, 2.0, 2.4, 2.9, 3.9 m
    weights = np.array(
        [0.15, 0.15, 0.40, 2.00, 2.00, 2.00, 1.60, 1.80, 1.80, 1.80, 1.00, 0.05],
        dtype=np.float32,
    )
    # Pad or trim if N_PROFILE_CHUNKS was changed from 12
    if len(weights) != N_PROFILE_CHUNKS:
        weights = np.interp(
            np.linspace(0, 1, N_PROFILE_CHUNKS),
            np.linspace(0, 1, 12),
            weights,
        ).astype(np.float32)

    # Compare only H and S (columns 0 and 1)
    diff_h = np.abs(a[:, 0] - b[:, 0])
    diff_h = np.minimum(diff_h, 180.0 - diff_h)   # hue wrap
    diff_s = np.abs(a[:, 1] - b[:, 1])

    # Normalise to [0, 1] range: H max=90 (half-circle), S max=255
    err = (diff_h / 90.0 + diff_s / 255.0) / 2.0   # per-chunk error in [0,1]
    weighted_err = np.dot(err, weights) / weights.sum()
    return float(np.clip(1.0 - weighted_err, 0.0, 1.0))


# =========================================================================== #
# Target lock
# =========================================================================== #

class TargetLock:
    def __init__(self):
        self.locked_id               = None
        self.last_seen               = 0.0
        self.last_dist               = None
        self.last_angle              = None
        self.reference_profile       = None   # FIXED on first detection, never updated
        self.color_profile           = None   # EMA-smoothed, used for display/debug
        self.switch_candidate_id     = None
        self.switch_candidate_frames = 0

    def choose(self, measurements, now):
        if not measurements:
            self.switch_candidate_id     = None
            self.switch_candidate_frames = 0
            if self.locked_id is not None and now - self.last_seen > TARGET_LOST_TIMEOUT_S:
                self.locked_id = None
            return None

        # ------------------------------------------------------------------ #
        # First ever detection: lock onto whoever we see and save their
        # color profile as the permanent reference.  Never overwrite it.
        # ------------------------------------------------------------------ #
        if self.reference_profile is None:
            # Pick the closest person (lowest score = closest + most centred)
            best = min(measurements, key=lambda m: m["score"])
            self._lock_first(best, now)
            return best["index"]

        # ------------------------------------------------------------------ #
        # Score every detection against the fixed reference profile
        # ------------------------------------------------------------------ #
        for m in measurements:
            m["ref_similarity"] = profile_similarity(self.reference_profile,
                                                      m["color_profile"])

        # The TARGET candidate is whichever tracked person best matches the
        # reference.  Untracked detections (tid=None) are ignored for locking
        # but still returned as OBSTACLE.
        tracked = [m for m in measurements if m["track_id"] is not None]

        if not tracked:
            # No tracker IDs yet — fall back to closest match by color
            best_color = max(measurements,
                             key=lambda m: m["ref_similarity"] or 0.0)
            if (best_color["ref_similarity"] is not None and
                    best_color["ref_similarity"] >= TARGET_COLOR_REID_MIN_SCORE):
                self._remember(best_color, now)
                return best_color["index"]
            if now - self.last_seen > TARGET_LOST_TIMEOUT_S:
                self.locked_id = None
            return None

        # Among tracked detections, find whoever currently holds the locked ID
        current = next((m for m in tracked if m["track_id"] == self.locked_id), None)

        # Re-ID: if we lost the locked ID, search by color + geometry
        if current is None:
            if now - self.last_seen <= TARGET_LOST_TIMEOUT_S:
                reid = self._reidentify(tracked)
                if reid is not None:
                    self._lock(reid["track_id"], now, reid)
                    return reid["index"]
                return None
            # Timeout expired — pick the best color match above threshold
            best_match = max(tracked, key=lambda m: m["ref_similarity"] or 0.0)
            if (best_match["ref_similarity"] is not None and
                    best_match["ref_similarity"] >= TARGET_COLOR_REID_MIN_SCORE):
                self._lock(best_match["track_id"], now, best_match)
                return best_match["index"]
            self.locked_id = None
            return None

        # Happy path: current target still visible
        self._remember(current, now)

        # Consider switching only if another tracked person is a *better*
        # color match AND significantly closer/more central
        for candidate in tracked:
            if candidate["track_id"] == self.locked_id:
                continue
            c_sim = candidate["ref_similarity"] or 0.0
            t_sim = current["ref_similarity"]   or 0.0
            color_improvement = c_sim - t_sim

            if (color_improvement > 0.15 and
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

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _lock_first(self, measurement, now):
        """Called exactly once — saves the permanent reference profile."""
        self.locked_id         = measurement["track_id"]
        self.last_seen         = now
        self.last_dist         = measurement["dist"]
        self.last_angle        = measurement["angle"]
        # Store as both the fixed reference AND the live EMA profile
        if measurement["color_profile"] is not None:
            self.reference_profile = measurement["color_profile"].copy()
            self.color_profile     = measurement["color_profile"].copy()
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
            # Use reference_profile (not the EMA) for re-ID — more stable
            color_score = profile_similarity(self.reference_profile,
                                             m["color_profile"])
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

    def _update_profile(self, measurement):
        """EMA update of the live color profile — reference_profile is never touched."""
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

    def _target_score(self, measurement):
        score       = measurement["score"]
        color_score = profile_similarity(self.reference_profile,
                                         measurement["color_profile"])
        if self.reference_profile is not None and color_score is not None:
            score += TARGET_COLOR_SWITCH_PENALTY * (1.0 - color_score)
        return score


# =========================================================================== #
# Annotation
# =========================================================================== #

def annotate_frame(frame, detections: sv.Detections, smooth_state: dict,
                   target_lock: TargetLock, now: float):
    img_h, img_w = frame.shape[:2]
    out = frame.copy()

    # Convert the whole frame to HSV once; pass the slice into color profiling.
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    ids = detections.tracker_id if detections.tracker_id is not None else [None] * len(detections)

    measurements = []
    for i, ((x1, y1, x2, y2), conf, tid_raw) in enumerate(zip(
        detections.xyxy, detections.confidence, ids
    )):
        tid      = normalize_track_id(tid_raw)
        bbox_h   = y2 - y1
        bbox_cx  = (x1 + x2) / 2
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

        measurements.append({
            "index":         i,
            "track_id":      tid,
            "xyxy":          (x1, y1, x2, y2),
            "confidence":    conf,
            "dist":          dist,
            "angle":         angle,
            "score":         detection_score(dist, angle),
            "color_profile": clothing_color_profile(hsv_frame, (x1, y1, x2, y2)),
        })

    for tid, state in list(smooth_state.items()):
        if now - state.get("last_seen", now) > TARGET_LOST_TIMEOUT_S * 4:
            smooth_state.pop(tid, None)

    target_idx = target_lock.choose(measurements, now)

    rows = []
    for m in measurements:
        i        = m["index"]
        x1, y1, x2, y2 = m["xyxy"]
        conf     = m["confidence"]
        tid      = m["track_id"]
        is_target = (i == target_idx)
        color     = COLOR_TARGET if is_target else COLOR_OBSTACLE
        role      = "TARGET"    if is_target else "OBSTACLE"
        label_id  = f"ID{tid}"  if tid is not None else "?"

        dist    = m["dist"]
        angle   = m["angle"]
        profile = m["color_profile"]

        thickness = 3 if is_target else 2
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        label = f"{role} {label_id} {dist:.1f}m {angle:+.1f}deg"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        top = max(int(y1) - 10, th + 4)
        cv2.rectangle(out, (int(x1), top - th - 4), (int(x1) + tw, top), color, -1)
        cv2.putText(out, label, (int(x1), top - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        # Draw the color profile as a vertical strip of swatches just inside
        # the left edge of the bounding box.  Each swatch is one chunk; its
        # BGR is converted back from the stored HSV (V re-scaled to a legible
        # fixed brightness so swatches are readable regardless of scene lighting).
        if profile is not None:
            swatch_w = max(6, int((x2 - x1) * 0.06))
            swatch_h = max(2, int((y2 - y1) / N_PROFILE_CHUNKS))
            strip_x  = int(x1) + thickness + 1
            for ci in range(N_PROFILE_CHUNKS):
                sy1 = int(y1) + ci * swatch_h
                sy2 = sy1 + swatch_h
                h_val = float(profile[ci, 0])
                s_val = float(profile[ci, 1])
                v_val = float(np.clip(profile[ci, 2] * 128, 0, 255))
                hsv_px  = np.uint8([[[h_val, s_val, v_val]]])
                bgr_px  = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0, 0].tolist()
                cv2.rectangle(out, (strip_x, sy1), (strip_x + swatch_w, sy2),
                              bgr_px, -1)
            cv2.rectangle(out,
                          (strip_x, int(y1) + thickness + 1),
                          (strip_x + swatch_w,
                           int(y1) + thickness + 1 + N_PROFILE_CHUNKS * swatch_h),
                          (255, 255, 255), 1)

        rows.append((role, label_id, conf, dist, angle))

    return out, rows


# =========================================================================== #
# Main loop
# =========================================================================== #

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
    target_lock  = TargetLock()

    motors   = None if no_drive else P.MotorDriver()
    odometry = P.Odometry(motors.read_encoder_deltas if motors else lambda: (0, 0))

    frame_idx   = 0
    frame_times = []
    t_last_tick = time.monotonic()
    latest_target = None

    quit_msg  = "" if no_display else " — press Q to quit"
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
            out, rows = annotate_frame(frame, tracked, smooth_state, target_lock, time.monotonic())

            target_row = next((r for r in rows if r[0] == "TARGET"), None)
            if target_row is not None:
                _role, _id, _conf, t_dist, t_angle = target_row
                latest_target = (t_dist, t_angle)
            else:
                latest_target = None

            obstacle_rows = [r for r in rows if r[0] == "OBSTACLE"]

            now = time.monotonic()
            if now - t_last_tick >= P.DT:
                pos, heading = odometry.update()
                obs = [(r[3], r[4]) for r in obstacle_rows]
                v_left, v_right = P.tick(latest_target, pos, heading, obstacles=obs)
                if motors is not None:
                    motors.send(v_left, v_right)
                t_last_tick = now

            t1 = time.time()
            latency_ms = (t1 - t0) * 1000
            frame_times.append(t1)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps_live = (
                (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                if len(frame_times) > 1 else 0.0
            )
            cv2.putText(out, f"NPU  FPS: {fps_live:.1f}  {latency_ms:.0f}ms  [{P.S.mode}]",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            for role, label_id, conf, dist, angle in rows:
                print(f"{role:<10} {label_id:<8} {conf:>6.0%}  {dist:>8.1f}m  {angle:>+7.1f}°  [f{frame_idx}]")

            if not no_display:
                cv2.imshow("Cart View", out)
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
    parser = argparse.ArgumentParser(
        description="SSD person tracker on IMX500 NPU + ByteTrack → Pathfinding_algorithm"
    )
    parser.add_argument("--no-display", dest="no_display", action="store_true",
                        help="suppress cv2 windows (headless / SSH use)")
    parser.add_argument("--no-drive", dest="no_drive", action="store_true",
                        help="run camera and detection but do not command motors")
    args = parser.parse_args()

    if not args.no_display and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        args.no_display = True
        print("No display detected — running headless (use --no-display to silence this message).")

    run(no_display=args.no_display, no_drive=args.no_drive)
