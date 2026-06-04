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

PERSON_CONFIDENCE = 0.5
PERSON_CLASS_ID   = 0   # COCO "person" in the labels shipped with imx500-models

PERSON_HEIGHT_M   = 1.8
H_FOV_DEG         = 66.0   # Pi AI Camera horizontal field of view
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

TARGET_LOST_TIMEOUT_S        = 0.75
TARGET_SWITCH_MARGIN         = 2.0
TARGET_SWITCH_FRAMES         = 12
TARGET_REID_MAX_DIST_DELTA_M = 1.0
TARGET_REID_MAX_ANGLE_DELTA_DEG = 12.0

TARGET_COLOR_EMA_ALPHA        = 0.25
TARGET_COLOR_REID_MIN_SCORE   = 0.70
TARGET_COLOR_SWITCH_MIN_SCORE = 0.70

# Number of frames over which ref_similarity is averaged per track ID
SIM_SMOOTH_WINDOW = 10
TARGET_COLOR_SWITCH_PENALTY   = 1.5

# --------------------------------------------------------------------------- #
# Color profile tuning
# --------------------------------------------------------------------------- #
N_PROFILE_CHUNKS = 12

BOX_TRIM_X  = 0.12
BOX_TRIM_Y  = 0.08
SURROUND_PX = 12
MIN_ROW_PX  = 4

BG_HUE_DELTA = 18   # degrees (0-179 OpenCV scale)
BG_SAT_DELTA = 30   # 0-255


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
        req   = self._picam2.capture_request()
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
# Color profile
# =========================================================================== #

def clothing_color_profile(hsv_frame, xyxy):
    """
    Build a 1-D vertical color profile for the person in *xyxy*.

    Pipeline
    --------
    1. Tighten the bounding box inward by BOX_TRIM_X / BOX_TRIM_Y.
    2. Sample the surrounding band (SURROUND_PX wide) to get the local
       background average H, S, V.
    3. For each pixel row inside the tightened box:
         - reject pixels whose H and S are both within BG_HUE_DELTA /
           BG_SAT_DELTA of the surrounding band average (background bleed).
         - average the remaining HSV values.
         - rows with fewer than MIN_ROW_PX valid pixels are skipped (NaN).
    4. Compress the per-row averages into N_PROFILE_CHUNKS chunks.
    5. Divide each chunk's V by the surrounding band's average V to normalise
       for local lighting.

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

    # Tighten box
    tx = int(box_w * BOX_TRIM_X)
    ty = int(box_h * BOX_TRIM_Y)
    ix1, ix2 = x1 + tx, x2 - tx
    iy1, iy2 = y1 + ty, y2 - ty
    if ix2 <= ix1 or iy2 <= iy1:
        return None

    # Sample surrounding band
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
    surround_avg = surround_all.mean(axis=0)
    surround_v   = max(surround_avg[2], 1.0)

    # Per-row averages with background rejection
    inner  = hsv_frame[iy1:iy2, ix1:ix2].astype(np.float32)
    n_rows = inner.shape[0]

    row_avgs = np.full((n_rows, 3), np.nan, dtype=np.float32)
    for r in range(n_rows):
        row   = inner[r]
        h_diff = np.abs(row[:, 0] - surround_avg[0])
        h_diff = np.minimum(h_diff, 180.0 - h_diff)
        s_diff = np.abs(row[:, 1] - surround_avg[1])
        fg_mask = ~((h_diff < BG_HUE_DELTA) & (s_diff < BG_SAT_DELTA))
        if fg_mask.sum() >= MIN_ROW_PX:
            row_avgs[r] = row[fg_mask].mean(axis=0)

    # Compress into N_PROFILE_CHUNKS
    chunk_size = n_rows / N_PROFILE_CHUNKS
    profile    = np.full((N_PROFILE_CHUNKS, 3), np.nan, dtype=np.float32)
    for c in range(N_PROFILE_CHUNKS):
        r_start    = int(round(c * chunk_size))
        r_end      = int(round((c + 1) * chunk_size))
        chunk_rows = row_avgs[r_start:r_end]
        valid      = chunk_rows[~np.isnan(chunk_rows[:, 0])]
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

    # V-channel lighting normalisation
    profile[:, 2] = np.clip(profile[:, 2] / surround_v, 0.0, 4.0)

    return profile


def profile_similarity(a, b):
    """
    Weighted similarity between two (N_PROFILE_CHUNKS, 3) profiles.

    Chunk mapping for N_PROFILE_CHUNKS=12:
      0-1   head          0.05  (skin/hair varies little between people)
      2     neck/shoulder 0.30
      3-5   upper torso   3.50  (shirt/hoodie — most distinctive)
      6     waist         2.00
      7-9   legs          3.00  (trousers/jeans — highly distinctive)
      10    lower legs    0.80
      11    feet          0.02  (often cut off)

    Error scaling:
      - H normalised over 45deg half-range (not 90) so different colours
        hit max error more readily.
      - Power of 1.5 amplifies large differences while being lenient on
        small lighting-induced shifts.
      - H weighted 70%, S weighted 30%.

    Returns a float in [0, 1], or None if either profile is None.
    """
    if a is None or b is None:
        return None

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

    diff_h = np.abs(a[:, 0] - b[:, 0])
    diff_h = np.minimum(diff_h, 180.0 - diff_h)
    diff_s = np.abs(a[:, 1] - b[:, 1])

    err_h = np.clip(diff_h / 45.0, 0.0, 1.0) ** 1.5
    err_s = np.clip(diff_s / 80.0, 0.0, 1.0) ** 1.5

    err = err_h * 0.70 + err_s * 0.30

    weighted_err = np.dot(err, weights) / weights.sum()
    return float(np.clip(1.0 - weighted_err, 0.0, 1.0))


# =========================================================================== #
# Target lock
# =========================================================================== #

class TargetLock:
    """
    Locks onto the first person detected and uses their saved color profile
    as a fixed reference to distinguish them from all subsequent detections.

    reference_profile  — set once on first detection, never modified.
    color_profile      — EMA-smoothed live profile, used for swatch display only.
    """

    def __init__(self):
        self.locked_id               = None
        self.last_seen               = 0.0
        self.last_dist               = None
        self.last_angle              = None
        self.reference_profile       = None   # set once accumulation is complete
        self.color_profile           = None   # EMA for display only
        self.switch_candidate_id     = None
        self.switch_candidate_frames = 0

        # Accumulation state: collect profiles over the first unbroken bbox run
        # before committing a reference.  Once reference_profile is set, these
        # are no longer used.
        self._accum_id       = None    # track ID being accumulated
        self._accum_profiles = []      # list of (N_PROFILE_CHUNKS, 3) arrays

    def choose(self, measurements, now):
        if not measurements:
            self.switch_candidate_id     = None
            self.switch_candidate_frames = 0
            if self.locked_id is not None and now - self.last_seen > TARGET_LOST_TIMEOUT_S:
                print(f"[LOCK] Target lost — no detections for >{TARGET_LOST_TIMEOUT_S}s")
                self.locked_id = None
            return None

        # ------------------------------------------------------------------ #
        # Accumulation phase: before reference_profile is committed, collect
        # color profiles over every frame the first person's bbox is unbroken.
        # The reference is finalised (averaged) as soon as that bbox breaks.
        # ------------------------------------------------------------------ #
        if self.reference_profile is None:
            best = min(measurements, key=lambda m: m["score"])
            tid  = best["track_id"]

            if self._accum_id is None:
                # Very first detection — start accumulating
                self._accum_id = tid
                if best["color_profile"] is not None:
                    self._accum_profiles.append(best["color_profile"].copy())
                self.locked_id  = tid
                self.last_seen  = now
                self.last_dist  = best["dist"]
                self.last_angle = best["angle"]
                print(f"[LOCK] First detection ID{tid} — accumulating profile "
                      f"(frame 1)  dist={best['dist']:.1f}m")
                return best["index"]

            # Check whether the accumulated person is still in frame
            accum_present = any(m["track_id"] == self._accum_id for m in measurements)

            if accum_present:
                m_accum = next(m for m in measurements if m["track_id"] == self._accum_id)
                if m_accum["color_profile"] is not None:
                    self._accum_profiles.append(m_accum["color_profile"].copy())
                self.last_seen  = now
                self.last_dist  = m_accum["dist"]
                self.last_angle = m_accum["angle"]
                print(f"[LOCK] Accumulating ID{self._accum_id} — "
                      f"{len(self._accum_profiles)} frames so far  "
                      f"dist={m_accum['dist']:.1f}m")
                return next(i for i, m in enumerate(measurements)
                            if m["track_id"] == self._accum_id)
            else:
                # Bbox broke — commit the averaged reference now
                self._commit_reference()
                print(f"[LOCK] *** REFERENCE COMMITTED from {len(self._accum_profiles)} "
                      f"frames for ID{self._accum_id} ***  "
                      f"profile={'OK' if self.reference_profile is not None else 'NONE'}")
                # Fall through to normal tracking logic below

        # ------------------------------------------------------------------ #
        # ref_similarity is pre-computed (10-frame smoothed) by annotate_frame.
        # Just log it here.
        # ------------------------------------------------------------------ #
        for m in measurements:
            sim_str = f"{m['ref_similarity']:.2f}" if m["ref_similarity"] is not None else "N/A"
            print(f"[SCORE] ID{m['track_id']}  dist={m['dist']:.1f}m  "
                  f"angle={m['angle']:+.1f}deg  ref_sim={sim_str}")

        tracked = [m for m in measurements if m["track_id"] is not None]

        # No tracker IDs yet — fall back to color-only match
        if not tracked:
            best_color = max(measurements, key=lambda m: m["ref_similarity"] or 0.0)
            sim = best_color["ref_similarity"]
            sim_str = f"{sim:.2f}" if sim is not None else "N/A"
            print(f"[LOCK] No tracked IDs — best color match sim={sim_str} "
                  f"(need >={TARGET_COLOR_REID_MIN_SCORE})")
            if sim is not None and sim >= TARGET_COLOR_REID_MIN_SCORE:
                self._remember(best_color, now)
                return best_color["index"]
            if now - self.last_seen > TARGET_LOST_TIMEOUT_S:
                self.locked_id = None
            return None

        current = next((m for m in tracked if m["track_id"] == self.locked_id), None)

        # Locked ID not visible — attempt re-identification
        if current is None:
            elapsed = now - self.last_seen
            if elapsed <= TARGET_LOST_TIMEOUT_S:
                print(f"[LOCK] ID{self.locked_id} not in frame — re-ID attempt "
                      f"({elapsed:.2f}s since last seen)")
                reid = self._reidentify(tracked)
                if reid is not None:
                    sim_str = f"{reid.get('ref_similarity', '?'):.2f}" \
                              if reid.get("ref_similarity") is not None else "N/A"
                    print(f"[LOCK] Re-identified as ID{reid['track_id']}  sim={sim_str}")
                    self._lock(reid["track_id"], now, reid)
                    return reid["index"]
                print("[LOCK] Re-ID failed — no candidate passed thresholds")
                return None

            # Timeout expired — try best color match
            best_match = max(tracked, key=lambda m: m["ref_similarity"] or 0.0)
            sim = best_match["ref_similarity"]
            sim_str = f"{sim:.2f}" if sim is not None else "N/A"
            print(f"[LOCK] Timeout expired — best color match ID{best_match['track_id']} "
                  f"sim={sim_str} (need >={TARGET_COLOR_REID_MIN_SCORE})")
            if sim is not None and sim >= TARGET_COLOR_REID_MIN_SCORE:
                self._lock(best_match["track_id"], now, best_match)
                print(f"[LOCK] Re-locked on ID{best_match['track_id']} after timeout")
                return best_match["index"]
            self.locked_id = None
            print("[LOCK] Lock dropped — no match above threshold")
            return None

        # Happy path: current target still visible
        self._remember(current, now)
        sim = current.get("ref_similarity")
        sim_str = f"{sim:.2f}" if sim is not None else "N/A"
        print(f"[LOCK] Tracking ID{self.locked_id}  dist={current['dist']:.1f}m  "
              f"angle={current['angle']:+.1f}deg  ref_sim={sim_str}")

        # Consider switching only if another person is a clearly better color
        # match AND significantly closer/more central
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
                print(f"[LOCK] Switch candidate ID{candidate['track_id']}  "
                      f"color_improvement={color_improvement:.2f}  "
                      f"frames={self.switch_candidate_frames}/{TARGET_SWITCH_FRAMES}")

                if self.switch_candidate_frames >= TARGET_SWITCH_FRAMES:
                    print(f"[LOCK] *** SWITCHING to ID{candidate['track_id']} ***")
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

    def _commit_reference(self):
        """Average all accumulated profiles into the fixed reference."""
        if self._accum_profiles:
            stacked = np.stack(self._accum_profiles, axis=0)  # (N, chunks, 3)
            avg     = stacked.mean(axis=0).astype(np.float32)
            self.reference_profile = avg
            self.color_profile     = avg.copy()
        # Lock onto the accumulated ID
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
            # Use the pre-computed smoothed ref_similarity (already in the measurement)
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

    def _update_profile(self, measurement):
        """EMA update of the live color profile. reference_profile is never touched."""
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
        color_score = measurement.get("ref_similarity")
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

        color_profile = clothing_color_profile(hsv_frame, (x1, y1, x2, y2))

        # Compute raw ref_similarity then smooth over last SIM_SMOOTH_WINDOW frames
        # so momentary profile glitches don't flip the lock decision.
        raw_sim = profile_similarity(
            target_lock.reference_profile, color_profile
        ) if target_lock.reference_profile is not None else None

        if tid is not None and raw_sim is not None:
            buf = smooth_state.get(tid, {}).get("sim_buf", [])
            buf = (buf + [raw_sim])[-SIM_SMOOTH_WINDOW:]
            smooth_state.setdefault(tid, {})["sim_buf"] = buf
            smoothed_sim = float(np.mean(buf))
        else:
            smoothed_sim = raw_sim   # None or untracked — pass through as-is

        measurements.append({
            "index":          i,
            "track_id":       tid,
            "xyxy":           (x1, y1, x2, y2),
            "confidence":     conf,
            "dist":           dist,
            "angle":          angle,
            "score":          detection_score(dist, angle),
            "color_profile":  color_profile,
            "ref_similarity": smoothed_sim,   # pre-computed smoothed value
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
        color           = COLOR_TARGET   if is_target else COLOR_OBSTACLE
        role            = "TARGET"       if is_target else "OBSTACLE"
        label_id        = f"ID{tid}"     if tid is not None else "?"

        dist    = m["dist"]
        angle   = m["angle"]
        profile = m["color_profile"]
        ref_sim = m.get("ref_similarity")

        thickness = 3 if is_target else 2
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        sim_str = f" sim:{ref_sim:.2f}" if ref_sim is not None else ""
        label = f"{role} {label_id} {dist:.1f}m {angle:+.1f}deg{sim_str}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        top = max(int(y1) - 10, th + 4)
        cv2.rectangle(out, (int(x1), top - th - 4), (int(x1) + tw, top), color, -1)
        cv2.putText(out, label, (int(x1), top - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        # Color profile swatches
        if profile is not None:
            swatch_w = max(6, int((x2 - x1) * 0.06))
            swatch_h = max(2, int((y2 - y1) / N_PROFILE_CHUNKS))
            strip_x  = int(x1) + thickness + 1
            for ci in range(N_PROFILE_CHUNKS):
                sy1    = int(y1) + ci * swatch_h
                sy2    = sy1 + swatch_h
                h_val  = float(profile[ci, 0])
                s_val  = float(profile[ci, 1])
                v_val  = float(np.clip(profile[ci, 2] * 128, 0, 255))
                hsv_px = np.uint8([[[h_val, s_val, v_val]]])
                bgr_px = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0, 0].tolist()
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

def print_header():
    print(f"\n{'Role':<10} {'ID':<8} {'Conf':>6}  {'Distance':>10}  {'Angle':>8}  {'Sim':>6}")
    print("-" * 62)


def run(no_display=False, no_drive=False):
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
            t0  = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            dets    = cap.get_detections()
            tracked = tracker.update_with_detections(dets)
            out, rows = annotate_frame(frame, tracked, smooth_state, target_lock,
                                       time.monotonic())

            target_row = next((r for r in rows if r[0] == "TARGET"), None)
            if target_row is not None:
                _role, _id, _conf, t_dist, t_angle, _sim = target_row
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

            t1         = time.time()
            latency_ms = (t1 - t0) * 1000
            frame_times.append(t1)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps_live = (
                (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                if len(frame_times) > 1 else 0.0
            )
            cv2.putText(out,
                        f"NPU  FPS: {fps_live:.1f}  {latency_ms:.0f}ms  [{P.S.mode}]",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            for role, label_id, conf, dist, angle, ref_sim in rows:
                sim_str = f"{ref_sim:.2f}" if ref_sim is not None else " N/A"
                print(f"{role:<10} {label_id:<8} {conf:>6.0%}  "
                      f"{dist:>8.1f}m  {angle:>+7.1f}°  sim:{sim_str}  [f{frame_idx}]")

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

    if not args.no_display and not (os.environ.get("DISPLAY") or
                                     os.environ.get("WAYLAND_DISPLAY")):
        args.no_display = True
        print("No display detected — running headless "
              "(use --no-display to silence this message).")

    run(no_display=args.no_display, no_drive=args.no_drive)
