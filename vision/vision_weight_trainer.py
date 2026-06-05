#!/usr/bin/env python3
"""
Outfit color re-identification weight trainer.

Learns a weighted Mahalanobis-style distance metric from pairwise HSV
color profiles. The output weights answer: "which body chunks and color
channels matter most for telling two people apart?"

At inference time, a new detection is matched to a registry of recently-seen
profiles using weighted nearest-centroid distance — no fixed person labels
needed. Anyone who walks in can be tracked.

Expected CSV columns:
  person_id, dist_m, frame, H0,S0,V0, H1,S1,V1, ..., H11,S11,V11

person_id is used only to construct same/different pairs for training.
dist_m and frame are ignored.

Install:
  pip install numpy pandas

Run:
  python3 reid_weight_trainer.py vectors.csv

Outputs (written to --out-dir, default: current directory):
  weights_vector.json       — full model: weights, bias, normalization
  weights_by_hsv_chunk.csv  — human-readable per-chunk weights
  vectors_features.csv      — scaled feature matrix (for debugging)

Data collection advice
----------------------
More data helps, but diversity matters more than count. Collect samples:
  - At multiple distances (1m, 2m, 3m+)
  - Under different lighting conditions
  - On different days / outfit changes

You do NOT need to label specific people — any two recordings of the
same person wearing the same outfit count as a "same" pair. Different
people (or same person on a different day in different clothes) are
"different" pairs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_CHUNKS = 12

# How many gradient steps before we consider the model converged
DEFAULT_EPOCHS = 3000

# Simplex projection learning rate — 0.1–0.2 works well for most datasets
DEFAULT_LR = 0.15

# L2 penalty keeps weights from collapsing onto a single feature
DEFAULT_L2 = 1e-3

# Cap pairwise combinations — large datasets can explode quadratically
DEFAULT_MAX_PAIRS = 20_000


# --------------------------------------------------------------------------- #
# Simplex projection (enforces weights >= 0, sum = 1)
# --------------------------------------------------------------------------- #

def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """
    Euclidean projection onto the probability simplex:
      w_i >= 0  for all i
      sum(w) = 1

    O(n log n) algorithm via Duchi et al. 2008.
    """
    v = np.asarray(v, dtype=float)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, len(v) + 1, dtype=float)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.ones_like(v) / len(v)
    rho = int(ind[cond][-1])
    theta = cssv[cond][-1] / rho
    return np.maximum(v - theta, 0.0)


# --------------------------------------------------------------------------- #
# Numerically stable sigmoid
# --------------------------------------------------------------------------- #

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #

def load_and_featurize(
    csv_path: Path,
    chunks: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], dict]:
    """
    Load CSV and build a scaled feature matrix.

    Per chunk, four features are produced:
      H_sin, H_cos  — circular encoding of OpenCV hue (0–179)
      S             — saturation, scaled to [0, 1]
      V             — brightness, scaled to [0, 1]

    Circular hue encoding prevents the 0/179 wraparound discontinuity
    that breaks Euclidean distance ("red" would otherwise be far from
    "also red").

    Features are then z-score normalised (zero mean, unit variance) so
    that gradient descent converges at a consistent scale regardless of
    which channels dominate raw values.

    Returns
    -------
    df            : cleaned DataFrame
    X_scaled      : (n_samples, n_features) float64 feature matrix
    y             : (n_samples,) int array of person_id labels
    feature_names : list of feature name strings matching X columns
    normalization : dict with 'mean' and 'std' lists for inference-time use
    """
    df = pd.read_csv(csv_path)

    if "person_id" not in df.columns:
        raise ValueError("CSV must contain a 'person_id' column.")

    needed = [f"{ch}{i}" for i in range(chunks) for ch in ("H", "S", "V")]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV missing required columns. First missing: {missing[:6]}"
        )

    df = df.copy()
    df["person_id"] = pd.to_numeric(df["person_id"], errors="coerce")
    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["person_id", *needed]).reset_index(drop=True)

    if len(df) < 4:
        raise ValueError("Need at least 4 valid labelled rows.")
    if df["person_id"].nunique() < 2:
        raise ValueError("Need samples from at least 2 different person_ids.")

    y = df["person_id"].astype(int).to_numpy()

    blocks: list[np.ndarray] = []
    feature_names: list[str] = []

    for i in range(chunks):
        h = df[f"H{i}"].to_numpy(dtype=float)
        s = df[f"S{i}"].to_numpy(dtype=float) / 255.0
        v = df[f"V{i}"].to_numpy(dtype=float) / 255.0

        angle = 2.0 * np.pi * h / 180.0
        blocks.append(np.column_stack([np.sin(angle), np.cos(angle), s, v]))
        feature_names += [f"H{i}_sin", f"H{i}_cos", f"S{i}", f"V{i}"]

    X = np.concatenate(blocks, axis=1)          # (n, chunks*4)

    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    std[std < 1e-9] = 1.0                        # avoid divide-by-zero
    X_scaled = (X - mean) / std

    normalization = {"mean": mean.tolist(), "std": std.tolist()}

    return df, X_scaled, y, feature_names, normalization


# --------------------------------------------------------------------------- #
# Pairwise dataset construction
# --------------------------------------------------------------------------- #

def build_pairs(
    X: np.ndarray,
    y: np.ndarray,
    max_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert per-sample features into pairwise squared-difference features.

    For samples A and B:
      pair_feature[k] = (A[k] - B[k])^2

    Label:
      1 = same person (small distance desired)
      0 = different person (large distance desired)

    Same/different pairs are balanced so the model cannot win by predicting
    the majority class. If one class has more raw pairs it is randomly
    downsampled to match the other.

    Why squared differences?
    The logistic model scores: p_same = sigmoid(bias - w · d²)
    Large weighted distance → low p_same. This is a learned Mahalanobis
    distance where w encodes per-feature importance.
    """
    rng = np.random.default_rng(seed)

    by_label: dict[int, list[int]] = {}
    for idx, label in enumerate(y):
        by_label.setdefault(int(label), []).append(idx)

    same_pairs: list[tuple[int, int]] = []
    diff_pairs: list[tuple[int, int]] = []

    for indices in by_label.values():
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                same_pairs.append((indices[a], indices[b]))

    labels = sorted(by_label)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            for a in by_label[labels[i]]:
                for b in by_label[labels[j]]:
                    diff_pairs.append((a, b))

    if not same_pairs:
        raise ValueError(
            "No same-person pairs found. Each person_id needs at least 2 samples."
        )
    if not diff_pairs:
        raise ValueError(
            "No different-person pairs found. Need at least 2 distinct person_ids."
        )

    per_class = min(len(same_pairs), len(diff_pairs), max_pairs // 2)

    same_idx = rng.choice(len(same_pairs), size=per_class, replace=False)
    diff_idx = rng.choice(len(diff_pairs), size=per_class, replace=False)

    pairs   = [same_pairs[i] for i in same_idx] + [diff_pairs[i] for i in diff_idx]
    targets = np.array([1] * per_class + [0] * per_class, dtype=float)

    order  = rng.permutation(len(pairs))
    pairs  = [pairs[i] for i in order]
    targets = targets[order]

    D = np.empty((len(pairs), X.shape[1]), dtype=float)
    for row, (a, b) in enumerate(pairs):
        diff = X[a] - X[b]
        D[row] = diff * diff

    print(f"  Same-person pairs available:      {len(same_pairs):,}")
    print(f"  Different-person pairs available: {len(diff_pairs):,}")
    print(f"  Pairs used for training:          {len(pairs):,} "
          f"({per_class:,} same + {per_class:,} different)")

    return D, targets


# --------------------------------------------------------------------------- #
# Metric learning via constrained gradient descent
# --------------------------------------------------------------------------- #

def train_weights(
    D: np.ndarray,
    same_label: np.ndarray,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
    log_every: int = 500,
) -> tuple[np.ndarray, float, list[float]]:
    """
    Learn distance metric weights by minimising binary cross-entropy.

    Model:
      distance(A, B) = w · (A - B)²       (weighted squared distance)
      p_same(A, B)   = sigmoid(bias - distance)

    Constraints on w (enforced after every gradient step via simplex
    projection):
      w_i >= 0  for all i
      sum(w) = 1

    The sum-to-1 constraint keeps the scale of the distance consistent
    across different training runs and datasets, making the bias term
    and similarity threshold interpretable.

    L2 regularisation on w prevents weights from concentrating on a
    single feature (which would be equivalent to ignoring all others).
    """
    rng = np.random.default_rng(seed)
    n_feat = D.shape[1]

    # Initialise on the simplex
    w = project_to_simplex(rng.random(n_feat))

    # Start bias near the median distance so sigmoid is in a useful range
    bias = float(np.median(D @ w))

    losses: list[float] = []
    eps = 1e-12

    for epoch in range(epochs):
        dist  = D @ w
        score = bias - dist
        p     = sigmoid(score)

        loss = -np.mean(
            same_label * np.log(p + eps)
            + (1.0 - same_label) * np.log(1.0 - p + eps)
        ) + 0.5 * l2 * float(np.dot(w, w))
        losses.append(float(loss))

        err = p - same_label                      # (n_pairs,)

        grad_bias = float(err.mean())
        grad_w    = -(D.T @ err) / len(err) + l2 * w

        bias -= lr * grad_bias
        w     = project_to_simplex(w - lr * grad_w)

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:5d}/{epochs}  loss={loss:.5f}")

    return w, bias, losses


# --------------------------------------------------------------------------- #
# Evaluation: leave-one-out nearest weighted centroid
# --------------------------------------------------------------------------- #

def evaluate_loo(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Leave-one-out nearest weighted-centroid accuracy.

    For each sample i, compute its weighted squared distance to the
    centroid of every person_id class (excluding sample i from that
    class). Predict the closest centroid.

    This is the same decision rule used at inference time in the
    ReIDRegistry below, so this metric directly reflects deployment
    accuracy.
    """
    correct = 0
    labels  = sorted(set(int(v) for v in y))

    for i in range(len(y)):
        best_label    = None
        best_dist     = float("inf")

        for label in labels:
            mask    = (y == label)
            mask[i] = False                       # leave-one-out
            if not mask.any():
                continue

            centroid = X[mask].mean(axis=0)
            dist     = float(np.dot(weights, (X[i] - centroid) ** 2))

            if dist < best_dist:
                best_dist  = dist
                best_label = label

        if best_label == int(y[i]):
            correct += 1

    return correct / len(y)


# --------------------------------------------------------------------------- #
# Inference helper: online re-identification registry
# --------------------------------------------------------------------------- #

class ReIDRegistry:
    """
    Lightweight online re-identification tracker.

    Usage
    -----
    Load weights from the JSON output of this trainer, then call
    match_or_register() on each new detection's feature vector.

    Example
    -------
    import json, numpy as np
    from reid_weight_trainer import ReIDRegistry

    with open("weights_vector.json") as f:
        model = json.load(f)

    registry = ReIDRegistry(
        weights      = np.array(model["weights_vector"]),
        norm_mean    = np.array(model["normalization"]["mean"]),
        norm_std     = np.array(model["normalization"]["std"]),
        match_thresh = 1.5,   # tune this on your data
        max_age      = 90,    # frames before a track is dropped
    )

    # Each frame, for each HSV profile from the detector:
    track_id = registry.match_or_register(hsv_profile)

    What match_thresh means
    -----------------------
    The threshold is in units of weighted squared distance in normalised
    feature space. Lower = stricter matching (fewer false positive ID
    merges, more track splits). Tune by visualising distances between
    known-same and known-different pairs in your environment.
    """

    def __init__(
        self,
        weights: np.ndarray,
        norm_mean: np.ndarray,
        norm_std: np.ndarray,
        match_thresh: float = 1.5,
        max_age: int = 90,
        chunks: int = DEFAULT_CHUNKS,
    ) -> None:
        self.weights      = weights
        self.norm_mean    = norm_mean
        self.norm_std     = norm_std
        self.match_thresh = match_thresh
        self.max_age      = max_age
        self.chunks       = chunks

        self._next_id: int = 0
        # track_id -> {"centroid": np.ndarray, "age": int, "n": int}
        self._tracks: dict[int, dict] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def match_or_register(
        self,
        hsv_profile: np.ndarray,
    ) -> int:
        """
        Given an (N_PROFILE_CHUNKS, 3) HSV profile from the detector,
        return an integer track ID. If the profile is close enough to an
        existing track it is assigned that track's ID; otherwise a new
        track is created.

        Call tick() once per frame to age out lost tracks.
        """
        feat = self._featurize(hsv_profile)

        best_id   = None
        best_dist = float("inf")

        for tid, track in self._tracks.items():
            d = float(np.dot(self.weights, (feat - track["centroid"]) ** 2))
            if d < best_dist:
                best_dist = d
                best_id   = tid

        if best_id is not None and best_dist <= self.match_thresh:
            # Update running centroid with exponential moving average
            n = self._tracks[best_id]["n"] + 1
            alpha = 1.0 / n
            self._tracks[best_id]["centroid"] = (
                (1 - alpha) * self._tracks[best_id]["centroid"] + alpha * feat
            )
            self._tracks[best_id]["age"] = 0
            self._tracks[best_id]["n"]   = n
            return best_id

        # New track
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = {"centroid": feat.copy(), "age": 0, "n": 1}
        return tid

    def tick(self) -> None:
        """
        Age all tracks by one frame and remove stale ones.
        Call once per frame after processing all detections.
        """
        stale = []
        for tid in self._tracks:
            self._tracks[tid]["age"] += 1
            if self._tracks[tid]["age"] > self.max_age:
                stale.append(tid)
        for tid in stale:
            del self._tracks[tid]

    @property
    def active_tracks(self) -> list[int]:
        return list(self._tracks.keys())

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _featurize(self, hsv_profile: np.ndarray) -> np.ndarray:
        """
        Convert raw (N_CHUNKS, 3) HSV profile to a normalised feature vector.
        Mirrors load_and_featurize() so inference matches training exactly.
        """
        parts: list[np.ndarray] = []
        for i in range(self.chunks):
            h = float(hsv_profile[i, 0])
            s = float(hsv_profile[i, 1]) / 255.0
            v = float(hsv_profile[i, 2]) / 255.0
            angle = 2.0 * np.pi * h / 180.0
            parts.append(np.array([np.sin(angle), np.cos(angle), s, v]))

        raw = np.concatenate(parts)
        return (raw - self.norm_mean) / self.norm_std


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #

def save_outputs(
    out_dir: Path,
    csv_path: Path,
    feature_names: list[str],
    weights: np.ndarray,
    bias: float,
    losses: list[float],
    normalization: dict,
    X_scaled: np.ndarray,
    y: np.ndarray,
    accuracy: float,
    chunks: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-chunk human-readable weights
    # H weight = sum of sin and cos components
    rows = []
    for i in range(chunks):
        h_w = float(
            weights[feature_names.index(f"H{i}_sin")]
            + weights[feature_names.index(f"H{i}_cos")]
        )
        rows += [
            {"chunk": i, "channel": "H", "weight": h_w},
            {"chunk": i, "channel": "S", "weight": float(weights[feature_names.index(f"S{i}")])},
            {"chunk": i, "channel": "V", "weight": float(weights[feature_names.index(f"V{i}")])},
        ]
    pd.DataFrame(rows).to_csv(out_dir / "weights_by_hsv_chunk.csv", index=False)

    # Scaled feature matrix (useful for debugging and distance inspection)
    feat_df = pd.DataFrame(X_scaled, columns=feature_names)
    feat_df.insert(0, "person_id", y)
    feat_df.to_csv(out_dir / "vectors_features.csv", index=False)

    # Full JSON model (everything needed for inference)
    label_counts = pd.Series(y).value_counts().sort_index()
    payload = {
        "model": "reid_metric_learning_constrained_logistic",
        "input_csv": str(csv_path),
        "meaning": (
            "Weighted squared distance metric for person re-identification. "
            "Use weights in: distance = sum(w_i * (A_i - B_i)^2) on normalised features. "
            "Small distance = likely same person."
        ),
        "constraint": "all weights >= 0, sum(weights) = 1",
        "n_samples": int(len(y)),
        "n_features": int(len(weights)),
        "person_id_counts": {str(k): int(v) for k, v in label_counts.items()},
        "feature_names": feature_names,
        "weights_vector": weights.tolist(),
        "bias": float(bias),
        "normalization": normalization,
        "training": {
            "initial_loss": float(losses[0]),
            "final_loss": float(losses[-1]),
        },
        "evaluation": {
            "method": "leave_one_out_nearest_weighted_centroid",
            "accuracy": float(accuracy),
            "note": (
                "LOO centroid accuracy mirrors the ReIDRegistry decision rule. "
                "With small datasets this is optimistic — collect more data across "
                "varied lighting and distances for a reliable estimate."
            ),
        },
        "inference": {
            "class": "ReIDRegistry",
            "module": "reid_weight_trainer",
            "match_thresh_default": 1.5,
            "note": (
                "Tune match_thresh on held-out pairs. Lower = stricter matching "
                "(fewer false merges, more track splits)."
            ),
        },
    }

    with open(out_dir / "weights_vector.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Learn a re-identification distance metric from outfit HSV profiles."
    )
    parser.add_argument("csv",        type=Path,  help="Input vectors CSV")
    parser.add_argument("--out-dir",  type=Path,  default=Path("."))
    parser.add_argument("--chunks",   type=int,   default=DEFAULT_CHUNKS)
    parser.add_argument("--epochs",   type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--lr",       type=float, default=DEFAULT_LR)
    parser.add_argument("--l2",       type=float, default=DEFAULT_L2)
    parser.add_argument("--max-pairs",type=int,   default=DEFAULT_MAX_PAIRS)
    parser.add_argument("--seed",     type=int,   default=42)
    args, unknown = parser.parse_known_args() # Modified line

    if unknown: # Optional: print a warning for unknown args
        print(f"Warning: Unrecognized arguments ignored: {unknown}")

    # Manually set the CSV path to the available vectors.csv file
    args.csv = Path('/content/vectors.csv')

    print(f"\nLoading: {args.csv}")
    df, X, y, feature_names, normalization = load_and_featurize(args.csv, args.chunks)
    print(f"  Samples: {len(df)}  |  People: {df['person_id'].nunique()}  |  Features: {X.shape[1]}")

    print("\nBuilding pairs...")
    D, pair_labels = build_pairs(X, y, args.max_pairs, args.seed)

    print(f"\nTraining ({args.epochs} epochs, lr={args.lr}, l2={args.l2})...")
    weights, bias, losses = train_weights(
        D=D, same_label=pair_labels,
        epochs=args.epochs, lr=args.lr, l2=args.l2, seed=args.seed,
    )

    print("\nEvaluating (leave-one-out nearest centroid)...")
    acc = evaluate_loo(X, y.copy(), weights)

    save_outputs(
        out_dir=args.out_dir, csv_path=args.csv,
        feature_names=feature_names, weights=weights, bias=bias,
        losses=losses, normalization=normalization,
        X_scaled=X, y=y, accuracy=acc, chunks=args.chunks,
    )

    print(f"\n{'─'*50}")
    print(f"  Rows used:              {len(df)}")
    print(f"  Weights sum:            {weights.sum():.6f}")
    print(f"  Initial loss:           {losses[0]:.5f}")
    print(f"  Final loss:             {losses[-1]:.5f}")
    print(f"  LOO centroid accuracy:  {acc:.1%}")
    print(f"{'─'*50}")
    print(f"\nWeights vector ({len(weights)} values):")
    print(",".join(f"{w:.8f}" for w in weights))
    print(f"\nWrote: {args.out_dir / 'weights_vector.json'}")
    print(f"Wrote: {args.out_dir / 'weights_by_hsv_chunk.csv'}")
    print(f"Wrote: {args.out_dir / 'vectors_features.csv'}")


if __name__ == "__main__":
    main()
