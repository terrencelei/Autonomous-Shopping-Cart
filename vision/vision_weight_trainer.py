#!/usr/bin/env python3
"""
Standalone CSV -> outfit color weight vector trainer.

No Git repo needed. No camera code needed. Just give it a CSV.

Expected CSV columns:
  person_id, dist_m, frame, H0,S0,V0, H1,S1,V1, ..., H11,S11,V11

Only person_id and the H/S/V columns are required. dist_m and frame are ignored if present.

Install:
  pip install numpy pandas

Run:
  python3 standalone_weight_trainer.py vectors.csv

Outputs by default:
  weights_vector.csv
  weights_vector.json
  weights_by_hsv_chunk.csv

The core learned output is a vector of nonnegative weights that sums to 1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CHUNKS = 12


def expected_columns(chunks: int) -> list[str]:
    cols = []
    for i in range(chunks):
        cols += [f"H{i}", f"S{i}", f"V{i}"]
    return cols


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """
    Project vector v onto:
      w >= 0
      sum(w) = 1
    """
    v = np.asarray(v, dtype=float)
    if v.ndim != 1:
        raise ValueError("Expected 1D vector.")

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, len(v) + 1)
    cond = u - cssv / ind > 0

    if not np.any(cond):
        return np.ones_like(v) / len(v)

    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    return np.maximum(v - theta, 0.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)

    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))

    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)

    return out


def load_csv_features(csv_path: Path, chunks: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], dict]:
    """
    Load CSV and convert HSV profile into ML features.

    OpenCV Hue is circular:
      H=0 and H=179 are close colors.
    So each H value becomes:
      sin(2*pi*H/180), cos(2*pi*H/180)

    S and V are scaled to 0..1.

    Final features per body chunk:
      H_sin, H_cos, S, V
    """
    df = pd.read_csv(csv_path)

    if "person_id" not in df.columns:
        raise ValueError("CSV must include a person_id column.")

    needed = expected_columns(chunks)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            "CSV is missing required HSV columns. "
            f"First missing columns: {missing[:10]}"
        )

    df = df.copy()

    df["person_id"] = pd.to_numeric(df["person_id"], errors="coerce")
    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["person_id", *needed]).reset_index(drop=True)

    if len(df) < 4:
        raise ValueError("Need at least 4 usable labeled rows.")

    y = df["person_id"].astype(int).to_numpy()

    if len(set(y)) < 2:
        raise ValueError("Need at least 2 different person_id labels.")

    feature_blocks = []
    feature_names = []

    for i in range(chunks):
        h = df[f"H{i}"].to_numpy(dtype=float)
        s = df[f"S{i}"].to_numpy(dtype=float) / 255.0
        v = df[f"V{i}"].to_numpy(dtype=float) / 255.0

        angle = 2.0 * np.pi * h / 180.0
        h_sin = np.sin(angle)
        h_cos = np.cos(angle)

        feature_blocks.append(np.column_stack([h_sin, h_cos, s, v]))
        feature_names += [f"H{i}_sin", f"H{i}_cos", f"S{i}", f"V{i}"]

    X = np.concatenate(feature_blocks, axis=1)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-9] = 1.0
    X_scaled = (X - mean) / std

    normalization = {
        "mean": mean.tolist(),
        "std": std.tolist(),
    }

    return df, X_scaled, y, feature_names, normalization


def build_pairs(
    X: np.ndarray,
    y: np.ndarray,
    max_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build pairwise training examples.

    For two samples A and B:
      pair_feature = (A - B)^2

    Label:
      1 = same person
      0 = different person
    """
    rng = np.random.default_rng(seed)

    by_label = {}
    for idx, label in enumerate(y):
        by_label.setdefault(int(label), []).append(idx)

    same_pairs = []
    diff_pairs = []

    for indices in by_label.values():
        for a_pos in range(len(indices)):
            for b_pos in range(a_pos + 1, len(indices)):
                same_pairs.append((indices[a_pos], indices[b_pos]))

    labels = sorted(by_label)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            for a in by_label[labels[i]]:
                for b in by_label[labels[j]]:
                    diff_pairs.append((a, b))

    if not same_pairs:
        raise ValueError("Need at least one person_id with 2 or more samples.")
    if not diff_pairs:
        raise ValueError("Need samples from at least two different people.")

    per_class = min(len(same_pairs), len(diff_pairs), max_pairs // 2)

    same_choice = rng.choice(len(same_pairs), size=per_class, replace=False)
    diff_choice = rng.choice(len(diff_pairs), size=per_class, replace=False)

    pairs = [same_pairs[i] for i in same_choice] + [diff_pairs[i] for i in diff_choice]
    pair_labels = np.array([1] * per_class + [0] * per_class, dtype=float)

    order = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in order]
    pair_labels = pair_labels[order]

    D = np.empty((len(pairs), X.shape[1]), dtype=float)

    for row, (a, b) in enumerate(pairs):
        diff = X[a] - X[b]
        D[row] = diff * diff

    return D, pair_labels


def train_weights(
    D: np.ndarray,
    same_label: np.ndarray,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
) -> tuple[np.ndarray, float, list[float]]:
    """
    Logistic model:

      distance = D @ weights
      p_same = sigmoid(bias - distance)

    Same person should create small distance.
    Different people should create large distance.

    Constraint:
      weights >= 0
      sum(weights) = 1
    """
    rng = np.random.default_rng(seed)
    n_features = D.shape[1]

    weights = rng.random(n_features)
    weights = project_to_simplex(weights)

    bias = float(np.median(D @ weights))

    losses = []

    for _ in range(epochs):
        distance = D @ weights
        score = bias - distance
        p_same = sigmoid(score)

        eps = 1e-12
        loss = -np.mean(
            same_label * np.log(p_same + eps)
            + (1.0 - same_label) * np.log(1.0 - p_same + eps)
        )
        loss += 0.5 * l2 * float(np.sum(weights * weights))
        losses.append(float(loss))

        error = p_same - same_label

        grad_bias = float(np.mean(error))
        grad_weights = -(D.T @ error) / len(error) + l2 * weights

        bias -= lr * grad_bias
        weights = project_to_simplex(weights - lr * grad_weights)

    return weights, bias, losses


def accuracy_nearest_centroid(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    """
    Simple leave-one-out nearest weighted-centroid accuracy.
    """
    correct = 0
    labels = sorted(set(int(v) for v in y))

    for i in range(len(y)):
        best_label = None
        best_distance = float("inf")

        for label in labels:
            mask = y == label
            mask[i] = False

            if not np.any(mask):
                continue

            centroid = X[mask].mean(axis=0)
            distance = float(np.sum(weights * (X[i] - centroid) ** 2))

            if distance < best_distance:
                best_distance = distance
                best_label = label

        if best_label == int(y[i]):
            correct += 1

    return correct / len(y)


def save_outputs(
    out_dir: Path,
    csv_path: Path,
    feature_names: list[str],
    weights: np.ndarray,
    bias: float,
    losses: list[float],
    normalization: dict,
    y: np.ndarray,
    accuracy: float,
    chunks: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    vector_df = pd.DataFrame({
        "index": np.arange(len(weights)),
        "feature": feature_names,
        "weight": weights,
    })
    vector_df.to_csv(out_dir / "weights_vector.csv", index=False)

    readable_rows = []
    for i in range(chunks):
        h_weight = float(
            weights[feature_names.index(f"H{i}_sin")]
            + weights[feature_names.index(f"H{i}_cos")]
        )
        s_weight = float(weights[feature_names.index(f"S{i}")])
        v_weight = float(weights[feature_names.index(f"V{i}")])

        readable_rows += [
            {"chunk": i, "channel": "H", "weight": h_weight},
            {"chunk": i, "channel": "S", "weight": s_weight},
            {"chunk": i, "channel": "V", "weight": v_weight},
        ]

    readable_df = pd.DataFrame(readable_rows)
    readable_df.to_csv(out_dir / "weights_by_hsv_chunk.csv", index=False)

    label_counts = pd.Series(y).value_counts().sort_index()

    payload = {
        "input_csv": str(csv_path),
        "model": "standalone_constrained_logistic_weight_vector",
        "meaning": "Use these weights in a weighted squared-distance comparison between color profiles.",
        "constraint": "all weights are nonnegative and sum to 1",
        "n_samples": int(len(y)),
        "person_id_counts": {str(k): int(v) for k, v in label_counts.items()},
        "n_weights": int(len(weights)),
        "feature_names": feature_names,
        "weights_vector": weights.tolist(),
        "weights_by_feature": {
            name: float(weight)
            for name, weight in zip(feature_names, weights)
        },
        "bias": float(bias),
        "normalization": normalization,
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "nearest_centroid_leave_one_out_accuracy": float(accuracy),
    }

    with open(out_dir / "weights_vector.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone CSV-to-weights-vector trainer for outfit color profiles."
    )
    parser.add_argument("csv", type=Path, help="Input vectors CSV")
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="Output directory")
    parser.add_argument("--chunks", type=int, default=DEFAULT_CHUNKS, help="Number of vertical body chunks")
    parser.add_argument("--epochs", type=int, default=2500, help="Gradient descent steps")
    parser.add_argument("--lr", type=float, default=0.15, help="Learning rate")
    parser.add_argument("--l2", type=float, default=1e-3, help="L2 regularization strength")
    parser.add_argument("--max-pairs", type=int, default=20000, help="Max pairwise examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    df, X, y, feature_names, normalization = load_csv_features(args.csv, args.chunks)
    D, pair_labels = build_pairs(X, y, args.max_pairs, args.seed)

    weights, bias, losses = train_weights(
        D=D,
        same_label=pair_labels,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
    )

    acc = accuracy_nearest_centroid(X, y.copy(), weights)

    save_outputs(
        out_dir=args.out_dir,
        csv_path=args.csv,
        feature_names=feature_names,
        weights=weights,
        bias=bias,
        losses=losses,
        normalization=normalization,
        y=y,
        accuracy=acc,
        chunks=args.chunks,
    )

    print("Done.")
    print(f"Rows used: {len(df)}")
    print(f"Number of weights: {len(weights)}")
    print(f"Weights sum: {weights.sum():.6f}")
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Leave-one-out centroid accuracy: {acc:.3f}")
    print()
    print("Weights vector:")
    print(",".join(f"{w:.10f}" for w in weights))
    print()
    print(f"Wrote: {args.out_dir / 'weights_vector.csv'}")
    print(f"Wrote: {args.out_dir / 'weights_vector.json'}")
    print(f"Wrote: {args.out_dir / 'weights_by_hsv_chunk.csv'}")


if __name__ == "__main__":
    main()
