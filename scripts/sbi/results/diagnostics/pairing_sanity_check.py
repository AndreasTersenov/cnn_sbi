#!/usr/bin/env python3
"""Check whether (summary, theta) pairing carries real signal vs shuffled control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np


PARAM_NAMES = ["Omega_m", "sigma_8", "w0", "h0", "n_s", "Omega_b"]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_cache = repo_root / "scripts" / "sbi" / "systematic_runs_l1_snr10_rerun" / "cache" / "l1_tomo4_20deg160"
    default_out = repo_root / "scripts" / "sbi" / "diagnostics" / "pairing_sanity_l1_tomo4_20deg160.json"
    p = argparse.ArgumentParser(
        description="Pairing sanity check: paired data vs shuffled-theta baseline."
    )
    p.add_argument("--cache-dir", type=Path, default=default_cache)
    p.add_argument("--train-file", type=str, default="l1_train.npz")
    p.add_argument("--val-file", type=str, default="l1_val.npz")
    p.add_argument("--alpha", type=float, default=10.0, help="Ridge regularization.")
    p.add_argument("--pca-components", type=int, default=50, help="0 disables PCA.")
    p.add_argument("--max-train", type=int, default=40000)
    p.add_argument("--max-val", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-shuffles", type=int, default=8)
    p.add_argument("--output-json", type=Path, default=default_out)
    return p.parse_args()


def sample_rows(x: np.ndarray, y: np.ndarray, n_rows: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) <= n_rows:
        return x, y
    idx = rng.choice(len(x), size=n_rows, replace=False)
    return x[idx], y[idx]


def standardize_train_val(x_train: np.ndarray, x_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (x_train - mean) / std, (x_val - mean) / std


def fit_pca(x_train: np.ndarray, x_val: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    if n_components <= 0:
        return x_train, x_val
    # x_train is already zero-mean approximately after standardization; this keeps it simple.
    _, _, vt = np.linalg.svd(x_train, full_matrices=False)
    k = min(n_components, vt.shape[0])
    w = vt[:k].T
    return x_train @ w, x_val @ w


def ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    xtx = x_train.T @ x_train
    xty = x_train.T @ y_train
    w = np.linalg.solve(xtx + alpha * np.eye(xtx.shape[0]), xty)
    return x_val @ w, w


def r2_per_param(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def cross_singular_values(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x0 = (x - x.mean(axis=0, keepdims=True)) / np.maximum(x.std(axis=0, keepdims=True), 1e-12)
    y0 = (y - y.mean(axis=0, keepdims=True)) / np.maximum(y.std(axis=0, keepdims=True), 1e-12)
    c = (x0.T @ y0) / max(len(x0) - 1, 1)
    return np.linalg.svd(c, compute_uv=False)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    train_npz = np.load(args.cache_dir / args.train_file)
    val_npz = np.load(args.cache_dir / args.val_file)

    x_train = train_npz["x"].astype(np.float64)
    y_train = train_npz["theta"].astype(np.float64)
    x_val = val_npz["x"].astype(np.float64)
    y_val = val_npz["theta"].astype(np.float64)

    x_train, y_train = sample_rows(x_train, y_train, args.max_train, rng)
    x_val, y_val = sample_rows(x_val, y_val, args.max_val, rng)

    # L1 summaries are non-negative and heavy-tailed.
    if np.min(x_train) >= 0.0:
        x_train = np.log1p(x_train)
        x_val = np.log1p(x_val)

    x_train, x_val = standardize_train_val(x_train, x_val)
    x_train, x_val = fit_pca(x_train, x_val, args.pca_components)

    y_pred_pair, _ = ridge_predict(x_train, y_train, x_val, args.alpha)
    r2_pair = r2_per_param(y_val, y_pred_pair)

    shuffle_r2 = []
    for _ in range(args.n_shuffles):
        y_shuf = y_train.copy()
        rng.shuffle(y_shuf, axis=0)
        y_pred_shuf, _ = ridge_predict(x_train, y_shuf, x_val, args.alpha)
        shuffle_r2.append(r2_per_param(y_val, y_pred_shuf))
    shuffle_r2 = np.stack(shuffle_r2, axis=0)

    # Cross-cov spectrum proxy (paired vs one shuffled draw).
    sv_pair = cross_singular_values(x_train, y_train)
    y_once = y_train.copy()
    rng.shuffle(y_once, axis=0)
    sv_shuf = cross_singular_values(x_train, y_once)

    k = min(6, len(sv_pair), len(sv_shuf))
    out = {
        "cache_dir": str(args.cache_dir.resolve()),
        "train_file": args.train_file,
        "val_file": args.val_file,
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "summary_dim_after_pca": int(x_train.shape[1]),
        "theta_dim": int(y_train.shape[1]),
        "alpha": float(args.alpha),
        "pca_components": int(args.pca_components),
        "n_shuffles": int(args.n_shuffles),
        "r2_paired_per_param": r2_pair.tolist(),
        "r2_shuffled_mean_per_param": shuffle_r2.mean(axis=0).tolist(),
        "r2_shuffled_std_per_param": shuffle_r2.std(axis=0).tolist(),
        "r2_delta_vs_shuffled_mean_per_param": (r2_pair - shuffle_r2.mean(axis=0)).tolist(),
        "r2_paired_first3_mean": float(np.mean(r2_pair[:3])),
        "r2_paired_last3_mean": float(np.mean(r2_pair[3:])),
        "r2_shuffled_first3_mean": float(np.mean(shuffle_r2.mean(axis=0)[:3])),
        "r2_shuffled_last3_mean": float(np.mean(shuffle_r2.mean(axis=0)[3:])),
        "cross_singular_values_paired_top6": sv_pair[:k].tolist(),
        "cross_singular_values_shuffled_top6": sv_shuf[:k].tolist(),
        "cross_singular_value_ratio_top6": (sv_pair[:k] / np.maximum(sv_shuf[:k], 1e-12)).tolist(),
        "param_names": PARAM_NAMES[: y_train.shape[1]],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Saved pairing sanity diagnostics: {args.output_json}")
    print("Paired R2:", np.array2string(r2_pair, precision=3, suppress_small=True))
    print(
        "Shuffled R2 mean:",
        np.array2string(shuffle_r2.mean(axis=0), precision=3, suppress_small=True),
    )
    print(
        "Top singular-value ratio paired/shuffled:",
        np.array2string(sv_pair[:k] / np.maximum(sv_shuf[:k], 1e-12), precision=3),
    )


if __name__ == "__main__":
    main()
