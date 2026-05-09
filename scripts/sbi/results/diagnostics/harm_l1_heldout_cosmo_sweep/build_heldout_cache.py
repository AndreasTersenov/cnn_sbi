#!/usr/bin/env python3
"""B4: build a held-out L1 cache for calibration testing.

Loads the pre-computed L1 train/val caches from the harmonic seed-41 run,
moves N_HELDOUT cosmologies from train to the val set, and writes the
filtered caches to a new directory.  The held-out cosmologies are chosen
by random seed to be spread across the (Omega_m, sigma_8) plane, at least
MIN_DIST from fiducial in L2 norm.

Outputs
-------
heldout_cache/l1_train.npz   — original train minus held-out rows
heldout_cache/l1_val.npz     — original val plus held-out rows appended
heldout_meta.json            — held-out cosmology thetas and indices

Usage
-----
conda run -n jaxili python build_heldout_cache.py [--n-heldout 5] [--seed 77]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PARAM_NAMES = ["Omega_m", "sigma_8", "w0", "h0", "n_s", "Omega_b"]
FIDUCIAL_THETA = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493], dtype=np.float64)

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_CACHE = (
    REPO_ROOT
    / "scripts"
    / "sbi"
    / "results"
    / "exploratory"
    / "cross_maps_campaign"
    / "jaxili_harm_cross_nobnt"
    / "l1_cache_seed41"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "heldout_cache"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build held-out L1 cache for B4.")
    p.add_argument("--l1-cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--n-heldout", type=int, default=5)
    p.add_argument("--seed", type=int, default=77)
    p.add_argument(
        "--min-dist",
        type=float,
        default=0.01,
        help="Minimum L2 distance from fiducial in (Omega_m, sigma_8) to be held out.",
    )
    return p.parse_args()


def pick_heldout_indices(
    theta: np.ndarray,
    n_heldout: int,
    seed: int,
    min_dist: float,
) -> np.ndarray:
    """Return indices of n_heldout distinct cosmologies well-separated from fiducial."""
    om_sig8 = theta[:, :2].astype(np.float64)
    fid_om_sig8 = FIDUCIAL_THETA[:2]
    dist_from_fid = np.linalg.norm(om_sig8 - fid_om_sig8[None, :], axis=1)
    candidate_mask = dist_from_fid >= min_dist

    # Group rows by unique cosmology (same theta → same cosmology).
    # We pick whole cosmologies so that no partial leakage occurs.
    unique_thetas, inverse = np.unique(theta, axis=0, return_inverse=True)
    cosmo_indices = {i: np.where(inverse == i)[0] for i in range(len(unique_thetas))}

    # Only cosmologies where ALL their rows are outside min_dist from fiducial.
    valid_cosmo_ids = [
        ci for ci, rows in cosmo_indices.items()
        if np.all(candidate_mask[rows])
    ]
    if len(valid_cosmo_ids) < n_heldout:
        raise ValueError(
            f"Only {len(valid_cosmo_ids)} cosmologies at distance >= {min_dist} "
            f"from fiducial, need {n_heldout}."
        )

    rng = np.random.default_rng(seed)
    chosen_cosmo_ids = rng.choice(valid_cosmo_ids, size=n_heldout, replace=False)
    held_rows = np.concatenate([cosmo_indices[ci] for ci in chosen_cosmo_ids])
    return held_rows, chosen_cosmo_ids, unique_thetas


def main() -> None:
    args = parse_args()
    train_path = args.l1_cache_dir / "l1_train.npz"
    val_path = args.l1_cache_dir / "l1_val.npz"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing: {val_path}")

    print(f"Loading train cache from {train_path} ...")
    train_npz = np.load(train_path, mmap_mode="r")
    theta_train = np.asarray(train_npz["theta"], dtype=np.float64)
    x_train = np.asarray(train_npz["x"], dtype=np.float32)
    print(f"  train: theta={theta_train.shape}, x={x_train.shape}")

    print(f"Loading val cache from {val_path} ...")
    val_npz = np.load(val_path, mmap_mode="r")
    theta_val = np.asarray(val_npz["theta"], dtype=np.float64)
    x_val = np.asarray(val_npz["x"], dtype=np.float32)
    print(f"  val:   theta={theta_val.shape}, x={x_val.shape}")

    held_rows, held_cosmo_ids, unique_thetas = pick_heldout_indices(
        theta_train, args.n_heldout, args.seed, args.min_dist
    )
    held_rows_set = set(held_rows.tolist())
    keep_rows = np.array([i for i in range(len(theta_train)) if i not in held_rows_set])

    theta_train_filtered = theta_train[keep_rows]
    x_train_filtered = x_train[keep_rows]
    theta_heldout = theta_train[held_rows]
    x_heldout = x_train[held_rows]

    theta_val_augmented = np.concatenate([theta_val, theta_heldout], axis=0)
    x_val_augmented = np.concatenate([x_val, x_heldout], axis=0)

    print(f"\nHeld-out summary:")
    print(f"  {len(held_rows)} rows held out ({args.n_heldout} cosmologies)")
    print(f"  Held-out thetas (Omega_m, sigma_8):")
    for ci in held_cosmo_ids:
        th = unique_thetas[ci]
        print(f"    Omega_m={th[0]:.4f}  sigma_8={th[1]:.4f}  w0={th[2]:.3f}")
    print(f"  New train size: {len(theta_train_filtered)} (was {len(theta_train)})")
    print(f"  New val size:   {len(theta_val_augmented)} (was {len(theta_val)})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "l1_train.npz", theta=theta_train_filtered, x=x_train_filtered)
    np.savez_compressed(args.output_dir / "l1_val.npz", theta=theta_val_augmented, x=x_val_augmented)
    # Held-out rows as a standalone eval set for targeted SBC.
    # The SBC script reads l1_val.npz; we create a separate subdir so it can be
    # pointed there directly without mixing in the regular val set.
    eval_dir = args.output_dir / "heldout_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(eval_dir / "l1_val.npz", theta=theta_heldout, x=x_heldout)

    meta = {
        "source_l1_cache_dir": str(args.l1_cache_dir.resolve()),
        "n_heldout": args.n_heldout,
        "seed": args.seed,
        "min_dist_from_fiducial": args.min_dist,
        "n_heldout_rows": int(len(held_rows)),
        "train_size_original": int(len(theta_train)),
        "train_size_filtered": int(len(theta_train_filtered)),
        "val_size_original": int(len(theta_val)),
        "val_size_augmented": int(len(theta_val_augmented)),
        "heldout_cosmo_ids": [int(ci) for ci in held_cosmo_ids],
        "heldout_thetas": [
            {name: float(unique_thetas[ci][j]) for j, name in enumerate(PARAM_NAMES)}
            for ci in held_cosmo_ids
        ],
        "parameter_order": PARAM_NAMES,
        "fiducial_theta": FIDUCIAL_THETA.tolist(),
    }
    meta_path = args.output_dir / "heldout_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote filtered caches and metadata to {args.output_dir}")
    print(f"  l1_train.npz, l1_val.npz, heldout_meta.json")
    print(f"  heldout_eval/l1_val.npz  ({len(theta_heldout)} held-out rows for SBC eval)")


if __name__ == "__main__":
    main()
