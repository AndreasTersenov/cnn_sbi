#!/usr/bin/env python3
"""GATE A1 — CNN flat-local cross construction correctness (no campaign).

Pulls a real batch of RAW auto patches (ch 0..nbins-1) from the cross TFDS and
checks the *in-pipeline* on-device JAX transform used by --cnn-map-route flat_local
against the numpy oracle, for every --cross-op. Mirrors the L1 GATE A1.

Checks per op:
  1. JAX build == numpy build (FFT float32 roundoff for conv; exact for product/autos).
  2. channel count == n_output_channels(nbins, op)  (4 / 10 / 10 / 16).
  3. ch 0..nbins-1 of the built tensor are the UNTOUCHED raw autos.
  4. whitening: transform(scale) == transform(None) / channel_scale  (exact).
  5. per-channel RMS finite + sane ratio; built channels finite (no NaN/Inf).
  6. determinism: transform(autos) twice -> bit-identical (the obs<->train identity:
     obs and train share this one callable, so equality here proves they match).

Run (GPU 1 only):
  CUDA_VISIBLE_DEVICES=1 conda run -n jaxili python scripts/sbi/gate_a_flat_cross_cnn.py
"""
from __future__ import annotations

import os

# Pin GPU + XLA BEFORE importing jax (via npe_cnn_nbody_tomo).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

from flatsky_cross import build_channels_np, n_output_channels
from tfds_cross_tfdata_loader import iter_cross_tfds_batches
from npe_cnn_nbody_tomo import (
    compute_flat_cross_channel_rms,
    make_flat_cross_transform,
)

TFDS = "nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180"
DDIR = "/home/tersenov/tensorflow_datasets"
NBINS = 4
ROLL = 0.10
OPS = ("none", "conv", "product", "both")


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    """max|a-b| / (max|b| + eps) — scale-aware relative discrepancy."""
    denom = float(np.max(np.abs(b))) + 1e-30
    return float(np.max(np.abs(a - b))) / denom


def main() -> int:
    print(f"=== GATE A1: flat-local CNN cross construction (GPU {os.environ['CUDA_VISIBLE_DEVICES']}) ===")
    # One batch of RAW autos (ch 0..3, NO scaling) from the train split.
    autos = None
    for maps_np, _theta in iter_cross_tfds_batches(
        tfds_name=TFDS, data_dir=DDIR, split="train", batch_size=64, seed=0,
        flip=False, channel_scale=None, channel_slice=slice(0, NBINS),
        perm_lo=None, perm_hi=None,
    ):
        if np.isnan(maps_np).any():
            continue
        autos = np.asarray(maps_np, dtype=np.float32)
        break
    assert autos is not None, "No clean auto batch pulled from the TFDS."
    print(f"  autos batch: shape={autos.shape}, dtype={autos.dtype}, "
          f"abs-max={np.abs(autos).max():.3e}")

    all_pass = True
    # Tolerances: autos/product are FFT-free (near-exact); conv goes through an
    # rfft2/irfft2 round trip (numpy computes in float64, jax in float32).
    REL_TOL = {"none": 1e-6, "product": 1e-5, "conv": 2e-3, "both": 2e-3}

    for op in OPS:
        print(f"\n--- op = {op} ---")
        n_exp = n_output_channels(NBINS, op)

        built_np = build_channels_np(autos, op, ROLL).astype(np.float32)
        transform_noscale = make_flat_cross_transform(op, None, ROLL)
        built_jax = np.asarray(transform_noscale(autos), dtype=np.float32)

        # 1. JAX == numpy
        rel = _rel(built_jax, built_np)
        ok1 = rel < REL_TOL[op]
        print(f"  [1] jax-vs-numpy rel-diff = {rel:.2e}  (tol {REL_TOL[op]:.0e})  "
              f"{'OK' if ok1 else 'FAIL'}")

        # 2. channel count
        ok2 = built_jax.shape[-1] == n_exp
        print(f"  [2] channels = {built_jax.shape[-1]} (expected {n_exp})  "
              f"{'OK' if ok2 else 'FAIL'}")

        # 3. raw autos preserved as ch 0..nbins-1
        rel_auto = _rel(built_jax[..., :NBINS], autos)
        ok3 = rel_auto < 1e-6
        print(f"  [3] autos preserved (ch 0..{NBINS-1}) rel-diff = {rel_auto:.2e}  "
              f"{'OK' if ok3 else 'FAIL'}")

        # 4. whitening exactness: transform(scale) == transform(None) / scale
        scale = compute_flat_cross_channel_rms(
            tfds_name=TFDS, data_dir=DDIR, op=op, nbins=NBINS,
            split="train", n_sample=2000, roll_frac=ROLL,
        )
        transform_scaled = make_flat_cross_transform(op, scale, ROLL)
        built_white = np.asarray(transform_scaled(autos), dtype=np.float32)
        expect_white = built_jax / scale.astype(np.float32)
        rel_w = _rel(built_white, expect_white)
        ok4 = (scale.shape[0] == n_exp) and (rel_w < 1e-5)
        print(f"  [4] whiten len={scale.shape[0]} rel-diff(scaled vs ref/scale) = "
              f"{rel_w:.2e}  {'OK' if ok4 else 'FAIL'}")

        # 5. RMS sane + built finite
        finite = bool(np.isfinite(built_white).all()) and bool(np.isfinite(scale).all())
        ratio = float(scale.max() / scale.min())
        ok5 = finite and (scale.min() > 0)
        print(f"  [5] RMS per-channel: min={scale.min():.3e} max={scale.max():.3e} "
              f"ratio={ratio:.1f}x  finite={finite}  {'OK' if ok5 else 'FAIL'}")
        print(f"      whitened built: mean={built_white.mean():.3f} "
              f"std={built_white.std():.3f} (per-channel std should be ~O(1))")
        per_ch_std = built_white.std(axis=(0, 1, 2))
        print(f"      per-channel std: {np.array2string(per_ch_std, precision=2)}")

        # 6. determinism (obs<->train identity proxy)
        built_again = np.asarray(transform_scaled(autos), dtype=np.float32)
        ok6 = np.array_equal(built_white, built_again)
        print(f"  [6] determinism (transform twice bit-identical)  "
              f"{'OK' if ok6 else 'FAIL'}")

        op_pass = all([ok1, ok2, ok3, ok4, ok5, ok6])
        all_pass = all_pass and op_pass
        print(f"  => op {op}: {'PASS' if op_pass else 'FAIL'}")

    print(f"\n=== GATE A1 {'PASS' if all_pass else 'FAIL'} ===")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
