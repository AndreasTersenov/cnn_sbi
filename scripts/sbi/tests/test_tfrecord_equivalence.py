#!/usr/bin/env python
"""§4.1 numerical-equivalence test -- THE critical gate.

Proves the TFRecord serialization round-trips the `.npz` patches and theta
*bit-identically*. This must pass before the reader or any production use.

Run (CPU-only, never touches GPU):
    conda run -n jaxili python scripts/sbi/tests/test_tfrecord_equivalence.py
"""

from __future__ import annotations

import os

# CPU-only: this is a correctness test, it must not grab a GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

SBI_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SBI_DIR))

import build_harmonic_tfrecord as builder  # noqa: E402
import npe_cnn_nbody_tomo as npe  # noqa: E402

CACHE = SBI_DIR / "results/exploratory/cross_maps_campaign/full_sphere_cache_grid"
REGIME = "nobnt"
N_FILES = 5

_FEATURE_DESC = {
    "patch": tf.io.FixedLenFeature([], tf.string),
    "theta": tf.io.FixedLenFeature([], tf.string),
    "cosmo_id": tf.io.FixedLenFeature([], tf.string),
    "perm": tf.io.FixedLenFeature([], tf.int64),
    "patch_idx": tf.io.FixedLenFeature([], tf.int64),
}


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    if not CACHE.is_dir():
        _fail(f"cache dir not found: {CACHE}")

    # Step 1: first 5 sorted .npz of nobnt/train (240 patches).
    npz_files = npe._list_harmonic_cache_files(CACHE, REGIME, "train")[:N_FILES]
    print(f"  Using {len(npz_files)} source files:")
    for f in npz_files:
        print(f"    {f.name}")

    # Step 2: build a tiny TFRecord from exactly those 5 files (NONE compression).
    tmp = Path(tempfile.mkdtemp(prefix="tfrec_eq_"))
    try:
        for src in npz_files:
            out_path = tmp / REGIME / "train" / (src.stem + ".tfrecord")
            builder._convert_one_file(
                {
                    "src_path": str(src),
                    "out_path": str(out_path),
                    "regime": REGIME,
                    "split": "train",
                    "compress": "NONE",
                    "overwrite": True,
                }
            )

        # Step 3: Path A (.npz). Concatenate patches; build theta via the
        # production helper (broadcast + H0/100). Tag every row.
        A_patches_list, A_theta_list, A_tags = [], [], []
        for src in npz_files:
            with np.load(src, allow_pickle=False) as d:
                p = np.asarray(d["patches"], dtype=np.float32)
                th = np.asarray(d["theta"], dtype=np.float64)
                cid = str(d["cosmo_id"])
                perm = int(d["perm"])
            A_patches_list.append(p)
            A_theta_list.append(npe._theta_batch_from_harmonic(th, p.shape[0]))
            A_tags.extend((cid, perm, pi) for pi in range(p.shape[0]))
        A_patches = np.concatenate(A_patches_list, axis=0)
        A_theta = np.concatenate(A_theta_list, axis=0)

        # Step 4: Path B (TFRecord). Deterministic, no shuffle, flip=False.
        shard_paths = sorted(
            str(p) for p in (tmp / REGIME / "train").glob("*.tfrecord")
        )
        B_patches_list, B_theta_list, B_tags = [], [], []
        for raw in tf.data.TFRecordDataset(shard_paths, compression_type=""):
            ex = tf.io.parse_single_example(raw, _FEATURE_DESC)
            patch = tf.reshape(
                tf.io.decode_raw(ex["patch"], tf.float32), (160, 160, 10)
            ).numpy()
            theta_raw = tf.io.decode_raw(ex["theta"], tf.float64).numpy()
            B_patches_list.append(patch)
            B_theta_list.append(theta_raw)
            B_tags.append(
                (
                    ex["cosmo_id"].numpy().decode("utf-8"),
                    int(ex["perm"].numpy()),
                    int(ex["patch_idx"].numpy()),
                )
            )
        B_patches = np.stack(B_patches_list, axis=0)
        # The .npz path applies H0/100 at read time; replicate on the per-record
        # raw theta (H0=68.5 -> h0=0.685).
        B_theta = np.stack(B_theta_list, axis=0).copy()
        B_theta[:, 3] /= 100.0

        # Step 5: match by identity. Sort both by (cosmo_id, perm, patch_idx).
        A_order = sorted(range(len(A_tags)), key=lambda i: A_tags[i])
        B_order = sorted(range(len(B_tags)), key=lambda i: B_tags[i])
        A_tags_sorted = [A_tags[i] for i in A_order]
        B_tags_sorted = [B_tags[i] for i in B_order]
        if A_tags_sorted != B_tags_sorted:
            _fail("tag sequences differ -- not the same 240 patches present.")
        if len(A_tags_sorted) != N_FILES * 48:
            _fail(f"expected {N_FILES * 48} patches, got {len(A_tags_sorted)}.")
        A_p = A_patches[A_order]
        A_t = A_theta[A_order]
        B_p = B_patches[B_order]
        B_t = B_theta[B_order]

        # Step 6: bit-exact raw patches.
        diff_raw = float(np.abs(A_p - B_p).max())
        if not np.array_equal(A_p, B_p):
            _fail(f"raw patches differ: max abs diff {diff_raw:.3e} (want 0.0).")
        print(f"  [6] raw patches bit-identical: max abs diff = {diff_raw:.1e}")

        # Step 7: theta bit-exact.
        diff_theta = float(np.abs(A_t - B_t).max())
        if not np.array_equal(A_t, B_t):
            _fail(f"theta differs: max abs diff {diff_theta:.3e} (want 0.0).")
        print(
            f"  [7] theta bit-identical (post H0/100): max abs diff = "
            f"{diff_theta:.1e}; sample h0={A_t[0, 3]:.4f}"
        )

        # Step 8: with channel_slice(0, 4).
        A_sl = A_p[..., 0:4]
        B_sl = B_p[..., 0:4]
        diff_sl = float(np.abs(A_sl - B_sl).max())
        if not np.array_equal(A_sl, B_sl):
            _fail(f"sliced patches differ: max abs diff {diff_sl:.3e} (want 0.0).")
        print(f"  [8] sliced[0:4] bit-identical: max abs diff = {diff_sl:.1e}")

        # Step 9: with channel_scale (slice -> divide) on both paths.
        cs = npe.compute_harmonic_channel_rms(
            CACHE,
            REGIME,
            "train",
            max_realizations=N_FILES,
            channel_slice=slice(0, 4),
            use_disk_cache=False,
        )
        A_scaled = A_p[..., 0:4] / cs
        B_scaled = B_p[..., 0:4] / cs
        diff_scaled = float(np.abs(A_scaled - B_scaled).max())
        if diff_scaled != 0.0:
            # Tolerate tiny float32 op-ordering noise but flag loudly.
            if diff_scaled > 1e-6:
                _fail(
                    f"scaled patches differ beyond tol: {diff_scaled:.3e} > 1e-6."
                )
            print(
                f"  [9] WARNING scaled differ by {diff_scaled:.3e} "
                f"(<=1e-6, float32 op-ordering)."
            )
        else:
            print(f"  [9] scaled[0:4] bit-identical: max abs diff = {diff_scaled:.1e}")

        # Step 10: zero-mean on B's raw patches.
        npe._assert_zero_mean_patches(B_p, "tfrecord-path-B")
        print("  [10] zero-mean assertion passed on TFRecord patches.")

        print(
            "\nPASS: TFRecord path is bit-identical to the .npz path "
            f"(raw/theta/slice max abs diff 0.0; scaled diff {diff_scaled:.1e})."
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
