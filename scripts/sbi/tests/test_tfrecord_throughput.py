#!/usr/bin/env python
"""§4.5 throughput-sanity test (informational, not correctness).

Times 200 batches (batch 128) from the TFRecord iterator vs the `.npz`
iterator on the SAME 50 nobnt/train realizations and prints patches/s for both.
Expect TFRecord >= 3x the `.npz` rate; a smaller ratio signals a misconfigured
pipeline (missing interleave/prefetch/AUTOTUNE).

CPU/IO only -- touches no GPU.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import shutil
import sys
import time
from pathlib import Path

SBI_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SBI_DIR))

import build_harmonic_tfrecord as builder  # noqa: E402
import npe_cnn_nbody_tomo as npe  # noqa: E402

CACHE = SBI_DIR / "results/exploratory/cross_maps_campaign/full_sphere_cache_grid"
REGIME = "nobnt"
N_SHARDS = 50
BATCH = 128
N_TIME = 200
N_WARMUP = 10
# Build bench shards on /nas (abundant space, reflects the FUSE target).
BENCH_ROOT = Path("/nas/tersenov/_tfrec_bench_tmp")


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def _time_iter(it, n_warmup, n_time, batch):
    for _ in range(n_warmup):
        next(it)
    t0 = time.time()
    for _ in range(n_time):
        next(it)
    elapsed = time.time() - t0
    patches = n_time * batch
    return patches / elapsed, elapsed


def main() -> None:
    if not CACHE.is_dir():
        _fail(f"cache dir not found: {CACHE}")

    npz_files = npe._list_harmonic_cache_files(CACHE, REGIME, "train")[:N_SHARDS]
    tfr_dir = BENCH_ROOT / "full_sphere_cache_grid"
    if tfr_dir.exists():
        shutil.rmtree(tfr_dir, ignore_errors=True)
    try:
        print(f"  Building {len(npz_files)} bench shards (NONE) under {tfr_dir} ...")
        for src in npz_files:
            out_path = tfr_dir / REGIME / "train" / (src.stem + ".tfrecord")
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

        tfr_it = npe.build_harmonic_tfrecord_iterator(
            tfrecord_dir=tfr_dir, regime=REGIME, split="train", batch_size=BATCH,
            seed=42, flip=True, max_realizations=N_SHARDS, channel_scale=None,
            channel_slice=None, compression="NONE",
        )
        tfr_rate, tfr_t = _time_iter(tfr_it, N_WARMUP, N_TIME, BATCH)
        print(f"  TFRecord: {tfr_rate:,.0f} patches/s  ({N_TIME} batches in {tfr_t:.1f}s)")

        npz_it = npe.build_harmonic_batch_iterator(
            cache_dir=CACHE, regime=REGIME, split="train", batch_size=BATCH,
            seed=42, flip=True, max_realizations=N_SHARDS, channel_scale=None,
            channel_slice=None,
        )
        npz_rate, npz_t = _time_iter(npz_it, N_WARMUP, N_TIME, BATCH)
        print(f"  .npz:     {npz_rate:,.0f} patches/s  ({N_TIME} batches in {npz_t:.1f}s)")

        ratio = tfr_rate / npz_rate if npz_rate > 0 else float("inf")
        print(f"  Speedup (TFRecord / .npz): {ratio:.2f}x")
        if ratio >= 3.0:
            print("\nPASS: TFRecord >= 3x the .npz rate.")
        else:
            print(
                f"\nWARN: TFRecord only {ratio:.2f}x (< 3x). Pipeline may be "
                "misconfigured (interleave/prefetch/AUTOTUNE). Informational."
            )
    finally:
        shutil.rmtree(tfr_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
