#!/usr/bin/env python
"""§4.4 output-contract test (+ a numerical reader check).

Part A (spec §4.4): one batch from `build_harmonic_tfrecord_iterator`
(batch_size=128, train, flip=True) must have the right shapes/dtypes/keys, for
both the unsliced (C==10) and channel_slice(0,4) (C==4) cases.

Part B (extra): validate the *production reader's* tf-graph slice+scale and
H0/100 numerically against the `.npz` computation on a single shard, flip=False
(deterministic order). §4.1 proved the raw bytes round-trip; this closes the
loop on the in-graph ops the reader actually applies at train time.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import shutil
import sys
import tempfile
from pathlib import Path

import jax
import numpy as np

SBI_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SBI_DIR))

import build_harmonic_tfrecord as builder  # noqa: E402
import npe_cnn_nbody_tomo as npe  # noqa: E402

CACHE = SBI_DIR / "results/exploratory/cross_maps_campaign/full_sphere_cache_grid"
REGIME = "nobnt"


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def _build_shards(npz_files, tmp, split):
    for src in npz_files:
        out_path = tmp / REGIME / split / (src.stem + ".tfrecord")
        builder._convert_one_file(
            {
                "src_path": str(src),
                "out_path": str(out_path),
                "regime": REGIME,
                "split": split,
                "compress": "NONE",
                "overwrite": True,
            }
        )


def main() -> None:
    if not CACHE.is_dir():
        _fail(f"cache dir not found: {CACHE}")

    npz_files = npe._list_harmonic_cache_files(CACHE, REGIME, "train")[:5]
    tmp = Path(tempfile.mkdtemp(prefix="tfrec_contract_"))
    try:
        _build_shards(npz_files, tmp, "train")

        # Part A: contract -- no slice (C==10).
        it10 = npe.build_harmonic_tfrecord_iterator(
            tfrecord_dir=tmp, regime=REGIME, split="train", batch_size=128,
            seed=42, flip=True, channel_slice=None, channel_scale=None,
            compression="NONE",
        )
        b = next(it10)
        if set(b.keys()) != {"maps", "theta"}:
            _fail(f"unexpected keys: {set(b.keys())}")
        # maps is now a JAX device array (DLPack zero-copy handoff); theta stays
        # numpy float64.
        if not isinstance(b["maps"], jax.Array):
            _fail(f"maps is {type(b['maps'])}, expected jax.Array (device)")
        maps = np.asarray(b["maps"])
        if maps.shape != (128, 160, 160, 10):
            _fail(f"maps shape {maps.shape} != (128,160,160,10)")
        if maps.dtype != np.float32:
            _fail(f"maps dtype {maps.dtype} != float32")
        if not isinstance(b["theta"], np.ndarray):
            _fail(f"theta is {type(b['theta'])}, expected np.ndarray")
        if b["theta"].shape != (128, 6):
            _fail(f"theta shape {b['theta'].shape} != (128,6)")
        if b["theta"].dtype != np.float64:
            _fail(f"theta dtype {b['theta'].dtype} != float64")
        print("  [A] unsliced contract OK: maps(128,160,160,10) f32 jax.Array, theta(128,6) f64 np.")

        # Part A: contract -- slice(0,4) (C==4).
        it4 = npe.build_harmonic_tfrecord_iterator(
            tfrecord_dir=tmp, regime=REGIME, split="train", batch_size=128,
            seed=42, flip=True, channel_slice=slice(0, 4), channel_scale=None,
            compression="NONE",
        )
        b4 = next(it4)
        maps4 = np.asarray(b4["maps"])
        if maps4.shape != (128, 160, 160, 4):
            _fail(f"sliced maps shape {maps4.shape} != (128,160,160,4)")
        if maps4.dtype != np.float32:
            _fail(f"sliced maps dtype {maps4.dtype} != float32")
        print("  [A] sliced contract OK: maps(128,160,160,4) f32.")

        # Part B: numerical reader check on ONE shard, flip=False (deterministic
        # order 0..47), with slice(0,4) + channel_scale. Compares the reader's
        # in-graph slice/scale/H0 against the .npz computation.
        one_file = npz_files[0]
        single_tmp = Path(tempfile.mkdtemp(prefix="tfrec_contract_single_"))
        try:
            _build_shards([one_file], single_tmp, "train")
            cs = npe.compute_harmonic_channel_rms(
                CACHE, REGIME, "train", max_realizations=1,
                channel_slice=slice(0, 4), use_disk_cache=False,
            )
            it = npe.build_harmonic_tfrecord_iterator(
                tfrecord_dir=single_tmp, regime=REGIME, split="train",
                batch_size=48, seed=0, flip=False, channel_slice=slice(0, 4),
                channel_scale=cs, compression="NONE",
            )
            rb = next(it)
            with np.load(one_file, allow_pickle=False) as d:
                p = np.asarray(d["patches"], dtype=np.float32)
                th = np.asarray(d["theta"], dtype=np.float64)
            expect_maps = p[..., 0:4] / cs
            expect_theta = npe._theta_batch_from_harmonic(th, 48)
            # maps is a JAX device array; bring it to host for comparison.
            rb_maps = np.asarray(rb["maps"])
            dmap = float(np.abs(rb_maps - expect_maps).max())
            dth = float(np.abs(rb["theta"] - expect_theta).max())
            if dmap > 1e-6:
                _fail(f"reader maps differ from .npz: max abs diff {dmap:.3e}.")
            if dth != 0.0:
                _fail(f"reader theta differs from .npz: max abs diff {dth:.3e}.")
            print(
                f"  [B] reader slice+scale+H0 matches .npz: "
                f"maps diff={dmap:.1e}, theta diff={dth:.1e}."
            )
        finally:
            shutil.rmtree(single_tmp, ignore_errors=True)

        print("\nPASS: output contract correct; reader numerics match .npz.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
