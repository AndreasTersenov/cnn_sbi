#!/usr/bin/env python
"""§4.3 epoch-completeness (set-equality) test.

Converts a small split (nobnt/val limited to 10 shards = 480 patches),
iterates the TFRecord shards for exactly one epoch (no repeat, no shuffle),
collects (cosmo_id, perm, patch_idx) tags, and asserts the set equals the
`.npz` set: every patch present exactly once (spec §1.8).
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

import numpy as np
import tensorflow as tf

SBI_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SBI_DIR))

import build_harmonic_tfrecord as builder  # noqa: E402
import npe_cnn_nbody_tomo as npe  # noqa: E402

CACHE = SBI_DIR / "results/exploratory/cross_maps_campaign/full_sphere_cache_grid"
REGIME = "nobnt"
SPLIT = "val"
N_SHARDS = 10

_FEATURE_DESC = {
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

    npz_files = npe._list_harmonic_cache_files(CACHE, REGIME, SPLIT)[:N_SHARDS]

    # .npz expected tag set.
    npz_tags = set()
    for src in npz_files:
        with np.load(src, allow_pickle=False) as d:
            n_patches = int(np.asarray(d["patches"]).shape[0])
            cid = str(d["cosmo_id"])
            perm = int(d["perm"])
        for pi in range(n_patches):
            npz_tags.add((cid, perm, pi))

    tmp = Path(tempfile.mkdtemp(prefix="tfrec_epoch_"))
    try:
        for src in npz_files:
            out_path = tmp / REGIME / SPLIT / (src.stem + ".tfrecord")
            builder._convert_one_file(
                {
                    "src_path": str(src),
                    "out_path": str(out_path),
                    "regime": REGIME,
                    "split": SPLIT,
                    "compress": "NONE",
                    "overwrite": True,
                }
            )

        shard_paths = sorted(
            str(p) for p in (tmp / REGIME / SPLIT).glob("*.tfrecord")
        )
        tfr_tags = []
        for raw in tf.data.TFRecordDataset(shard_paths, compression_type=""):
            ex = tf.io.parse_single_example(raw, _FEATURE_DESC)
            tfr_tags.append(
                (
                    ex["cosmo_id"].numpy().decode("utf-8"),
                    int(ex["perm"].numpy()),
                    int(ex["patch_idx"].numpy()),
                )
            )

        tfr_set = set(tfr_tags)
        if len(tfr_tags) != len(tfr_set):
            _fail(
                f"duplicate patches in one epoch: {len(tfr_tags)} read, "
                f"{len(tfr_set)} unique."
            )
        if tfr_set != npz_tags:
            missing = npz_tags - tfr_set
            extra = tfr_set - npz_tags
            _fail(
                f"tag set mismatch: {len(missing)} missing, {len(extra)} extra."
            )
        print(
            f"  {len(npz_files)} shards, {len(tfr_tags)} patches; "
            f"set equals .npz set, no duplicates."
        )
        print("\nPASS: one TFRecord epoch covers exactly the .npz patch set.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
