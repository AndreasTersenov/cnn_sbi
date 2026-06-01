#!/usr/bin/env python
"""Validate the full TFRecord cross dataset (read-only): shards present, tfds.load
returns the expected schema, and bit-exact spot-check vs the .npz cache for a few
examples per split.

Pure CPU, no destructive ops. Run after the full build PID exits:
  FULL_TFRECORD_DIR=/nas/tersenov/tfds_cross_tfrecord_full \
    python scripts/sbi/tests/validate_full_tfrecord_build.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

DATA_DIR = os.environ.get(
    "FULL_TFRECORD_DIR", "/nas/tersenov/tfds_cross_tfrecord_full"
)
NAME = "nbody_cosmogrid_dataset_tomo_cross"
CONFIG = "grid_20deg_160px_nonoverlap48"
N_CHECK_PER_SPLIT = 10  # spot-check sample size per split
SPLIT_MAP = {"train": "train", "test": "val", "obs": "obs"}

_HERE = Path(__file__).resolve().parent
_SBI = _HERE.parent
if str(_SBI) not in sys.path:
    sys.path.insert(0, str(_SBI))

CACHE = _SBI / "results" / "exploratory" / "cross_maps_campaign" / "full_sphere_cache_grid"


def main() -> int:
    # 1. shards on disk
    version_dir = Path(DATA_DIR) / NAME / CONFIG / "0.0.1"
    if not version_dir.exists():
        print(f"FAIL: version dir missing: {version_dir}")
        return 1
    shards = sorted(version_dir.glob("*.tfrecord*"))
    info_json = version_dir / "dataset_info.json"
    print(f"shards on disk: {len(shards)}  ({sum(f.stat().st_size for f in shards) / 1e9:.1f} GB)")
    print(f"dataset_info.json present: {info_json.exists()}")
    if len(shards) == 0 or not info_json.exists():
        print("FAIL: dataset incomplete")
        return 1

    # Build cosmo_idx -> cosmo_id map by scanning the .npz cache dir filenames once,
    # reading just the small cosmo_idx member (np.load is lazy for .npz members).
    print("building cosmo_idx -> path map from .npz cache (one pass)...", flush=True)
    idx_to_path: dict[tuple[int, int, str], Path] = {}
    for sub in set(SPLIT_MAP.values()):
        for f in sorted((CACHE / "nobnt" / sub).glob("*.npz")):
            d = np.load(f, allow_pickle=True)
            idx_to_path[(int(d["cosmo_idx"]), int(d["perm"]), sub)] = f
    print(f"  mapped {len(idx_to_path)} (cosmo_idx, perm, split) entries")

    # 2. tfds.load works + 3. bit-exact spot-check
    import tensorflow_datasets as tfds
    import tf_dataset_nbody_tomo_cross as _cross_builder  # noqa: F401  (registers)

    builder = _cross_builder.NbodyCosmogridDatasetTomoCross(
        config=CONFIG, data_dir=DATA_DIR
    )

    total_checked = 0
    max_diff = 0.0
    fail = False
    npz_cache: dict[Path, np.ndarray] = {}

    for tfds_split, cache_sub in SPLIT_MAP.items():
        try:
            ds = builder.as_dataset(split=tfds_split)
        except Exception as e:
            print(f"  split {tfds_split!r}: as_dataset FAILED ({e})")
            fail = True
            continue
        n_local = 0
        for ex in tfds.as_numpy(ds.take(N_CHECK_PER_SPLIT)):
            ci, perm, k = int(ex["cosmo_idx"]), int(ex["perm"]), int(ex["patch"])
            key = (ci, perm, cache_sub)
            if key not in idx_to_path:
                print(f"  MISSING npz ref for ci={ci} perm={perm} split={cache_sub}")
                fail = True
                continue
            npz_path = idx_to_path[key]
            if npz_path not in npz_cache:
                npz_cache[npz_path] = np.load(npz_path, allow_pickle=True)["patches"]
            ref_patch = npz_cache[npz_path][k]
            diff = float(np.max(np.abs(ex["map_nbody"] - ref_patch)))
            max_diff = max(max_diff, diff)
            n_local += 1
            if diff != 0.0:
                fail = True
                print(f"  MISMATCH split={tfds_split} ci={ci} perm={perm} k={k} diff={diff:.3e}")
        total_checked += n_local
        print(f"  split {tfds_split!r}: checked {n_local} / {N_CHECK_PER_SPLIT}")

    print(f"\nTOTAL checked: {total_checked}  max abs diff: {max_diff:.3e}")
    ok = (not fail) and total_checked > 0 and max_diff == 0.0
    print("RESULT:", "PASS (bit-exact)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
