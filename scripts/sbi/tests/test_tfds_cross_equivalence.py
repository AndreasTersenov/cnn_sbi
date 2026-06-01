#!/usr/bin/env python
"""Bit-exact gate: TFDS-cross examples == the .npz cross cache they reserialize.

The TFDS builder reserializes the validated `.npz` cache. This gate builds a
1-cosmology-per-split subset (CROSS_TFDS_COSMO_LIMIT=1; fast — just reads ~7 .npz/split),
loads it back via `as_data_source` (ArrayRecord), and for a stride sample of examples
confirms the map round-trips bit-exactly (max abs diff 0.0) and the (cosmo_idx, perm,
patch) provenance indexes the right patch. (The .npz cache itself is proven bit-exact to
the compute by tests/validate_cross_compute_refactor.py.)

CPU-only, ~1 min. Run:
  /home/tersenov/anaconda3/envs/jaxili/bin/python -u scripts/sbi/tests/test_tfds_cross_equivalence.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SBI = _HERE.parent
if str(_SBI) not in sys.path:
    sys.path.insert(0, str(_SBI))

CACHE = _SBI / "results" / "exploratory" / "cross_maps_campaign" / "full_sphere_cache_grid"
REGIME = "nobnt"
# TFDS split -> .npz cache subdir.
SPLIT_MAP = {"train": "train", "test": "val", "obs": "obs"}
LIMIT = 1


def load_subset(cache_sub: str, limit: int) -> dict[tuple[int, int], np.ndarray]:
    """Mirror the builder's file selection; return {(cosmo_idx, perm): patches}."""
    files = sorted((CACHE / REGIME / cache_sub).glob("*.npz"))
    seen: set[str] = set()
    kept = []
    for f in files:
        cid = f.name.split("_perm")[0]
        if cid not in seen:
            if len(seen) >= limit:
                continue
            seen.add(cid)
        kept.append(f)
    out = {}
    for f in kept:
        d = np.load(f, allow_pickle=True)
        out[(int(d["cosmo_idx"]), int(d["perm"]))] = (d["patches"], d["theta"].astype(np.float32))
    return out


def main() -> int:
    os.environ["CROSS_TFDS_COSMO_LIMIT"] = str(LIMIT)
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    import tf_dataset_nbody_tomo_cross as M

    tmp = Path(tempfile.mkdtemp(prefix="tfds_cross_gate_"))
    print(f"building subset (limit={LIMIT}/split) -> {tmp}  [array_record]", flush=True)
    builder = M.NbodyCosmogridDatasetTomoCross(
        config="grid_20deg_160px_nonoverlap48", data_dir=str(tmp), file_format="array_record",
    )
    builder.download_and_prepare()
    print("build done; verifying vs .npz cache ...", flush=True)

    n_checked = 0
    max_map_diff = 0.0
    max_theta_diff = 0.0
    fail = False

    for tfds_split, cache_sub in SPLIT_MAP.items():
        ref = load_subset(cache_sub, LIMIT)
        try:
            ds = builder.as_data_source(split=tfds_split)
        except Exception as e:
            print(f"  split {tfds_split!r}: could not load ({e})")
            fail = True
            continue
        n = len(ds)
        stride = max(1, n // 60)
        idxs = list(range(0, n, stride))
        print(f"  split {tfds_split!r}: {n} examples, checking {len(idxs)} sampled", flush=True)
        for i in idxs:
            ex = ds[i]
            ci, perm, k = int(ex["cosmo_idx"]), int(ex["perm"]), int(ex["patch"])
            if (ci, perm) not in ref:
                print(f"  MISSING ref for cosmo_idx={ci} perm={perm} split={tfds_split}")
                fail = True
                continue
            ref_patches, ref_theta = ref[(ci, perm)]
            md = float(np.max(np.abs(ex["map_nbody"] - ref_patches[k])))
            td = float(np.max(np.abs(ex["theta"] - ref_theta)))
            max_map_diff = max(max_map_diff, md)
            max_theta_diff = max(max_theta_diff, td)
            n_checked += 1
            if md != 0.0:
                fail = True
                print(f"  MISMATCH split={tfds_split} ci={ci} perm={perm} patch={k} map_diff={md:.3e}")

    print(f"checked {n_checked} examples")
    print(f"MAX MAP ABS DIFF:   {max_map_diff:.3e}")
    print(f"MAX THETA ABS DIFF: {max_theta_diff:.3e}")
    ok = (not fail) and n_checked > 0 and max_map_diff == 0.0
    print("RESULT:", "PASS (bit-exact)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
