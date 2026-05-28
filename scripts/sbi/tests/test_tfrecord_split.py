#!/usr/bin/env python
"""§4.2 split-slicing equivalence test.

Confirms the TFRecord shard selection for a sliced split equals the `.npz`
file selection (same stems, same count) -- i.e. `round(frac*n)` slicing picks
the identical realizations on both paths (spec §1.5). Also asserts the 70/30
compressor/NDE file split is disjoint.

The shard lister only inspects filenames (iterdir + suffix + sort + slice), so
we mirror the *full* real nobnt/train population with empty marker `.tfrecord`
files in a temp dir; that exercises the true round() boundaries at n=6293
without converting any data.
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

SBI_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SBI_DIR))

import npe_cnn_nbody_tomo as npe  # noqa: E402

CACHE = SBI_DIR / "results/exploratory/cross_maps_campaign/full_sphere_cache_grid"
REGIME = "nobnt"
SPLITS = ["train", "train[:70%]", "train[70%:]"]


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    if not CACHE.is_dir():
        _fail(f"cache dir not found: {CACHE}")

    # Mirror the full nobnt/train population as empty marker .tfrecord files.
    full_npz = npe._list_harmonic_cache_files(CACHE, REGIME, "train")
    tmp = Path(tempfile.mkdtemp(prefix="tfrec_split_"))
    try:
        marker_dir = tmp / REGIME / "train"
        marker_dir.mkdir(parents=True)
        for p in full_npz:
            (marker_dir / (p.stem + ".tfrecord")).touch()
        print(f"  Mirrored {len(full_npz)} train stems as marker shards.")

        for split in SPLITS:
            npz_stems = [
                p.stem for p in npe._list_harmonic_cache_files(CACHE, REGIME, split)
            ]
            tfr_stems = [
                p.stem
                for p in npe._list_harmonic_tfrecord_shards(tmp, REGIME, split)
            ]
            if npz_stems != tfr_stems:
                _fail(
                    f"split {split!r}: stem selection differs "
                    f"(npz n={len(npz_stems)}, tfr n={len(tfr_stems)})."
                )
            print(f"  [{split}] selection matches: {len(npz_stems)} shards.")

        # Disjointness of the 70/30 compressor vs NDE split (run on real .npz).
        audit = npe.audit_harmonic_split_overlap(
            CACHE, REGIME, "train[:70%]", "train[70%:]"
        )
        if int(audit["overlap_count"]) != 0:
            _fail(
                f"train[:70%] vs train[70%:] overlap_count="
                f"{audit['overlap_count']} (expected 0)."
            )
        print(
            f"  70/30 disjoint: comp_files={audit['compressor_train_files']} "
            f"nde_files={audit['nde_train_files']} overlap=0."
        )
        print("\nPASS: TFRecord split slicing == .npz selection; 70/30 disjoint.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
