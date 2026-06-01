#!/usr/bin/env python
"""Standalone CPU smoke for the standard tf.data loader on the FULL TFRecord cross
dataset: pull a few batches, check shape/dtype/theta scaling. ~10 s.

Verifies the loader works end-to-end on the full data (not just the subset) before
the integrated training smoke.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_SBI = Path(__file__).resolve().parent.parent
if str(_SBI) not in sys.path:
    sys.path.insert(0, str(_SBI))

DATA = "/nas/tersenov/tfds_cross_tfrecord_full"
NAME = "nbody_cosmogrid_dataset_tomo_cross/grid_20deg_160px_nonoverlap48"


def main() -> int:
    from tfds_cross_tfdata_loader import build_tfds_tfdata_iterator

    it = build_tfds_tfdata_iterator(
        tfds_name=NAME, data_dir=DATA, split="train", batch_size=128, seed=42,
        flip=True, channel_scale=None, channel_slice=None, shuffle_buffer=512,
    )
    t0 = time.time()
    last = t0
    for i in range(4):
        b = next(it)
        now = time.time()
        print(f"batch {i}: maps {b['maps'].shape} {b['maps'].dtype}  "
              f"theta {b['theta'].shape} {b['theta'].dtype}  dt={now-last:.3f}s", flush=True)
        last = now
    maps, theta = b["maps"], b["theta"]
    shape_ok = (maps.shape == (128, 160, 160, 10) and maps.dtype == np.float32
                and theta.shape == (128, 6) and theta.dtype == np.float32)
    h0_ok = bool(np.all(theta[:, 3] < 2.0))  # h0 (~0.6-0.8), not H0 (~60-80)
    map_range_ok = bool(np.isfinite(maps).all() and maps.std() > 0)
    print(f"shape/dtype ok: {shape_ok} | theta[:,3] in h0 range: {h0_ok} "
          f"(sample {theta[0,3]:.4f}) | maps finite + nonzero std: {map_range_ok}")
    ok = shape_ok and h0_ok and map_range_ok
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
