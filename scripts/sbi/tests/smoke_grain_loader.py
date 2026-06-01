#!/usr/bin/env python
"""Standalone smoke for the Grain loader: pull a few batches, check shapes/dtypes.

Verifies the Grain pipeline end-to-end (tfds.data_source -> shuffle -> random_map ->
batch -> mp_prefetch worker processes) on the subset ArrayRecord dataset, before the
heavier integrated benchmark. CPU-only.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_SBI = Path(__file__).resolve().parent.parent
if str(_SBI) not in sys.path:
    sys.path.insert(0, str(_SBI))

DATA = "/nas/tersenov/tfds_cross_arrayrecord_subset20"
NAME = "nbody_cosmogrid_dataset_tomo_cross/grid_20deg_160px_nonoverlap48"


def main() -> int:
    from grain_loader import build_grain_iterator

    it = build_grain_iterator(
        tfds_name=NAME, data_dir=DATA, split="train", batch_size=128, seed=42,
        flip=True, channel_scale=None, channel_slice=None, num_workers=8,
    )
    t0 = time.time()
    last = t0
    for i in range(6):
        b = next(it)
        now = time.time()
        print(f"batch {i}: maps {b['maps'].shape} {b['maps'].dtype} "
              f"theta {b['theta'].shape} {b['theta'].dtype}  dt={now-last:.3f}s", flush=True)
        last = now
    maps, theta = b["maps"], b["theta"]
    ok = (maps.shape == (128, 160, 160, 10) and maps.dtype == np.float32
          and theta.shape == (128, 6) and theta.dtype == np.float32)
    # theta[:,3] should be h0 (~0.6-0.8) after /100, not H0 (~60-80)
    h0_ok = bool(np.all(theta[:, 3] < 2.0))
    print(f"shape/dtype ok: {ok}; theta[:,3] looks like h0 (<2): {h0_ok} "
          f"(sample {theta[0, 3]:.4f})")
    print("RESULT:", "PASS" if (ok and h0_ok) else "FAIL")
    return 0 if (ok and h0_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
