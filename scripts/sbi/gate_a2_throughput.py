#!/usr/bin/env python3
"""GATE A2 — flat_local L1 datavector THROUGHPUT (augment ON vs OFF).

Measures patches/s for the on-device flat-sky cross + frozen-sigma wavelet-l1 over a
bounded pass of the real cross TFDS (autos only). 'none' (autos, no cross) is the OFF
baseline; conv/product/both are ON. If ON ~ OFF, the on-device cross is NOT starving the
GPU (the wavelet dominates); if ON << OFF, the cross build is the bottleneck. Reports >=2
batch sizes (feedback_benchmark_dont_assume: measured, repeated). Sample nvidia-smi -i 1
externally while this runs to confirm GPU is fed. Runs on GPU 1.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flatsky_cross as fx
import flatsky_cross_l1 as fxl

SIGMA = HERE + "/results/exploratory/flatsky_cross_2026_06/flatsky_cross_noise_sigma.npz"
TFDS = "nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180"
DDIR = "/home/tersenov/tensorflow_datasets"
RESO = 7.5
CAP = 12000          # patches per (op,batch) measurement
WARMUP = 2           # warmup batches excluded from timing


def measure(op, batch_size, stats, dev):
    import torch
    from tfds_cross_tfdata_loader import iter_cross_tfds_batches
    C = fx.n_output_channels(4, op)
    sig, names, _ = fxl.select_frozen_sigma(SIGMA, op, 4, dev)
    ranges = np.tile([-8.0, 8.0], (C, 1))
    n, t0, nb = 0, None, 0
    for autos_np, _ in iter_cross_tfds_batches(
        TFDS, DDIR, "train", batch_size, flip=True,
        channel_scale=None, channel_slice=slice(0, 4), perm_lo=5, perm_hi=5, seed=41,
    ):
        _ = fxl.build_and_l1(autos_np, op, sig, stats, 40, ranges, clamp_overflow=True)
        nb += 1
        if nb == WARMUP:
            torch.cuda.synchronize(); t0 = time.time(); n = 0
        elif nb > WARMUP:
            n += autos_np.shape[0]
            if n >= CAP:
                break
    torch.cuda.synchronize()
    el = time.time() - t0
    return n / el, el, C


def main():
    import torch
    from wl_stats_torch import WLStatistics
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stats = WLStatistics(n_scales=5, device=dev, pixel_arcmin=RESO, dtype=torch.float64)
    print(f"############ GATE A2 throughput (device={dev}, cap={CAP}/measurement) ############")
    print(f"{'op':9s} {'C':>3s} {'batch':>6s} {'patches/s':>11s} {'rel-to-none':>12s}")
    base = {}
    for bs in (512, 1024):
        for op in ("none", "conv", "product", "both"):
            rate, el, C = measure(op, bs, stats, dev)
            base.setdefault(bs, rate if op == "none" else base[bs])
            rel = rate / base[bs]
            print(f"{op:9s} {C:3d} {bs:6d} {rate:11.0f} {rel:11.2f}x")
        print()
    print("Interpretation: rel-to-none ~1 => on-device cross build is negligible (wavelet-bound, "
          "GPU fed); rel << 1 => cross build starves the GPU (fix placement).")


if __name__ == "__main__":
    main()
