#!/usr/bin/env python3
"""L1-norm histograms vs sigma8 across MANY cosmologies, viridis color-coded.

Bins a wide TFDS sample into N_LEVELS sigma8 levels (equal-count), takes the mean op=both
L1 datavector per level (frozen sigma + calibrated per-channel ranges = the exact inference
datavector, pre log1p-zscore), and plots the per-scale L1 histograms for representative
channels (auto kappa4, conv 3x4, product 3x4) as a viridis gradient in sigma8 + a colorbar.
Runs on GPU 1. -> results/exploratory/flatsky_cross_2026_06/figs/l1_hist_vs_s8_viridis.{png,pdf}
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flatsky_cross as fx
import flatsky_cross_l1 as fxl

SIG = HERE + "/results/exploratory/flatsky_cross_2026_06/flatsky_cross_noise_sigma.npz"
TFDS = "nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180"
DDIR = "/home/tersenov/tensorflow_datasets"
OUT = HERE + "/results/exploratory/flatsky_cross_2026_06/figs"
RESO, L1_NBINS, NBINS, NS = 7.5, 40, 4, 5
N_PATCHES, N_LEVELS = 70000, 16
plt.rcParams.update({"font.size": 9, "figure.dpi": 130})


def main():
    import torch
    from wl_stats_torch import WLStatistics
    from tfds_cross_tfdata_loader import iter_cross_tfds_batches
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stats = WLStatistics(n_scales=NS, device=dev, pixel_arcmin=RESO, dtype=torch.float64)
    sig, names, _ = fxl.select_frozen_sigma(SIG, "both", NBINS, dev)
    ranges = fxl.calibrate_snr_range_flat_local(
        TFDS, DDIR, "both", sig, stats, NBINS, names,
        n_calibration_examples=3600, perm_lo=0, perm_hi=0, seed=0)

    xs, s8 = [], []
    n, t0 = 0, time.time()
    for autos_np, theta in iter_cross_tfds_batches(
        TFDS, DDIR, "train", 512, flip=False, channel_scale=None,
        channel_slice=slice(0, 4), perm_lo=0, perm_hi=5, seed=23,
    ):
        xs.append(fxl.build_and_l1(autos_np, "both", sig, stats, L1_NBINS, ranges, clamp_overflow=True))
        s8.append(theta[:, 1].copy()); n += autos_np.shape[0]
        if n >= N_PATCHES:
            break
    X = np.concatenate(xs)[:N_PATCHES]; S8 = np.concatenate(s8)[:N_PATCHES]
    print(f"  {X.shape[0]} patches in {time.time()-t0:.0f}s; sigma8 {S8.min():.3f}..{S8.max():.3f}")

    # equal-count sigma8 levels
    order = np.argsort(S8); edges = np.linspace(0, len(S8), N_LEVELS + 1).astype(int)
    lvl_s8 = np.zeros(N_LEVELS); lvl_X = np.zeros((N_LEVELS, X.shape[1]))
    for b in range(N_LEVELS):
        idx = order[edges[b]:edges[b + 1]]
        lvl_s8[b] = S8[idx].mean(); lvl_X[b] = X[idx].mean(0)

    feat = NS * L1_NBINS
    norm = Normalize(vmin=lvl_s8.min(), vmax=lvl_s8.max())
    sm = cm.ScalarMappable(norm=norm, cmap="viridis"); sm.set_array([])
    chsel = [("auto κ4", 3), ("conv 3×4", 9), ("product 3×4", 15)]
    fig, axes = plt.subplots(3, NS, figsize=(16, 7.4), sharex=True)
    for r, (cname, ch) in enumerate(chsel):
        for s in range(NS):
            ax = axes[r, s]
            lo = ch * feat + s * L1_NBINS
            for b in range(N_LEVELS):
                ax.plot(lvl_X[b, lo:lo + L1_NBINS], color=cm.viridis(norm(lvl_s8[b])), lw=1.0)
            if r == 0:
                ax.set_title(f"scale {s}")
            if s == 0:
                ax.set_ylabel(cname, fontsize=10)
    fig.suptitle(f"L1 histograms vs σ8 — {N_LEVELS} cosmology levels (op=both, exact datavector)",
                 fontsize=12)
    fig.supxlabel("SNR bin")
    fig.tight_layout(rect=[0, 0, 0.93, 1])
    cax = fig.add_axes([0.945, 0.12, 0.012, 0.76])
    fig.colorbar(sm, cax=cax, label="σ₈")
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/l1_hist_vs_s8_viridis.{ext}", bbox_inches="tight")
    print(f"  wrote {OUT}/l1_hist_vs_s8_viridis.{{png,pdf}}")


if __name__ == "__main__":
    main()
