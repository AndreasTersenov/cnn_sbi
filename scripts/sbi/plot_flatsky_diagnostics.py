#!/usr/bin/env python3
"""Diagnostic plots for the flat-local cross-map L1 pipeline (pre-campaign interpretation).

Fig 1  maps_examples : the actual inference inputs for one patch — 4 noisy auto patches,
                       6 convolution cross maps, 6 product cross maps.
Fig 2  datavector_full: the EXACT op=both L1 datavector (frozen sigma + calibrated per-channel
                       ranges), per-cosmology mean, overlaid for low/mid/high sigma8; channel
                       segments marked. This is what feeds the L1 NDE (before log1p-zscore).
Fig 3  l1_hist_zoom  : per-scale L1 histograms for 3 representative channels (auto_bin4,
                       conv_34, prod_34), overlaid by sigma8 tercile — the datavector pieces
                       that move with cosmology.
Fig 4  datavector_Om : same as Fig 2 but split by Omega_m terciles (product tracks Om > conv).

Runs on GPU 1. Outputs -> results/exploratory/flatsky_cross_2026_06/figs/.
"""
import os, sys, glob, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flatsky_cross as fx
import flatsky_cross_l1 as fxl

SIG = HERE + "/results/exploratory/flatsky_cross_2026_06/flatsky_cross_noise_sigma.npz"
FIDP = (HERE + "/results/exploratory/cross_maps_campaign/"
        "full_sphere_cache_fiducial_10deg/nobnt/obs/cosmo_fiducial_perm0.npz")
TFDS = "nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180"
DDIR = "/home/tersenov/tensorflow_datasets"
OUT = HERE + "/results/exploratory/flatsky_cross_2026_06/figs"
os.makedirs(OUT, exist_ok=True)
RESO, L1_NBINS, NBINS, NS = 7.5, 40, 4, 5
PAIRS = fx.cross_pairs(NBINS)
N_PATCHES = 36000
plt.rcParams.update({"font.size": 9, "figure.dpi": 130})


def fig_maps(patch_idx=90):
    z = np.load(FIDP)
    autos = z["patches"][patch_idx, :, :, :4].astype(np.float32)        # (80,80,4)
    autos = autos - autos.mean((0, 1), keepdims=True)
    chans = fx.build_channels_np(autos[None], "both")[0]               # (80,80,16)
    fig, axes = plt.subplots(3, 6, figsize=(15, 7.6))
    titles = ([f"auto κ{b+1}" for b in range(4)] + ["", ""]
              + [f"conv {i+1}×{j+1}" for i, j in PAIRS]
              + [f"product {i+1}×{j+1}" for i, j in PAIRS])
    for k, ax in enumerate(axes.ravel()):
        if titles[k] == "":
            ax.axis("off"); continue
        m = chans[:, :, k if k < 4 else k - 2]   # map cols 4,5 are blank in row 0
        v = np.percentile(np.abs(m), 99) + 1e-30
        ax.imshow(m, cmap="RdBu_r", vmin=-v, vmax=v); ax.set_title(titles[k], fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("noisy autos", fontsize=11)
    axes[1, 0].set_ylabel("convolution cross", fontsize=11)
    axes[2, 0].set_ylabel("product cross", fontsize=11)
    fig.suptitle(f"Flat-local inference inputs — fiducial patch {patch_idx} "
                 f"(10°/80px, demeaned; per-panel symmetric scaling)", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/maps_examples.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote maps_examples.{{png,pdf}}")


def gather_datavectors(stats, sig, ranges, dev):
    import torch
    from tfds_cross_tfdata_loader import iter_cross_tfds_batches
    xs, ths = [], []
    n, t0 = 0, time.time()
    for autos_np, theta in iter_cross_tfds_batches(
        TFDS, DDIR, "train", 512, flip=False, channel_scale=None,
        channel_slice=slice(0, 4), perm_lo=0, perm_hi=3, seed=11,
    ):
        xs.append(fxl.build_and_l1(autos_np, "both", sig, stats, L1_NBINS, ranges, clamp_overflow=True))
        ths.append(theta.copy()); n += autos_np.shape[0]
        if n >= N_PATCHES:
            break
    X = np.concatenate(xs)[:N_PATCHES]; TH = np.concatenate(ths)[:N_PATCHES]
    print(f"  datavectors: {X.shape} in {time.time()-t0:.0f}s")
    return X, TH


def _tercile_means(X, p):
    o = np.argsort(p); e = np.linspace(0, len(p), 4).astype(int)
    out, lab = [], []
    for b in range(3):
        idx = o[e[b]:e[b + 1]]
        out.append(X[idx].mean(0)); lab.append(p[idx].mean())
    return out, lab


def fig_datavector_full(X, TH, pidx, pname, fname):
    feat = NS * L1_NBINS                                   # 200 per channel
    means, labs = _tercile_means(X, TH[:, pidx])
    names = ([f"a{b+1}" for b in range(4)] + [f"c{i+1}{j+1}" for i, j in PAIRS]
             + [f"p{i+1}{j+1}" for i, j in PAIRS])
    fig, ax = plt.subplots(figsize=(15, 4.2))
    colors = ["#2166ac", "#777777", "#b2182b"]
    for m, lab, col in zip(means, labs, colors):
        ax.plot(m, color=col, lw=0.8, label=f"{pname}≈{lab:.3f}")
    for c in range(1, 16):
        ax.axvline(c * feat, color="k", lw=0.4, alpha=0.25)
    for c in range(16):
        ax.text((c + 0.5) * feat, ax.get_ylim()[1] * 0.92, names[c], ha="center", fontsize=6,
                color=("#1a1a1a" if c < 4 else "#0a5" if c < 10 else "#a05"))
    ax.axvspan(0, 4 * feat, color="#1f77b4", alpha=0.05)
    ax.axvspan(4 * feat, 10 * feat, color="#2ca02c", alpha=0.05)
    ax.axvspan(10 * feat, 16 * feat, color="#d62728", alpha=0.05)
    ax.set_xlim(0, 16 * feat); ax.set_yscale("log")
    ax.set_xlabel("L1 feature index  (16 channels × 5 scales × 40 SNR bins)  — autos | conv | product")
    ax.set_ylabel("mean L1 norm")
    ax.set_title(f"Exact op=both L1 datavector (frozen σ + calibrated ranges) vs {pname}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{fname}.{ext}", bbox_inches="tight")
    plt.close(fig); print(f"  wrote {fname}.{{png,pdf}}")


def fig_hist_zoom(X, TH):
    feat = NS * L1_NBINS
    means, labs = _tercile_means(X, TH[:, 1])             # sigma8
    chsel = [("auto κ4", 3), ("conv 3×4", 9), ("product 3×4", 15)]
    fig, axes = plt.subplots(3, NS, figsize=(15, 7), sharex=True)
    colors = ["#2166ac", "#777777", "#b2182b"]
    for r, (cname, ch) in enumerate(chsel):
        for s in range(NS):
            ax = axes[r, s]
            lo = ch * feat + s * L1_NBINS
            for m, lab, col in zip(means, labs, colors):
                ax.plot(m[lo:lo + L1_NBINS], color=col, lw=1.2,
                        label=(f"σ8≈{lab:.2f}" if (r == 0 and s == NS - 1) else None))
            if r == 0:
                ax.set_title(f"scale {s}")
            if s == 0:
                ax.set_ylabel(cname, fontsize=10)
    axes[0, NS - 1].legend(fontsize=7)
    fig.suptitle("L1 histograms (40 SNR bins) per scale — datavector pieces moving with σ8", fontsize=12)
    fig.supxlabel("SNR bin")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/l1_hist_zoom.{ext}", bbox_inches="tight")
    plt.close(fig); print("  wrote l1_hist_zoom.{png,pdf}")


def main():
    import torch
    from wl_stats_torch import WLStatistics
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("############ flat-local diagnostics ############")
    print("Fig 1: example maps"); fig_maps()
    stats = WLStatistics(n_scales=NS, device=dev, pixel_arcmin=RESO, dtype=torch.float64)
    sig, names, _ = fxl.select_frozen_sigma(SIG, "both", NBINS, dev)
    print("Calibrating per-channel SNR ranges (exact inference ranges)...")
    ranges = fxl.calibrate_snr_range_flat_local(
        TFDS, DDIR, "both", sig, stats, NBINS, names,
        n_calibration_examples=3600, perm_lo=0, perm_hi=0, seed=0)
    X, TH = gather_datavectors(stats, sig, ranges, dev)
    print("Fig 2/4: full datavector vs sigma8 / Omega_m")
    fig_datavector_full(X, TH, 1, "σ8", "datavector_full_sigma8")
    fig_datavector_full(X, TH, 0, "Ωm", "datavector_full_Om")
    print("Fig 3: per-scale L1 histogram zoom")
    fig_hist_zoom(X, TH)
    print(f"\nAll figures in {OUT}")


if __name__ == "__main__":
    main()
