#!/usr/bin/env python3
"""Update the CNN calibration figures to the best pipeline (resnet18+RealNVP) + a CNN-vs-L1 panel.

(#1) In-place refresh of the CNN gate-C figures (the old ones were plain-CNN + common-MAF, 4 arms):
     cnn_phase/gate_c/tarp_drp/figures/tarp_resnet18_rnvp_dim3.{png,pdf}  (un-stratified TARP-DRP)
     cnn_phase/gate_c/sbc/sbc_rank_histograms_resnet18.{png,pdf}          (SBC, 99% binomial band)
(#2) New comparison panel: cnn_phase/nde_sweep_2026_06_13/figs/tarp_cnn_vs_l1_calibrated.{png,pdf}
     — CNN (resnet18+RealNVP) and L1+product (MAF) TARP-DRP on one plot, both on the diagonal
       ("both summaries are trustworthy"). No GPU — reads GATE-C dumps/curves.
"""
from pathlib import Path
import glob
import numpy as np
from scipy.stats import binom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SBI = "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi"
B = f"{SBI}/results/exploratory/flatsky_cross_2026_06"
CNN_G = f"{B}/cnn_phase/arch_sweep_2026_06_13/gate_c_resnet18"
L1_G = f"{B}/gate_c/tarp_drp"
CNN_GATE_OUT = f"{B}/cnn_phase/gate_c"          # in-place refresh target
FIGS = Path(f"{B}/cnn_phase/nde_sweep_2026_06_13/figs")
SEEDS = [41, 42, 43]; PARAMS = [r"$\Omega_m$", r"$\sigma_8$", r"$w_0$"]; NB = 20; N_INDEP = 600


def tarp_unstrat(curve_glob):
    ecps, alpha = [], None
    for f in sorted(glob.glob(curve_glob)):
        d = np.load(f, allow_pickle=True); alpha = d["alpha"]; ecps.append(d["ecp_bootstrap"].mean(0))
    e = np.array(ecps)
    return alpha, e.mean(0), e.min(0), e.max(0), float(np.mean(e.mean(0) - alpha))


def sbc_ranks(dumps_root, arm):
    by_seed = {}
    for s in SEEDS:
        samp, th = [], []
        for t in ["LOW", "MID", "HIGH"]:
            g = glob.glob(f"{dumps_root}/{arm}_{t}/seed_{s}/n*/posterior_samples.npz")
            if g:
                z = np.load(g[0]); samp.append(z["samples"]); th.append(z["theta"])
        S = np.concatenate(samp, 0); T = np.concatenate(th, 0)
        by_seed[s] = np.stack([(S[:, :, p] < T[:, p, None]).mean(1) for p in range(3)], 1)
    return by_seed


def draw_tarp(ax, alpha, mean, lo, hi, net, color, label):
    ax.fill_between(alpha, lo, hi, color=color, alpha=0.2)
    ax.plot(alpha, mean, color=color, lw=2, label=f"{label} (net {net:+.3f})")


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    Path(f"{CNN_GATE_OUT}/tarp_drp/figures").mkdir(parents=True, exist_ok=True)
    Path(f"{CNN_GATE_OUT}/sbc").mkdir(parents=True, exist_ok=True)
    # TARP curves
    aC, mC, loC, hiC, netC = tarp_unstrat(f"{CNN_G}/all/curves/*resnet18_all*dim3*.npz")
    aL, mL, loL, hiL, netL = tarp_unstrat(f"{L1_G}/all_product/curves/*flat_product_all*dim3*.npz")

    # ---- (#1a) CNN TARP-DRP standalone (in-place refresh) ----
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7, label="ideal")
    draw_tarp(ax, aC, mC, loC, hiC, netC, "#1f77b4", "CNN auto-only (resnet18+RealNVP)")
    ax.set_xlabel("credibility level"); ax.set_ylabel("expected coverage probability")
    ax.set_title("CNN TARP-DRP (resnet18+RealNVP) — calibrated", fontsize=10)
    ax.legend(fontsize=8, loc="upper left"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(f"{CNN_GATE_OUT}/tarp_drp/figures/tarp_resnet18_rnvp_dim3.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- (#1b) CNN SBC standalone (in-place refresh) ----
    ranks = sbc_ranks(f"{CNN_G}/dumps", "resnet18_rnvp")
    bw = 1.0 / NB; band_lo = binom.ppf(0.005, N_INDEP, 1 / NB) / (N_INDEP * bw)
    band_hi = binom.ppf(0.995, N_INDEP, 1 / NB) / (N_INDEP * bw); centers = (np.arange(NB) + 0.5) / NB
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for pi in range(3):
        ax = axes[pi]
        dens = np.mean([np.histogram(ranks[s][:, pi], bins=NB, range=(0, 1), density=True)[0] for s in SEEDS], 0)
        nout = int(np.sum((dens < band_lo) | (dens > band_hi)))
        ax.axhspan(band_lo, band_hi, color="0.85", zorder=0, label="99% uniform band" if pi == 0 else None)
        ax.bar(centers, dens, width=bw, color="#1f77b4", alpha=0.75, edgecolor="white", linewidth=0.4)
        ax.set_title(f"SBC — {PARAMS[pi]}  ({nout}/{NB} out)", fontsize=10); ax.set_xlim(0, 1)
        ax.set_ylim(0, max(dens.max() * 1.15, band_hi * 1.3)); ax.set_xlabel("posterior rank of truth")
        if pi == 0:
            ax.set_ylabel("density"); ax.legend(fontsize=8)
    fig.suptitle("CNN SBC rank histograms (resnet18+RealNVP) — flat within band = calibrated", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for e in ("pdf", "png"):
        fig.savefig(f"{CNN_GATE_OUT}/sbc/sbc_rank_histograms_resnet18.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- (#2) Combined CNN + L1 TARP-DRP ("both trustworthy") ----
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7, label="ideal (calibrated)")
    draw_tarp(ax, aL, mL, loL, hiL, netL, "#d62728", "L1+product, MAF")
    draw_tarp(ax, aC, mC, loC, hiC, netC, "#1f77b4", "CNN auto-only, resnet18+RealNVP")
    ax.set_xlabel("credibility level"); ax.set_ylabel("expected coverage probability")
    ax.set_title("Both summaries are calibrated (TARP-DRP, full val ensemble)", fontsize=10)
    ax.legend(fontsize=9, loc="upper left"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(FIGS / f"tarp_cnn_vs_l1_calibrated.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"CNN net-bias {netC:+.3f}  |  L1+product net-bias {netL:+.3f}")
    print(f"#1 -> {CNN_GATE_OUT}/tarp_drp/figures/tarp_resnet18_rnvp_dim3 + sbc/sbc_rank_histograms_resnet18")
    print(f"#2 -> {FIGS}/tarp_cnn_vs_l1_calibrated")


if __name__ == "__main__":
    main()
