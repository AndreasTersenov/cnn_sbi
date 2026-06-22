#!/usr/bin/env python3
"""Regenerate the best-CNN calibration panel (calibration_best_cnn_resnet18) with the PROPER TARP band.

The original used a 'seed spread' (min/max over 3 near-identical seed curves) for the TARP band, which
is ~200x too small and carried the old single-flow net (~+0.059). Here the TARP panel uses the proper
sightline-bootstrap 1-sigma band recomputed in tarp_3way.npz (CNN, un-stratified, net +0.033). The 3
SBC panels are unchanged in spirit (per-seed rank density + 99% binomial band). CPU only.
"""
from pathlib import Path
import glob
import numpy as np
from scipy.stats import binom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CR = Path("/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/flatsky_cross_2026_06/cnn_phase/calib_refine_2026_06")
G = "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/flatsky_cross_2026_06/cnn_phase/arch_sweep_2026_06_13/gate_c_resnet18"
OUT = Path("/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/flatsky_cross_2026_06/cnn_phase/nde_sweep_2026_06_13/figs")
PARAMS = [r"$\Omega_m$", r"$\sigma_8$", r"$w_0$"]
NB = 20; N_INDEP = 600; SEEDS = [41, 42, 43]


def sbc_density():
    per = {p: [] for p in range(3)}
    for s in SEEDS:
        S, T = [], []
        for t in ("LOW", "MID", "HIGH"):
            g = glob.glob(f"{G}/dumps/resnet18_rnvp_{t}/seed_{s}/n*_m*/posterior_samples.npz")
            if g:
                z = np.load(g[0]); S.append(z["samples"]); T.append(z["theta"])
        S = np.concatenate(S, 0); T = np.concatenate(T, 0)
        for p in range(3):
            per[p].append((S[:, :, p] < T[:, p, None]).mean(1))
    return per


def main():
    z = np.load(CR / "tarp_3way.npz")
    a = z["alpha"]; mean = z["CNN_mean"]; lo = z["CNN_lo"]; hi = z["CNN_hi"]; net = float(z["CNN_net"])
    per = sbc_density()
    bw = 1.0 / NB
    band_lo = binom.ppf(0.005, N_INDEP, 1 / NB) / (N_INDEP * bw)
    band_hi = binom.ppf(0.995, N_INDEP, 1 / NB) / (N_INDEP * bw)
    centers = (np.arange(NB) + 0.5) / NB

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1.1, alpha=0.7, label="ideal")
    ax.fill_between(a, lo, hi, color="#1f77b4", alpha=0.28, label=r"1$\sigma$ (sightline bootstrap)")
    ax.plot(a, mean, color="#1f77b4", lw=2.2, label=f"CNN (net {net:+.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("credibility level"); ax.set_ylabel("expected coverage probability")
    ax.set_title("TARP-DRP (full val ensemble)", fontsize=10); ax.legend(fontsize=8, loc="upper left")

    for pi in range(3):
        ax = axes[pi + 1]
        dens = np.mean([np.histogram(r, bins=NB, range=(0, 1), density=True)[0] for r in per[pi]], 0)
        std_p = float(np.mean([np.std(r) for r in per[pi]]))
        n_out = int(np.sum((dens < band_lo) | (dens > band_hi)))
        ax.axhspan(band_lo, band_hi, color="0.85", zorder=0, label="99% band" if pi == 0 else None)
        ax.bar(centers, dens, width=bw, color="#1f77b4", alpha=0.78, edgecolor="white", linewidth=0.4)
        ax.set_title(f"SBC rank — {PARAMS[pi]}  (std {std_p:.3f}; {n_out}/{NB} out)", fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, max(dens.max() * 1.15, band_hi * 1.3)); ax.set_xlabel("rank of truth")
        if pi == 0:
            ax.set_ylabel("density"); ax.legend(fontsize=8)
    fig.suptitle("Best CNN (resnet18 + sbi_lens RealNVP, FoM3 3326) — calibration "
                 "(TARP near diagonal + SBC flat within grey band = calibrated; mildly conservative)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"calibration_best_cnn_resnet18.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}/calibration_best_cnn_resnet18.{{pdf,png}}  TARP net {net:+.3f}")


if __name__ == "__main__":
    main()
