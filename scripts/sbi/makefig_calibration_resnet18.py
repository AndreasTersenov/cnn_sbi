#!/usr/bin/env python3
"""Calibration figure for the best CNN (resnet18 + RealNVP, FoM3 3326): un-stratified TARP-DRP + SBC.

No GPU — reads the GATE-C artifacts (gate_c_resnet18/). TARP-DRP is the FULL val ensemble (terciles
pooled, 600 pts/seed; NOT split). SBC rank histograms follow the project convention: bars = mean of
the 3 per-seed density histograms, grey band = 99% binomial uniform null for N=600 (Talts 2018);
flat within the band = calibrated.
"""
from pathlib import Path
import glob
import numpy as np
from scipy.stats import binom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SBI = "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi"
G = f"{SBI}/results/exploratory/flatsky_cross_2026_06/cnn_phase/arch_sweep_2026_06_13/gate_c_resnet18"
OUT = Path(f"{SBI}/results/exploratory/flatsky_cross_2026_06/cnn_phase/nde_sweep_2026_06_13/figs")
SEEDS = [41, 42, 43]
PARAMS = [r"$\Omega_m$", r"$\sigma_8$", r"$w_0$"]
NB = 20
N_INDEP = 600

# ---- un-stratified TARP-DRP (dim 3), pooled over seeds ----
ecp_seeds, alpha = [], None
for f in sorted(glob.glob(f"{G}/all/curves/*resnet18_all*dim3*.npz")):
    d = np.load(f, allow_pickle=True)
    alpha = d["alpha"]; ecp_seeds.append(d["ecp_bootstrap"].mean(0))   # per-seed mean ECP
ecp_seeds = np.array(ecp_seeds)                                        # (3, 21)
ecp_mean = ecp_seeds.mean(0)
ecp_lo, ecp_hi = ecp_seeds.min(0), ecp_seeds.max(0)
net = float(np.mean(ecp_mean - alpha))

# ---- SBC ranks (terciles pooled within seed) ----
ranks_by_seed = {}
for s in SEEDS:
    samp, th = [], []
    for terc in ["LOW", "MID", "HIGH"]:
        g = glob.glob(f"{G}/dumps/resnet18_rnvp_{terc}/seed_{s}/n*/posterior_samples.npz")
        if g:
            z = np.load(g[0]); samp.append(z["samples"]); th.append(z["theta"])
    S = np.concatenate(samp, 0); T = np.concatenate(th, 0)             # (600,M,6),(600,6)
    ranks_by_seed[s] = np.stack([(S[:, :, p] < T[:, p, None]).mean(1) for p in range(3)], 1)  # (600,3)

bw = 1.0 / NB
band_lo = binom.ppf(0.005, N_INDEP, 1 / NB) / (N_INDEP * bw)
band_hi = binom.ppf(0.995, N_INDEP, 1 / NB) / (N_INDEP * bw)
centers = (np.arange(NB) + 0.5) / NB

fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
# TARP panel
ax = axes[0]
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7, label="ideal")
ax.fill_between(alpha, ecp_lo, ecp_hi, color="#1f77b4", alpha=0.2, label="seed spread")
ax.plot(alpha, ecp_mean, color="#1f77b4", lw=2, label=f"resnet18+RealNVP (net {net:+.3f})")
ax.set_xlabel("credibility level"); ax.set_ylabel("expected coverage probability")
ax.set_title("TARP-DRP (full val ensemble)", fontsize=10)
ax.legend(fontsize=8, loc="upper left"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
# SBC panels
for pi in range(3):
    ax = axes[pi + 1]
    dens = np.mean([np.histogram(ranks_by_seed[s][:, pi], bins=NB, range=(0, 1), density=True)[0]
                    for s in SEEDS], axis=0)
    n_out = int(np.sum((dens < band_lo) | (dens > band_hi)))
    ax.axhspan(band_lo, band_hi, color="0.85", zorder=0, label="99% uniform band")
    ax.bar(centers, dens, width=bw, color="#1f77b4", alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.set_title(f"SBC rank — {PARAMS[pi]}  ({n_out}/{NB} bins out)", fontsize=10)
    ax.set_xlabel("posterior rank of truth"); ax.set_xlim(0, 1); ax.set_ylim(0, max(dens.max() * 1.15, band_hi * 1.3))
    if pi == 0:
        ax.set_ylabel("density"); ax.legend(fontsize=8)
fig.suptitle("Best CNN (resnet18 + sbi_lens RealNVP, FoM3 3326) — calibration "
             "(TARP on diagonal + SBC flat within grey band = calibrated)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"calibration_best_cnn_resnet18.{ext}", dpi=200, bbox_inches="tight")
print(f"wrote {OUT}/calibration_best_cnn_resnet18.{{pdf,png}}  TARP net {net:+.3f}")
for pi in range(3):
    dens = np.mean([np.histogram(ranks_by_seed[s][:, pi], bins=NB, range=(0, 1), density=True)[0]
                    for s in SEEDS], axis=0)
    print(f"  SBC {PARAMS[pi]}: {int(np.sum((dens<band_lo)|(dens>band_hi)))}/{NB} bins outside 99% band")
