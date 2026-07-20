#!/usr/bin/env python3
"""Calibration figure (SLIDE style): TARP-DRP + SBC for the best CNN (resnet18 + RealNVP).

Same data/logic as makefig_calibration_resnet18.py — reads the GATE-C artifacts, no GPU — but with
slide-sized fonts, the locked Wong blue (#0072B2), and a short title (the slide's assertion headline
carries the message). Output -> talk_figures/_new_figs/calibration_slide.{pdf,png}.
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
OUT = Path("/mnt/home/tersenov/software/cnn_sbi/talk_figures/_new_figs")
SEEDS = [41, 42, 43]
PARAMS = [r"$\Omega_m$", r"$\sigma_8$", r"$w_0$"]
NB, N_INDEP = 20, 600
C_CNN = "#0072B2"   # locked Wong blue
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
FS_TITLE, FS_LABEL, FS_TICK, FS_LEG, FS_SUP = 15, 15, 12.5, 12, 16

# ---- un-stratified TARP-DRP (dim 3), pooled over seeds ----
ecp_seeds, alpha = [], None
for f in sorted(glob.glob(f"{G}/all/curves/*resnet18_all*dim3*.npz")):
    d = np.load(f, allow_pickle=True)
    alpha = d["alpha"]; ecp_seeds.append(d["ecp_bootstrap"].mean(0))
ecp_seeds = np.array(ecp_seeds)
ecp_mean = ecp_seeds.mean(0); ecp_lo, ecp_hi = ecp_seeds.min(0), ecp_seeds.max(0)
net = float(np.mean(ecp_mean - alpha))

# ---- SBC ranks (terciles pooled within seed) ----
ranks_by_seed = {}
for s in SEEDS:
    samp, th = [], []
    for terc in ["LOW", "MID", "HIGH"]:
        g = glob.glob(f"{G}/dumps/resnet18_rnvp_{terc}/seed_{s}/n*/posterior_samples.npz")
        if g:
            z = np.load(g[0]); samp.append(z["samples"]); th.append(z["theta"])
    S = np.concatenate(samp, 0); T = np.concatenate(th, 0)
    ranks_by_seed[s] = np.stack([(S[:, :, p] < T[:, p, None]).mean(1) for p in range(3)], 1)

bw = 1.0 / NB
band_lo = binom.ppf(0.005, N_INDEP, 1 / NB) / (N_INDEP * bw)
band_hi = binom.ppf(0.995, N_INDEP, 1 / NB) / (N_INDEP * bw)
centers = (np.arange(NB) + 0.5) / NB

fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
ax = axes[0]
ax.plot([0, 1], [0, 1], "k--", lw=1.3, alpha=0.7, label="ideal")
ax.fill_between(alpha, ecp_lo, ecp_hi, color=C_CNN, alpha=0.2, label="seed spread")
ax.plot(alpha, ecp_mean, color=C_CNN, lw=2.6, label=f"CNN (net {net:+.3f})")
ax.set_xlabel("credibility level", fontsize=FS_LABEL)
ax.set_ylabel("expected coverage", fontsize=FS_LABEL)
ax.set_title("TARP-DRP coverage", fontsize=FS_TITLE)
ax.legend(fontsize=FS_LEG, loc="upper left"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.tick_params(labelsize=FS_TICK)
for pi in range(3):
    ax = axes[pi + 1]
    dens = np.mean([np.histogram(ranks_by_seed[s][:, pi], bins=NB, range=(0, 1), density=True)[0]
                    for s in SEEDS], axis=0)
    n_out = int(np.sum((dens < band_lo) | (dens > band_hi)))
    ax.axhspan(band_lo, band_hi, color="0.85", zorder=0, label="99% uniform band")
    ax.bar(centers, dens, width=bw, color=C_CNN, alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.axhline(1.0, color="k", ls="--", lw=0.9, alpha=0.5)
    ax.set_title(f"SBC rank — {PARAMS[pi]}  ({n_out}/{NB} out)", fontsize=FS_TITLE)
    ax.set_xlabel("posterior rank of truth", fontsize=FS_LABEL)
    ax.set_xlim(0, 1); ax.set_ylim(0, max(dens.max() * 1.15, band_hi * 1.3))
    ax.tick_params(labelsize=FS_TICK)
    if pi == 0:
        ax.set_ylabel("density", fontsize=FS_LABEL); ax.legend(fontsize=FS_LEG)
fig.suptitle("The contours are trustworthy: TARP on the diagonal + SBC flat within the band",
             fontsize=FS_SUP)
fig.tight_layout(rect=[0, 0, 1, 0.94])
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"calibration_slide.{ext}", dpi=200, bbox_inches="tight")
print(f"wrote {OUT}/calibration_slide.{{pdf,png}}  TARP net {net:+.3f}")
