#!/usr/bin/env python3
"""Per-patch FoM3 distribution: best CNN (RealNVP) vs best L1 (MAF) over the 9000-patch test set.

This is the figure that carries the QUANTITATIVE claim (a single contour can't show a ~9% median
edge legibly). Uses the existing full-sweep per-patch FoM3 (no GPU): A0_full (CNN sbi_lens RealNVP
4x128) and population_sweep/flat_product (L1 jaxili MAF). Violin + box, log-y, medians annotated.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SBI = "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi"
BASE = f"{SBI}/results/exploratory/flatsky_cross_2026_06"
OUT = Path(f"{BASE}/cnn_phase/nde_sweep_2026_06_13/figs")

cnn = np.load(f"{BASE}/cnn_phase/nde_sweep_2026_06_13/A0_full_sbilens_4x128/per_patch_metrics.npz")["fom3"]
l1 = np.load(f"{BASE}/population_sweep/flat_product/per_patch_metrics.npz")["fom3"]
cnn = cnn[np.isfinite(cnn)]; l1 = l1[np.isfinite(l1)]
cm, lm = np.median(cnn), np.median(l1)
data = [l1, cnn]
labels = [f"L1+product\nMAF\n(median {lm:.0f})", f"CNN auto-only\nRealNVP\n(median {cm:.0f})"]
colors = ["#d62728", "#1f77b4"]

fig, ax = plt.subplots(figsize=(5.2, 4.2))
parts = ax.violinplot(data, positions=[0, 1], showextrema=False, widths=0.8)
for pc, c in zip(parts["bodies"], colors):
    pc.set_facecolor(c); pc.set_alpha(0.35); pc.set_edgecolor(c)
bp = ax.boxplot(data, positions=[0, 1], widths=0.18, patch_artist=True, showfliers=False,
                medianprops=dict(color="k", lw=1.6), whiskerprops=dict(color="0.4"),
                capprops=dict(color="0.4"), boxprops=dict(alpha=0.85))
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.5)
ax.set_yscale("log")
ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel(r"per-patch FoM3 $= 1/\sqrt{\det C_3}$  ($\Omega_m,\sigma_8,w_0$)")
ax.axhline(lm, color=colors[0], ls=":", lw=1, alpha=0.7)
ax.axhline(cm, color=colors[1], ls=":", lw=1, alpha=0.7)
# fraction of patches where CNN > L1 (paired; index orders verified identical)
n = min(len(cnn), len(l1)); frac = float(np.mean(cnn[:n] > l1[:n]))
ax.set_title(f"CNN tighter at {100*frac:.0f}% of patches   |   median ratio {cm/lm:.2f}×",
             fontsize=11)
ax.text(0.02, 0.02, "each probe at its best calibrated NDE\n(L1: MAF; CNN: RealNVP — which craters on 2000-D L1)",
        transform=ax.transAxes, fontsize=8, va="bottom", color="0.3")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fom3_distribution_cnn_vs_l1.{ext}", dpi=200, bbox_inches="tight")
print(f"CNN median {cm:.0f}  L1 median {lm:.0f}  ratio {cm/lm:.3f}  | per-patch CNN>L1 frac {frac:.2f}")
print(f"wrote {OUT}/fom3_distribution_cnn_vs_l1.{{pdf,png}}")
