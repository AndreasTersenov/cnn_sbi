#!/usr/bin/env python3
"""FoM3 bar comparison — L1 (auto, +product) vs best CNN (resnet18+RealNVP), no-BNT vs BNT.

The two messages in one figure: (1) no-BNT, the CNN modestly beats L1's best (+product); (2) under
BNT the L1 norm COLLAPSES (0.15x / 0.22x) while the CNN is LOSSLESS (0.96x) — so in BNT space the CNN
beats L1 by ~5-9x. All numbers read from disk (pooled 9000-obs median FoM3); CNN = resnet18 + RealNVP
(seed 41 headline), L1 = jaxili MAF. CPU-only.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SBI = "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi"
B = f"{SBI}/results/exploratory/flatsky_cross_2026_06"
OUT = Path(f"{B}/cnn_phase/nde_sweep_2026_06_13/figs")


def fom3(path):
    return float(json.load(open(path))["fom3"])


# arms: label -> (noBNT path, BNT path)
L1POP = f"{B}/population_sweep"; L1BNT = f"{B}/bnt_campaign/population_sweep"


def cnn_mean(template):  # 3-compressor-seed mean (seed-robust)
    return float(np.mean([fom3(template.format(s=s)) for s in (41, 42, 43)]))


CNN_NOBNT = cnn_mean(f"{B}/cnn_phase/arch_sweep_2026_06_13/cnn_resnet18_s{{s}}/readout_full/median_summary.json")
CNN_BNT = cnn_mean(f"{B}/cnn_phase/bnt_resnet18_2026_06_14/cnn_resnet18_bnt_s{{s}}/readout/median_summary.json")
ARMS = [
    ("L1\nauto-only", fom3(f"{L1POP}/flat_none/median_summary.json"), fom3(f"{L1BNT}/l1_none/median_summary.json"), "#d62728"),
    ("L1\n+product", fom3(f"{L1POP}/flat_product/median_summary.json"), fom3(f"{L1BNT}/l1_product/median_summary.json"), "#d62728"),
    ("CNN auto-only\n(resnet18+RealNVP)", CNN_NOBNT, CNN_BNT, "#1f77b4"),
]

labels = [a[0] for a in ARMS]
nobnt = [a[1] for a in ARMS]; bnt = [a[2] for a in ARMS]; cols = [a[3] for a in ARMS]
x = np.arange(len(ARMS)); w = 0.38

fig, ax = plt.subplots(figsize=(7.2, 4.6))
b1 = ax.bar(x - w / 2, nobnt, w, color=cols, edgecolor="black", linewidth=0.6, label="no BNT")
b2 = ax.bar(x + w / 2, bnt, w, color=cols, edgecolor="black", linewidth=0.6, hatch="///",
            alpha=0.55, label="BNT")
for i in range(len(ARMS)):
    ax.text(x[i] - w / 2, nobnt[i] + 40, f"{nobnt[i]:.0f}", ha="center", va="bottom", fontsize=9)
    ax.text(x[i] + w / 2, bnt[i] + 40, f"{bnt[i]:.0f}", ha="center", va="bottom", fontsize=9)
    ax.text(x[i] + w / 2, bnt[i] + 230, f"({bnt[i]/nobnt[i]:.2f}×)", ha="center", va="bottom",
            fontsize=8.5, color="0.3")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel(r"FoM3 = $1/\sqrt{\det C_3}$  (pooled 9000-obs median)")
ax.set_ylim(0, max(nobnt) * 1.18)
# legend with hatch meaning
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="0.5", edgecolor="black", label="no BNT"),
                   Patch(facecolor="0.5", edgecolor="black", hatch="///", alpha=0.55, label="BNT")],
          loc="upper left", fontsize=9)
ax.set_title("Under BNT the L1 norm collapses; the CNN is lossless", fontsize=11)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fom3_bars_l1_cnn_bnt.{ext}", dpi=200, bbox_inches="tight")
print("FoM3 bars (noBNT / BNT / ratio):")
for lab, n, b, _ in ARMS:
    print(f"  {lab.replace(chr(10),' '):32s} {n:6.0f} / {b:6.0f}  ({b/n:.2f}x)")
print(f"wrote {OUT}/fom3_bars_l1_cnn_bnt.{{pdf,png}}")
