#!/usr/bin/env python3
"""no-BNT vs BNT median FoM3 bars (matched best-NDE setup): l1+product and CNN ResNet18.
Reads median_summary.json per arm; annotates the BNT/no-BNT ratio (l1 should collapse, CNN ~lossless).
Output: bnt_fom3_bars_l1_vs_cnn.{png,pdf}
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/flatsky_cross_2026_06"
HERE = Path(ROOT) / "analytical_nde_match"

# (method, color) -> {nobnt path, bnt path}
PATHS = {
    "l1+product": dict(color="#d62728",
        nobnt=f"{HERE}/l1product_rnvp_s41_n9000/median_summary.json",
        bnt=f"{HERE}/l1product_bnt_rnvp_s41_n9000/median_summary.json"),
    "CNN ResNet18": dict(color="#1f77b4",
        nobnt=f"{ROOT}/cnn_phase/arch_sweep_2026_06_13/cnn_resnet18_s41/readout_full/median_summary.json",
        bnt=f"{ROOT}/cnn_phase/bnt_resnet18_2026_06_14/cnn_resnet18_bnt_s41/readout/median_summary.json"),
}


def fom(p):
    try:
        return json.load(open(p))["fom3"]
    except Exception:
        return None


fig, ax = plt.subplots(figsize=(7, 5.4))
x = np.arange(len(PATHS)); w = 0.38
for k, (method, cfg) in enumerate(PATHS.items()):
    nb, bn = fom(cfg["nobnt"]), fom(cfg["bnt"])
    b1 = ax.bar(k - w / 2, nb or 0, w, color=cfg["color"], alpha=0.95, label="no-BNT" if k == 0 else None)
    b2 = ax.bar(k + w / 2, bn or 0, w, color=cfg["color"], alpha=0.45, hatch="//",
                label="BNT" if k == 0 else None)
    if nb:
        ax.text(k - w / 2, nb + 40, f"{nb:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    if bn:
        ax.text(k + w / 2, bn + 40, f"{bn:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    if nb and bn:
        ax.text(k, max(nb, bn) + 260, f"BNT/no-BNT = {bn / nb:.2f}×", ha="center", fontsize=11,
                color=cfg["color"], fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(list(PATHS), fontsize=12)
ax.set_ylabel("median FoM3 = 1/√det C₃(Ωm,σ8,w0)")
ax.set_title("BNT vs no-BNT (matched VMIM→RealNVP NDE)\n"
             "per-channel l1 COLLAPSES under BNT; the channel-mixing CNN is ~lossless", fontsize=11)
ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, None)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"bnt_fom3_bars_l1_vs_cnn.{ext}", dpi=150, bbox_inches="tight")
print("wrote bnt_fom3_bars_l1_vs_cnn.png")
