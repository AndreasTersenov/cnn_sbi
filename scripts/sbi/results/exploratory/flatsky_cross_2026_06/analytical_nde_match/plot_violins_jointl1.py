#!/usr/bin/env python3
"""Per-patch distribution violins over the full fiducial population (~9000 obs), 3 arms:
l1+product, jointl1 (Q1 winner), CNN ResNet18 — all VMIM->sbi_lens RealNVP.
Reads per_patch_metrics.npz (sigma (N,3)=Om/s8/w0, fom3 (N,)). Panels: σ(Om),σ(s8),σ(w0),FoM3
+ a standalone FoM3 violin. Output: violins_jointl1_3arm.{png,pdf}, violin_fom3_jointl1_3arm.{png,pdf}
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/flatsky_cross_2026_06"
HERE = Path(ROOT) / "analytical_nde_match"

# (label, short, per_patch_metrics path, color)
ARMS = [
    ("l1+product → RealNVP", "l1+product", f"{HERE}/l1product_rnvp_s41_n9000/per_patch_metrics.npz", "#d62728"),
    ("joint l1 → RealNVP", "joint l1", f"{HERE}/jointl1_nobnt/n9000/per_patch_metrics.npz", "#2ca02c"),
    ("CNN ResNet18 → RealNVP", "CNN", f"{ROOT}/cnn_phase/arch_sweep_2026_06_13/cnn_resnet18_s41/readout_full/per_patch_metrics.npz", "#1f77b4"),
]
data = []
for label, short, path, color in ARMS:
    d = np.load(path)
    f = d["fom3"]; g = np.isfinite(f); s = d["sigma"]
    data.append(dict(label=label, short=short, color=color, n=int(g.sum()),
                     vals=[s[g, 0], s[g, 1], s[g, 2], f[g]]))
    print(f"{label}: N={int(g.sum())} med FoM3 {np.median(f[g]):.0f} "
          f"sig {np.median(s[g,0]):.3f}/{np.median(s[g,1]):.3f}/{np.median(s[g,2]):.3f}")

N = len(data); pos = list(range(N))
PANELS = [(r"$\sigma(\Omega_m)$", 0, None), (r"$\sigma(\sigma_8)$", 1, None),
          (r"$\sigma(w_0)$", 2, None), ("FoM3", 3, "fom")]
fig, axes = plt.subplots(1, 4, figsize=(17, 4.8))
for ax, (title, j, kind) in zip(axes, PANELS):
    series = [arm["vals"][j] for arm in data]
    if kind == "fom":
        hi = max(np.percentile(v, 99) for v in series)
        series = [np.clip(v, None, hi) for v in series]
    parts = ax.violinplot(series, positions=pos, showextrema=False, widths=0.8)
    for body, arm in zip(parts["bodies"], data):
        body.set_facecolor(arm["color"]); body.set_alpha(0.5); body.set_edgecolor(arm["color"])
    for i, arm in enumerate(data):
        med = float(np.median(arm["vals"][j]))
        ax.hlines(med, i - 0.32, i + 0.32, color=arm["color"], lw=2.5)
        ax.text(i, med, f"  {med:.3f}" if kind != "fom" else f"  {med:.0f}",
                va="center", ha="left", fontsize=8.5, color=arm["color"], fontweight="bold")
    ax.set_xticks(pos); ax.set_xticklabels([arm["short"] for arm in data], fontsize=9)
    ax.set_title(title, fontsize=12); ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("per-patch value (over fiducial patches)")
fig.suptitle("Per-patch distributions over the fiducial population (~%d obs): "
             "l1+product vs joint l1 vs CNN — all VMIM → sbi_lens RealNVP\n"
             "bars = medians · joint l1 matches the CNN and is tighter than l1+product"
             % data[0]["n"], fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.90])
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"violins_jointl1_3arm.{ext}", dpi=150, bbox_inches="tight")
print("wrote violins_jointl1_3arm.png")
plt.close(fig)

# ---- standalone FoM3-only violin ----
figf, axf = plt.subplots(figsize=(6.6, 5.6))
series = [arm["vals"][3] for arm in data]
hi = max(np.percentile(v, 99) for v in series)
series = [np.clip(v, None, hi) for v in series]
parts = axf.violinplot(series, positions=pos, showextrema=False, widths=0.8)
for body, arm in zip(parts["bodies"], data):
    body.set_facecolor(arm["color"]); body.set_alpha(0.5); body.set_edgecolor(arm["color"]); body.set_linewidth(1.5)
meds = []
for i, arm in enumerate(data):
    med = float(np.median(arm["vals"][3])); meds.append(med)
    axf.hlines(med, i - 0.36, i + 0.36, color=arm["color"], lw=2.8)
    axf.text(i + 0.38, med, f"{med:.0f}", va="center", ha="left", color=arm["color"],
             fontweight="bold", fontsize=12)
axf.set_xticks(pos)
axf.set_xticklabels([arm["short"] + "\n→ RealNVP" for arm in data], fontsize=10)
axf.set_ylabel(r"per-patch FoM3 = $1/\sqrt{\det C_3(\Omega_m,\sigma_8,w_0)}$", fontsize=11)
axf.set_title("Per-patch FoM3 over the fiducial population (N=%d)\n"
              "medians: l1+product %.0f · joint l1 %.0f · CNN %.0f"
              % (data[0]["n"], meds[0], meds[1], meds[2]), fontsize=10.5)
axf.grid(axis="y", alpha=0.3)
figf.tight_layout()
for ext in ("png", "pdf"):
    figf.savefig(HERE / f"violin_fom3_jointl1_3arm.{ext}", dpi=150, bbox_inches="tight")
print("wrote violin_fom3_jointl1_3arm.png")
