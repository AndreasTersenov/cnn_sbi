#!/usr/bin/env python3
"""Per-patch distribution violins (the headline quantitative comparison), recreated with the CURRENT
proper arms: l1+product VMIM->sbi_lens RealNVP vs CNN ResNet18 VMIM->sbi_lens RealNVP.

Reads per_patch_metrics.npz (sigma (N,3)=Om/s8/w0, fom3 (N,)) for both arms over the fiducial patch
population. Four panels: sigma(Om), sigma(s8), sigma(w0), FoM3 — violin per arm with median marked.
Output: violins_l1_vs_cnn.{png,pdf}
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/flatsky_cross_2026_06"
HERE = Path(ROOT) / "analytical_nde_match"

ap = argparse.ArgumentParser()
ap.add_argument("--l1", default=f"{HERE}/l1product_rnvp_s41_n9000/per_patch_metrics.npz")
ap.add_argument("--cnn", default=f"{ROOT}/cnn_phase/arch_sweep_2026_06_13/cnn_resnet18_s41/readout_full/per_patch_metrics.npz")
ap.add_argument("--out-tag", default="")
ap.add_argument("--l1-label", default="l1+product → RealNVP")
ap.add_argument("--cnn-label", default="CNN ResNet18 → RealNVP")
a = ap.parse_args()

ARMS = [(a.l1_label, a.l1, "#d62728"), (a.cnn_label, a.cnn, "#1f77b4")]
data = []
for label, path, color in ARMS:
    d = np.load(path)
    f = d["fom3"]; g = np.isfinite(f); s = d["sigma"]
    data.append(dict(label=label, color=color, n=int(g.sum()),
                     vals=[s[g, 0], s[g, 1], s[g, 2], f[g]]))
    print(f"{label}: N={int(g.sum())} med FoM3 {np.median(f[g]):.0f} "
          f"sig {np.median(s[g,0]):.3f}/{np.median(s[g,1]):.3f}/{np.median(s[g,2]):.3f}")

PANELS = [(r"$\sigma(\Omega_m)$", 0, None), (r"$\sigma(\sigma_8)$", 1, None),
          (r"$\sigma(w_0)$", 2, None), ("FoM3", 3, "fom")]
fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
for ax, (title, j, kind) in zip(axes, PANELS):
    series = [arm["vals"][j] for arm in data]
    # clip the long FoM3 tail for a readable violin (keep <=99th pct of the wider arm)
    if kind == "fom":
        hi = max(np.percentile(v, 99) for v in series)
        series = [np.clip(v, None, hi) for v in series]
    parts = ax.violinplot(series, positions=[0, 1], showextrema=False, widths=0.85)
    for body, arm in zip(parts["bodies"], data):
        body.set_facecolor(arm["color"]); body.set_alpha(0.5); body.set_edgecolor(arm["color"])
    for i, arm in enumerate(data):
        med = float(np.median(arm["vals"][j]))
        ax.hlines(med, i - 0.35, i + 0.35, color=arm["color"], lw=2.5)
        ax.text(i, med, f"  {med:.3f}" if kind != "fom" else f"  {med:.0f}",
                va="center", ha="left", fontsize=9, color=arm["color"], fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["l1+product", "CNN"], fontsize=10)
    ax.set_title(title, fontsize=12); ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("per-patch value (over fiducial patches)")
fig.suptitle(f"Per-patch distributions over the fiducial population: l1+product → RealNVP "
             f"(N={data[0]['n']}) vs CNN ResNet18 → RealNVP (N={data[1]['n']})\n"
             "bars = medians · CNN posteriors are tighter+more stable; l1 has larger per-patch scatter",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92])
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"violins{a.out_tag}_l1_vs_cnn.{ext}", dpi=150, bbox_inches="tight")
print(f"wrote violins{a.out_tag}_l1_vs_cnn.png")
plt.close(fig)

# ---- standalone FoM3-only violin ----
figf, axf = plt.subplots(figsize=(5.6, 5.6))
series = [arm["vals"][3] for arm in data]
hi = max(np.percentile(v, 99) for v in series)
series = [np.clip(v, None, hi) for v in series]
parts = axf.violinplot(series, positions=[0, 1], showextrema=False, widths=0.85)
for body, arm in zip(parts["bodies"], data):
    body.set_facecolor(arm["color"]); body.set_alpha(0.5); body.set_edgecolor(arm["color"]); body.set_linewidth(1.5)
meds = []
for i, arm in enumerate(data):
    med = float(np.median(arm["vals"][3])); meds.append(med)
    axf.hlines(med, i - 0.38, i + 0.38, color=arm["color"], lw=2.8)
    axf.text(i + 0.40, med, f"{med:.0f}", va="center", ha="left", color=arm["color"],
             fontweight="bold", fontsize=12)
axf.set_xticks([0, 1])
axf.set_xticklabels(["l1+product\n→ RealNVP", "CNN ResNet18\n→ RealNVP"], fontsize=11)
axf.set_ylabel(r"per-patch FoM3 = $1/\sqrt{\det C_3(\Omega_m,\sigma_8,w_0)}$", fontsize=11)
axf.set_title(f"Per-patch FoM3 over the fiducial population (N={data[0]['n']})\n"
              f"medians: l1 {meds[0]:.0f} · CNN {meds[1]:.0f} (CNN +{100*(meds[1]/meds[0]-1):.0f}%); "
              "l1 has the longer high-FoM3 tail", fontsize=10.5)
axf.grid(axis="y", alpha=0.3)
figf.tight_layout()
for ext in ("png", "pdf"):
    figf.savefig(HERE / f"violin_fom3{a.out_tag}_l1_vs_cnn.{ext}", dpi=150, bbox_inches="tight")
print(f"wrote violin_fom3{a.out_tag}_l1_vs_cnn.png")
