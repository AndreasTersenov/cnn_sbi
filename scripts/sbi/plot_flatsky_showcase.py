#!/usr/bin/env python3
"""Showcase the flat-sky de-leaking result: FoM3 bars (flat-local vs leaky full-sphere) +
per-patch cross/auto FoM3 ratio distributions over the 9000 fiducial obs (median vs single-obs)."""
import json, glob, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi"
POP = HERE + "/results/exploratory/flatsky_cross_2026_06/population_sweep"
OUT = HERE + "/results/exploratory/flatsky_cross_2026_06/figs"
d = {a: json.load(open(f"{POP}/flat_{a}/median_summary.json")) for a in ("none", "conv", "product", "both")}
FS_AUTO, FS_CROSS = 2200, 8530   # full-sphere L1 (SUMMARY_PHASE_D, pooled 9000-obs)

# paired per-patch cross/auto FoM3 ratios
m = {a: np.load(f"{POP}/flat_{a}/per_patch_metrics.npz") for a in ("none", "conv", "product", "both")}
key = {(int(p), int(q)): i for i, (p, q) in enumerate(zip(m["none"]["perm"], m["none"]["patch"]))}
def ratios(a):
    r = []
    for i, (p, q) in enumerate(zip(m[a]["perm"], m[a]["patch"])):
        j = key.get((int(p), int(q)))
        if j is not None and np.isfinite(m[a]["fom3"][i]) and m["none"]["fom3"][j] > 0:
            r.append(m[a]["fom3"][i] / m["none"]["fom3"][j])
    return np.array(r)
def single(a):   # perm0/patch90
    ia = np.where((m[a]["perm"] == 0) & (m[a]["patch"] == 90))[0][0]
    jn = np.where((m["none"]["perm"] == 0) & (m["none"]["patch"] == 90))[0][0]
    return m[a]["fom3"][ia] / m["none"]["fom3"][jn]

fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))

# (a) FoM3 bars
arms = ["auto-only", "+conv", "+product", "+both"]
vals = [d["none"]["fom3"], d["conv"]["fom3"], d["product"]["fom3"], d["both"]["fom3"]]
xb = np.arange(4)
ax[0].bar(xb, vals, color=["#555", "#1f77b4", "#2ca02c", "#d62728"], width=0.62, label="flat-local (patch-local)")
ax[0].axhline(FS_AUTO, color="k", ls=":", lw=1.2)
ax[0].axhline(FS_CROSS, color="#8B0000", ls="--", lw=2)
ax[0].text(3.5, FS_CROSS, "  full-sphere auto+cross\n  8530 (LEAKY, 3.88×)", color="#8B0000",
           va="center", ha="left", fontsize=9, fontweight="bold")
ax[0].text(3.5, FS_AUTO, "  full-sphere auto-only 2200", color="k", va="bottom", ha="left", fontsize=8)
for x, v in zip(xb, vals):
    ax[0].text(x, v + 80, f"{v:.0f}\n({v/vals[0]:.2f}×)", ha="center", fontsize=9)
ax[0].set_xticks(xb); ax[0].set_xticklabels(arms)
ax[0].set_ylabel("FoM3 (pooled 3-seed, 9000-obs median)"); ax[0].set_ylim(0, 9200)
ax[0].set_title("De-leaked cross gain is MODEST\n~92% of the full-sphere cross gain was leakage")
ax[0].annotate("", xy=(3.7, FS_CROSS), xytext=(3.7, 2910),
               arrowprops=dict(arrowstyle="<->", color="#8B0000", lw=1.5))
ax[0].text(3.78, (FS_CROSS + 2910) / 2, "leakage", color="#8B0000", rotation=90, va="center", fontsize=9)

# (b) per-patch ratio distributions
for a, col, lab in [("conv", "#1f77b4", "+conv"), ("product", "#2ca02c", "+product"), ("both", "#d62728", "+both")]:
    r = ratios(a)
    ax[1].hist(r, bins=60, range=(0.7, 1.7), histtype="step", color=col, lw=2,
               label=f"{lab}: median {np.median(r):.2f}×")
    ax[1].axvline(np.median(r), color=col, ls="-", lw=1, alpha=0.5)
    ax[1].plot(single(a), 1, "v", color=col, ms=11, mec="k")
ax[1].axvline(1.0, color="k", ls=":", lw=1.2)
ax[1].set_xlabel("per-patch cross/auto-only FoM3 ratio (9000 fiducial obs)")
ax[1].set_ylabel("patches"); ax[1].legend(fontsize=9, loc="upper right")
ax[1].set_title("Cross gain is patch-variable; conv ≈ 1 (fragile),\nproduct robust. ▼ = single-obs (perm0/patch90)")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/flatsky_showcase.{ext}", bbox_inches="tight", dpi=130)
print(f"wrote {OUT}/flatsky_showcase.png ; single-obs ratios:",
      {a: round(single(a), 2) for a in ("conv", "product", "both")},
      "medians:", {a: round(float(np.median(ratios(a))), 2) for a in ("conv", "product", "both")})
