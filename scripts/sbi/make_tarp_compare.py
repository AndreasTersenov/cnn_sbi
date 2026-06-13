#!/usr/bin/env python
"""TARP-DRP coverage: A1 (VMIM joint PDF) vs l1+product (prev best, gate-C clean) vs raw
pair2d K=10. dim-3 science subspace, HIGH-FoM3 tercile + all-tercile pooled, 3-seed mean
band. The calibration companion to the A1-vs-product comparison. CPU, existing curves."""
import glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FC = ("/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/"
      "flatsky_cross_2026_06")
OUT = f"{FC}/overnight_menu_2/lane_a_plots"
ARMS = {  # label -> (gate_root, arm, color)
    "A1 VMIM joint PDF (net +0.021)": (f"{FC}/overnight_menu_2/gate_c", "A1_pair2d_vmim", "tab:red"),
    "l1 + product (gate-C clean, −0.015)": (f"{FC}/gate_c", "flat_product", "tab:purple"),
    "pair2d K=10 raw (−0.044)": (f"{FC}/overnight_menu/gate_c", "pair2dq_nobnt", "tab:blue"),
}
SEEDS = (41, 42, 43)


def curve(gr, arm, terc, seed):
    f = f"{gr}/tarp_drp/curves/tarp_curve_{arm}_{terc}_seed{seed}_dim3.npz"
    if not Path(f).exists():
        return None
    z = np.load(f)
    return np.asarray(z["alpha"]), np.asarray(z["ecp_bootstrap"]).mean(0)


def pooled(gr, arm, terc):
    cs = [curve(gr, arm, terc, s) for s in SEEDS]
    cs = [c for c in cs if c is not None]
    if not cs:
        return None, None, None
    a0 = cs[0][0]
    e = np.stack([np.interp(a0, c[0], c[1]) for c in cs])
    return a0, e.mean(0), e.std(0)


fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
for ax, terc, title in zip(axes, ("HIGH", "ALL"),
                           ("HIGH-FoM3 tercile (tightest posteriors)",
                            "all terciles pooled")):
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="calibrated")
    for label, (gr, arm, col) in ARMS.items():
        if terc == "ALL":
            acc = [pooled(gr, arm, t)[:2] for t in ("HIGH", "MID", "LOW")]
            acc = [(a, m) for a, m in acc if a is not None]
            a0 = acc[0][0]
            m = np.mean([np.interp(a0, a, mm) for a, mm in acc], axis=0)
            s = None
        else:
            a0, m, s = pooled(gr, arm, terc)
        ax.plot(a0, m, color=col, lw=1.8, label=label)
        if s is not None:
            ax.fill_between(a0, m - s, m + s, color=col, alpha=0.15)
    ax.set_xlabel("credibility level  α"); ax.set_ylabel("expected coverage  ECP")
    ax.set_title(title, fontsize=10); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.97, 0.06, "below diagonal =\nover-confident", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="gray")
    ax.legend(fontsize=8, loc="upper left")
fig.suptitle("TARP-DRP coverage — A1 (VMIM joint PDF) vs previous best (l1+product), dim-3",
             fontsize=11)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/tarp_a1_vs_product.{ext}", dpi=140, bbox_inches="tight")
print(f"wrote {OUT}/tarp_a1_vs_product.png/pdf")
