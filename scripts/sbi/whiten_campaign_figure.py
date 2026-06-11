#!/usr/bin/env python3
"""Whitening-test decomposition figure (CPU): noBNT vs whitened vs BNT FoM3 bars for
the two L1 arms, annotated with the recovered fraction (whiten − BNT)/(noBNT − BNT).
Companion to fom3_bnt_inflation (bnt_campaign_figures.py); all numbers read from
artifacts on disk — nothing hardcoded."""
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = "/home/tersenov/.claude/skills/figure-polish/style/aa.mplstyle"
SBI = Path("/mnt/home/tersenov/software/cnn_sbi/scripts/sbi")
FC = SBI / "results/exploratory/flatsky_cross_2026_06"
WH = FC / "whiten_campaign"
FIGS = WH / "figures"
C_NOBNT, C_WHITEN, C_BNT = "#0072B2", "#009E73", "#D55E00"


def fom3(path):
    f = Path(path) / "median_summary.json"
    if not f.exists():
        raise FileNotFoundError(f"missing {f} — run the whiten campaign first")
    return json.load(open(f))["fom3"]


def main():
    try:
        plt.style.use(STYLE)
    except OSError:
        pass
    FIGS.mkdir(parents=True, exist_ok=True)

    arms = []
    for op, label in (("none", "L1 auto"), ("product", "L1 +product")):
        n = fom3(FC / f"population_sweep/flat_{op}")
        w = fom3(WH / f"population_sweep/l1_{op}")
        b = fom3(FC / f"bnt_campaign/population_sweep/l1_{op}")
        arms.append((label, n, w, b, (w - b) / (n - b)))
    # optional extra bars (auto arm only): the §5.4 deep-channel ladder
    def opt_fom3(rel):
        f = FC / rel
        return json.load(open(f))["fom3"] if f.exists() else None
    d5 = opt_fom3("bntdeep_campaign/population_sweep/l1_none/median_summary.json")
    d6 = opt_fom3("bntdeep2_campaign/population_sweep/l1_none/median_summary.json")

    x = np.arange(len(arms))
    width = 0.26 if d5 is None else 0.18
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for k, (vals, color, label) in enumerate((
            ([a[1] for a in arms], C_NOBNT, "no BNT"),
            ([a[2] for a in arms], C_WHITEN, "whitened (Q = (BBᵀ)$^{-1/2}$B)"),
            ([a[3] for a in arms], C_BNT, "BNT"))):
        ax.bar(x + (k - 1) * width, vals, width, color=color, edgecolor="k",
               lw=0.5, label=label)
    for xi, (label, n, w, b, rec) in enumerate(arms):
        ax.text(xi, w * 1.15, f"recovered {rec:.2f}", ha="center", fontsize=9,
                color=C_WHITEN)
        ax.text(xi + width, b * 1.3, f"{b/n:.2f}×", ha="center", fontsize=9,
                color=C_BNT)
    n0, b0 = arms[0][1], arms[0][3]
    for k, (val, color, label) in enumerate((
            (d5, "#CC79A7", "BNT + deep (avg)"),
            (d6, "#E69F00", "BNT + deep2 (avg + bin4)"))):
        if val is None:
            continue
        ax.bar((2 + k) * width, val, width, color=color, edgecolor="k", lw=0.5,
               label=label)
        ax.text((2 + k) * width, val * 1.15, f"rec.\n{(val-b0)/(n0-b0):.2f}", ha="center",
                fontsize=8, color=color)
    ax.set_yscale("log")
    ax.set_ylim(top=max(a[1] for a in arms) * 4)
    ax.set_ylabel(r"FoM$_3$ (pooled 9000-obs median)")
    ax.set_xticks(x)
    ax.set_xticklabels([a[0] for a in arms])
    ax.legend(frameon=False, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), fontsize=8)
    ax.set_title("Whitening decomposition of the L1 BNT collapse", fontsize=10,
                 pad=42)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"fom3_whiten_decomposition.{ext}", dpi=200)
    plt.close(fig)
    for label, n, w, b, rec in arms:
        print(f"  {label}: noBNT {n:.0f} | whiten {w:.0f} | BNT {b:.0f} | "
              f"recovered {rec:.3f}")
    print(f"  wrote {FIGS}/fom3_whiten_decomposition.{{png,pdf}}")


if __name__ == "__main__":
    main()
