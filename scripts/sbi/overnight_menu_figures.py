#!/usr/bin/env python3
"""Overnight-menu figures (CPU; all numbers read from median_summary.json artifacts):
1. fom3_joint_stats     — joint statistics vs the l1 baselines, both bases (full rigor)
2. invariance_ratios    — BNT/noBNT ratios incl. the grid-transport ladder
3. rescue_ladder        — recovered fraction of the l1's BNT loss, all rescue routes
"""
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = "/home/tersenov/.claude/skills/figure-polish/style/aa.mplstyle"
SBI = Path("/mnt/home/tersenov/software/cnn_sbi/scripts/sbi")
FC = SBI / "results/exploratory/flatsky_cross_2026_06"
OM = FC / "overnight_menu"
FIGS = OM / "figures"
C_NOBNT, C_BNT, C_BASE = "#0072B2", "#D55E00", "#999999"


def fom3(path):
    f = Path(path) / "median_summary.json"
    return json.load(open(f))["fom3"] if f.exists() else None


L1_AUTO = fom3(FC / "population_sweep/flat_none")            # 2405
L1_PROD = fom3(FC / "population_sweep/flat_product")         # 2875
L1_BNT = fom3(FC / "bnt_campaign/population_sweep/l1_none")  # 364


def arm(name, full=True):
    return fom3(OM / name / ("population_sweep_full" if full else "population_sweep"))


def fig_joint_stats():
    stats = [("pair2d\n(K=10)", "pair2dq"), ("joint wavelet l1\n(K=10)", "jointl1q"),
             ("full 4D (K=4)\nfixed grid", "full4dq"), ("full 4D (K=4)\nadaptive grid", "full4da")]
    labels = [s[0] for s in stats]
    nob = [arm(f"{s[1]}_nobnt") for s in stats]
    bnt = [arm(f"{s[1]}_bnt") for s in stats]
    x = np.arange(len(stats)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.5, 3.9))
    ax.bar(x - w/2, nob, w, color=C_NOBNT, edgecolor="k", lw=0.5, label="original basis")
    ax.bar(x + w/2, bnt, w, color=C_BNT, edgecolor="k", lw=0.5, label="BNT basis")
    ax.axhline(L1_AUTO, color=C_BASE, ls="--", lw=1.2)
    ax.axhline(L1_PROD, color=C_BASE, ls=":", lw=1.2)
    ax.text(len(stats) - 0.45, L1_AUTO * 1.01, "l1 auto", fontsize=8, color="0.35")
    ax.text(len(stats) - 0.45, L1_PROD * 1.01, "l1 +product", fontsize=8, color="0.35")
    for xi, (n, b) in enumerate(zip(nob, bnt)):
        ax.text(xi + w/2, b + 60, f"{b/n:.2f}", ha="center", fontsize=8, color=C_BNT)
    ax.set_ylabel(r"FoM$_3$ (pooled 3-seed 9000-obs median)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, max(L1_PROD, max(nob)) * 1.15)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.set_title("Joint one-point statistics (overnight, dequantized, full rigor)", fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"fom3_joint_stats.{ext}", dpi=200)
    plt.close(fig); print("  wrote fom3_joint_stats")


def fig_invariance():
    rows = [("pair2d", "pair2dq"), ("joint l1", "jointl1q"),
            ("full 4D\nfixed grid", "full4dq"), ("full 4D\nadaptive grid", "full4da")]
    ratios = [arm(f"{p}_bnt") / arm(f"{p}_nobnt") for _, p in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    colors = ["#56B4E9", "#56B4E9", "#E69F00", "#009E73"]
    ax.bar(x, ratios, 0.55, color=colors, edgecolor="k", lw=0.5)
    ax.axhline(1.0, color="k", lw=1.0, ls="--")
    ax.text(0.02, 1.015, "exact basis-covariance (P4b, distribution level)",
            fontsize=8, transform=ax.get_yaxis_transform())
    for xi, r in enumerate(ratios):
        ax.text(xi, r + 0.02, f"{r:.2f}", ha="center", fontsize=9)
    ax.annotate("", xy=(3, ratios[3] - 0.01), xytext=(2, ratios[2] + 0.01),
                arrowprops=dict(arrowstyle="->", color="0.3", lw=1.0))
    ax.text(2.5, (ratios[2] + ratios[3]) / 2 + 0.03, "grid\ntransport", fontsize=8,
            ha="center", color="0.3")
    ax.set_ylabel("FoM$_3$ ratio  (BNT basis / original basis)")
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_title("Binned joint estimators are only as invariant\nas their grid is transported",
                 fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"invariance_ratios.{ext}", dpi=200)
    plt.close(fig); print("  wrote invariance_ratios")


def fig_rescue_ladder():
    entries = [
        ("BNT l1 alone", L1_BNT, C_BNT),
        ("+ 2nd moments (cov50)", arm("A1_cov_bnt"), "#E69F00"),
        ("+ deep channel (avg)", fom3(FC / "bntdeep_campaign/population_sweep/l1_none"), "#CC79A7"),
        ("whitened (rotation)", fom3(FC / "whiten_campaign/population_sweep/l1_none"), "#009E73"),
        ("+ deep2 (avg + bin4)", fom3(FC / "bntdeep2_campaign/population_sweep/l1_none"), "#CC79A7"),
        ("+ 6 union channels", arm("A2_unions6_bnt"), "#56B4E9"),
    ]
    rec = [(v - L1_BNT) / (L1_AUTO - L1_BNT) for _, v, _ in entries]
    y = np.arange(len(entries))
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.barh(y, rec, 0.6, color=[e[2] for e in entries], edgecolor="k", lw=0.5)
    ax.axvline(1.0, color="k", ls="--", lw=1.0)
    ax.text(1.02, 0.25, "full recovery\n(no-BNT level)", fontsize=8, va="top")
    for yi, (r, (_, v, _)) in enumerate(zip(rec, entries)):
        ax.text(max(r, 0) + 0.02, yi, f"{r:.2f}  (FoM$_3$ {v:.0f})", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels([e[0] for e in entries], fontsize=9)
    ax.set_xlabel("recovered fraction of the l1's BNT loss")
    ax.set_xlim(0, 1.45)
    ax.invert_yaxis()
    ax.set_title("Rescuing the per-channel l1 on nulled maps — all routes", fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"rescue_ladder.{ext}", dpi=200)
    plt.close(fig); print("  wrote rescue_ladder")


def main():
    try:
        plt.style.use(STYLE)
    except OSError:
        pass
    FIGS.mkdir(parents=True, exist_ok=True)
    print(f"baselines: l1 auto {L1_AUTO:.0f} | l1+product {L1_PROD:.0f} | l1 BNT {L1_BNT:.0f}")
    fig_joint_stats()
    fig_invariance()
    fig_rescue_ladder()


if __name__ == "__main__":
    main()
