#!/usr/bin/env python3
"""L-C2ST local-calibration plot for the flat-local CNN arms (built from saved lc2st_results.npz).

Per arm: the permutation-null distribution of the L-C2ST statistic (grey) vs the 30 observed
statistics T_obs at typical fiducial obs (coloured rug), with the p<0.05 reject threshold (95th
null percentile). Statistics within the null = LOCALLY CALIBRATED; shifted right = miscalibrated.
Annotated with frac_reject(p<0.05) + median p + the self-test power gate (ST_H1).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = "/home/tersenov/.claude/skills/figure-polish/style/aa.mplstyle"
CNNP = Path("/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/flatsky_cross_2026_06/cnn_phase")
ARMS = [("none", "auto-only"), ("conv", "+conv"), ("product", "+product"), ("both", "+both")]
COL = {"none": "#555555", "conv": "#56B4E9", "product": "#009E73", "both": "#CC79A7"}


def main():
    plt.style.use(STYLE)
    plt.rcParams["figure.constrained_layout.use"] = True
    fig, axes = plt.subplots(2, 2, figsize=(7.087, 5.0), sharex=False)
    for ax, (op, lab) in zip(axes.ravel(), ARMS):
        base = CNNP / "gate_c/lc2st" / f"flat_{op}" / f"flat_{op}"
        d = np.load(base / "lc2st_results.npz")
        s = json.load(open(base / "lc2st_summary.json"))
        T_obs, T_null = d["T_obs"], d["T_null"].ravel()
        thr = np.percentile(T_null, 95)               # p<0.05 reject threshold
        ax.hist(T_null, bins=40, density=True, color="0.80", label="permutation null")
        # observed statistics as a rug + their median
        y = ax.get_ylim()[1]
        ax.plot(T_obs, np.full_like(T_obs, 0.04 * y), "|", color=COL[op], ms=11, mew=1.4,
                label=r"observed $T(x_0)$ (30 obs)")
        ax.axvline(np.median(T_obs), color=COL[op], lw=1.6)
        ax.axvline(thr, color="k", ls="--", lw=1.0, label="p=0.05 threshold")
        ax.set_title(f"{lab}", fontsize=10)
        ax.text(0.97, 0.95,
                f"reject@p<0.05: {s['frac_reject_p05']*100:.0f}%\nmedian p = {s['median_p']:.2f}\n"
                f"self-test ST$_{{H1}}$ p = {s['gate']['st_h1_median_p']:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.6))
        ax.set_yticks([])
        if op in ("product", "both"):
            ax.set_xlabel("L-C2ST statistic $T$")
        ax.legend(fontsize=7, loc="center right", frameon=False)
    fig.suptitle("GATE C (CNN) — L-C2ST local calibration at the fiducial", fontsize=11)
    out = CNNP / "gate_c/lc2st"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"lc2st_cnn.{ext}", dpi=200)
    print(f"wrote {out}/lc2st_cnn.{{png,pdf}}")


if __name__ == "__main__":
    main()
