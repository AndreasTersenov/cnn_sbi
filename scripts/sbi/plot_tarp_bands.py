#!/usr/bin/env python
"""TARP-DRP with clearly visible bootstrap uncertainty bands. Two views:
 - residual (ECP − α): diagonal → flat 0 line, so the band + any deviation are obvious.
 - standard (ECP vs α) with bands, for the record.
Pools the 3 seeds × 200 bootstraps (=600) per (arm, FoM3 tercile) → median + 68% band. CPU-only.
"""
import os, glob
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CUR = "results/exploratory/definitive_comparison_10deg/phase_c/analysis/tarp_drp/curves"
OUT = "results/exploratory/definitive_comparison_10deg/phase_c/analysis/figs"
os.makedirs(OUT, exist_ok=True)
ARMS = ["l1_auto_cross", "cnn_auto_cross", "l1_auto_only", "cnn_auto_only"]
LBL = {"l1_auto_cross": "L1 a+c", "cnn_auto_cross": "CNN a+c",
       "l1_auto_only": "L1 auto", "cnn_auto_only": "CNN auto"}
TERC = ["LOW", "MID", "HIGH"]
TCOL = {"LOW": "#4daf4a", "MID": "#377eb8", "HIGH": "#e41a1c"}


def band(arm, terc, dim=3):
    boots = []
    for f in sorted(glob.glob(f"{CUR}/tarp_curve_{arm}_{terc}_seed*_dim{dim}.npz")):
        d = np.load(f); boots.append(d["ecp_bootstrap"]); alpha = d["alpha"]
    B = np.concatenate(boots, 0)  # (n_seed*200, 21)
    return alpha, np.median(B, 0), np.percentile(B, 16, 0), np.percentile(B, 84, 0)


def make(dim, residual):
    fig, ax = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    ax = ax.ravel()
    for i, arm in enumerate(ARMS):
        a = ax[i]
        for terc in TERC:
            al, med, lo, hi = band(arm, terc, dim)
            y = (med - al, lo - al, hi - al) if residual else (med, lo, hi)
            a.fill_between(al, y[1], y[2], color=TCOL[terc], alpha=0.30, lw=0)
            a.plot(al, y[0], color=TCOL[terc], lw=2, label=f"{terc} FoM3")
        if residual:
            a.axhline(0, color="k", lw=1)
            a.set_ylim(-0.12, 0.12)
        else:
            a.plot([0, 1], [0, 1], "k--", lw=1)
        a.set_title(LBL[arm]); a.grid(alpha=0.25)
        if i == 0:
            a.legend(fontsize=9, title="(68% band = 3 seeds × 200 boot)")
        if i in (2, 3):
            a.set_xlabel("nominal credibility α")
        if i in (0, 2):
            a.set_ylabel("ECP − α (calibration residual)" if residual else "expected coverage")
    kind = "residual" if residual else "coverage"
    fig.suptitle(f"TARP-DRP {dim}-D — {kind} with 68% bootstrap bands "
                 f"({'flat 0 = calibrated' if residual else 'diagonal = calibrated'}; "
                 "HIGH tercile = tight posteriors)", fontsize=12)
    fig.tight_layout()
    tag = f"{'residual' if residual else 'coverage'}_dim{dim}"
    fig.savefig(f"{OUT}/D3b_tarp_{tag}.png", dpi=130, bbox_inches="tight")
    fig.savefig(f"{OUT}/D3b_tarp_{tag}.pdf", bbox_inches="tight")
    print(f"wrote D3b_tarp_{tag}")


if __name__ == "__main__":
    make(3, residual=True)
    make(3, residual=False)
    make(6, residual=True)
