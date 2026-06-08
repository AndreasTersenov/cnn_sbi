#!/usr/bin/env python
"""Re-plot the TARP-DRP per-arm + overlay figures (ECP vs α, same as run_tarp_coverage)
but with distinct colors and higher-alpha (opaque) bootstrap bands. Pools 3 seeds × 200
bootstraps for smooth 68% bands. CPU-only."""
import os, glob
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CUR = "results/exploratory/definitive_comparison_10deg/phase_c/analysis/tarp_drp/curves"
OUT = "results/exploratory/definitive_comparison_10deg/phase_c/analysis/tarp_drp/figures"
ARMS = ["l1_auto_cross", "cnn_auto_cross", "l1_auto_only", "cnn_auto_only"]
DISP = {"l1_auto_cross": "L1 auto+cross", "cnn_auto_cross": "CNN auto+cross",
        "l1_auto_only": "L1 auto-only", "cnn_auto_only": "CNN auto-only"}
ARMC = {"l1_auto_cross": "#2ca02c", "cnn_auto_cross": "#1f77b4",
        "l1_auto_only": "#ff7f0e", "cnn_auto_only": "#d62728"}
TERC = ["LOW", "MID", "HIGH"]
TERCC = {"LOW": "#1b9e77", "MID": "#d95f02", "HIGH": "#7570b3"}
BAND_ALPHA = 0.40


def boots(arm, terc, dim):
    out = []
    for f in sorted(glob.glob(f"{CUR}/tarp_curve_{arm}_{terc}_seed*_dim{dim}.npz")):
        d = np.load(f); out.append(d["ecp_bootstrap"]); al = d["alpha"]
    return al, np.concatenate(out, 0)  # (n_seed*200, 21)


def stat(B):
    return np.median(B, 0), np.percentile(B, 16, 0), np.percentile(B, 84, 0)


def per_arm(dim):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.2), constrained_layout=True)
    for ax, arm in zip(axes.flat, ARMS):
        for terc in TERC:
            al, B = boots(arm, terc, dim)
            med, lo, hi = stat(B)
            ax.fill_between(al, lo, hi, color=TERCC[terc], alpha=BAND_ALPHA, lw=0)
            ax.plot(al, med, color=TERCC[terc], lw=2.2, label=f"{terc} FoM3 tercile")
        ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.7)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
        ax.set_xlabel("Nominal credibility α"); ax.set_ylabel("Expected coverage")
        ax.set_title(DISP[arm]); ax.legend(loc="lower right", fontsize=9)
    fig.suptitle(f"TARP-DRP per arm — {dim}-D ({'Ωm,σ8,w0' if dim==3 else 'all 6 params'}); "
                 "68% bootstrap band (3 seeds × 200)", fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/tarp_per_arm_dim{dim}_colored.{ext}", dpi=150)
    print(f"wrote tarp_per_arm_dim{dim}")


def overlay(dim):
    """4 arms, each pooling all FoM3 terciles → one curve + band per arm."""
    fig, ax = plt.subplots(figsize=(6.6, 6.0), constrained_layout=True)
    for arm in ARMS:
        Bs = []
        for terc in TERC:
            al, B = boots(arm, terc, dim); Bs.append(B)
        med, lo, hi = stat(np.concatenate(Bs, 0))
        ax.fill_between(al, lo, hi, color=ARMC[arm], alpha=0.28, lw=0)
        ax.plot(al, med, color=ARMC[arm], lw=2.4, label=DISP[arm])
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.7)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
    ax.set_xlabel("Nominal credibility α"); ax.set_ylabel("Expected coverage")
    ax.set_title(f"TARP joint coverage — {dim}-D ({'Ωm,σ8,w0' if dim==3 else 'all 6 params'})")
    ax.legend(loc="lower right", fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/tarp_overlay_dim{dim}_colored.{ext}", dpi=150)
    print(f"wrote tarp_overlay_dim{dim}")


if __name__ == "__main__":
    for dim in (3, 6):
        per_arm(dim); overlay(dim)
