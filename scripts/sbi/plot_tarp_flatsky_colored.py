#!/usr/bin/env python3
"""Recolor the flat-local GATE-C TARP-DRP curves: per-arm colors + visible bootstrap bands.

One panel per FoM3 tercile (LOW/MID/HIGH); 4 arms colored (none/conv/product/both, matching the
corner plot); mean ECP line + filled 16-84 bootstrap band (3 seeds x 200 bootstraps pooled) at
higher alpha. Diagonal = perfect calibration; band ABOVE = conservative, BELOW = over-confident.
"""
import os, glob, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
G = HERE + "/results/exploratory/flatsky_cross_2026_06/gate_c/tarp_drp"
ARMS = [("flat_none", "auto-only", "#555555"), ("flat_conv", "+conv", "#1f77b4"),
        ("flat_product", "+product", "#2ca02c"), ("flat_both", "+both", "#d62728")]
TERC = ["LOW", "MID", "HIGH"]


def curve(arm, terc, dim):
    fs = sorted(glob.glob(f"{G}/curves/tarp_curve_{arm}_{terc}_seed*_dim{dim}.npz"))
    if not fs:
        return None
    alpha = None
    boots = []
    for f in fs:
        z = np.load(f)
        alpha = np.asarray(z["alpha"]).ravel()
        boots.append(np.asarray(z["ecp_bootstrap"]))   # (200, 21)
    B = np.concatenate(boots, axis=0)                  # (3*200, 21) pooled over seeds
    return alpha, np.median(B, 0), np.percentile(B, 16, 0), np.percentile(B, 84, 0)


def make(dim, fname):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True, sharey=True)
    for ax, terc in zip(axes, TERC):
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7, zorder=1)
        for arm, lab, col in ARMS:
            c = curve(arm, terc, dim)
            if c is None:
                continue
            alpha, med, lo, hi = c
            ax.fill_between(alpha, lo, hi, color=col, alpha=0.28, lw=0, zorder=2)
            ax.plot(alpha, med, color=col, lw=2.0, label=lab, zorder=3)
        ax.set_title(f"{terc}-FoM3 tercile", fontsize=12)
        ax.set_xlabel("nominal credibility α")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.25)
    axes[0].set_ylabel("expected coverage")
    axes[2].legend(loc="upper left", fontsize=10, framealpha=0.9)
    ttl = "Ω_m,σ_8,w_0" if dim == 3 else "all 6 params"
    fig.suptitle(f"GATE C TARP-DRP — {ttl}  (600 held-out val obs; band = 16-84 bootstrap, 3 seeds)",
                 fontsize=13)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{G}/figures/{fname}.{ext}", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  wrote {G}/figures/{fname}.{{png,pdf}}")


if __name__ == "__main__":
    make(3, "tarp_colored_dim3")
    make(6, "tarp_colored_dim6")
