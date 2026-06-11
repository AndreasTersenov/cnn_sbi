#!/usr/bin/env python3
"""Recolor the BNT GATE-C TARP-DRP curves (preferred style, cf. plot_tarp_flatsky_colored.py):
per-arm colors + clearly visible bootstrap bands (higher alpha, per Andreas 2026-06-11).
One panel per FoM3 tercile; 4 BNT arms; median ECP + 16-84 band (3 seeds x 200 bootstraps
pooled). Diagonal = calibrated; ABOVE = conservative, BELOW = over-confident."""
import os, glob, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
G = HERE + "/results/exploratory/flatsky_cross_2026_06/bnt_campaign/gate_c/tarp_drp"
OUT = HERE + "/results/exploratory/flatsky_cross_2026_06/bnt_campaign/figures"
ARMS = [("bnt_cnn_none", "CNN auto", "#0072B2"),
        ("bnt_cnn_product", "CNN +product", "#56B4E9"),
        ("bnt_l1_none", "L1 auto", "#D55E00"),
        ("bnt_l1_product", "L1 +product", "#E69F00")]
TERC = ["LOW", "MID", "HIGH"]
BAND_ALPHA = 0.40


def curve(arm, terc, dim):
    fs = sorted(glob.glob(f"{G}/curves/tarp_curve_{arm}_{terc}_seed*_dim{dim}.npz"))
    if not fs:
        return None
    alpha, boots = None, []
    for f in fs:
        z = np.load(f)
        alpha = np.asarray(z["alpha"]).ravel()
        boots.append(np.asarray(z["ecp_bootstrap"]))
    B = np.concatenate(boots, axis=0)
    return alpha, np.median(B, 0), np.percentile(B, 16, 0), np.percentile(B, 84, 0)


def make(dim, fname):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True, sharey=True)
    for ax, terc in zip(axes, TERC):
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7, zorder=1)
        for arm, lab, col in ARMS:
            c = curve(arm, terc, dim)
            if c is None:
                continue
            a, med, lo, hi = c
            ax.fill_between(a, lo, hi, color=col, alpha=BAND_ALPHA, lw=0, zorder=2)
            ax.plot(a, med, color=col, lw=2.0, label=lab, zorder=3)
        ax.set_title(f"{terc}-FoM3 tercile", fontsize=12)
        ax.set_xlabel(r"nominal credibility $\alpha$")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.25)
    axes[0].set_ylabel("expected coverage")
    axes[2].legend(loc="upper left", fontsize=10, framealpha=0.9)
    ttl = r"$\Omega_m,\sigma_8,w_0$" if dim == 3 else "all 6 params"
    fig.suptitle(f"GATE C (BNT arms) — TARP-DRP, {ttl} "
                 "(band = 16–84% over 3 seeds × 200 bootstraps)", fontsize=13)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{fname}.{ext}", dpi=180)
    plt.close(fig)
    print(f"  wrote {OUT}/{fname}.{{png,pdf}}")


def make_full(dim, fname):
    """Un-split (all 600 val points) TARP per arm. The three terciles are equal-size
    (n=200 each), so the full-set ECP is EXACTLY the mean of the tercile ECPs; the band
    combines per-bootstrap-draw means (independent terciles => variance matches a
    600-point bootstrap to good approximation)."""
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7, zorder=1)
    for arm, lab, col in ARMS:
        per_terc, alpha = [], None
        for terc in TERC:
            fs = sorted(glob.glob(f"{G}/curves/tarp_curve_{arm}_{terc}_seed*_dim{dim}.npz"))
            if not fs:
                break
            boots = []
            for f in fs:
                z = np.load(f)
                alpha = np.asarray(z["alpha"]).ravel()
                boots.append(np.asarray(z["ecp_bootstrap"]))
            per_terc.append(np.concatenate(boots, axis=0))   # (3 seeds * 200, 21)
        if len(per_terc) != 3:
            continue
        n = min(p.shape[0] for p in per_terc)
        full = np.mean([p[:n] for p in per_terc], axis=0)    # per-draw mean over terciles
        ax.fill_between(alpha, np.percentile(full, 16, 0), np.percentile(full, 84, 0),
                        color=col, alpha=BAND_ALPHA, lw=0, zorder=2)
        ax.plot(alpha, np.median(full, 0), color=col, lw=2.2, label=lab, zorder=3)
    ax.set_xlabel(r"nominal credibility $\alpha$")
    ax.set_ylabel("expected coverage")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ttl = r"$\Omega_m,\sigma_8,w_0$" if dim == 3 else "all 6 params"
    ax.set_title(f"GATE C (BNT arms) — TARP-DRP, {ttl}, all 600 val points", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{fname}.{ext}", dpi=180)
    plt.close(fig)
    print(f"  wrote {OUT}/{fname}.{{png,pdf}}")


if __name__ == "__main__":
    make(3, "tarp_bnt_colored_dim3")
    make(6, "tarp_bnt_colored_dim6")
    make_full(3, "tarp_bnt_colored_full_dim3")
    make_full(6, "tarp_bnt_colored_full_dim6")
