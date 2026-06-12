#!/usr/bin/env python3
"""σ8-coded JOINT-statistic datavectors (the analog of l1_hist_vs_s8_viridis for the
overnight pair2d / jointl1 arms, both bases).

Two views per statistic×basis, from the on-disk training caches (dequantized builds; the
+U(0,1) is ≤1 per cell vs ~60 typical — invisible at this scale):
  1. grid of curves — rows = pairs (1×2, 1×4, 3×4), cols = scales 0–4; x = flattened 10×10
     cell index (row-major; light gridlines every K mark the 2D rows); 16 σ8-quantile
     levels, viridis.
  2. native 2D view — mean 10×10 histogram for the lowest/highest σ8 level + difference,
     pair 3×4, scale 3 (the most responsive in the l1 reference figure), nobnt vs bnt rows.
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SBI = Path("/mnt/home/tersenov/software/cnn_sbi/scripts/sbi")
OM = SBI / "results/exploratory/flatsky_cross_2026_06/overnight_menu"
FIGS = OM / "figures"
K, NS = 10, 5
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]   # fx.cross_pairs(4) order
SHOW_PAIRS = [(0, (0, 1)), (2, (0, 3)), (5, (2, 3))]        # 1x2, 1x4, 3x4
N_LEVELS = 16


def levels_mean(x, s8):
    qs = np.quantile(s8, np.linspace(0, 1, N_LEVELS + 1))
    out, centers = [], []
    for i in range(N_LEVELS):
        m = (s8 >= qs[i]) & (s8 <= qs[i + 1] if i == N_LEVELS - 1 else s8 < qs[i + 1])
        out.append(x[m].mean(axis=0))
        centers.append(0.5 * (qs[i] + qs[i + 1]))
    return np.array(out), np.array(centers)


def fig_curves(name, title):
    z = np.load(OM / name / "cache" / "l1_train.npz")
    x = z["x"]; s8 = z["theta"][:, 1].astype(np.float64)
    means, centers = levels_mean(x, s8)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(centers.min(), centers.max())
    fig, axes = plt.subplots(len(SHOW_PAIRS), NS, figsize=(16, 8), sharex=True)
    for r, (pi, (i, j)) in enumerate(SHOW_PAIRS):
        for s in range(NS):
            ax = axes[r, s]
            base = (pi * NS + s) * K * K
            for lv in range(N_LEVELS):
                ax.plot(means[lv, base:base + K * K], color=cmap(norm(centers[lv])),
                        lw=0.8)
            for g in range(K, K * K, K):
                ax.axvline(g, color="0.85", lw=0.4, zorder=0)
            if r == 0:
                ax.set_title(f"scale {s}", fontsize=11)
            if s == 0:
                ax.set_ylabel(f"pair $\\kappa_{i+1}\\times\\kappa_{j+1}$", fontsize=10)
            if r == len(SHOW_PAIRS) - 1:
                ax.set_xlabel("cell index (10×10, row-major)", fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=axes, label=r"$\sigma_8$", fraction=0.025, pad=0.01)
    fig.suptitle(title, fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"jointdv_{name}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote jointdv_{name}")
    return means, centers


def fig_heatmaps(stat):
    pi, (i, j) = 5, (2, 3)   # pair 3x4
    s = 3
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6))
    for r, basis in enumerate(("nobnt", "bnt")):
        z = np.load(OM / f"{stat}_{basis}" / "cache" / "l1_train.npz")
        x = z["x"]; s8 = z["theta"][:, 1].astype(np.float64)
        means, centers = levels_mean(x, s8)
        base = (pi * NS + s) * K * K
        lo = means[0, base:base + K * K].reshape(K, K)
        hi = means[-1, base:base + K * K].reshape(K, K)
        vmax = max(lo.max(), hi.max())
        for c, (img, ttl, cm, vmin, vmx) in enumerate((
                (lo, f"low $\\sigma_8$ ({centers[0]:.2f})", "magma", 0, vmax),
                (hi, f"high $\\sigma_8$ ({centers[-1]:.2f})", "magma", 0, vmax),
                (hi - lo, "high − low", "RdBu_r", -np.abs(hi - lo).max(),
                 np.abs(hi - lo).max()))):
            ax = axes[r, c]
            im = ax.imshow(img, origin="lower", cmap=cm, vmin=vmin, vmax=vmx,
                           extent=[0, K, 0, K])
            ax.set_title(("" if r else f"{ttl}\n") + (ttl if r else ""), fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{basis.upper()}\nSNR bin ($\\kappa_{i+1}$)", fontsize=10)
            ax.set_xlabel(f"SNR bin ($\\kappa_{j+1}$)", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{stat}: mean joint histogram, pair $\\kappa_3\\times\\kappa_4$, "
                 f"scale 3 — low vs high $\\sigma_8$", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"jointdv_2d_{stat}.{ext}", dpi=170)
    plt.close(fig)
    print(f"  wrote jointdv_2d_{stat}")


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    for stat, pretty in (("pair2dq", "joint PDF (counts)"), ("jointl1q", "joint wavelet $\\ell_1$")):
        for basis in ("nobnt", "bnt"):
            fig_curves(f"{stat}_{basis}",
                       f"{pretty} vs $\\sigma_8$ — 16 levels, {basis.upper()} basis "
                       f"(dequantized training cache)")
    fig_heatmaps("pair2dq")
    fig_heatmaps("jointl1q")


if __name__ == "__main__":
    main()
