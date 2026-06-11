#!/usr/bin/env python3
"""L1 'both' (16-channel) datavectors vs sigma8, no-BNT (top) vs BNT (bottom).

Sibling of plot_flatsky_diagnostics.py Fig 2, per Andreas 2026-06-11: datavectors for
different cosmologies COLOR-CODED BY sigma8 (continuous colormap over sigma8-quantile-bin
means), in the BNT basis, next to the familiar no-BNT version. Reads the EXACT inference
datavectors from the training caches on disk (frozen sigma + calibrated ranges) — no
recompute. CPU-only."""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

SBI = Path("/mnt/home/tersenov/software/cnn_sbi/scripts/sbi")
FC = SBI / "results/exploratory/flatsky_cross_2026_06"
CACHES = [
    ("no BNT", FC / "l1_matrix/l1_both_cache/flat_local_both/l1_train.npz"),
    ("BNT", FC / "bnt_campaign/l1_matrix/l1_both_cache/flat_local_both_bnt/l1_train.npz"),
]
OUT = FC / "bnt_campaign/figures"
NSC, L1N, NCH = 5, 40, 16
FEAT = NSC * L1N
BLOCKS = [(0, 4, "auto (4 ch)", "#1f77b4"), (4, 10, "conv (6 ch)", "#2ca02c"),
          (10, 16, "product (6 ch)", "#d62728")]
NQ = 10   # sigma8 quantile bins
CMAP = plt.get_cmap("coolwarm")


def binned_means(npz_path):
    z = np.load(npz_path)
    s8 = z["theta"][:, 1].astype(np.float64)
    X = z["x"]
    edges = np.quantile(s8, np.linspace(0, 1, NQ + 1))
    means, mids = [], []
    for i in range(NQ):
        m = (s8 >= edges[i]) & (s8 <= edges[i + 1] if i == NQ - 1 else s8 < edges[i + 1])
        means.append(X[m].mean(axis=0))
        mids.append(0.5 * (edges[i] + edges[i + 1]))
    return np.array(means), np.array(mids)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    norm = None
    for ax, (lab, path) in zip(axes, CACHES):
        means, mids = binned_means(path)
        if norm is None:
            norm = Normalize(vmin=mids.min(), vmax=mids.max())
        for m, s8 in zip(means, mids):
            ax.plot(m, color=CMAP(norm(s8)), lw=0.9, alpha=0.9)
        for c0, c1, blab, bcol in BLOCKS:
            ax.axvspan(c0 * FEAT, c1 * FEAT, color=bcol, alpha=0.05)
            ax.text(0.5 * (c0 + c1) * FEAT, ax.get_ylim()[1] * 0.97, blab,
                    ha="center", va="top", fontsize=9, color=bcol)
        for c in range(1, NCH):
            ax.axvline(c * FEAT, color="k", lw=0.4, alpha=0.25)
        ax.set_ylabel(f"L1 datavector — {lab}")
        ax.set_xlim(0, NCH * FEAT)
    axes[1].set_xlabel(f"datavector index (16 channels × {NSC} scales × {L1N} SNR bins; "
                       "auto | conv | product)")
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), ax=axes, fraction=0.025, pad=0.01)
    cb.set_label(r"$\sigma_8$ (quantile-bin mean)")
    fig.suptitle(r"L1 'both' datavectors vs $\sigma_8$ — original vs BNT basis "
                 "(frozen per-basis $\\sigma$ + calibrated ranges)", fontsize=12)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"datavectors_bnt_vs_nobnt_s8.{ext}", dpi=180)
    print(f"wrote {OUT}/datavectors_bnt_vs_nobnt_s8.{{png,pdf}}")
    plt.close(fig)

    # Relative version: per-bin mean / global mean - 1 => the sigma8 RESPONSE of each
    # datavector entry, directly comparable across bases (the information story).
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, sharey=True)
    norm = None
    for ax, (lab, path) in zip(axes, CACHES):
        means, mids = binned_means(path)
        if norm is None:
            norm = Normalize(vmin=mids.min(), vmax=mids.max())
        ref = means.mean(axis=0)
        ref = np.where(np.abs(ref) > 1e-12, ref, 1.0)
        for m, s8 in zip(means, mids):
            ax.plot(m / ref - 1.0, color=CMAP(norm(s8)), lw=0.8, alpha=0.85)
        ax.axhline(0, color="k", lw=0.6)
        for c0, c1, blab, bcol in BLOCKS:
            ax.axvspan(c0 * FEAT, c1 * FEAT, color=bcol, alpha=0.05)
            ax.text(0.5 * (c0 + c1) * FEAT, 0.95, blab, ha="center", va="top",
                    fontsize=9, color=bcol, transform=ax.get_xaxis_transform())
        for c in range(1, NCH):
            ax.axvline(c * FEAT, color="k", lw=0.4, alpha=0.25)
        ax.set_ylabel(f"fractional deviation — {lab}")
        ax.set_xlim(0, NCH * FEAT)
    axes[1].set_xlabel(f"datavector index (16 channels × {NSC} scales × {L1N} SNR bins)")
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), ax=axes, fraction=0.025, pad=0.01)
    cb.set_label(r"$\sigma_8$ (quantile-bin mean)")
    fig.suptitle(r"$\sigma_8$ response of the L1 datavector — original vs BNT basis "
                 "(bin mean / global mean − 1)", fontsize=12)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"datavectors_bnt_vs_nobnt_s8_relative.{ext}", dpi=180)
    print(f"wrote {OUT}/datavectors_bnt_vs_nobnt_s8_relative.{{png,pdf}}")
    plt.close(fig)

    hist_grid()


def hist_grid(nlev: int = 16):
    """BNT twin of figs/l1_hist_vs_s8_viridis: rows = (auto bin4, conv 3x4, prod 3x4),
    cols = 5 scales; each panel = the 40-SNR-bin L1 histogram at `nlev` sigma8-quantile
    levels, viridis-coded. Exact inference datavectors from the BNT both-cache."""
    rows = [("auto κ4 (BNT)", 3), ("conv 3×4 (BNT)", 9), ("product 3×4 (BNT)", 15)]
    z = np.load(CACHES[1][1])     # the BNT both-cache
    s8 = z["theta"][:, 1].astype(np.float64)
    X = z["x"]
    edges = np.quantile(s8, np.linspace(0, 1, nlev + 1))
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=s8.min(), vmax=s8.max())
    fig, axes = plt.subplots(3, NSC, figsize=(15.5, 7.2))
    for r, (rlab, ch) in enumerate(rows):
        for sc in range(NSC):
            ax = axes[r, sc]
            cols = ch * FEAT + sc * L1N + np.arange(L1N)
            for i in range(nlev):
                m = (s8 >= edges[i]) & (s8 <= edges[i + 1] if i == nlev - 1 else s8 < edges[i + 1])
                mid = 0.5 * (edges[i] + edges[i + 1])
                ax.plot(X[m][:, cols].mean(axis=0), color=cmap(norm(mid)), lw=1.1)
            if r == 0:
                ax.set_title(f"scale {sc}", fontsize=11)
            if sc == 0:
                ax.set_ylabel(rlab, fontsize=11)
            if r == 2:
                ax.set_xlabel("SNR bin")
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, fraction=0.02, pad=0.01)
    cb.set_label(r"$\sigma_8$")
    fig.suptitle(f"L1 histograms vs σ8 — {nlev} cosmology levels "
                 "(op=both, exact datavector, BNT basis)", fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"l1_hist_vs_s8_viridis_bnt.{ext}", dpi=170)
    print(f"wrote {OUT}/l1_hist_vs_s8_viridis_bnt.{{png,pdf}}")


if __name__ == "__main__":
    main()
