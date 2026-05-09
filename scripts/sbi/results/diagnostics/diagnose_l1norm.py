#!/usr/bin/env python
"""
Diagnostic script: visualize L1-norm summary statistics and diagnose NaN flow training.

Loads the cached L1-norm data and produces diagnostic plots:
1. Raw L1-norm distributions (per scale, per tomo bin)
2. Effect of log1p transform
3. Standardized distributions (before/after log1p)
4. Feature correlation structure

Usage:
    cd scripts/sbi
    python diagnose_l1norm.py --cache-dir ./cache_l1
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default="./cache_l1")
    parser.add_argument("--out-dir", type=str, default="./fig/l1norm_diagnostics")
    parser.add_argument("--n-scales", type=int, default=5)
    parser.add_argument("--l1-nbins", type=int, default=40)
    parser.add_argument("--nbins", type=int, default=4)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load data
    d_tr = np.load(f"{args.cache_dir}/l1_train.npz")
    x_raw = d_tr["x"]    # (N, 800)
    theta = d_tr["theta"] # (N, 6)
    N, D = x_raw.shape
    print(f"Loaded {N} training samples, {D} features")

    n_scales = args.n_scales
    l1_nbins = args.l1_nbins
    nbins = args.nbins
    features_per_bin = n_scales * l1_nbins  # 200

    # =========================================================================
    # PLOT 1: Raw L1-norm values per scale and tomo bin
    # =========================================================================
    fig, axes = plt.subplots(nbins, n_scales, figsize=(4 * n_scales, 3 * nbins),
                              sharex=False, sharey=False)
    fig.suptitle("Raw L1-norm: distribution of sum across histogram bins\n"
                 "(each subplot = one wavelet scale, one tomo bin)", fontsize=12)

    for tomo_bin in range(nbins):
        for scale in range(n_scales):
            ax = axes[tomo_bin, scale]
            # Features for this (tomo_bin, scale): sum over l1 histogram bins
            start = tomo_bin * features_per_bin + scale * l1_nbins
            end = start + l1_nbins
            feature_block = x_raw[:, start:end]  # (N, l1_nbins)

            # Sum across histogram bins for a scalar per map
            total_l1 = feature_block.sum(axis=1)
            ax.hist(total_l1, bins=80, alpha=0.7, color=f"C{tomo_bin}", density=True)
            ax.set_title(f"Bin {tomo_bin+1}, Scale {scale+1}", fontsize=9)
            if scale == 0:
                ax.set_ylabel("Density")
            if tomo_bin == nbins - 1:
                ax.set_xlabel("Total L1")

    plt.tight_layout()
    plt.savefig(out / "raw_l1_per_scale_tomobin.png", dpi=120)
    plt.close()
    print(f"Saved: {out / 'raw_l1_per_scale_tomobin.png'}")

    # =========================================================================
    # PLOT 2: Raw L1-norm histogram features for one example map
    # =========================================================================
    fig, axes = plt.subplots(nbins, n_scales, figsize=(4 * n_scales, 3 * nbins))
    fig.suptitle("L1-norm histogram features for 3 example maps\n"
                 "(x-axis = histogram bin index, y-axis = L1 value)", fontsize=12)

    example_indices = [0, N // 2, N - 1]
    for tomo_bin in range(nbins):
        for scale in range(n_scales):
            ax = axes[tomo_bin, scale]
            start = tomo_bin * features_per_bin + scale * l1_nbins
            end = start + l1_nbins

            for color_idx, ex_idx in enumerate(example_indices):
                ax.plot(x_raw[ex_idx, start:end], alpha=0.7, label=f"Map {ex_idx}",
                        color=f"C{color_idx}")

            ax.set_title(f"Bin {tomo_bin+1}, Scale {scale+1}", fontsize=9)
            if scale == 0:
                ax.set_ylabel("L1 value")
            if tomo_bin == 0 and scale == 0:
                ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out / "l1_histograms_examples.png", dpi=120)
    plt.close()
    print(f"Saved: {out / 'l1_histograms_examples.png'}")

    # =========================================================================
    # PLOT 3: Standardization comparison — mean/std vs log1p+mean/std
    # =========================================================================
    # Method A: plain standardization
    mean_a = x_raw.mean(axis=0)
    std_a = x_raw.std(axis=0)
    std_a[std_a < 1e-12] = 1.0
    x_std_a = (x_raw - mean_a) / std_a

    # Method B: log1p then standardize
    x_log = np.log1p(x_raw)
    mean_b = x_log.mean(axis=0)
    std_b = x_log.std(axis=0)
    std_b[std_b < 1e-12] = 1.0
    x_std_b = (x_log - mean_b) / std_b

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Standardization comparison: plain vs log1p", fontsize=14)

    # Row 0: plain standardization
    # Histogram of all standardized values
    axes[0, 0].hist(x_std_a.ravel(), bins=200, range=(-5, 15), alpha=0.7, density=True)
    axes[0, 0].set_title("Plain: all feature values")
    axes[0, 0].set_xlabel("Standardized value")
    axes[0, 0].axvline(5, color="red", ls="--", label="|x|=5")
    axes[0, 0].legend()

    # Max |x| per sample
    max_abs_a = np.abs(x_std_a).max(axis=1)
    axes[0, 1].hist(max_abs_a, bins=100, range=(0, 50), alpha=0.7, density=True)
    axes[0, 1].set_title(f"Plain: max|x| per sample\nmedian={np.median(max_abs_a):.1f}")
    axes[0, 1].set_xlabel("max|standardized feature|")

    # Feature-wise std after standardization (should all be 1)
    axes[0, 2].hist(x_std_a.std(axis=0), bins=50, alpha=0.7)
    axes[0, 2].set_title("Plain: per-feature std (should be 1)")

    # Row 1: log1p standardization
    axes[1, 0].hist(x_std_b.ravel(), bins=200, range=(-5, 5), alpha=0.7, density=True,
                     color="C1")
    axes[1, 0].set_title("Log1p: all feature values")
    axes[1, 0].set_xlabel("Standardized value")
    axes[1, 0].axvline(5, color="red", ls="--", label="|x|=5")
    axes[1, 0].legend()

    max_abs_b = np.abs(x_std_b).max(axis=1)
    axes[1, 1].hist(max_abs_b, bins=100, range=(0, 10), alpha=0.7, density=True,
                     color="C1")
    axes[1, 1].set_title(f"Log1p: max|x| per sample\nmedian={np.median(max_abs_b):.1f}")
    axes[1, 1].set_xlabel("max|standardized feature|")

    axes[1, 2].hist(x_std_b.std(axis=0), bins=50, alpha=0.7, color="C1")
    axes[1, 2].set_title("Log1p: per-feature std (should be 1)")

    plt.tight_layout()
    plt.savefig(out / "standardization_comparison.png", dpi=120)
    plt.close()
    print(f"Saved: {out / 'standardization_comparison.png'}")

    # Print stats
    print("\n=== Standardization stats ===")
    print(f"Plain  : max|x|={np.abs(x_std_a).max():.1f}, "
          f"median(max|x|)={np.median(max_abs_a):.1f}, "
          f"frac(max|x|>5)={100*(max_abs_a>5).mean():.1f}%")
    print(f"Log1p  : max|x|={np.abs(x_std_b).max():.1f}, "
          f"median(max|x|)={np.median(max_abs_b):.1f}, "
          f"frac(max|x|>5)={100*(max_abs_b>5).mean():.1f}%")

    # =========================================================================
    # PLOT 4: Feature correlation matrix (log1p standardized)
    # =========================================================================
    # Subsample for speed
    n_sub = min(5000, N)
    idx = np.random.choice(N, n_sub, replace=False)
    corr = np.corrcoef(x_std_b[idx].T)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Mark tomo bin boundaries
    for i in range(1, nbins):
        pos = i * features_per_bin
        ax.axhline(pos - 0.5, color="black", lw=1)
        ax.axvline(pos - 0.5, color="black", lw=1)

    # Mark scale boundaries within each tomo bin
    for tomo in range(nbins):
        for s in range(1, n_scales):
            pos = tomo * features_per_bin + s * l1_nbins
            ax.axhline(pos - 0.5, color="gray", lw=0.3, ls="--")
            ax.axvline(pos - 0.5, color="gray", lw=0.3, ls="--")

    ax.set_title("Feature correlation matrix (log1p standardized)\n"
                 "Black lines = tomo bin boundaries, gray = scale boundaries")
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Feature index")
    plt.tight_layout()
    plt.savefig(out / "correlation_matrix.png", dpi=120)
    plt.close()
    print(f"Saved: {out / 'correlation_matrix.png'}")

    # =========================================================================
    # PLOT 5: L1-norm mean profile per scale (averaged over maps)
    # =========================================================================
    fig, axes = plt.subplots(1, nbins, figsize=(4 * nbins, 4), sharey=True)
    fig.suptitle("Mean L1-norm profile per wavelet scale (averaged over all maps)", fontsize=12)

    for tomo_bin in range(nbins):
        ax = axes[tomo_bin]
        for scale in range(n_scales):
            start = tomo_bin * features_per_bin + scale * l1_nbins
            end = start + l1_nbins
            mean_profile = x_raw[:, start:end].mean(axis=0)
            ax.plot(mean_profile, label=f"Scale {scale+1}", color=f"C{scale}")
        ax.set_title(f"Tomo bin {tomo_bin+1}")
        ax.set_xlabel("L1 histogram bin index")
        if tomo_bin == 0:
            ax.set_ylabel("Mean L1 value")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out / "mean_l1_profiles.png", dpi=120)
    plt.close()
    print(f"Saved: {out / 'mean_l1_profiles.png'}")

    # =========================================================================
    # PLOT 6: Sensitivity to cosmology — L1-norm vs Omega_m and sigma_8
    # =========================================================================
    fig, axes = plt.subplots(2, n_scales, figsize=(4 * n_scales, 6))
    fig.suptitle("L1-norm total vs cosmological parameters (tomo bin 1)", fontsize=12)

    n_show = min(5000, N)
    idx = np.random.choice(N, n_show, replace=False)

    for scale in range(n_scales):
        start = 0 * features_per_bin + scale * l1_nbins
        end = start + l1_nbins
        total_l1 = x_raw[idx, start:end].sum(axis=1)

        ax0 = axes[0, scale]
        sc0 = ax0.scatter(theta[idx, 0], total_l1, c=theta[idx, 1], s=1, alpha=0.3, cmap="viridis")
        ax0.set_xlabel(r"$\Omega_m$")
        ax0.set_ylabel("Total L1")
        ax0.set_title(f"Scale {scale+1}")
        if scale == n_scales - 1:
            plt.colorbar(sc0, ax=ax0, label=r"$\sigma_8$")

        ax1 = axes[1, scale]
        sc1 = ax1.scatter(theta[idx, 1], total_l1, c=theta[idx, 0], s=1, alpha=0.3, cmap="plasma")
        ax1.set_xlabel(r"$\sigma_8$")
        ax1.set_ylabel("Total L1")
        if scale == n_scales - 1:
            plt.colorbar(sc1, ax=ax1, label=r"$\Omega_m$")

    plt.tight_layout()
    plt.savefig(out / "l1_vs_cosmology.png", dpi=120)
    plt.close()
    print(f"Saved: {out / 'l1_vs_cosmology.png'}")

    print(f"\nAll plots saved to {out.resolve()}")


if __name__ == "__main__":
    main()
