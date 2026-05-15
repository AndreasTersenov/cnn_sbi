#!/usr/bin/env python
"""Diagnostics for the cross-only campaign inputs.

Produces two figures under
`scripts/sbi/results/exploratory/cross_only_campaign/diagnostics/`:

1. `cnn_training_example_cross_maps.{pdf,png}` — the 6 cross maps that the
   CNN sees as a single training example (one realization), one subplot per
   channel.
2. `l1_histograms_cross_channels.{pdf,png}` — the L1 wavelet histograms for
   the same training-example-equivalent: per (cross channel, scale) panel,
   one realization's histogram overlaid on the mean ± std band across many
   cosmologies.

The plots use the production-calibrated SNR range and bin count from the
seed=41 L1 cache meta, so they reflect exactly what the trained flow saw.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/mnt/home/tersenov/software/cnn_sbi")
HARM_CACHE = REPO / "scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid"
L1_CACHE = REPO / "scripts/sbi/results/exploratory/cross_only_campaign/l1_cross_only/seed_41/cache"
OUT = REPO / "scripts/sbi/results/exploratory/cross_only_campaign/diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

CROSS_LABELS = [
    r"$\kappa_1 \times \kappa_2$", r"$\kappa_1 \times \kappa_3$",
    r"$\kappa_1 \times \kappa_4$", r"$\kappa_2 \times \kappa_3$",
    r"$\kappa_2 \times \kappa_4$", r"$\kappa_3 \times \kappa_4$",
]

# ---------------------------------------------------------------------------
# 1) CNN training example: 6 cross maps
# ---------------------------------------------------------------------------
print("[1/2] Plotting CNN training example (6 cross maps)...")

train_files = sorted((HARM_CACHE / "nobnt/train").glob("*.npz"))
print(f"  {len(train_files)} train cache files; using {train_files[0].name}")

with np.load(train_files[0], allow_pickle=False) as d:
    patches = np.asarray(d["patches"])          # (N_patches, H, W, 10)
    theta_one = np.asarray(d["theta"])           # (6,)

cross_maps = patches[0, :, :, 4:10]              # (H, W, 6)

# Symmetric color range based on robust percentile across all 6 channels
vmax = float(np.percentile(np.abs(cross_maps), 99))

fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
for k in range(6):
    ax = axes[k // 3, k % 3]
    im = ax.imshow(cross_maps[..., k], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(CROSS_LABELS[k])
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
fig.suptitle(
    f"CNN training example — 6 cross maps "
    f"(post-noise, post-cross, post-zero-mean)\n"
    f"θ = (Ω_m={theta_one[0]:.3f}, σ_8={theta_one[1]:.3f}, "
    f"w_0={theta_one[2]:.2f}, h_0={theta_one[3]/100:.3f}, "
    f"n_s={theta_one[4]:.3f}, Ω_b={theta_one[5]:.3f})"
)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"cnn_training_example_cross_maps.{ext}",
                dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ saved {OUT}/cnn_training_example_cross_maps.{{pdf,png}}")

# ---------------------------------------------------------------------------
# 2) L1 wavelet histograms (per channel × scale)
# ---------------------------------------------------------------------------
print("[2/2] Plotting L1 wavelet histograms with cosmology spread...")

cache_meta = dict(np.load(L1_CACHE / "l1_cache_meta.npz", allow_pickle=True))
n_scales   = int(cache_meta["n_scales"])
l1_nbins   = int(cache_meta["l1_nbins"])
n_channels = int(cache_meta["n_l1_channels"])
snr_lo     = float(cache_meta["l1_min_snr_cross"])
snr_hi     = float(cache_meta["l1_max_snr_cross"])
print(f"  cache meta: {n_channels} channels × {n_scales} scales × {l1_nbins} bins, "
      f"SNR ∈ [{snr_lo:+g}, {snr_hi:+g}]")

d_train = np.load(L1_CACHE / "l1_train.npz")
X      = d_train["x"]                  # (N, 1200)
THETA  = d_train["theta"]              # (N, 6)
assert X.shape[1] == n_channels * n_scales * l1_nbins, \
    f"feature size {X.shape[1]} != {n_channels * n_scales * l1_nbins}"

# Reshape: (N, channels, scales, bins).
# Order: compute_l1_batch outer-loops channels, then concatenates scales
# (each scale contributes `l1_nbins` bins).
Xr = X.reshape(X.shape[0], n_channels, n_scales, l1_nbins)
print(f"  reshaped L1 dataset: {Xr.shape}")

# Sample N_COSMOS distinct cosmologies for variance estimation, plus one
# specific realization for the foreground curve.
N_COSMOS_SAMPLES = 500
rng = np.random.default_rng(42)
# Random sample of indices is the cheapest way; the dataset is shuffled so this
# spans many distinct cosmologies.
sample_idx = rng.choice(Xr.shape[0], size=N_COSMOS_SAMPLES, replace=False)
X_sample = Xr[sample_idx]            # (500, 6, 5, 40)
example_idx = sample_idx[0]
example = Xr[example_idx]            # (6, 5, 40)
example_theta = THETA[example_idx]
print(f"  example idx={example_idx}, θ={example_theta}")

# SNR bin centers
snr_edges   = np.linspace(snr_lo, snr_hi, l1_nbins + 1)
snr_centers = 0.5 * (snr_edges[:-1] + snr_edges[1:])

mean_curves = X_sample.mean(axis=0)   # (6, 5, 40)
std_curves  = X_sample.std(axis=0)    # (6, 5, 40)

# Layout: n_channels rows × n_scales cols
fig, axes = plt.subplots(n_channels, n_scales,
                         figsize=(2.7 * n_scales, 1.9 * n_channels),
                         sharex=True, constrained_layout=True)
for c in range(n_channels):
    for s in range(n_scales):
        ax = axes[c, s]
        # Mean ± std band across cosmologies
        m, sd = mean_curves[c, s], std_curves[c, s]
        ax.fill_between(snr_centers, np.maximum(m - sd, 0.0), m + sd,
                        color="tab:blue", alpha=0.25,
                        label="mean ± std (500 cosmos)" if (c == 0 and s == 0) else None)
        ax.plot(snr_centers, m, color="tab:blue", lw=1.0,
                label="mean (500 cosmos)" if (c == 0 and s == 0) else None)
        # Single example overlay
        ax.step(snr_centers, example[c, s], color="k", lw=1.0, where="mid",
                label=f"example #{example_idx}" if (c == 0 and s == 0) else None)
        if c == 0:
            ax.set_title(f"scale {s+1}")
        if s == 0:
            ax.set_ylabel(CROSS_LABELS[c], fontsize=9)
        if c == n_channels - 1:
            ax.set_xlabel("SNR")
        ax.axvline(0.0, color="gray", lw=0.5, alpha=0.5)
        ax.grid(True, alpha=0.25)
axes[0, 0].legend(loc="upper right", fontsize=7, framealpha=0.9)

fig.suptitle(
    "L1 wavelet histograms — cross-only channels, all scales\n"
    f"(black = realization #{example_idx}; "
    f"blue band = mean ± std across {N_COSMOS_SAMPLES} cosmologies; "
    f"SNR range [{snr_lo:+g}, {snr_hi:+g}] from cache meta)",
    fontsize=11,
)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"l1_histograms_cross_channels.{ext}",
                dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ saved {OUT}/l1_histograms_cross_channels.{{pdf,png}}")
print()
print("Done.")
