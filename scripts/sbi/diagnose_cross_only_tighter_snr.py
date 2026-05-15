#!/usr/bin/env python
"""Re-compute the L1 wavelet histograms for the cross-only channels at SEVERAL
candidate SNR ranges on a small batch of harmonic patches, and plot a
side-by-side comparison.

Why: the production calibration set the cross-SNR range to [-5, +5] but the
cached histograms show ALL non-zero mass in 2 bins centered at SNR=±0.125 —
effectively only 2 of 40 bins per (channel, scale) carry any data. Most of
the 1200-D L1 feature vector is structurally zero.

This script:
- loads ~few cosmologies × 48 patches from the harmonic cache,
- slices the 6 cross channels,
- computes L1 histograms with the production WLStatistics setup at multiple
  candidate SNR ranges,
- plots the per-(channel, scale) mean + (min,max) band across cosmologies for
  each candidate range, side by side.

Light: ~1-2 minutes on GPU. No production data is overwritten.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path("/mnt/home/tersenov/software/cnn_sbi")
sys.path.insert(0, str(REPO / "scripts/sbi"))
sys.path.insert(0, "/home/tersenov/software/wl_stats_torch")

# Import the production L1 functions to guarantee parity with the campaign.
from npe_l1norm_cross_jaxili_nbody_tomo import (  # noqa: E402
    build_l1_computer,
    compute_l1_batch,
    pixel_noise_sigma,
)

HARM_CACHE = REPO / "scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid"
OUT = REPO / "scripts/sbi/results/exploratory/cross_only_campaign/diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

CROSS_LABELS = [
    r"$\kappa_1 \times \kappa_2$", r"$\kappa_1 \times \kappa_3$",
    r"$\kappa_1 \times \kappa_4$", r"$\kappa_2 \times \kappa_3$",
    r"$\kappa_2 \times \kappa_4$", r"$\kappa_3 \times \kappa_4$",
]

# --- Production config (matches l1_cache_meta.npz / the running campaign) ---
SIGMA_E         = 0.26
GAL_DENSITY     = 7.5         # gal / arcmin^2
FIELD_SIZE      = 20          # deg
FIELD_NPIX      = 160
N_SCALES        = 5
L1_NBINS        = 40
SUBTRACT_COARSE = True
PIXEL_ARCMIN    = FIELD_SIZE * 60.0 / FIELD_NPIX  # = 7.5

NOISE_SIGMA = pixel_noise_sigma(SIGMA_E, GAL_DENSITY, FIELD_SIZE, FIELD_NPIX)
print(f"noise_sigma per auto-bin = {NOISE_SIGMA:.6g}")

# --- Candidate SNR ranges to test --------------------------------------------
# (label, min_snr, max_snr)
RANGES = [
    ("current  [-5, +5]",   -5.0,  5.0),
    ("[-1, +1]",            -1.0,  1.0),
    ("[-0.25, +0.25]",      -0.25, 0.25),
]

# --- Load a small set of cosmologies ----------------------------------------
N_COSMOS = 10           # different cosmology files
PATCHES_PER_COSMO = 48  # all patches in each file

train_files = sorted((HARM_CACHE / "nobnt/train").glob("*.npz"))
print(f"Found {len(train_files)} train cache files; using first {N_COSMOS}")

all_patches = []   # list of (48, 160, 160, 6) cross-only slices
all_theta   = []
for fp in train_files[:N_COSMOS]:
    with np.load(fp, allow_pickle=False) as d:
        all_patches.append(np.asarray(d["patches"])[..., 4:10])
        all_theta.append(np.asarray(d["theta"]))
all_patches = np.concatenate(all_patches, axis=0).astype(np.float64)  # (N_COSMOS*48, 160, 160, 6)
all_theta = np.stack(all_theta)
print(f"Loaded patches: {all_patches.shape}, theta: {all_theta.shape}")

# --- Build the L1 computer (same as the production runner) ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stats = build_l1_computer(
    n_scales=N_SCALES,
    pixel_arcmin=PIXEL_ARCMIN,
    torch_device=device,
    l1_implementation="cnn_sbi",
)
print(f"WLStatistics on {device}")

# --- Recompute L1 for each candidate range ----------------------------------
# Output: dict[range_label] -> (N, 6, 5, 40)
results = {}
for label, lo, hi in RANGES:
    print(f"\n[L1] computing for range {label} ...")
    flat = compute_l1_batch(
        maps_batch=all_patches,
        noise_sigma=NOISE_SIGMA,
        stats=stats,
        l1_nbins=L1_NBINS,
        nbins=0,                            # all 6 channels treated as cross
        l1_min_snr=lo,
        l1_max_snr=hi,
        clamp_overflow=False,
        subtract_coarse_mean=SUBTRACT_COARSE,
        l1_implementation="cnn_sbi",
        n_l1_channels=6,
        l1_min_snr_cross=lo,
        l1_max_snr_cross=hi,
    )
    # shape (N, 6 * 5 * 40); reshape
    arr = flat.reshape(flat.shape[0], 6, N_SCALES, L1_NBINS)
    results[label] = (arr, lo, hi)
    nz_per_panel = (arr.mean(axis=0) > 0).sum(axis=-1)
    print(f"  L1 shape={arr.shape}  non-zero bins per (channel, scale) min={nz_per_panel.min()} mean={nz_per_panel.mean():.1f} max={nz_per_panel.max()}")

# --- Plot side-by-side -------------------------------------------------------
# Layout: rows = cross channels (6), columns = scales (5).
# Each panel overlays the three SNR ranges (in normalized SNR space so they fit
# on the same x-axis), as separate lines.
print("\n[plot] building grid ...")
n_channels, n_scales = 6, N_SCALES
fig, axes = plt.subplots(
    n_channels, n_scales,
    figsize=(2.8 * n_scales, 2.0 * n_channels),
    sharex=False, constrained_layout=True,
)
colors = {RANGES[0][0]: "tab:red", RANGES[1][0]: "tab:orange", RANGES[2][0]: "tab:green"}
example_idx = 0

for c in range(n_channels):
    for s in range(n_scales):
        ax = axes[c, s]
        for label, lo, hi in RANGES:
            arr, _, _ = results[label]
            edges = np.linspace(lo, hi, L1_NBINS + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            mean = arr[:, c, s].mean(axis=0)
            mn = arr[:, c, s].min(axis=0)
            mx = arr[:, c, s].max(axis=0)
            color = colors[label]
            ax.fill_between(centers, mn, mx, color=color, alpha=0.18)
            ax.plot(centers, mean, color=color, lw=1.1,
                    label=label if (c == 0 and s == 0) else None)
        if c == 0:
            ax.set_title(f"scale {s+1}", fontsize=9)
        if s == 0:
            ax.set_ylabel(CROSS_LABELS[c], fontsize=9)
        if c == n_channels - 1:
            ax.set_xlabel("SNR")
        ax.axvline(0.0, color="gray", lw=0.4, alpha=0.6)
        ax.grid(True, alpha=0.2)

# Place legend at the top
fig.legend(*axes[0, 0].get_legend_handles_labels(),
           loc="upper center", bbox_to_anchor=(0.5, 1.02),
           ncol=len(RANGES), fontsize=9, frameon=True)
fig.suptitle(
    f"L1 wavelet histograms — cross-only, 3 candidate SNR ranges "
    f"({N_COSMOS} cosmos × {PATCHES_PER_COSMO} patches)\n"
    f"(line = mean across realizations; band = min-to-max across realizations)",
    fontsize=11, y=1.06,
)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"l1_histograms_snr_range_comparison.{ext}",
                dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✓ saved {OUT}/l1_histograms_snr_range_comparison.{{pdf,png}}")

# --- Summary table -----------------------------------------------------------
print("\n=== Summary: fraction of non-zero bins per (channel, scale) ===")
print(f"{'range':>20}  {'mean nz':>9}  {'min nz':>7}  {'max nz':>7}  ({L1_NBINS} bins total)")
for label, _, _ in RANGES:
    arr, _, _ = results[label]
    nz = (arr.mean(axis=0) > 0).sum(axis=-1)
    print(f"{label:>20}  {nz.mean():>9.1f}  {int(nz.min()):>7}  {int(nz.max()):>7}")
