#!/usr/bin/env python
"""Two final sanity checks before declaring the noise model broken:

(1) Auto-channel L1 histograms under the production noise model — these
    SHOULD fill ~all 40 bins (validating the setup for autos).
(2) Cosmology-discrimination metric for cross channels under BOTH noise
    models — quantifies how much of the L1-feature variance is driven by
    cosmology vs by within-cosmology noise/seed scatter. If the production
    noise model truly nukes cross-channel information, this metric should
    drop ~5 orders of magnitude relative to the channel-aware model.

The metric:
    For each L1 feature, with N_COSMOS cosmologies and N_PER patches per
    cosmology, compute
        var_between = var across cosmology means
        var_within  = mean across cosmologies of (var within cosmology)
        F = var_between / (var_between + var_within)         in [0, 1]
    F → 1 means "all variance is cosmology-driven" (perfect signal),
    F → 0 means "all variance is within-cosmology" (no signal).
    Average F across bins gives a global cross-cosmology discriminative
    power.
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
from npe_l1norm_cross_jaxili_nbody_tomo import (  # noqa
    build_l1_computer,
    pixel_noise_sigma,
)

HARM_CACHE = REPO / "scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid"
OUT = REPO / "scripts/sbi/results/exploratory/cross_only_campaign/diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

# Config
SIGMA_E, GAL_DENSITY, FIELD_SIZE, FIELD_NPIX = 0.26, 7.5, 20, 160
N_SCALES, L1_NBINS, SNR_LO, SNR_HI = 5, 40, -5.0, 5.0
PIXEL_ARCMIN = FIELD_SIZE * 60.0 / FIELD_NPIX
NOISE_AUTO = pixel_noise_sigma(SIGMA_E, GAL_DENSITY, FIELD_SIZE, FIELD_NPIX)
N_COSMOS = 30           # how many distinct cosmologies to sample
N_PER_COSMO = 48        # patches per cosmology (full file)

print(f"noise_sigma (auto) = {NOISE_AUTO:.3e}")

# --- Load patches per cosmology --------------------------------------------
fps = sorted((HARM_CACHE / "nobnt/train").glob("*.npz"))[:N_COSMOS]
patches_by_cosmo = []   # list of (48, H, W, 10)
theta_by_cosmo = []
for fp in fps:
    with np.load(fp, allow_pickle=False) as d:
        patches_by_cosmo.append(np.asarray(d["patches"]).astype(np.float64))
        theta_by_cosmo.append(np.asarray(d["theta"]))
print(f"Loaded {N_COSMOS} cosmologies × {N_PER_COSMO} patches each")

# --- Build L1 computer ------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stats = build_l1_computer(N_SCALES, PIXEL_ARCMIN, device, "cnn_sbi")


def compute_l1(maps_1ch_np, noise_sigma):
    img = torch.from_numpy(maps_1ch_np.astype(np.float64)).to(device)
    stats.compute_wavelet_transform(img, noise_sigma, subtract_coarse_mean=True)
    _, l1_list = stats.compute_wavelet_l1_norms(
        n_bins=L1_NBINS, min_snr=SNR_LO, max_snr=SNR_HI, clamp_overflow=False,
    )
    bin_vec = torch.cat(l1_list, dim=-1).detach().cpu().numpy()
    return bin_vec.reshape(bin_vec.shape[0], N_SCALES, L1_NBINS)


# --- (1) Auto-channel sanity check: production noise, channel 0 (auto κ_1) -
print("\n[1] AUTO channel 0 under PRODUCTION noise — expect ~all 40 bins active")
auto_l1 = np.zeros((N_COSMOS, N_PER_COSMO, N_SCALES, L1_NBINS))
for ci, p in enumerate(patches_by_cosmo):
    auto_l1[ci] = compute_l1(p[..., 0], NOISE_AUTO)
nz = (auto_l1.mean(axis=(0, 1)) > 0).sum(axis=-1)
print(f"  non-zero bins per scale: {nz.tolist()}")


# --- (2) Discriminative-power metric (variance ratio) ---------------------
def disc_power(arr):
    """arr shape: (n_cosmos, n_per_cosmo, n_scales, n_bins).
    Returns per-(scale, bin) F = var_between / (var_between + var_within)."""
    mean_per_cosmo = arr.mean(axis=1)             # (n_cosmos, n_scales, n_bins)
    var_between = mean_per_cosmo.var(axis=0)      # (n_scales, n_bins)
    var_within = arr.var(axis=1).mean(axis=0)     # (n_scales, n_bins)
    total = var_between + var_within
    F = np.where(total > 0, var_between / total, 0.0)
    return F, var_between, var_within


# Auto-channel discrimination (baseline)
F_auto, vb_a, vw_a = disc_power(auto_l1)
print(f"\n[2a] AUTO channel 0: discriminative-power F (mean)={F_auto.mean():.3f}  "
      f"median={np.median(F_auto):.3f}  max={F_auto.max():.3f}")

# Cross channel 0, both noise models
print("\n[2b] CROSS channel 0 — production vs channel-aware noise")
cross_l1_A = np.zeros((N_COSMOS, N_PER_COSMO, N_SCALES, L1_NBINS))   # production
cross_l1_B = np.zeros((N_COSMOS, N_PER_COSMO, N_SCALES, L1_NBINS))   # channel-aware
for ci, p in enumerate(patches_by_cosmo):
    # CROSS channel 0 = harmonic-cache channel index 4
    cmap = p[..., 4]
    cross_l1_A[ci] = compute_l1(cmap, NOISE_AUTO)
    # channel-aware σ: use std of THIS cosmology's cross map
    sigma_c = float(cmap.std())
    cross_l1_B[ci] = compute_l1(cmap, sigma_c)

F_A, _, _ = disc_power(cross_l1_A)
F_B, _, _ = disc_power(cross_l1_B)
print(f"  (A) production:   F mean={F_A.mean():.4f}  max={F_A.max():.4f}")
print(f"  (B) channel-aware: F mean={F_B.mean():.4f}  max={F_B.max():.4f}")
print(f"  → ratio of fixed/broken mean F = {F_B.mean()/max(F_A.mean(), 1e-12):.2f}×")

# Also compute on the NON-ZERO bins only of model A — even if only 2 bins
# carry mass, do those 2 still discriminate cosmology?
nz_A = cross_l1_A.mean(axis=(0,1)) > 0
F_A_nz = F_A[nz_A]
print(f"  (A) production, non-zero bins only ({nz_A.sum()} bins): "
      f"F mean={F_A_nz.mean():.4f}  max={F_A_nz.max():.4f}")

# --- (3) Composite plot ----------------------------------------------------
fig, axes = plt.subplots(3, N_SCALES, figsize=(2.7 * N_SCALES, 7.5),
                         sharex=True, constrained_layout=True)
snr_centers = 0.5 * (np.linspace(SNR_LO, SNR_HI, L1_NBINS + 1)[:-1] +
                     np.linspace(SNR_LO, SNR_HI, L1_NBINS + 1)[1:])

def per_realization(arr):
    return arr.reshape(-1, *arr.shape[2:])  # (n_cosmos*n_per_cosmo, n_scales, n_bins)

panels = [
    ("auto κ_1, production σ", per_realization(auto_l1), "tab:blue"),
    ("cross κ_1×κ_2, production σ", per_realization(cross_l1_A), "tab:red"),
    ("cross κ_1×κ_2, channel-aware σ", per_realization(cross_l1_B), "tab:green"),
]
for row, (title, arr, color) in enumerate(panels):
    for s in range(N_SCALES):
        ax = axes[row, s]
        m, mn, mx = arr[:, s].mean(0), arr[:, s].min(0), arr[:, s].max(0)
        ax.fill_between(snr_centers, mn, mx, color=color, alpha=0.20)
        ax.plot(snr_centers, m, color=color, lw=1.2)
        if row == 0:
            ax.set_title(f"scale {s+1}", fontsize=10)
        if s == 0:
            ax.set_ylabel(title, fontsize=9)
        if row == 2:
            ax.set_xlabel("SNR")
        ax.axvline(0.0, color="gray", lw=0.4, alpha=0.6)
        ax.grid(True, alpha=0.2)
        if arr[:, s].max() > 1:
            ax.set_yscale("log")
fig.suptitle(
    f"L1 wavelet histogram sanity — {N_COSMOS} cosmos × {N_PER_COSMO} patches\n"
    f"top = auto (production σ, expected to fill); middle = cross (production σ, broken); "
    f"bottom = cross (channel-aware σ, fixed)",
    fontsize=11, y=1.04,
)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"l1_histograms_signal_check.{ext}",
                dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n✓ saved {OUT}/l1_histograms_signal_check.{{pdf,png}}")
