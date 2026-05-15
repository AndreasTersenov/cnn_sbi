#!/usr/bin/env python
"""Compare cross-only L1 wavelet histograms with two noise models:

(A) PRODUCTION:    noise_sigma = auto pixel noise (= sigma_e / sqrt(n*Δ²))
                   — same scalar for ALL channels (auto + cross).
                   Cross maps are ~30 000× smaller in amplitude, so SNR
                   collapses to ~10⁻⁵ and L1 histograms are 95% zero.

(B) CHANNEL-AWARE: per-channel noise_sigma_c = std(cross_map_c) across pixels
                   on the batch. SNR is then properly O(1)–O(10).

Both use the SAME L1_NBINS=40, SAME range [-5, +5], SAME wavelets and
subtract_coarse_mean=True. The plot shows per-(channel, scale) histograms
mean ± min/max band across cosmologies, side by side.

Small batch (~few cosmologies). Designed to inform whether a channel-aware
noise model would unblock the L1 cross-only experiment without launching a
big retrain.
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

from npe_l1norm_cross_jaxili_nbody_tomo import (  # noqa: E402
    build_l1_computer,
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

SIGMA_E, GAL_DENSITY = 0.26, 7.5
FIELD_SIZE, FIELD_NPIX = 20, 160
N_SCALES, L1_NBINS = 5, 40
PIXEL_ARCMIN = FIELD_SIZE * 60.0 / FIELD_NPIX
NOISE_AUTO = pixel_noise_sigma(SIGMA_E, GAL_DENSITY, FIELD_SIZE, FIELD_NPIX)
SNR_LO, SNR_HI = -5.0, 5.0
N_COSMOS = 10
print(f"noise_sigma (auto) = {NOISE_AUTO:.3e}")

# --- Load patches ----------------------------------------------------------
fps = sorted((HARM_CACHE / "nobnt/train").glob("*.npz"))[:N_COSMOS]
patches = np.concatenate(
    [np.load(fp)["patches"][..., 4:10] for fp in fps], axis=0
).astype(np.float64)
print(f"Patches: {patches.shape}  (= {N_COSMOS} cosmos × 48 patches × 6 cross channels)")

# Per-channel empirical noise estimate (use median absolute deviation rather
# than std so signal outliers don't blow it up; for now keep simple = std)
sigma_per_ch = np.array([patches[..., c].std() for c in range(6)])
print(f"σ per channel (empirical, used in model B): {sigma_per_ch}")

# --- Wavelet computer + L1 helper -----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stats = build_l1_computer(N_SCALES, PIXEL_ARCMIN, device, "cnn_sbi")


def compute_l1_one_channel(maps_1ch_np, noise_sigma):
    """Returns (N, N_SCALES, L1_NBINS) of L1 norms for a single channel."""
    img = torch.from_numpy(maps_1ch_np.astype(np.float64)).to(device)
    stats.compute_wavelet_transform(img, noise_sigma, subtract_coarse_mean=True)
    _, l1_list = stats.compute_wavelet_l1_norms(
        n_bins=L1_NBINS, min_snr=SNR_LO, max_snr=SNR_HI, clamp_overflow=False,
    )
    # l1_list is a list of N_SCALES tensors, each shape (N, L1_NBINS) (presumably)
    # Concatenate along last dim → (N, N_SCALES*L1_NBINS), then reshape.
    bin_vec = torch.cat(l1_list, dim=-1).detach().cpu().numpy()
    return bin_vec.reshape(bin_vec.shape[0], N_SCALES, L1_NBINS)


# --- Run both models -------------------------------------------------------
print("\n[A] PRODUCTION noise model: auto noise_sigma for all channels")
l1_A = np.zeros((patches.shape[0], 6, N_SCALES, L1_NBINS))
for c in range(6):
    l1_A[:, c] = compute_l1_one_channel(patches[..., c], NOISE_AUTO)
    nz = (l1_A[:, c].mean(axis=0) > 0).sum(axis=-1)
    print(f"  ch {c}: nz bins per scale = {nz.tolist()}")

print("\n[B] CHANNEL-AWARE noise model: σ_c = std(cross_map_c)")
l1_B = np.zeros((patches.shape[0], 6, N_SCALES, L1_NBINS))
for c in range(6):
    l1_B[:, c] = compute_l1_one_channel(patches[..., c], float(sigma_per_ch[c]))
    nz = (l1_B[:, c].mean(axis=0) > 0).sum(axis=-1)
    print(f"  ch {c}: nz bins per scale = {nz.tolist()}")

# --- Plot ------------------------------------------------------------------
snr_edges = np.linspace(SNR_LO, SNR_HI, L1_NBINS + 1)
snr_centers = 0.5 * (snr_edges[:-1] + snr_edges[1:])

fig, axes = plt.subplots(6, 5, figsize=(15, 14), sharex=True, constrained_layout=True)
for c in range(6):
    for s in range(5):
        ax = axes[c, s]
        # (A) production
        mA, mnA, mxA = l1_A[:, c, s].mean(0), l1_A[:, c, s].min(0), l1_A[:, c, s].max(0)
        ax.fill_between(snr_centers, mnA, mxA, color="tab:red", alpha=0.18)
        ax.plot(snr_centers, mA, color="tab:red", lw=1.1,
                label="(A) production (auto σ)" if (c == 0 and s == 0) else None)
        # (B) channel-aware
        mB, mnB, mxB = l1_B[:, c, s].mean(0), l1_B[:, c, s].min(0), l1_B[:, c, s].max(0)
        ax.fill_between(snr_centers, mnB, mxB, color="tab:green", alpha=0.18)
        ax.plot(snr_centers, mB, color="tab:green", lw=1.1,
                label="(B) channel-aware σ" if (c == 0 and s == 0) else None)
        if c == 0:
            ax.set_title(f"scale {s+1}", fontsize=10)
        if s == 0:
            ax.set_ylabel(CROSS_LABELS[c], fontsize=9)
        if c == 5:
            ax.set_xlabel("SNR")
        ax.axvline(0.0, color="gray", lw=0.4, alpha=0.6)
        ax.grid(True, alpha=0.2)
        # Use log-scale on y when (B) has dynamic range, since one or both
        # curves can have huge differences
        if mB.max() > 1:
            ax.set_yscale("log")
fig.legend(*axes[0, 0].get_legend_handles_labels(),
           loc="upper center", bbox_to_anchor=(0.5, 1.025),
           ncol=2, fontsize=10, frameon=True)
fig.suptitle(
    "L1 wavelet histograms — cross-only — production vs channel-aware noise\n"
    f"(SNR range fixed [-5, +5]; {N_COSMOS} cosmos × 48 patches; "
    f"line = mean, band = min-max across realizations; y axis log if model B exceeds 1)",
    fontsize=11, y=1.07,
)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"l1_histograms_noise_model_comparison.{ext}",
                dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n✓ saved {OUT}/l1_histograms_noise_model_comparison.{{pdf,png}}")

# --- Summary table ---------------------------------------------------------
def nz_summary(arr, label):
    nz = (arr.mean(axis=0) > 0).sum(axis=-1)
    print(f"\n[{label}] non-zero bins per (channel, scale): "
          f"mean={nz.mean():.1f}  min={int(nz.min())}  max={int(nz.max())}  /{L1_NBINS}")
nz_summary(l1_A, "A) production")
nz_summary(l1_B, "B) channel-aware")
