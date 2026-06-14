"""Phase 2 (Approach B): 2D-1D Haar SCATTERING ℓ1 — modulus between the 2D starlet and the 1D Haar.

`2D starlet → |·| → 1D Haar across the 4 (pre-mixed) channels → S/N-binned ℓ1`. Because the modulus
is nonlinear, this is NOT reducible to the ordinary ℓ1 of a linear recombination (unlike the linear
Phase-1 Haar) — it is the only version that can exceed the linear-recombination ceiling and that may
survive BNT (the Haar SUM-of-moduli does not suffer the sign cancellation that nulls the deep mode).

pre-basis P (applied to the 4 autos BEFORE the wavelet+modulus):
  none → P=I  (no-BNT)         bnt → P=B  (the scattering computed in BNT space)
post-mix: the orthonormal 4-bin Haar H (deep ¼Σ, coarse diff, 2 fine diffs) across the |wavelet| fields.

Noise σ per (mode, scale) MUST be empirical (the modulus folds the Gaussian) — see
freeze_haar_scatter_noise.py. Reuses the wl_stats_torch L1 seam (set wavelet_coeffs/noise_levels/
snr_coeffs, call compute_wavelet_l1_norms) exactly as flatsky_cross_l1._l1_with_frozen_sigma.
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
import torch

import flatsky_cross as fx


def haar4() -> np.ndarray:
    """Orthonormal 2-level Haar over 4 bins; row 0 = deep mode ¼Σκ (×2)."""
    s = 1.0 / np.sqrt(2.0)
    return np.array([[0.5, 0.5, 0.5, 0.5],
                     [0.5, 0.5, -0.5, -0.5],
                     [s, -s, 0.0, 0.0],
                     [0.0, 0.0, s, -s]], dtype=np.float64)


def modulus_haar_fields(channels4: torch.Tensor, H: torch.Tensor, stats,
                        subtract_coarse_mean: bool = True) -> torch.Tensor:
    """channels4: (B,H,W,4) maps already in the pre-modulus basis (autos or B·autos).
    Returns J: (n_modes, B, n_scales, H, W) = Haar mix across the 4 channels of |starlet coeffs|."""
    aWc = []
    for b in range(4):
        stats.compute_wavelet_transform(channels4[..., b].contiguous(), 1.0,
                                        subtract_coarse_mean=subtract_coarse_mean)
        aWc.append(stats.wavelet_coeffs.abs())          # (B, n_scales, H, W)
    aWc = torch.stack(aWc, 0)                            # (4, B, n_scales, H, W)
    return torch.einsum("mb,b...->m...", H.to(aWc.dtype), aWc)   # (n_modes, B, n_scales, H, W)


def scatter_l1(channels4: torch.Tensor, H: torch.Tensor, sigma_modes: torch.Tensor,
               stats, l1_nbins: int, snr_ranges: np.ndarray,
               subtract_coarse_mean: bool = True, clamp_overflow: bool = False) -> np.ndarray:
    """(B, n_modes*n_scales*l1_nbins). Per mode: histogram |J_m| over its [min,max] S/N range,
    S/N = J_m / sigma_modes[m] (frozen empirical, the folded-noise level). Reuses the L1 seam."""
    J = modulus_haar_fields(channels4, H, stats, subtract_coarse_mean)   # (M, B, ns, Hh, Ww)
    M, B, ns = J.shape[0], J.shape[1], stats.n_scales
    out = []
    for m in range(M):
        field = J[m]                                     # (B, ns, Hh, Ww)
        sig = sigma_modes[m].view(1, ns, 1, 1)
        stats.wavelet_coeffs = field
        stats.noise_levels = sig.expand_as(field)
        stats.snr_coeffs = field / sig
        _, l1 = stats.compute_wavelet_l1_norms(
            n_bins=l1_nbins, min_snr=float(snr_ranges[m][0]),
            max_snr=float(snr_ranges[m][1]), clamp_overflow=clamp_overflow)
        out.append(torch.cat(l1, dim=-1))                # (B, ns*l1_nbins)
    return torch.cat(out, dim=-1).cpu().numpy()


def build_and_scatter_l1(autos_np: np.ndarray, pre_basis, H: torch.Tensor,
                         sigma_modes: torch.Tensor, stats, l1_nbins: int, snr_ranges: np.ndarray,
                         subtract_coarse_mean: bool = True, clamp_overflow: bool = False) -> np.ndarray:
    """autos_np: (B,H,W,4) raw autos. pre_basis: False (P=I) or 'bnt' (P=B) or an ndarray."""
    autos = torch.from_numpy(np.ascontiguousarray(autos_np, np.float64)).to(stats.device)
    channels4 = fx.build_channels_torch(autos, "none", bnt=pre_basis)   # (B,H,W,4) in pre-basis
    return scatter_l1(channels4, H, sigma_modes, stats, l1_nbins, snr_ranges,
                      subtract_coarse_mean=subtract_coarse_mean, clamp_overflow=clamp_overflow)


def calibrate_scatter_snr_range(tfds_name, data_dir, pre_basis, H, sigma_modes, stats,
                                n_calibration_examples=20 * 180, perm_lo=5, perm_hi=6,
                                subtract_coarse_mean=True, margin=0.05, q_lo=0.5, q_hi=99.5,
                                seed=0, batch_size=480, reservoir_per_batch=4000,
                                mode_names=None) -> np.ndarray:
    """Per-mode [min,max] S/N range from robust percentiles of J_m/sigma_modes[m]. Returns (M,2)."""
    from tfds_cross_tfdata_loader import iter_cross_tfds_batches
    print("######## [scatter] CALIBRATING PER-MODE SNR RANGE (empirical sigma) ########", flush=True)
    M = H.shape[0]; ns = stats.n_scales
    reservoirs: List[List[torch.Tensor]] = [[] for _ in range(M)]
    gen = torch.Generator(device=stats.device); gen.manual_seed(0xC0FFEE)
    n = 0
    for autos_np, _ in iter_cross_tfds_batches(
        tfds_name, data_dir, "train", batch_size, flip=False,
        channel_scale=None, channel_slice=slice(0, 4), perm_lo=perm_lo, perm_hi=perm_hi, seed=seed):
        if np.isnan(autos_np).any():
            continue
        autos = torch.from_numpy(np.ascontiguousarray(autos_np, np.float64)).to(stats.device)
        channels4 = fx.build_channels_torch(autos, "none", bnt=pre_basis)
        J = modulus_haar_fields(channels4, H, stats, subtract_coarse_mean)   # (M,B,ns,H,W)
        for m in range(M):
            snr = (J[m] / sigma_modes[m].view(1, ns, 1, 1)).reshape(-1)
            if snr.numel() > reservoir_per_batch:
                idx = torch.randint(0, snr.numel(), (reservoir_per_batch,), generator=gen, device=snr.device)
                snr = snr[idx]
            reservoirs[m].append(snr.detach().cpu())
        n += autos_np.shape[0]
        if n >= n_calibration_examples:
            break
    ranges = np.zeros((M, 2), np.float64)
    print(f"  calibrated from {n} maps (percentiles {q_lo}/{q_hi}, margin {margin}):", flush=True)
    for m in range(M):
        pooled = torch.cat(reservoirs[m])
        lo = float(torch.quantile(pooled, q_lo / 100.0)); hi = float(torch.quantile(pooled, q_hi / 100.0))
        span = hi - lo; ranges[m] = [lo - margin * span, hi + margin * span]
        nm = mode_names[m] if mode_names and m < len(mode_names) else f"mode{m}"
        print(f"    {nm:14s}: [{ranges[m,0]:9.3f}, {ranges[m,1]:9.3f}]  "
              f"(raw {float(pooled.min()):.2f}/{float(pooled.max()):.2f})", flush=True)
    return ranges


def compute_scatter_dataset(tfds_name, data_dir, split, pre_basis, H, sigma_modes, stats,
                            l1_nbins, snr_ranges, perm_lo=None, perm_hi=None, flip=True,
                            seed=1001, batch_size=512, subtract_coarse_mean=True,
                            clamp_overflow=True):
    """Finite deterministic pass: autos -> scatter ℓ1. theta H0->h0 to match the cache builder."""
    from tfds_cross_tfdata_loader import iter_cross_tfds_batches
    print(f"  [scatter] {split} perms {perm_lo}-{perm_hi} pre_basis={pre_basis} flip={flip} ...", flush=True)
    xs, ths, n, t0 = [], [], 0, time.time()
    for autos_np, theta_np in iter_cross_tfds_batches(
        tfds_name, data_dir, split, batch_size, flip=flip,
        channel_scale=None, channel_slice=slice(0, 4), perm_lo=perm_lo, perm_hi=perm_hi, seed=seed):
        if np.isnan(autos_np).any():
            print("    [!] skipped NaN batch"); continue
        x = build_and_scatter_l1(autos_np, pre_basis, H, sigma_modes, stats, l1_nbins, snr_ranges,
                                 subtract_coarse_mean=subtract_coarse_mean, clamp_overflow=clamp_overflow)
        th = theta_np.copy(); th[:, 3] = th[:, 3] / 100.0
        xs.append(x); ths.append(th); n += autos_np.shape[0]
        if n % 20000 < batch_size:
            print(f"    {n} ({time.time()-t0:.0f}s)", flush=True)
    if not xs:
        raise RuntimeError(f"no scatter examples split={split} perms {perm_lo}-{perm_hi}")
    print(f"  [scatter] done {n} in {time.time()-t0:.1f}s", flush=True)
    return {"theta": np.concatenate(ths, 0), "x": np.concatenate(xs, 0)}
