"""Flat-sky (patch-local) cross-map L1 datavectors with FROZEN per-(channel,scale) noise sigma.

The de-leaked replacement for the harmonic-cache cross L1 path. Reads ONLY the 4 auto
channels (ch 0-3) of the cross TFDS, builds the 6 (conv|product) or 12 (both) cross
channels ON-DEVICE (torch.fft), and computes the wavelet-l1 datavector using the frozen
per-(channel,scale) NOISE sigma from freeze_flatsky_cross_noise.py as the SNR denominator
(repo NOISE-based convention; overrides the wavelet's WHITE-noise propagation, which is
invalid for the colored cross-noise AND — we measured — for the band-limited auto finest scale).

Single source of truth for the cross operators is flatsky_cross.py (np/torch/jax bit-identical).
This module owns: frozen-sigma selection, the sigma-override L1, the train/val dataset pass,
the obs single-map, and the SNR-range calibration — all reused by npe_l1norm_cross_jaxili_nbody_tomo.py
at the `flat_local` route seams.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import os

import numpy as np
import torch

import flatsky_cross as fx

# Cap maps per batched wavelet call (channel-chunking) to bound GPU memory; >~16k maps
# OOMs a 40GB A100. Override via FLATSKY_MAX_WAVELET_MAPS. 6144 saturates an A100 well
# while leaving headroom (measured GATE A2).
_MAX_WAVELET_MAPS = int(os.environ.get("FLATSKY_MAX_WAVELET_MAPS", "6144"))


# ---------------------------------------------------------------------------
# Frozen sigma table
# ---------------------------------------------------------------------------
def op_channel_rows(op: str, nbins: int) -> List[int]:
    """Channel indices of `op` into the 16-channel 'both' layout (auto, conv, product) —
    the SAME order build_channels_* emits. Lets every arm be a column-slice of `both`."""
    npairs = len(fx.cross_pairs(nbins))
    auto = list(range(nbins))
    conv = list(range(nbins, nbins + npairs))
    prod = list(range(nbins + npairs, nbins + 2 * npairs))
    return {"none": auto, "conv": auto + conv, "product": auto + prod,
            "both": auto + conv + prod}[op]


def op_feature_columns(op: str, nbins: int, feat_per_ch: int) -> np.ndarray:
    """L1-datavector column indices for `op` given the 'both' (16-channel) layout, so the
    arm's raw datavector = both_datavector[:, op_feature_columns(...)]. feat_per_ch =
    n_scales * l1_nbins."""
    rows = op_channel_rows(op, nbins)
    return np.concatenate([np.arange(c * feat_per_ch, (c + 1) * feat_per_ch) for c in rows]).astype(np.int64)


def select_frozen_sigma(npz_path: str, op: str, nbins: int, device, dtype=torch.float64,
                        expected_bnt: bool | None = None):
    """Load the 16-channel frozen sigma table and select the rows for `op`, in the
    SAME channel order flatsky_cross.build_channels_* emits: auto(nbins) + cross(es).

    expected_bnt: when given, enforce that the table was frozen with the matching
    BNT setting (tables written before 2026-06-10 lack the key = no-BNT). Using a
    no-BNT sigma for a BNT arm (or vice versa) is the silent-wrong-noise-model
    failure mode (cf. feedback_l1_cross_must_use_harmonic_route) — hard error.

    Returns (sigma_torch[(C, n_scales)], channel_names[list[str]], n_scales[int]).
    """
    z = np.load(npz_path)
    if expected_bnt is not None:
        table_bnt = bool(z["bnt"]) if "bnt" in z.files else False
        if table_bnt != bool(expected_bnt):
            raise ValueError(
                f"frozen sigma table {npz_path} has bnt={table_bnt} but the arm requires "
                f"bnt={bool(expected_bnt)}; refreeze with freeze_flatsky_cross_noise.py"
                f"{' --bnt' if expected_bnt else ''}."
            )
    sigma16 = np.asarray(z["sigma"], dtype=np.float64)          # (16, n_scales)
    names16 = [str(x) for x in z["channel_names"]]
    n_scales = int(z["n_scales"])
    npairs = len(fx.cross_pairs(nbins))
    auto = list(range(nbins))                                   # 0..3
    conv = list(range(nbins, nbins + npairs))                   # 4..9
    prod = list(range(nbins + npairs, nbins + 2 * npairs))      # 10..15
    if op == "none":
        rows = auto
    elif op == "conv":
        rows = auto + conv
    elif op == "product":
        rows = auto + prod
    elif op == "both":
        rows = auto + conv + prod
    else:
        raise ValueError(f"Unknown cross op={op!r}")
    sigma = torch.from_numpy(sigma16[rows]).to(device=device, dtype=dtype)
    names = [names16[r] for r in rows]
    return sigma, names, n_scales


# ---------------------------------------------------------------------------
# Core: wavelet-l1 with frozen per-(channel,scale) sigma injected
# ---------------------------------------------------------------------------
def _l1_with_frozen_sigma(
    channels: torch.Tensor,        # (B, H, W, C) float64 on stats.device
    frozen_sigma: torch.Tensor,    # (C, n_scales) float64 on stats.device
    stats,
    l1_nbins: int,
    snr_ranges: np.ndarray,        # (C, 2) per-channel [min_snr, max_snr]
    subtract_coarse_mean: bool = True,
    clamp_overflow: bool = False,
) -> np.ndarray:
    """Wavelet-transform ALL channels in ONE batched call (C*B maps), override the
    noise levels with the frozen per-(channel,scale) sigma, then histogram each channel
    over its OWN [min,max] range (per-channel because the pointwise product is heavy-tailed
    while conv/auto are not). One GPU->CPU sync per batch. Returns (B, C*n_scales*l1_nbins).

    Batching the wavelet over channels (vs a Python per-channel loop with a CPU sync each)
    is the throughput fix: the per-channel loop left the GPU ~29% utilized (launch/sync bound).
    """
    B, H, W, C = channels.shape
    n_scales = stats.n_scales
    # Batch the wavelet over channels in chunks, capping maps-per-call to bound GPU
    # memory (decouples the loader batch size from wavelet memory; >~16k maps OOMs a
    # 40GB A100). chans_per_call*B <= _MAX_WAVELET_MAPS. Within a chunk: one wavelet
    # call (saturates the GPU), then per-channel histogram with the channel's own range.
    chans_per_call = max(1, _MAX_WAVELET_MAPS // B)
    out = []
    for g0 in range(0, C, chans_per_call):
        g = list(range(g0, min(g0 + chans_per_call, C)))
        ng = len(g)
        x = channels[..., g].permute(3, 0, 1, 2).reshape(ng * B, H, W).contiguous()
        stats.compute_wavelet_transform(x, 1.0, subtract_coarse_mean=subtract_coarse_mean)
        wc = stats.wavelet_coeffs                              # (ng*B, n_scales, H, W)
        sig = frozen_sigma[g].repeat_interleave(B, dim=0).view(ng * B, n_scales, 1, 1)
        snr_all = wc / sig
        for ci, c in enumerate(g):
            sl = slice(ci * B, (ci + 1) * B)
            stats.wavelet_coeffs = wc[sl]
            stats.noise_levels = sig[sl]
            stats.snr_coeffs = snr_all[sl]
            _, l1_norms = stats.compute_wavelet_l1_norms(
                n_bins=l1_nbins, min_snr=float(snr_ranges[c][0]),
                max_snr=float(snr_ranges[c][1]), clamp_overflow=clamp_overflow)
            out.append(torch.cat(l1_norms, dim=-1))            # (B, n_scales*l1_nbins) GPU
    return torch.cat(out, dim=-1).cpu().numpy()                # ONE sync per batch


def build_and_l1(
    autos_np: np.ndarray,          # (B, H, W, 4) raw autos (float32/64)
    op: str,
    frozen_sigma: torch.Tensor,
    stats,
    l1_nbins: int,
    snr_ranges: np.ndarray,        # (C, 2)
    subtract_coarse_mean: bool = True,
    clamp_overflow: bool = False,
    bnt: bool = False,
) -> np.ndarray:
    autos = torch.from_numpy(np.ascontiguousarray(autos_np, dtype=np.float64)).to(stats.device)
    channels = fx.build_channels_torch(autos, op, bnt=bnt)     # (B,H,W,C) float64
    return _l1_with_frozen_sigma(channels, frozen_sigma, stats, l1_nbins, snr_ranges,
                                 subtract_coarse_mean=subtract_coarse_mean,
                                 clamp_overflow=clamp_overflow)


def compute_l1_single_map_flat_local(
    autos_np: np.ndarray,          # (H, W, 4) obs autos
    op: str, frozen_sigma, stats, l1_nbins, snr_ranges,
    subtract_coarse_mean=True, clamp_overflow=False, bnt: bool = False,
) -> np.ndarray:
    vec = build_and_l1(autos_np[None], op, frozen_sigma, stats, l1_nbins, snr_ranges,
                       subtract_coarse_mean=subtract_coarse_mean, clamp_overflow=clamp_overflow,
                       bnt=bnt)
    return vec[0]


def compute_l1_dataset_flat_local(
    tfds_name: str, data_dir: str, split: str,
    op: str, frozen_sigma, stats, l1_nbins: int,
    snr_ranges: np.ndarray,
    perm_lo: Optional[int] = None, perm_hi: Optional[int] = None,
    flip: bool = True, seed: int = 1001, batch_size: int = 480,
    subtract_coarse_mean: bool = True, clamp_overflow: bool = False,
    log_every_examples: int = 20000, bnt: bool = False,
) -> Dict[str, np.ndarray]:
    """Finite deterministic pass over the cross TFDS (autos only), on-device cross,
    frozen-sigma L1. theta H0->h0 to match the cache builder."""
    from tfds_cross_tfdata_loader import iter_cross_tfds_batches
    print(f"  [flat_local] L1 from cross TFDS [{split} perms {perm_lo}-{perm_hi} "
          f"op={op} flip={flip} bnt={bnt}] ...")
    x_list: List[np.ndarray] = []
    theta_list: List[np.ndarray] = []
    n, t0, nxt = 0, time.time(), log_every_examples
    for autos_np, theta_np in iter_cross_tfds_batches(
        tfds_name, data_dir, split, batch_size, flip=flip,
        channel_scale=None, channel_slice=slice(0, 4),
        perm_lo=perm_lo, perm_hi=perm_hi, seed=seed,
    ):
        if np.isnan(autos_np).any():
            print("    [!] skipped batch with NaN autos"); continue
        x = build_and_l1(autos_np, op, frozen_sigma, stats, l1_nbins, snr_ranges,
                         subtract_coarse_mean=subtract_coarse_mean, clamp_overflow=clamp_overflow,
                         bnt=bnt)
        theta = theta_np.copy(); theta[:, 3] = theta[:, 3] / 100.0
        x_list.append(x); theta_list.append(theta)
        n += autos_np.shape[0]
        if n >= nxt:
            el = time.time() - t0
            print(f"    {n} patches ({el:.1f}s, {n/max(el,1e-9):.0f}/s)"); nxt += log_every_examples
    if not x_list:
        raise RuntimeError(f"No flat_local examples for split={split} perms {perm_lo}-{perm_hi}.")
    print(f"  [flat_local] done: {n} patches in {time.time()-t0:.1f}s")
    return {"theta": np.concatenate(theta_list, 0), "x": np.concatenate(x_list, 0)}


def calibrate_snr_range_flat_local(
    tfds_name: str, data_dir: str, op: str, frozen_sigma, stats,
    nbins: int, channel_names: List[str],
    n_calibration_examples: int = 5760,
    perm_lo: Optional[int] = None, perm_hi: Optional[int] = None,
    subtract_coarse_mean: bool = True, margin: float = 0.05,
    q_lo: float = 0.5, q_hi: float = 99.5,
    seed: int = 0, batch_size: int = 480, reservoir_per_batch: int = 4000,
    bnt: bool = False,
) -> np.ndarray:
    """PER-CHANNEL [min,max] SNR range under the frozen sigma, from robust percentiles
    (q_lo/q_hi) + margin. Per-channel (not a single auto/cross pair) because the pointwise
    PRODUCT is intrinsically heavy-tailed (peaky kappa_i*kappa_j) while conv/auto are not —
    a min/max range would waste ~all bins on the product's rare tail. NOT the old band-aid:
    the sigma denominator is the correct frozen per-scale value; percentiles only set the
    histogram extent (the standard heavy-tailed-HOS practice, cf. Zurcher's fixed [-4,4]).
    Returns (C, 2)."""
    from tfds_cross_tfdata_loader import iter_cross_tfds_batches
    print("######## [flat_local] CALIBRATING PER-CHANNEL SNR RANGE (frozen sigma) ########")
    C = fx.n_output_channels(nbins, op)
    n_scales = stats.n_scales
    reservoirs: List[List[torch.Tensor]] = [[] for _ in range(C)]
    gen = torch.Generator(device=stats.device); gen.manual_seed(0xC0FFEE)
    n = 0
    for autos_np, _ in iter_cross_tfds_batches(
        tfds_name, data_dir, "train", batch_size, flip=False,
        channel_scale=None, channel_slice=slice(0, 4),
        perm_lo=perm_lo, perm_hi=perm_hi, seed=seed,
    ):
        if np.isnan(autos_np).any():
            continue
        autos = torch.from_numpy(np.ascontiguousarray(autos_np, dtype=np.float64)).to(stats.device)
        ch = fx.build_channels_torch(autos, op, bnt=bnt)
        for c in range(C):
            stats.compute_wavelet_transform(ch[..., c], 1.0, subtract_coarse_mean=subtract_coarse_mean)
            snr = (stats.wavelet_coeffs / frozen_sigma[c].view(1, n_scales, 1, 1)).reshape(-1)
            if snr.numel() > reservoir_per_batch:
                idx = torch.randint(0, snr.numel(), (reservoir_per_batch,),
                                    generator=gen, device=snr.device)
                snr = snr[idx]
            reservoirs[c].append(snr.detach().cpu())
        n += autos_np.shape[0]
        if n >= n_calibration_examples:
            break

    ranges = np.zeros((C, 2), dtype=np.float64)
    print(f"  calibrated from {n} maps (percentiles {q_lo}/{q_hi}, margin {margin}):")
    for c in range(C):
        pooled = torch.cat(reservoirs[c])
        lo = float(torch.quantile(pooled, q_lo / 100.0))
        hi = float(torch.quantile(pooled, q_hi / 100.0))
        span = hi - lo
        ranges[c] = [lo - margin * span, hi + margin * span]
        nm = channel_names[c] if c < len(channel_names) else f"ch{c}"
        print(f"    {nm:11s}: [{ranges[c,0]:9.3f}, {ranges[c,1]:9.3f}]  "
              f"(raw min/max {float(pooled.min()):.2f}/{float(pooled.max()):.2f})")
    return ranges
