"""Joint one-point statistics on the flat-sky channel stack (overnight menu 2026-06-11).

Reductions over the per-pixel wavelet-coefficient vectors (W_s x_1..x_C)(p), per scale:
  - cov:     per-scale (co)variance vector — upper-tri of the CxC pixel covariance
             (10 numbers/scale for C=4). The P7 Gaussian-sector block.
  - pair2d:  pairwise 2-D joint histograms in SNR units (counts), all C(C,2) pairs,
             fixed [-RANGE, RANGE], K bins/axis, out-of-range clamped to edge cells.
  - full4d:  full joint 4-D histogram (counts), exactly basis-covariant (P4b).
  - jointl1: pairwise 2-D cells holding sum (|u_i|+|u_j|)/2 over the cell's pixels —
             the joint generalization of the SNR-binned wavelet l1 (Andreas 2026-06-11).

All reductions consume the SAME wavelet pass (WLStatistics, subtract_coarse_mean=True) and
the frozen per-(channel,scale) noise sigma of the arm's basis. Feature ordering:
cov: (scale, tri-entry); pair2d/jointl1: (pair, scale, cell_row*K+cell_col);
full4d: (scale, flat cell). Documented here = the contract for any later reader.
"""
from __future__ import annotations

import numpy as np
import torch

import flatsky_cross as fx

SNR_RANGE = 5.0   # fixed symmetric SNR range for joint histograms (clamp-to-edge)


def wavelet_stack(autos_np: np.ndarray, basis, stats) -> torch.Tensor:
    """(B,H,W,4) raw autos -> wavelet coefficients (B, C, S, H, W) float64 on stats.device.
    basis: False (nobnt) or 'bnt' (channel mix before the wavelet)."""
    autos = torch.from_numpy(np.ascontiguousarray(autos_np, dtype=np.float64)).to(stats.device)
    ch = fx.build_channels_torch(autos, "none", bnt=basis)        # (B,H,W,C)
    B, H, W, C = ch.shape
    x = ch.permute(3, 0, 1, 2).reshape(C * B, H, W).contiguous()
    stats.compute_wavelet_transform(x, 1.0, subtract_coarse_mean=True)
    wc = stats.wavelet_coeffs                                      # (C*B, S, H, W)
    S = wc.shape[1]
    return wc.view(C, B, S, H, W).permute(1, 0, 2, 3, 4)           # (B, C, S, H, W)


def cov_features(wc: torch.Tensor) -> torch.Tensor:
    """(B,C,S,H,W) -> (B, S*C*(C+1)/2) per-scale pixel covariances (upper-tri incl diag)."""
    B, C, S, H, W = wc.shape
    X = wc.reshape(B, C, S, H * W).permute(0, 2, 1, 3)             # (B,S,C,P)
    m = X.mean(dim=-1, keepdim=True)
    Xc = X - m
    cov = torch.einsum("bsip,bsjp->bsij", Xc, Xc) / (H * W - 1)    # (B,S,C,C)
    iu = torch.triu_indices(C, C)
    feat = cov[:, :, iu[0], iu[1]]                                 # (B,S,ntri)
    return feat.reshape(B, -1)


def _snr_bins(wc: torch.Tensor, sigma: torch.Tensor, k: int):
    """SNR u = wc/sigma[c,s]; bin indices in [0,k) with clamp-to-edge. Returns (u, bins)."""
    B, C, S, H, W = wc.shape
    u = wc / sigma.view(1, C, S, 1, 1)
    width = 2.0 * SNR_RANGE / k
    bins = torch.clamp(((u + SNR_RANGE) / width).long(), 0, k - 1)
    return u, bins


def pair2d_features(wc, sigma, k: int, weighted: bool = False) -> torch.Tensor:
    """(B,C,S,H,W) -> (B, npairs*S*k*k). weighted=False: counts (the joint PDF estimate);
    weighted=True: cells hold sum (|u_i|+|u_j|)/2 (the joint wavelet l1)."""
    B, C, S, H, W = wc.shape
    P = H * W
    u, bins = _snr_bins(wc, sigma, k)
    u = u.reshape(B, C, S, P)
    bins = bins.reshape(B, C, S, P)
    pairs = fx.cross_pairs(C)
    ncell = k * k
    row = (torch.arange(B, device=wc.device) * ncell).view(B, 1)
    out = []
    for (i, j) in pairs:
        for s in range(S):
            cell = bins[:, i, s] * k + bins[:, j, s]               # (B,P)
            flat = (cell + row).reshape(-1)
            if weighted:
                w = 0.5 * (u[:, i, s].abs() + u[:, j, s].abs()).reshape(-1)
                h = torch.zeros(B * ncell, dtype=torch.float64, device=wc.device)
                h.scatter_add_(0, flat, w)
            else:
                h = torch.bincount(flat, minlength=B * ncell).to(torch.float64)
            out.append(h.view(B, ncell))
    return torch.cat(out, dim=1)                                   # (B, npairs*S*k*k)


def full4d_features(wc, sigma, k: int, dequant_gen=None) -> torch.Tensor:
    """(B,C,S,H,W) -> (B, S*k^C) full joint histogram counts (exactly basis-covariant).
    dequant_gen: optional torch.Generator — adds U(0,1) dequantization noise to every cell
    (the standard flows-on-count-data fix; quasi-discrete sparse cells NaN the MAF —
    diagnosed 2026-06-11 night: median surviving dim had ~4 distinct values)."""
    B, C, S, H, W = wc.shape
    P = H * W
    _, bins = _snr_bins(wc, sigma, k)
    bins = bins.reshape(B, C, S, P)
    ncell = k ** C
    row = (torch.arange(B, device=wc.device) * ncell).view(B, 1)
    out = []
    for s in range(S):
        cell = bins[:, 0, s]
        for c in range(1, C):
            cell = cell * k + bins[:, c, s]                        # (B,P)
        flat = (cell + row).reshape(-1)
        h = torch.bincount(flat, minlength=B * ncell).to(torch.float64)
        out.append(h.view(B, ncell))
    f = torch.cat(out, dim=1)                                      # (B, S*k^C)
    if dequant_gen is not None:
        f = f + torch.rand(f.shape, generator=dequant_gen,
                           device=f.device, dtype=f.dtype)
    return f


def compute_features(autos_np, stat: str, basis, sigma, stats, k: int,
                     dequant_gen=None) -> np.ndarray:
    wc = wavelet_stack(autos_np, basis, stats)
    if stat == "cov":
        f = cov_features(wc)
    elif stat == "pair2d":
        f = pair2d_features(wc, sigma, k, weighted=False)
    elif stat == "jointl1":
        f = pair2d_features(wc, sigma, k, weighted=True)
    elif stat == "full4d":
        f = full4d_features(wc, sigma, k, dequant_gen=dequant_gen)
    else:
        raise ValueError(f"unknown stat {stat!r}")
    return f.cpu().numpy()
