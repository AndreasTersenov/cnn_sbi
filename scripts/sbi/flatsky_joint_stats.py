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


def _snr_bins(wc: torch.Tensor, sigma: torch.Tensor, k: int, ranges=None):
    """SNR u = wc/sigma[c,s]; bin indices in [0,k) with clamp-to-edge. Returns (u, bins).
    ranges: optional (C,S,2) per-(channel,scale) [lo,hi] in SNR units (signal-adapted
    percentile grid — the 'transported binning' variant); default = fixed [-5,5]."""
    B, C, S, H, W = wc.shape
    u = wc / sigma.view(1, C, S, 1, 1)
    if ranges is None:
        width = 2.0 * SNR_RANGE / k
        bins = torch.clamp(((u + SNR_RANGE) / width).long(), 0, k - 1)
    else:
        lo = ranges[..., 0].view(1, C, S, 1, 1)
        hi = ranges[..., 1].view(1, C, S, 1, 1)
        bins = torch.clamp(((u - lo) / (hi - lo) * k).long(), 0, k - 1)
    return u, bins


def calibrate_joint_ranges(autos_iter, basis, sigma, stats, k_unused,
                           n_examples=3600, q_lo=0.5, q_hi=99.5) -> torch.Tensor:
    """Per-(channel,scale) SNR percentile ranges from ~n_examples maps — the
    signal-adapted grid. Returns (C,S,2) on stats.device."""
    pooled = None
    n = 0
    for autos_np in autos_iter:
        wc = wavelet_stack(autos_np, basis, stats)                 # (B,C,S,H,W)
        B, C, S, H, W = wc.shape
        u = (wc / sigma.view(1, C, S, 1, 1)).reshape(B, C, S, -1)
        sub = u[:, :, :, ::37].reshape(-1, C, S)                   # thin pixel subsample
        pooled = sub if pooled is None else torch.cat([pooled, sub], 0)
        n += B
        if n >= n_examples:
            break
    lo = torch.quantile(pooled, q_lo / 100.0, dim=0)               # (C,S)
    hi = torch.quantile(pooled, q_hi / 100.0, dim=0)
    span = hi - lo
    return torch.stack([lo - 0.05 * span, hi + 0.05 * span], dim=-1)  # (C,S,2)


def calibrate_joint_rotation(autos_iter, basis, sigma, stats, k,
                             n_examples=3600, q_lo=0.5, q_hi=99.5, eps=1e-8):
    """Per-(pair,scale) 2-D PCA-whitening of the (u_i,u_j) SNR cloud (shear-aware transport):
    rotates the pair's 2-D grid onto the cloud's eigen-axes AND scales to unit variance, so the
    histogram follows BNT's full 2-D tilt (the shear the axis-aligned `ranges` cannot). Returns a
    dict {pairs, mu(npairs,S,2), L(npairs,S,2,2)=diag(1/sqrt λ)·Vᵀ, rng(npairs,S,2,2)=[lo,hi] on
    the whitened axes}. Calibrated ONCE on ~n_examples train maps; applied identically to all passes.
    NB BNT mixes all 4 channels, so the BNT pair is NOT a 2-D rotation of the original pair —
    this is a genuine pairwise transport, not the trivial full rotate-back."""
    pooled, n = None, 0
    for autos_np in autos_iter:
        wc = wavelet_stack(autos_np, basis, stats)
        B, C, S, H, W = wc.shape
        u = (wc / sigma.view(1, C, S, 1, 1)).reshape(B, C, S, -1)
        sub = u[:, :, :, ::37].reshape(-1, C, S)
        pooled = sub if pooled is None else torch.cat([pooled, sub], 0)
        n += B
        if n >= n_examples:
            break
    pairs = fx.cross_pairs(pooled.shape[1])
    S = pooled.shape[2]
    dev, dt = pooled.device, pooled.dtype
    npairs = len(pairs)
    mu = torch.zeros(npairs, S, 2, device=dev, dtype=dt)
    L = torch.zeros(npairs, S, 2, 2, device=dev, dtype=dt)
    rng = torch.zeros(npairs, S, 2, 2, device=dev, dtype=dt)
    eye = torch.eye(2, device=dev, dtype=dt)
    for pidx, (i, j) in enumerate(pairs):
        for s in range(S):
            x = torch.stack([pooled[:, i, s], pooled[:, j, s]], dim=1)  # (m,2)
            m = x.mean(0)
            xc = x - m
            cov = (xc.T @ xc) / (xc.shape[0] - 1) + eps * eye
            evals, evecs = torch.linalg.eigh(cov)                       # ascending
            Lmat = torch.diag(evals.clamp_min(eps).rsqrt()) @ evecs.T   # PCA-whitening (2,2)
            v = xc @ Lmat.T                                             # (m,2) whitened
            lo = torch.quantile(v, q_lo / 100.0, dim=0)
            hi = torch.quantile(v, q_hi / 100.0, dim=0)
            span = hi - lo
            mu[pidx, s] = m
            L[pidx, s] = Lmat
            rng[pidx, s, :, 0] = lo - 0.05 * span
            rng[pidx, s, :, 1] = hi + 0.05 * span
    return {"pairs": pairs, "mu": mu, "L": L, "rng": rng}


def _bin_axis(v, lo, hi, k):
    """v (...) -> bin index in [0,k) with clamp-to-edge given scalar lo/hi."""
    return torch.clamp(((v - lo) / (hi - lo) * k).long(), 0, k - 1)


def pair2d_features(wc, sigma, k: int, weighted: bool = False,
                    dequant_gen=None, ranges=None, rotation=None) -> torch.Tensor:
    """(B,C,S,H,W) -> (B, npairs*S*k*k). weighted=False: counts (the joint PDF estimate);
    weighted=True: cells hold sum (|u_i|+|u_j|)/2 (the joint wavelet l1).
    dequant_gen: optional — U(0,1) noise per cell (kills the zero-point-mass of
    rarely-occupied cells, which seed-dependently NaN the MAF in the sparse BNT basis)."""
    B, C, S, H, W = wc.shape
    P = H * W
    u_snr = (wc / sigma.view(1, C, S, 1, 1)).reshape(B, C, S, P)    # SNR (weights + rotation input)
    ncell = k * k
    row = (torch.arange(B, device=wc.device) * ncell).view(B, 1)
    out = []
    if rotation is None:
        _, bins = _snr_bins(wc, sigma, k, ranges=ranges)
        bins = bins.reshape(B, C, S, P)
        pairs = fx.cross_pairs(C)
        for (i, j) in pairs:
            for s in range(S):
                cell = bins[:, i, s] * k + bins[:, j, s]           # (B,P)
                flat = (cell + row).reshape(-1)
                if weighted:
                    w = 0.5 * (u_snr[:, i, s].abs() + u_snr[:, j, s].abs()).reshape(-1)
                    h = torch.zeros(B * ncell, dtype=torch.float64, device=wc.device)
                    h.scatter_add_(0, flat, w)
                else:
                    h = torch.bincount(flat, minlength=B * ncell).to(torch.float64)
                out.append(h.view(B, ncell))
    else:
        pairs, mu, L, rng = (rotation[x] for x in ("pairs", "mu", "L", "rng"))
        for pidx, (i, j) in enumerate(pairs):
            for s in range(S):
                ui, uj = u_snr[:, i, s], u_snr[:, j, s]            # (B,P)
                x = torch.stack([ui, uj], dim=-1)                  # (B,P,2)
                v = (x - mu[pidx, s]) @ L[pidx, s].T               # (B,P,2) whitened (rot+scale)
                b0 = _bin_axis(v[..., 0], rng[pidx, s, 0, 0], rng[pidx, s, 0, 1], k)
                b1 = _bin_axis(v[..., 1], rng[pidx, s, 1, 0], rng[pidx, s, 1, 1], k)
                cell = b0 * k + b1                                 # (B,P)
                flat = (cell + row).reshape(-1)
                if weighted:
                    w = 0.5 * (ui.abs() + uj.abs()).reshape(-1)
                    h = torch.zeros(B * ncell, dtype=torch.float64, device=wc.device)
                    h.scatter_add_(0, flat, w)
                else:
                    h = torch.bincount(flat, minlength=B * ncell).to(torch.float64)
                out.append(h.view(B, ncell))
    f = torch.cat(out, dim=1)                                      # (B, npairs*S*k*k)
    if dequant_gen is not None:
        f = f + torch.rand(f.shape, generator=dequant_gen,
                           device=f.device, dtype=f.dtype)
    return f


def full4d_features(wc, sigma, k: int, dequant_gen=None, ranges=None) -> torch.Tensor:
    """(B,C,S,H,W) -> (B, S*k^C) full joint histogram counts (exactly basis-covariant).
    dequant_gen: optional torch.Generator — adds U(0,1) dequantization noise to every cell
    (the standard flows-on-count-data fix; quasi-discrete sparse cells NaN the MAF —
    diagnosed 2026-06-11 night: median surviving dim had ~4 distinct values)."""
    B, C, S, H, W = wc.shape
    P = H * W
    _, bins = _snr_bins(wc, sigma, k, ranges=ranges)
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
                     dequant_gen=None, ranges=None, rotation=None) -> np.ndarray:
    wc = wavelet_stack(autos_np, basis, stats)
    if stat == "cov":
        f = cov_features(wc)
    elif stat == "pair2d":
        f = pair2d_features(wc, sigma, k, weighted=False, dequant_gen=dequant_gen,
                            ranges=ranges, rotation=rotation)
    elif stat == "jointl1":
        f = pair2d_features(wc, sigma, k, weighted=True, dequant_gen=dequant_gen,
                            ranges=ranges, rotation=rotation)
    elif stat == "full4d":
        f = full4d_features(wc, sigma, k, dequant_gen=dequant_gen, ranges=ranges)
    else:
        raise ValueError(f"unknown stat {stat!r}")
    return f.cpu().numpy()
