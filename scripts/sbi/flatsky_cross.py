"""Patch-local (flat-sky) tomographic cross-maps — single source of truth.

Replaces the LEAKY full-sphere harmonic construction (every full-sphere cross-patch
pixel is a global functional of the whole sky; see CROSS_MAP_LEAKAGE_FINDING.md).
These operators are strictly patch-local: each cross channel is a function of the
patch's own auto maps only.

Two operators (LOCKED with Andreas 2026-06-08; FLATSKY_CROSS_BUILD_PLAN.md §1):
  - convolution  C_ij = irfft2( rfft2(k_i * W) * rfft2(k_j * W) )
      The faithful flat-sky analog of Zürcher Eq. 12 (alm-product); apodized-circular,
      smooth/large-scale, carries cross-info in morphology/phase. ONE definition only
      (the zero-pad+crop variant is dropped — it differs by a 39-px lag-registration
      shift + a small edge wrap; REDESIGN_NOTES §12-13).
  - product      P_ij = k_i * k_j
      Raw pointwise (NO apodization, NO FFT); strictly local; its spatial mean = xi_ij
      (validated, REDESIGN_NOTES §14). Per Andreas: do NOT multiply by W^2 — that would
      break the mean=xi_ij property and the consistency with the raw autos.

Channel conventions:
  - convolution inputs are APODIZED (k_i * W); product inputs are RAW (k_i).
  - the auto channels appended to the output are RAW (un-apodized) — matches what the
    autos get downstream (autos are fed raw to the wavelet / per-channel RMS).
  - op='conv'    -> [auto(4), conv(6)]            = 10 channels
    op='product' -> [auto(4), product(6)]         = 10 channels
    op='both'    -> [auto(4), conv(6), product(6)] = 16 channels

Three numerically-identical backends:
  - numpy  : CPU reference (correctness oracle, GATE A bit-match anchor)
  - torch  : GPU, for the L1 wavelet-l1 pipeline (computed right before WLStatistics)
  - jax    : GPU, for the CNN compressor input step

All three must agree to FFT float32 roundoff (GATE A). FFT normalization is the
numpy/"backward" convention in all three (no forward scaling, 1/N on inverse), so a
common `s=(H, W)` on the inverse keeps them aligned.

Apodization window = separable cosine taper, roll_frac default 0.10 (LOCKED).
"""
from __future__ import annotations

import numpy as np

CROSS_OPS = ("none", "conv", "product", "both", "product3")


def cross_pairs(nbins: int) -> list[tuple[int, int]]:
    """Ordered upper-triangular bin pairs (i<j). For nbins=4 -> 6 pairs."""
    return [(i, j) for i in range(nbins) for j in range(i + 1, nbins)]


def cross_triples(nbins: int) -> list[tuple[int, ...]]:
    """Ordered bin triples (i<j<k) plus the single all-bins tuple — the 'product3'
    channels (order-3 closure test, PLAN_OVERNIGHT_MENU_2.md lane D). nbins=4 -> 5."""
    triples = [(i, j, k) for i in range(nbins) for j in range(i + 1, nbins)
               for k in range(j + 1, nbins)]
    return triples + [tuple(range(nbins))]


def _mix_requested(mode) -> bool:
    """True when a channel mix should be applied. `mode` may be False/None (no mix),
    a registered mode string / True, or a raw (rows, nbins) ndarray — the ndarray
    form is what the post-cut builders use (`if mode:` would be ambiguous for arrays)."""
    return mode is not None and mode is not False


# ---------------------------------------------------------------------------
# Optional BNT pre-step (paper pillar 2): apply the nulling transform to the 4
# RAW autos BEFORE any channel build, so both the auto channels and the cross
# channels live in BNT space — the "observed maps" a survey would null first.
# Matrix source of truth: bnt_utils.BNT_MATRIX (tomo4_bnt_v1; channel-last
# convention out[..., i] = sum_j B[i, j] * autos[..., j]). Lazy import keeps
# the heavy bnt_utils->tensorflow dependency out of non-BNT consumers.
# ---------------------------------------------------------------------------
def bnt_matrix_np() -> np.ndarray:
    from bnt_utils import BNT_MATRIX
    return np.asarray(BNT_MATRIX, dtype=np.float32)


def whiten_matrix_np() -> np.ndarray:
    """Q = (B B^T)^(-1/2) B — the noise-whitened BNT basis. Orthogonal for equal
    per-bin noise (Q Q^T = I), i.e. an orthogonal rotation of the ORIGINAL basis:
    per-channel statistics on Q kappa see independent equal-variance noise again.
    Diagnostic basis for decomposing the BNT inflation (PAPER_BNT_* Part II)."""
    B = bnt_matrix_np().astype(np.float64)
    w, V = np.linalg.eigh(B @ B.T)
    M = V @ np.diag(w ** -0.5) @ V.T
    return (M @ B).astype(np.float32)


def deep_matrix_np() -> np.ndarray:
    """1x4 'deep channel' mix: the plain bin average of the ORIGINAL autos —
    the single most kernel-deep direction in channel space (BNT_THEORY_DEEP_DIVE.md
    §5.3/§5.4). Used standalone to build the appended 5th channel of the bnt+deep test."""
    return np.full((1, 4), 0.25, dtype=np.float32)


def bnt_deep_matrix_np() -> np.ndarray:
    """5x4 mix: the 4 BNT rows + the deep (bin-average) row — the §5.4 test frame."""
    return np.concatenate([bnt_matrix_np(), deep_matrix_np()], axis=0)


def deep2_matrix_np() -> np.ndarray:
    """2x4 mix: the bin average + the deepest bin (kappa_4) alone — two depth-distinct
    deep directions (the §5.4 two-channel rung; span-calibration of the deep account)."""
    return np.array([[0.25, 0.25, 0.25, 0.25],
                     [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)


def unions6_matrix_np() -> np.ndarray:
    """6x4 mix: the equal-weight pair averages (kappa_i+kappa_j)/2, i<j — the survey-
    practice union-map analogs (M2). Basis-agnostic: constructible from BNT maps."""
    rows = []
    for i, j in cross_pairs(4):
        r = np.zeros(4, dtype=np.float32)
        r[i] = r[j] = 0.5
        rows.append(r)
    return np.stack(rows)


def mix_matrix_np(mode) -> np.ndarray:
    """Channel-mix matrix for `mode`: True/'bnt' -> BNT; 'whiten' -> noise-whitened BNT;
    'deep' -> 1x4 bin average; 'deep2' -> 2x4 [average; e4]; 'bnt_deep' -> 5x4
    [BNT; average]; 'unions6' -> 6x4 pair averages; a raw (rows, 4) ndarray is passed
    through (the post-cut builders' custom per-scale-masked rows). Mixes need not be
    square: output channels = rows."""
    if isinstance(mode, np.ndarray):
        if mode.ndim != 2:
            raise ValueError(f"ndarray mix mode must be 2-D (rows, nbins); got {mode.shape}")
        return mode.astype(np.float32)
    if mode is True or mode == "bnt":
        return bnt_matrix_np()
    if mode == "whiten":
        return whiten_matrix_np()
    if mode == "deep":
        return deep_matrix_np()
    if mode == "deep2":
        return deep2_matrix_np()
    if mode == "bnt_deep":
        return bnt_deep_matrix_np()
    if mode == "unions6":
        return unions6_matrix_np()
    raise ValueError(
        f"Unknown channel-mix mode {mode!r}; expected one of "
        "'bnt'/'whiten'/'deep'/'deep2'/'bnt_deep'/'unions6'.")


def n_built_channels(nbins: int, op: str, mode=False) -> int:
    """Channel count build_channels_* actually emits: the mix (if any) sets the auto count
    (= mix rows), and the cross pairs/triples are built from the MIXED channels."""
    m = mix_matrix_np(mode).shape[0] if _mix_requested(mode) else nbins
    npairs = len(cross_pairs(m))
    if op == "none":
        return m
    if op in ("conv", "product"):
        return m + npairs
    if op == "both":
        return m + 2 * npairs
    if op == "product3":
        return m + len(cross_triples(m))
    raise ValueError(f"Unknown cross op={op!r}; expected one of {CROSS_OPS}")


def _check_bnt_nbins(n: int) -> None:
    if n != 4:
        raise ValueError(f"channel mix (tomo4) requires 4 auto channels, got {n}.")


def apply_bnt_np(autos_b: np.ndarray, mode=True) -> np.ndarray:
    _check_bnt_nbins(autos_b.shape[-1])
    M = mix_matrix_np(mode)
    out = np.tensordot(autos_b.astype(np.float32), M, axes=([autos_b.ndim - 1], [1]))
    return out.astype(np.float32)


def apply_bnt_torch(autos_b, mode=True):
    import torch
    _check_bnt_nbins(autos_b.shape[-1])
    M = torch.from_numpy(mix_matrix_np(mode)).to(device=autos_b.device, dtype=autos_b.dtype)
    return torch.tensordot(autos_b, M, dims=([autos_b.ndim - 1], [1]))


def apply_bnt_jax(autos_b, mode=True):
    import jax.numpy as jnp
    _check_bnt_nbins(autos_b.shape[-1])
    M = jnp.asarray(mix_matrix_np(mode), dtype=autos_b.dtype)
    return jnp.tensordot(autos_b, M, axes=([autos_b.ndim - 1], [1]))


def n_output_channels(nbins: int, op: str) -> int:
    npairs = len(cross_pairs(nbins))
    if op == "none":          # autos only (the auto-only baseline arm)
        return nbins
    if op == "conv" or op == "product":
        return nbins + npairs
    if op == "both":
        return nbins + 2 * npairs
    if op == "product3":
        return nbins + len(cross_triples(nbins))
    raise ValueError(f"Unknown cross op={op!r}; expected one of {CROSS_OPS}")


# ---------------------------------------------------------------------------
# Apodization window (shared across backends; built in numpy, cast per backend)
# ---------------------------------------------------------------------------
def apod_window_np(npix: int, roll_frac: float = 0.10) -> np.ndarray:
    """Separable cosine (Hann-ramp) taper. roll_frac fraction of each edge ramps
    0->1 by a raised cosine; the interior is flat 1. Matches _apod_window_np in the
    L1 script but with the LOCKED 10% roll."""
    ramp = np.ones(npix, dtype=np.float32)
    n_roll = max(1, int(roll_frac * npix))
    cos_ramp = (0.5 * (1.0 - np.cos(np.pi * np.arange(n_roll) / n_roll))).astype(np.float32)
    ramp[:n_roll] = cos_ramp
    ramp[-n_roll:] = cos_ramp[::-1]
    return np.outer(ramp, ramp).astype(np.float32)


def _as_batched(autos):
    """Return (autos_b, was_batched) where autos_b is (B, H, W, n)."""
    if autos.ndim == 3:
        return autos[None], False
    if autos.ndim == 4:
        return autos, True
    raise ValueError(f"autos must be (H,W,n) or (B,H,W,n); got shape {tuple(autos.shape)}")


# ---------------------------------------------------------------------------
# numpy backend (reference)
# ---------------------------------------------------------------------------
def _conv_np(autos_b: np.ndarray, W: np.ndarray) -> np.ndarray:
    B, H, Wd, n = autos_b.shape
    pairs = cross_pairs(n)
    xa = (autos_b * W[None, :, :, None]).astype(np.float32)
    F = np.fft.rfft2(np.transpose(xa, (0, 3, 1, 2)), axes=(-2, -1))  # (B, n, H, W//2+1)
    cross = np.stack([F[:, i] * F[:, j] for i, j in pairs], axis=1)   # (B, npairs, ...)
    xc = np.fft.irfft2(cross, s=(H, Wd), axes=(-2, -1)).astype(np.float32)
    return np.transpose(xc, (0, 2, 3, 1))                            # (B, H, W, npairs)


def _product_np(autos_b: np.ndarray) -> np.ndarray:
    n = autos_b.shape[-1]
    pairs = cross_pairs(n)
    return np.stack([autos_b[..., i] * autos_b[..., j] for i, j in pairs], axis=-1).astype(np.float32)


def _product3_np(autos_b: np.ndarray) -> np.ndarray:
    n = autos_b.shape[-1]
    out = []
    for tup in cross_triples(n):
        m = autos_b[..., tup[0]].copy()
        for t in tup[1:]:
            m = m * autos_b[..., t]
        out.append(m)
    return np.stack(out, axis=-1).astype(np.float32)


def build_channels_np(autos: np.ndarray, op: str, roll_frac: float = 0.10,
                      bnt: bool = False) -> np.ndarray:
    """autos: (H,W,n) or (B,H,W,n) RAW auto maps. Returns RAW autos + cross channels.
    bnt=True applies the nulling transform to the autos FIRST (auto + cross channels
    then all live in BNT space)."""
    autos_b, was_batched = _as_batched(np.asarray(autos, dtype=np.float32))
    if _mix_requested(bnt):
        autos_b = apply_bnt_np(autos_b, mode=bnt)
    npix = autos_b.shape[1]
    parts = [autos_b]
    if op in ("conv", "both"):
        parts.append(_conv_np(autos_b, apod_window_np(npix, roll_frac)))
    if op in ("product", "both"):
        parts.append(_product_np(autos_b))
    if op == "product3":
        parts.append(_product3_np(autos_b))
    if op not in CROSS_OPS:
        raise ValueError(f"Unknown cross op={op!r}; expected one of {CROSS_OPS}")
    out = np.concatenate(parts, axis=-1)
    return out if was_batched else out[0]


# ---------------------------------------------------------------------------
# torch backend (L1, GPU)
# ---------------------------------------------------------------------------
def apod_window_torch(npix, roll_frac, device, dtype):
    import torch
    return torch.from_numpy(apod_window_np(npix, roll_frac)).to(device=device, dtype=dtype)


def _conv_torch(autos_b, W):
    import torch
    B, H, Wd, n = autos_b.shape
    pairs = cross_pairs(n)
    xa = autos_b * W[None, :, :, None]
    xa = xa.permute(0, 3, 1, 2)                                  # (B, n, H, W)
    F = torch.fft.rfft2(xa, dim=(-2, -1))                        # (B, n, H, W//2+1)
    cross = torch.stack([F[:, i] * F[:, j] for i, j in pairs], dim=1)
    xc = torch.fft.irfft2(cross, s=(H, Wd), dim=(-2, -1))
    return xc.permute(0, 2, 3, 1)                                # (B, H, W, npairs)


def _product_torch(autos_b):
    import torch
    n = autos_b.shape[-1]
    pairs = cross_pairs(n)
    return torch.stack([autos_b[..., i] * autos_b[..., j] for i, j in pairs], dim=-1)


def _product3_torch(autos_b):
    import torch
    n = autos_b.shape[-1]
    out = []
    for tup in cross_triples(n):
        m = autos_b[..., tup[0]]
        for t in tup[1:]:
            m = m * autos_b[..., t]
        out.append(m)
    return torch.stack(out, dim=-1)


def build_channels_torch(autos, op, roll_frac: float = 0.10, bnt: bool = False):
    """autos: torch tensor (H,W,n) or (B,H,W,n) RAW autos on the target device.
    bnt=True applies the nulling transform to the autos first."""
    import torch
    was_batched = autos.ndim == 4
    autos_b = autos if was_batched else autos.unsqueeze(0)
    if _mix_requested(bnt):
        autos_b = apply_bnt_torch(autos_b, mode=bnt)
    npix = autos_b.shape[1]
    parts = [autos_b]
    if op in ("conv", "both"):
        W = apod_window_torch(npix, roll_frac, autos_b.device, autos_b.dtype)
        parts.append(_conv_torch(autos_b, W))
    if op in ("product", "both"):
        parts.append(_product_torch(autos_b))
    if op == "product3":
        parts.append(_product3_torch(autos_b))
    if op not in CROSS_OPS:
        raise ValueError(f"Unknown cross op={op!r}; expected one of {CROSS_OPS}")
    out = torch.cat(parts, dim=-1)
    return out if was_batched else out.squeeze(0)


# ---------------------------------------------------------------------------
# jax backend (CNN, GPU)
# ---------------------------------------------------------------------------
def _conv_jax(autos_b, W):
    import jax.numpy as jnp
    B, H, Wd, n = autos_b.shape
    pairs = cross_pairs(n)
    xa = autos_b * W[None, :, :, None]
    xa = jnp.transpose(xa, (0, 3, 1, 2))                         # (B, n, H, W)
    F = jnp.fft.rfft2(xa, axes=(-2, -1))
    cross = jnp.stack([F[:, i] * F[:, j] for i, j in pairs], axis=1)
    xc = jnp.fft.irfft2(cross, s=(H, Wd), axes=(-2, -1))
    return jnp.transpose(xc, (0, 2, 3, 1))


def _product_jax(autos_b):
    import jax.numpy as jnp
    n = autos_b.shape[-1]
    pairs = cross_pairs(n)
    return jnp.stack([autos_b[..., i] * autos_b[..., j] for i, j in pairs], axis=-1)


def _product3_jax(autos_b):
    import jax.numpy as jnp
    n = autos_b.shape[-1]
    out = []
    for tup in cross_triples(n):
        m = autos_b[..., tup[0]]
        for t in tup[1:]:
            m = m * autos_b[..., t]
        out.append(m)
    return jnp.stack(out, axis=-1)


def build_channels_jax(autos, op, roll_frac: float = 0.10, bnt: bool = False):
    """autos: jax array (H,W,n) or (B,H,W,n) RAW autos.
    bnt=True applies the nulling transform to the autos first."""
    import jax.numpy as jnp
    was_batched = autos.ndim == 4
    autos_b = autos if was_batched else autos[None]
    if _mix_requested(bnt):
        autos_b = apply_bnt_jax(autos_b, mode=bnt)
    npix = autos_b.shape[1]
    parts = [autos_b]
    if op in ("conv", "both"):
        W = jnp.asarray(apod_window_np(npix, roll_frac))
        parts.append(_conv_jax(autos_b, W))
    if op in ("product", "both"):
        parts.append(_product_jax(autos_b))
    if op == "product3":
        parts.append(_product3_jax(autos_b))
    if op not in CROSS_OPS:
        raise ValueError(f"Unknown cross op={op!r}; expected one of {CROSS_OPS}")
    out = jnp.concatenate(parts, axis=-1)
    return out if was_batched else out[0]
