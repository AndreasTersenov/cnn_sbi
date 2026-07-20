#!/usr/bin/env python3
"""GPU-native posterior metrics (jnp), drop-in for the CPU numpy path in the per-patch
sweeps. The CPU path (compute_fom3/fom2d/marginal_stats in train_jaxili_from_compressed)
transfers 30000x6 samples device->host and runs np.cov/slogdet on CPU per patch — which is
the bottleneck on a load-heavy node (the GPU sits ~idle). This keeps everything on device and
transfers only the ~17 final scalars per patch.

Numerically matches the CPU reference up to float32-vs-float64 accumulation (validated by
validate_gpu_metrics.py to rtol<1e-3; typical ~1e-4). Conventions copied EXACTLY:
  - FoM3  = exp(-0.5*logdet(cov3)), cov via jnp.cov (ddof=1), valid iff slogdet sign>0
  - FoM2D = 1/sqrt(det(cov2)) per pair (0,1),(0,2),(1,2); cov2 = submatrix of cov6 (== np.cov of pair)
  - sigma = std(ddof=0) per param   (marginal_stats uses np.std, ddof=0 — NOT sqrt(diag cov))
  - bias  = mean - FIDUCIAL ; pull = bias/sigma
"""
from __future__ import annotations
import numpy as np

PARAM_KEYS = ["Omega_m", "sigma_8", "w_0", "h_0", "n_s", "Omega_b"]
FOM2D_PAIRS = [(0, 1), (0, 2), (1, 2)]


def all_metrics_gpu(samples, fiducial) -> dict:
    """samples: jnp (N,6) (may contain non-finite rows; they are dropped, matching CPU).
    Returns a flat dict identical in keys to the CPU combination."""
    import jax.numpy as jnp
    s = samples
    s = s[jnp.all(jnp.isfinite(s), axis=1)]            # drop non-finite (small host sync on count)
    cov6 = jnp.cov(s, rowvar=False)                    # ddof=1, matches np.cov
    sign3, logdet3 = jnp.linalg.slogdet(cov6[:3, :3])
    fom3 = jnp.where(sign3 > 0, jnp.exp(-0.5 * logdet3), jnp.nan)
    f2 = []
    for i, j in FOM2D_PAIRS:
        det = cov6[i, i] * cov6[j, j] - cov6[i, j] * cov6[j, i]
        f2.append(jnp.where(det > 0, 1.0 / jnp.sqrt(det), jnp.nan))
    sigma = jnp.std(s, axis=0)                          # ddof=0, matches np.std
    mean = jnp.mean(s, axis=0)
    # pack into ONE vector -> single device->host transfer
    packed = jnp.concatenate([jnp.array([fom3, (sign3 > 0).astype(jnp.float32),
                                         f2[0], f2[1], f2[2]]), sigma, mean])
    p = np.asarray(packed)                              # the only transfer per patch
    fid = np.asarray(fiducial, float)
    sig = p[5:11]; mu = p[11:17]; bias = mu - fid
    out = {"fom3": float(p[0]), "valid_fom3": bool(p[1] > 0.5),
           "fom2d_Omega_m_sigma_8": float(p[2]), "fom2d_Omega_m_w_0": float(p[3]),
           "fom2d_sigma_8_w_0": float(p[4])}
    for k, name in enumerate(PARAM_KEYS):
        out[f"sig_{name}"] = float(sig[k])
        out[f"mean_{name}"] = float(mu[k])
        out[f"bias_{name}"] = float(bias[k])
        out[f"pull_{name}"] = float(bias[k] / sig[k]) if sig[k] > 0 else float("nan")
    return out


def all_metrics_cpu(samples, fiducial) -> dict:
    """CPU reference: exactly the existing train_jaxili functions, packed into the same dict."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_jaxili_from_compressed import compute_fom3, fom2d, marginal_stats
    s = np.asarray(samples)
    s = s[np.all(np.isfinite(s), axis=1)]
    f3 = compute_fom3(s); f2 = fom2d(s); mg = marginal_stats(s)
    fid = np.asarray(fiducial, float)
    out = {"fom3": f3["fom3"], "valid_fom3": f3["valid_fom3"], **f2}
    for k, name in enumerate(PARAM_KEYS):
        sig = mg["sigma"][name]; bias = mg["bias"][name]
        out[f"sig_{name}"] = sig
        out[f"mean_{name}"] = bias + fid[k]
        out[f"bias_{name}"] = bias
        out[f"pull_{name}"] = bias / sig if sig > 0 else float("nan")
    return out
