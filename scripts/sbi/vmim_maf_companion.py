"""Conditional MAF companion for VMIM compressor training (Haiku).

Drop-in alternative to the sbi_lens ConditionalRealNVP companion. Exposes the
SAME interface used in npe_cnn_nbody_tomo.py:

    nf = hk.without_apply_rng(hk.transform(
        lambda theta, y: conditional_maf_log_prob(theta, y, n_dim, n_cond, ...)))
    ...
    logp = nf.apply(params, theta, y)            # shape (batch,)

A MAF only needs density evaluation (no sampling) for VMIM, which is the cheap
("forward"/inverse-pass) direction of a MAF — one MADE pass per transform.

Design (careful, hand-rolled so it composes with the Haiku VMIM loop, no new dep):
- `n_transforms` autoregressive affine transforms; standard Germain MADE masking.
- Conditioning y enters every transform's first layer UNMASKED (full coupling).
- Fixed per-transform permutations of the theta coords (seeded) for mixing.
- Gaussian base. Last layer zero-initialised → each transform starts at identity
  → the whole flow starts EXACTLY at the standard normal (stable, testable).
- log_scale soft-clipped to [-7, 7] for numerical safety.

Validated by tests/test_vmim_maf_companion.py (shapes, finite, identity-init ==
N(0,I), triangular-Jacobian log-det consistency vs jax.jacfwd, and overfit).
"""
from __future__ import annotations

from functools import partial
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

_LOG2PI = float(np.log(2.0 * np.pi))


def _made_degrees(n_dim: int, hidden_sizes):
    """Germain MADE degrees: theta inputs 1..D, hidden spread over [1, D-1]."""
    degrees = [np.arange(1, n_dim + 1)]
    for h in hidden_sizes:
        degrees.append((np.arange(h) % max(1, n_dim - 1)) + 1)
    return degrees


def _made_masks(n_dim: int, hidden_sizes):
    """Return (input_masks list, out_mask). Mask[l] shape (out_units, in_units)."""
    deg = _made_degrees(n_dim, hidden_sizes)
    masks = []
    for l in range(len(deg) - 1):
        din, dout = deg[l], deg[l + 1]
        masks.append((dout[:, None] >= din[None, :]).astype(np.float32))
    # output: 2 params (shift, log_scale) per dim; strict '>' so out_i depends on theta_{<i}
    d_out = np.concatenate([np.arange(1, n_dim + 1), np.arange(1, n_dim + 1)])
    masks.append((d_out[:, None] > deg[-1][None, :]).astype(np.float32))
    return masks


class _ConditionalMADE(hk.Module):
    """One MADE: (theta[.,D], y[.,C]) -> (shift[.,D], log_scale[.,D]), autoregressive in theta."""

    def __init__(self, n_dim, n_cond, hidden_sizes, name=None):
        super().__init__(name=name)
        self.n_dim = int(n_dim)
        self.n_cond = int(n_cond)
        self.hidden_sizes = list(hidden_sizes)
        self._masks = [jnp.asarray(m) for m in _made_masks(n_dim, hidden_sizes)]

    def __call__(self, theta, y):
        sizes_in = [self.n_dim] + self.hidden_sizes
        sizes_out = self.hidden_sizes + [2 * self.n_dim]
        h = theta
        n_layers = len(self.hidden_sizes) + 1
        for li in range(n_layers):
            fin, fout = sizes_in[li], sizes_out[li]
            mask = self._masks[li]  # (fout, fin)
            is_out = (li == n_layers - 1)
            # zero-init the OUTPUT layer -> shift=log_scale=0 -> transform starts
            # at identity -> whole flow starts at N(0,I) (stable + unit-testable).
            w_init = jnp.zeros if is_out else hk.initializers.TruncatedNormal(
                1.0 / np.sqrt(max(1, fin)))
            w = hk.get_parameter(f"w{li}", (fin, fout), init=w_init)
            b = hk.get_parameter(f"b{li}", (fout,), init=jnp.zeros)
            pre = jnp.matmul(h, (w * mask.T)) + b
            if li == 0:
                # conditioning y: full (unmasked) into the first hidden layer
                wy = hk.get_parameter(
                    "wy", (self.n_cond, fout), init=hk.initializers.TruncatedNormal(
                        1.0 / np.sqrt(max(1, self.n_cond))))
                pre = pre + jnp.matmul(y, wy)
            if li < n_layers - 1:
                h = jax.nn.silu(pre)
            else:
                h = pre  # linear output
        shift, log_scale = jnp.split(h, 2, axis=-1)
        log_scale = 7.0 * jnp.tanh(log_scale / 7.0)  # soft clip to (-7, 7)
        return shift, log_scale

    @staticmethod
    def zero_last_layer_name(idx):
        return f"_made_{idx}/w{0}"


class ConditionalMAF(hk.Module):
    """Stack of autoregressive affine transforms; .log_prob(theta, y) -> (batch,)."""

    def __init__(self, n_dim, n_cond, n_transforms=8, hidden=(256, 256),
                 perm_seed=0, name=None):
        super().__init__(name=name)
        self.n_dim = int(n_dim)
        self.n_cond = int(n_cond)
        self.n_transforms = int(n_transforms)
        self.hidden = tuple(hidden)
        rng = np.random.default_rng(perm_seed)
        # transform 0 = identity order; the rest = random fixed permutations
        perms = [np.arange(self.n_dim)]
        for _ in range(self.n_transforms - 1):
            perms.append(rng.permutation(self.n_dim))
        self._perms = [jnp.asarray(p) for p in perms]

    def to_base(self, theta, y):
        """Map data theta -> base z (the density-eval direction); returns (z, logdet)."""
        z = theta
        logdet = jnp.zeros(theta.shape[:-1], dtype=theta.dtype)
        for k in range(self.n_transforms):
            z = z[..., self._perms[k]]
            made = _ConditionalMADE(self.n_dim, self.n_cond, self.hidden, name=f"made_{k}")
            shift, log_scale = made(z, y)
            z = (z - shift) * jnp.exp(-log_scale)
            logdet = logdet - jnp.sum(log_scale, axis=-1)
        return z, logdet

    def log_prob(self, theta, y):
        z, logdet = self.to_base(theta, y)
        base = -0.5 * jnp.sum(z ** 2 + _LOG2PI, axis=-1)
        return base + logdet


def conditional_maf_log_prob(theta, y, n_dim, n_cond, n_transforms=8,
                             hidden=(256, 256)):
    """Functional entry point matching the RealNVP companion's call signature."""
    return ConditionalMAF(n_dim, n_cond, n_transforms=n_transforms,
                          hidden=hidden).log_prob(theta, y)
