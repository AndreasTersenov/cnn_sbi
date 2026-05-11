#!/usr/bin/env python
"""
CNN (VMIM) + NPE for tomographic weak lensing cosmological inference.

Uses a pretrained CNN compressor (trained with VMIM loss) to compress
4-bin tomographic convergence maps to a low-dimensional summary vector,
then feeds it to a conditional RealNVP normalizing flow (JAX / Haiku)
for Neural Posterior Estimation of cosmological parameters.

Main stages:
 1. Set CUDA device
 2. Load observed (fiducial) 4-bin tomographic map and add shape noise
 3. Build CNN compressor and load pretrained VMIM weights
 4. Compress observed map, train set, and test set through the CNN
 5. Define & train conditional RealNVP normalizing flow for p(theta | y)
 6. Sample the posterior and produce contour
  plots

Requires:
  - Pretrained CNN compressor weights (from train_compressor_tomographic.py)
  - NbodyCosmogridDatasetTomo TFDS dataset (from tf_dataset_nbody_tomo.py)
  - sbi_lens (normalizing flow components)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import time
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional, Tuple

import wandb

import h5py
import haiku as hk
import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from jax.lib import xla_bridge
from tensorflow_probability.substrates import jax as tfp

from bnt_utils import (
    BNT_MATRIX_VERSION,
    apply_bnt_numpy,
    apply_bnt_tf,
    validate_bnt_configuration,
)

# NumPy 2.x compatibility for older TFP/JAX substrate code paths.
if not hasattr(np, "issctype"):
    def _np_issctype(rep) -> bool:
        try:
            return issubclass(np.dtype(rep).type, np.generic)
        except Exception:
            return False
    np.issctype = _np_issctype  # type: ignore[attr-defined]

# sbi_lens normalizing flow
from sbi_lens.normflow.models import AffineCoupling, ConditionalRealNVP
from sbi_lens.normflow.train_model import TrainModel

# Register the local TFDS dataset builder so tfds.load can find it
import tf_dataset_nbody_tomo as _tomo_builder  # noqa: F401, E402

tfb = tfp.bijectors
tfd = tfp.distributions

HARMONIC_CACHE_CHANNELS = 10


# =============================================================================
# CLI
# =============================================================================

def parse_tomo_bin_indices(spec: str) -> tuple[int, ...]:
    """Parse comma-separated tomographic bin indices."""
    values = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        b = int(token)
        if b < 1 or b > 4:
            raise ValueError(
                f"Invalid tomo bin '{b}' in --tomo-bin-indices. Allowed range: 1..4."
            )
        values.append(b)
    if not values:
        raise ValueError("--tomo-bin-indices must contain at least one bin.")
    deduped = []
    seen = set()
    for b in values:
        if b not in seen:
            deduped.append(b)
            seen.add(b)
    return tuple(deduped)


def parse_positive_int_list(spec: str, arg_name: str) -> tuple[int, ...]:
    """Parse comma-separated positive integers."""
    values = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 1:
            raise ValueError(
                f"Invalid value '{value}' in {arg_name}. Values must be >= 1."
            )
        values.append(value)
    if not values:
        raise ValueError(f"{arg_name} must contain at least one integer.")
    return tuple(values)


def parse_nonnegative_float_list(spec: str, arg_name: str) -> tuple[float, ...]:
    """Parse comma-separated finite non-negative floats."""
    values = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if not np.isfinite(value):
            raise ValueError(f"Invalid value '{token}' in {arg_name}: not finite.")
        if value < 0.0:
            raise ValueError(
                f"Invalid value '{value}' in {arg_name}. Values must be >= 0."
            )
        values.append(value)
    if not values:
        raise ValueError(f"{arg_name} must contain at least one float.")
    return tuple(values)


def parse_positive_float_list(spec: str, arg_name: str) -> tuple[float, ...]:
    """Parse comma-separated finite positive floats."""
    values = parse_nonnegative_float_list(spec, arg_name)
    if any(v <= 0.0 for v in values):
        raise ValueError(f"{arg_name} values must be > 0.")
    return values


def allocate_stage_steps(total_steps: int, stage_fracs: tuple[float, ...]) -> tuple[int, ...]:
    """Allocate integer stage steps from fractions, preserving total exactly."""
    raw = [float(total_steps) * frac for frac in stage_fracs]
    floored = [int(np.floor(v)) for v in raw]
    remainder = int(total_steps - sum(floored))
    # Distribute leftover steps to largest residuals first.
    residual_order = sorted(
        range(len(raw)),
        key=lambda idx: (raw[idx] - floored[idx]),
        reverse=True,
    )
    for idx in residual_order[:remainder]:
        floored[idx] += 1
    return tuple(int(v) for v in floored)


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"batch(\d+)\.pkl$", path.name)
    return int(match.group(1)) if match is not None else -1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CNN (VMIM) + NPE for tomographic weak lensing"
    )

    # Hardware
    p.add_argument("--cuda-visible-devices", type=str, default="0")

    # Survey / map configuration
    p.add_argument("--field-size", type=int, default=10, help="Field size in degrees")
    p.add_argument("--field-npix", type=int, default=80, help="Pixels per side")
    p.add_argument("--nside", type=int, default=512, help="HEALPix NSIDE")
    p.add_argument("--sigma-e", type=float, default=0.26,
                    help="Shape noise dispersion per component")
    p.add_argument("--galaxy-density", type=float, default=30 / 4,
                    help="Galaxy number density [arcmin^{-2}]")
    p.add_argument("--nbins", type=int, default=4, help="Number of tomographic bins")
    p.add_argument("--n-cosmo", type=int, default=6,
                    help="Number of cosmological parameters")

    # Map kind
    p.add_argument("--map-kind", type=str, default="nbody",
                    choices=["nbody", "nbody_with_baryon_ia", "gaussian"])

    # Compressor
    p.add_argument("--compressor-dim", type=int, default=6,
                    help="CNN compressor output dimension")
    p.add_argument(
        "--compressor-arch",
        type=str,
        default="plain",
        choices=["plain", "resnet_small", "resnet18", "resnet34", "resnet50", "resnet50_gn"],
        help=(
            "Compressor architecture family: "
            "'plain' (existing 3-conv CNN), "
            "'resnet_small' (handcrafted residual CNN), "
            "'resnet18'/'resnet34'/'resnet50' (canonical Haiku ResNets, BatchNorm), "
            "'resnet50_gn' (custom ResNet50 with GroupNorm — for cosmology-batched inputs)."
        ),
    )
    p.add_argument(
        "--compressor-conv-channels",
        type=str,
        default="32,64,128",
        help="Comma-separated Conv2D channel widths for compressor trunk",
    )
    p.add_argument(
        "--compressor-dense-width",
        type=int,
        default=64,
        help="Hidden width of the compressor dense head",
    )
    p.add_argument(
        "--compressor-pool-window",
        type=int,
        default=16,
        help="AvgPool window size in compressor head",
    )
    p.add_argument(
        "--compressor-pool-stride",
        type=int,
        default=8,
        help="AvgPool stride in compressor head",
    )
    p.add_argument(
        "--resnet-small-channels",
        type=str,
        default="64,128,256",
        help="Comma-separated stage channels for --compressor-arch=resnet_small",
    )
    p.add_argument(
        "--resnet-small-blocks",
        type=str,
        default="2,2,2",
        help="Comma-separated residual blocks per stage for resnet_small",
    )
    p.add_argument(
        "--resnet-head-width",
        type=int,
        default=256,
        help="Dense head width used by ResNet compressors before output projection",
    )
    p.add_argument(
        "--resnet-v2",
        action="store_true",
        help=(
            "Use ResNet-v2 variant for canonical ResNets "
            "(--compressor-arch=resnet18/resnet34/resnet50)"
        ),
    )
    p.add_argument("--compressor-params", type=str,
                    default="/home/tersenov/software/cnn_sbi/tomo/save_params/"
                            "vmim/nbody/sigma_0.26/gal_density_30/bin_4/"
                            "params_nd_compressor_batch150000.pkl",
                    help="Path to pretrained compressor params pickle")
    p.add_argument("--compressor-state", type=str,
                    default="/home/tersenov/software/cnn_sbi/tomo/save_params/"
                            "vmim/nbody/sigma_0.26/gal_density_30/bin_4/"
                            "opt_state_resnet_batch150000.pkl",
                    help="Path to pretrained compressor state pickle")

    # Paths
    p.add_argument("--cosmogrid-meta", type=str,
                    default="/home/tersenov/CosmoGridV1/CosmoGridV1_metainfo.h5")
    p.add_argument("--fiducial-map", type=str,
                    default="/home/tersenov/CosmoGridV1/stage3_forecast/fiducial/"
                            "cosmo_fiducial/perm_0000/"
                            "projected_probes_maps_nobaryons512.h5")
    p.add_argument("--save-dir", type=str,
                    default="/home/tersenov/software/cnn_sbi/scripts/sbi/save_params")
    p.add_argument("--posterior-out", type=str,
                    default="posterior_cnn_tomo.npy")
    p.add_argument("--figure-out", type=str,
                    default="posterior_cnn_tomo.png")
    p.add_argument("--cache-dir", type=str, default=None,
                    help="Directory to cache compressed datasets "
                         "(skip recomputation)")
    p.add_argument(
        "--tfds-name",
        type=str,
        default="NbodyCosmogridDatasetTomo/grid",
        help="TFDS dataset name/config for training and validation maps",
    )
    p.add_argument(
        "--compressor-train-split",
        type=str,
        default="train",
        help="TFDS split used to train the compressor",
    )
    p.add_argument(
        "--compressor-val-split",
        type=str,
        default="test",
        help="TFDS split used for compressor validation/test loss",
    )
    p.add_argument(
        "--nde-train-split",
        type=str,
        default="train",
        help="TFDS split used to build NDE training summaries",
    )
    p.add_argument(
        "--nde-val-split",
        type=str,
        default="test",
        help="TFDS split used to build NDE validation summaries",
    )
    p.add_argument(
        "--require-disjoint-train-examples",
        action="store_true",
        help=(
            "Require zero overlap of exact training examples between "
            "--compressor-train-split and --nde-train-split "
            "(example identity = (cosmology, patch))."
        ),
    )
    p.add_argument(
        "--tomo-bin-indices",
        type=str,
        default="1,2,3,4",
        help="Tomographic bins to use, e.g. '1,2,3,4' or '3'",
    )
    p.add_argument(
        "--apply-bnt",
        action="store_true",
        help="Apply BNT transform after shape-noise injection.",
    )
    p.add_argument(
        "--full-sphere-cross-cache",
        type=str,
        default=None,
        help=(
            "Path to a cache built by build_full_sphere_cross_cache.py. "
            "When provided, CNN maps are loaded from harmonic cache files "
            "(4 auto + 6 cross channels) instead of TFDS maps."
        ),
    )
    p.add_argument(
        "--cnn-map-route",
        type=str,
        default=None,
        choices=["tfds", "harmonic"],
        help=(
            "Force map input route. Defaults to 'harmonic' when "
            "--full-sphere-cross-cache is set, 'tfds' otherwise."
        ),
    )
    p.add_argument(
        "--harmonic-cache-regime",
        type=str,
        default=None,
        choices=["bnt", "nobnt"],
        help=(
            "Regime subdir used under --full-sphere-cross-cache. Defaults to "
            "'bnt' when --apply-bnt is set, else 'nobnt'."
        ),
    )
    p.add_argument(
        "--harmonic-obs-cosmo-id",
        type=str,
        default="cosmo_fiducial",
        help="Observed cosmology id when using --full-sphere-cross-cache.",
    )
    p.add_argument(
        "--harmonic-obs-perm",
        type=int,
        default=0,
        help="Observed realization perm when using --full-sphere-cross-cache.",
    )
    p.add_argument(
        "--harmonic-obs-patch-idx",
        type=int,
        default=0,
        help="Observed patch index when using --full-sphere-cross-cache.",
    )
    p.add_argument(
        "--harmonic-train-realizations-limit",
        type=int,
        default=None,
        help=(
            "Optional cap on harmonic-cache train realizations loaded. "
            "Useful for smoke tests."
        ),
    )
    p.add_argument(
        "--harmonic-val-realizations-limit",
        type=int,
        default=None,
        help=(
            "Optional cap on harmonic-cache val realizations loaded. "
            "Useful for smoke tests."
        ),
    )

    # Flow training hyperparameters
    p.add_argument("--total-steps", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr-init", type=float, default=1e-3)
    p.add_argument("--lr-end", type=float, default=1e-5)
    p.add_argument("--nvp-layers", type=int, default=4,
                    help="Number of RealNVP coupling layers")
    p.add_argument("--nvp-hidden", type=int, default=128,
                    help="Hidden layer width in coupling networks")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                    help="AdamW weight decay")
    p.add_argument("--grad-clip", type=float, default=1.0,
                    help="Global gradient norm clipping (0 = disabled)")
    p.add_argument("--patience", type=int, default=20,
                    help="Early stopping patience "
                         "(in val-check intervals, 0 = disabled)")
    p.add_argument("--seed", type=int, default=42)

    # Posterior sampling
    p.add_argument("--npe-samples", type=int, default=100_000)

    # Weights & Biases
    p.add_argument("--wandb-project", type=str, default="cnn-npe-tomo",
                    help="W&B project name")
    p.add_argument("--wandb-entity", type=str, default=None,
                    help="W&B entity (team or username)")
    p.add_argument("--wandb-run-name", type=str, default=None,
                    help="W&B run name (auto-generated if None)")
    p.add_argument("--no-wandb", action="store_true",
                    help="Disable W&B logging entirely")

    # Compressor training
    p.add_argument("--train-compressor", action="store_true",
                    help="Train the CNN compressor from scratch "
                         "(VMIM loss) before inference")
    p.add_argument("--compressor-steps", type=int, default=150_000,
                    help="Total training steps for the compressor")
    p.add_argument("--compressor-lr", type=float, default=5e-4,
                    help="Initial learning rate for compressor training")
    p.add_argument("--compressor-batch-size", type=int, default=128,
                    help="Batch size for compressor training")
    p.add_argument("--compressor-save-every", type=int, default=2000,
                    help="Save compressor checkpoint every N steps")
    p.add_argument(
        "--compressor-noise-curriculum",
        action="store_true",
        help=(
            "Train compressor with staged shape-noise levels that ramp to "
            "target --sigma-e."
        ),
    )
    p.add_argument(
        "--compressor-curriculum-sigma-factors",
        type=str,
        default="0.0,0.25,0.5,0.75,1.0",
        help=(
            "Comma-separated multipliers of --sigma-e for curriculum stages "
            "(used when --compressor-noise-curriculum is set)."
        ),
    )
    p.add_argument(
        "--compressor-curriculum-stage-fracs",
        type=str,
        default="0.10,0.15,0.20,0.25,0.30",
        help=(
            "Comma-separated stage fractions summing to 1.0 for curriculum "
            "step allocation (used when --compressor-noise-curriculum is set)."
        ),
    )
    p.add_argument(
        "--compressor-paired-bnt-nobnt-consistency",
        action="store_true",
        help=(
            "Train compressor on paired no-BNT/BNT views of the same maps and "
            "optimize VMIM with an explicit summary consistency penalty."
        ),
    )
    p.add_argument(
        "--compressor-consistency-weight",
        type=float,
        default=0.1,
        help=(
            "Weight for paired summary consistency loss when "
            "--compressor-paired-bnt-nobnt-consistency is enabled."
        ),
    )
    p.add_argument(
        "--compressor-domain-adversarial",
        action="store_true",
        help=(
            "Enable a domain-adversarial head (BNT vs no-BNT) during compressor "
            "training to encourage domain-invariant summaries."
        ),
    )
    p.add_argument(
        "--compressor-domain-adv-weight",
        type=float,
        default=0.05,
        help=(
            "Adversarial weight for domain invariance. The compressor minimizes "
            "VMIM + consistency - w * domain_ce."
        ),
    )
    p.add_argument(
        "--compressor-domain-hidden",
        type=int,
        default=64,
        help="Hidden width of the domain-adversarial MLP head.",
    )
    p.add_argument(
        "--vmim-nf-hidden",
        type=int,
        default=128,
        help=(
            "Hidden width of the VMIM companion RealNVP auxiliary network "
            "(AffineCoupling layers=[vmim_nf_hidden]*2). Default 128. "
            "Increase (e.g. 256, 512) to test whether the VMIM bound is saturated."
        ),
    )
    p.add_argument(
        "--compressor-plot-contours",
        action="store_true",
        help="Plot compressor contour diagnostics at each compressor checkpoint",
    )

    # Execution flags
    p.add_argument("--no-train", action="store_true",
                    help="Load saved flow params instead of training")
    p.add_argument("--no-sample", action="store_true",
                    help="Skip posterior sampling")
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--plot", action="store_true",
                    help="Generate triangle plot")
    p.add_argument("--ds-batch-size", type=int, default=500,
                    help="Batch size for CNN compression of datasets")
    p.add_argument("--shuffle-theta-train", action="store_true",
                    help="Control test: shuffle training theta labels before "
                         "flow training (should degrade posterior quality)")
    p.add_argument("--standardize-summary", dest="standardize_summary",
                    action="store_true",
                    help="Z-score normalize compressed summaries using "
                         "training-set statistics before flow training")
    p.add_argument("--no-standardize-summary", dest="standardize_summary",
                    action="store_false",
                    help="Disable summary standardization")
    p.set_defaults(standardize_summary=True)
    p.add_argument("--summary-clip-value", type=float, default=5.0,
                    help="Clip standardized summary features to ±this value "
                         "(0 = disabled)")
    p.add_argument("--zero-mean-maps", dest="zero_mean_maps",
                    action="store_true",
                    help="Subtract the per-example, per-channel spatial mean "
                         "from input maps before the compressor (mass-sheet "
                         "degeneracy). Applied to observed and augmented "
                         "training/eval maps.")
    p.add_argument("--no-zero-mean-maps", dest="zero_mean_maps",
                    action="store_false",
                    help="Disable per-channel map demeaning (default).")
    p.set_defaults(zero_mean_maps=False)
    p.add_argument(
        "--harmonic-normalize-input-channels",
        dest="harmonic_normalize_input_channels",
        action="store_true",
        default=False,
        help=(
            "Harmonic-cache route only. Divide each input channel by its "
            "dataset-level RMS (computed from the training split, pooled over "
            "all spatial pixels and examples). This equalizes the gradient "
            "signal from auto-channels (C_ii, large amplitude) and "
            "cross-channels (C_ij i≠j, ~100× smaller amplitude) without "
            "discarding inter-example cosmological scale variation. "
            "The 10 per-channel RMS values are saved in the run meta JSON."
        ),
    )

    return p.parse_args()


# =============================================================================
# Environment setup
# =============================================================================

def setup_environment(cuda_devices: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    print(f"CUDA_VISIBLE_DEVICES = {cuda_devices}")
    # TF: restrict memory growth
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"TF GPU config: {e}")
    print(f"JAX backend    : {xla_bridge.get_backend().platform}")


# =============================================================================
# Survey helpers
# =============================================================================

def pixel_noise_sigma(sigma_e: float, galaxy_density: float,
                      field_size: float, field_npix: int) -> float:
    """Per-pixel noise standard deviation for shape noise."""
    reso_arcmin = field_size * 60.0 / field_npix  # arcmin / pixel
    return sigma_e / np.sqrt(galaxy_density * reso_arcmin ** 2)


# =============================================================================
# Observed (fiducial) map
# =============================================================================

def load_observed_map(
    meta_path: str,
    fid_path: str,
    field_size: int,
    field_npix: int,
    nside: int,
    nbins: int,
    tomo_bin_indices: tuple[int, ...],
    sigma_e: float,
    galaxy_density: float,
    rng_key: jax.Array,
    apply_bnt: bool = False,
    zero_mean_maps: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load fiducial 4-bin tomographic map, project, and add shape noise."""
    print("######## OBSERVED DATA ########")
    # Truth parameters
    with h5py.File(meta_path, "r") as f:
        ds = f["parameters"]["fiducial"]
        cosmo_params = np.array([
            ds["Om"], ds["s8"], ds["w0"],
            np.array(ds["H0"]) / 100.0,
            ds["ns"], ds["Ob"],
        ], dtype=np.float64).T
    truth = cosmo_params[0].copy()
    print(f"  Truth = {truth}")

    # Load & project each tomographic bin
    reso = field_size * 60.0 / field_npix
    proj = hp.projector.GnomonicProj(
        rot=[0, 0, 0], xsize=field_npix, ysize=field_npix, reso=reso,
    )

    if len(tomo_bin_indices) != nbins:
        raise ValueError(
            f"nbins={nbins} is inconsistent with selected bins {tomo_bin_indices}."
        )

    with h5py.File(fid_path, "r") as f:
        kg = f["kg"]
        proj_bins = []
        for b in tomo_bin_indices:
            full_map = np.array(kg[f"stage3_lensing{b}"])
            patch = proj.projmap(
                full_map, vec2pix_func=partial(hp.vec2pix, nside),
            )
            proj_bins.append(patch)

    # Stack to (H, W, nbins) and add shape noise
    m_data = np.stack(proj_bins, axis=-1).astype(np.float32)
    noise_std = pixel_noise_sigma(sigma_e, galaxy_density, field_size, field_npix)
    noise = jax.random.normal(rng_key, (field_npix, field_npix, nbins)) * noise_std
    m_data = np.array(jnp.asarray(m_data) + noise)
    if zero_mean_maps:
        # Mass-sheet degeneracy: each tomographic channel is recoverable only
        # up to an additive constant in real data. Remove the per-channel
        # spatial mean so the compressor cannot exploit absolute levels.
        per_channel_mean = m_data.mean(axis=(0, 1), keepdims=True)
        m_data = m_data - per_channel_mean
        residual = np.abs(m_data.mean(axis=(0, 1))).max()
        assert residual < 1e-5, (
            f"Observed map per-channel mean residual {residual:.3e} after "
            "demeaning exceeds tolerance."
        )
        print(f"  Applied zero-mean-maps to observed map "
              f"(subtracted means = {per_channel_mean.squeeze()})")
    if apply_bnt:
        m_data = apply_bnt_numpy(m_data)
    print(f"  Observed map shape = {m_data.shape}, "
          f"noise_std/pixel = {noise_std:.6f}")
    return m_data, cosmo_params, truth


def _read_harmonic_manifest(cache_dir: Path) -> Dict[str, object]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing harmonic-cache manifest at {manifest_path}. "
            "Build the cache with build_full_sphere_cross_cache.py first."
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    args_sha = payload.get("args_sha256")
    if not isinstance(args_sha, str) or not args_sha:
        raise ValueError(f"Manifest at {manifest_path} missing 'args_sha256'.")
    n_channels = int(payload.get("n_channels", -1))
    if n_channels != HARMONIC_CACHE_CHANNELS:
        raise ValueError(
            "Harmonic cache channel mismatch: expected "
            f"{HARMONIC_CACHE_CHANNELS}, got {n_channels}."
        )
    return payload


def _list_harmonic_cache_files(cache_dir: Path, regime: str, split: str) -> list[Path]:
    if regime not in ("bnt", "nobnt"):
        raise ValueError(f"regime must be 'bnt' or 'nobnt', got {regime}")
    split_dir = cache_dir / regime / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Harmonic cache split missing: {split_dir}. "
            "Did the build script complete this regime/split?"
        )
    files = sorted(p for p in split_dir.iterdir() if p.suffix == ".npz")
    if not files:
        raise FileNotFoundError(f"No .npz files found under {split_dir}.")
    return files


def _harmonic_random_flip(maps: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = maps
    flip_lr = rng.integers(0, 2, size=maps.shape[0]).astype(bool)
    flip_ud = rng.integers(0, 2, size=maps.shape[0]).astype(bool)
    if flip_lr.any():
        idx = np.where(flip_lr)[0]
        out = out.copy()
        out[idx] = out[idx, :, ::-1, :]
    if flip_ud.any():
        idx = np.where(flip_ud)[0]
        if out is maps:
            out = out.copy()
        out[idx] = out[idx, ::-1, :, :]
    return out


def _assert_zero_mean_patches(
    patches: np.ndarray,
    source: str,
    atol: float = 1e-4,
) -> None:
    residual = float(np.abs(patches.mean(axis=(1, 2))).max())
    if residual > atol:
        raise ValueError(
            "Harmonic cache zero-mean compatibility check failed for "
            f"{source}: max per-channel patch mean residual {residual:.3e} > {atol:.1e}."
        )


def iter_harmonic_examples(
    cache_dir: Path,
    regime: str,
    split: str,
    rng: np.random.Generator | None = None,
    flip: bool = True,
    max_realizations: int | None = None,
):
    files = _list_harmonic_cache_files(cache_dir, regime, split)
    if max_realizations is not None:
        files = files[:max_realizations]
    if rng is None:
        rng = np.random.default_rng(0)
    for f in files:
        with np.load(f, allow_pickle=False) as d:
            patches = np.asarray(d["patches"], dtype=np.float32)
            theta = np.asarray(d["theta"], dtype=np.float64)
        if patches.ndim != 4 or patches.shape[-1] != HARMONIC_CACHE_CHANNELS:
            raise ValueError(
                f"Unexpected patch shape in {f}: {patches.shape} "
                f"(expected (..., {HARMONIC_CACHE_CHANNELS}))."
            )
        _assert_zero_mean_patches(patches, str(f))
        if flip:
            patches = _harmonic_random_flip(patches, rng)
        yield patches, theta, str(f)


def compute_harmonic_channel_rms(
    cache_dir: Path,
    regime: str,
    split: str,
    max_realizations: int | None = None,
) -> np.ndarray:
    """Compute per-channel RMS over the training split (pooled over all pixels and examples).

    Returns shape (n_channels,) float32 array. Patches must be zero-mean
    (enforced by cache), so RMS == std here. Using RMS rather than per-example
    std ensures inter-example cosmological amplitude variation is preserved.
    """
    files = _list_harmonic_cache_files(cache_dir, regime, split)
    if max_realizations is not None:
        files = files[:max_realizations]
    if not files:
        raise ValueError(
            f"No harmonic cache files for channel-stats computation "
            f"(split={split}, regime={regime})."
        )
    sum_sq: np.ndarray | None = None
    n_pixels = 0
    for path in files:
        with np.load(path, allow_pickle=False) as d:
            patches = np.asarray(d["patches"], dtype=np.float32)  # (N, H, W, C)
        if sum_sq is None:
            sum_sq = np.zeros(patches.shape[-1], dtype=np.float64)
        sum_sq += np.sum(patches.astype(np.float64) ** 2, axis=(0, 1, 2))
        n_pixels += patches.shape[0] * patches.shape[1] * patches.shape[2]
    assert sum_sq is not None
    rms = np.sqrt(sum_sq / n_pixels).astype(np.float32)
    return rms


def _theta_batch_from_harmonic(theta: np.ndarray, n_samples: int) -> np.ndarray:
    theta_batch = np.broadcast_to(theta, (n_samples, theta.shape[0])).copy()
    theta_batch[:, 3] = theta_batch[:, 3] / 100.0
    return theta_batch


def _normalize_harmonic_split(
    split: str,
    arg_name: str,
    allowed: tuple[str, ...],
) -> str:
    mapping = {
        "train": "train",
        "val": "val",
        "validation": "val",
        "test": "val",
        "obs": "obs",
    }
    key = split.strip().lower()
    normalized = mapping.get(key)
    if normalized is None or normalized not in allowed:
        allowed_str = ", ".join(allowed)
        raise ValueError(
            f"{arg_name}={split!r} is invalid for harmonic-cache route. "
            f"Allowed: {allowed_str}."
        )
    if normalized != split:
        print(f"  Overriding {arg_name} from '{split}' to '{normalized}' for harmonic cache.")
    return normalized


def load_observed_from_harmonic_cache(
    cache_dir: Path,
    regime: str,
    cosmo_id: str = "cosmo_fiducial",
    perm: int = 0,
    patch_idx: int = 0,
    meta_path: str | None = None,
    channel_scale: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    print("######## OBSERVED DATA (harmonic cache) ########")
    npz_path = cache_dir / regime / "obs" / f"{cosmo_id}_perm{perm}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Observed cache file missing: {npz_path}. The cache must include "
            f"cosmo_id={cosmo_id} (split=obs) for regime={regime}."
        )
    with np.load(npz_path, allow_pickle=False) as d:
        patches = np.asarray(d["patches"], dtype=np.float32)
        theta_npz = np.asarray(d["theta"], dtype=np.float64)
    _assert_zero_mean_patches(patches, str(npz_path))
    if patch_idx < 0 or patch_idx >= patches.shape[0]:
        raise IndexError(
            f"--harmonic-obs-patch-idx={patch_idx} out of range "
            f"[0, {patches.shape[0]})."
        )
    m_data = patches[patch_idx]
    if channel_scale is not None:
        m_data = m_data / channel_scale

    if meta_path is not None and Path(meta_path).exists():
        with h5py.File(meta_path, "r") as f:
            ds = f["parameters"]["fiducial"]
            cosmo_params = np.array(
                [
                    ds["Om"],
                    ds["s8"],
                    ds["w0"],
                    np.array(ds["H0"]) / 100.0,
                    ds["ns"],
                    ds["Ob"],
                ],
                dtype=np.float64,
            ).T
        truth = cosmo_params[0].copy()
    else:
        truth = theta_npz.copy()
        truth[3] = truth[3] / 100.0
        cosmo_params = truth.reshape(1, -1)

    print(f"  Source = {npz_path} (patch {patch_idx} of {patches.shape[0]})")
    print(f"  Truth  = {truth}")
    print(f"  Observed map shape = {m_data.shape}")
    return m_data, cosmo_params, truth


def build_harmonic_batch_iterator(
    cache_dir: Path,
    regime: str,
    split: str,
    batch_size: int,
    seed: int,
    flip: bool,
    max_realizations: int | None = None,
    channel_scale: np.ndarray | None = None,
) -> Iterator[Dict[str, np.ndarray]]:
    files = _list_harmonic_cache_files(cache_dir, regime, split)
    if max_realizations is not None:
        files = files[:max_realizations]
    if not files:
        raise ValueError(
            f"No harmonic cache files available for split={split}, regime={regime}."
        )
    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(len(files))
        for idx in order:
            path = files[int(idx)]
            with np.load(path, allow_pickle=False) as d:
                maps_np = np.asarray(d["patches"], dtype=np.float32)
                theta_np = np.asarray(d["theta"], dtype=np.float64)
            _assert_zero_mean_patches(maps_np, str(path))
            if channel_scale is not None:
                maps_np = maps_np / channel_scale
            if flip:
                maps_np = _harmonic_random_flip(maps_np, rng)
            perm = rng.permutation(maps_np.shape[0])
            maps_np = maps_np[perm]
            theta_batch = _theta_batch_from_harmonic(theta_np, maps_np.shape[0])[perm]
            for start in range(0, maps_np.shape[0], batch_size):
                end = start + batch_size
                yield {"maps": maps_np[start:end], "theta": theta_batch[start:end]}


def compress_dataset_from_harmonic_cache(
    cache_dir: Path,
    regime: str,
    split: str,
    compressor,
    comp_params: hk.Params,
    comp_state: hk.State,
    ds_batch_size: int,
    rng: np.random.Generator | None = None,
    flip: bool = True,
    max_realizations: int | None = None,
    channel_scale: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    print(f"  Loading harmonic cache [{regime}/{split}] ...")
    theta_list = []
    x_list = []
    n_processed = 0
    n_realizations = 0
    t0 = time.time()
    first_batch_reported = False
    for maps_np, theta_np, _path in iter_harmonic_examples(
        cache_dir=cache_dir,
        regime=regime,
        split=split,
        rng=rng,
        flip=flip,
        max_realizations=max_realizations,
    ):
        if channel_scale is not None:
            maps_np = maps_np / channel_scale
        theta_batch = _theta_batch_from_harmonic(theta_np, maps_np.shape[0])
        for start in range(0, maps_np.shape[0], ds_batch_size):
            end = start + ds_batch_size
            maps_batch = maps_np[start:end]
            theta_chunk = theta_batch[start:end]
            if np.isnan(maps_batch).any():
                print("    [!] Skipped batch with NaN maps")
                continue
            if not first_batch_reported:
                per_map_means = maps_batch.mean(axis=(1, 2))
                print(
                    "    First-batch per-channel spatial-mean stats: "
                    f"abs max = {np.abs(per_map_means).max():.3e}, "
                    f"mean = {per_map_means.mean():.3e}"
                )
                first_batch_reported = True
            comp_y, _ = compressor.apply(comp_params, comp_state, None, maps_batch)
            x_list.append(np.array(comp_y))
            theta_list.append(theta_chunk)
            n_processed += len(theta_chunk)
        n_realizations += 1
        if n_realizations % 100 == 0:
            elapsed = time.time() - t0
            print(
                f"    Processed {n_realizations} realizations / "
                f"{n_processed} patches ({elapsed:.1f}s)"
            )

    if not theta_list or not x_list:
        raise RuntimeError(
            f"No harmonic examples were processed for split={split}, regime={regime}."
        )

    elapsed = time.time() - t0
    print(
        f"  Done: {n_realizations} realizations / "
        f"{n_processed} patches in {elapsed:.1f}s"
    )
    return {
        "theta": np.concatenate(theta_list, axis=0),
        "x": np.concatenate(x_list, axis=0),
    }


# =============================================================================
# CNN Compressor (Haiku)
# =============================================================================

class CompressorCNN2D(hk.Module):
    """CNN compressor: (B, H, W, nbins) -> (B, output_dim)."""

    def __init__(
        self,
        output_dim: int,
        conv_channels: tuple[int, ...],
        dense_width: int,
        pool_window: int,
        pool_stride: int,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.output_dim = output_dim
        self.conv_channels = conv_channels
        self.dense_width = dense_width
        self.pool_window = pool_window
        self.pool_stride = pool_stride

    def __call__(self, x):
        net_x = x
        for channels in self.conv_channels:
            net_x = hk.Conv2D(channels, 3, 2)(net_x)
            net_x = jax.nn.leaky_relu(net_x)
        net_x = hk.AvgPool(self.pool_window, self.pool_stride, "SAME")(net_x)
        net_x = hk.Flatten()(net_x)
        net_x = hk.Linear(self.dense_width)(net_x)
        net_x = jax.nn.leaky_relu(net_x)
        net_x = hk.Linear(self.output_dim)(net_x)
        return net_x.squeeze()


class ResidualBlock2D(hk.Module):
    """Simple residual block used by handcrafted resnet_small compressor."""

    def __init__(self, channels: int, stride: int = 1, name: str | None = None):
        super().__init__(name=name)
        self.channels = channels
        self.stride = stride

    def __call__(self, x):
        shortcut = x
        y = hk.Conv2D(self.channels, 3, stride=self.stride, padding="SAME")(x)
        y = jax.nn.leaky_relu(y)
        y = hk.Conv2D(self.channels, 3, stride=1, padding="SAME")(y)

        if self.stride != 1 or x.shape[-1] != self.channels:
            shortcut = hk.Conv2D(
                self.channels, 1, stride=self.stride, padding="SAME"
            )(shortcut)
        return jax.nn.leaky_relu(y + shortcut)


class CompressorResNetSmall(hk.Module):
    """Handcrafted residual CNN compressor for (B, H, W, nbins) maps."""

    def __init__(
        self,
        output_dim: int,
        stage_channels: tuple[int, ...],
        blocks_per_stage: tuple[int, ...],
        head_width: int,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.output_dim = output_dim
        self.stage_channels = stage_channels
        self.blocks_per_stage = blocks_per_stage
        self.head_width = head_width

    def __call__(self, x):
        y = hk.Conv2D(self.stage_channels[0], 7, stride=2, padding="SAME")(x)
        y = jax.nn.leaky_relu(y)
        y = hk.max_pool(y, window_shape=3, strides=2, padding="SAME")

        for stage_idx, (channels, n_blocks) in enumerate(
            zip(self.stage_channels, self.blocks_per_stage)
        ):
            for block_idx in range(n_blocks):
                stride = 2 if (stage_idx > 0 and block_idx == 0) else 1
                y = ResidualBlock2D(channels=channels, stride=stride)(
                    y
                )

        y = jnp.mean(y, axis=(1, 2))
        y = hk.Linear(self.head_width)(y)
        y = jax.nn.leaky_relu(y)
        y = hk.Linear(self.output_dim)(y)
        return y.squeeze()


def _gn_groups(channels: int, target: int = 32) -> int:
    """Pick a GroupNorm group count that divides ``channels`` (≤ ``target``)."""
    g = min(target, channels)
    while g > 1 and (channels % g) != 0:
        g -= 1
    return max(g, 1)


class _BottleneckGN(hk.Module):
    """ResNet-v1 bottleneck block with hk.GroupNorm replacing BatchNorm."""

    def __init__(
        self,
        channels: int,
        bottleneck_channels: int,
        stride: int,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.channels = channels
        self.bottleneck_channels = bottleneck_channels
        self.stride = stride

    def __call__(self, x):
        out = hk.Conv2D(
            self.bottleneck_channels, 1, stride=1, padding="SAME", with_bias=False,
        )(x)
        out = hk.GroupNorm(_gn_groups(self.bottleneck_channels))(out)
        out = jax.nn.relu(out)

        out = hk.Conv2D(
            self.bottleneck_channels, 3, stride=self.stride, padding="SAME",
            with_bias=False,
        )(out)
        out = hk.GroupNorm(_gn_groups(self.bottleneck_channels))(out)
        out = jax.nn.relu(out)

        out = hk.Conv2D(
            self.channels, 1, stride=1, padding="SAME", with_bias=False,
        )(out)
        out = hk.GroupNorm(_gn_groups(self.channels))(out)

        if self.stride != 1 or x.shape[-1] != self.channels:
            shortcut = hk.Conv2D(
                self.channels, 1, stride=self.stride, padding="SAME",
                with_bias=False,
            )(x)
            shortcut = hk.GroupNorm(_gn_groups(self.channels))(shortcut)
        else:
            shortcut = x
        return jax.nn.relu(out + shortcut)


class CompressorResNet50GN(hk.Module):
    """Custom ResNet-50 (v1) compressor with GroupNorm instead of BatchNorm.

    Mirrors the canonical 4-stage layout (3, 4, 6, 3 bottlenecks) at output
    channels (256, 512, 1024, 2048) with bottleneck channels = output / 4.
    Replacing BN with GN avoids cross-cosmology contamination of the
    normalization statistics — the failure mode that collapsed the canonical
    ResNet-50 compressor on harmonic auto+cross inputs (FoM3 ≈ 700).
    """

    def __init__(
        self,
        output_dim: int,
        head_width: int,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.output_dim = output_dim
        self.head_width = head_width

    def __call__(self, x, is_training: bool = True):  # is_training kept for API parity
        del is_training
        y = hk.Conv2D(64, 7, stride=2, padding="SAME", with_bias=False)(x)
        y = hk.GroupNorm(_gn_groups(64))(y)
        y = jax.nn.relu(y)
        y = hk.max_pool(y, window_shape=3, strides=2, padding="SAME")

        stage_specs = [
            (256, 3, 1),
            (512, 4, 2),
            (1024, 6, 2),
            (2048, 3, 2),
        ]
        for channels, n_blocks, first_stride in stage_specs:
            for block_idx in range(n_blocks):
                stride = first_stride if block_idx == 0 else 1
                y = _BottleneckGN(
                    channels=channels,
                    bottleneck_channels=channels // 4,
                    stride=stride,
                )(y)

        y = jnp.mean(y, axis=(1, 2))
        y = hk.Linear(self.head_width)(y)
        y = jax.nn.relu(y)
        y = hk.Linear(self.output_dim)(y)
        return y.squeeze()


class CompressorCanonicalResNet(hk.Module):
    """Canonical Haiku ResNet compressor for selected depth."""

    def __init__(
        self,
        output_dim: int,
        arch: str,
        head_width: int,
        resnet_v2: bool,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.output_dim = output_dim
        self.arch = arch
        self.head_width = head_width
        self.resnet_v2 = resnet_v2

    def __call__(self, x, is_training: bool):
        if self.arch == "resnet18":
            backbone_cls = hk.nets.ResNet18
        elif self.arch == "resnet34":
            backbone_cls = hk.nets.ResNet34
        elif self.arch == "resnet50":
            backbone_cls = hk.nets.ResNet50
        else:
            raise ValueError(f"Unsupported canonical ResNet arch '{self.arch}'")

        backbone = backbone_cls(
            num_classes=self.head_width,
            resnet_v2=self.resnet_v2,
            initial_conv_config={
                "output_channels": 64,
                "kernel_shape": 7,
                "stride": 2,
                "padding": "SAME",
            },
            name=f"{self.arch}_backbone",
        )
        y = backbone(x, is_training=is_training)
        y = jax.nn.leaky_relu(y)
        y = hk.Linear(self.output_dim)(y)
        return y.squeeze()


def build_compressors(
    dim: int,
    arch: str,
    conv_channels: tuple[int, ...],
    dense_width: int,
    pool_window: int,
    pool_stride: int,
    resnet_small_channels: tuple[int, ...],
    resnet_small_blocks: tuple[int, ...],
    resnet_head_width: int,
    resnet_v2: bool,
):
    """Build train/eval Haiku compressor transforms for selected architecture."""
    if arch == "plain":
        def _forward(y):
            return CompressorCNN2D(
                dim,
                conv_channels=conv_channels,
                dense_width=dense_width,
                pool_window=pool_window,
                pool_stride=pool_stride,
                name="compressor_plain",
            )(y)
        compressor_train = hk.transform_with_state(_forward)
        compressor_eval = hk.transform_with_state(_forward)
        return compressor_train, compressor_eval

    if arch == "resnet_small":
        def _forward(y):
            return CompressorResNetSmall(
                dim,
                stage_channels=resnet_small_channels,
                blocks_per_stage=resnet_small_blocks,
                head_width=resnet_head_width,
                name="compressor_resnet_small",
            )(y)
        compressor_train = hk.transform_with_state(_forward)
        compressor_eval = hk.transform_with_state(_forward)
        return compressor_train, compressor_eval

    if arch in ("resnet18", "resnet34", "resnet50"):
        def _forward_train(y):
            return CompressorCanonicalResNet(
                dim,
                arch=arch,
                head_width=resnet_head_width,
                resnet_v2=resnet_v2,
                name=f"compressor_{arch}",
            )(y, is_training=True)

        def _forward_eval(y):
            return CompressorCanonicalResNet(
                dim,
                arch=arch,
                head_width=resnet_head_width,
                resnet_v2=resnet_v2,
                name=f"compressor_{arch}",
            )(y, is_training=False)

        compressor_train = hk.transform_with_state(_forward_train)
        compressor_eval = hk.transform_with_state(_forward_eval)
        return compressor_train, compressor_eval

    if arch == "resnet50_gn":
        def _forward(y):
            return CompressorResNet50GN(
                dim,
                head_width=resnet_head_width,
                name="compressor_resnet50_gn",
            )(y)
        compressor_train = hk.transform_with_state(_forward)
        compressor_eval = hk.transform_with_state(_forward)
        return compressor_train, compressor_eval

    raise ValueError(f"Unknown --compressor-arch '{arch}'")


def load_compressor_params(
    params_path: str, state_path: str,
) -> Tuple[hk.Params, hk.State]:
    """Load pretrained compressor params and state from pickle files."""
    print(f"  Loading compressor params: {params_path}")
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    print(f"  Loading compressor state:  {state_path}")
    with open(state_path, "rb") as f:
        state = pickle.load(f)
    return params, state


def file_sha256(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_cnn_cache_metadata(
    args: argparse.Namespace,
    compressor_source: str,
    compressor_params_path: Optional[str],
    compressor_state_path: Optional[str],
    tomo_bin_indices: tuple[int, ...],
    cnn_map_route: str = "tfds",
    full_sphere_cache_manifest_sha256: str = "",
    harmonic_regime: str = "",
    cnn_input_channels: Optional[int] = None,
) -> Dict[str, object]:
    """Build metadata used to validate cached compressed datasets."""
    params_path = Path(compressor_params_path).resolve() if compressor_params_path else None
    state_path = Path(compressor_state_path).resolve() if compressor_state_path else None

    meta: Dict[str, object] = {
        "compressor_source": compressor_source,
        "compressor_arch": str(args.compressor_arch),
        "compressor_dim": int(args.compressor_dim),
        "compressor_conv_channels": str(args.compressor_conv_channels),
        "compressor_dense_width": int(args.compressor_dense_width),
        "compressor_pool_window": int(args.compressor_pool_window),
        "compressor_pool_stride": int(args.compressor_pool_stride),
        "compressor_noise_curriculum": int(bool(args.compressor_noise_curriculum)),
        "compressor_curriculum_sigma_factors": str(
            args.compressor_curriculum_sigma_factors
        ),
        "compressor_curriculum_stage_fracs": str(
            args.compressor_curriculum_stage_fracs
        ),
        "compressor_paired_bnt_nobnt_consistency": int(
            bool(args.compressor_paired_bnt_nobnt_consistency)
        ),
        "compressor_consistency_weight": float(args.compressor_consistency_weight),
        "compressor_domain_adversarial": int(bool(args.compressor_domain_adversarial)),
        "compressor_domain_adv_weight": float(args.compressor_domain_adv_weight),
        "compressor_domain_hidden": int(args.compressor_domain_hidden),
        "resnet_small_channels": str(args.resnet_small_channels),
        "resnet_small_blocks": str(args.resnet_small_blocks),
        "resnet_head_width": int(args.resnet_head_width),
        "resnet_v2": int(bool(args.resnet_v2)),
        "tfds_name": str(args.tfds_name),
        "compressor_train_split": str(args.compressor_train_split),
        "compressor_val_split": str(args.compressor_val_split),
        "nde_train_split": str(args.nde_train_split),
        "nde_val_split": str(args.nde_val_split),
        "require_disjoint_train_examples": int(bool(args.require_disjoint_train_examples)),
        "tomo_bin_indices": ",".join(str(b) for b in tomo_bin_indices),
        "map_kind": str(args.map_kind),
        "cnn_map_route": str(cnn_map_route),
        "full_sphere_cache_manifest_sha256": str(full_sphere_cache_manifest_sha256),
        "harmonic_regime": str(harmonic_regime),
        "field_size": int(args.field_size),
        "field_npix": int(args.field_npix),
        "nbins": int(args.nbins),
        "cnn_input_channels": int(
            args.nbins if cnn_input_channels is None else cnn_input_channels
        ),
        "sigma_e": float(args.sigma_e),
        "galaxy_density": float(args.galaxy_density),
        "apply_bnt": bool(args.apply_bnt),
        "bnt_matrix_version": BNT_MATRIX_VERSION if args.apply_bnt else "none",
        "zero_mean_maps": int(bool(args.zero_mean_maps)),
        "harmonic_normalize_input_channels": int(
            bool(getattr(args, "harmonic_normalize_input_channels", False))
        ),
        "compressor_params_path": str(params_path) if params_path else "",
        "compressor_state_path": str(state_path) if state_path else "",
        "compressor_params_sha256": (
            file_sha256(params_path) if params_path and params_path.exists() else ""
        ),
        "compressor_state_sha256": (
            file_sha256(state_path) if state_path and state_path.exists() else ""
        ),
    }
    return meta


def compare_cache_metadata(
    meta_npz: np.lib.npyio.NpzFile,
    expected: Dict[str, object],
) -> Tuple[bool, list[str]]:
    """Compare cached metadata against expected values."""
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        if key not in meta_npz.files:
            mismatches.append(f"missing:{key}")
            continue
        cached_raw = meta_npz[key]
        if isinstance(expected_value, float):
            cached_value = float(cached_raw)
            if abs(cached_value - expected_value) > 1e-12:
                mismatches.append(
                    f"{key}={cached_value} (expected {expected_value})"
                )
        elif isinstance(expected_value, int):
            cached_value = int(cached_raw)
            if cached_value != expected_value:
                mismatches.append(
                    f"{key}={cached_value} (expected {expected_value})"
                )
        else:
            cached_value = str(cached_raw)
            if cached_value != str(expected_value):
                mismatches.append(
                    f"{key}={cached_value} (expected {expected_value})"
                )

    return len(mismatches) == 0, mismatches


def _normalize_tfds_id(raw_id: object) -> str:
    if isinstance(raw_id, bytes):
        return raw_id.decode("utf-8")
    return str(raw_id)


def _collect_split_identity(
    tfds_name: str,
    split: str,
    theta_decimals: int = 8,
) -> Tuple[set[str], set[tuple[float, ...]]]:
    import tensorflow_datasets as tfds

    ds = tfds.load(
        tfds_name,
        split=split,
        read_config=tfds.ReadConfig(add_tfds_id=True),
    )
    id_set: set[str] = set()
    theta_set: set[tuple[float, ...]] = set()
    for ex in tfds.as_numpy(ds):
        id_set.add(_normalize_tfds_id(ex["tfds_id"]))
        theta = np.asarray(ex["theta"], dtype=np.float64).copy()
        # Match pipeline convention where H0 is expressed as h0.
        theta[3] = theta[3] / 100.0
        theta_set.add(tuple(np.round(theta, theta_decimals).tolist()))
    return id_set, theta_set


def audit_train_split_overlap(
    tfds_name: str,
    compressor_train_split: str,
    nde_train_split: str,
) -> Dict[str, object]:
    """Audit overlap of exact examples between compressor and NDE train splits."""
    ids_comp, theta_comp = _collect_split_identity(tfds_name, compressor_train_split)
    ids_nde, theta_nde = _collect_split_identity(tfds_name, nde_train_split)
    shared_ids = ids_comp.intersection(ids_nde)
    shared_theta = theta_comp.intersection(theta_nde)
    return {
        "compressor_train_split": str(compressor_train_split),
        "nde_train_split": str(nde_train_split),
        "compressor_train_examples": int(len(ids_comp)),
        "nde_train_examples": int(len(ids_nde)),
        "shared_example_count": int(len(shared_ids)),
        "compressor_theta_count": int(len(theta_comp)),
        "nde_theta_count": int(len(theta_nde)),
        "shared_theta_count": int(len(shared_theta)),
    }


def log_compressor_checkpoint_provenance(
    params_path: str,
    state_path: str,
) -> None:
    """Print and log compressor checkpoint provenance info."""
    p_params = Path(params_path).resolve()
    p_state = Path(state_path).resolve()
    params_size = p_params.stat().st_size if p_params.exists() else -1
    state_size = p_state.stat().st_size if p_state.exists() else -1
    params_hash = file_sha256(p_params) if p_params.exists() else ""
    state_hash = file_sha256(p_state) if p_state.exists() else ""

    print("  Compressor checkpoint provenance:")
    print(f"    params: {p_params} ({params_size} bytes)")
    print(f"    state:  {p_state} ({state_size} bytes)")
    print(f"    params_sha256: {params_hash[:16]}...")
    print(f"    state_sha256:  {state_hash[:16]}...")

    wandb.config.update({
        "compressor/params_path": str(p_params),
        "compressor/state_path": str(p_state),
        "compressor/params_size_bytes": params_size,
        "compressor/state_size_bytes": state_size,
        "compressor/params_sha256": params_hash,
        "compressor/state_sha256": state_hash,
    }, allow_val_change=True)


# =============================================================================
# Compressor training (VMIM)
# =============================================================================

def train_compressor_vmim(
    compressor,
    augmentation_fn,
    n_cosmo: int,
    compressor_dim: int,
    field_npix: int,
    nbins: int,
    total_steps: int,
    lr_init: float,
    batch_size: int,
    save_every: int,
    save_dir: Path,
    m_data_obs: np.ndarray,
    truth: np.ndarray,
    param_names: list[str],
    tfds_name: str,
    compressor_train_split: str,
    compressor_val_split: str,
    plot_contours: bool = False,
    noise_curriculum_stages: Optional[list[Dict[str, object]]] = None,
    paired_bnt_nobnt_consistency: bool = False,
    consistency_weight: float = 0.0,
    domain_adversarial: bool = False,
    domain_adv_weight: float = 0.0,
    domain_hidden: int = 64,
    vmim_nf_hidden: int = 128,
    dataset_iter_factory: Optional[
        Callable[[str, int], Iterator[Dict[str, np.ndarray]]]
    ] = None,
) -> Tuple[hk.Params, hk.State]:
    """Train the CNN compressor from scratch using VMIM loss.

    Follows the same recipe as train_compressor_tomographic.py:
      - Companion RealNVP (4 layers, [vmim_nf_hidden]*2, silu) for VMIM objective
      - Piecewise constant LR schedule (init × 0.7 at every 10% milestone)
      - Adam optimizer
      - TrainModel from sbi_lens
    """
    from tqdm import tqdm

    print("######## TRAINING COMPRESSOR (VMIM) ########")
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Companion normalizing flow for VMIM ---
    bijector_fn = partial(
        AffineCoupling, layers=[vmim_nf_hidden] * 2, activation=jax.nn.silu,
    )
    NF_compressor = partial(
        ConditionalRealNVP, n_layers=4, bijector_fn=bijector_fn,
    )

    class FlowNdCompressor(hk.Module):
        def __call__(self, y):
            return NF_compressor(n_cosmo)(y)

    nf = hk.without_apply_rng(
        hk.transform(
            lambda theta, y: FlowNdCompressor()(y).log_prob(theta).squeeze()
        )
    )

    # --- Initialize params ---
    params_cnn, state_cnn = compressor.init(
        jax.random.PRNGKey(0),
        y=0.5 * jnp.ones([1, field_npix, field_npix, nbins]),
    )
    params_nf = nf.init(
        jax.random.PRNGKey(0),
        theta=0.5 * jnp.ones([1, n_cosmo]),
        y=0.5 * jnp.ones([1, compressor_dim]),
    )
    params_merged = hk.data_structures.merge(params_cnn, params_nf)
    del params_cnn, params_nf

    n_params = sum(x.size for x in jax.tree.leaves(params_merged))
    print(f"  Compressor + NF parameters: {n_params:,}")

    # --- Piecewise constant LR schedule (same as original) ---
    schedule_steps = total_steps - total_steps // 3
    lr_schedule = optax.piecewise_constant_schedule(
        init_value=lr_init,
        boundaries_and_scales={
            int(schedule_steps * f): 0.7
            for f in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        },
    )
    optimizer = optax.adam(learning_rate=lr_schedule)
    opt_state = optimizer.init(params_merged)

    model = TrainModel(
        compressor=compressor, nf=nf,
        optimizer=optimizer, loss_name="train_compressor_vmim",
    )
    update = jax.jit(model.update)

    paired_training = bool(paired_bnt_nobnt_consistency or domain_adversarial)
    if paired_training:
        print(
            "  Paired BNT/no-BNT compressor training enabled "
            f"(consistency_weight={consistency_weight:.4g}, "
            f"domain_adversarial={bool(domain_adversarial)}, "
            f"domain_adv_weight={domain_adv_weight:.4g})."
        )

    domain_params = None
    domain_optimizer = None
    domain_opt_state = None

    def _domain_logits(dparams, y):
        hidden = jnp.tanh(jnp.matmul(y, dparams["w1"]) + dparams["b1"])
        return jnp.matmul(hidden, dparams["w2"]) + dparams["b2"]

    if domain_adversarial:
        key_dom = jax.random.PRNGKey(17)
        key_w1, key_w2 = jax.random.split(key_dom)
        scale_in = 1.0 / np.sqrt(float(max(1, compressor_dim)))
        scale_hidden = 1.0 / np.sqrt(float(max(1, domain_hidden)))
        domain_params = {
            "w1": jax.random.normal(key_w1, (compressor_dim, domain_hidden)) * scale_in,
            "b1": jnp.zeros((domain_hidden,), dtype=jnp.float32),
            "w2": jax.random.normal(key_w2, (domain_hidden, 2)) * scale_hidden,
            "b2": jnp.zeros((2,), dtype=jnp.float32),
        }
        domain_optimizer = optax.adam(learning_rate=lr_schedule)
        domain_opt_state = domain_optimizer.init(domain_params)

        @jax.jit
        def _update_domain_head(
            current_domain_params,
            current_domain_opt_state,
            model_params,
            state_resnet,
            x_nobnt,
            x_bnt,
        ):
            def _domain_loss_fn(dparams):
                y_nobnt, _ = compressor.apply(model_params, state_resnet, None, x_nobnt)
                y_bnt, _ = compressor.apply(model_params, state_resnet, None, x_bnt)
                y_domain = jnp.concatenate(
                    [
                        jax.lax.stop_gradient(y_nobnt),
                        jax.lax.stop_gradient(y_bnt),
                    ],
                    axis=0,
                )
                labels = jnp.concatenate(
                    [
                        jnp.zeros((y_nobnt.shape[0],), dtype=jnp.int32),
                        jnp.ones((y_bnt.shape[0],), dtype=jnp.int32),
                    ],
                    axis=0,
                )
                logits = _domain_logits(dparams, y_domain)
                ce = jnp.mean(
                    optax.softmax_cross_entropy_with_integer_labels(logits, labels)
                )
                acc = jnp.mean((jnp.argmax(logits, axis=-1) == labels).astype(jnp.float32))
                return ce, acc

            (domain_loss, domain_acc), domain_grads = jax.value_and_grad(
                _domain_loss_fn, has_aux=True
            )(current_domain_params)
            domain_updates, new_domain_opt_state = domain_optimizer.update(  # type: ignore[union-attr]
                domain_grads, current_domain_opt_state, current_domain_params
            )
            new_domain_params = optax.apply_updates(current_domain_params, domain_updates)
            return domain_loss, domain_acc, new_domain_params, new_domain_opt_state

    def _paired_objective(
        model_params,
        state_resnet,
        theta,
        x_nobnt,
        x_bnt,
        current_domain_params,
    ):
        y_nobnt, state_mid = compressor.apply(model_params, state_resnet, None, x_nobnt)
        y_bnt, state_out = compressor.apply(model_params, state_mid, None, x_bnt)

        log_prob_nobnt = nf.apply(model_params, theta, y_nobnt)
        log_prob_bnt = nf.apply(model_params, theta, y_bnt)
        vmim_loss = -0.5 * (jnp.mean(log_prob_nobnt) + jnp.mean(log_prob_bnt))

        consistency_loss = jnp.mean(jnp.sum((y_nobnt - y_bnt) ** 2, axis=-1))
        total_loss = vmim_loss + float(consistency_weight) * consistency_loss

        domain_ce = jnp.asarray(0.0, dtype=vmim_loss.dtype)
        domain_acc = jnp.asarray(0.0, dtype=vmim_loss.dtype)
        if domain_adversarial:
            y_domain = jnp.concatenate([y_nobnt, y_bnt], axis=0)
            labels = jnp.concatenate(
                [
                    jnp.zeros((y_nobnt.shape[0],), dtype=jnp.int32),
                    jnp.ones((y_bnt.shape[0],), dtype=jnp.int32),
                ],
                axis=0,
            )
            logits = _domain_logits(current_domain_params, y_domain)
            domain_ce = jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(logits, labels)
            )
            domain_acc = jnp.mean((jnp.argmax(logits, axis=-1) == labels).astype(jnp.float32))
            total_loss = total_loss - float(domain_adv_weight) * domain_ce

        return total_loss, state_out, vmim_loss, consistency_loss, domain_ce, domain_acc

    @jax.jit
    def _update_paired(
        model_params,
        current_opt_state,
        state_resnet,
        theta,
        x_nobnt,
        x_bnt,
        current_domain_params,
    ):
        def _loss_fn(params):
            total, state_out, vmim_loss, consistency_loss, domain_ce, domain_acc = (
                _paired_objective(
                    params,
                    state_resnet,
                    theta,
                    x_nobnt,
                    x_bnt,
                    current_domain_params,
                )
            )
            aux = (state_out, vmim_loss, consistency_loss, domain_ce, domain_acc)
            return total, aux

        (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(model_params)
        updates, new_opt_state = optimizer.update(grads, current_opt_state, model_params)
        new_params = optax.apply_updates(model_params, updates)
        state_out, vmim_loss, consistency_loss, domain_ce, domain_acc = aux
        return (
            loss,
            new_params,
            new_opt_state,
            state_out,
            vmim_loss,
            consistency_loss,
            domain_ce,
            domain_acc,
        )

    @jax.jit
    def _eval_paired(
        model_params,
        state_resnet,
        theta,
        x_nobnt,
        x_bnt,
        current_domain_params,
    ):
        total, _, vmim_loss, consistency_loss, domain_ce, domain_acc = _paired_objective(
            model_params,
            state_resnet,
            theta,
            x_nobnt,
            x_bnt,
            current_domain_params,
        )
        return total, vmim_loss, consistency_loss, domain_ce, domain_acc

    if dataset_iter_factory is None:
        import tensorflow_datasets as tfds

        def _dataset_iter(split: str, shuffle_buffer: int, aug_fn):
            ds = tfds.load(tfds_name, split=split)
            ds = ds.repeat().shuffle(shuffle_buffer)
            ds = ds.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.batch(batch_size)
            ds = ds.prefetch(tf.data.AUTOTUNE)
            return iter(tfds.as_numpy(ds))
    else:
        if paired_training:
            raise ValueError(
                "paired BNT/no-BNT compressor training is not supported for "
                "custom dataset_iter_factory routes."
            )

        def _dataset_iter(split: str, _shuffle_buffer: int, _aug_fn):
            return dataset_iter_factory(split, batch_size)

    stage_specs: list[Dict[str, object]]
    if noise_curriculum_stages is None:
        stage_specs = [
            {
                "name": "target_noise",
                "stage_index": 1,
                "sigma_factor": 1.0,
                "sigma_e": None,
                "steps": int(total_steps),
                "augmentation_fn": augmentation_fn,
            }
        ]
    else:
        stage_specs = []
        for stage in noise_curriculum_stages:
            stage_steps = int(stage.get("steps", 0))
            if stage_steps <= 0:
                continue
            if "augmentation_fn" not in stage:
                raise ValueError("Curriculum stage is missing augmentation_fn.")
            stage_specs.append(stage)
        if not stage_specs:
            raise ValueError("Noise curriculum enabled but no non-empty stages were built.")
        planned_steps = sum(int(stage["steps"]) for stage in stage_specs)
        if planned_steps != int(total_steps):
            raise ValueError(
                "Noise curriculum stage steps must sum to --compressor-steps "
                f"({planned_steps} != {total_steps})."
            )
        print("  Noise curriculum schedule:")
        for stage in stage_specs:
            sigma_e_stage = stage.get("sigma_e")
            sigma_desc = (
                f"{float(sigma_e_stage):.6f}" if sigma_e_stage is not None else "target"
            )
            print(
                "   - "
                f"stage {int(stage.get('stage_index', -1))}: "
                f"factor={float(stage.get('sigma_factor', 1.0)):.4f}, "
                f"sigma_e={sigma_desc}, "
                f"steps={int(stage['steps'])}"
            )

    # --- Training loop ---
    store_loss = []
    loss_train_hist = []
    loss_test_hist = []

    stop_training = False
    global_step = 0
    last_step = 0
    last_saved_step = 0
    for stage in stage_specs:
        stage_idx = int(stage.get("stage_index", len(stage_specs)))
        stage_name = str(stage.get("name", f"stage_{stage_idx}"))
        stage_sigma_factor = float(stage.get("sigma_factor", 1.0))
        stage_sigma_e = stage.get("sigma_e")
        stage_steps = int(stage["steps"])
        stage_aug = stage["augmentation_fn"]
        stage_sigma_desc = (
            f"{float(stage_sigma_e):.6f}" if stage_sigma_e is not None else "target"
        )

        wandb.log(
            {
                "compressor/curriculum_stage_index": stage_idx,
                "compressor/curriculum_stage_sigma_factor": stage_sigma_factor,
                "compressor/step": global_step,
            }
        )
        print(
            f"  Stage {stage_idx}/{len(stage_specs)} [{stage_name}] | "
            f"sigma_factor={stage_sigma_factor:.4f} | "
            f"sigma_e={stage_sigma_desc} | "
            f"steps={stage_steps}"
        )

        ds_train_iter = _dataset_iter(compressor_train_split, 800, stage_aug)
        ds_test_iter = _dataset_iter(compressor_val_split, 200, stage_aug)

        for _ in tqdm(range(stage_steps), desc=f"Compressor[{stage_name}]"):
            step = global_step + 1
            ex = next(ds_train_iter)
            if paired_training:
                has_nan = bool(np.isnan(ex["maps_nobnt"]).any()) or bool(
                    np.isnan(ex["maps_bnt"]).any()
                )
            else:
                has_nan = bool(np.isnan(ex["maps"]).any())
            if has_nan:
                global_step = step
                continue

            domain_loss = 0.0
            domain_acc = 0.0
            if domain_adversarial:
                (
                    domain_loss,
                    domain_acc,
                    domain_params,
                    domain_opt_state,
                ) = _update_domain_head(
                    domain_params,
                    domain_opt_state,
                    params_merged,
                    state_cnn,
                    ex["maps_nobnt"],
                    ex["maps_bnt"],
                )

            if paired_training:
                (
                    b_loss,
                    params_merged,
                    opt_state,
                    state_cnn,
                    vmim_train_loss,
                    consistency_train_loss,
                    domain_ce_train,
                    domain_acc_train,
                ) = _update_paired(
                    params_merged,
                    opt_state,
                    state_cnn,
                    ex["theta"],
                    ex["maps_nobnt"],
                    ex["maps_bnt"],
                    domain_params,
                )
            else:
                b_loss, params_merged, opt_state, state_cnn = update(
                    model_params=params_merged,
                    opt_state=opt_state,
                    theta=ex["theta"],
                    x=ex["maps"],
                    state_resnet=state_cnn,
                )
                vmim_train_loss = b_loss
                consistency_train_loss = jnp.asarray(0.0, dtype=b_loss.dtype)
                domain_ce_train = jnp.asarray(0.0, dtype=b_loss.dtype)
                domain_acc_train = jnp.asarray(0.0, dtype=b_loss.dtype)
            store_loss.append(float(b_loss))

            if jnp.isnan(b_loss):
                print("  [!] NaN loss — stopping compressor training")
                stop_training = True
                global_step = step
                break
            last_step = step
            global_step = step

            # Log to wandb every 100 steps
            if step % 100 == 0:
                wandb.log({
                    "compressor/train_loss": float(b_loss),
                    "compressor/train_vmim_loss": float(vmim_train_loss),
                    "compressor/train_consistency_loss": float(consistency_train_loss),
                    "compressor/train_domain_ce": float(domain_ce_train),
                    "compressor/train_domain_acc": float(domain_acc_train),
                    "compressor/domain_head_loss": float(domain_loss),
                    "compressor/domain_head_acc": float(domain_acc),
                    "compressor/step": step,
                    "compressor/curriculum_stage_index": stage_idx,
                    "compressor/curriculum_stage_sigma_factor": stage_sigma_factor,
                })

            if step % save_every == 0:
                # Save checkpoint
                ckpt_params = save_dir / f"params_nd_compressor_batch{step}.pkl"
                with open(ckpt_params, "wb") as f:
                    pickle.dump(params_merged, f)
                ckpt_state = save_dir / f"opt_state_resnet_batch{step}.pkl"
                with open(ckpt_state, "wb") as f:
                    pickle.dump(state_cnn, f)
                last_saved_step = step

                # Test loss
                ex_test = next(ds_test_iter)
                if paired_training:
                    (
                        b_loss_test,
                        vmim_test_loss,
                        consistency_test_loss,
                        domain_ce_test,
                        domain_acc_test,
                    ) = _eval_paired(
                        params_merged,
                        state_cnn,
                        ex_test["theta"],
                        ex_test["maps_nobnt"],
                        ex_test["maps_bnt"],
                        domain_params,
                    )
                else:
                    b_loss_test, _, _, _ = update(
                        model_params=params_merged,
                        opt_state=opt_state,
                        theta=ex_test["theta"],
                        x=ex_test["maps"],
                        state_resnet=state_cnn,
                    )
                    vmim_test_loss = b_loss_test
                    consistency_test_loss = jnp.asarray(0.0, dtype=b_loss_test.dtype)
                    domain_ce_test = jnp.asarray(0.0, dtype=b_loss_test.dtype)
                    domain_acc_test = jnp.asarray(0.0, dtype=b_loss_test.dtype)
                loss_train_hist.append(float(b_loss))
                loss_test_hist.append(float(b_loss_test))

                wandb.log({
                    "compressor/test_loss": float(b_loss_test),
                    "compressor/test_vmim_loss": float(vmim_test_loss),
                    "compressor/test_consistency_loss": float(consistency_test_loss),
                    "compressor/test_domain_ce": float(domain_ce_test),
                    "compressor/test_domain_acc": float(domain_acc_test),
                    "compressor/step": step,
                    "compressor/curriculum_stage_index": stage_idx,
                    "compressor/curriculum_stage_sigma_factor": stage_sigma_factor,
                })
                print(
                    f"  Step {step:6d} | "
                    f"train {b_loss:.4f} | test {b_loss_test:.4f}"
                )

                # Save loss curves
                np.save(save_dir / "loss_compressor_train.npy",
                        np.array(loss_train_hist))
                np.save(save_dir / "loss_compressor_test.npy",
                        np.array(loss_test_hist))

                # Optional contour diagnostic from compressor + companion NF
                if plot_contours:
                    _plot_compressor_contours(
                        compressor, params_merged, state_cnn, nf,
                        m_data_obs, field_npix, nbins, n_cosmo, compressor_dim,
                        truth, param_names, save_dir, step,
                    )
        if stop_training:
            break

    if last_step > 0 and last_saved_step != last_step:
        ckpt_params = save_dir / f"params_nd_compressor_batch{last_step}.pkl"
        with open(ckpt_params, "wb") as f:
            pickle.dump(params_merged, f)
        ckpt_state = save_dir / f"opt_state_resnet_batch{last_step}.pkl"
        with open(ckpt_state, "wb") as f:
            pickle.dump(state_cnn, f)
        print(f"  Saved final checkpoint @ step {last_step}.")

    print(f"  Compressor training done ({len(store_loss)} steps).")
    wandb.run.summary["compressor/total_steps"] = len(store_loss)
    return params_merged, state_cnn


def _plot_compressor_contours(
    compressor, params_merged, state_cnn, nf,
    m_data_obs, field_npix, nbins, n_cosmo, compressor_dim,
    truth, param_names, save_dir, step,
):
    """Plot posterior contours from the companion NF during training."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from getdist import MCSamples, plots as gplot
    except ImportError:
        return

    y, _ = compressor.apply(
        params_merged, state_cnn, None,
        m_data_obs.reshape([1, field_npix, field_npix, nbins]),
    )
    nvp_sample = hk.transform(
        lambda x: hk.Module.__init_subclass__  # dummy; use class below
    )
    # Re-use the same companion NF architecture for sampling
    bijector_fn = partial(
        AffineCoupling, layers=[128] * 2, activation=jax.nn.silu,
    )
    NF_comp = partial(
        ConditionalRealNVP, n_layers=4, bijector_fn=bijector_fn,
    )

    class _FlowSample(hk.Module):
        def __call__(self, x):
            return NF_comp(n_cosmo)(x)

    nvp_sample_fn = hk.transform(
        lambda x: _FlowSample()(x).sample(100_000, seed=hk.next_rng_key())
    )
    sample_nd = nvp_sample_fn.apply(
        params_merged, rng=jax.random.PRNGKey(43),
        x=y * jnp.ones([100_000, compressor_dim]),
    )
    # Remove NaN samples
    idx = jnp.where(jnp.isnan(sample_nd))[0]
    sample_nd = jnp.delete(sample_nd, idx, axis=0)

    truth_arr = np.array(truth)
    mcsamples = MCSamples(
        samples=np.array(sample_nd), names=param_names, labels=param_names,
    )
    g = gplot.get_subplot_plotter(subplot_size=1.5)
    g.triangle_plot(
        [mcsamples], filled=True,
        markers=truth_arr,
        marker_args={"color": "red", "lw": 1.2},
    )
    fig_path = save_dir / f"contour_compressor_batch{step}.png"
    plt.savefig(fig_path, dpi=100, bbox_inches="tight")
    wandb.log({
        "compressor/contour_plot": wandb.Image(str(fig_path)),
        "compressor/step": step,
    })
    plt.close()


# =============================================================================
# Data augmentation (TF) — same as train_compressor_tomographic.py
# =============================================================================

def build_augmentation(
    map_kind: str,
    sigma_e: float,
    galaxy_density: float,
    field_size: int,
    field_npix: int,
    nbins: int,
    tomo_bin_indices: tuple[int, ...],
    apply_bnt: bool = False,
    sigma_e_override: Optional[float] = None,
    paired_bnt_nobnt_consistency: bool = False,
    zero_mean_maps: bool = False,
):
    """Build the TF augmentation pipeline for the tomographic dataset."""
    effective_sigma_e = sigma_e if sigma_e_override is None else sigma_e_override
    noise_std = effective_sigma_e / jnp.sqrt(
        galaxy_density * (field_size * 60 / field_npix) ** 2
    )

    map_key = {
        "nbody": "map_nbody",
        "nbody_with_baryon_ia": "map_nbody_w_baryon_ia",
        "gaussian": "map_gaussian",
    }[map_kind]
    gather_indices = tf.constant([b - 1 for b in tomo_bin_indices], dtype=tf.int32)

    def _rescale_h(theta):
        return tf.tensor_scatter_nd_update(theta, [[3]], [theta[3] / 100.0])

    def _flip_pair(x_nobnt, x_bnt):
        flip_lr = tf.less(tf.random.uniform([], 0.0, 1.0), 0.5)
        flip_ud = tf.less(tf.random.uniform([], 0.0, 1.0), 0.5)

        def _apply_lr(x):
            return tf.cond(flip_lr, lambda: tf.image.flip_left_right(x), lambda: x)

        def _apply_ud(x):
            return tf.cond(flip_ud, lambda: tf.image.flip_up_down(x), lambda: x)

        x_nobnt = _apply_ud(_apply_lr(x_nobnt))
        x_bnt = _apply_ud(_apply_lr(x_bnt))
        return x_nobnt, x_bnt

    if paired_bnt_nobnt_consistency:
        def augmentation(example):
            x = tf.gather(example[map_key], gather_indices, axis=-1)
            noise = tf.random.normal(
                shape=(field_npix, field_npix, nbins),
                stddev=noise_std,
            )
            x_nobnt = x + noise
            if zero_mean_maps:
                # Demean once, before BNT split. BNT is linear across channels
                # so B(x - m·1) is also zero-mean per channel — both views see
                # a mass-sheet-degeneracy-respecting input.
                x_nobnt = x_nobnt - tf.reduce_mean(
                    x_nobnt, axis=[0, 1], keepdims=True,
                )
            x_bnt = apply_bnt_tf(x_nobnt)
            x_nobnt, x_bnt = _flip_pair(x_nobnt, x_bnt)
            theta = _rescale_h(example["theta"])
            return {"maps_nobnt": x_nobnt, "maps_bnt": x_bnt, "theta": theta}

        return augmentation

    def augmentation_noise(example):
        x = tf.gather(example[map_key], gather_indices, axis=-1)
        x += tf.random.normal(
            shape=(field_npix, field_npix, nbins), stddev=noise_std,
        )
        if zero_mean_maps:
            x = x - tf.reduce_mean(x, axis=[0, 1], keepdims=True)
        if apply_bnt:
            x = apply_bnt_tf(x)
        return {"maps": x, "theta": example["theta"]}

    def augmentation_flip(example):
        x = example["maps"]
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
        return {"maps": x, "theta": example["theta"]}

    def rescale_h(example):
        x = _rescale_h(example["theta"])
        return {"maps": example["maps"], "theta": x}

    def augmentation(example):
        return rescale_h(augmentation_flip(augmentation_noise(example)))

    return augmentation


# =============================================================================
# Dataset compression
# =============================================================================

def compress_dataset(
    tfds_name: str,
    split: str,
    augmentation_fn,
    compressor,
    comp_params: hk.Params,
    comp_state: hk.State,
    ds_batch_size: int,
    paired_map_view: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Load TFDS dataset, apply augmentation, compress via CNN.

    Returns dict with 'theta' (N, n_cosmo) and 'x' (N, compressor_dim).
    """
    import tensorflow_datasets as tfds

    print(f"  Loading {tfds_name} [{split}] ...")
    ds = tfds.load(tfds_name, split=split)
    ds = ds.map(augmentation_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(ds_batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    theta_list = []
    x_list = []
    n_processed = 0
    t0 = time.time()
    first_batch_reported = False

    for example in ds.as_numpy_iterator():
        if "maps" in example:
            maps_np = example["maps"]   # (B, H, W, nbins)
        elif "maps_nobnt" in example and "maps_bnt" in example:
            if paired_map_view == "nobnt":
                maps_np = example["maps_nobnt"]
            elif paired_map_view == "bnt":
                maps_np = example["maps_bnt"]
            else:
                raise KeyError(
                    "Augmentation produced paired maps but no valid paired_map_view "
                    f"was provided (got {paired_map_view!r})."
                )
        else:
            raise KeyError(
                "Augmentation output missing map tensor. Expected one of "
                "['maps'] or paired ['maps_nobnt', 'maps_bnt']; got keys "
                f"{sorted(example.keys())}."
            )
        theta_np = example["theta"]  # (B, 6)

        # Skip any batch with NaNs
        if np.isnan(maps_np).any():
            print("    [!] Skipped batch with NaN maps")
            continue

        if not first_batch_reported:
            per_map_means = maps_np.mean(axis=(1, 2))  # (B, nbins)
            print(
                "    First-batch per-channel spatial-mean stats: "
                f"abs max = {np.abs(per_map_means).max():.3e}, "
                f"mean = {per_map_means.mean():.3e}"
            )
            first_batch_reported = True

        comp_y, _ = compressor.apply(
            comp_params, comp_state, None, maps_np,
        )
        x_list.append(np.array(comp_y))
        theta_list.append(theta_np)
        n_processed += len(theta_np)
        if n_processed % (ds_batch_size * 20) == 0:
            elapsed = time.time() - t0
            print(f"    Processed {n_processed} maps ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Done: {n_processed} maps in {elapsed:.1f}s")
    return {
        "theta": np.concatenate(theta_list, axis=0),
        "x": np.concatenate(x_list, axis=0),
    }


# =============================================================================
# Compressor diagnostics
# =============================================================================

def plot_compressor_diagnostics(
    obs_compressed: np.ndarray,
    train_x: np.ndarray,
    train_theta: np.ndarray,
    param_names: list[str],
):
    """Log CNN compressor diagnostics to wandb.

    Scatter plot of compressed-vs-true for each parameter, plus
    the observed compressed value marked.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_params = len(param_names)
    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4))
    if n_params == 1:
        axes = [axes]

    # Subsample for faster plotting
    n_plot = min(5000, len(train_theta))
    idx = np.random.choice(len(train_theta), n_plot, replace=False)

    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.scatter(
            train_theta[idx, i], train_x[idx, i],
            s=1, alpha=0.3, label="train",
        )
        ax.axhline(obs_compressed[i], color="red", lw=1.5,
                    ls="--", label="obs")
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Compressed dim {i}")
        ax.set_title(name)
        if i == 0:
            ax.legend(fontsize=8)

    fig.suptitle("CNN Compressor: compressed vs true params", fontsize=13)
    fig.tight_layout()
    wandb.log({"diagnostics/compressor_scatter": wandb.Image(fig)})
    plt.close(fig)


def log_compressed_summary_health_diagnostics(
    obs_summary: np.ndarray,
    train_x: np.ndarray,
    val_x: np.ndarray,
) -> None:
    """Log compact health diagnostics for CNN compressed summaries."""
    train_std = train_x.std(axis=0)
    p01 = np.percentile(train_x, 1.0, axis=0)
    p99 = np.percentile(train_x, 99.0, axis=0)
    obs_inlier_frac = np.mean((obs_summary >= p01) & (obs_summary <= p99))

    train_mean = train_x.mean(axis=0)
    val_mean = val_x.mean(axis=0)
    val_shift = np.abs(val_mean - train_mean) / (train_std + 1e-8)

    diag = {
        "diagnostics/summary_std_min": float(train_std.min()),
        "diagnostics/summary_std_median": float(np.median(train_std)),
        "diagnostics/summary_std_max": float(train_std.max()),
        "diagnostics/summary_dead_feature_frac": float(np.mean(train_std < 1e-8)),
        "diagnostics/obs_in_train_p01_p99_frac": float(obs_inlier_frac),
        "diagnostics/val_train_mean_shift_median_sigma": float(np.median(val_shift)),
        "diagnostics/val_train_mean_shift_max_sigma": float(np.max(val_shift)),
    }
    wandb.log(diag)
    print(
        "  Summary health | "
        f"std[min,med,max]=[{diag['diagnostics/summary_std_min']:.4e}, "
        f"{diag['diagnostics/summary_std_median']:.4e}, "
        f"{diag['diagnostics/summary_std_max']:.4e}] | "
        f"obs_inlier_frac={diag['diagnostics/obs_in_train_p01_p99_frac']:.3f}"
    )


def apply_summary_standardization(
    train_x: np.ndarray,
    val_x: np.ndarray,
    obs_x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    clip_value: Optional[float] = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply precomputed z-score standardization to train/val/obs summaries."""
    train_std = (train_x - mean) / std
    val_std = (val_x - mean) / std
    obs_std = (obs_x - mean) / std

    if clip_value is not None and clip_value > 0:
        train_std = np.clip(train_std, -clip_value, clip_value)
        val_std = np.clip(val_std, -clip_value, clip_value)
        obs_std = np.clip(obs_std, -clip_value, clip_value)

    return train_std, val_std, obs_std


def fit_and_apply_summary_standardization(
    train_x: np.ndarray,
    val_x: np.ndarray,
    obs_x: np.ndarray,
    clip_value: Optional[float] = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit z-score stats on train summaries, then apply to train/val/obs."""
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-8] = 1.0
    train_std, val_std, obs_std = apply_summary_standardization(
        train_x, val_x, obs_x, mean, std, clip_value=clip_value,
    )
    return train_std, val_std, obs_std, mean, std


# =============================================================================
# Normalizing Flow (Conditional RealNVP) — JAX / Haiku
# =============================================================================

def build_flow(n_cosmo_params: int, n_layers: int, hidden: int):
    """
    Build conditional RealNVP for NPE:  p(theta | y).

    Returns (nf_logp, nf_sample) — full hk.Transformed objects.
    """
    bijector_fn = partial(
        AffineCoupling,
        layers=[hidden] * 2,
        activation=jax.nn.silu,
    )
    NF_factory = partial(
        ConditionalRealNVP,
        n_layers=n_layers,
        bijector_fn=bijector_fn,
    )

    class NF(hk.Module):
        def __call__(self, y):
            return NF_factory(n_cosmo_params)(y)

    @hk.transform
    def nf_log_prob(theta, y):
        return NF()(y).log_prob(theta).squeeze()

    @hk.transform
    def nf_sample(y, n_samples):
        return NF()(y).sample(n_samples, seed=hk.next_rng_key())

    nf_logp = hk.without_apply_rng(nf_log_prob)
    return nf_logp, nf_sample


def make_update_fn(nf_logp, optimizer):
    """JIT-compiled training update step."""
    def loss_fn(params, theta_batch, y_batch):
        return -jnp.mean(nf_logp.apply(params, theta_batch, y_batch))

    @jax.jit
    def update(params, opt_state, theta_batch, y_batch):
        loss, grads = jax.value_and_grad(loss_fn)(
            params, theta_batch, y_batch,
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_opt_state

    return update


def train_flow(
    rng_key: jax.Array,
    nf_logp,
    dataset_train: Dict[str, np.ndarray],
    dataset_val: Dict[str, np.ndarray],
    n_cosmo: int,
    summary_dim: int,
    total_steps: int,
    batch_size: int,
    save_every: int,
    save_dir: Path,
    lr_init: float,
    end_lr: float,
    grad_clip: float = 1.0,
    weight_decay: float = 1e-4,
    patience: int = 20,
    lr_schedule_fn=None,
) -> hk.Params:
    """Train the conditional normalizing flow with early stopping."""
    print("######## TRAINING FLOW ########")
    key_init, _ = jax.random.split(rng_key)

    # Initialise params — dummy inputs of correct shape
    theta_dummy = 0.5 * jnp.zeros([1, n_cosmo])
    y_dummy = jnp.zeros([1, summary_dim])
    params = nf_logp.init(key_init, theta_dummy, y_dummy)

    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"  Flow parameters: {n_params:,}")

    # Cosine LR schedule + AdamW + gradient clipping
    lr_schedule = optax.cosine_decay_schedule(
        init_value=lr_init,
        decay_steps=total_steps,
        alpha=end_lr / max(lr_init, 1e-12),
    )
    opt_parts = []
    if grad_clip > 0:
        opt_parts.append(optax.clip_by_global_norm(grad_clip))
    opt_parts.append(
        optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay),
    )
    optimizer = optax.chain(*opt_parts)
    opt_state = optimizer.init(params)
    update = make_update_fn(nf_logp, optimizer)

    theta_train = dataset_train["theta"]
    x_train = dataset_train["x"]
    theta_val = dataset_val["theta"]
    x_val = dataset_val["x"]
    n_train = len(theta_train)
    n_val = len(theta_val)

    batch_losses: list[float] = []
    val_losses: list[float] = []
    val_steps: list[int] = []

    # Early stopping state
    best_val_loss = float("inf")
    best_step = 0
    best_params = params
    patience_counter = 0
    val_batch_size = min(512, n_val)

    for step in range(1, total_steps + 1):
        idx = np.random.randint(0, n_train, batch_size)
        loss, params, opt_state = update(
            params, opt_state, theta_train[idx], x_train[idx],
        )
        batch_losses.append(float(loss))

        if step % 100 == 0:
            log_dict = {"train/loss": float(loss), "step": step}
            if lr_schedule_fn is not None:
                log_dict["train/lr"] = float(lr_schedule_fn(step))
            wandb.log(log_dict, step=step)
            print(f"  Step {step:6d} | train loss {loss:.4f}")

        if step % save_every == 0 or step == total_steps:
            save_dir.mkdir(parents=True, exist_ok=True)

            # Validation loss (larger batch for stability)
            vidx = np.random.randint(0, n_val, val_batch_size)
            val_l = float(
                -jnp.mean(
                    nf_logp.apply(params, theta_val[vidx], x_val[vidx])
                )
            )
            val_losses.append(val_l)
            val_steps.append(step)

            improved = ""
            if val_l < best_val_loss:
                best_val_loss = val_l
                best_step = step
                best_params = params
                patience_counter = 0
                improved = " ***"
                # Save best model
                with open(save_dir / "params_cnn_flow_best.pkl", "wb") as f:
                    pickle.dump(params, f)
            else:
                patience_counter += 1

            with open(
                save_dir / f"params_cnn_flow_batch{step}.pkl", "wb",
            ) as f:
                pickle.dump(params, f)

            wandb.log({
                "val/loss": val_l,
                "val/best_loss": best_val_loss,
                "val/patience_counter": patience_counter,
                "step": step,
            }, step=step)
            print(
                f"  Saved @ step {step}. Val loss = {val_l:.4f}{improved}"
                f"  (best = {best_val_loss:.4f}, "
                f"patience = {patience_counter})"
            )

            # Early stopping
            if patience > 0 and patience_counter >= patience:
                print(
                    f"  Early stopping at step {step} "
                    f"(no val improvement for {patience} checks)"
                )
                break

    # Save loss curves
    np.save(save_dir / "loss_train_cnn.npy", np.array(batch_losses))
    np.save(save_dir / "loss_val_cnn.npy", np.array(val_losses))
    np.save(save_dir / "loss_val_steps.npy", np.array(val_steps))
    summary = {
        "best_val_loss": float(best_val_loss),
        "best_step": int(best_step),
        "final_step": int(step),
        "best_at_final_step": bool(best_step == step),
        "total_steps_requested": int(total_steps),
        "save_every": int(save_every),
        "patience": int(patience),
        "n_val_checks": int(len(val_losses)),
    }
    (save_dir / "flow_training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    wandb.run.summary["best_val_loss"] = best_val_loss
    wandb.run.summary["best_val_step"] = best_step
    wandb.run.summary["final_step"] = step
    print(f"  Best validation loss: {best_val_loss:.4f}")
    if best_step == step:
        print(
            "  WARNING: Best val loss occurred at final step. "
            "Flow may be underconverged; consider increasing --total-steps "
            "and/or reducing --save-every."
        )
    return best_params


# =============================================================================
# Posterior sampling
# =============================================================================

def sample_posterior(
    rng_key: jax.Array,
    nf_sample,
    flow_params: hk.Params,
    summary_obs: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Draw posterior samples via the trained NPE flow."""
    print("######## SAMPLING POSTERIOR ########")
    summary_dim = summary_obs.shape[-1]
    y_obs = jnp.asarray(summary_obs).reshape(1, summary_dim)
    y_cond = jnp.repeat(y_obs, repeats=n_samples, axis=0)
    samples = nf_sample.apply(flow_params, rng_key, y_cond, n_samples)

    # Remove NaN samples
    nan_rows = jnp.any(jnp.isnan(samples), axis=-1)
    samples = samples[~nan_rows]
    print(f"  Generated {len(samples)} valid posterior samples.")
    return np.array(samples)


# =============================================================================
# Plotting
# =============================================================================

def plot_posterior(
    samples: np.ndarray,
    truth: np.ndarray,
    output_path: str,
    param_names: list[str] | None = None,
    log_to_wandb: bool = True,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from getdist import MCSamples, plots as gplot
    except ImportError:
        print("Plotting requires 'getdist' and 'matplotlib'. Skipping.")
        return

    if param_names is None:
        param_names = [
            r"\Omega_m", r"\sigma_8", r"w_0",
            r"h_0", r"n_s", r"\Omega_b",
        ]

    mcsamples = MCSamples(
        samples=samples, names=param_names, labels=param_names,
    )

    g = gplot.get_subplot_plotter(subplot_size=1.5)
    g.triangle_plot(
        [mcsamples], filled=True,
        markers=truth,
        marker_args={"color": "red", "lw": 1.2},
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved posterior plot → {output_path}")

    if log_to_wandb and wandb.run is not None:
        wandb.log({"posterior/triangle_plot": wandb.Image(output_path)})

    plt.close()

    # Log per-parameter summary statistics to wandb
    if log_to_wandb and wandb.run is not None:
        for i, name in enumerate(param_names):
            s = samples[:, i]
            wandb.run.summary[f"posterior/{name}/mean"] = float(s.mean())
            wandb.run.summary[f"posterior/{name}/std"] = float(s.std())
            wandb.run.summary[f"posterior/{name}/truth"] = float(truth[i])
            wandb.run.summary[f"posterior/{name}/bias"] = float(
                s.mean() - truth[i]
            )


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    tomo_bin_indices = parse_tomo_bin_indices(args.tomo_bin_indices)
    compressor_conv_channels = parse_positive_int_list(
        args.compressor_conv_channels, "--compressor-conv-channels",
    )
    resnet_small_channels = parse_positive_int_list(
        args.resnet_small_channels, "--resnet-small-channels",
    )
    resnet_small_blocks = parse_positive_int_list(
        args.resnet_small_blocks, "--resnet-small-blocks",
    )
    args.compressor_conv_channels = ",".join(str(v) for v in compressor_conv_channels)
    args.resnet_small_channels = ",".join(str(v) for v in resnet_small_channels)
    args.resnet_small_blocks = ",".join(str(v) for v in resnet_small_blocks)
    if args.compressor_dense_width < 1:
        raise ValueError("--compressor-dense-width must be >= 1.")
    if args.compressor_pool_window < 1:
        raise ValueError("--compressor-pool-window must be >= 1.")
    if args.compressor_pool_stride < 1:
        raise ValueError("--compressor-pool-stride must be >= 1.")
    if args.resnet_head_width < 1:
        raise ValueError("--resnet-head-width must be >= 1.")
    if args.compressor_domain_hidden < 1:
        raise ValueError("--compressor-domain-hidden must be >= 1.")
    if args.compressor_consistency_weight < 0.0:
        raise ValueError("--compressor-consistency-weight must be >= 0.")
    if args.compressor_domain_adv_weight < 0.0:
        raise ValueError("--compressor-domain-adv-weight must be >= 0.")
    if (
        args.compressor_domain_adversarial
        and not args.compressor_paired_bnt_nobnt_consistency
    ):
        print(
            "  Enabling --compressor-paired-bnt-nobnt-consistency because "
            "--compressor-domain-adversarial requires paired domain views."
        )
        args.compressor_paired_bnt_nobnt_consistency = True
    if len(resnet_small_channels) != len(resnet_small_blocks):
        raise ValueError(
            "--resnet-small-channels and --resnet-small-blocks must have "
            "the same number of entries."
        )
    curriculum_sigma_factors: Optional[tuple[float, ...]] = None
    curriculum_stage_fracs: Optional[tuple[float, ...]] = None
    if args.compressor_noise_curriculum:
        curriculum_sigma_factors = parse_nonnegative_float_list(
            args.compressor_curriculum_sigma_factors,
            "--compressor-curriculum-sigma-factors",
        )
        curriculum_stage_fracs = parse_positive_float_list(
            args.compressor_curriculum_stage_fracs,
            "--compressor-curriculum-stage-fracs",
        )
        if len(curriculum_sigma_factors) != len(curriculum_stage_fracs):
            raise ValueError(
                "--compressor-curriculum-sigma-factors and "
                "--compressor-curriculum-stage-fracs must have the same length."
            )
        frac_sum = float(np.sum(np.asarray(curriculum_stage_fracs, dtype=np.float64)))
        if not np.isclose(frac_sum, 1.0, atol=1e-6):
            raise ValueError(
                "--compressor-curriculum-stage-fracs must sum to 1.0 "
                f"(got {frac_sum:.8f})."
            )
        args.compressor_curriculum_sigma_factors = ",".join(
            f"{v:.10g}" for v in curriculum_sigma_factors
        )
        args.compressor_curriculum_stage_fracs = ",".join(
            f"{v:.10g}" for v in curriculum_stage_fracs
        )
    if args.nbins != len(tomo_bin_indices):
        print(
            f"  Overriding nbins from {args.nbins} to {len(tomo_bin_indices)} "
            f"to match selected bins {tomo_bin_indices}."
        )
        args.nbins = len(tomo_bin_indices)
    if args.apply_bnt:
        validate_bnt_configuration(args.nbins, tomo_bin_indices)

    if args.harmonic_train_realizations_limit is not None and args.harmonic_train_realizations_limit < 1:
        raise ValueError("--harmonic-train-realizations-limit must be >= 1.")
    if args.harmonic_val_realizations_limit is not None and args.harmonic_val_realizations_limit < 1:
        raise ValueError("--harmonic-val-realizations-limit must be >= 1.")

    cnn_map_route = args.cnn_map_route or (
        "harmonic" if args.full_sphere_cross_cache else "tfds"
    )
    full_sphere_cache_dir: Optional[Path] = None
    full_sphere_cache_manifest_sha = ""
    harmonic_regime = ""
    if args.full_sphere_cross_cache:
        if cnn_map_route != "harmonic":
            raise ValueError(
                "--full-sphere-cross-cache requires --cnn-map-route=harmonic "
                f"(got {cnn_map_route})."
            )
        full_sphere_cache_dir = Path(args.full_sphere_cross_cache).resolve()
        manifest = _read_harmonic_manifest(full_sphere_cache_dir)
        full_sphere_cache_manifest_sha = str(manifest["args_sha256"])
        harmonic_regime = (
            args.harmonic_cache_regime
            if args.harmonic_cache_regime is not None
            else ("bnt" if args.apply_bnt else "nobnt")
        )
        if harmonic_regime not in ("bnt", "nobnt"):
            raise ValueError(
                "--harmonic-cache-regime must be one of {'bnt','nobnt'} "
                f"(got {harmonic_regime})."
            )
        harmonic_apply_bnt = harmonic_regime == "bnt"
        if bool(args.apply_bnt) != harmonic_apply_bnt:
            print(
                "  Overriding --apply-bnt to match --harmonic-cache-regime="
                f"{harmonic_regime}."
            )
            args.apply_bnt = harmonic_apply_bnt
            if args.apply_bnt:
                validate_bnt_configuration(args.nbins, tomo_bin_indices)
        if not args.zero_mean_maps:
            print(
                "  [warn] Forcing zero_mean_maps=True for harmonic-cache route "
                "(cache patches are already demeaned)."
            )
        args.zero_mean_maps = True
        if args.require_disjoint_train_examples:
            raise ValueError(
                "--require-disjoint-train-examples is TFDS-only and not "
                "supported with --full-sphere-cross-cache."
            )
        args.compressor_train_split = _normalize_harmonic_split(
            args.compressor_train_split,
            "--compressor-train-split",
            allowed=("train", "val"),
        )
        args.compressor_val_split = _normalize_harmonic_split(
            args.compressor_val_split,
            "--compressor-val-split",
            allowed=("train", "val"),
        )
        args.nde_train_split = _normalize_harmonic_split(
            args.nde_train_split,
            "--nde-train-split",
            allowed=("train", "val"),
        )
        args.nde_val_split = _normalize_harmonic_split(
            args.nde_val_split,
            "--nde-val-split",
            allowed=("train", "val"),
        )
        print(f"  cnn_map_route = harmonic")
        print(f"  harmonic cache = {full_sphere_cache_dir}")
        print(f"  harmonic regime = {harmonic_regime}")
        print(f"  manifest sha256 = {full_sphere_cache_manifest_sha[:16]}...")
    else:
        if cnn_map_route != "tfds":
            raise ValueError(
                "--cnn-map-route=harmonic requires --full-sphere-cross-cache."
            )

    if cnn_map_route == "harmonic" and args.compressor_paired_bnt_nobnt_consistency:
        raise ValueError(
            "Harmonic-cache route currently supports single-regime maps only; "
            "--compressor-paired-bnt-nobnt-consistency is unsupported."
        )
    if cnn_map_route == "harmonic" and args.compressor_noise_curriculum:
        raise ValueError(
            "--compressor-noise-curriculum is unsupported for harmonic-cache "
            "route because shape-noise levels are baked into cache files."
        )

    cnn_input_channels = (
        HARMONIC_CACHE_CHANNELS if cnn_map_route == "harmonic" else args.nbins
    )

    # Per-channel RMS normalization for harmonic route (computed once here so it
    # is available for both the observed map and the compressor training iterator).
    harmonic_channel_scale: np.ndarray | None = None
    if cnn_map_route == "harmonic" and args.harmonic_normalize_input_channels:
        if full_sphere_cache_dir is None:
            raise RuntimeError("Internal error: harmonic route selected with no cache dir.")
        print(
            "  [harmonic-normalize-input-channels] Computing per-channel RMS "
            f"from {args.compressor_train_split} split ..."
        )
        harmonic_channel_scale = compute_harmonic_channel_rms(
            cache_dir=full_sphere_cache_dir,
            regime=harmonic_regime,
            split=args.compressor_train_split,
            max_realizations=args.harmonic_train_realizations_limit,
        )
        print(f"  Per-channel RMS (auto first, then cross): {harmonic_channel_scale}")
        print(
            f"  RMS range: min={harmonic_channel_scale.min():.4e}, "
            f"max={harmonic_channel_scale.max():.4e}, "
            f"ratio={harmonic_channel_scale.max() / harmonic_channel_scale.min():.1f}×"
        )

    setup_environment(args.cuda_visible_devices)
    rng = jax.random.PRNGKey(args.seed)
    rng, rng_obs, rng_sample = jax.random.split(rng, 3)

    # Derived quantities
    summary_dim = args.compressor_dim
    print(f"  summary_dim    = {summary_dim}")
    save_path = Path(args.save_dir) / "cnn_vmim" / args.map_kind
    if cnn_map_route == "harmonic":
        save_path = save_path / f"harmonic_{harmonic_regime}"
    summary_stats_path = save_path / "cnn_summary_standardization.npz"

    param_names = [
        r"\Omega_m", r"\sigma_8", r"w_0",
        r"h_0", r"n_s", r"\Omega_b",
    ]

    # ------------------------------------------------------------------
    # 0. Initialize Weights & Biases
    # ------------------------------------------------------------------
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
            tags=[args.map_kind, "cnn-vmim",
                  f"nvp{args.nvp_layers}"],
        )
    else:
        wandb.init(mode="disabled")

    # ------------------------------------------------------------------
    # 1. Observed map
    # ------------------------------------------------------------------
    if cnn_map_route == "harmonic":
        if full_sphere_cache_dir is None:
            raise RuntimeError("Internal error: harmonic route selected with no cache dir.")
        m_data, cosmo_params, truth = load_observed_from_harmonic_cache(
            cache_dir=full_sphere_cache_dir,
            regime=harmonic_regime,
            cosmo_id=args.harmonic_obs_cosmo_id,
            perm=args.harmonic_obs_perm,
            patch_idx=args.harmonic_obs_patch_idx,
            meta_path=args.cosmogrid_meta,
            channel_scale=harmonic_channel_scale,
        )
    else:
        m_data, cosmo_params, truth = load_observed_map(
            args.cosmogrid_meta, args.fiducial_map,
            args.field_size, args.field_npix, args.nside, args.nbins,
            tomo_bin_indices,
            args.sigma_e, args.galaxy_density, rng_obs,
            apply_bnt=args.apply_bnt,
            zero_mean_maps=args.zero_mean_maps,
        )

    # ------------------------------------------------------------------
    # 2. CNN compressor
    # ------------------------------------------------------------------
    print("######## CNN COMPRESSOR ########")
    print(f"  Input route: {cnn_map_route} (channels={cnn_input_channels})")
    if args.compressor_arch == "plain":
        arch_desc = (
            f"plain conv={compressor_conv_channels} "
            f"dense={args.compressor_dense_width} "
            f"pool=({args.compressor_pool_window},{args.compressor_pool_stride})"
        )
    elif args.compressor_arch == "resnet_small":
        arch_desc = (
            "resnet_small "
            f"channels={resnet_small_channels} "
            f"blocks={resnet_small_blocks} "
            f"head={args.resnet_head_width}"
        )
    elif args.compressor_arch in ("resnet18", "resnet34", "resnet50"):
        arch_desc = (
            f"{args.compressor_arch} "
            f"head={args.resnet_head_width} "
            f"resnet_v2={bool(args.resnet_v2)}"
        )
    elif args.compressor_arch == "resnet50_gn":
        arch_desc = (
            f"resnet50_gn (GroupNorm, custom) "
            f"head={args.resnet_head_width}"
        )
    else:
        raise ValueError(f"Unsupported --compressor-arch '{args.compressor_arch}'")
    print(f"  Compressor architecture: {arch_desc}")
    print(
        "  Split config: "
        f"compressor[{args.compressor_train_split}/{args.compressor_val_split}] "
        f"NDE[{args.nde_train_split}/{args.nde_val_split}]"
    )

    split_overlap_info = None
    if args.require_disjoint_train_examples and cnn_map_route == "tfds":
        print(
            "  Auditing train-split overlap (example identity = "
            "(cosmology, patch)) ..."
        )
        split_overlap_info = audit_train_split_overlap(
            args.tfds_name,
            args.compressor_train_split,
            args.nde_train_split,
        )
        print(
            "  Train overlap audit | "
            f"comp_examples={split_overlap_info['compressor_train_examples']} "
            f"nde_examples={split_overlap_info['nde_train_examples']} "
            f"shared_examples={split_overlap_info['shared_example_count']} "
            f"shared_theta={split_overlap_info['shared_theta_count']}"
        )
        if int(split_overlap_info["shared_example_count"]) > 0:
            raise ValueError(
                "Detected shared training examples between compressor and NDE "
                f"splits: {split_overlap_info['shared_example_count']}."
            )
        wandb.config.update(
            {"data/train_split_overlap": split_overlap_info},
            allow_val_change=True,
        )

    compressor_train, compressor_eval = build_compressors(
        args.compressor_dim,
        arch=args.compressor_arch,
        conv_channels=compressor_conv_channels,
        dense_width=args.compressor_dense_width,
        pool_window=args.compressor_pool_window,
        pool_stride=args.compressor_pool_stride,
        resnet_small_channels=resnet_small_channels,
        resnet_small_blocks=resnet_small_blocks,
        resnet_head_width=args.resnet_head_width,
        resnet_v2=bool(args.resnet_v2),
    )
    compressor_source = "train_compressor" if args.train_compressor else "pretrained"
    compressor_params_ref: Optional[str] = None
    compressor_state_ref: Optional[str] = None

    paired_consistency_training = bool(
        args.train_compressor and args.compressor_paired_bnt_nobnt_consistency
    )
    augmentation = None
    compressor_dataset_iter_factory: Optional[
        Callable[[str, int], Iterator[Dict[str, np.ndarray]]]
    ] = None
    curriculum_stages: Optional[list[Dict[str, object]]] = None
    if cnn_map_route == "tfds":
        augmentation = build_augmentation(
            args.map_kind, args.sigma_e, args.galaxy_density,
            args.field_size, args.field_npix, args.nbins, tomo_bin_indices,
            apply_bnt=args.apply_bnt,
            paired_bnt_nobnt_consistency=paired_consistency_training,
            zero_mean_maps=args.zero_mean_maps,
        )
        if args.train_compressor and args.compressor_noise_curriculum:
            if curriculum_sigma_factors is None or curriculum_stage_fracs is None:
                raise ValueError(
                    "Internal error: curriculum flag set but parsed schedule missing."
                )
            stage_steps = allocate_stage_steps(
                int(args.compressor_steps),
                curriculum_stage_fracs,
            )
            curriculum_stages = []
            for idx, (sigma_factor, steps) in enumerate(
                zip(curriculum_sigma_factors, stage_steps),
                start=1,
            ):
                if int(steps) <= 0:
                    continue
                stage_sigma_e = float(args.sigma_e) * float(sigma_factor)
                stage_aug = build_augmentation(
                    args.map_kind,
                    args.sigma_e,
                    args.galaxy_density,
                    args.field_size,
                    args.field_npix,
                    args.nbins,
                    tomo_bin_indices,
                    apply_bnt=args.apply_bnt,
                    sigma_e_override=stage_sigma_e,
                    paired_bnt_nobnt_consistency=paired_consistency_training,
                    zero_mean_maps=args.zero_mean_maps,
                )
                curriculum_stages.append(
                    {
                        "name": f"curriculum_s{idx}",
                        "stage_index": idx,
                        "sigma_factor": float(sigma_factor),
                        "sigma_e": float(stage_sigma_e),
                        "steps": int(steps),
                        "augmentation_fn": stage_aug,
                    }
                )
            if not curriculum_stages:
                raise ValueError(
                    "Curriculum schedule produced zero non-empty stages. "
                    "Increase --compressor-steps or adjust stage fractions."
                )
            allocated = int(sum(int(stage["steps"]) for stage in curriculum_stages))
            if allocated != int(args.compressor_steps):
                raise ValueError(
                    "Curriculum stage allocation does not match compressor steps "
                    f"({allocated} != {args.compressor_steps})."
                )
            wandb.config.update(
                {
                    "compressor/noise_curriculum": True,
                    "compressor/curriculum_sigma_factors": list(curriculum_sigma_factors),
                    "compressor/curriculum_stage_fracs": list(curriculum_stage_fracs),
                    "compressor/curriculum_stage_steps": [
                        int(stage["steps"]) for stage in curriculum_stages
                    ],
                },
                allow_val_change=True,
            )
        else:
            wandb.config.update(
                {"compressor/noise_curriculum": False},
                allow_val_change=True,
            )
    else:
        if full_sphere_cache_dir is None:
            raise RuntimeError("Internal error: harmonic route selected with no cache dir.")

        def _harmonic_dataset_iter_factory(
            split: str,
            batch_size: int,
        ) -> Iterator[Dict[str, np.ndarray]]:
            is_train_split = split == args.compressor_train_split
            split_seed = int(args.seed) + (1001 if is_train_split else 2001)
            split_limit = (
                args.harmonic_train_realizations_limit
                if is_train_split
                else args.harmonic_val_realizations_limit
            )
            return build_harmonic_batch_iterator(
                cache_dir=full_sphere_cache_dir,
                regime=harmonic_regime,
                split=split,
                batch_size=batch_size,
                seed=split_seed,
                flip=is_train_split,
                max_realizations=split_limit,
                channel_scale=harmonic_channel_scale,
            )

        compressor_dataset_iter_factory = _harmonic_dataset_iter_factory
        wandb.config.update(
            {"compressor/noise_curriculum": False},
            allow_val_change=True,
        )
    wandb.config.update(
        {
            "compressor/paired_bnt_nobnt_consistency": bool(
                paired_consistency_training
            ),
            "compressor/consistency_weight": float(args.compressor_consistency_weight),
            "compressor/domain_adversarial": bool(
                args.compressor_domain_adversarial and paired_consistency_training
            ),
            "compressor/domain_adv_weight": float(args.compressor_domain_adv_weight),
            "compressor/domain_hidden": int(args.compressor_domain_hidden),
        },
        allow_val_change=True,
    )

    if args.train_compressor:
        comp_save_dir = (
            Path(args.save_dir) / "vmim" / args.map_kind
            / f"sigma_{args.sigma_e}"
            / f"gal_density_{int(args.galaxy_density * 4)}"
            / f"bin_{args.nbins}"
        )
        if cnn_map_route == "harmonic":
            comp_save_dir = comp_save_dir / f"harmonic_{harmonic_regime}_ch{cnn_input_channels}"
        comp_params, comp_state = train_compressor_vmim(
            compressor=compressor_train,
            augmentation_fn=augmentation,
            n_cosmo=args.n_cosmo,
            compressor_dim=args.compressor_dim,
            field_npix=args.field_npix,
            nbins=cnn_input_channels,
            total_steps=args.compressor_steps,
            lr_init=args.compressor_lr,
            batch_size=args.compressor_batch_size,
            save_every=args.compressor_save_every,
            save_dir=comp_save_dir,
            m_data_obs=m_data,
            truth=truth,
            param_names=param_names,
            tfds_name=args.tfds_name,
            compressor_train_split=args.compressor_train_split,
            compressor_val_split=args.compressor_val_split,
            plot_contours=args.compressor_plot_contours,
            noise_curriculum_stages=curriculum_stages,
            paired_bnt_nobnt_consistency=paired_consistency_training,
            consistency_weight=float(args.compressor_consistency_weight),
            domain_adversarial=bool(
                args.compressor_domain_adversarial and paired_consistency_training
            ),
            domain_adv_weight=float(args.compressor_domain_adv_weight),
            domain_hidden=int(args.compressor_domain_hidden),
            vmim_nf_hidden=int(args.vmim_nf_hidden),
            dataset_iter_factory=compressor_dataset_iter_factory,
        )
        # Invalidate any cached compressed datasets
        cache_dir = Path(args.cache_dir) if args.cache_dir else None
        if cache_dir is not None:
            for f in ["cnn_train.npz", "cnn_val.npz", "cnn_cache_meta.npz"]:
                p = cache_dir / f
                if p.exists():
                    p.unlink()
                    print(f"  Deleted stale cache: {p}")
    else:
        compressor_params_ref = args.compressor_params
        compressor_state_ref = args.compressor_state
        comp_params, comp_state = load_compressor_params(
            args.compressor_params, args.compressor_state,
        )
        log_compressor_checkpoint_provenance(
            args.compressor_params, args.compressor_state,
        )

    # 2a. Compress observed map
    print("######## COMPRESS: OBSERVED MAP ########")
    obs_compressed, _ = compressor_eval.apply(
        comp_params, comp_state, None,
        m_data.reshape([1, args.field_npix, args.field_npix, cnn_input_channels]),
    )
    obs_compressed = np.array(obs_compressed).squeeze()
    print(f"  Observed compressed shape = {obs_compressed.shape}")
    print(f"  Observed compressed = {obs_compressed}")

    # ------------------------------------------------------------------
    # 3. Compress train / test datasets
    # ------------------------------------------------------------------
    print("######## COMPRESS: DATASETS ########")

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    cache_ok = False
    cache_meta_expected = build_cnn_cache_metadata(
        args=args,
        compressor_source=compressor_source,
        compressor_params_path=compressor_params_ref,
        compressor_state_path=compressor_state_ref,
        tomo_bin_indices=tomo_bin_indices,
        cnn_map_route=cnn_map_route,
        full_sphere_cache_manifest_sha256=full_sphere_cache_manifest_sha,
        harmonic_regime=harmonic_regime,
        cnn_input_channels=cnn_input_channels,
    )

    if cache_dir is not None and cache_dir.exists():
        train_cache = cache_dir / "cnn_train.npz"
        val_cache = cache_dir / "cnn_val.npz"
        meta_cache = cache_dir / "cnn_cache_meta.npz"
        if train_cache.exists() and val_cache.exists() and meta_cache.exists():
            meta = np.load(meta_cache)
            cache_ok, mismatches = compare_cache_metadata(
                meta, cache_meta_expected,
            )
            if cache_ok:
                print("  Loading cached compressed datasets (metadata matches) ...")
                d_tr = np.load(train_cache)
                dataset_train = {"theta": d_tr["theta"], "x": d_tr["x"]}
                d_va = np.load(val_cache)
                dataset_val = {"theta": d_va["theta"], "x": d_va["x"]}
                print(
                    f"  Train: {len(dataset_train['theta'])} | "
                    f"Val: {len(dataset_val['theta'])}"
                )
            else:
                print(
                    "  CNN cache metadata mismatch; recomputing compressed datasets. "
                    f"First mismatch: {mismatches[0]}"
                )
        elif train_cache.exists() and val_cache.exists():
            print(
                "  CNN cache metadata file missing "
                "(cnn_cache_meta.npz); recomputing to avoid stale-cache reuse."
            )

    if not cache_ok:
        if cnn_map_route == "harmonic":
            if full_sphere_cache_dir is None:
                raise RuntimeError("Internal error: harmonic route selected with no cache dir.")
            dataset_train = compress_dataset_from_harmonic_cache(
                cache_dir=full_sphere_cache_dir,
                regime=harmonic_regime,
                split=args.nde_train_split,
                compressor=compressor_eval,
                comp_params=comp_params,
                comp_state=comp_state,
                ds_batch_size=args.ds_batch_size,
                rng=np.random.default_rng(int(args.seed) + 3001),
                flip=True,
                max_realizations=args.harmonic_train_realizations_limit,
                channel_scale=harmonic_channel_scale,
            )
            dataset_val = compress_dataset_from_harmonic_cache(
                cache_dir=full_sphere_cache_dir,
                regime=harmonic_regime,
                split=args.nde_val_split,
                compressor=compressor_eval,
                comp_params=comp_params,
                comp_state=comp_state,
                ds_batch_size=args.ds_batch_size,
                rng=np.random.default_rng(int(args.seed) + 4001),
                flip=False,
                max_realizations=args.harmonic_val_realizations_limit,
                channel_scale=harmonic_channel_scale,
            )
        else:
            paired_map_view = (
                "bnt" if args.apply_bnt else "nobnt"
            ) if paired_consistency_training else None
            dataset_train = compress_dataset(
                args.tfds_name, args.nde_train_split,
                augmentation, compressor_eval, comp_params, comp_state,
                args.ds_batch_size,
                paired_map_view=paired_map_view,
            )
            dataset_val = compress_dataset(
                args.tfds_name, args.nde_val_split,
                augmentation, compressor_eval, comp_params, comp_state,
                args.ds_batch_size,
                paired_map_view=paired_map_view,
            )
        # Save cache
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_dir / "cnn_train.npz",
                theta=dataset_train["theta"], x=dataset_train["x"],
            )
            np.savez(
                cache_dir / "cnn_val.npz",
                theta=dataset_val["theta"], x=dataset_val["x"],
            )
            np.savez(cache_dir / "cnn_cache_meta.npz", **cache_meta_expected)
            print(f"  Cached datasets to {cache_dir}")

    print(f"  Train x shape = {dataset_train['x'].shape}")
    print(f"  Val   x shape = {dataset_val['x'].shape}")

    # ------------------------------------------------------------------
    # 3b. Compressor diagnostics
    # ------------------------------------------------------------------
    print("######## COMPRESSOR DIAGNOSTICS ########")
    plot_compressor_diagnostics(
        obs_compressed, dataset_train["x"], dataset_train["theta"],
        param_names,
    )
    log_compressed_summary_health_diagnostics(
        obs_compressed, dataset_train["x"], dataset_val["x"],
    )

    # ------------------------------------------------------------------
    # 3c. Optional summary standardization
    # ------------------------------------------------------------------
    standardization_applied = False
    standardization_mean: Optional[np.ndarray] = None
    standardization_std: Optional[np.ndarray] = None
    summary_clip_value = (
        args.summary_clip_value if args.summary_clip_value > 0 else None
    )
    if args.standardize_summary:
        if args.no_train and summary_stats_path.exists():
            std_data = np.load(summary_stats_path)
            standardization_mean = np.array(std_data["mean"])
            standardization_std = np.array(std_data["std"])
            if "clip_value" in std_data.files:
                loaded_clip = float(std_data["clip_value"])
                if np.isfinite(loaded_clip) and loaded_clip > 0:
                    summary_clip_value = loaded_clip
                else:
                    summary_clip_value = None
            dataset_train["x"], dataset_val["x"], obs_compressed = (
                apply_summary_standardization(
                    dataset_train["x"], dataset_val["x"], obs_compressed,
                    standardization_mean, standardization_std,
                    clip_value=summary_clip_value,
                )
            )
            standardization_applied = True
            print(
                "  Loaded summary standardization stats from "
                f"{summary_stats_path}"
            )
        elif args.no_train and not summary_stats_path.exists():
            print(
                "  WARNING: --no-train set but no saved CNN summary "
                "standardization stats found; skipping standardization to avoid "
                "checkpoint mismatch."
            )
        else:
            (
                dataset_train["x"],
                dataset_val["x"],
                obs_compressed,
                standardization_mean,
                standardization_std,
            ) = fit_and_apply_summary_standardization(
                dataset_train["x"], dataset_val["x"], obs_compressed,
                clip_value=summary_clip_value,
            )
            standardization_applied = True
            print(
                "  Applied summary standardization "
                f"(clip={summary_clip_value if summary_clip_value is not None else 'off'}). "
                f"std range=[{standardization_std.min():.4e}, "
                f"{standardization_std.max():.4e}]"
            )

    if args.shuffle_theta_train:
        np.random.seed(args.seed)
        perm = np.random.permutation(len(dataset_train["theta"]))
        dataset_train["theta"] = dataset_train["theta"][perm]
        print("  [control] Shuffled training theta labels before flow training.")
        wandb.config.update(
            {"control/shuffle_theta_train": True}, allow_val_change=True,
        )

    # Log dataset statistics to wandb
    wandb.log({
        "data/train_size": len(dataset_train["theta"]),
        "data/val_size": len(dataset_val["theta"]),
        "data/summary_dim": summary_dim,
        "data/cnn_map_route": 1 if cnn_map_route == "harmonic" else 0,
        "data/cnn_input_channels": int(cnn_input_channels),
        "data/compressor_arch": args.compressor_arch,
        "data/compressor_noise_curriculum": int(args.compressor_noise_curriculum),
        "data/compressor_paired_bnt_nobnt_consistency": int(
            args.compressor_paired_bnt_nobnt_consistency
        ),
        "data/compressor_consistency_weight": float(args.compressor_consistency_weight),
        "data/compressor_domain_adversarial": int(args.compressor_domain_adversarial),
        "data/compressor_domain_adv_weight": float(args.compressor_domain_adv_weight),
        "data/require_disjoint_train_examples": int(args.require_disjoint_train_examples),
        "data/train_x_min": float(dataset_train["x"].min()),
        "data/train_x_max": float(dataset_train["x"].max()),
        "data/train_x_mean": float(dataset_train["x"].mean()),
        "data/train_x_std": float(dataset_train["x"].std()),
        "data/summary_standardized": int(standardization_applied),
        "data/summary_clip_value": (
            float(summary_clip_value) if summary_clip_value is not None else 0.0
        ),
        "data/shared_train_examples": (
            int(split_overlap_info["shared_example_count"])
            if split_overlap_info is not None
            else -1
        ),
        "data/shared_train_theta": (
            int(split_overlap_info["shared_theta_count"])
            if split_overlap_info is not None
            else -1
        ),
    })

    # Log theta distributions as histograms
    for i, name in enumerate(param_names):
        wandb.log({
            f"data/theta_train/{name}": wandb.Histogram(
                dataset_train["theta"][:, i],
            ),
            f"data/theta_val/{name}": wandb.Histogram(
                dataset_val["theta"][:, i],
            ),
        })

    # ------------------------------------------------------------------
    # 4. Build & train flow
    # ------------------------------------------------------------------
    nf_logp, nf_sample = build_flow(
        n_cosmo_params=args.n_cosmo,
        n_layers=args.nvp_layers,
        hidden=args.nvp_hidden,
    )

    flow_summary_path = save_path / "flow_training_summary.json"
    flow_params = None
    flow_params_source = "unknown"

    if args.no_train:
        # Prefer best model, fall back to latest checkpoint
        best_path = save_path / "params_cnn_flow_best.pkl"
        if best_path.exists():
            load_path = best_path
        else:
            candidates = sorted(
                save_path.glob("params_cnn_flow_batch*.pkl"),
                key=_checkpoint_step,
            )
            if not candidates:
                raise FileNotFoundError(
                    f"No saved flow params in {save_path} "
                    f"and --no-train set"
                )
            load_path = candidates[-1]
        with open(load_path, "rb") as f:
            flow_params = pickle.load(f)
        flow_params_source = str(load_path.resolve())
        print(f"  Loaded flow params from {load_path}")
    else:
        # Build LR schedule for logging
        _lr_schedule = optax.cosine_decay_schedule(
            init_value=args.lr_init,
            decay_steps=args.total_steps,
            alpha=args.lr_end / max(args.lr_init, 1e-12),
        )
        flow_params = train_flow(
            rng, nf_logp,
            dataset_train, dataset_val,
            n_cosmo=args.n_cosmo,
            summary_dim=summary_dim,
            total_steps=args.total_steps,
            batch_size=args.batch_size,
            save_every=args.save_every,
            save_dir=save_path,
            lr_init=args.lr_init,
            end_lr=args.lr_end,
            grad_clip=args.grad_clip,
            weight_decay=args.weight_decay,
            patience=args.patience,
            lr_schedule_fn=_lr_schedule,
        )
        flow_params_source = "trained_best_val_in_memory"

    if (not args.no_train and standardization_applied and
            standardization_mean is not None and standardization_std is not None):
        save_path.mkdir(parents=True, exist_ok=True)
        np.savez(
            summary_stats_path,
            mean=standardization_mean,
            std=standardization_std,
            clip_value=(
                np.nan if summary_clip_value is None else float(summary_clip_value)
            ),
        )

    # ------------------------------------------------------------------
    # 5. Posterior sampling
    # ------------------------------------------------------------------
    if not args.no_sample:
        posterior_samples = sample_posterior(
            rng_sample, nf_sample, flow_params,
            obs_compressed, args.npe_samples,
        )
        out = Path(args.posterior_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, posterior_samples)
        metadata = {
            "method": "cnn",
            "posterior_file": str(out.resolve()),
            "flow_params_source": flow_params_source,
            "compressor_arch": str(args.compressor_arch),
            "compressor_conv_channels": str(args.compressor_conv_channels),
            "compressor_dense_width": int(args.compressor_dense_width),
            "compressor_pool_window": int(args.compressor_pool_window),
            "compressor_pool_stride": int(args.compressor_pool_stride),
            "compressor_noise_curriculum": bool(args.compressor_noise_curriculum),
            "compressor_curriculum_sigma_factors": str(
                args.compressor_curriculum_sigma_factors
            ),
            "compressor_curriculum_stage_fracs": str(
                args.compressor_curriculum_stage_fracs
            ),
            "compressor_paired_bnt_nobnt_consistency": bool(
                args.compressor_paired_bnt_nobnt_consistency
            ),
            "compressor_consistency_weight": float(args.compressor_consistency_weight),
            "compressor_domain_adversarial": bool(args.compressor_domain_adversarial),
            "compressor_domain_adv_weight": float(args.compressor_domain_adv_weight),
            "compressor_domain_hidden": int(args.compressor_domain_hidden),
            "resnet_small_channels": str(args.resnet_small_channels),
            "resnet_small_blocks": str(args.resnet_small_blocks),
            "resnet_head_width": int(args.resnet_head_width),
            "resnet_v2": bool(args.resnet_v2),
            "total_steps": int(args.total_steps),
            "save_every": int(args.save_every),
            "patience": int(args.patience),
            "npe_samples": int(args.npe_samples),
            "summary_standardized": bool(standardization_applied),
            "summary_standardization_file": (
                str(summary_stats_path.resolve())
                if standardization_applied and summary_stats_path.exists()
                else None
            ),
            "compressor_train_split": str(args.compressor_train_split),
            "compressor_val_split": str(args.compressor_val_split),
            "nde_train_split": str(args.nde_train_split),
            "nde_val_split": str(args.nde_val_split),
            "require_disjoint_train_examples": bool(args.require_disjoint_train_examples),
            "train_split_overlap": split_overlap_info,
            "apply_bnt": bool(args.apply_bnt),
            "bnt_matrix_version": BNT_MATRIX_VERSION if args.apply_bnt else "none",
            "zero_mean_maps": bool(args.zero_mean_maps),
            "harmonic_normalize_input_channels": bool(
                args.harmonic_normalize_input_channels
            ),
            "harmonic_channel_scale": (
                harmonic_channel_scale.tolist()
                if harmonic_channel_scale is not None
                else None
            ),
            "cnn_map_route": str(cnn_map_route),
            "cnn_input_channels": int(cnn_input_channels),
            "full_sphere_cache_dir": (
                str(full_sphere_cache_dir) if full_sphere_cache_dir is not None else None
            ),
            "full_sphere_cache_manifest_sha256": str(full_sphere_cache_manifest_sha),
            "harmonic_regime": str(harmonic_regime) if harmonic_regime else None,
        }
        if flow_summary_path.exists():
            metadata["flow_training_summary"] = json.loads(
                flow_summary_path.read_text(encoding="utf-8")
            )
        out.with_suffix(".meta.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        print(f"  Saved posterior samples → {out.resolve()}")

        if args.plot:
            fig_out = Path(args.figure_out)
            fig_out.parent.mkdir(parents=True, exist_ok=True)
            plot_posterior(
                posterior_samples, truth, str(fig_out),
                param_names, log_to_wandb=(not args.no_wandb),
            )
    else:
        print("  Skipping posterior sampling (--no-sample)")

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
