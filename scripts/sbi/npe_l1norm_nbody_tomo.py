#!/usr/bin/env python
"""
L1-norm + NPE for tomographic weak lensing cosmological inference.

Replaces the CNN compressor in the standard NPE pipeline with wavelet L1-norm
summary statistics computed via the wl_stats_torch package on GPU (PyTorch).
The L1-norm summary vector is then fed to a conditional RealNVP normalizing
flow (JAX / Haiku) for Neural Posterior Estimation of cosmological parameters.

Main stages:
 1. Set CUDA device
 2. Load observed (fiducial) 4-bin tomographic map and add shape noise
 3. Compute L1-norm summary for the observed map (PyTorch / GPU)
 4. Load TFDS tomographic dataset, apply augmentation, compute L1-norm
    summaries for train/test (PyTorch / GPU)
 5. Standardize summaries (zero-mean, unit-variance)
 6. Define & train conditional RealNVP normalizing flow for p(theta | y)
 7. Sample the posterior and produce contour plots

Requires:
  - wl_stats_torch (located at /home/tersenov/software/wl_stats_torch)
  - NbodyCosmogridDatasetTomo TFDS dataset (from tf_dataset_nbody_tomo.py)
  - sbi_lens (normalizing flow components)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import wandb

import h5py
import haiku as hk
import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import torch
from jax.lib import xla_bridge
from sklearn.decomposition import PCA
from tensorflow_probability.substrates import jax as tfp

if not hasattr(np, "issctype"):
    # Compatibility for NumPy >= 2.0 with tensorflow_probability paths that
    # still reference np.issctype.
    def _np_issctype(rep):
        try:
            return issubclass(np.dtype(rep).type, np.generic)
        except Exception:
            return False

    np.issctype = _np_issctype  # type: ignore[attr-defined]

from bnt_utils import (
    BNT_MATRIX_VERSION,
    apply_bnt_numpy,
    apply_bnt_tf,
    validate_bnt_configuration,
)

# sbi_lens normalizing flow
from sbi_lens.normflow.models import AffineCoupling, ConditionalRealNVP

# wl_stats_torch — add to path if not installed
_WL_STATS_PATH = "/home/tersenov/software/wl_stats_torch"
if _WL_STATS_PATH not in sys.path:
    sys.path.insert(0, _WL_STATS_PATH)
from wl_stats_torch import WLStatistics  # noqa: E402

# Register the local TFDS dataset builder so tfds.load can find it
import tf_dataset_nbody_tomo as _tomo_builder  # noqa: F401, E402

tfb = tfp.bijectors
tfd = tfp.distributions


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


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"batch(\d+)\.pkl$", path.name)
    return int(match.group(1)) if match is not None else -1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="L1-norm + NPE for tomographic weak lensing"
    )

    # Hardware
    p.add_argument("--cuda-visible-devices", type=str, default="0")

    # Survey / map configuration
    p.add_argument("--field-size", type=int, default=10, help="Field size in degrees")
    p.add_argument("--field-npix", type=int, default=80, help="Pixels per side")
    p.add_argument("--nside", type=int, default=512, help="HEALPix NSIDE")
    p.add_argument("--sigma-e", type=float, default=0.26, help="Shape noise dispersion per component")
    p.add_argument("--galaxy-density", type=float, default=30 / 4,
                    help="Galaxy number density [arcmin^{-2}]")
    p.add_argument("--nbins", type=int, default=4, help="Number of tomographic bins")
    p.add_argument("--n-cosmo", type=int, default=6, help="Number of cosmological parameters")

    # L1-norm configuration
    p.add_argument("--n-scales", type=int, default=5, help="Number of starlet wavelet scales")
    p.add_argument("--l1-nbins", type=int, default=40, help="Number of L1-norm histogram bins per scale")
    p.add_argument("--l1-min-snr", type=float, default=-10.0,
                    help="Fixed min SNR for L1-norm binning (recommended default)")
    p.add_argument("--l1-max-snr", type=float, default=10.0,
                    help="Fixed max SNR for L1-norm binning (recommended default)")
    p.add_argument("--auto-calibrate-snr", action="store_true",
                    help="Estimate global SNR range from data instead of using fixed --l1-min/max-snr")
    p.add_argument("--calibration-samples", type=int, default=512,
                    help="Number of maps to use for SNR range calibration")
    p.add_argument("--calibration-margin", type=float, default=0.05,
                    help="Fractional margin to add to calibrated SNR range")
    p.add_argument("--l1-clamp-overflow", action="store_true",
                    help="Clamp SNR values outside [l1-min-snr, l1-max-snr] into edge bins")
    p.add_argument("--subtract-coarse-mean", dest="subtract_coarse_mean", action="store_true",
                    help="Subtract coarse-scale mean before SNR (default: True)")
    p.add_argument("--no-subtract-coarse-mean", dest="subtract_coarse_mean", action="store_false",
                    help="Disable coarse-scale mean subtraction before SNR")
    p.set_defaults(subtract_coarse_mean=True)

    # Map kind
    p.add_argument("--map-kind", type=str, default="nbody",
                    choices=["nbody", "nbody_with_baryon_ia", "gaussian"])

    # Paths
    p.add_argument("--cosmogrid-meta", type=str,
                    default="/home/tersenov/CosmoGridV1/CosmoGridV1_metainfo.h5")
    p.add_argument("--fiducial-map", type=str,
                    default="/home/tersenov/CosmoGridV1/stage3_forecast/fiducial/"
                            "cosmo_fiducial/perm_0000/projected_probes_maps_nobaryons512.h5")
    p.add_argument("--save-dir", type=str,
                    default="/home/tersenov/software/cnn_sbi/scripts/sbi/save_params")
    p.add_argument("--posterior-out", type=str, default="posterior_l1norm_tomo.npy")
    p.add_argument("--figure-out", type=str, default="posterior_l1norm_tomo.png")
    p.add_argument("--cache-dir", type=str, default=None,
                    help="Directory to cache precomputed L1-norm datasets (skip recomputation)")
    p.add_argument(
        "--tfds-name",
        type=str,
        default="NbodyCosmogridDatasetTomo/grid",
        help="TFDS dataset name/config for training and validation maps",
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

    # Dimensionality reduction
    p.add_argument("--pca-components", type=int, default=50,
                    help="Number of PCA components for summary reduction (0 = no PCA)")

    # Flow training hyperparameters
    p.add_argument("--total-steps", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr-init", type=float, default=1e-3)
    p.add_argument("--lr-end", type=float, default=1e-5)
    p.add_argument("--nvp-layers", type=int, default=4, help="Number of RealNVP coupling layers")
    p.add_argument("--nvp-hidden", type=int, default=128, help="Hidden layer width in coupling networks")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    p.add_argument("--grad-clip", type=float, default=1.0,
                    help="Global gradient norm clipping (0 = disabled)")
    p.add_argument("--clip-value", type=float, default=5.0,
                    help="Clip standardized features to ±this value (0 = disabled)")
    p.add_argument("--patience", type=int, default=20,
                    help="Early stopping patience (in val-check intervals, 0 = disabled)")
    p.add_argument("--seed", type=int, default=42)

    # Posterior sampling
    p.add_argument("--npe-samples", type=int, default=100_000)

    # Weights & Biases
    p.add_argument("--wandb-project", type=str, default="l1norm-npe-tomo",
                    help="W&B project name")
    p.add_argument("--wandb-entity", type=str, default=None,
                    help="W&B entity (team or username)")
    p.add_argument("--wandb-run-name", type=str, default=None,
                    help="W&B run name (auto-generated if None)")
    p.add_argument("--no-wandb", action="store_true",
                    help="Disable W&B logging entirely")

    # Execution flags
    p.add_argument("--no-train", action="store_true", help="Load saved flow params instead of training")
    p.add_argument("--no-sample", action="store_true", help="Skip posterior sampling")
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--plot", action="store_true", help="Generate triangle plot")
    p.add_argument("--ds-batch-size", type=int, default=256,
                    help="Batch size for L1-norm computation on datasets")

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
    # PyTorch device
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch device : {torch_device}")
    print(f"JAX backend    : {xla_bridge.get_backend().platform}")
    return torch_device


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
    proj = hp.projector.GnomonicProj(rot=[0, 0, 0], xsize=field_npix, ysize=field_npix, reso=reso)

    if len(tomo_bin_indices) != nbins:
        raise ValueError(
            f"nbins={nbins} is inconsistent with selected bins {tomo_bin_indices}."
        )

    with h5py.File(fid_path, "r") as f:
        kg = f["kg"]
        proj_bins = []
        for b in tomo_bin_indices:
            full_map = np.array(kg[f"stage3_lensing{b}"])
            patch = proj.projmap(full_map, vec2pix_func=partial(hp.vec2pix, nside))
            proj_bins.append(patch)

    # Stack to (H, W, nbins) and add shape noise
    m_data = np.stack(proj_bins, axis=-1).astype(np.float32)
    noise_std = pixel_noise_sigma(sigma_e, galaxy_density, field_size, field_npix)
    noise = jax.random.normal(rng_key, (field_npix, field_npix, nbins)) * noise_std
    m_data = np.array(jnp.asarray(m_data) + noise)
    if apply_bnt:
        m_data = apply_bnt_numpy(m_data)
    print(f"  Observed map shape = {m_data.shape}, noise_std/pixel = {noise_std:.6f}")
    return m_data, cosmo_params, truth


# =============================================================================
# L1-norm computation  (PyTorch / GPU)
# =============================================================================

def build_l1_computer(
    n_scales: int,
    pixel_arcmin: float,
    torch_device: torch.device,
) -> WLStatistics:
    """Instantiate the wavelet L1-norm computer."""
    return WLStatistics(n_scales=n_scales, device=torch_device,
                        pixel_arcmin=pixel_arcmin, dtype=torch.float64)


def calibrate_snr_range(
    stats: WLStatistics,
    augmentation_fn,
    tfds_name: str,
    noise_sigma: float,
    nbins: int,
    n_calibration: int = 512,
    ds_batch_size: int = 64,
    subtract_coarse_mean: bool = True,
    margin: float = 0.05,
) -> Tuple[float, float]:
    """
    Determine global min/max SNR from a pilot sample of training data.

    This ensures all subsequent L1-norm computations use identical bin edges,
    which is critical for the summary vectors to be comparable.
    """
    import tensorflow_datasets as tfds

    print("######## CALIBRATING SNR RANGE ########")
    ds = tfds.load(tfds_name, split="train")
    ds = ds.take(n_calibration)
    ds = ds.map(augmentation_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(ds_batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    device = stats.device
    global_min = float("inf")
    global_max = float("-inf")
    n_used = 0

    for example in ds.as_numpy_iterator():
        maps_np = example["maps"]  # (B, H, W, nbins)
        if np.isnan(maps_np).any():
            continue
        for b in range(nbins):
            img_batch = torch.from_numpy(
                maps_np[:, :, :, b].astype(np.float64)
            ).to(device)
            stats.compute_wavelet_transform(
                img_batch, noise_sigma,
                subtract_coarse_mean=subtract_coarse_mean,
            )
            # Read SNR coefficients directly
            snr = stats.snr_coeffs  # (B, n_scales, H, W)
            batch_min = snr.min().item()
            batch_max = snr.max().item()
            global_min = min(global_min, batch_min)
            global_max = max(global_max, batch_max)
        n_used += len(maps_np)

    # Add margin so no real value falls exactly on the boundary
    span = global_max - global_min
    global_min -= margin * span
    global_max += margin * span

    print(f"  Calibrated from {n_used} maps")
    print(f"  SNR range: [{global_min:.4f}, {global_max:.4f}]")
    return global_min, global_max


def compute_l1_single_map(
    kappa: np.ndarray,
    noise_sigma: float,
    stats: WLStatistics,
    l1_nbins: int,
    nbins: int,
    l1_min_snr: float,
    l1_max_snr: float,
    clamp_overflow: bool = False,
    subtract_coarse_mean: bool = True,
) -> np.ndarray:
    """
    Compute L1-norm summary vector for a single (H, W, nbins) map.

    IMPORTANT: l1_min_snr and l1_max_snr must be fixed globally
    (e.g., from calibrate_snr_range) to ensure consistent bin edges.

    Returns shape (n_scales * l1_nbins * nbins,).
    """
    device = stats.device
    all_l1 = []
    for b in range(nbins):
        img = torch.from_numpy(kappa[:, :, b].astype(np.float64)).to(device)
        stats.compute_wavelet_transform(img, noise_sigma,
                                        subtract_coarse_mean=subtract_coarse_mean)
        _, l1_norms = stats.compute_wavelet_l1_norms(
            n_bins=l1_nbins, min_snr=l1_min_snr, max_snr=l1_max_snr,
            clamp_overflow=clamp_overflow,
        )
        # l1_norms is a list of (l1_nbins,) tensors, one per scale
        bin_vec = torch.cat(l1_norms, dim=-1)  # (n_scales * l1_nbins,)
        all_l1.append(bin_vec.cpu().numpy())
    return np.concatenate(all_l1)  # (n_scales * l1_nbins * nbins,)


def compute_l1_batch(
    maps_batch: np.ndarray,
    noise_sigma: float,
    stats: WLStatistics,
    l1_nbins: int,
    nbins: int,
    l1_min_snr: float,
    l1_max_snr: float,
    clamp_overflow: bool = False,
    subtract_coarse_mean: bool = True,
) -> np.ndarray:
    """
    Compute L1-norm summary vectors for a batch of (B, H, W, nbins) maps.

    IMPORTANT: l1_min_snr and l1_max_snr must be fixed globally
    (e.g., from calibrate_snr_range) to ensure consistent bin edges.

    Returns shape (B, n_scales * l1_nbins * nbins).
    """
    device = stats.device
    B = maps_batch.shape[0]
    all_l1 = []
    for b in range(nbins):
        # (B, H, W) for this tomographic bin
        img_batch = torch.from_numpy(
            maps_batch[:, :, :, b].astype(np.float64)
        ).to(device)
        stats.compute_wavelet_transform(img_batch, noise_sigma,
                                        subtract_coarse_mean=subtract_coarse_mean)
        _, l1_norms = stats.compute_wavelet_l1_norms(
            n_bins=l1_nbins, min_snr=l1_min_snr, max_snr=l1_max_snr,
            clamp_overflow=clamp_overflow,
        )
        # l1_norms: list of (B, l1_nbins) per scale
        bin_vec = torch.cat(l1_norms, dim=-1)  # (B, n_scales * l1_nbins)
        all_l1.append(bin_vec.cpu().numpy())
    # Concatenate across tomo bins: (B, n_scales * l1_nbins * nbins)
    return np.concatenate(all_l1, axis=-1)


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
):
    """Build the TF augmentation pipeline for the tomographic dataset."""
    noise_std = sigma_e / jnp.sqrt(galaxy_density * (field_size * 60 / field_npix) ** 2)

    map_key = {
        "nbody": "map_nbody",
        "nbody_with_baryon_ia": "map_nbody_w_baryon_ia",
        "gaussian": "map_gaussian",
    }[map_kind]
    gather_indices = tf.constant([b - 1 for b in tomo_bin_indices], dtype=tf.int32)

    def augmentation_noise(example):
        x = tf.gather(example[map_key], gather_indices, axis=-1)
        x += tf.random.normal(shape=(field_npix, field_npix, nbins), stddev=noise_std)
        if apply_bnt:
            x = apply_bnt_tf(x)
        return {"maps": x, "theta": example["theta"]}

    def augmentation_flip(example):
        x = example["maps"]
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
        return {"maps": x, "theta": example["theta"]}

    def rescale_h(example):
        x = example["theta"]
        x = tf.tensor_scatter_nd_update(x, [[3]], [x[3] / 100.0])
        return {"maps": example["maps"], "theta": x}

    def augmentation(example):
        return rescale_h(augmentation_flip(augmentation_noise(example)))

    return augmentation


# =============================================================================
# Dataset L1-norm computation
# =============================================================================

def compute_l1_dataset(
    tfds_name: str,
    split: str,
    augmentation_fn,
    stats: WLStatistics,
    noise_sigma: float,
    l1_nbins: int,
    nbins: int,
    ds_batch_size: int,
    l1_min_snr: float,
    l1_max_snr: float,
    clamp_overflow: bool = False,
    subtract_coarse_mean: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Load TFDS dataset, apply augmentation, compute L1-norm summaries.

    IMPORTANT: l1_min_snr and l1_max_snr must be fixed globally
    (e.g., from calibrate_snr_range) to ensure consistent bin edges.

    Returns dict with 'theta' (N, n_cosmo) and 'x' (N, summary_dim).
    """
    import tensorflow_datasets as tfds

    print(f"  Loading {tfds_name} [{split}] ...")
    ds = tfds.load(tfds_name, split=split)
    ds = ds.map(augmentation_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(ds_batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    theta_list: List[np.ndarray] = []
    x_list: List[np.ndarray] = []
    n_processed = 0
    t0 = time.time()

    for example in ds.as_numpy_iterator():
        maps_np = example["maps"]  # (B, H, W, nbins)
        theta_np = example["theta"]  # (B, 6)

        # Skip any batch with NaNs
        if np.isnan(maps_np).any():
            print("    [!] Skipped batch with NaN maps")
            continue

        l1_vec = compute_l1_batch(
            maps_np, noise_sigma, stats, l1_nbins, nbins,
            l1_min_snr=l1_min_snr, l1_max_snr=l1_max_snr,
            clamp_overflow=clamp_overflow,
            subtract_coarse_mean=subtract_coarse_mean,
        )
        x_list.append(l1_vec)
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
# L1-norm diagnostics
# =============================================================================

def plot_l1_diagnostics(
    obs_l1: np.ndarray,
    train_x: np.ndarray,
    train_theta: np.ndarray,
    n_scales: int,
    l1_nbins: int,
    nbins: int,
    param_names: list[str],
    l1_min_snr: float,
    l1_max_snr: float,
):
    """Log L1-norm data vector diagnostics to wandb."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins_per_scale = l1_nbins
    features_per_bin = n_scales * l1_nbins  # per tomo bin
    snr_edges = np.linspace(l1_min_snr, l1_max_snr, l1_nbins + 1)
    snr_centers = 0.5 * (snr_edges[:-1] + snr_edges[1:])

    train_mean = train_x.mean(axis=0)
    train_std_vec = train_x.std(axis=0)

    # Per tomo-bin, per scale: observed L1(SNR) curves
    fig, axes = plt.subplots(nbins, 1, figsize=(10, 3 * nbins), sharex=True)
    if nbins == 1:
        axes = [axes]
    for b in range(nbins):
        ax = axes[b]
        offset = b * features_per_bin
        for s in range(n_scales):
            s_start = offset + s * bins_per_scale
            s_end = s_start + bins_per_scale
            obs_curve = obs_l1[s_start:s_end]
            train_curve_mean = train_mean[s_start:s_end]
            train_curve_std = train_std_vec[s_start:s_end]
            ax.fill_between(snr_centers, train_curve_mean - train_curve_std,
                            train_curve_mean + train_curve_std, alpha=0.15)
            ax.plot(snr_centers, train_curve_mean, lw=0.7, ls="--",
                    label=f"Scale {s} train" if b == 0 else "")
            ax.plot(snr_centers, obs_curve, lw=1.0,
                    label=f"Scale {s} obs" if b == 0 else "")
        ax.set_ylabel(f"Bin {b+1}")
        ax.set_title(f"Tomo bin {b+1}: L1-norm vs SNR")
    axes[-1].set_xlabel("SNR")
    if nbins > 0:
        axes[0].legend(ncol=n_scales, fontsize=7, loc="upper right")
    fig.tight_layout()
    wandb.log({"diagnostics/l1_per_scale_per_bin": wandb.Image(fig)})
    plt.close(fig)


def log_l1_health_diagnostics(
    train_raw_x: np.ndarray,
    obs_raw_x: np.ndarray,
    train_std_x: np.ndarray,
    obs_std_x: np.ndarray,
    clip_value: Optional[float],
):
    """Log compact health diagnostics for raw and standardized L1 summaries."""
    raw_std = train_raw_x.std(axis=0)
    p01 = np.percentile(train_raw_x, 1.0, axis=0)
    p99 = np.percentile(train_raw_x, 99.0, axis=0)
    obs_inlier_frac = np.mean((obs_raw_x >= p01) & (obs_raw_x <= p99))

    diag = {
        "diagnostics/raw_feature_std_min": float(raw_std.min()),
        "diagnostics/raw_feature_std_median": float(np.median(raw_std)),
        "diagnostics/raw_feature_std_max": float(raw_std.max()),
        "diagnostics/raw_dead_feature_frac": float(np.mean(raw_std < 1e-12)),
        "diagnostics/raw_zero_entry_frac": float(np.mean(train_raw_x <= 0.0)),
        "diagnostics/raw_obs_in_train_p01_p99_frac": float(obs_inlier_frac),
        "diagnostics/std_train_absmax": float(np.abs(train_std_x).max()),
        "diagnostics/std_obs_absmax": float(np.abs(obs_std_x).max()),
    }

    if clip_value is not None and clip_value > 0:
        clip_edge = clip_value - 1e-8
        diag["diagnostics/std_train_clipped_frac"] = float(
            np.mean(np.abs(train_std_x) >= clip_edge)
        )
        diag["diagnostics/std_obs_clipped_frac"] = float(
            np.mean(np.abs(obs_std_x) >= clip_edge)
        )

    wandb.log(diag)


# =============================================================================
# Summary standardization
# =============================================================================

def standardize(
    train_x: np.ndarray,
    val_x: np.ndarray,
    obs_x: np.ndarray,
    clip_value: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Log1p + zero-mean unit-variance standardization + clipping.

    L1-norms are non-negative sums of absolute values with heavy right tails.
    Plain mean/std standardization leaves extreme outliers (>200 sigma) that
    cause NaN in the normalizing flow's affine coupling layers.

    The log1p transform compresses the right tail, and clipping at ±clip_value
    bounds any remaining outliers.
    """
    # 1. Log-transform (L1-norms are strictly non-negative)
    train_log = np.log1p(train_x)
    val_log = np.log1p(val_x)
    obs_log = np.log1p(obs_x)

    # 2. Zero-mean, unit-variance from training set
    mean = train_log.mean(axis=0)
    std = train_log.std(axis=0)
    std[std < 1e-12] = 1.0  # avoid division by zero for dead features

    train_std = (train_log - mean) / std
    val_std = (val_log - mean) / std
    obs_std = (obs_log - mean) / std

    # 3. Clip to ±clip_value to prevent flow overflow
    if clip_value is not None and clip_value > 0:
        train_std = np.clip(train_std, -clip_value, clip_value)
        val_std = np.clip(val_std, -clip_value, clip_value)
        obs_std = np.clip(obs_std, -clip_value, clip_value)

    return train_std, val_std, obs_std, mean, std


# =============================================================================
# PCA dimensionality reduction
# =============================================================================

def fit_pca(
    train_x: np.ndarray,
    val_x: np.ndarray,
    obs_x: np.ndarray,
    n_components: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PCA]:
    """Fit whitened PCA on training set and transform all datasets.

    PCA reduces the 800-dim L1-norm summary to a compact representation,
    preventing the flow's coupling layers from being overparameterized
    (which causes overfitting and val-loss divergence).
    Whitening ensures all PCA dimensions have unit variance.
    """
    pca = PCA(n_components=n_components, whiten=True)
    train_pca = pca.fit_transform(train_x).astype(np.float32)
    val_pca = pca.transform(val_x).astype(np.float32)
    obs_pca = pca.transform(obs_x.reshape(1, -1)).astype(np.float32).squeeze(0)
    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"  PCA: {train_x.shape[1]} → {n_components} components "
          f"({explained:.1f}% variance explained)")
    return train_pca, val_pca, obs_pca, pca


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
        loss, grads = jax.value_and_grad(loss_fn)(params, theta_batch, y_batch)
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
    opt_parts.append(optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay))
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
        loss, params, opt_state = update(params, opt_state, theta_train[idx], x_train[idx])
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
            val_l = float(-jnp.mean(nf_logp.apply(params, theta_val[vidx], x_val[vidx])))
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
                with open(save_dir / "params_l1norm_flow_best.pkl", "wb") as f:
                    pickle.dump(params, f)
            else:
                patience_counter += 1

            with open(save_dir / f"params_l1norm_flow_batch{step}.pkl", "wb") as f:
                pickle.dump(params, f)
            wandb.log({
                "val/loss": val_l,
                "val/best_loss": best_val_loss,
                "val/patience_counter": patience_counter,
                "step": step,
            }, step=step)
            print(f"  Saved @ step {step}. Val loss = {val_l:.4f}{improved}"
                  f"  (best = {best_val_loss:.4f}, patience = {patience_counter})")

            # Early stopping
            if patience > 0 and patience_counter >= patience:
                print(f"  Early stopping at step {step} "
                      f"(no val improvement for {patience} checks)")
                break

    # Save loss curves
    np.save(save_dir / "loss_train_l1norm.npy", np.array(batch_losses))
    np.save(save_dir / "loss_val_l1norm.npy", np.array(val_losses))
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
    y_cond = jnp.ones([n_samples, summary_dim]) * jnp.asarray(summary_obs)
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
        param_names = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]

    mcsamples = MCSamples(samples=samples, names=param_names, labels=param_names)

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
            wandb.run.summary[f"posterior/{name}/bias"] = float(s.mean() - truth[i])


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    tomo_bin_indices = parse_tomo_bin_indices(args.tomo_bin_indices)
    if args.nbins != len(tomo_bin_indices):
        print(
            f"  Overriding nbins from {args.nbins} to {len(tomo_bin_indices)} "
            f"to match selected bins {tomo_bin_indices}."
        )
        args.nbins = len(tomo_bin_indices)
    if args.apply_bnt:
        validate_bnt_configuration(args.nbins, tomo_bin_indices)
    torch_device = setup_environment(args.cuda_visible_devices)
    rng = jax.random.PRNGKey(args.seed)
    rng, rng_obs, rng_sample = jax.random.split(rng, 3)

    # Derived quantities
    pixel_arcmin = args.field_size * 60.0 / args.field_npix
    noise_sigma = pixel_noise_sigma(
        args.sigma_e, args.galaxy_density, args.field_size, args.field_npix
    )
    raw_summary_dim = args.n_scales * args.l1_nbins * args.nbins
    summary_dim = args.pca_components if args.pca_components > 0 else raw_summary_dim
    print(f"  pixel_arcmin   = {pixel_arcmin:.2f}")
    print(f"  noise_sigma    = {noise_sigma:.6f}")
    print(f"  raw_summary    = {raw_summary_dim}  "
          f"({args.n_scales} scales × {args.l1_nbins} bins × {args.nbins} tomo bins)")
    print(f"  summary_dim    = {summary_dim}  "
          f"({'PCA-' + str(args.pca_components) if args.pca_components > 0 else 'no PCA'})")

    param_names = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]

    # ------------------------------------------------------------------
    # 0. Initialize Weights & Biases
    # ------------------------------------------------------------------
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
            tags=[args.map_kind, f"pca{args.pca_components}", f"nvp{args.nvp_layers}"],
        )
    else:
        wandb.init(mode="disabled")

    # ------------------------------------------------------------------
    # 1. Observed map
    # ------------------------------------------------------------------
    m_data, cosmo_params, truth = load_observed_map(
        args.cosmogrid_meta, args.fiducial_map,
        args.field_size, args.field_npix, args.nside, args.nbins,
        tomo_bin_indices,
        args.sigma_e, args.galaxy_density, rng_obs,
        apply_bnt=args.apply_bnt,
    )

    # ------------------------------------------------------------------
    # 2. L1-norm computer + SNR calibration
    # ------------------------------------------------------------------
    stats = build_l1_computer(args.n_scales, pixel_arcmin, torch_device)

    augmentation = build_augmentation(
        args.map_kind, args.sigma_e, args.galaxy_density,
        args.field_size, args.field_npix, args.nbins, tomo_bin_indices,
        apply_bnt=args.apply_bnt,
    )

    # Resolve cache directory once (used for both calibration and dataset caching)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    # Calibrate global SNR range if requested; otherwise use fixed values
    if not args.auto_calibrate_snr:
        l1_min_snr = args.l1_min_snr
        l1_max_snr = args.l1_max_snr
        print(f"  Using fixed SNR range: [{l1_min_snr}, {l1_max_snr}]")
    else:
        # Check for cached calibration
        calib_cache = cache_dir / "snr_calibration.npz" if cache_dir else None
        if calib_cache is not None and calib_cache.exists():
            calib = np.load(calib_cache)
            l1_min_snr = float(calib["min_snr"])
            l1_max_snr = float(calib["max_snr"])
            print(f"  Loaded cached SNR range: [{l1_min_snr:.4f}, {l1_max_snr:.4f}]")
        else:
            l1_min_snr, l1_max_snr = calibrate_snr_range(
                stats, augmentation,
                args.tfds_name,
                noise_sigma, args.nbins,
                n_calibration=args.calibration_samples,
                ds_batch_size=args.ds_batch_size,
                subtract_coarse_mean=args.subtract_coarse_mean,
                margin=args.calibration_margin,
            )
            if calib_cache is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)
                np.savez(calib_cache, min_snr=l1_min_snr, max_snr=l1_max_snr)
                print(f"  Cached SNR calibration to {calib_cache}")

    wandb.config.update({
        "l1_min_snr_calibrated": l1_min_snr,
        "l1_max_snr_calibrated": l1_max_snr,
        "l1_clamp_overflow": args.l1_clamp_overflow,
        "l1_auto_calibrate_snr": args.auto_calibrate_snr,
        "l1_subtract_coarse_mean": args.subtract_coarse_mean,
    }, allow_val_change=True)

    # 2a. L1-norm for observed map
    print("######## L1-NORM: OBSERVED MAP ########")
    obs_l1 = compute_l1_single_map(
        m_data, noise_sigma, stats, args.l1_nbins, args.nbins,
        l1_min_snr=l1_min_snr, l1_max_snr=l1_max_snr,
        clamp_overflow=args.l1_clamp_overflow,
        subtract_coarse_mean=args.subtract_coarse_mean,
    )
    print(f"  Observed L1-norm vector shape = {obs_l1.shape}")

    # ------------------------------------------------------------------
    # 3. L1-norm for train / test datasets
    # ------------------------------------------------------------------
    print("######## L1-NORM: DATASETS ########")

    cache_ok = False

    # Try loading from cache (only if SNR range matches)
    if cache_dir is not None and cache_dir.exists():
        train_cache = cache_dir / "l1_train.npz"
        val_cache = cache_dir / "l1_val.npz"
        meta_cache = cache_dir / "l1_cache_meta.npz"
        if train_cache.exists() and val_cache.exists() and meta_cache.exists():
            meta = np.load(meta_cache)
            required_meta = {
                "l1_min_snr", "l1_max_snr", "l1_nbins",
                "l1_clamp_overflow", "subtract_coarse_mean", "n_scales",
                "tfds_name", "tomo_bin_indices", "apply_bnt", "bnt_matrix_version",
            }
            if not required_meta.issubset(set(meta.files)):
                print("  Cache metadata is missing newer L1 settings; recomputing ...")
            else:
                cached_min = float(meta["l1_min_snr"])
                cached_max = float(meta["l1_max_snr"])
                cached_nbins = int(meta["l1_nbins"])
                cached_clamp = bool(meta["l1_clamp_overflow"])
                cached_subtract = bool(meta["subtract_coarse_mean"])
                cached_n_scales = int(meta["n_scales"])
                cached_tfds_name = str(meta["tfds_name"])
                cached_tomo_bins = str(meta["tomo_bin_indices"])
                cached_apply_bnt = bool(meta["apply_bnt"])
                cached_bnt_version = str(meta["bnt_matrix_version"])
                if (abs(cached_min - l1_min_snr) < 1e-6 and
                        abs(cached_max - l1_max_snr) < 1e-6 and
                        cached_nbins == args.l1_nbins and
                        cached_clamp == args.l1_clamp_overflow and
                        cached_subtract == args.subtract_coarse_mean and
                        cached_n_scales == args.n_scales and
                        cached_tfds_name == args.tfds_name and
                        cached_tomo_bins == ",".join(str(b) for b in tomo_bin_indices) and
                        cached_apply_bnt == bool(args.apply_bnt) and
                        cached_bnt_version == (BNT_MATRIX_VERSION if args.apply_bnt else "none")):
                    print("  Loading cached L1-norm datasets (metadata matches) ...")
                    d_tr = np.load(train_cache)
                    dataset_train = {"theta": d_tr["theta"], "x": d_tr["x"]}
                    d_va = np.load(val_cache)
                    dataset_val = {"theta": d_va["theta"], "x": d_va["x"]}
                    cache_ok = True
                    print(f"  Train: {len(dataset_train['theta'])} | Val: {len(dataset_val['theta'])}")
                else:
                    print(
                        "  Cache metadata does not match current L1 settings "
                        "(SNR range / nbins / clamp / coarse-mean / n_scales / BNT). Recomputing ..."
                    )

    if not cache_ok:
        dataset_train = compute_l1_dataset(
            args.tfds_name, "train", augmentation, stats,
            noise_sigma, args.l1_nbins, args.nbins, args.ds_batch_size,
            l1_min_snr=l1_min_snr, l1_max_snr=l1_max_snr,
            clamp_overflow=args.l1_clamp_overflow,
            subtract_coarse_mean=args.subtract_coarse_mean,
        )
        dataset_val = compute_l1_dataset(
            args.tfds_name, "test", augmentation, stats,
            noise_sigma, args.l1_nbins, args.nbins, args.ds_batch_size,
            l1_min_snr=l1_min_snr, l1_max_snr=l1_max_snr,
            clamp_overflow=args.l1_clamp_overflow,
            subtract_coarse_mean=args.subtract_coarse_mean,
        )
        # Save cache with metadata
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(cache_dir / "l1_train.npz",
                     theta=dataset_train["theta"], x=dataset_train["x"])
            np.savez(cache_dir / "l1_val.npz",
                     theta=dataset_val["theta"], x=dataset_val["x"])
            np.savez(cache_dir / "l1_cache_meta.npz",
                     l1_min_snr=l1_min_snr, l1_max_snr=l1_max_snr,
                     l1_nbins=args.l1_nbins,
                     l1_clamp_overflow=args.l1_clamp_overflow,
                     subtract_coarse_mean=args.subtract_coarse_mean,
                     n_scales=args.n_scales,
                     tfds_name=args.tfds_name,
                     tomo_bin_indices=",".join(str(b) for b in tomo_bin_indices),
                     apply_bnt=bool(args.apply_bnt),
                     bnt_matrix_version=(BNT_MATRIX_VERSION if args.apply_bnt else "none"))
            print(f"  Cached datasets to {cache_dir}")

    print(f"  Train x shape = {dataset_train['x'].shape}")
    print(f"  Val   x shape = {dataset_val['x'].shape}")

    # ------------------------------------------------------------------
    # 3b. L1-norm diagnostics (before standardization)
    # ------------------------------------------------------------------
    print("######## L1-NORM DIAGNOSTICS ########")
    plot_l1_diagnostics(
        obs_l1, dataset_train["x"], dataset_train["theta"],
        args.n_scales, args.l1_nbins, args.nbins,
        param_names, l1_min_snr, l1_max_snr,
    )

    # ------------------------------------------------------------------
    # 4. Standardize summaries
    # ------------------------------------------------------------------
    print("######## STANDARDIZE ########")
    train_x_raw = dataset_train["x"]
    obs_l1_raw = obs_l1
    clip_val = args.clip_value if args.clip_value > 0 else None
    dataset_train["x"], dataset_val["x"], obs_l1_std, mean, std = standardize(
        dataset_train["x"], dataset_val["x"], obs_l1, clip_value=clip_val,
    )
    print(f"  Summary mean range = [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  Summary std  range = [{std.min():.4f}, {std.max():.4f}]")
    log_l1_health_diagnostics(
        train_x_raw, obs_l1_raw, dataset_train["x"], obs_l1_std, clip_val
    )

    # Log dataset & summary statistics to wandb
    wandb.log({
        "data/train_size": len(dataset_train["theta"]),
        "data/val_size": len(dataset_val["theta"]),
        "data/raw_summary_dim": raw_summary_dim,
        "data/summary_mean_min": float(mean.min()),
        "data/summary_mean_max": float(mean.max()),
        "data/summary_std_min": float(std.min()),
        "data/summary_std_max": float(std.max()),
        "data/train_x_min": float(dataset_train["x"].min()),
        "data/train_x_max": float(dataset_train["x"].max()),
        "data/train_x_mean": float(dataset_train["x"].mean()),
        "data/train_x_std": float(dataset_train["x"].std()),
        "data/obs_l1_min": float(obs_l1.min()),
        "data/obs_l1_max": float(obs_l1.max()),
    })

    # Log theta distributions as histograms
    for i, name in enumerate(param_names):
        wandb.log({
            f"data/theta_train/{name}": wandb.Histogram(dataset_train["theta"][:, i]),
            f"data/theta_val/{name}": wandb.Histogram(dataset_val["theta"][:, i]),
        })

    # ------------------------------------------------------------------
    # 4b. PCA dimensionality reduction
    # ------------------------------------------------------------------
    pca_model = None
    if args.pca_components > 0:
        print("######## PCA REDUCTION ########")
        dataset_train["x"], dataset_val["x"], obs_l1_std, pca_model = fit_pca(
            dataset_train["x"], dataset_val["x"], obs_l1_std,
            n_components=args.pca_components,
        )
        print(f"  Train x shape after PCA = {dataset_train['x'].shape}")

    # ------------------------------------------------------------------
    # 5. Build & train flow
    # ------------------------------------------------------------------
    nf_logp, nf_sample = build_flow(
        n_cosmo_params=args.n_cosmo,
        n_layers=args.nvp_layers,
        hidden=args.nvp_hidden,
    )

    save_path = Path(args.save_dir) / "l1norm" / args.map_kind
    flow_summary_path = save_path / "flow_training_summary.json"
    flow_params = None
    flow_params_source = "unknown"

    if args.no_train:
        # Prefer best model, fall back to latest checkpoint
        best_path = save_path / "params_l1norm_flow_best.pkl"
        if best_path.exists():
            load_path = best_path
        else:
            candidates = sorted(
                save_path.glob("params_l1norm_flow_batch*.pkl"),
                key=_checkpoint_step,
            )
            if not candidates:
                raise FileNotFoundError(f"No saved flow params in {save_path} and --no-train set")
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

    # Save preprocessing artifacts alongside flow params
    save_path.mkdir(parents=True, exist_ok=True)
    save_dict = {"mean": mean, "std": std}
    if pca_model is not None:
        save_dict["pca_components"] = pca_model.components_
        save_dict["pca_mean"] = pca_model.mean_
        save_dict["pca_explained_variance"] = pca_model.explained_variance_
    np.savez(save_path / "l1_standardization.npz", **save_dict)

    # ------------------------------------------------------------------
    # 6. Posterior sampling
    # ------------------------------------------------------------------
    if not args.no_sample:
        posterior_samples = sample_posterior(
            rng_sample, nf_sample, flow_params,
            obs_l1_std, args.npe_samples,
        )
        out = Path(args.posterior_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, posterior_samples)
        metadata = {
            "method": "l1norm",
            "posterior_file": str(out.resolve()),
            "flow_params_source": flow_params_source,
            "total_steps": int(args.total_steps),
            "save_every": int(args.save_every),
            "patience": int(args.patience),
            "npe_samples": int(args.npe_samples),
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
            plot_posterior(posterior_samples, truth, str(fig_out), param_names,
                           log_to_wandb=(not args.no_wandb))
    else:
        print("  Skipping posterior sampling (--no-sample)")

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
