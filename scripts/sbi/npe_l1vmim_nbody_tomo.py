#!/usr/bin/env python
"""
L1 datavectors + VMIM MLP compressor + NPE for tomographic weak-lensing inference.

This script follows the CNN VMIM prescription, but replaces map-based compression with
an MLP compressor trained on wavelet L1 summary vectors.

Pipeline:
  1) Build observed/train/val L1 vectors using wl_stats_torch
  2) Train (or load) VMIM compressor on L1 vectors
  3) Compress observed/train/val L1 vectors
  4) Train (or load) conditional RealNVP on compressed summaries
  5) Sample posterior and save artifacts
"""
from __future__ import annotations

import argparse
import hashlib
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


if not hasattr(wandb, "init"):
    class _WandbConfig(dict):
        def update(self, *args, **kwargs):
            kwargs.pop("allow_val_change", None)
            return super().update(*args, **kwargs)

    class _WandbRun:
        def __init__(self):
            self.summary: dict[str, object] = {}

    class _WandbMedia:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _WandbStub:
        Image = _WandbMedia
        Histogram = _WandbMedia

        def __init__(self):
            self.run: Optional[_WandbRun] = None
            self.config = _WandbConfig()

        def init(self, *args, **kwargs):
            self.run = _WandbRun()
            return self.run

        def log(self, *args, **kwargs):
            return None

        def define_metric(self, *args, **kwargs):
            return None

        def finish(self):
            return None

    wandb = _WandbStub()

import h5py
import haiku as hk
import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tensorflow_probability as tfp
import torch
from jax.lib import xla_bridge

if not hasattr(np, "issctype"):
    # Compatibility for NumPy >= 2.0 with tensorflow_probability paths that
    # still reference np.issctype.
    def _np_issctype(rep):
        try:
            return issubclass(np.dtype(rep).type, np.generic)
        except Exception:
            return False

    np.issctype = _np_issctype  # type: ignore[attr-defined]

if not hasattr(tfp, "substrates"):
    # Compatibility shim for environments where top-level tfp does not expose
    # the substrates namespace expected by sbi_lens.
    import types
    from tensorflow_probability.substrates import jax as tfp_jax

    tfp.substrates = types.SimpleNamespace(jax=tfp_jax)

try:
    from sbi_lens.normflow.models import AffineCoupling, ConditionalRealNVP
    from sbi_lens.normflow.train_model import TrainModel
except ImportError as exc:  # pragma: no cover - runtime env dependent
    raise ImportError(
        "Failed to import sbi_lens. Install it in the active environment, e.g.:\n"
        "  pip install 'sbi_lens @ "
        "git+https://github.com/DifferentiableUniverseInitiative/sbi_lens.git'"
    ) from exc

# Register local TFDS dataset builder
import tf_dataset_nbody_tomo as _tomo_builder  # noqa: F401, E402

# wl_stats_torch
_WL_STATS_PATH = "/home/tersenov/software/wl_stats_torch"
if _WL_STATS_PATH not in sys.path:
    sys.path.insert(0, _WL_STATS_PATH)
from wl_stats_torch import WLStatistics  # noqa: E402

from bnt_utils import (
    BNT_MATRIX_VERSION,
    apply_bnt_numpy,
    apply_bnt_tf,
    validate_bnt_configuration,
)


def parse_tomo_bin_indices(spec: str) -> tuple[int, ...]:
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

    deduped: list[int] = []
    seen = set()
    for b in values:
        if b not in seen:
            deduped.append(b)
            seen.add(b)
    return tuple(deduped)


def parse_hidden_sizes(spec: str) -> tuple[int, ...]:
    vals: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        v = int(token)
        if v <= 0:
            raise ValueError("All --compressor-hidden sizes must be > 0.")
        vals.append(v)
    if not vals:
        raise ValueError("--compressor-hidden must define at least one hidden layer.")
    return tuple(vals)


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"batch(\d+)\.pkl$", path.name)
    return int(match.group(1)) if match is not None else -1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="L1 datavectors + VMIM MLP compressor + NPE"
    )

    # Hardware
    p.add_argument("--cuda-visible-devices", type=str, default="0")
    p.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Alias for --cuda-visible-devices",
    )

    # Survey / map configuration
    p.add_argument("--field-size", type=int, default=10, help="Field size in degrees")
    p.add_argument("--field-npix", type=int, default=80, help="Pixels per side")
    p.add_argument("--nside", type=int, default=512, help="HEALPix NSIDE")
    p.add_argument("--sigma-e", type=float, default=0.26, help="Shape-noise sigma_e")
    p.add_argument(
        "--galaxy-density",
        type=float,
        default=30 / 4,
        help="Galaxy number density [arcmin^-2]",
    )
    p.add_argument("--nbins", type=int, default=4, help="Number of tomographic bins")
    p.add_argument("--n-cosmo", type=int, default=6, help="Number of cosmological parameters")

    # L1 configuration
    p.add_argument("--n-scales", type=int, default=5, help="Number of starlet scales")
    p.add_argument("--l1-nbins", type=int, default=40, help="L1 histogram bins per scale")
    p.add_argument("--l1-min-snr", type=float, default=-10.0, help="Fixed minimum SNR")
    p.add_argument("--l1-max-snr", type=float, default=10.0, help="Fixed maximum SNR")
    p.add_argument(
        "--auto-calibrate-snr",
        action="store_true",
        help="Calibrate global SNR range from train data",
    )
    p.add_argument("--calibration-samples", type=int, default=512)
    p.add_argument("--calibration-margin", type=float, default=0.05)
    p.add_argument(
        "--l1-clamp-overflow",
        action="store_true",
        help="Clamp SNR values outside [l1-min-snr, l1-max-snr]",
    )
    p.add_argument(
        "--subtract-coarse-mean",
        dest="subtract_coarse_mean",
        action="store_true",
        help="Subtract coarse mean before SNR (default)",
    )
    p.add_argument(
        "--no-subtract-coarse-mean",
        dest="subtract_coarse_mean",
        action="store_false",
        help="Disable coarse-mean subtraction",
    )
    p.set_defaults(subtract_coarse_mean=True)
    p.add_argument(
        "--l1-implementation",
        type=str,
        default="cnn_sbi",
        choices=["cnn_sbi", "cosmoford"],
        help="L1 extraction mode",
    )

    # Map kind
    p.add_argument(
        "--map-kind",
        type=str,
        default="nbody",
        choices=["nbody", "nbody_with_baryon_ia", "gaussian"],
    )

    # Paths
    p.add_argument(
        "--cosmogrid-meta",
        type=str,
        default="/home/tersenov/CosmoGridV1/CosmoGridV1_metainfo.h5",
    )
    p.add_argument(
        "--fiducial-map",
        type=str,
        default=(
            "/home/tersenov/CosmoGridV1/stage3_forecast/fiducial/"
            "cosmo_fiducial/perm_0000/projected_probes_maps_nobaryons512.h5"
        ),
    )
    p.add_argument(
        "--save-dir",
        type=str,
        default="/home/tersenov/software/cnn_sbi/scripts/sbi/save_params",
    )
    p.add_argument("--posterior-out", type=str, default="posterior_l1vmim_tomo.npy")
    p.add_argument("--figure-out", type=str, default="posterior_l1vmim_tomo.png")
    p.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache dir for raw/compressed L1 datasets",
    )
    p.add_argument(
        "--tfds-name",
        type=str,
        default="NbodyCosmogridDatasetTomo/grid",
        help="TFDS dataset name/config",
    )
    p.add_argument(
        "--tomo-bin-indices",
        type=str,
        default="1,2,3,4",
        help="Tomographic bins to use",
    )
    p.add_argument(
        "--apply-bnt",
        action="store_true",
        help="Apply BNT transform after shape-noise injection.",
    )

    # VMIM compressor configuration
    p.add_argument("--compressor-dim", type=int, default=6, help="Compressed summary dimension")
    p.add_argument(
        "--compressor-hidden",
        type=str,
        default="256,256",
        help="Comma-separated hidden widths for MLP compressor",
    )
    p.add_argument(
        "--compressor-vmim-nf-layers",
        type=int,
        default=4,
        help="Number of RealNVP coupling layers for VMIM companion NF",
    )
    p.add_argument(
        "--compressor-vmim-nf-hidden",
        type=int,
        default=128,
        help="Hidden width for VMIM companion NF coupling MLPs",
    )
    p.add_argument(
        "--train-compressor",
        action="store_true",
        help="Train MLP compressor with VMIM objective",
    )
    p.add_argument(
        "--compressor-params",
        type=str,
        default=None,
        help="Path to pretrained compressor params (required if not training compressor)",
    )
    p.add_argument(
        "--compressor-state",
        type=str,
        default=None,
        help="Path to pretrained compressor state (required if not training compressor)",
    )
    p.add_argument("--compressor-steps", type=int, default=150_000)
    p.add_argument("--compressor-lr", type=float, default=5e-4)
    p.add_argument("--compressor-batch-size", type=int, default=128)
    p.add_argument("--compressor-save-every", type=int, default=2000)
    p.add_argument(
        "--abort-on-nonfinite-compressor",
        action="store_true",
        help="Abort compressor training immediately on any non-finite loss",
    )
    p.add_argument(
        "--max-nonfinite-compressor-events",
        type=int,
        default=20,
        help="Abort compressor training if non-finite events exceed this count",
    )
    p.add_argument(
        "--compress-batch-size",
        type=int,
        default=4096,
        help="Batch size for applying compressor to cached L1 vectors",
    )
    p.add_argument(
        "--compressor-log1p-input",
        action="store_true",
        help="Apply log1p transform to raw L1 vectors before VMIM compressor training/application",
    )
    p.add_argument(
        "--compressor-input-standardize",
        action="store_true",
        help=(
            "Z-score raw/log1p L1 vectors before VMIM compressor training/application "
            "(stats from train set only)"
        ),
    )
    p.add_argument(
        "--compressor-input-clip",
        type=float,
        default=0.0,
        help=(
            "Clip standardized compressor inputs to +/- value (<=0 disables). "
            "Only applies with --compressor-input-standardize"
        ),
    )

    # Flow training hyperparameters
    p.add_argument("--total-steps", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr-init", type=float, default=1e-3)
    p.add_argument("--lr-end", type=float, default=1e-5)
    p.add_argument("--nvp-layers", type=int, default=4)
    p.add_argument("--nvp-hidden", type=int, default=128)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)

    # Summary preprocessing before flow
    p.add_argument(
        "--standardize-summary",
        dest="standardize_summary",
        action="store_true",
        help="Z-score normalize compressed summaries before flow",
    )
    p.add_argument(
        "--no-standardize-summary",
        dest="standardize_summary",
        action="store_false",
        help="Disable compressed-summary standardization",
    )
    p.set_defaults(standardize_summary=True)
    p.add_argument(
        "--summary-clip-value",
        type=float,
        default=5.0,
        help="Clip standardized compressed summaries to +/- this value (0 disables)",
    )

    # Posterior sampling
    p.add_argument("--npe-samples", type=int, default=100_000)

    # Weights & Biases
    p.add_argument("--wandb-project", type=str, default="l1-vmim-npe-tomo")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Optional W&B group; default auto-groups by method/map kind",
    )
    p.add_argument(
        "--wandb-tags",
        type=str,
        default="",
        help="Additional comma-separated W&B tags",
    )
    p.add_argument("--no-wandb", action="store_true")

    # Execution flags
    p.add_argument("--no-train", action="store_true", help="Load flow checkpoint; skip flow training")
    p.add_argument("--no-sample", action="store_true", help="Skip posterior sampling")
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--ds-batch-size", type=int, default=256, help="Batch size for L1 extraction from TFDS")

    args = p.parse_args()
    if args.gpu is not None:
        args.cuda_visible_devices = str(args.gpu)

    args.compressor_hidden_sizes = parse_hidden_sizes(args.compressor_hidden)

    if args.compressor_dim <= 0:
        raise ValueError("--compressor-dim must be > 0.")
    if args.compressor_steps <= 0:
        raise ValueError("--compressor-steps must be > 0.")
    if args.compressor_batch_size <= 0:
        raise ValueError("--compressor-batch-size must be > 0.")
    if args.compressor_save_every <= 0:
        raise ValueError("--compressor-save-every must be > 0.")
    if args.compressor_steps < args.compressor_save_every:
        raise ValueError("--compressor-steps must be >= --compressor-save-every.")
    if args.total_steps <= 0:
        raise ValueError("--total-steps must be > 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.save_every <= 0:
        raise ValueError("--save-every must be > 0.")
    if args.compress_batch_size <= 0:
        raise ValueError("--compress-batch-size must be > 0.")
    if args.max_nonfinite_compressor_events < 0:
        raise ValueError("--max-nonfinite-compressor-events must be >= 0.")
    if args.compressor_vmim_nf_layers <= 0:
        raise ValueError("--compressor-vmim-nf-layers must be > 0.")
    if args.compressor_vmim_nf_hidden <= 0:
        raise ValueError("--compressor-vmim-nf-hidden must be > 0.")
    if args.no_train and args.train_compressor:
        raise ValueError("--no-train cannot be combined with --train-compressor.")

    return args


def setup_environment(cuda_devices: str) -> torch.device:
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    print(f"CUDA_VISIBLE_DEVICES = {cuda_devices}")

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"TF GPU config: {exc}")

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch device : {torch_device}")
    print(f"JAX backend    : {xla_bridge.get_backend().platform}")
    return torch_device


def pixel_noise_sigma(
    sigma_e: float,
    galaxy_density: float,
    field_size: float,
    field_npix: int,
) -> float:
    reso_arcmin = field_size * 60.0 / field_npix
    return sigma_e / np.sqrt(galaxy_density * reso_arcmin ** 2)


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
    print("######## OBSERVED DATA ########")
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
    print(f"  Truth = {truth}")

    reso = field_size * 60.0 / field_npix
    proj = hp.projector.GnomonicProj(
        rot=[0, 0, 0], xsize=field_npix, ysize=field_npix, reso=reso
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
            patch = proj.projmap(full_map, vec2pix_func=partial(hp.vec2pix, nside))
            proj_bins.append(patch)

    m_data = np.stack(proj_bins, axis=-1).astype(np.float32)
    noise_std = pixel_noise_sigma(sigma_e, galaxy_density, field_size, field_npix)
    noise = jax.random.normal(rng_key, (field_npix, field_npix, nbins)) * noise_std
    m_data = np.array(jnp.asarray(m_data) + noise)
    if apply_bnt:
        m_data = apply_bnt_numpy(m_data)
    print(f"  Observed map shape = {m_data.shape}, noise_std/pixel = {noise_std:.6f}")
    return m_data, cosmo_params, truth


def build_l1_computer(
    n_scales: int,
    pixel_arcmin: float,
    torch_device: torch.device,
    l1_implementation: str = "cnn_sbi",
) -> WLStatistics:
    dtype = torch.float32 if l1_implementation == "cosmoford" else torch.float64
    return WLStatistics(
        n_scales=n_scales,
        device=torch_device,
        pixel_arcmin=pixel_arcmin,
        dtype=dtype,
    )


def calibrate_snr_range(
    stats: WLStatistics,
    augmentation_fn,
    tfds_name: str,
    noise_sigma: float,
    nbins: int,
    l1_implementation: str = "cnn_sbi",
    n_calibration: int = 512,
    ds_batch_size: int = 64,
    subtract_coarse_mean: bool = True,
    margin: float = 0.05,
) -> Tuple[float, float]:
    import tensorflow_datasets as tfds

    print("######## CALIBRATING SNR RANGE ########")
    ds = tfds.load(tfds_name, split="train")
    ds = ds.take(n_calibration)
    ds = ds.map(augmentation_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(ds_batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    global_min = float("inf")
    global_max = float("-inf")
    n_used = 0

    for example in ds.as_numpy_iterator():
        maps_np = example["maps"]
        if np.isnan(maps_np).any():
            continue
        for b in range(nbins):
            map_dtype = np.float32 if l1_implementation == "cosmoford" else np.float64
            img_batch = torch.from_numpy(maps_np[:, :, :, b].astype(map_dtype)).to(stats.device)
            if l1_implementation == "cosmoford":
                stats.compute_wavelet_transform(img_batch.float(), float(noise_sigma))
            else:
                stats.compute_wavelet_transform(
                    img_batch,
                    noise_sigma,
                    subtract_coarse_mean=subtract_coarse_mean,
                )
            snr = stats.snr_coeffs
            global_min = min(global_min, snr.min().item())
            global_max = max(global_max, snr.max().item())
        n_used += len(maps_np)

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
    l1_implementation: str = "cnn_sbi",
) -> np.ndarray:
    all_l1 = []
    for b in range(nbins):
        map_dtype = np.float32 if l1_implementation == "cosmoford" else np.float64
        img = torch.from_numpy(kappa[:, :, b].astype(map_dtype)).to(stats.device)
        if l1_implementation == "cosmoford":
            stats.compute_wavelet_transform(img.float(), float(noise_sigma))
            clamp_this = False
        else:
            stats.compute_wavelet_transform(
                img,
                noise_sigma,
                subtract_coarse_mean=subtract_coarse_mean,
            )
            clamp_this = clamp_overflow
        _, l1_norms = stats.compute_wavelet_l1_norms(
            n_bins=l1_nbins,
            min_snr=l1_min_snr,
            max_snr=l1_max_snr,
            clamp_overflow=clamp_this,
        )
        bin_vec = torch.cat(l1_norms, dim=-1)
        all_l1.append(bin_vec.cpu().numpy())
    return np.concatenate(all_l1).astype(np.float32)


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
    l1_implementation: str = "cnn_sbi",
) -> np.ndarray:
    all_l1 = []
    for b in range(nbins):
        map_dtype = np.float32 if l1_implementation == "cosmoford" else np.float64
        img_batch = torch.from_numpy(maps_batch[:, :, :, b].astype(map_dtype)).to(stats.device)
        if l1_implementation == "cosmoford":
            stats.compute_wavelet_transform(img_batch.float(), float(noise_sigma))
            clamp_this = False
        else:
            stats.compute_wavelet_transform(
                img_batch,
                noise_sigma,
                subtract_coarse_mean=subtract_coarse_mean,
            )
            clamp_this = clamp_overflow
        _, l1_norms = stats.compute_wavelet_l1_norms(
            n_bins=l1_nbins,
            min_snr=l1_min_snr,
            max_snr=l1_max_snr,
            clamp_overflow=clamp_this,
        )
        bin_vec = torch.cat(l1_norms, dim=-1)
        all_l1.append(bin_vec.cpu().numpy())
    return np.concatenate(all_l1, axis=-1).astype(np.float32)


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
    l1_implementation: str = "cnn_sbi",
) -> Dict[str, np.ndarray]:
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
        maps_np = example["maps"]
        theta_np = example["theta"]
        if np.isnan(maps_np).any():
            print("    [!] Skipped batch with NaN maps")
            continue
        l1_vec = compute_l1_batch(
            maps_np,
            noise_sigma,
            stats,
            l1_nbins,
            nbins,
            l1_min_snr=l1_min_snr,
            l1_max_snr=l1_max_snr,
            clamp_overflow=clamp_overflow,
            subtract_coarse_mean=subtract_coarse_mean,
            l1_implementation=l1_implementation,
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
        "theta": np.concatenate(theta_list, axis=0).astype(np.float32),
        "x": np.concatenate(x_list, axis=0).astype(np.float32),
    }


def build_l1_cache_metadata(
    args: argparse.Namespace,
    tomo_bin_indices: tuple[int, ...],
    l1_min_snr: float,
    l1_max_snr: float,
    l1_clamp_overflow: bool,
    subtract_coarse_mean: bool,
) -> Dict[str, object]:
    # Keep raw L1 cache schema aligned with existing L1 pipelines so previously
    # produced datavector caches remain reusable.
    return {
        "l1_min_snr": float(l1_min_snr),
        "l1_max_snr": float(l1_max_snr),
        "l1_nbins": int(args.l1_nbins),
        "l1_clamp_overflow": bool(l1_clamp_overflow),
        "subtract_coarse_mean": bool(subtract_coarse_mean),
        "l1_implementation": str(args.l1_implementation),
        "n_scales": int(args.n_scales),
        "tfds_name": str(args.tfds_name),
        "tomo_bin_indices": ",".join(str(b) for b in tomo_bin_indices),
        "apply_bnt": bool(args.apply_bnt),
        "bnt_matrix_version": BNT_MATRIX_VERSION if args.apply_bnt else "none",
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_float_or_none(value: float) -> Optional[float]:
    value_f = float(value)
    return value_f if np.isfinite(value_f) else None


def fit_and_apply_compressor_input_standardization(
    train_x: np.ndarray,
    val_x: np.ndarray,
    obs_x: np.ndarray,
    clip_value: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0).astype(np.float32)
    std = train_x.std(axis=0).astype(np.float32)
    std_safe = std.copy()
    std_safe[std_safe < 1e-12] = 1.0

    train_std = (train_x - mean) / std_safe
    val_std = (val_x - mean) / std_safe
    obs_std = (obs_x - mean) / std_safe

    if clip_value is not None and clip_value > 0:
        train_std = np.clip(train_std, -clip_value, clip_value)
        val_std = np.clip(val_std, -clip_value, clip_value)
        obs_std = np.clip(obs_std, -clip_value, clip_value)

    return (
        train_std.astype(np.float32),
        val_std.astype(np.float32),
        obs_std.astype(np.float32),
        mean,
        std_safe,
    )


def apply_compressor_input_standardization(
    train_x: np.ndarray,
    val_x: np.ndarray,
    obs_x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    clip_value: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_std = (train_x - mean) / std
    val_std = (val_x - mean) / std
    obs_std = (obs_x - mean) / std

    if clip_value is not None and clip_value > 0:
        train_std = np.clip(train_std, -clip_value, clip_value)
        val_std = np.clip(val_std, -clip_value, clip_value)
        obs_std = np.clip(obs_std, -clip_value, clip_value)

    return (
        train_std.astype(np.float32),
        val_std.astype(np.float32),
        obs_std.astype(np.float32),
    )


def build_compressed_cache_metadata(
    args: argparse.Namespace,
    tomo_bin_indices: tuple[int, ...],
    l1_min_snr: float,
    l1_max_snr: float,
    l1_clamp_overflow: bool,
    subtract_coarse_mean: bool,
    compressor_source: str,
    compressor_params_path: Optional[str],
    compressor_state_path: Optional[str],
) -> Dict[str, object]:
    params_path = Path(compressor_params_path).resolve() if compressor_params_path else None
    state_path = Path(compressor_state_path).resolve() if compressor_state_path else None

    return {
        "compressor_source": compressor_source,
        "compressor_dim": int(args.compressor_dim),
        "compressor_hidden": ",".join(str(v) for v in args.compressor_hidden_sizes),
        "compressor_params_path": str(params_path) if params_path else "",
        "compressor_state_path": str(state_path) if state_path else "",
        "compressor_params_sha256": (
            file_sha256(params_path) if params_path and params_path.exists() else ""
        ),
        "compressor_state_sha256": (
            file_sha256(state_path) if state_path and state_path.exists() else ""
        ),
        "tfds_name": str(args.tfds_name),
        "tomo_bin_indices": ",".join(str(b) for b in tomo_bin_indices),
        "map_kind": str(args.map_kind),
        "field_size": int(args.field_size),
        "field_npix": int(args.field_npix),
        "nside": int(args.nside),
        "nbins": int(args.nbins),
        "sigma_e": float(args.sigma_e),
        "galaxy_density": float(args.galaxy_density),
        "l1_min_snr": float(l1_min_snr),
        "l1_max_snr": float(l1_max_snr),
        "l1_nbins": int(args.l1_nbins),
        "l1_clamp_overflow": bool(l1_clamp_overflow),
        "subtract_coarse_mean": bool(subtract_coarse_mean),
        "l1_implementation": str(args.l1_implementation),
        "n_scales": int(args.n_scales),
        "apply_bnt": bool(args.apply_bnt),
        "bnt_matrix_version": BNT_MATRIX_VERSION if args.apply_bnt else "none",
        "standardize_summary": bool(args.standardize_summary),
        "summary_clip_value": float(args.summary_clip_value),
        "compressor_log1p_input": bool(args.compressor_log1p_input),
        "compressor_input_standardize": bool(args.compressor_input_standardize),
        "compressor_input_clip": float(args.compressor_input_clip),
    }


def compare_cache_metadata(
    meta_npz: np.lib.npyio.NpzFile,
    expected: Dict[str, object],
) -> Tuple[bool, list[str]]:
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        if key not in meta_npz.files:
            mismatches.append(f"missing:{key}")
            continue
        cached_raw = meta_npz[key]
        if isinstance(expected_value, bool):
            cached_value = bool(cached_raw)
            if cached_value != expected_value:
                mismatches.append(f"{key}={cached_value} (expected {expected_value})")
        elif isinstance(expected_value, float):
            cached_value = float(cached_raw)
            if abs(cached_value - expected_value) > 1e-10:
                mismatches.append(f"{key}={cached_value} (expected {expected_value})")
        elif isinstance(expected_value, int):
            cached_value = int(cached_raw)
            if cached_value != expected_value:
                mismatches.append(f"{key}={cached_value} (expected {expected_value})")
        else:
            cached_value = str(cached_raw)
            if cached_value != str(expected_value):
                mismatches.append(f"{key}={cached_value} (expected {expected_value})")
    return len(mismatches) == 0, mismatches


class CompressorMLP(hk.Module):
    def __init__(self, output_dim: int, hidden_sizes: tuple[int, ...], name: str | None = None):
        super().__init__(name=name)
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes

    def __call__(self, x: jax.Array) -> jax.Array:
        net = x
        for width in self.hidden_sizes:
            net = hk.Linear(width)(net)
            net = jax.nn.leaky_relu(net)
        net = hk.Linear(self.output_dim)(net)
        return net


def build_compressor(output_dim: int, hidden_sizes: tuple[int, ...]):
    return hk.transform_with_state(lambda y: CompressorMLP(output_dim, hidden_sizes)(y))


def load_compressor_params(params_path: str, state_path: str):
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    with open(state_path, "rb") as f:
        state = pickle.load(f)
    return params, state


def train_compressor_vmim(
    compressor,
    train_theta: np.ndarray,
    train_x: np.ndarray,
    val_theta: np.ndarray,
    val_x: np.ndarray,
    n_cosmo: int,
    compressor_dim: int,
    hidden_sizes: tuple[int, ...],
    total_steps: int,
    lr_init: float,
    batch_size: int,
    save_every: int,
    save_dir: Path,
    seed: int,
    abort_on_nonfinite: bool = False,
    max_nonfinite_events: int = 20,
    vmim_nf_layers: int = 4,
    vmim_nf_hidden: int = 128,
) -> tuple[hk.Params, hk.State, Path, Path]:
    print("######## TRAINING L1 VMIM COMPRESSOR ########")
    save_dir.mkdir(parents=True, exist_ok=True)

    bijector_fn = partial(
        AffineCoupling,
        layers=[int(vmim_nf_hidden), int(vmim_nf_hidden)],
        activation=jax.nn.silu,
    )
    nf_factory = partial(
        ConditionalRealNVP,
        n_layers=int(vmim_nf_layers),
        bijector_fn=bijector_fn,
    )

    class FlowNdCompressor(hk.Module):
        def __call__(self, y):
            return nf_factory(n_cosmo)(y)

    nf = hk.without_apply_rng(
        hk.transform(lambda theta, y: FlowNdCompressor()(y).log_prob(theta).squeeze())
    )

    key0 = jax.random.PRNGKey(seed)
    params_c, state_c = compressor.init(
        key0,
        y=jnp.zeros((1, train_x.shape[1]), dtype=jnp.float32),
    )
    params_nf = nf.init(
        key0,
        theta=jnp.zeros((1, n_cosmo), dtype=jnp.float32),
        y=jnp.zeros((1, compressor_dim), dtype=jnp.float32),
    )
    params_merged = hk.data_structures.merge(params_c, params_nf)

    n_params = sum(x.size for x in jax.tree.leaves(params_merged))
    print(f"  Compressor+NF parameters: {n_params:,}")

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
        compressor=compressor,
        nf=nf,
        optimizer=optimizer,
        loss_name="train_compressor_vmim",
    )
    update = jax.jit(model.update)

    train_theta_j = jnp.asarray(train_theta, dtype=jnp.float32)
    train_x_j = jnp.asarray(train_x, dtype=jnp.float32)
    val_theta_j = jnp.asarray(val_theta, dtype=jnp.float32)
    val_x_j = jnp.asarray(val_x, dtype=jnp.float32)

    n_train = int(train_theta.shape[0])
    n_val = int(val_theta.shape[0])

    best_val = float("inf")
    best_params = params_merged
    best_state = state_c
    best_step = 0

    train_hist: list[float] = []
    val_hist: list[float] = []
    val_steps: list[int] = []
    final_step = 0
    nonfinite_train_events = 0
    nonfinite_val_events = 0
    warning_flags: list[str] = []
    caught_error: Optional[BaseException] = None

    try:
        for step in range(1, total_steps + 1):
            final_step = step
            idx = np.random.randint(0, n_train, size=batch_size)
            b_loss, params_merged, opt_state, state_c = update(
                model_params=params_merged,
                opt_state=opt_state,
                theta=train_theta_j[idx],
                x=train_x_j[idx],
                state_resnet=state_c,
            )
            loss_f = float(b_loss)
            if not np.isfinite(loss_f):
                nonfinite_train_events += 1
                warning_flags.append(f"nonfinite_train_loss@step{step}")
                print(f"  [warn] Non-finite compressor train loss at step {step}: {loss_f}")
                wandb.log(
                    {
                        "compressor/nonfinite_event": 1,
                        "compressor/nonfinite_train_event": 1,
                        "compressor/nonfinite_train_events_total": int(nonfinite_train_events),
                        "compressor/nonfinite_events_total": int(
                            nonfinite_train_events + nonfinite_val_events
                        ),
                        "compressor/step": step,
                    },
                )
                if abort_on_nonfinite:
                    warning_flags.append("abort_on_nonfinite_triggered")
                    raise FloatingPointError(
                        f"Non-finite compressor train loss at step {step}: {loss_f}."
                    )
                if (nonfinite_train_events + nonfinite_val_events) > max_nonfinite_events:
                    warning_flags.append("max_nonfinite_exceeded")
                    raise FloatingPointError(
                        "Too many non-finite compressor events: "
                        f"{nonfinite_train_events + nonfinite_val_events} > {max_nonfinite_events}."
                    )
                continue
            train_hist.append(loss_f)

            if step % 100 == 0:
                wandb.log(
                    {
                        "compressor/train_loss": loss_f,
                        "compressor/step": step,
                        "compressor/lr": float(lr_schedule(step)),
                        "compressor/nonfinite_train_events_total": int(nonfinite_train_events),
                        "compressor/nonfinite_val_events_total": int(nonfinite_val_events),
                        "compressor/nonfinite_events_total": int(
                            nonfinite_train_events + nonfinite_val_events
                        ),
                        "compressor/max_nonfinite_events": int(max_nonfinite_events),
                    },
                )
                print(f"  Compressor step {step:6d} | train loss {loss_f:.4f}")

            if step % save_every == 0 or step == total_steps:
                vidx = np.random.randint(0, n_val, size=min(batch_size, n_val))
                val_l_raw, _, _, _ = update(
                    model_params=params_merged,
                    opt_state=opt_state,
                    theta=val_theta_j[vidx],
                    x=val_x_j[vidx],
                    state_resnet=state_c,
                )
                val_l = float(val_l_raw)
                if not np.isfinite(val_l):
                    nonfinite_val_events += 1
                    warning_flags.append(f"nonfinite_val_loss@step{step}")
                    print(
                        f"  [warn] Non-finite compressor val loss at step {step}; "
                        "skipping best-checkpoint update for this step."
                    )
                    wandb.log(
                        {
                            "compressor/nonfinite_event": 1,
                            "compressor/nonfinite_val_event": 1,
                            "compressor/nonfinite_val_events_total": int(nonfinite_val_events),
                            "compressor/nonfinite_events_total": int(
                                nonfinite_train_events + nonfinite_val_events
                            ),
                            "compressor/step": step,
                        },
                    )
                    if abort_on_nonfinite:
                        warning_flags.append("abort_on_nonfinite_triggered")
                        raise FloatingPointError(
                            f"Non-finite compressor val loss at step {step}: {val_l}."
                        )
                    if (nonfinite_train_events + nonfinite_val_events) > max_nonfinite_events:
                        warning_flags.append("max_nonfinite_exceeded")
                        raise FloatingPointError(
                            "Too many non-finite compressor events: "
                            f"{nonfinite_train_events + nonfinite_val_events} > {max_nonfinite_events}."
                        )
                    val_l = float("inf")

                val_hist.append(val_l)
                val_steps.append(step)

                ckpt_params = save_dir / f"params_nd_compressor_batch{step}.pkl"
                ckpt_state = save_dir / f"opt_state_resnet_batch{step}.pkl"
                with open(ckpt_params, "wb") as f:
                    pickle.dump(params_merged, f)
                with open(ckpt_state, "wb") as f:
                    pickle.dump(state_c, f)

                improved = ""
                if val_l < best_val:
                    best_val = val_l
                    best_step = step
                    best_params = params_merged
                    best_state = state_c
                    with open(save_dir / "params_nd_compressor_best.pkl", "wb") as f:
                        pickle.dump(best_params, f)
                    with open(save_dir / "opt_state_resnet_best.pkl", "wb") as f:
                        pickle.dump(best_state, f)
                    improved = " ***"

                wandb.log(
                    {
                        "compressor/val_loss": val_l,
                        "compressor/best_val_loss": best_val,
                        "compressor/best_step": int(best_step),
                        "compressor/step": step,
                        "compressor/nonfinite_train_events_total": int(nonfinite_train_events),
                        "compressor/nonfinite_val_events_total": int(nonfinite_val_events),
                        "compressor/nonfinite_events_total": int(
                            nonfinite_train_events + nonfinite_val_events
                        ),
                        "compressor/max_nonfinite_events": int(max_nonfinite_events),
                    },
                )
                print(
                    f"  Compressor save @ {step} | val loss {val_l:.4f}{improved} "
                    f"(best={best_val:.4f})"
                )
    except FloatingPointError as exc:
        caught_error = exc
        print(f"  ERROR: {exc}")
    finally:
        np.save(save_dir / "loss_compressor_train.npy", np.array(train_hist))
        np.save(save_dir / "loss_compressor_val.npy", np.array(val_hist))
        np.save(save_dir / "loss_compressor_val_steps.npy", np.array(val_steps))

        repeated_nonfinite = (nonfinite_train_events + nonfinite_val_events) >= 2
        if best_step == final_step and final_step > 0:
            warning_flags.append("best_at_final_step")
            print(
                "  WARNING: Best compressor validation loss occurred at final step. "
                "Compressor may be underconverged."
            )
        if repeated_nonfinite:
            warning_flags.append("repeated_nonfinite_events")
            print(
                "  WARNING: Repeated non-finite compressor events detected. "
                "Inspect data/model stability."
            )
        if caught_error is not None:
            warning_flags.append("aborted_nonfinite")

        summary = {
            "best_val": finite_float_or_none(best_val),
            "best_val_loss": finite_float_or_none(best_val),
            "best_step": int(best_step),
            "final_step": int(final_step),
            "best_at_final_step": bool(best_step == final_step and final_step > 0),
            "total_steps_requested": int(total_steps),
            "save_every": int(save_every),
        "compressor_dim": int(compressor_dim),
        "hidden_sizes": [int(v) for v in hidden_sizes],
        "vmim_nf_layers": int(vmim_nf_layers),
        "vmim_nf_hidden": int(vmim_nf_hidden),
        "nonfinite_train_events": int(nonfinite_train_events),
            "nonfinite_val_events": int(nonfinite_val_events),
            "nonfinite_events_total": int(nonfinite_train_events + nonfinite_val_events),
            "max_nonfinite_events": int(max_nonfinite_events),
            "abort_on_nonfinite": bool(abort_on_nonfinite),
            "warning_flags": sorted(set(warning_flags)),
            "aborted": bool(caught_error is not None),
            "abort_reason": str(caught_error) if caught_error is not None else None,
        }
        (save_dir / "compressor_training_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        if wandb.run is not None:
            wandb.run.summary["compressor_best_val_loss"] = summary["best_val_loss"]
            wandb.run.summary["compressor_best_step"] = int(best_step)
            wandb.run.summary["compressor_final_step"] = int(final_step)
            wandb.run.summary["compressor_nonfinite_events_total"] = int(
                nonfinite_train_events + nonfinite_val_events
            )
            wandb.run.summary["compressor_warning_flags"] = ",".join(
                summary["warning_flags"]
            )
            wandb.run.summary["compressor_aborted"] = bool(summary["aborted"])

    best_params_path = save_dir / "params_nd_compressor_best.pkl"
    best_state_path = save_dir / "opt_state_resnet_best.pkl"
    best_val_msg = (
        f"{best_val:.4f}" if np.isfinite(best_val) else "non-finite/unavailable"
    )
    print(
        f"  Compressor training done. Best step={best_step}, best val={best_val_msg}"
    )
    if caught_error is not None:
        raise caught_error
    return best_params, best_state, best_params_path, best_state_path


def compress_features(
    x: np.ndarray,
    compressor,
    comp_params,
    comp_state,
    batch_size: int,
) -> np.ndarray:
    outs: list[np.ndarray] = []
    n = x.shape[0]
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        y, _ = compressor.apply(
            comp_params,
            comp_state,
            None,
            x[i:j].astype(np.float32),
        )
        outs.append(np.asarray(y, dtype=np.float32))
    return np.concatenate(outs, axis=0)


def apply_summary_standardization(
    train_x: np.ndarray,
    val_x: np.ndarray,
    obs_x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    clip_value: Optional[float] = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-8] = 1.0
    train_std, val_std, obs_std = apply_summary_standardization(
        train_x, val_x, obs_x, mean, std, clip_value=clip_value,
    )
    return train_std, val_std, obs_std, mean, std


def collect_compressed_summary_health_diagnostics(
    obs_summary: np.ndarray,
    train_x: np.ndarray,
    val_x: np.ndarray,
) -> Dict[str, float]:
    train_std = train_x.std(axis=0)
    p01 = np.percentile(train_x, 1.0, axis=0)
    p99 = np.percentile(train_x, 99.0, axis=0)
    obs_inlier_frac = np.mean((obs_summary >= p01) & (obs_summary <= p99))
    train_mean = train_x.mean(axis=0)
    val_mean = val_x.mean(axis=0)
    val_shift = np.abs(val_mean - train_mean) / (train_std + 1e-8)
    return {
        "summary_std_min": float(train_std.min()),
        "summary_std_median": float(np.median(train_std)),
        "summary_std_max": float(train_std.max()),
        "summary_dead_feature_frac": float(np.mean(train_std < 1e-8)),
        "obs_in_train_p01_p99_frac": float(obs_inlier_frac),
        "val_train_mean_shift_median_sigma": float(np.median(val_shift)),
        "val_train_mean_shift_max_sigma": float(np.max(val_shift)),
    }


def build_flow(n_cosmo_params: int, n_layers: int, hidden: int):
    bijector_fn = partial(
        AffineCoupling,
        layers=[hidden] * 2,
        activation=jax.nn.silu,
    )
    nf_factory = partial(
        ConditionalRealNVP,
        n_layers=n_layers,
        bijector_fn=bijector_fn,
    )

    class NF(hk.Module):
        def __call__(self, y):
            return nf_factory(n_cosmo_params)(y)

    @hk.transform
    def nf_log_prob(theta, y):
        return NF()(y).log_prob(theta).squeeze()

    @hk.transform
    def nf_sample(y, n_samples):
        return NF()(y).sample(n_samples, seed=hk.next_rng_key())

    nf_logp = hk.without_apply_rng(nf_log_prob)
    return nf_logp, nf_sample


def make_update_fn(nf_logp, optimizer):
    def loss_fn(params, theta_batch, y_batch):
        return -jnp.mean(nf_logp.apply(params, theta_batch, y_batch))

    @jax.jit
    def update(params, opt_state, theta_batch, y_batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, theta_batch, y_batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_opt_state

    return update


def validate_flow_inputs(
    dataset_train: Dict[str, np.ndarray],
    dataset_val: Dict[str, np.ndarray],
    obs_summary: np.ndarray,
    n_cosmo: int,
) -> None:
    for split_name, dataset in (("train", dataset_train), ("val", dataset_val)):
        if "theta" not in dataset or "x" not in dataset:
            raise ValueError(f"{split_name} dataset must contain 'theta' and 'x'.")
        theta = np.asarray(dataset["theta"])
        x = np.asarray(dataset["x"])
        if theta.ndim != 2 or x.ndim != 2:
            raise ValueError(
                f"{split_name} arrays must be 2D; got theta={theta.shape}, x={x.shape}."
            )
        if theta.shape[0] != x.shape[0]:
            raise ValueError(
                f"{split_name} sample mismatch: theta={theta.shape}, x={x.shape}."
            )
        if theta.shape[1] != n_cosmo:
            raise ValueError(
                f"{split_name} theta second dimension must be {n_cosmo}, got {theta.shape[1]}."
            )
        if x.shape[1] <= 0:
            raise ValueError(f"{split_name} summary dimension must be positive.")
        if not np.isfinite(theta).all() or not np.isfinite(x).all():
            raise ValueError(f"{split_name} contains non-finite values.")

    train_dim = int(np.asarray(dataset_train["x"]).shape[1])
    val_dim = int(np.asarray(dataset_val["x"]).shape[1])
    if train_dim != val_dim:
        raise ValueError(f"Train/val summary dim mismatch: {train_dim} vs {val_dim}.")

    obs_summary = np.asarray(obs_summary)
    if obs_summary.ndim != 1:
        raise ValueError(f"Observed summary must be 1D, got shape {obs_summary.shape}.")
    if obs_summary.shape[0] != train_dim:
        raise ValueError(
            f"Observed summary dim mismatch: obs={obs_summary.shape[0]} vs train={train_dim}."
        )
    if not np.isfinite(obs_summary).all():
        raise ValueError("Observed summary contains non-finite values.")


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
    print("######## TRAINING FLOW ########")
    key_init, _ = jax.random.split(rng_key)

    theta_dummy = 0.5 * jnp.zeros([1, n_cosmo])
    y_dummy = jnp.zeros([1, summary_dim])
    params = nf_logp.init(key_init, theta_dummy, y_dummy)

    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"  Flow parameters: {n_params:,}")

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

    best_val_loss = float("inf")
    best_step = 0
    best_params = params
    patience_counter = 0
    val_batch_size = min(512, n_val)
    final_step = 0
    nonfinite_train_events = 0
    nonfinite_val_events = 0
    warning_flags: list[str] = []
    caught_error: Optional[BaseException] = None

    try:
        for step in range(1, total_steps + 1):
            final_step = step
            idx = np.random.randint(0, n_train, batch_size)
            loss, params, opt_state = update(params, opt_state, theta_train[idx], x_train[idx])
            loss_f = float(loss)
            if not np.isfinite(loss_f):
                nonfinite_train_events += 1
                warning_flags.append(f"nonfinite_train_loss@step{step}")
                wandb.log(
                    {
                        "flow/nonfinite_event": 1,
                        "flow/nonfinite_train_event": 1,
                        "flow/nonfinite_train_events_total": int(nonfinite_train_events),
                        "flow/nonfinite_events_total": int(
                            nonfinite_train_events + nonfinite_val_events
                        ),
                        "flow/step": step,
                    },
                )
                raise FloatingPointError(
                    f"Non-finite train loss at step {step}: {loss_f}."
                )
            batch_losses.append(loss_f)

            if step % 100 == 0:
                lr_value = (
                    float(lr_schedule_fn(step))
                    if lr_schedule_fn is not None
                    else float(lr_schedule(step))
                )
                log_dict = {
                    "flow/train_loss": loss_f,
                    "flow/step": step,
                    "flow/lr": lr_value,
                    "flow/nonfinite_train_events_total": int(nonfinite_train_events),
                    "flow/nonfinite_val_events_total": int(nonfinite_val_events),
                    "flow/nonfinite_events_total": int(
                        nonfinite_train_events + nonfinite_val_events
                    ),
                    "train/loss": loss_f,
                    "train/lr": lr_value,
                }
                if np.isfinite(best_val_loss):
                    log_dict["flow/best_val_loss"] = float(best_val_loss)
                    log_dict["train/best_val_loss_so_far"] = float(best_val_loss)
                wandb.log(log_dict)
                print(f"  Step {step:6d} | train loss {loss_f:.4f}")

            if step % save_every == 0 or step == total_steps:
                save_dir.mkdir(parents=True, exist_ok=True)

                vidx = np.random.randint(0, n_val, val_batch_size)
                val_l = float(-jnp.mean(nf_logp.apply(params, theta_val[vidx], x_val[vidx])))
                if not np.isfinite(val_l):
                    nonfinite_val_events += 1
                    warning_flags.append(f"nonfinite_val_loss@step{step}")
                    wandb.log(
                        {
                            "flow/nonfinite_event": 1,
                            "flow/nonfinite_val_event": 1,
                            "flow/nonfinite_val_events_total": int(nonfinite_val_events),
                            "flow/nonfinite_events_total": int(
                                nonfinite_train_events + nonfinite_val_events
                            ),
                            "flow/step": step,
                        },
                    )
                    raise FloatingPointError(
                        f"Non-finite validation loss at step {step}: {val_l}."
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
                    with open(save_dir / "params_l1vmim_flow_best.pkl", "wb") as f:
                        pickle.dump(params, f)
                else:
                    patience_counter += 1

                with open(save_dir / f"params_l1vmim_flow_batch{step}.pkl", "wb") as f:
                    pickle.dump(params, f)

                lr_value = (
                    float(lr_schedule_fn(step))
                    if lr_schedule_fn is not None
                    else float(lr_schedule(step))
                )
                wandb.log(
                    {
                        "flow/val_loss": val_l,
                        "flow/best_val_loss": float(best_val_loss),
                        "flow/patience_counter": int(patience_counter),
                        "flow/patience": int(patience),
                        "flow/step": step,
                        "flow/lr": lr_value,
                        "flow/nonfinite_train_events_total": int(nonfinite_train_events),
                        "flow/nonfinite_val_events_total": int(nonfinite_val_events),
                        "flow/nonfinite_events_total": int(
                            nonfinite_train_events + nonfinite_val_events
                        ),
                        "val/loss": val_l,
                        "val/best_loss": float(best_val_loss),
                        "val/patience_counter": int(patience_counter),
                    },
                )
                print(
                    f"  Saved @ step {step}. Val loss = {val_l:.4f}{improved}"
                    f"  (best = {best_val_loss:.4f}, patience = {patience_counter})"
                )

                if patience > 0 and patience_counter >= patience:
                    print(
                        f"  Early stopping at step {step} "
                        f"(no val improvement for {patience} checks)"
                    )
                    break
    except FloatingPointError as exc:
        caught_error = exc
        print(f"  ERROR: {exc}")
    finally:
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "loss_train_l1vmim.npy", np.array(batch_losses))
        np.save(save_dir / "loss_val_l1vmim.npy", np.array(val_losses))
        np.save(save_dir / "loss_val_steps.npy", np.array(val_steps))

        total_nonfinite = nonfinite_train_events + nonfinite_val_events
        if best_step == final_step and final_step > 0:
            warning_flags.append("best_at_final_step")
            print(
                "  WARNING: Best val loss occurred at final step. "
                "Flow may be underconverged; consider increasing --total-steps "
                "and/or reducing --save-every."
            )
        if total_nonfinite >= 2:
            warning_flags.append("repeated_nonfinite_events")
            print(
                "  WARNING: Repeated non-finite flow events detected. "
                "Inspect training stability."
            )
        if caught_error is not None:
            warning_flags.append("aborted_nonfinite")

        summary = {
            "best_val": finite_float_or_none(best_val_loss),
            "best_val_loss": finite_float_or_none(best_val_loss),
            "best_step": int(best_step),
            "final_step": int(final_step),
            "best_at_final_step": bool(best_step == final_step and final_step > 0),
            "total_steps_requested": int(total_steps),
            "save_every": int(save_every),
            "patience": int(patience),
            "n_val_checks": int(len(val_losses)),
            "nonfinite_train_events": int(nonfinite_train_events),
            "nonfinite_val_events": int(nonfinite_val_events),
            "nonfinite_events_total": int(total_nonfinite),
            "warning_flags": sorted(set(warning_flags)),
            "aborted": bool(caught_error is not None),
            "abort_reason": str(caught_error) if caught_error is not None else None,
        }
        (save_dir / "flow_training_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        if wandb.run is not None:
            wandb.run.summary["best_val_loss"] = summary["best_val_loss"]
            wandb.run.summary["best_val_step"] = int(best_step)
            wandb.run.summary["final_step"] = int(final_step)
            wandb.run.summary["flow_nonfinite_events_total"] = int(total_nonfinite)
            wandb.run.summary["flow_warning_flags"] = ",".join(summary["warning_flags"])
            wandb.run.summary["flow_aborted"] = bool(summary["aborted"])

    best_val_msg = (
        f"{best_val_loss:.4f}" if np.isfinite(best_val_loss) else "non-finite/unavailable"
    )
    print(f"  Best validation loss: {best_val_msg}")
    if caught_error is not None:
        raise caught_error
    return best_params


def sample_posterior(
    rng_key: jax.Array,
    nf_sample,
    flow_params: hk.Params,
    summary_obs: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    print("######## SAMPLING POSTERIOR ########")
    summary_dim = summary_obs.shape[-1]
    y_obs = jnp.asarray(summary_obs).reshape(1, summary_dim)
    y_cond = jnp.repeat(y_obs, repeats=n_samples, axis=0)
    samples = nf_sample.apply(flow_params, rng_key, y_cond, n_samples)

    nan_rows = jnp.any(jnp.isnan(samples), axis=-1)
    samples = samples[~nan_rows]
    if len(samples) == 0:
        raise FloatingPointError("All posterior samples are non-finite (NaN).")
    print(f"  Generated {len(samples)} valid posterior samples.")
    return np.array(samples)


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
        [mcsamples],
        filled=True,
        markers=truth,
        marker_args={"color": "red", "lw": 1.2},
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved posterior plot -> {output_path}")

    if log_to_wandb and wandb.run is not None:
        wandb.log({"posterior/triangle_plot": wandb.Image(output_path)})

    plt.close()


def main() -> None:
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
    n_cross_pairs = (args.nbins * (args.nbins - 1)) // 2 if getattr(args, "cross_maps", False) else 0
    n_l1_channels = args.nbins + n_cross_pairs
    raw_summary_dim = args.n_scales * args.l1_nbins * n_l1_channels
    print(f"  pixel_arcmin   = {pixel_arcmin:.2f}")
    print(f"  noise_sigma    = {noise_sigma:.6f}")
    print(
        f"  raw_summary    = {raw_summary_dim} "
        f"({args.n_scales} scales x {args.l1_nbins} bins x {n_l1_channels} channels)"
    )
    print(f"  compressor_dim = {args.compressor_dim}")

    param_names = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]
    save_path = Path(args.save_dir) / "l1_vmim" / args.map_kind
    flow_summary_path = save_path / "flow_training_summary.json"
    compressor_summary_path: Optional[Path] = None
    summary_health_path = save_path / "compressed_summary_health.json"
    summary_stats_path = save_path / "l1_vmim_summary_standardization.npz"
    compressor_input_stats_path = save_path / "l1_vmim_compressor_input_standardization.npz"
    compressor_input_clip_value = (
        args.compressor_input_clip if args.compressor_input_clip > 0 else None
    )

    base_tags = [
        args.map_kind,
        "l1-vmim",
        f"nvp{args.nvp_layers}",
        f"cdim{args.compressor_dim}",
        f"std{int(args.standardize_summary)}",
        f"log1p{int(args.compressor_log1p_input)}",
        f"l1impl-{args.l1_implementation}",
        f"bnt{int(args.apply_bnt)}",
    ]
    user_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    wandb_tags = list(dict.fromkeys(base_tags + user_tags))
    wandb_group = (
        args.wandb_group
        if args.wandb_group
        else f"l1-vmim-{args.map_kind}-bins{args.nbins}-cdim{args.compressor_dim}"
    )
    run_variant = (
        f"map={args.map_kind}|impl={args.l1_implementation}|nbins={args.nbins}"
        f"|cdim={args.compressor_dim}|std={int(args.standardize_summary)}"
        f"|log1p={int(args.compressor_log1p_input)}"
        f"|bnt={int(args.apply_bnt)}"
    )

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
            tags=wandb_tags,
            group=wandb_group,
        )
        wandb.define_metric("compressor/step")
        wandb.define_metric("compressor/*", step_metric="compressor/step")
        wandb.define_metric("flow/step")
        wandb.define_metric("flow/*", step_metric="flow/step")
        wandb.define_metric("train/*", step_metric="flow/step")
        wandb.define_metric("val/*", step_metric="flow/step")
        wandb.config.update(
            {
                "run/method": "l1_vmim",
                "run/variant": run_variant,
                "run/tomo_bin_indices": ",".join(str(b) for b in tomo_bin_indices),
                "run/wandb_group": wandb_group,
                "run/wandb_tags": wandb_tags,
            },
            allow_val_change=True,
        )
    else:
        wandb.init(mode="disabled")

    # 1) Observed map
    m_data, _, truth = load_observed_map(
        args.cosmogrid_meta,
        args.fiducial_map,
        args.field_size,
        args.field_npix,
        args.nside,
        args.nbins,
        tomo_bin_indices,
        args.sigma_e,
        args.galaxy_density,
        rng_obs,
        apply_bnt=args.apply_bnt,
    )

    # 2) L1 computer + SNR policy
    stats = build_l1_computer(
        args.n_scales,
        pixel_arcmin,
        torch_device,
        l1_implementation=args.l1_implementation,
    )
    effective_l1_clamp = args.l1_clamp_overflow
    effective_subtract_coarse_mean = args.subtract_coarse_mean
    if args.l1_implementation == "cosmoford":
        if args.l1_clamp_overflow:
            print("  CosmOrford mode forces clamp_overflow=False.")
        if args.subtract_coarse_mean:
            print(
                "  CosmOrford mode uses WLStatistics default coarse-mean handling "
                "(ignoring --subtract-coarse-mean)."
            )
        effective_l1_clamp = False
        effective_subtract_coarse_mean = False

    augmentation = build_augmentation(
        args.map_kind,
        args.sigma_e,
        args.galaxy_density,
        args.field_size,
        args.field_npix,
        args.nbins,
        tomo_bin_indices,
        apply_bnt=args.apply_bnt,
    )

    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    if not args.auto_calibrate_snr:
        l1_min_snr = args.l1_min_snr
        l1_max_snr = args.l1_max_snr
        print(f"  Using fixed SNR range: [{l1_min_snr}, {l1_max_snr}]")
    else:
        calib_cache = cache_dir / "snr_calibration.npz" if cache_dir else None
        if calib_cache is not None and calib_cache.exists():
            calib = np.load(calib_cache)
            l1_min_snr = float(calib["min_snr"])
            l1_max_snr = float(calib["max_snr"])
            print(f"  Loaded cached SNR range: [{l1_min_snr:.4f}, {l1_max_snr:.4f}]")
        else:
            l1_min_snr, l1_max_snr = calibrate_snr_range(
                stats,
                augmentation,
                args.tfds_name,
                noise_sigma,
                args.nbins,
                l1_implementation=args.l1_implementation,
                n_calibration=args.calibration_samples,
                ds_batch_size=args.ds_batch_size,
                subtract_coarse_mean=effective_subtract_coarse_mean,
                margin=args.calibration_margin,
            )
            if calib_cache is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)
                np.savez(calib_cache, min_snr=l1_min_snr, max_snr=l1_max_snr)
                print(f"  Cached SNR calibration to {calib_cache}")

    cache_meta_expected = build_l1_cache_metadata(
        args=args,
        tomo_bin_indices=tomo_bin_indices,
        l1_min_snr=l1_min_snr,
        l1_max_snr=l1_max_snr,
        l1_clamp_overflow=effective_l1_clamp,
        subtract_coarse_mean=effective_subtract_coarse_mean,
    )

    print("######## L1-NORM: OBSERVED MAP ########")
    obs_l1 = compute_l1_single_map(
        m_data,
        noise_sigma,
        stats,
        args.l1_nbins,
        args.nbins,
        l1_min_snr=l1_min_snr,
        l1_max_snr=l1_max_snr,
        clamp_overflow=effective_l1_clamp,
        subtract_coarse_mean=effective_subtract_coarse_mean,
        l1_implementation=args.l1_implementation,
    )
    print(f"  Observed L1 shape = {obs_l1.shape}")

    print("######## L1-NORM: DATASETS ########")
    cache_ok = False
    if cache_dir is not None and cache_dir.exists():
        train_cache = cache_dir / "l1_train.npz"
        val_cache = cache_dir / "l1_val.npz"
        meta_cache = cache_dir / "l1_cache_meta.npz"
        if train_cache.exists() and val_cache.exists() and meta_cache.exists():
            meta = np.load(meta_cache)
            cache_ok, mismatches = compare_cache_metadata(meta, cache_meta_expected)
            if cache_ok:
                d_tr = np.load(train_cache)
                dataset_train = {"theta": d_tr["theta"], "x": d_tr["x"]}
                d_va = np.load(val_cache)
                dataset_val = {"theta": d_va["theta"], "x": d_va["x"]}
                print("  Loaded cached L1 train/val datasets (metadata matches).")
            else:
                first_mismatch = mismatches[0] if mismatches else "unknown mismatch"
                print(
                    "  Cache metadata mismatch. Recomputing L1 train/val datasets. "
                    f"First mismatch: {first_mismatch}"
                )

    if not cache_ok:
        dataset_train = compute_l1_dataset(
            args.tfds_name,
            "train",
            augmentation,
            stats,
            noise_sigma,
            args.l1_nbins,
            args.nbins,
            args.ds_batch_size,
            l1_min_snr=l1_min_snr,
            l1_max_snr=l1_max_snr,
            clamp_overflow=effective_l1_clamp,
            subtract_coarse_mean=effective_subtract_coarse_mean,
            l1_implementation=args.l1_implementation,
        )
        dataset_val = compute_l1_dataset(
            args.tfds_name,
            "test",
            augmentation,
            stats,
            noise_sigma,
            args.l1_nbins,
            args.nbins,
            args.ds_batch_size,
            l1_min_snr=l1_min_snr,
            l1_max_snr=l1_max_snr,
            clamp_overflow=effective_l1_clamp,
            subtract_coarse_mean=effective_subtract_coarse_mean,
            l1_implementation=args.l1_implementation,
        )
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(cache_dir / "l1_train.npz", theta=dataset_train["theta"], x=dataset_train["x"])
            np.savez(cache_dir / "l1_val.npz", theta=dataset_val["theta"], x=dataset_val["x"])
            np.savez(cache_dir / "l1_cache_meta.npz", **cache_meta_expected)
            print(f"  Cached raw L1 datasets to {cache_dir}")

    dataset_train["theta"] = np.asarray(dataset_train["theta"], dtype=np.float32)
    dataset_val["theta"] = np.asarray(dataset_val["theta"], dtype=np.float32)
    dataset_train["x"] = np.asarray(dataset_train["x"], dtype=np.float32)
    dataset_val["x"] = np.asarray(dataset_val["x"], dtype=np.float32)
    obs_l1 = np.asarray(obs_l1, dtype=np.float32)

    if args.compressor_log1p_input:
        if np.any(dataset_train["x"] < -1.0) or np.any(dataset_val["x"] < -1.0) or np.any(obs_l1 < -1.0):
            raise ValueError(
                "--compressor-log1p-input requires all raw L1 values to be >= -1."
            )
        dataset_train["x"] = np.log1p(dataset_train["x"]).astype(np.float32)
        dataset_val["x"] = np.log1p(dataset_val["x"]).astype(np.float32)
        obs_l1 = np.log1p(obs_l1).astype(np.float32)
        print("  Applied log1p transform to raw L1 vectors before VMIM compression.")

    compressor_input_standardized = False
    compressor_input_mean: Optional[np.ndarray] = None
    compressor_input_std: Optional[np.ndarray] = None

    if args.compressor_input_standardize:
        if args.no_train:
            if not compressor_input_stats_path.exists():
                raise FileNotFoundError(
                    "--no-train with --compressor-input-standardize requires saved stats at "
                    f"{compressor_input_stats_path}."
                )
            comp_std_data = np.load(compressor_input_stats_path)
            compressor_input_mean = np.array(comp_std_data["mean"], dtype=np.float32)
            compressor_input_std = np.array(comp_std_data["std"], dtype=np.float32)
            if "clip_value" in comp_std_data.files:
                loaded_clip = float(comp_std_data["clip_value"])
                if np.isfinite(loaded_clip) and loaded_clip > 0:
                    compressor_input_clip_value = loaded_clip
                else:
                    compressor_input_clip_value = None
            (
                dataset_train["x"],
                dataset_val["x"],
                obs_l1,
            ) = apply_compressor_input_standardization(
                dataset_train["x"],
                dataset_val["x"],
                obs_l1,
                compressor_input_mean,
                compressor_input_std,
                clip_value=compressor_input_clip_value,
            )
            compressor_input_standardized = True
            print(
                "  Loaded compressor-input standardization stats from "
                f"{compressor_input_stats_path}"
            )
        else:
            (
                dataset_train["x"],
                dataset_val["x"],
                obs_l1,
                compressor_input_mean,
                compressor_input_std,
            ) = fit_and_apply_compressor_input_standardization(
                dataset_train["x"],
                dataset_val["x"],
                obs_l1,
                clip_value=compressor_input_clip_value,
            )
            compressor_input_standardized = True
            print(
                "  Applied compressor-input standardization "
                f"(clip={compressor_input_clip_value if compressor_input_clip_value is not None else 'off'})."
            )

    print(f"  Train x shape = {dataset_train['x'].shape}")
    print(f"  Val   x shape = {dataset_val['x'].shape}")

    # 3) VMIM compressor
    compressor = build_compressor(args.compressor_dim, args.compressor_hidden_sizes)
    compressor_source = "train_compressor" if args.train_compressor else "pretrained"

    compressor_params_ref: Optional[str] = args.compressor_params
    compressor_state_ref: Optional[str] = args.compressor_state

    if args.train_compressor:
        comp_save_dir = (
            Path(args.save_dir)
            / "vmim_l1"
            / args.map_kind
            / f"sigma_{args.sigma_e}"
            / f"gal_density_{int(args.galaxy_density * 4)}"
            / f"bin_{args.nbins}"
        )
        comp_params, comp_state, best_params_path, best_state_path = train_compressor_vmim(
            compressor=compressor,
            train_theta=dataset_train["theta"],
            train_x=dataset_train["x"],
            val_theta=dataset_val["theta"],
            val_x=dataset_val["x"],
            n_cosmo=args.n_cosmo,
            compressor_dim=args.compressor_dim,
            hidden_sizes=args.compressor_hidden_sizes,
            total_steps=args.compressor_steps,
            lr_init=args.compressor_lr,
            batch_size=args.compressor_batch_size,
            save_every=args.compressor_save_every,
            save_dir=comp_save_dir,
            seed=args.seed,
            abort_on_nonfinite=args.abort_on_nonfinite_compressor,
            max_nonfinite_events=args.max_nonfinite_compressor_events,
            vmim_nf_layers=args.compressor_vmim_nf_layers,
            vmim_nf_hidden=args.compressor_vmim_nf_hidden,
        )
        compressor_params_ref = str(best_params_path.resolve())
        compressor_state_ref = str(best_state_path.resolve())
        compressor_summary_path = comp_save_dir / "compressor_training_summary.json"

        if compressor_input_standardized and compressor_input_mean is not None and compressor_input_std is not None:
            save_path.mkdir(parents=True, exist_ok=True)
            np.savez(
                compressor_input_stats_path,
                mean=compressor_input_mean,
                std=compressor_input_std,
                clip_value=(
                    np.nan
                    if compressor_input_clip_value is None
                    else float(compressor_input_clip_value)
                ),
            )

        if cache_dir is not None:
            for f in [
                "l1vmim_train.npz",
                "l1vmim_val.npz",
                "l1vmim_obs.npz",
                "l1vmim_cache_meta.npz",
            ]:
                p = cache_dir / f
                if p.exists():
                    p.unlink()
                    print(f"  Deleted stale compressed cache: {p}")
    else:
        if args.compressor_params is None or args.compressor_state is None:
            raise ValueError(
                "When --train-compressor is not set, both --compressor-params and "
                "--compressor-state are required."
            )
        comp_params, comp_state = load_compressor_params(
            args.compressor_params,
            args.compressor_state,
        )
        if args.compressor_params is not None:
            candidate_summary = (
                Path(args.compressor_params).resolve().parent
                / "compressor_training_summary.json"
            )
            if candidate_summary.exists():
                compressor_summary_path = candidate_summary

    compressed_cache_expected = build_compressed_cache_metadata(
        args=args,
        tomo_bin_indices=tomo_bin_indices,
        l1_min_snr=l1_min_snr,
        l1_max_snr=l1_max_snr,
        l1_clamp_overflow=effective_l1_clamp,
        subtract_coarse_mean=effective_subtract_coarse_mean,
        compressor_source=compressor_source,
        compressor_params_path=compressor_params_ref,
        compressor_state_path=compressor_state_ref,
    )

    # 4) Compress L1 datasets
    print("######## COMPRESS L1 VECTORS ########")
    compressed_cache_ok = False
    if cache_dir is not None and cache_dir.exists():
        train_cache = cache_dir / "l1vmim_train.npz"
        val_cache = cache_dir / "l1vmim_val.npz"
        obs_cache = cache_dir / "l1vmim_obs.npz"
        meta_cache = cache_dir / "l1vmim_cache_meta.npz"
        if train_cache.exists() and val_cache.exists() and obs_cache.exists() and meta_cache.exists():
            meta = np.load(meta_cache)
            compressed_cache_ok, mismatches = compare_cache_metadata(
                meta,
                compressed_cache_expected,
            )
            if compressed_cache_ok:
                d_tr = np.load(train_cache)
                d_va = np.load(val_cache)
                d_obs = np.load(obs_cache)
                dataset_train_comp = {
                    "theta": dataset_train["theta"],
                    "x": np.asarray(d_tr["x"], dtype=np.float32),
                }
                dataset_val_comp = {
                    "theta": dataset_val["theta"],
                    "x": np.asarray(d_va["x"], dtype=np.float32),
                }
                obs_comp = np.asarray(d_obs["x"], dtype=np.float32)
                print("  Loaded cached compressed L1 datasets (metadata matches).")
            else:
                first_mismatch = mismatches[0] if mismatches else "unknown mismatch"
                print(
                    "  Compressed cache metadata mismatch; recomputing. "
                    f"First mismatch: {first_mismatch}"
                )

    if not compressed_cache_ok:
        train_comp_x = compress_features(
            dataset_train["x"],
            compressor,
            comp_params,
            comp_state,
            batch_size=args.compress_batch_size,
        )
        val_comp_x = compress_features(
            dataset_val["x"],
            compressor,
            comp_params,
            comp_state,
            batch_size=args.compress_batch_size,
        )
        obs_comp_batch = compress_features(
            obs_l1.reshape(1, -1),
            compressor,
            comp_params,
            comp_state,
            batch_size=1,
        )
        obs_comp = obs_comp_batch.reshape(-1).astype(np.float32)

        dataset_train_comp = {"theta": dataset_train["theta"], "x": train_comp_x}
        dataset_val_comp = {"theta": dataset_val["theta"], "x": val_comp_x}

        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(cache_dir / "l1vmim_train.npz", x=train_comp_x)
            np.savez(cache_dir / "l1vmim_val.npz", x=val_comp_x)
            np.savez(cache_dir / "l1vmim_obs.npz", x=obs_comp)
            np.savez(cache_dir / "l1vmim_cache_meta.npz", **compressed_cache_expected)
            print(f"  Cached compressed datasets to {cache_dir}")

    print(f"  Train compressed shape = {dataset_train_comp['x'].shape}")
    print(f"  Val   compressed shape = {dataset_val_comp['x'].shape}")
    print(f"  Obs   compressed shape = {obs_comp.shape}")

    # 5) Optional compressed-summary standardization
    standardization_applied = False
    standardization_mean: Optional[np.ndarray] = None
    standardization_std: Optional[np.ndarray] = None
    summary_clip_value = (
        args.summary_clip_value if args.summary_clip_value > 0 else None
    )

    if args.standardize_summary:
        if args.no_train:
            if not summary_stats_path.exists():
                raise FileNotFoundError(
                    "--no-train with --standardize-summary requires saved stats at "
                    f"{summary_stats_path}."
                )
            std_data = np.load(summary_stats_path)
            standardization_mean = np.array(std_data["mean"])
            standardization_std = np.array(std_data["std"])
            if "clip_value" in std_data.files:
                loaded_clip = float(std_data["clip_value"])
                if np.isfinite(loaded_clip) and loaded_clip > 0:
                    summary_clip_value = loaded_clip
                else:
                    summary_clip_value = None
            (
                dataset_train_comp["x"],
                dataset_val_comp["x"],
                obs_comp,
            ) = apply_summary_standardization(
                dataset_train_comp["x"],
                dataset_val_comp["x"],
                obs_comp,
                standardization_mean,
                standardization_std,
                clip_value=summary_clip_value,
            )
            standardization_applied = True
            print(f"  Loaded compressed-summary stats from {summary_stats_path}")
        else:
            (
                dataset_train_comp["x"],
                dataset_val_comp["x"],
                obs_comp,
                standardization_mean,
                standardization_std,
            ) = fit_and_apply_summary_standardization(
                dataset_train_comp["x"],
                dataset_val_comp["x"],
                obs_comp,
                clip_value=summary_clip_value,
            )
            standardization_applied = True
            print(
                "  Applied compressed-summary standardization "
                f"(clip={summary_clip_value if summary_clip_value is not None else 'off'})."
            )

    summary_dim = int(dataset_train_comp["x"].shape[1])
    validate_flow_inputs(dataset_train_comp, dataset_val_comp, obs_comp, args.n_cosmo)

    summary_health_diag = collect_compressed_summary_health_diagnostics(
        obs_summary=obs_comp,
        train_x=dataset_train_comp["x"],
        val_x=dataset_val_comp["x"],
    )
    save_path.mkdir(parents=True, exist_ok=True)
    summary_health_path.write_text(
        json.dumps(summary_health_diag, indent=2),
        encoding="utf-8",
    )
    wandb.log(
        {
            "diagnostics/summary_std_min": summary_health_diag["summary_std_min"],
            "diagnostics/summary_std_median": summary_health_diag["summary_std_median"],
            "diagnostics/summary_std_max": summary_health_diag["summary_std_max"],
            "diagnostics/summary_dead_feature_frac": summary_health_diag[
                "summary_dead_feature_frac"
            ],
            "diagnostics/obs_in_train_p01_p99_frac": summary_health_diag[
                "obs_in_train_p01_p99_frac"
            ],
            "diagnostics/val_train_mean_shift_median_sigma": summary_health_diag[
                "val_train_mean_shift_median_sigma"
            ],
            "diagnostics/val_train_mean_shift_max_sigma": summary_health_diag[
                "val_train_mean_shift_max_sigma"
            ],
        }
    )
    print(
        "  Summary health | "
        f"std[min,med,max]=[{summary_health_diag['summary_std_min']:.4e}, "
        f"{summary_health_diag['summary_std_median']:.4e}, "
        f"{summary_health_diag['summary_std_max']:.4e}] | "
        f"dead_frac={summary_health_diag['summary_dead_feature_frac']:.3f} | "
        f"obs_inlier_frac={summary_health_diag['obs_in_train_p01_p99_frac']:.3f}"
    )

    wandb.log(
        {
            "data/train_size": len(dataset_train_comp["theta"]),
            "data/val_size": len(dataset_val_comp["theta"]),
            "data/raw_l1_dim": int(raw_summary_dim),
            "data/compressor_dim": int(args.compressor_dim),
            "data/summary_dim": int(summary_dim),
            "data/summary_standardized": int(standardization_applied),
            "data/summary_clip_value": (
                float(summary_clip_value) if summary_clip_value is not None else 0.0
            ),
            "data/train_x_min": float(dataset_train_comp["x"].min()),
            "data/train_x_max": float(dataset_train_comp["x"].max()),
            "data/train_x_mean": float(dataset_train_comp["x"].mean()),
            "data/train_x_std": float(dataset_train_comp["x"].std()),
            "diagnostics/summary_std_min": summary_health_diag["summary_std_min"],
            "diagnostics/summary_std_median": summary_health_diag["summary_std_median"],
            "diagnostics/summary_std_max": summary_health_diag["summary_std_max"],
            "diagnostics/summary_dead_feature_frac": summary_health_diag[
                "summary_dead_feature_frac"
            ],
            "diagnostics/obs_in_train_p01_p99_frac": summary_health_diag[
                "obs_in_train_p01_p99_frac"
            ],
        }
    )

    # 6) Build & train flow
    nf_logp, nf_sample = build_flow(
        n_cosmo_params=args.n_cosmo,
        n_layers=args.nvp_layers,
        hidden=args.nvp_hidden,
    )

    flow_params = None
    flow_params_source = "unknown"

    if args.no_train:
        best_path = save_path / "params_l1vmim_flow_best.pkl"
        if best_path.exists():
            load_path = best_path
        else:
            candidates = sorted(
                save_path.glob("params_l1vmim_flow_batch*.pkl"),
                key=_checkpoint_step,
            )
            if not candidates:
                raise FileNotFoundError(
                    f"No saved flow params in {save_path} and --no-train set"
                )
            load_path = candidates[-1]
        with open(load_path, "rb") as f:
            flow_params = pickle.load(f)
        flow_params_source = str(load_path.resolve())
        print(f"  Loaded flow params from {load_path}")
    else:
        _lr_schedule = optax.cosine_decay_schedule(
            init_value=args.lr_init,
            decay_steps=args.total_steps,
            alpha=args.lr_end / max(args.lr_init, 1e-12),
        )
        flow_params = train_flow(
            rng,
            nf_logp,
            dataset_train_comp,
            dataset_val_comp,
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

        if standardization_applied and standardization_mean is not None and standardization_std is not None:
            save_path.mkdir(parents=True, exist_ok=True)
            np.savez(
                summary_stats_path,
                mean=standardization_mean,
                std=standardization_std,
                clip_value=(
                    np.nan if summary_clip_value is None else float(summary_clip_value)
                ),
            )

    # 7) Posterior sampling
    if not args.no_sample:
        posterior_samples = sample_posterior(
            rng_sample,
            nf_sample,
            flow_params,
            obs_comp,
            args.npe_samples,
        )
        out = Path(args.posterior_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, posterior_samples)

        metadata = {
            "method": "l1_vmim",
            "posterior_file": str(out.resolve()),
            "flow_params_source": flow_params_source,
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
            "compressor_source": compressor_source,
            "compressor_params": compressor_params_ref,
            "compressor_state": compressor_state_ref,
            "compressor_abort_on_nonfinite": bool(args.abort_on_nonfinite_compressor),
            "compressor_max_nonfinite_events": int(args.max_nonfinite_compressor_events),
            "compressor_dim": int(args.compressor_dim),
            "compressor_hidden": [int(v) for v in args.compressor_hidden_sizes],
            "compressor_log1p_input": bool(args.compressor_log1p_input),
            "compressor_input_standardized": bool(compressor_input_standardized),
            "compressor_input_standardization_file": (
                str(compressor_input_stats_path.resolve())
                if compressor_input_standardized and compressor_input_stats_path.exists()
                else None
            ),
            "compressor_input_clip_value": (
                float(compressor_input_clip_value)
                if compressor_input_clip_value is not None
                else None
            ),
            "compressor_vmim_nf_layers": int(args.compressor_vmim_nf_layers),
            "compressor_vmim_nf_hidden": int(args.compressor_vmim_nf_hidden),
            "l1_implementation": args.l1_implementation,
            "wandb": {
                "project": args.wandb_project,
                "entity": args.wandb_entity,
                "group": wandb_group,
                "tags": wandb_tags,
            },
            "l1_settings": {
                "n_scales": int(args.n_scales),
                "l1_nbins": int(args.l1_nbins),
                "l1_min_snr": float(l1_min_snr),
                "l1_max_snr": float(l1_max_snr),
                "l1_clamp_overflow": bool(effective_l1_clamp),
                "subtract_coarse_mean": bool(effective_subtract_coarse_mean),
            },
            "tomo_bin_indices": list(tomo_bin_indices),
            "tfds_name": args.tfds_name,
            "map_kind": args.map_kind,
            "truth_parameters": [float(v) for v in np.asarray(truth).ravel()],
            "diagnostic_files": {
                "flow_training_summary_path": str(flow_summary_path.resolve()),
                "compressor_training_summary_path": (
                    str(compressor_summary_path.resolve())
                    if compressor_summary_path is not None and compressor_summary_path.exists()
                    else None
                ),
                "compressed_summary_health_path": (
                    str(summary_health_path.resolve())
                    if summary_health_path.exists()
                    else None
                ),
            },
            "compressed_summary_health": summary_health_diag,
        }
        if flow_summary_path.exists():
            metadata["flow_training_summary"] = json.loads(
                flow_summary_path.read_text(encoding="utf-8")
            )
        if compressor_summary_path is not None and compressor_summary_path.exists():
            metadata["compressor_training_summary"] = json.loads(
                compressor_summary_path.read_text(encoding="utf-8")
            )
        if summary_health_path.exists():
            metadata["compressed_summary_health"] = json.loads(
                summary_health_path.read_text(encoding="utf-8")
            )

        out.with_suffix(".meta.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        print(f"  Saved posterior samples -> {out.resolve()}")

        if args.plot:
            fig_out = Path(args.figure_out)
            fig_out.parent.mkdir(parents=True, exist_ok=True)
            plot_posterior(
                posterior_samples,
                truth,
                str(fig_out),
                param_names,
                log_to_wandb=(not args.no_wandb),
            )
    else:
        print("  Skipping posterior sampling (--no-sample)")

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
