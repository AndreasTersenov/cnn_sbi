#!/usr/bin/env python
"""
L1-norm + jaxili NPE for tomographic weak-lensing cosmological inference.

This pipeline intentionally reuses the same L1 datavector construction choices as
`npe_l1norm_nbody_tomo.py` and swaps only the density estimator to `jaxili.NPE`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import importlib
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import torch
from jax.lib import xla_bridge
from sklearn.decomposition import PCA

try:
    from jaxili.inference import NPE
except ImportError as exc:  # pragma: no cover - runtime env dependent
    raise ImportError(
        "Failed to import jaxili. Activate the conda environment 'jaxili' "
        "before running this script."
    ) from exc

# wl_stats_torch — add to path if not installed
_WL_STATS_PATH = "/home/tersenov/software/wl_stats_torch"
if _WL_STATS_PATH not in sys.path:
    sys.path.insert(0, _WL_STATS_PATH)
from wl_stats_torch import WLStatistics  # noqa: E402


def _ensure_tfds_builder_registered() -> None:
    """Import the local TFDS builder lazily to avoid hard dependency at --help time."""
    try:
        importlib.import_module("tf_dataset_nbody_tomo")
    except ModuleNotFoundError as exc:
        if exc.name == "tensorflow_datasets":
            raise ModuleNotFoundError(
                "Missing dependency 'tensorflow_datasets'. Install it in the 'jaxili' "
                "environment to run L1 datavector extraction."
            ) from exc
        raise


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
    deduped = []
    seen = set()
    for b in values:
        if b not in seen:
            deduped.append(b)
            seen.add(b)
    return tuple(deduped)


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
    _ensure_tfds_builder_registered()
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
        maps_np = example["maps"]
        if np.isnan(maps_np).any():
            continue
        for b in range(nbins):
            map_dtype = np.float32 if l1_implementation == "cosmoford" else np.float64
            img_batch = torch.from_numpy(maps_np[:, :, :, b].astype(map_dtype)).to(device)
            if l1_implementation == "cosmoford":
                stats.compute_wavelet_transform(img_batch.float(), float(noise_sigma))
            else:
                stats.compute_wavelet_transform(
                    img_batch,
                    noise_sigma,
                    subtract_coarse_mean=subtract_coarse_mean,
                )
            snr = stats.snr_coeffs
            batch_min = snr.min().item()
            batch_max = snr.max().item()
            global_min = min(global_min, batch_min)
            global_max = max(global_max, batch_max)
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
    return np.concatenate(all_l1)


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
    return np.concatenate(all_l1, axis=-1)


def build_augmentation(
    map_kind: str,
    sigma_e: float,
    galaxy_density: float,
    field_size: int,
    field_npix: int,
    nbins: int,
    tomo_bin_indices: tuple[int, ...],
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
    _ensure_tfds_builder_registered()
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
    return {"theta": np.concatenate(theta_list, axis=0), "x": np.concatenate(x_list, axis=0)}


def _summary_transform_flags(summary_transform: str) -> tuple[bool, bool, str]:
    if summary_transform == "log1p-zscore":
        return True, True, "log1p"
    if summary_transform == "log10p-zscore":
        return True, True, "log10p"
    if summary_transform == "zscore":
        return False, True, "none"
    if summary_transform == "log1p":
        return True, False, "log1p"
    if summary_transform == "log10p":
        return True, False, "log10p"
    if summary_transform == "none":
        return False, False, "none"
    raise ValueError(
        f"Unknown summary transform '{summary_transform}'. "
        "Expected one of: log1p-zscore, log10p-zscore, zscore, log1p, log10p, none."
    )


def preprocess_summaries(
    train_x: np.ndarray,
    val_x: np.ndarray,
    obs_x: np.ndarray,
    summary_transform: str = "log1p-zscore",
    clip_value: Optional[float] = 5.0,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    apply_log, apply_standardize, log_kind = _summary_transform_flags(summary_transform)

    if apply_log:
        for arr_name, arr in (("train_x", train_x), ("val_x", val_x), ("obs_x", obs_x)):
            if np.any(arr < -1.0):
                raise ValueError(
                    f"{arr_name} contains values < -1, cannot apply {log_kind} safely "
                    f"(minimum={arr.min():.6e})."
                )
        if log_kind == "log1p":
            train_proc = np.log1p(train_x)
            val_proc = np.log1p(val_x)
            obs_proc = np.log1p(obs_x)
        elif log_kind == "log10p":
            train_proc = np.log10(train_x + 1.0)
            val_proc = np.log10(val_x + 1.0)
            obs_proc = np.log10(obs_x + 1.0)
        else:
            raise ValueError(f"Unexpected log kind: {log_kind}")
    else:
        train_proc = train_x
        val_proc = val_x
        obs_proc = obs_x

    if apply_standardize:
        if (mean is None) ^ (std is None):
            raise ValueError("Both mean and std must be provided together.")
        if mean is None or std is None:
            mean = train_proc.mean(axis=0)
            std = train_proc.std(axis=0)
        else:
            mean = np.asarray(mean)
            std = np.asarray(std)
            if mean.shape != (train_proc.shape[1],) or std.shape != (train_proc.shape[1],):
                raise ValueError(
                    "Loaded standardization stats have incompatible shape: "
                    f"mean={mean.shape}, std={std.shape}, expected={(train_proc.shape[1],)}."
                )
        std = std.copy()
        std[std < 1e-12] = 1.0
        train_out = (train_proc - mean) / std
        val_out = (val_proc - mean) / std
        obs_out = (obs_proc - mean) / std
    else:
        mean = np.zeros(train_proc.shape[1], dtype=train_proc.dtype)
        std = np.ones(train_proc.shape[1], dtype=train_proc.dtype)
        train_out = train_proc
        val_out = val_proc
        obs_out = obs_proc

    if clip_value is not None and clip_value > 0:
        train_out = np.clip(train_out, -clip_value, clip_value)
        val_out = np.clip(val_out, -clip_value, clip_value)
        obs_out = np.clip(obs_out, -clip_value, clip_value)
    return train_out, val_out, obs_out, mean, std


def fit_pca(
    train_x: np.ndarray,
    val_x: np.ndarray,
    obs_x: np.ndarray,
    n_components: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PCA]:
    pca = PCA(n_components=n_components, whiten=True)
    train_pca = pca.fit_transform(train_x).astype(np.float32)
    val_pca = pca.transform(val_x).astype(np.float32)
    obs_pca = pca.transform(obs_x.reshape(1, -1)).astype(np.float32).squeeze(0)
    explained = pca.explained_variance_ratio_.sum() * 100
    print(
        f"  PCA: {train_x.shape[1]} → {n_components} components "
        f"({explained:.1f}% variance explained)"
    )
    return train_pca, val_pca, obs_pca, pca


def apply_saved_pca(
    train_x: np.ndarray,
    val_x: np.ndarray,
    obs_x: np.ndarray,
    pca_components: np.ndarray,
    pca_mean: np.ndarray,
    pca_explained_variance: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pca_components = np.asarray(pca_components)
    pca_mean = np.asarray(pca_mean)
    if pca_components.ndim != 2:
        raise ValueError(f"pca_components must be 2D, got shape {pca_components.shape}.")
    if pca_mean.shape != (pca_components.shape[1],):
        raise ValueError(
            f"pca_mean shape {pca_mean.shape} incompatible with "
            f"pca_components shape {pca_components.shape}."
        )
    if pca_explained_variance is None:
        raise ValueError("Missing pca_explained_variance for whitening.")
    pca_explained_variance = np.asarray(pca_explained_variance)
    if pca_explained_variance.shape != (pca_components.shape[0],):
        raise ValueError(
            f"pca_explained_variance shape {pca_explained_variance.shape} incompatible with "
            f"number of PCA components {pca_components.shape[0]}."
        )

    whitening = np.sqrt(np.maximum(pca_explained_variance, 1e-12))

    def _transform(x: np.ndarray) -> np.ndarray:
        centered = x - pca_mean
        projected = centered @ pca_components.T
        return (projected / whitening).astype(np.float32)

    train_pca = _transform(train_x)
    val_pca = _transform(val_x)
    obs_pca = _transform(obs_x.reshape(1, -1)).squeeze(0)
    return train_pca, val_pca, obs_pca


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
    print(f"  Saved posterior plot → {output_path}")
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="L1-norm datavectors + jaxili NPE for tomographic weak lensing"
    )

    # Hardware
    p.add_argument("--cuda-visible-devices", type=str, default="0")
    p.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU index alias for --cuda-visible-devices (e.g. --gpu 2).",
    )

    # Survey / map configuration
    p.add_argument("--field-size", type=int, default=10, help="Field size in degrees")
    p.add_argument("--field-npix", type=int, default=80, help="Pixels per side")
    p.add_argument("--nside", type=int, default=512, help="HEALPix NSIDE")
    p.add_argument("--sigma-e", type=float, default=0.26, help="Shape noise dispersion per component")
    p.add_argument(
        "--galaxy-density",
        type=float,
        default=30 / 4,
        help="Galaxy number density [arcmin^{-2}]",
    )
    p.add_argument("--nbins", type=int, default=4, help="Number of tomographic bins")
    p.add_argument("--n-cosmo", type=int, default=6, help="Number of cosmological parameters")

    # L1 configuration
    p.add_argument("--n-scales", type=int, default=5, help="Number of starlet wavelet scales")
    p.add_argument("--l1-nbins", type=int, default=40, help="Number of L1 histogram bins per scale")
    p.add_argument("--l1-min-snr", type=float, default=-10.0, help="Fixed minimum SNR")
    p.add_argument("--l1-max-snr", type=float, default=10.0, help="Fixed maximum SNR")
    p.add_argument(
        "--auto-calibrate-snr",
        action="store_true",
        help="Estimate global SNR range from data instead of fixed bounds",
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
    )
    p.add_argument(
        "--no-subtract-coarse-mean",
        dest="subtract_coarse_mean",
        action="store_false",
    )
    p.set_defaults(subtract_coarse_mean=True)
    p.add_argument(
        "--l1-implementation",
        type=str,
        default="cnn_sbi",
        choices=["cnn_sbi", "cosmoford"],
        help="L1 extraction implementation mode",
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
    p.add_argument("--posterior-out", type=str, default="posterior_l1norm_jaxili_tomo.npy")
    p.add_argument("--figure-out", type=str, default="posterior_l1norm_jaxili_tomo.png")
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument(
        "--tfds-name",
        type=str,
        default="NbodyCosmogridDatasetTomo/grid",
        help="TFDS dataset name/config for train and test maps",
    )
    p.add_argument(
        "--tomo-bin-indices",
        type=str,
        default="1,2,3,4",
        help="Tomographic bins to use, e.g. '1,2,3,4' or '3'",
    )

    # Summary preprocessing
    p.add_argument("--pca-components", type=int, default=50, help="PCA components (0 disables PCA)")
    p.add_argument(
        "--summary-transform",
        type=str,
        default="log1p-zscore",
        choices=[
            "log1p-zscore",
            "log10p-zscore",
            "zscore",
            "log1p",
            "log10p",
            "none",
        ],
    )
    p.add_argument("--clip-value", type=float, default=5.0)

    # jaxili NPE training
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="jaxili training epochs; defaults to --total-steps for compatibility",
    )
    p.add_argument(
        "--total-steps",
        type=int,
        default=5_000,
        help="Compatibility alias used as default epochs when --epochs is unset",
    )
    p.add_argument("--batch-size", type=int, default=128, help="jaxili training batch size")
    p.add_argument("--learning-rate", type=float, default=1e-4, help="jaxili learning rate")
    p.add_argument("--nan-retries", type=int, default=10, help="Max retry attempts if NaN loss occurs")
    p.add_argument(
        "--npe-warmup-steps",
        type=int,
        default=128,
        help="Learning-rate warmup steps passed to jaxili optimizer",
    )
    p.add_argument(
        "--npe-decay-steps",
        type=int,
        default=10_000,
        help="Learning-rate decay steps passed to jaxili optimizer",
    )
    p.add_argument(
        "--checkpoint-name",
        type=str,
        default="params_l1norm_jaxili",
        help="Checkpoint basename saved under save_dir/l1norm_jaxili/<map-kind>",
    )
    p.add_argument(
        "--min-feature-variance",
        type=float,
        default=1e-5,
        help="Minimum variance threshold for zero-variance feature filtering",
    )

    # Compatibility no-op args (passed by existing sweep runner)
    p.add_argument("--save-every", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--patience", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--lr-init", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--lr-end", type=float, default=None, help=argparse.SUPPRESS)

    # Sampling / execution
    p.add_argument("--npe-samples", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-train", action="store_true")
    p.add_argument("--no-sample", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--ds-batch-size", type=int, default=256)
    p.add_argument("--no-wandb", action="store_true", help="Compatibility flag; ignored in this script")

    args = p.parse_args()
    if args.gpu is not None:
        args.cuda_visible_devices = str(args.gpu)
    if args.epochs is None:
        args.epochs = args.total_steps
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.nan_retries <= 0:
        raise ValueError("--nan-retries must be > 0.")
    if args.npe_warmup_steps <= 0:
        raise ValueError("--npe-warmup-steps must be > 0.")
    if args.npe_decay_steps <= 0:
        raise ValueError("--npe-decay-steps must be > 0.")
    return args


def build_l1_cache_metadata(
    args: argparse.Namespace,
    tomo_bin_indices: tuple[int, ...],
    l1_min_snr: float,
    l1_max_snr: float,
    l1_clamp_overflow: bool,
    subtract_coarse_mean: bool,
) -> Dict[str, object]:
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
        "map_kind": str(args.map_kind),
        "field_size": int(args.field_size),
        "field_npix": int(args.field_npix),
        "nside": int(args.nside),
        "nbins": int(args.nbins),
        "sigma_e": float(args.sigma_e),
        "galaxy_density": float(args.galaxy_density),
        "ds_batch_size": int(args.ds_batch_size),
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


def filter_zero_variance_bins(
    data: np.ndarray,
    min_variance: float = 1e-5,
    verbose: bool = True,
) -> tuple[np.ndarray, int]:
    """Return mask of features whose sample variance exceeds min_variance."""
    variances = np.var(data, axis=0)
    valid_mask = variances > min_variance
    n_removed = int(valid_mask.size - np.sum(valid_mask))
    if verbose:
        print("######## FEATURE FILTERING ########")
        print(f"  Total features: {valid_mask.size}")
        print(f"  Kept features : {int(np.sum(valid_mask))}")
        print(f"  Removed       : {n_removed} (variance <= {min_variance})")
    return valid_mask, n_removed


def _metric_has_nan(metric_value) -> bool:
    if isinstance(metric_value, (list, tuple)):
        arr = np.asarray(metric_value, dtype=np.float64)
        return bool(np.any(np.isnan(arr)))
    if isinstance(metric_value, np.ndarray):
        return bool(np.any(np.isnan(metric_value)))
    try:
        return bool(np.isnan(metric_value))
    except TypeError:
        return False


def _resolve_latest_jaxili_checkpoint_dir(checkpoint_root: Path) -> Path:
    """Resolve latest completed jaxili checkpoint subdir (NDE_w_Standardization/version_*)."""
    nde_root = checkpoint_root / "NDE_w_Standardization"
    if not nde_root.exists():
        raise FileNotFoundError(
            f"Missing jaxili checkpoint directory '{nde_root}'. "
            "Run without --no-train first."
        )

    version_dirs = sorted(
        [p for p in nde_root.glob("version_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1,
        reverse=True,
    )
    if not version_dirs:
        raise FileNotFoundError(
            f"No version_* checkpoint folders found under '{nde_root}'."
        )

    for version_dir in version_dirs:
        has_hparams = (version_dir / "hparams.json").exists()
        has_numeric_ckpt = any(
            p.is_dir() and p.name.isdigit() for p in version_dir.iterdir()
        )
        if has_hparams and has_numeric_ckpt:
            return version_dir

    raise FileNotFoundError(
        f"No completed jaxili checkpoints found under '{nde_root}'. "
        "Found only temporary/incomplete checkpoint directories."
    )


def _normalize_jaxili_hparams_embedding_arrays(version_dir: Path) -> None:
    """Patch multiline Array([...]) entries so jaxili regex-based loader can parse them."""
    hparams_path = version_dir / "hparams.json"
    if not hparams_path.exists():
        return
    raw = hparams_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    embedding = data.get("model_hparams", {}).get("embedding_net")
    if not isinstance(embedding, str):
        return

    normalized_embedding = embedding.replace("\n", " ")
    normalized_embedding = re.sub(r"\s+", " ", normalized_embedding)
    normalized_embedding = re.sub(r"\s*,\s*", ", ", normalized_embedding)
    if normalized_embedding != embedding:
        data["model_hparams"]["embedding_net"] = normalized_embedding
        hparams_path.write_text(json.dumps(data, indent=4), encoding="utf-8")


def train_with_nan_retry(
    inference: NPE,
    checkpoint_path: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    warmup_steps: int,
    decay_steps: int,
    params: jnp.ndarray,
    data: jnp.ndarray,
    split_key: jax.Array,
    max_retries: int = 10,
) -> tuple[NPE, object, object]:
    """Train jaxili NPE with bounded retries if train/val losses become NaN."""
    for attempt in range(1, max_retries + 1):
        print(f"######## JAXILI TRAINING ATTEMPT {attempt}/{max_retries} ########")
        try:
            metrics, density_estimator = inference.train(
                checkpoint_path=checkpoint_path,
                num_epochs=epochs,
                learning_rate=learning_rate,
                training_batch_size=batch_size,
                warmup=warmup_steps,
                decay_steps=decay_steps,
            )

            nan_source = None
            for metric_name in ("train_loss", "val_loss"):
                if hasattr(metrics, metric_name) and _metric_has_nan(getattr(metrics, metric_name)):
                    nan_source = metric_name
                    break

            if nan_source is not None:
                print(f"  NaN detected in {nan_source}; reinitializing inference object.")
                inference = NPE()
                inference = inference.append_simulations(params, data, key=split_key)
                continue

            if hasattr(metrics, "test_loss") and _metric_has_nan(getattr(metrics, "test_loss")):
                print("  Note: test_loss has NaN values; continuing because train/val are finite.")

            print("  Training completed successfully.")
            return inference, metrics, density_estimator
        except Exception as exc:
            print(f"  Training attempt failed: {exc}")
            if attempt == max_retries:
                raise
            inference = NPE()
            inference = inference.append_simulations(params, data, key=split_key)

    raise RuntimeError("Training failed after exhausting NaN-retry budget.")


def validate_npe_inputs(
    dataset_train: Dict[str, np.ndarray],
    dataset_val: Dict[str, np.ndarray],
    obs_summary: np.ndarray,
    n_cosmo: int,
) -> None:
    """Fail fast if train/val/obs arrays are malformed for NPE."""
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


def _metrics_summary(metrics: object) -> dict:
    summary: dict[str, float] = {}
    for key in ("train_loss", "val_loss", "test_loss"):
        if not hasattr(metrics, key):
            continue
        value = getattr(metrics, key)
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            summary[f"{key}_last"] = float(arr)
            summary[f"{key}_nan"] = bool(np.isnan(arr))
        elif arr.size > 0:
            summary[f"{key}_last"] = float(arr.ravel()[-1])
            summary[f"{key}_min"] = float(np.nanmin(arr))
            summary[f"{key}_nan"] = bool(np.any(np.isnan(arr)))
    return summary


def main() -> None:
    args = parse_args()
    if args.no_wandb:
        print("  --no-wandb is accepted for compatibility and has no effect in this script.")

    tomo_bin_indices = parse_tomo_bin_indices(args.tomo_bin_indices)
    if args.nbins != len(tomo_bin_indices):
        print(
            f"  Overriding nbins from {args.nbins} to {len(tomo_bin_indices)} "
            f"to match selected bins {tomo_bin_indices}."
        )
        args.nbins = len(tomo_bin_indices)

    torch_device = setup_environment(args.cuda_visible_devices)
    rng = jax.random.PRNGKey(args.seed)
    rng, rng_obs, rng_sample = jax.random.split(rng, 3)
    split_seed = int(args.seed) + 1
    split_key = jax.random.PRNGKey(split_seed)

    pixel_arcmin = args.field_size * 60.0 / args.field_npix
    noise_sigma = pixel_noise_sigma(
        args.sigma_e, args.galaxy_density, args.field_size, args.field_npix
    )
    raw_summary_dim = args.n_scales * args.l1_nbins * args.nbins
    print(f"  pixel_arcmin   = {pixel_arcmin:.2f}")
    print(f"  noise_sigma    = {noise_sigma:.6f}")
    print(
        f"  raw_summary    = {raw_summary_dim} "
        f"({args.n_scales} scales × {args.l1_nbins} bins × {args.nbins} tomo bins)"
    )

    save_path = (Path(args.save_dir) / "l1norm_jaxili" / args.map_kind).resolve()
    preprocessing_stats_path = save_path / "l1_jaxili_standardization.npz"
    feature_mask_path = save_path / "l1_jaxili_feature_mask.npz"
    training_summary_path = save_path / "jaxili_training_summary.json"
    checkpoint_path = (save_path / args.checkpoint_name).resolve()

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
    )

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    cache_meta_expected: Optional[Dict[str, object]] = None
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

    # 3) Train/val L1 datasets with cache metadata checks
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
            np.savez(
                cache_dir / "l1_cache_meta.npz",
                **cache_meta_expected,
            )
            print(f"  Cached datasets to {cache_dir}")

    print(f"  Train x shape = {dataset_train['x'].shape}")
    print(f"  Val   x shape = {dataset_val['x'].shape}")

    # 4) Summary preprocessing and optional PCA
    print("######## SUMMARY PREPROCESSING ########")
    requested_clip_val = args.clip_value if args.clip_value > 0 else None
    requested_transform = args.summary_transform
    effective_clip_val = requested_clip_val
    effective_transform = requested_transform
    loaded_stats: Optional[dict[str, np.ndarray]] = None

    if args.no_train:
        if not preprocessing_stats_path.exists():
            raise FileNotFoundError(
                f"--no-train requires preprocessing stats at {preprocessing_stats_path}."
            )
        with np.load(preprocessing_stats_path, allow_pickle=False) as saved:
            loaded_stats = {k: np.array(saved[k]) for k in saved.files}
        effective_transform = (
            str(loaded_stats["summary_transform"])
            if "summary_transform" in loaded_stats
            else "log1p-zscore"
        )
        if "clip_value" in loaded_stats:
            clip_raw = float(loaded_stats["clip_value"])
            effective_clip_val = None if np.isnan(clip_raw) else clip_raw
        if effective_transform != requested_transform:
            print(
                f"  Overriding requested --summary-transform={requested_transform} "
                f"with saved '{effective_transform}' for checkpoint compatibility."
            )
        if effective_clip_val != requested_clip_val:
            print(
                f"  Overriding requested clip value {requested_clip_val} "
                f"with saved {effective_clip_val}."
            )

    loaded_mean = None if loaded_stats is None else loaded_stats.get("mean")
    loaded_std = None if loaded_stats is None else loaded_stats.get("std")
    dataset_train["x"], dataset_val["x"], obs_l1_std, mean, std = preprocess_summaries(
        dataset_train["x"],
        dataset_val["x"],
        obs_l1,
        summary_transform=effective_transform,
        clip_value=effective_clip_val,
        mean=loaded_mean,
        std=loaded_std,
    )

    apply_log, apply_standardize, log_kind = _summary_transform_flags(effective_transform)
    print(
        f"  Transform = {effective_transform} "
        f"(log={log_kind}, zscore={int(apply_standardize)}, clip={effective_clip_val})"
    )
    if apply_standardize:
        print(f"  Summary mean range = [{mean.min():.4f}, {mean.max():.4f}]")
        print(f"  Summary std  range = [{std.min():.4f}, {std.max():.4f}]")

    pca_model = None
    pca_applied = False
    pca_source = "none"
    saved_has_pca = (
        loaded_stats is not None
        and {"pca_components", "pca_mean", "pca_explained_variance"}.issubset(loaded_stats.keys())
    )
    if args.no_train and saved_has_pca:
        saved_n_components = int(loaded_stats["pca_components"].shape[0])
        if args.pca_components != saved_n_components:
            print(
                f"  Overriding requested --pca-components={args.pca_components} "
                f"with saved PCA components={saved_n_components}."
            )
        dataset_train["x"], dataset_val["x"], obs_l1_std = apply_saved_pca(
            dataset_train["x"],
            dataset_val["x"],
            obs_l1_std,
            pca_components=loaded_stats["pca_components"],
            pca_mean=loaded_stats["pca_mean"],
            pca_explained_variance=loaded_stats["pca_explained_variance"],
        )
        pca_applied = True
        pca_source = str(preprocessing_stats_path.resolve())
    elif args.no_train and args.pca_components > 0:
        raise ValueError(
            f"--no-train with --pca-components={args.pca_components} requires saved PCA "
            f"entries in {preprocessing_stats_path}."
        )
    elif args.pca_components > 0:
        dataset_train["x"], dataset_val["x"], obs_l1_std, pca_model = fit_pca(
            dataset_train["x"],
            dataset_val["x"],
            obs_l1_std,
            n_components=args.pca_components,
        )
        pca_applied = True
        pca_source = "fitted_current_run"
        print(f"  Train x shape after PCA = {dataset_train['x'].shape}")

    # 4b) Zero-variance feature filtering on final summary features
    if args.no_train:
        if not feature_mask_path.exists():
            raise FileNotFoundError(
                f"--no-train requires feature mask at {feature_mask_path}."
            )
        with np.load(feature_mask_path, allow_pickle=False) as saved:
            valid_mask = np.asarray(saved["valid_mask"], dtype=bool)
        if valid_mask.ndim != 1:
            raise ValueError(f"Saved valid_mask must be 1D, got {valid_mask.shape}.")
        if valid_mask.shape[0] != dataset_train["x"].shape[1]:
            raise ValueError(
                "Saved valid_mask dimension does not match current processed summary "
                f"dim ({valid_mask.shape[0]} vs {dataset_train['x'].shape[1]})."
            )
        print(f"  Loaded feature mask from {feature_mask_path}")
    else:
        valid_mask, n_removed = filter_zero_variance_bins(
            dataset_train["x"],
            min_variance=args.min_feature_variance,
            verbose=True,
        )
        if int(np.sum(valid_mask)) == 0:
            raise ValueError(
                "All features were removed by variance filtering. "
                "Lower --min-feature-variance or inspect L1 preprocessing."
            )
        save_path.mkdir(parents=True, exist_ok=True)
        np.savez(
            feature_mask_path,
            valid_mask=valid_mask,
            min_variance=np.array(args.min_feature_variance, dtype=np.float64),
            n_removed=np.array(n_removed, dtype=np.int64),
        )
        print(f"  Saved feature mask to {feature_mask_path}")

    dataset_train["x"] = dataset_train["x"][:, valid_mask]
    dataset_val["x"] = dataset_val["x"][:, valid_mask]
    obs_l1_std = obs_l1_std[valid_mask]
    print(f"  Final summary_dim used by jaxili NPE = {dataset_train['x'].shape[1]}")

    validate_npe_inputs(dataset_train, dataset_val, obs_l1_std, args.n_cosmo)

    # 5) Train or load jaxili NPE
    save_path.mkdir(parents=True, exist_ok=True)
    theta_train = jnp.asarray(dataset_train["theta"])
    x_train = jnp.asarray(dataset_train["x"])
    metrics = None
    loaded_checkpoint_dir: Optional[Path] = None
    if args.no_train:
        try:
            loaded_checkpoint_dir = _resolve_latest_jaxili_checkpoint_dir(checkpoint_path)
            _normalize_jaxili_hparams_embedding_arrays(loaded_checkpoint_dir)
            exmp_input = (
                jnp.zeros((1, args.n_cosmo), dtype=jnp.float32),
                jnp.zeros((1, int(dataset_train["x"].shape[1])), dtype=jnp.float32),
            )
            inference = NPE.load_from_checkpoints(
                checkpoint=str(loaded_checkpoint_dir),
                exmp_input=exmp_input,
            )
            print(f"  Loaded jaxili checkpoint from {loaded_checkpoint_dir}")
        except Exception as exc:
            raise FileNotFoundError(
                f"Could not load checkpoint '{checkpoint_path}'. "
                "Run without --no-train first."
            ) from exc
    else:
        inference = NPE()
        inference = inference.append_simulations(theta_train, x_train, key=split_key)
        if args.no_sample and args.epochs < 2:
            print(
                "  --epochs < 2 can fail in jaxili scheduler setup; "
                "consider --epochs >= 2 for smoke runs."
            )
        inference, metrics, _ = train_with_nan_retry(
            inference=inference,
            checkpoint_path=str(checkpoint_path),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            warmup_steps=args.npe_warmup_steps,
            decay_steps=args.npe_decay_steps,
            params=theta_train,
            data=x_train,
            split_key=split_key,
            max_retries=args.nan_retries,
        )
        training_summary = {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "nan_retries": int(args.nan_retries),
            "npe_warmup_steps": int(args.npe_warmup_steps),
            "npe_decay_steps": int(args.npe_decay_steps),
            "npe_split_seed": split_seed,
            "checkpoint_path": str(checkpoint_path),
        }
        if metrics is not None:
            training_summary.update(_metrics_summary(metrics))
        training_summary_path.write_text(
            json.dumps(training_summary, indent=2),
            encoding="utf-8",
        )
        print(f"  Saved training summary to {training_summary_path}")

        save_dict = {
            "mean": mean,
            "std": std,
            "summary_transform": np.array(effective_transform),
            "clip_value": np.array(
                np.nan if effective_clip_val is None else float(effective_clip_val),
                dtype=np.float64,
            ),
        }
        if pca_model is not None:
            save_dict["pca_components"] = pca_model.components_
            save_dict["pca_mean"] = pca_model.mean_
            save_dict["pca_explained_variance"] = pca_model.explained_variance_
        np.savez(preprocessing_stats_path, **save_dict)
        print(f"  Saved preprocessing stats to {preprocessing_stats_path}")

    # 6) Posterior sampling
    if args.no_sample:
        print("  Skipping posterior sampling (--no-sample).")
        return

    posterior = inference.build_posterior()
    sample_key, _ = jax.random.split(rng_sample)
    samples = posterior.sample(
        x=jnp.asarray(obs_l1_std),
        num_samples=args.npe_samples,
        key=sample_key,
    )
    samples_np = np.asarray(samples)
    finite_mask = np.all(np.isfinite(samples_np), axis=1)
    samples_np = samples_np[finite_mask]
    if samples_np.shape[0] == 0:
        raise FloatingPointError("All posterior samples are non-finite.")

    out = Path(args.posterior_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, samples_np)
    print(f"  Saved posterior samples → {out.resolve()}")

    metadata = {
        "method": "l1norm_jaxili",
        "posterior_file": str(out.resolve()),
        "checkpoint_path": (
            str(loaded_checkpoint_dir)
            if loaded_checkpoint_dir is not None
            else str(checkpoint_path)
        ),
        "preprocessing_stats_source": str(preprocessing_stats_path.resolve()),
        "feature_mask_source": str(feature_mask_path.resolve()),
        "training_summary_source": (
            str(training_summary_path.resolve()) if training_summary_path.exists() else None
        ),
        "l1_implementation": args.l1_implementation,
        "summary_transform": effective_transform,
        "summary_clip_value": None if effective_clip_val is None else float(effective_clip_val),
        "pca_applied": bool(pca_applied),
        "pca_source": pca_source,
        "npe_samples_requested": int(args.npe_samples),
        "npe_samples_finite": int(samples_np.shape[0]),
        "npe_epochs": int(args.epochs),
        "npe_batch_size": int(args.batch_size),
        "npe_learning_rate": float(args.learning_rate),
        "npe_warmup_steps": int(args.npe_warmup_steps),
        "npe_decay_steps": int(args.npe_decay_steps),
        "npe_split_seed": split_seed,
        "min_feature_variance": float(args.min_feature_variance),
        "truth_parameters": [float(v) for v in np.asarray(truth).ravel()],
        "tomo_bin_indices": list(tomo_bin_indices),
        "tfds_name": args.tfds_name,
        "map_kind": args.map_kind,
    }
    out.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    if args.plot:
        fig_out = Path(args.figure_out)
        fig_out.parent.mkdir(parents=True, exist_ok=True)
        param_names = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]
        plot_posterior(
            samples_np,
            np.asarray(truth),
            str(fig_out),
            param_names=param_names,
            log_to_wandb=False,
        )

    print("Done.")


if __name__ == "__main__":
    main()
