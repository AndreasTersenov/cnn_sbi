#!/usr/bin/env python
"""
CNN-compressed summaries + jaxili NPE for tomographic weak-lensing inference.

This script reuses the existing CNN compressor workflow from cnn_sbi and swaps
only the downstream density estimator to jaxili.NPE.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import pickle
import re
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import wandb
from jax.lib import xla_bridge

from bnt_utils import (
    BNT_MATRIX_VERSION,
    apply_bnt_numpy,
    apply_bnt_tf,
    validate_bnt_configuration,
)

def _ensure_tfds_builder_registered() -> None:
    """Import local TFDS builder lazily to avoid hard dependency at --help time."""
    try:
        importlib.import_module("tf_dataset_nbody_tomo")
    except ModuleNotFoundError as exc:
        if exc.name == "tensorflow_datasets":
            raise ModuleNotFoundError(
                "Missing dependency 'tensorflow_datasets'. Install it in the "
                "environment to run dataset compression."
            ) from exc
        raise


def _import_haiku():
    try:
        import haiku as hk
    except ImportError as exc:  # pragma: no cover - runtime env dependent
        raise ImportError(
            "Failed to import haiku. Install it in your environment, e.g. "
            "`pip install dm-haiku`."
        ) from exc
    return hk


def _import_npe_class():
    try:
        from jaxili.inference import NPE
    except ImportError as exc:  # pragma: no cover - runtime env dependent
        raise ImportError(
            "Failed to import jaxili. Activate the conda environment 'jaxili' "
            "before running this script."
        ) from exc
    return NPE


def _new_inference():
    return _import_npe_class()()


def _resolve_latest_jaxili_checkpoint_dir(checkpoint_root: Path) -> Path:
    """Resolve latest jaxili checkpoint subdir compatible with load_from_checkpoints."""
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CNN-compressed summaries + jaxili NPE for tomographic weak lensing"
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
    p.add_argument("--sigma-e", type=float, default=0.26, help="Shape noise dispersion")
    p.add_argument(
        "--galaxy-density",
        type=float,
        default=30 / 4,
        help="Galaxy number density [arcmin^{-2}]",
    )
    p.add_argument("--nbins", type=int, default=4, help="Number of tomographic bins")
    p.add_argument("--n-cosmo", type=int, default=6, help="Number of cosmological parameters")

    # Map kind
    p.add_argument(
        "--map-kind",
        type=str,
        default="nbody",
        choices=["nbody", "nbody_with_baryon_ia", "gaussian"],
    )

    # Compressor configuration
    p.add_argument("--compressor-dim", type=int, default=6, help="CNN compressor output dim")
    p.add_argument(
        "--compressor-params",
        type=str,
        default=(
            "/home/tersenov/software/cnn_sbi/tomo/save_params/"
            "vmim/nbody/sigma_0.26/gal_density_30/bin_4/"
            "params_nd_compressor_batch150000.pkl"
        ),
        help="Path to pretrained compressor params pickle",
    )
    p.add_argument(
        "--compressor-state",
        type=str,
        default=(
            "/home/tersenov/software/cnn_sbi/tomo/save_params/"
            "vmim/nbody/sigma_0.26/gal_density_30/bin_4/"
            "opt_state_resnet_batch150000.pkl"
        ),
        help="Path to pretrained compressor state pickle",
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
    p.add_argument("--posterior-out", type=str, default="posterior_cnn_jaxili_tomo.npy")
    p.add_argument("--figure-out", type=str, default="posterior_cnn_jaxili_tomo.pdf")
    p.add_argument("--cache-dir", type=str, default=None)
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

    # Summary preprocessing (compressed summary)
    p.add_argument("--standardize-summary", dest="standardize_summary", action="store_true")
    p.add_argument("--no-standardize-summary", dest="standardize_summary", action="store_false")
    p.set_defaults(standardize_summary=True)
    p.add_argument(
        "--summary-clip-value",
        type=float,
        default=5.0,
        help="Clip standardized summary features to ±this value (0 = disabled)",
    )
    p.add_argument(
        "--min-feature-variance",
        type=float,
        default=1e-8,
        help="Minimum variance threshold for zero-variance feature filtering",
    )

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
    p.add_argument("--nan-retries", type=int, default=10, help="Max retries on NaN losses")
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
        default="params_cnn_jaxili",
        help="Checkpoint basename under save_dir/cnn_jaxili/<map-kind>",
    )

    # Compatibility no-op args (used by existing sweep runner)
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
    p.add_argument("--wandb-project", type=str, default="cnn-jaxili-npe-tomo")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Optional W&B group; defaults to method/map/BNT grouping.",
    )
    p.add_argument(
        "--wandb-tags",
        type=str,
        default="",
        help="Additional comma-separated W&B tags.",
    )
    p.add_argument(
        "--ds-batch-size",
        type=int,
        default=500,
        help="Batch size for CNN compression of TFDS datasets",
    )
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")

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


def setup_environment(cuda_devices: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    print(f"CUDA_VISIBLE_DEVICES = {cuda_devices}")
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"TF GPU config: {exc}")
    print(f"JAX backend    : {xla_bridge.get_backend().platform}")


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


def build_compressor(dim: int):
    hk = _import_haiku()

    class CompressorCNN2D(hk.Module):
        """CNN compressor: (B, H, W, nbins) -> (B, output_dim)."""

        def __init__(self, output_dim: int, name: str | None = None):
            super().__init__(name=name)
            self.output_dim = output_dim

        def __call__(self, x):
            net_x = hk.Conv2D(32, 3, 2)(x)
            net_x = jax.nn.leaky_relu(net_x)
            net_x = hk.Conv2D(64, 3, 2)(net_x)
            net_x = jax.nn.leaky_relu(net_x)
            net_x = hk.Conv2D(128, 3, 2)(net_x)
            net_x = jax.nn.leaky_relu(net_x)
            net_x = hk.AvgPool(16, 8, "SAME")(net_x)
            net_x = hk.Flatten()(net_x)
            net_x = hk.Linear(64)(net_x)
            net_x = jax.nn.leaky_relu(net_x)
            net_x = hk.Linear(self.output_dim)(net_x)
            return net_x.squeeze()

    return hk.transform_with_state(lambda y: CompressorCNN2D(dim)(y))


def load_compressor_params(params_path: str, state_path: str) -> Tuple[Any, Any]:
    print(f"  Loading compressor params: {params_path}")
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    print(f"  Loading compressor state:  {state_path}")
    with open(state_path, "rb") as f:
        state = pickle.load(f)
    return params, state


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_cnn_cache_metadata(
    args: argparse.Namespace,
    compressor_params_path: str,
    compressor_state_path: str,
    tomo_bin_indices: tuple[int, ...],
) -> Dict[str, object]:
    params_path = Path(compressor_params_path).resolve()
    state_path = Path(compressor_state_path).resolve()
    return {
        "compressor_dim": int(args.compressor_dim),
        "tfds_name": str(args.tfds_name),
        "tomo_bin_indices": ",".join(str(b) for b in tomo_bin_indices),
        "map_kind": str(args.map_kind),
        "field_size": int(args.field_size),
        "field_npix": int(args.field_npix),
        "nbins": int(args.nbins),
        "sigma_e": float(args.sigma_e),
        "galaxy_density": float(args.galaxy_density),
        "apply_bnt": bool(args.apply_bnt),
        "bnt_matrix_version": BNT_MATRIX_VERSION if args.apply_bnt else "none",
        "compressor_params_path": str(params_path),
        "compressor_state_path": str(state_path),
        "compressor_params_sha256": file_sha256(params_path),
        "compressor_state_sha256": file_sha256(state_path),
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
        if isinstance(expected_value, float):
            cached_value = float(cached_raw)
            if abs(cached_value - expected_value) > 1e-12:
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


def compress_dataset(
    tfds_name: str,
    split: str,
    augmentation_fn,
    compressor,
    comp_params: Any,
    comp_state: Any,
    ds_batch_size: int,
) -> Dict[str, np.ndarray]:
    _ensure_tfds_builder_registered()
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
    for example in ds.as_numpy_iterator():
        maps_np = example["maps"]
        theta_np = example["theta"]
        if np.isnan(maps_np).any():
            print("    [!] Skipped batch with NaN maps")
            continue
        comp_y, _ = compressor.apply(comp_params, comp_state, None, maps_np)
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
        train_x, val_x, obs_x, mean, std, clip_value=clip_value
    )
    return train_std, val_std, obs_std, mean, std


def filter_zero_variance_bins(
    data: np.ndarray,
    min_variance: float = 1e-8,
    verbose: bool = True,
) -> tuple[np.ndarray, int]:
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


def _extract_metric(metrics: object, keys: list[str]):
    if isinstance(metrics, dict):
        for key in keys:
            if key in metrics:
                return metrics[key]
        return None
    for key in keys:
        if hasattr(metrics, key):
            return getattr(metrics, key)
    return None


def train_with_nan_retry(
    inference: Any,
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

            train_metric = _extract_metric(metrics, ["train/loss", "train_loss"])
            val_metric = _extract_metric(metrics, ["val/loss", "val_loss"])
            if train_metric is not None and _metric_has_nan(train_metric):
                print("  NaN detected in training loss; reinitializing inference object.")
                inference = _new_inference()
                inference = inference.append_simulations(params, data, key=split_key)
                continue
            if val_metric is not None and _metric_has_nan(val_metric):
                print("  NaN detected in validation loss; reinitializing inference object.")
                inference = _new_inference()
                inference = inference.append_simulations(params, data, key=split_key)
                continue
            print("  Training completed successfully.")
            return inference, metrics, density_estimator
        except Exception as exc:
            print(f"  Training attempt failed: {exc}")
            if attempt == max_retries:
                raise
            inference = _new_inference()
            inference = inference.append_simulations(params, data, key=split_key)

    raise RuntimeError("Training failed after exhausting NaN-retry budget.")


def validate_npe_inputs(
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
                f"{split_name} theta second dim must be {n_cosmo}, got {theta.shape[1]}."
            )
        if x.shape[1] <= 0:
            raise ValueError(f"{split_name} summary dim must be positive.")
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
    summary: dict[str, float | bool] = {}
    if isinstance(metrics, dict):
        for key in ("train/loss", "val/loss", "test/loss"):
            if key not in metrics:
                continue
            arr = np.asarray(metrics[key], dtype=np.float64)
            if arr.ndim == 0:
                summary[f"{key}_last"] = float(arr)
                summary[f"{key}_nan"] = bool(np.isnan(arr))
            elif arr.size > 0:
                summary[f"{key}_last"] = float(arr.ravel()[-1])
                summary[f"{key}_min"] = float(np.nanmin(arr))
                summary[f"{key}_nan"] = bool(np.any(np.isnan(arr)))
        return summary

    for key in ("train_loss", "val_loss", "test_loss"):
        if not hasattr(metrics, key):
            continue
        arr = np.asarray(getattr(metrics, key), dtype=np.float64)
        if arr.ndim == 0:
            summary[f"{key}_last"] = float(arr)
            summary[f"{key}_nan"] = bool(np.isnan(arr))
        elif arr.size > 0:
            summary[f"{key}_last"] = float(arr.ravel()[-1])
            summary[f"{key}_min"] = float(np.nanmin(arr))
            summary[f"{key}_nan"] = bool(np.any(np.isnan(arr)))
    return summary


def compute_fom3(samples: np.ndarray) -> dict[str, float | bool]:
    if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] < 3:
        return {
            "fom3": float("nan"),
            "det_cov3": float("nan"),
            "logdet_cov3": float("nan"),
            "valid_fom3": False,
        }
    cov3 = np.cov(samples[:, :3], rowvar=False)
    sign, logdet = np.linalg.slogdet(cov3)
    if sign <= 0:
        return {
            "fom3": float("nan"),
            "det_cov3": float("nan"),
            "logdet_cov3": float(logdet),
            "valid_fom3": False,
        }
    return {
        "fom3": float(np.exp(-0.5 * logdet)),
        "det_cov3": float(np.exp(logdet)),
        "logdet_cov3": float(logdet),
        "valid_fom3": True,
    }


def plot_posterior(
    samples: np.ndarray,
    truth: np.ndarray,
    output_path: str,
    param_names: list[str] | None = None,
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

    setup_environment(args.cuda_visible_devices)
    rng = jax.random.PRNGKey(args.seed)
    rng, rng_obs, rng_sample = jax.random.split(rng, 3)
    split_seed = int(args.seed) + 1
    split_key = jax.random.PRNGKey(split_seed)

    wandb_enabled = not args.no_wandb
    wandb_group = (
        args.wandb_group
        if args.wandb_group
        else f"cnn-jaxili-{args.map_kind}-{'bnt' if args.apply_bnt else 'nobnt'}-bins{args.nbins}"
    )
    extra_wandb_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    wandb_tags = [
        "cnn",
        "jaxili",
        args.map_kind,
        f"bnt:{int(args.apply_bnt)}",
        f"tomo:{','.join(str(b) for b in tomo_bin_indices)}",
        *extra_wandb_tags,
    ]
    if wandb_enabled:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            group=wandb_group,
            tags=wandb_tags,
            config=vars(args),
        )

    summary_dim = args.compressor_dim
    print(f"  summary_dim    = {summary_dim}")

    save_path = (Path(args.save_dir) / "cnn_jaxili" / args.map_kind).resolve()
    summary_stats_path = save_path / "cnn_jaxili_summary_standardization.npz"
    feature_mask_path = save_path / "cnn_jaxili_feature_mask.npz"
    training_summary_path = save_path / "jaxili_training_summary.json"
    checkpoint_path = (save_path / args.checkpoint_name).resolve()

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

    # Compressor
    print("######## CNN COMPRESSOR ########")
    compressor = build_compressor(args.compressor_dim)
    comp_params, comp_state = load_compressor_params(
        args.compressor_params, args.compressor_state
    )
    comp_params_path = Path(args.compressor_params).resolve()
    comp_state_path = Path(args.compressor_state).resolve()
    print(f"  Compressor params sha256: {file_sha256(comp_params_path)[:16]}...")
    print(f"  Compressor state  sha256: {file_sha256(comp_state_path)[:16]}...")

    # Compress observed map
    print("######## COMPRESS: OBSERVED MAP ########")
    obs_compressed, _ = compressor.apply(
        comp_params,
        comp_state,
        None,
        m_data.reshape([1, args.field_npix, args.field_npix, args.nbins]),
    )
    obs_compressed = np.array(obs_compressed).reshape(-1)
    print(f"  Observed compressed shape = {obs_compressed.shape}")

    # Compress train/test datasets
    print("######## COMPRESS: DATASETS ########")
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

    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else None
    cache_ok = False
    cache_meta_expected = build_cnn_cache_metadata(
        args=args,
        compressor_params_path=args.compressor_params,
        compressor_state_path=args.compressor_state,
        tomo_bin_indices=tomo_bin_indices,
    )
    if cache_dir is not None and cache_dir.exists():
        train_cache = cache_dir / "cnn_train.npz"
        val_cache = cache_dir / "cnn_val.npz"
        meta_cache = cache_dir / "cnn_cache_meta.npz"
        if train_cache.exists() and val_cache.exists() and meta_cache.exists():
            meta = np.load(meta_cache)
            cache_ok, mismatches = compare_cache_metadata(meta, cache_meta_expected)
            if cache_ok:
                print("  Loading cached compressed datasets (metadata matches) ...")
                d_tr = np.load(train_cache)
                dataset_train = {"theta": d_tr["theta"], "x": d_tr["x"]}
                d_va = np.load(val_cache)
                dataset_val = {"theta": d_va["theta"], "x": d_va["x"]}
            else:
                print(
                    "  CNN cache metadata mismatch; recomputing compressed datasets. "
                    f"First mismatch: {mismatches[0]}"
                )

    if not cache_ok:
        dataset_train = compress_dataset(
            args.tfds_name,
            "train",
            augmentation,
            compressor,
            comp_params,
            comp_state,
            args.ds_batch_size,
        )
        dataset_val = compress_dataset(
            args.tfds_name,
            "test",
            augmentation,
            compressor,
            comp_params,
            comp_state,
            args.ds_batch_size,
        )
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_dir / "cnn_train.npz",
                theta=dataset_train["theta"],
                x=dataset_train["x"],
            )
            np.savez(
                cache_dir / "cnn_val.npz",
                theta=dataset_val["theta"],
                x=dataset_val["x"],
            )
            np.savez(cache_dir / "cnn_cache_meta.npz", **cache_meta_expected)
            print(f"  Cached datasets to {cache_dir}")

    print(f"  Train x shape = {dataset_train['x'].shape}")
    print(f"  Val   x shape = {dataset_val['x'].shape}")

    # Summary preprocessing
    standardization_applied = False
    summary_clip_value = (
        args.summary_clip_value if args.summary_clip_value > 0 else None
    )
    standardization_mean: Optional[np.ndarray] = None
    standardization_std: Optional[np.ndarray] = None
    if args.standardize_summary:
        if args.no_train:
            if not summary_stats_path.exists():
                raise FileNotFoundError(
                    f"--no-train requires summary standardization stats at "
                    f"{summary_stats_path}."
                )
            std_data = np.load(summary_stats_path)
            standardization_mean = np.array(std_data["mean"])
            standardization_std = np.array(std_data["std"])
            if "clip_value" in std_data.files:
                loaded_clip = float(std_data["clip_value"])
                summary_clip_value = loaded_clip if np.isfinite(loaded_clip) and loaded_clip > 0 else None
            dataset_train["x"], dataset_val["x"], obs_compressed = apply_summary_standardization(
                dataset_train["x"],
                dataset_val["x"],
                obs_compressed,
                standardization_mean,
                standardization_std,
                clip_value=summary_clip_value,
            )
            standardization_applied = True
            print(f"  Loaded summary standardization stats from {summary_stats_path}")
        else:
            (
                dataset_train["x"],
                dataset_val["x"],
                obs_compressed,
                standardization_mean,
                standardization_std,
            ) = fit_and_apply_summary_standardization(
                dataset_train["x"],
                dataset_val["x"],
                obs_compressed,
                clip_value=summary_clip_value,
            )
            standardization_applied = True
            print(
                "  Applied summary standardization "
                f"(clip={summary_clip_value if summary_clip_value is not None else 'off'})."
            )

    # Feature filtering
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
                "Lower --min-feature-variance or inspect compressed summaries."
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
    obs_compressed = obs_compressed[valid_mask]
    print(f"  Final summary_dim used by jaxili NPE = {dataset_train['x'].shape[1]}")

    validate_npe_inputs(dataset_train, dataset_val, obs_compressed, args.n_cosmo)
    if wandb_enabled:
        wandb.log(
            {
                "data/train_size": int(dataset_train["theta"].shape[0]),
                "data/val_size": int(dataset_val["theta"].shape[0]),
                "data/summary_dim": int(dataset_train["x"].shape[1]),
                "data/summary_standardized": int(standardization_applied),
                "data/apply_bnt": int(args.apply_bnt),
                "data/min_feature_variance": float(args.min_feature_variance),
            }
        )

    # Train/load jaxili NPE
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
            NPEClass = _import_npe_class()
            inference = NPEClass.load_from_checkpoints(
                checkpoint=str(loaded_checkpoint_dir),
                exmp_input=exmp_input,
            )
            print(f"  Loaded jaxili checkpoint from {loaded_checkpoint_dir}")
        except Exception as exc:
            raise FileNotFoundError(
                f"Could not load checkpoint from '{checkpoint_path}'. "
                "Run without --no-train first."
            ) from exc
    else:
        inference = _new_inference()
        inference = inference.append_simulations(theta_train, x_train, key=split_key)
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
        if wandb_enabled:
            wandb.log(
                {
                    "train/epochs": int(args.epochs),
                    "train/batch_size": int(args.batch_size),
                    "train/learning_rate": float(args.learning_rate),
                    **{
                        f"train/{k}": v
                        for k, v in training_summary.items()
                        if isinstance(v, (int, float, bool))
                    },
                }
            )

        if standardization_applied and standardization_mean is not None and standardization_std is not None:
            np.savez(
                summary_stats_path,
                mean=standardization_mean,
                std=standardization_std,
                clip_value=(
                    np.nan if summary_clip_value is None else float(summary_clip_value)
                ),
            )
            print(f"  Saved summary standardization stats to {summary_stats_path}")

    # Posterior sampling
    if args.no_sample:
        print("  Skipping posterior sampling (--no-sample).")
        if wandb_enabled:
            wandb.finish()
        return

    posterior = inference.build_posterior()
    sample_key, _ = jax.random.split(rng_sample)
    samples = posterior.sample(
        x=jnp.asarray(obs_compressed),
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
    fom3 = compute_fom3(samples_np)
    fom_out = out.with_suffix(".fom.json")
    fom_out.write_text(json.dumps(fom3, indent=2), encoding="utf-8")

    metadata = {
        "method": "cnn_jaxili",
        "posterior_file": str(out.resolve()),
        "fom_file": str(fom_out.resolve()),
        "checkpoint_path": (
            str(loaded_checkpoint_dir)
            if loaded_checkpoint_dir is not None
            else str(checkpoint_path)
        ),
        "summary_standardized": bool(standardization_applied),
        "summary_standardization_file": (
            str(summary_stats_path.resolve()) if summary_stats_path.exists() else None
        ),
        "feature_mask_source": str(feature_mask_path.resolve()),
        "training_summary_source": (
            str(training_summary_path.resolve()) if training_summary_path.exists() else None
        ),
        "compressor_params_path": str(comp_params_path),
        "compressor_state_path": str(comp_state_path),
        "compressor_params_sha256": file_sha256(comp_params_path),
        "compressor_state_sha256": file_sha256(comp_state_path),
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
        "apply_bnt": bool(args.apply_bnt),
        "bnt_matrix_version": BNT_MATRIX_VERSION if args.apply_bnt else "none",
        "compressor_dim": int(args.compressor_dim),
        "fom3": fom3,
    }
    out.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    if args.plot:
        fig_out = Path(args.figure_out)
        fig_out.parent.mkdir(parents=True, exist_ok=True)
        param_names = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]
        plot_posterior(samples_np, np.asarray(truth), str(fig_out), param_names=param_names)

    if wandb_enabled:
        wandb.log(
            {
                "posterior/n_samples_finite": int(samples_np.shape[0]),
                "posterior/fom3": float(fom3["fom3"]),
                "posterior/det_cov3": float(fom3["det_cov3"]),
                "posterior/logdet_cov3": float(fom3["logdet_cov3"]),
                "posterior/valid_fom3": int(bool(fom3["valid_fom3"])),
            }
        )
        if args.plot:
            try:
                wandb.log({"posterior/corner": wandb.Image(str(Path(args.figure_out)))})
            except Exception:
                pass
        wandb.finish()

    print("Done.")


if __name__ == "__main__":
    main()
