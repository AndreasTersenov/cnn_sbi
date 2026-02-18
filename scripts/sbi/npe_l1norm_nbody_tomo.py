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
import os
import pickle
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from tensorflow_probability.substrates import jax as tfp

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
    p.add_argument("--l1-min-snr", type=float, default=None, help="Min SNR for L1-norm binning (None = auto)")
    p.add_argument("--l1-max-snr", type=float, default=None, help="Max SNR for L1-norm binning (None = auto)")
    p.add_argument("--subtract-coarse-mean", action="store_true", default=True,
                    help="Subtract coarse-scale mean before SNR (default: True)")

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

    # Flow training hyperparameters
    p.add_argument("--total-steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr-init", type=float, default=1e-3)
    p.add_argument("--lr-decay-rate", type=float, default=0.9)
    p.add_argument("--lr-transition-fraction", type=float, default=0.8)
    p.add_argument("--lr-end", type=float, default=1e-6)
    p.add_argument("--nvp-layers", type=int, default=6, help="Number of RealNVP coupling layers")
    p.add_argument("--nvp-hidden", type=int, default=256, help="Hidden layer width in coupling networks")
    p.add_argument("--seed", type=int, default=42)

    # Posterior sampling
    p.add_argument("--npe-samples", type=int, default=100_000)

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
    sigma_e: float,
    galaxy_density: float,
    rng_key: jax.Array,
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

    with h5py.File(fid_path, "r") as f:
        kg = f["kg"]
        proj_bins = []
        for b in range(1, nbins + 1):
            full_map = np.array(kg[f"stage3_lensing{b}"])
            patch = proj.projmap(full_map, vec2pix_func=partial(hp.vec2pix, nside))
            proj_bins.append(patch)

    # Stack to (H, W, nbins) and add shape noise
    m_data = np.stack(proj_bins, axis=-1).astype(np.float32)
    noise_std = pixel_noise_sigma(sigma_e, galaxy_density, field_size, field_npix)
    noise = jax.random.normal(rng_key, (field_npix, field_npix, nbins)) * noise_std
    m_data = np.array(jnp.asarray(m_data) + noise)
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


def compute_l1_single_map(
    kappa: np.ndarray,
    noise_sigma: float,
    stats: WLStatistics,
    l1_nbins: int,
    nbins: int,
    l1_min_snr: Optional[float] = None,
    l1_max_snr: Optional[float] = None,
    subtract_coarse_mean: bool = True,
) -> np.ndarray:
    """
    Compute L1-norm summary vector for a single (H, W, nbins) map.

    Returns shape (n_scales * l1_nbins * nbins,).
    """
    device = stats.device
    all_l1 = []
    for b in range(nbins):
        img = torch.from_numpy(kappa[:, :, b].astype(np.float64)).to(device)
        stats.compute_wavelet_transform(img, noise_sigma,
                                        subtract_coarse_mean=subtract_coarse_mean)
        _, l1_norms = stats.compute_wavelet_l1_norms(
            n_bins=l1_nbins, min_snr=l1_min_snr, max_snr=l1_max_snr
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
    l1_min_snr: Optional[float] = None,
    l1_max_snr: Optional[float] = None,
    subtract_coarse_mean: bool = True,
) -> np.ndarray:
    """
    Compute L1-norm summary vectors for a batch of (B, H, W, nbins) maps.

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
            n_bins=l1_nbins, min_snr=l1_min_snr, max_snr=l1_max_snr
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
):
    """Build the TF augmentation pipeline for the tomographic dataset."""
    noise_std = sigma_e / jnp.sqrt(galaxy_density * (field_size * 60 / field_npix) ** 2)

    map_key = {
        "nbody": "map_nbody",
        "nbody_with_baryon_ia": "map_nbody_w_baryon_ia",
        "gaussian": "map_gaussian",
    }[map_kind]

    def augmentation_noise(example):
        x = example[map_key]
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
    l1_min_snr: Optional[float] = None,
    l1_max_snr: Optional[float] = None,
    subtract_coarse_mean: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Load TFDS dataset, apply augmentation, compute L1-norm summaries.

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
# Summary standardization
# =============================================================================

def standardize(
    train_x: np.ndarray,
    val_x: np.ndarray,
    obs_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Zero-mean unit-variance standardization based on training set."""
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-12] = 1.0  # avoid division by zero for dead features
    return (
        (train_x - mean) / std,
        (val_x - mean) / std,
        (obs_x - mean) / std,
        mean,
        std,
    )


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
    decay_rate: float,
    end_lr: float,
    transition_fraction: float,
) -> hk.Params:
    """Train the conditional normalizing flow."""
    print("######## TRAINING FLOW ########")
    key_init, _ = jax.random.split(rng_key)

    # Initialise params — dummy inputs of correct shape
    theta_dummy = 0.5 * jnp.zeros([1, n_cosmo])
    y_dummy = jnp.zeros([1, summary_dim])
    params = nf_logp.init(key_init, theta_dummy, y_dummy)

    # LR schedule & optimizer
    nb_steps = int(total_steps * transition_fraction)
    lr_schedule = optax.exponential_decay(
        init_value=lr_init,
        transition_steps=max(nb_steps // 50, 1),
        decay_rate=decay_rate,
        end_value=end_lr,
    )
    optimizer = optax.adam(learning_rate=lr_schedule)
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

    for step in range(1, total_steps + 1):
        idx = np.random.randint(0, n_train, batch_size)
        loss, params, opt_state = update(params, opt_state, theta_train[idx], x_train[idx])
        batch_losses.append(float(loss))

        if step % 100 == 0:
            print(f"  Step {step:6d} | train loss {loss:.4f}")

        if step % save_every == 0 or step == total_steps:
            save_dir.mkdir(parents=True, exist_ok=True)
            with open(save_dir / f"params_l1norm_flow_batch{step}.pkl", "wb") as f:
                pickle.dump(params, f)
            # Quick validation
            vidx = np.random.randint(0, n_val, min(batch_size, n_val))
            val_l = -jnp.mean(nf_logp.apply(params, theta_val[vidx], x_val[vidx]))
            val_losses.append(float(val_l))
            print(f"  Saved @ step {step}. Val loss = {val_l:.4f}")

    # Save loss curves
    np.save(save_dir / "loss_train_l1norm.npy", np.array(batch_losses))
    np.save(save_dir / "loss_val_l1norm.npy", np.array(val_losses))
    return params


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

    # Sensible axis ranges
    param_limits = {}
    for i, name in enumerate(param_names):
        lo, hi = float(samples[:, i].min()), float(samples[:, i].max())
        tv = truth[i]
        margin = 0.05 * max(abs(tv), abs(hi - lo), 1e-3)
        param_limits[name] = (min(lo, tv) - margin, max(hi, tv) + margin)

    g = gplot.get_subplot_plotter(subplot_size=1.5)
    g.triangle_plot(
        [mcsamples], filled=True,
        markers=truth,
        marker_args={"color": "red", "lw": 1.2},
        param_limits=param_limits,
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved posterior plot → {output_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    torch_device = setup_environment(args.cuda_visible_devices)
    rng = jax.random.PRNGKey(args.seed)
    rng, rng_obs, rng_sample = jax.random.split(rng, 3)

    # Derived quantities
    pixel_arcmin = args.field_size * 60.0 / args.field_npix
    noise_sigma = pixel_noise_sigma(
        args.sigma_e, args.galaxy_density, args.field_size, args.field_npix
    )
    summary_dim = args.n_scales * args.l1_nbins * args.nbins
    print(f"  pixel_arcmin   = {pixel_arcmin:.2f}")
    print(f"  noise_sigma    = {noise_sigma:.6f}")
    print(f"  summary_dim    = {summary_dim}  "
          f"({args.n_scales} scales × {args.l1_nbins} bins × {args.nbins} tomo bins)")

    param_names = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]

    # ------------------------------------------------------------------
    # 1. Observed map
    # ------------------------------------------------------------------
    m_data, cosmo_params, truth = load_observed_map(
        args.cosmogrid_meta, args.fiducial_map,
        args.field_size, args.field_npix, args.nside, args.nbins,
        args.sigma_e, args.galaxy_density, rng_obs,
    )

    # ------------------------------------------------------------------
    # 2. L1-norm computer
    # ------------------------------------------------------------------
    stats = build_l1_computer(args.n_scales, pixel_arcmin, torch_device)

    # 2a. L1-norm for observed map
    print("######## L1-NORM: OBSERVED MAP ########")
    obs_l1 = compute_l1_single_map(
        m_data, noise_sigma, stats, args.l1_nbins, args.nbins,
        l1_min_snr=args.l1_min_snr, l1_max_snr=args.l1_max_snr,
        subtract_coarse_mean=args.subtract_coarse_mean,
    )
    print(f"  Observed L1-norm vector shape = {obs_l1.shape}")

    # ------------------------------------------------------------------
    # 3. L1-norm for train / test datasets
    # ------------------------------------------------------------------
    print("######## L1-NORM: DATASETS ########")
    augmentation = build_augmentation(
        args.map_kind, args.sigma_e, args.galaxy_density,
        args.field_size, args.field_npix, args.nbins,
    )

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    cache_ok = False

    # Try loading from cache
    if cache_dir is not None and cache_dir.exists():
        train_cache = cache_dir / "l1_train.npz"
        val_cache = cache_dir / "l1_val.npz"
        if train_cache.exists() and val_cache.exists():
            print("  Loading cached L1-norm datasets ...")
            d_tr = np.load(train_cache)
            dataset_train = {"theta": d_tr["theta"], "x": d_tr["x"]}
            d_va = np.load(val_cache)
            dataset_val = {"theta": d_va["theta"], "x": d_va["x"]}
            cache_ok = True
            print(f"  Train: {len(dataset_train['theta'])} | Val: {len(dataset_val['theta'])}")

    if not cache_ok:
        dataset_train = compute_l1_dataset(
            "NbodyCosmogridDatasetTomo/grid", "train", augmentation, stats,
            noise_sigma, args.l1_nbins, args.nbins, args.ds_batch_size,
            l1_min_snr=args.l1_min_snr, l1_max_snr=args.l1_max_snr,
            subtract_coarse_mean=args.subtract_coarse_mean,
        )
        dataset_val = compute_l1_dataset(
            "NbodyCosmogridDatasetTomo/grid", "test", augmentation, stats,
            noise_sigma, args.l1_nbins, args.nbins, args.ds_batch_size,
            l1_min_snr=args.l1_min_snr, l1_max_snr=args.l1_max_snr,
            subtract_coarse_mean=args.subtract_coarse_mean,
        )
        # Save cache
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(cache_dir / "l1_train.npz",
                     theta=dataset_train["theta"], x=dataset_train["x"])
            np.savez(cache_dir / "l1_val.npz",
                     theta=dataset_val["theta"], x=dataset_val["x"])
            print(f"  Cached datasets to {cache_dir}")

    print(f"  Train x shape = {dataset_train['x'].shape}")
    print(f"  Val   x shape = {dataset_val['x'].shape}")

    # ------------------------------------------------------------------
    # 4. Standardize summaries
    # ------------------------------------------------------------------
    print("######## STANDARDIZE ########")
    dataset_train["x"], dataset_val["x"], obs_l1_std, mean, std = standardize(
        dataset_train["x"], dataset_val["x"], obs_l1,
    )
    print(f"  Summary mean range = [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  Summary std  range = [{std.min():.4f}, {std.max():.4f}]")

    # ------------------------------------------------------------------
    # 5. Build & train flow
    # ------------------------------------------------------------------
    nf_logp, nf_sample = build_flow(
        n_cosmo_params=args.n_cosmo,
        n_layers=args.nvp_layers,
        hidden=args.nvp_hidden,
    )

    save_path = Path(args.save_dir) / "l1norm" / args.map_kind
    flow_params = None

    if args.no_train:
        candidates = sorted(save_path.glob("params_l1norm_flow_batch*.pkl"))
        if not candidates:
            raise FileNotFoundError(f"No saved flow params in {save_path} and --no-train set")
        with open(candidates[-1], "rb") as f:
            flow_params = pickle.load(f)
        print(f"  Loaded flow params from {candidates[-1]}")
    else:
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
            decay_rate=args.lr_decay_rate,
            end_lr=args.lr_end,
            transition_fraction=args.lr_transition_fraction,
        )

    # Save standardization constants alongside flow params
    save_path.mkdir(parents=True, exist_ok=True)
    np.savez(save_path / "l1_standardization.npz", mean=mean, std=std)

    # ------------------------------------------------------------------
    # 6. Posterior sampling
    # ------------------------------------------------------------------
    if not args.no_sample:
        posterior_samples = sample_posterior(
            rng_sample, nf_sample, flow_params,
            obs_l1_std, args.npe_samples,
        )
        out = Path(args.posterior_out)
        np.save(out, posterior_samples)
        print(f"  Saved posterior samples → {out.resolve()}")

        if args.plot:
            plot_posterior(posterior_samples, truth, args.figure_out, param_names)
    else:
        print("  Skipping posterior sampling (--no-sample)")

    print("Done.")


if __name__ == "__main__":
    main()
