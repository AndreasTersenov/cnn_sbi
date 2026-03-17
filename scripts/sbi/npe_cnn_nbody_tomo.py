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
 6. Sample the posterior and produce contour plots

Requires:
  - Pretrained CNN compressor weights (from train_compressor_tomographic.py)
  - NbodyCosmogridDatasetTomo TFDS dataset (from tf_dataset_nbody_tomo.py)
  - sbi_lens (normalizing flow components)
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from functools import partial
from pathlib import Path
from typing import Dict, Tuple

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

# sbi_lens normalizing flow
from sbi_lens.normflow.models import AffineCoupling, ConditionalRealNVP
from sbi_lens.normflow.train_model import TrainModel

# Register the local TFDS dataset builder so tfds.load can find it
import tf_dataset_nbody_tomo as _tomo_builder  # noqa: F401, E402

tfb = tfp.bijectors
tfd = tfp.distributions


# =============================================================================
# CLI
# =============================================================================

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
    p.add_argument("--wandb-project", type=str, default="l1norm-npe-tomo",
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
    proj = hp.projector.GnomonicProj(
        rot=[0, 0, 0], xsize=field_npix, ysize=field_npix, reso=reso,
    )

    with h5py.File(fid_path, "r") as f:
        kg = f["kg"]
        proj_bins = []
        for b in range(1, nbins + 1):
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
    print(f"  Observed map shape = {m_data.shape}, "
          f"noise_std/pixel = {noise_std:.6f}")
    return m_data, cosmo_params, truth


# =============================================================================
# CNN Compressor (Haiku)
# =============================================================================

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


def build_compressor(dim: int):
    """Build the Haiku compressor transform."""
    return hk.transform_with_state(lambda y: CompressorCNN2D(dim)(y))


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
) -> Tuple[hk.Params, hk.State]:
    """Train the CNN compressor from scratch using VMIM loss.

    Follows the same recipe as train_compressor_tomographic.py:
      - Companion RealNVP (4 layers, [128]*2, silu) for VMIM objective
      - Piecewise constant LR schedule (init × 0.7 at every 10% milestone)
      - Adam optimizer
      - TrainModel from sbi_lens
    """
    import tensorflow_datasets as tfds
    from tqdm import tqdm

    print("######## TRAINING COMPRESSOR (VMIM) ########")
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Companion normalizing flow for VMIM ---
    bijector_fn = partial(
        AffineCoupling, layers=[128] * 2, activation=jax.nn.silu,
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

    # --- Streaming datasets ---
    ds_tr = tfds.load("NbodyCosmogridDatasetTomo/grid", split="train")
    ds_tr = ds_tr.repeat().shuffle(800)
    ds_tr = ds_tr.map(augmentation_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds_tr = ds_tr.batch(batch_size)
    ds_tr = ds_tr.prefetch(tf.data.AUTOTUNE)
    ds_train_iter = iter(tfds.as_numpy(ds_tr))

    ds_te = tfds.load("NbodyCosmogridDatasetTomo/grid", split="test")
    ds_te = ds_te.repeat().shuffle(200)
    ds_te = ds_te.map(augmentation_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds_te = ds_te.batch(batch_size)
    ds_te = ds_te.prefetch(tf.data.AUTOTUNE)
    ds_test_iter = iter(tfds.as_numpy(ds_te))

    # --- Training loop ---
    store_loss = []
    loss_train_hist = []
    loss_test_hist = []

    for step in tqdm(range(1, total_steps + 1), desc="Compressor"):
        ex = next(ds_train_iter)
        if jnp.isnan(ex["maps"]).any():
            continue

        b_loss, params_merged, opt_state, state_cnn = update(
            model_params=params_merged,
            opt_state=opt_state,
            theta=ex["theta"],
            x=ex["maps"],
            state_resnet=state_cnn,
        )
        store_loss.append(float(b_loss))

        if jnp.isnan(b_loss):
            print("  [!] NaN loss — stopping compressor training")
            break

        # Log to wandb every 100 steps
        if step % 100 == 0:
            wandb.log({
                "compressor/train_loss": float(b_loss),
                "compressor/step": step,
            })

        if step % save_every == 0:
            # Save checkpoint
            ckpt_params = save_dir / f"params_nd_compressor_batch{step}.pkl"
            with open(ckpt_params, "wb") as f:
                pickle.dump(params_merged, f)
            ckpt_state = save_dir / f"opt_state_resnet_batch{step}.pkl"
            with open(ckpt_state, "wb") as f:
                pickle.dump(state_cnn, f)

            # Test loss
            ex_test = next(ds_test_iter)
            b_loss_test, _, _, _ = update(
                model_params=params_merged,
                opt_state=opt_state,
                theta=ex_test["theta"],
                x=ex_test["maps"],
                state_resnet=state_cnn,
            )
            loss_train_hist.append(float(b_loss))
            loss_test_hist.append(float(b_loss_test))

            wandb.log({
                "compressor/test_loss": float(b_loss_test),
                "compressor/step": step,
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

            # Quick contour plot from compressor + companion NF
            _plot_compressor_contours(
                compressor, params_merged, state_cnn, nf,
                m_data_obs, field_npix, nbins, n_cosmo, compressor_dim,
                truth, param_names, save_dir, step,
            )

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
):
    """Build the TF augmentation pipeline for the tomographic dataset."""
    noise_std = sigma_e / jnp.sqrt(
        galaxy_density * (field_size * 60 / field_npix) ** 2
    )

    map_key = {
        "nbody": "map_nbody",
        "nbody_with_baryon_ia": "map_nbody_w_baryon_ia",
        "gaussian": "map_gaussian",
    }[map_kind]

    def augmentation_noise(example):
        x = example[map_key]
        x += tf.random.normal(
            shape=(field_npix, field_npix, nbins), stddev=noise_std,
        )
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

    for example in ds.as_numpy_iterator():
        maps_np = example["maps"]   # (B, H, W, nbins)
        theta_np = example["theta"]  # (B, 6)

        # Skip any batch with NaNs
        if np.isnan(maps_np).any():
            print("    [!] Skipped batch with NaN maps")
            continue

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
    wandb.run.summary["best_val_loss"] = best_val_loss
    wandb.run.summary["final_step"] = step
    print(f"  Best validation loss: {best_val_loss:.4f}")
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
    setup_environment(args.cuda_visible_devices)
    rng = jax.random.PRNGKey(args.seed)
    rng, rng_obs, rng_sample = jax.random.split(rng, 3)

    # Derived quantities
    summary_dim = args.compressor_dim
    print(f"  summary_dim    = {summary_dim}")

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
    m_data, cosmo_params, truth = load_observed_map(
        args.cosmogrid_meta, args.fiducial_map,
        args.field_size, args.field_npix, args.nside, args.nbins,
        args.sigma_e, args.galaxy_density, rng_obs,
    )

    # ------------------------------------------------------------------
    # 2. CNN compressor
    # ------------------------------------------------------------------
    print("######## CNN COMPRESSOR ########")
    compressor = build_compressor(args.compressor_dim)

    augmentation = build_augmentation(
        args.map_kind, args.sigma_e, args.galaxy_density,
        args.field_size, args.field_npix, args.nbins,
    )

    if args.train_compressor:
        comp_save_dir = (
            Path(args.save_dir) / "vmim" / args.map_kind
            / f"sigma_{args.sigma_e}"
            / f"gal_density_{int(args.galaxy_density * 4)}"
            / f"bin_{args.nbins}"
        )
        comp_params, comp_state = train_compressor_vmim(
            compressor=compressor,
            augmentation_fn=augmentation,
            n_cosmo=args.n_cosmo,
            compressor_dim=args.compressor_dim,
            field_npix=args.field_npix,
            nbins=args.nbins,
            total_steps=args.compressor_steps,
            lr_init=args.compressor_lr,
            batch_size=args.compressor_batch_size,
            save_every=args.compressor_save_every,
            save_dir=comp_save_dir,
            m_data_obs=m_data,
            truth=truth,
            param_names=param_names,
        )
        # Invalidate any cached compressed datasets
        cache_dir = Path(args.cache_dir) if args.cache_dir else None
        if cache_dir is not None:
            for f in ["cnn_train.npz", "cnn_val.npz"]:
                p = cache_dir / f
                if p.exists():
                    p.unlink()
                    print(f"  Deleted stale cache: {p}")
    else:
        comp_params, comp_state = load_compressor_params(
            args.compressor_params, args.compressor_state,
        )

    # 2a. Compress observed map
    print("######## COMPRESS: OBSERVED MAP ########")
    obs_compressed, _ = compressor.apply(
        comp_params, comp_state, None,
        m_data.reshape([1, args.field_npix, args.field_npix, args.nbins]),
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

    if cache_dir is not None and cache_dir.exists():
        train_cache = cache_dir / "cnn_train.npz"
        val_cache = cache_dir / "cnn_val.npz"
        if train_cache.exists() and val_cache.exists():
            print("  Loading cached compressed datasets ...")
            d_tr = np.load(train_cache)
            dataset_train = {"theta": d_tr["theta"], "x": d_tr["x"]}
            d_va = np.load(val_cache)
            dataset_val = {"theta": d_va["theta"], "x": d_va["x"]}
            cache_ok = True
            print(
                f"  Train: {len(dataset_train['theta'])} | "
                f"Val: {len(dataset_val['theta'])}"
            )

    if not cache_ok:
        dataset_train = compress_dataset(
            "NbodyCosmogridDatasetTomo/grid", "train",
            augmentation, compressor, comp_params, comp_state,
            args.ds_batch_size,
        )
        dataset_val = compress_dataset(
            "NbodyCosmogridDatasetTomo/grid", "test",
            augmentation, compressor, comp_params, comp_state,
            args.ds_batch_size,
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

    # Log dataset statistics to wandb
    wandb.log({
        "data/train_size": len(dataset_train["theta"]),
        "data/val_size": len(dataset_val["theta"]),
        "data/summary_dim": summary_dim,
        "data/train_x_min": float(dataset_train["x"].min()),
        "data/train_x_max": float(dataset_train["x"].max()),
        "data/train_x_mean": float(dataset_train["x"].mean()),
        "data/train_x_std": float(dataset_train["x"].std()),
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

    save_path = Path(args.save_dir) / "cnn_vmim" / args.map_kind
    flow_params = None

    if args.no_train:
        # Prefer best model, fall back to latest checkpoint
        best_path = save_path / "params_cnn_flow_best.pkl"
        if best_path.exists():
            load_path = best_path
        else:
            candidates = sorted(
                save_path.glob("params_cnn_flow_batch*.pkl"),
            )
            if not candidates:
                raise FileNotFoundError(
                    f"No saved flow params in {save_path} "
                    f"and --no-train set"
                )
            load_path = candidates[-1]
        with open(load_path, "rb") as f:
            flow_params = pickle.load(f)
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

    # ------------------------------------------------------------------
    # 5. Posterior sampling
    # ------------------------------------------------------------------
    if not args.no_sample:
        posterior_samples = sample_posterior(
            rng_sample, nf_sample, flow_params,
            obs_compressed, args.npe_samples,
        )
        out = Path(args.posterior_out)
        np.save(out, posterior_samples)
        print(f"  Saved posterior samples → {out.resolve()}")

        if args.plot:
            plot_posterior(
                posterior_samples, truth, args.figure_out,
                param_names, log_to_wandb=(not args.no_wandb),
            )
    else:
        print("  Skipping posterior sampling (--no-sample)")

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
