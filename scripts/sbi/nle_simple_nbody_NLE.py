#!/usr/bin/env python
"""
End-to-end Normalizing Flow based SBI example converted from the Jupyter notebook
`nle_simple_nbody.ipynb`.

Main stages:
 1. (Optional) Set CUDA device
 2. Load observed (fiducial) map and add shape noise
 3. Build & load pretrained CNN compressor (Haiku)
 4. Data augmentation + compression of TFDS cosmology maps (train/test)
 5. Define & train conditional RealNVP normalizing flow p(theta | y)
 6. Likelihood sampling & posterior inference with HMC (TFP-JAX)
 7. (Optional) Produce diagnostic / triangle plots & save posterior samples

This script adds CLI arguments, modularizes the code and removes notebook-only constructs.

NOTE: Requires availability of the custom dataset (e.g. NbodyCosmogridDataset) and the
pretrained compressor parameter pickle files produced previously.
"""
from __future__ import annotations

import os
import argparse
import pickle
from functools import partial
from pathlib import Path
from typing import Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.lib import xla_bridge
import healpy as hp
import h5py
import tensorflow as tf
from tensorflow_probability.substrates import jax as tfp

# External / custom imports
# sbi_lens modules (normalizing flow components)
from sbi_lens.normflow.models import AffineCoupling, ConditionalRealNVP  # type: ignore

# -----------------------------------------------------------------------------
# Utilities & Configuration
# -----------------------------------------------------------------------------

tfb = tfp.bijectors
tfd = tfp.distributions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalizing Flow SBI for cosmological parameters")
    # Environment / hardware
    parser.add_argument("--cuda-visible-devices", type=str, default="0", help="CUDA_VISIBLE_DEVICES value")

    # Data & model configuration
    parser.add_argument("--map-kind", type=str, default="nbody", choices=["nbody", "nbody_with_baryon_ia", "gaussian"], help="Type of simulation maps")
    parser.add_argument("--loss-name", type=str, default="vmim", help="Name of the loss directory for saved params")
    parser.add_argument("--field-size", type=int, default=10, help="Field size (deg)")
    parser.add_argument("--field-npix", type=int, default=80, help="Number of pixels per side in projected map")
    parser.add_argument("--nside", type=int, default=512, help="Healpix NSIDE of full-sky maps")
    parser.add_argument("--sigma-e", type=float, default=0.26, help="Shape noise dispersion")
    parser.add_argument("--galaxy-density", type=float, default=10/4, help="Galaxy density per arcmin^2")
    parser.add_argument("--nbins", type=int, default=1, help="Number of tomographic bins (kept for compatibility)")
    parser.add_argument("--dim", type=int, default=6, help="Number of cosmological parameters")

    # Paths
    parser.add_argument("--cosmogrid-meta", type=str, default="/home/tersenov/CosmoGridV1/CosmoGridV1_metainfo.h5", help="Path to meta info HDF5")
    parser.add_argument("--fiducial-map", type=str, default="/home/tersenov/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial/perm_0002/projected_probes_maps_nobaryons512.h5", help="Path to fiducial maps HDF5")
    parser.add_argument("--compressor-param-dir", type=str, default="/home/tersenov/software/Learn2Map/scripts/sbi/save_params", help="Directory with saved compressor params")
    parser.add_argument("--flow-save-dir", type=str, default="/home/tersenov/software/Learn2Map/scripts/sbi/save_params", help="Directory to save flow params")
    parser.add_argument("--posterior-out", type=str, default="posterior.npy", help="File name for posterior samples output")

    # Training hyperparameters
    parser.add_argument("--total-steps", type=int, default=50_000, help="Number of training iterations for flow")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for flow training")
    parser.add_argument("--lr-init", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--lr-decay-rate", type=float, default=0.9, help="Exponential decay rate")
    parser.add_argument("--lr-transition-fraction", type=float, default=0.8, help="Fraction of steps used for LR schedule end value")
    parser.add_argument("--lr-end", type=float, default=1e-6, help="Final learning rate value")
    parser.add_argument("--nvp-layers", type=int, default=4, help="Number of RealNVP coupling layers")
    parser.add_argument("--nvp-hidden", type=int, default=128, help="Hidden layer width in coupling networks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # HMC / Posterior
    parser.add_argument("--hmc-num-results", type=int, default=40_000, help="Number of HMC samples (after burn-in)")
    parser.add_argument("--hmc-burnin", type=int, default=400, help="Number of burn-in steps")
    parser.add_argument("--hmc-num-chains", type=int, default=10, help="Number of HMC chains")
    parser.add_argument("--hmc-step-size", type=float, default=1e-2, help="Initial HMC step size")
    parser.add_argument("--hmc-leapfrog", type=int, default=3, help="Number of leapfrog steps")

    # Execution & options
    parser.add_argument("--no-train", action="store_true", help="Skip training (load latest saved flow params)")
    parser.add_argument("--no-mcmc", action="store_true", help="Skip HMC posterior sampling")
    parser.add_argument("--save-every", type=int, default=1000, help="Save flow params every N steps")
    parser.add_argument("--plot", action="store_true", help="Generate plots (requires matplotlib, chainconsumer, getdist)")

    return parser.parse_args()


def set_cuda_environment(cuda_devices: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            print(f"Memory growth set for GPUs: {gpus}")
    except Exception as e:  # noqa: BLE001
        print(f"Could not set GPU memory growth: {e}")


# -----------------------------------------------------------------------------
# Observed (fiducial) data
# -----------------------------------------------------------------------------

def load_observed_map(meta_path: str, fid_path: str, field_size: int, field_npix: int, nside: int, galaxy_density: float, sigma_e: float, rng_key: jax.Array) -> Tuple[jnp.ndarray, np.ndarray, np.ndarray]:
    print("######## OBSERVED DATA ########")
    with h5py.File(meta_path, "r") as f:
        dataset_grid = f['parameters']['fiducial']
        cosmo_parameters = jnp.array([
            dataset_grid['Om'],
            dataset_grid['s8'],
            dataset_grid['w0'],
            dataset_grid['H0']/100,
            dataset_grid['ns'],
            dataset_grid['Ob']
        ]).T
    truth = np.array(cosmo_parameters[0])

    # Load full-sky map & project
    with h5py.File(fid_path, "r") as f:
        m_data_full = np.array(f['kg'][f'stage3_lensing{4}'])  # adapt index for tomographic bin

    # Gnomonic projection to small patch
    reso = field_size * 60 / field_npix
    proj = hp.projector.GnomonicProj(rot=[0, 0, 0], xsize=field_npix, ysize=field_npix, reso=reso)
    m_data_patch = proj.projmap(m_data_full, vec2pix_func=partial(hp.vec2pix, nside))

    # Add noise
    noisy = tfd.Independent(
        tfd.Normal(
            loc=m_data_patch,
            scale=sigma_e / jnp.sqrt((galaxy_density * (field_size * 60 / field_npix) ** 2))
        ),
        1  # single map flattened
    ).sample(seed=rng_key, sample_shape=(1,))

    return noisy, cosmo_parameters, truth


# -----------------------------------------------------------------------------
# Compressor
# -----------------------------------------------------------------------------

class CompressorCNN2D(hk.Module):
    def __init__(self, output_dim: int, name: str | None = None):
        super().__init__(name=name)
        self.output_dim = output_dim

    def __call__(self, x):  # x: [B,H,W,1]
        net_x = hk.Conv2D(32, 3, 2)(x)
        net_x = jax.nn.leaky_relu(net_x)
        net_x = hk.Conv2D(64, 3, 2)(net_x)
        net_x = jax.nn.leaky_relu(net_x)
        net_x = hk.Conv2D(128, 3, 2)(net_x)
        net_x = jax.nn.leaky_relu(net_x)
        net_x = hk.AvgPool(16, 8, 'SAME')(net_x)
        net_x = hk.Flatten()(net_x)
        net_x = hk.Linear(64)(net_x)
        net_x = jax.nn.leaky_relu(net_x)
        net_x = hk.Linear(self.output_dim)(net_x)
        return net_x.squeeze()


def build_compressor(dim: int):
    return hk.transform_with_state(lambda y: CompressorCNN2D(dim)(y))


def load_compressor_params(base_dir: str, loss_name: str, map_kind: str) -> Tuple[hk.Params, hk.State]:
    """Load compressor parameters & opt_state (state placeholder)."""
    param_path = Path(base_dir) / loss_name / map_kind / "params_nd_compressor_batch100000.pkl"
    opt_state_path = Path(base_dir) / loss_name / map_kind / "opt_state_resnet_batch100000.pkl"
    with open(param_path, "rb") as f:
        params = pickle.load(f)
    with open(opt_state_path, "rb") as f:
        state = pickle.load(f)
    return params, state


# -----------------------------------------------------------------------------
# Data augmentation & dataset compression
# -----------------------------------------------------------------------------

def build_augmentation(map_kind: str, sigma_e: float, galaxy_density: float, field_size: int, field_npix: int):
    def augmentation_noise(example):
        key = {
            'nbody': 'map_nbody',
            'nbody_with_baryon_ia': 'map_nbody_w_baryon_ia',
            'gaussian': 'map_gaussian'
        }[map_kind]
        x = example[key]
        x += tf.random.normal(
            shape=(field_npix, field_npix),
            stddev=sigma_e / jnp.sqrt(galaxy_density * (field_size * 60 / field_npix) ** 2),
        )
        return {"maps": x, "theta": example["theta"]}

    def augmentation_flip(example):
        x = tf.expand_dims(example["maps"], -1)
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
        return {"maps": x, "theta": example["theta"]}

    def rescale_h(example):
        x = example['theta']
        idx = 3
        x = tf.tensor_scatter_nd_update(x, [[idx]], [x[idx] / 100])
        return {"maps": example["maps"], "theta": x}

    def augmentation(example):
        return rescale_h(augmentation_flip(augmentation_noise(example)))

    return augmentation


def compress_tfds_dataset(tfds_name: str, split: str, augmentation_fn, compressor, comp_params, comp_state, batch_size: int, dim: int) -> Dict[str, np.ndarray]:
    import tensorflow_datasets as tfds  # local import to allow script usage without tfds globally
    ds = tfds.load(tfds_name, split=split)
    ds = ds.map(augmentation_fn)
    ds = ds.batch(batch_size)

    y_list = []
    theta_list = []
    for example in ds.as_numpy_iterator():
        comp_y, _ = compressor.apply(
            comp_params,
            comp_state,
            None,
            example['maps']
        )
        y_list.append(comp_y)
        theta_list.append(example['theta'])

    return {
        'theta': np.concatenate(theta_list, axis=0),
        'x': np.concatenate(y_list, axis=0)
    }


def compress_single_map(map_array: jnp.ndarray, compressor, comp_params, comp_state, field_npix: int, nbins: int) -> jnp.ndarray:
    comp, _ = compressor.apply(comp_params, comp_state, None, map_array.reshape([1, field_npix, field_npix, nbins]))
    return comp


# -----------------------------------------------------------------------------
# Normalizing Flow (Conditional RealNVP)
# -----------------------------------------------------------------------------

def build_flow(nb_params: int, n_layers: int, hidden: int):
    bijector_ff = partial(
        AffineCoupling,
        layers=[hidden] * 4,
        activation=jax.nn.silu
    )
    NF_ff = partial(
        ConditionalRealNVP,
        n_layers=n_layers,
        bijector_fn=bijector_ff
    )

    class NF(hk.Module):
        def __call__(self, y):
            nvp = NF_ff(nb_params)(y)
            return nvp

    nf_logp = hk.without_apply_rng(
        hk.transform(lambda theta, y: NF()(theta).log_prob(y).squeeze())
    )
    return nf_logp


def make_update_fn(nf_logp, optimizer):
    def loss_nll(params, theta_batch, y_batch):
        return -jnp.mean(nf_logp.apply(params, theta_batch, y_batch))

    @jax.jit
    def update(params, opt_state, theta_batch, y_batch):
        loss, grads = jax.value_and_grad(loss_nll)(params, theta_batch, y_batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_opt_state

    return update


def train_flow(rng_key, nf_logp, dataset_train, dataset_val, total_steps: int, batch_size: int, save_every: int, save_dir: Path, loss_name: str, map_kind: str, lr_init: float, decay_rate: float, end_lr: float, transition_fraction: float) -> hk.Params:
    print("######## TRAINING FLOW ########")
    key_init, key_train = jax.random.split(rng_key)
    params = nf_logp.init(key_init, 0.5 * jnp.zeros([1, 6]), 0.5 * jnp.zeros([1, 6]))

    nb_steps = int(total_steps * transition_fraction)
    lr_schedule = optax.exponential_decay(
        init_value=lr_init,
        transition_steps=nb_steps // 50,
        decay_rate=decay_rate,
        end_value=end_lr,
    )
    optimizer = optax.adam(learning_rate=lr_schedule)
    opt_state = optimizer.init(params)
    update = make_update_fn(nf_logp, optimizer)

    theta_train = dataset_train['theta']
    x_train = dataset_train['x']
    theta_val = dataset_val['theta']
    x_val = dataset_val['x']

    n_train = len(theta_train)
    n_val = len(theta_val)

    batch_loss = []
    val_loss = []

    for step in range(1, total_steps + 1):
        # Sample train batch
        idx = np.random.randint(0, n_train, batch_size)
        loss, params, opt_state = update(params, opt_state, theta_train[idx], x_train[idx])
        batch_loss.append(float(loss))

        if step % 100 == 0:
            print(f"Step {step:6d} | train loss {loss:.4f}")

        if step % save_every == 0 or step == total_steps:
            save_path = save_dir / loss_name / map_kind
            save_path.mkdir(parents=True, exist_ok=True)
            with open(save_path / f"params_nd_flow_batch{step}.pkl", "wb") as f:
                pickle.dump(params, f)
            # Validation (no parameter update)
            vidx = np.random.randint(0, n_val, min(batch_size, n_val))
            val_l = -jnp.mean(nf_logp.apply(params, theta_val[vidx], x_val[vidx]))
            val_loss.append(float(val_l))
            print(f"Saved flow params at step {step}. Validation loss {val_l:.4f}")

    return params


# -----------------------------------------------------------------------------
# Posterior (Prior + Likelihood) & HMC
# -----------------------------------------------------------------------------

def build_prior():
    # Uniform priors matching notebook
    def log_prob_prior(theta: jnp.ndarray):
        Om_logprob = tfd.Uniform(0.1, 0.5).log_prob(theta[0])
        s8_logprob = tfd.Uniform(0.4, 1.4).log_prob(theta[1])
        h_logprob = tfd.Uniform(0.64, 0.82).log_prob(theta[3])
        Ob_logprob = tfd.Uniform(0.03, 0.06).log_prob(theta[5])
        ns_logprob = tfd.Uniform(0.87, 1.07).log_prob(theta[4])
        w0_logprob = tfd.Uniform(-2.0, -0.333).log_prob(theta[2])
        return Om_logprob + s8_logprob + h_logprob + Ob_logprob + ns_logprob + w0_logprob
    return log_prob_prior


def make_posterior_log_prob(nf_logp, flow_params, compressed_obs):
    prior_logp = build_prior()

    def log_prob(theta):
        # theta shape [6]
        likelihood = nf_logp.apply(flow_params, theta.reshape([1, 6]), compressed_obs.reshape([1, 6]))
        return prior_logp(theta) + likelihood.squeeze()

    return jax.vmap(log_prob)


def run_hmc(rng_key, unnormalized_log_prob, num_results: int, num_burnin: int, num_chains: int, step_size: float, leapfrog: int, init_theta: np.ndarray):
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=unnormalized_log_prob,
            num_leapfrog_steps=leapfrog,
            step_size=step_size,
        ),
        num_adaptation_steps=int(num_burnin * 0.8),
    )

    def run_chain():
        samples, is_accepted = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin,
            current_state=jnp.array(init_theta) * jnp.ones([num_chains, 6]),
            kernel=adaptive_hmc,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
            seed=rng_key,
        )
        return samples, is_accepted

    samples_hmc, accepted = run_chain()
    print(f"HMC acceptance rate: {float(jnp.mean(accepted)):.3f}")
    posterior_samples = samples_hmc[accepted]
    return posterior_samples, samples_hmc, accepted


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    set_cuda_environment(args.cuda_visible_devices)
    print("Platform =", xla_bridge.get_backend().platform)

    rng = jax.random.PRNGKey(args.seed)
    rng, rng_obs = jax.random.split(rng)

    # 1. Observed map
    m_data, cosmo_parameters, truth = load_observed_map(
        args.cosmogrid_meta,
        args.fiducial_map,
        args.field_size,
        args.field_npix,
        args.nside,
        args.galaxy_density,
        args.sigma_e,
        rng_obs,
    )

    # 2. Compressor
    compressor = build_compressor(args.dim)
    params_compressor, state_compressor = load_compressor_params(args.compressor_param_dir, args.loss_name, args.map_kind)

    # 3. Data augmentation & dataset compression
    augmentation = build_augmentation(args.map_kind, args.sigma_e, args.galaxy_density, args.field_size, args.field_npix)
    print("######## COMPRESS DATA ########")
    dataset_val = compress_tfds_dataset('NbodyCosmogridDataset/grid', 'test', augmentation, compressor, params_compressor, state_compressor, 500, args.dim)
    dataset_train = compress_tfds_dataset('NbodyCosmogridDataset/grid', 'train', augmentation, compressor, params_compressor, state_compressor, 500, args.dim)
    compressed_obs = compress_single_map(m_data, compressor, params_compressor, state_compressor, args.field_npix, args.nbins)

    # 4. Flow
    nf_logp = build_flow(nb_params=args.dim, n_layers=args.nvp_layers, hidden=args.nvp_hidden)
    flow_param_path = None

    if args.no_train:
        # Attempt to load latest params
        flow_dir = Path(args.flow_save_dir) / args.loss_name / args.map_kind
        candidates = sorted(flow_dir.glob("params_nd_flow_batch*.pkl"))
        if not candidates:
            raise FileNotFoundError("No saved flow parameters found and --no-train specified")
        flow_param_path = candidates[-1]
        with open(flow_param_path, "rb") as f:
            flow_params = pickle.load(f)
        print(f"Loaded flow params from {flow_param_path}")
    else:
        flow_params = train_flow(
            rng,
            nf_logp,
            dataset_train,
            dataset_val,
            total_steps=args.total_steps,
            batch_size=args.batch_size,
            save_every=args.save_every,
            save_dir=Path(args.flow_save_dir),
            loss_name=args.loss_name,
            map_kind=args.map_kind,
            lr_init=args.lr_init,
            decay_rate=args.lr_decay_rate,
            end_lr=args.lr_end,
            transition_fraction=args.lr_transition_fraction,
        )

    # 5. Posterior via HMC
    if not args.no_mcmc:
        print("######## RUN MCMC ########")
        unnorm = make_posterior_log_prob(nf_logp, flow_params, compressed_obs)
        posterior_samples, all_samples, accepted = run_hmc(
            rng,
            unnorm,
            num_results=args.hmc_num_results,
            num_burnin=args.hmc_burnin,
            num_chains=args.hmc_num_chains,
            step_size=args.hmc_step_size,
            leapfrog=args.hmc_leapfrog,
            init_theta=truth,
        )
        out_path = Path(args.posterior_out)
        np.save(out_path, np.array(posterior_samples))
        print(f"Saved posterior samples to {out_path.resolve()}")

    else:
        print("Skipping MCMC (--no-mcmc)")

    print("Done.")


if __name__ == "__main__":  # pragma: no cover
    main()
