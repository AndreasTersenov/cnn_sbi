import argparse
import pickle
from functools import partial

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import h5py
import haiku as hk
import healpy as hp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from chainconsumer import ChainConsumer
from jax.lib import xla_bridge
from numpyro import distributions as dist
from sbi_lens.normflow.models import AffineCoupling, ConditionalRealNVP
from sbi_lens.normflow.train_model import TrainModel
from tqdm import tqdm

import getdist.plots as gplot
from getdist import MCSamples

# Set memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

print(xla_bridge.get_backend().platform)

# tfp = tfp.experimental.substrates.jax
from tensorflow_probability.substrates import jax as tfp
tfb = tfp.bijectors
tfd = tfp.distributions

# Register TFDS tomo dataset builder
import tf_dataset_nbody_tomo as tf_dataset_nbody

sigma_e = 0.26
galaxy_density = 30 / 4
field_size = size = 10
field_npix = xsize = 80
nside = 512
reso = size * 60 / xsize
nbins = 4
dim = 6

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument("--total_steps", type=int, default=150_000)
parser.add_argument(
    "--map_kind", type=str, default="nbody"
)  # nbody_with_baryon_ia or gaussian or nbody
parser.add_argument("--loss", type=str, default="mse")
parser.add_argument("--bin", type=int, default=4, help="Specify the lensing bin number")
parser.add_argument("--sigma_e", type=float, default=0.26, help="Noise level sigma_e")

args = parser.parse_args()
# override sigma_e from command line
sigma_e = args.sigma_e

bin_number = args.bin

if args.loss == "mse":
    loss_name = "train_compressor_mse"
elif args.loss == "vmim":
    loss_name = "train_compressor_vmim"
    

# Ensure the save_params and fig directories exist
os.makedirs(f"./save_params/{args.loss}/{args.map_kind}/sigma_{sigma_e}/gal_density_{int(galaxy_density*4)}/bin_{bin_number}", exist_ok=True)
os.makedirs(f"./fig/{args.loss}/{args.map_kind}/sigma_{sigma_e}/gal_density_{int(galaxy_density*4)}/bin_{bin_number}", exist_ok=True)

print("######## CONFIG ########")


print("######## OBSERVED DATA ########")
filename = "/home/tersenov/CosmoGridV1/CosmoGridV1_metainfo.h5"
f = h5py.File(filename, "r")
dataset_grid = f["parameters"]["fiducial"]
cosmo_parameters = jnp.array(
    [
        dataset_grid["Om"],
        dataset_grid["s8"],
        dataset_grid["w0"],
        dataset_grid["H0"] / 100,
        dataset_grid["ns"],
        dataset_grid["Ob"],
    ]
).T
truth = list(cosmo_parameters[0])
print('TRUTH=', truth)
path = "/home/tersenov/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial/perm_0000/projected_probes_maps_nobaryons512.h5"
m_data_h5 = h5py.File(path, "r")
# Build 4-channel tomographic observed map (bins 1..4), then project each with same projector
proj = hp.projector.GnomonicProj(rot=[0, 0, 0], xsize=xsize, ysize=xsize, reso=reso)
proj_bins = []
for b in (1, 2, 3, 4):
    full_map = np.array(m_data_h5["kg"][f"stage3_lensing{b}"])
    proj_map = proj.projmap(full_map, vec2pix_func=partial(hp.vec2pix, nside))
    proj_bins.append(proj_map)
# Stack into (H, W, 4)
m_data = np.stack(proj_bins, axis=-1).astype(np.float32)
# Add shape noise equally to each bin
stddev = sigma_e / jnp.sqrt(galaxy_density * (field_size * 60 / field_npix) ** 2)
m_data = jnp.asarray(m_data) + jax.random.normal(jax.random.PRNGKey(0), (field_npix, field_npix, nbins)) * stddev

# Apply BNT transformation to observed data
BNT_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [-1.0, 1.0, 0.0, 0.0],
        [0.4521097, -1.4521097, 1.0, 0.0],
        [0.0, 0.25127807, -1.251278, 1.0],
    ],
    dtype=np.float32,
)
m_data = jnp.tensordot(m_data, BNT_MATRIX, axes=[[2], [1]])

# params_name = [
#     r"$\\Omega_m$",
#     r"$\\sigma_8$",
#     r"$w_0$",
#     r"$h_0$",
#     r"$n_s$",
#     r"$\\Omega_b$",
# ]
params_name = [
    r'\Omega_m', 
    r'\sigma_8', 
    r'w_0', 
    r'h_0', 
    r'n_s', 
    r'\Omega_b']

print("######## DATA AUGMENTATION ########")
tf.random.set_seed(1)

if args.map_kind == "nbody_with_baryon_ia":
    print("nbody w baryon and ia")

    def augmentation_noise(
        example, sigma_e=0.26, galaxy_density=27, field_size=5, field_npix=256
    ):
        x = example["map_nbody_w_baryon_ia"]
        x += tf.random.normal(
            shape=(field_npix, field_npix, nbins),
            stddev=sigma_e
            / jnp.sqrt(galaxy_density * (field_size * 60 / field_npix) ** 2),
        )

        return {"maps": x, "theta": example["theta"]}

elif args.map_kind == "nbody":
    print("nbody")

    def augmentation_noise(
        example, sigma_e=0.26, galaxy_density=27, field_size=5, field_npix=256
    ):
        x = example["map_nbody"]
        x += tf.random.normal(
            shape=(field_npix, field_npix, nbins),
            stddev=sigma_e
            / jnp.sqrt(galaxy_density * (field_size * 60 / field_npix) ** 2),
        )

        return {"maps": x, "theta": example["theta"]}

elif args.map_kind == "gaussian":
    print("gaussian")

    def augmentation_noise(
        example, sigma_e=0.26, galaxy_density=27, field_size=5, field_npix=256
    ):
        x = example["map_gaussian"]
        x += tf.random.normal(
            shape=(field_npix, field_npix, nbins),
            stddev=sigma_e
            / jnp.sqrt(galaxy_density * (field_size * 60 / field_npix) ** 2),
        )

        return {"maps": x, "theta": example["theta"]}


def augmentation_flip(example):
    x = example["maps"]
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return {"maps": x, "theta": example["theta"]}


def augmentation_bnt(example):
    BNT_MATRIX = tf.constant(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0, 0.0],
            [0.4521097, -1.4521097, 1.0, 0.0],
            [0.0, 0.25127807, -1.251278, 1.0],
        ],
        dtype=tf.float32,
    )
    x = example["maps"]
    x = tf.tensordot(x, BNT_MATRIX, axes=[[2], [1]])
    return {"maps": x, "theta": example["theta"]}


def rescale_h(example):
    x = example["theta"]
    index_to_update = 3
    x = tf.tensor_scatter_nd_update(x, [[index_to_update]], [x[index_to_update] / 100])
    return {"maps": example["maps"], "theta": x}


def augmentation(example):
    return rescale_h(
        augmentation_bnt(
            augmentation_flip(
                augmentation_noise(
                    example=example,
                    sigma_e=sigma_e,
                    galaxy_density=galaxy_density,
                    field_size=field_size,
                    field_npix=field_npix,
                )
            )
        )
    )


print("######## CREATE COMPRESSOR ########")

# nf
bijector_layers_compressor = [128] * 2

bijector_compressor = partial(
    AffineCoupling, layers=bijector_layers_compressor, activation=jax.nn.silu
)

NF_compressor = partial(ConditionalRealNVP, n_layers=4, bijector_fn=bijector_compressor)


# theta_bijector = tfb.Chain([
#     tfb.Scale(
#         jnp.array([
#             0.09285661,
#             0.23046516,
#             0.27378845,
#             4.458831,
#             0.04350383,
#             0.00650289
#         ])
#     ),
#     tfb.Shift(
#         -jnp.array([
#             2.9245374e-01,
#             8.2852399e-01,
#             -9.4738042e-01,
#             7.1496910e+01,
#             9.6793532e-01,
#             4.5024041e-02
#         ])
#     ),
# ])

# class Flow_nd_Compressor(hk.Module):
#     def __call__(self, y):
#         nvp = NF_compressor(dim)(y)
#         return tfd.TransformedDistribution(
#             nvp,
#             tfb.Chain([tfb.Invert(theta_bijector)])
#         )


class Flow_nd_Compressor(hk.Module):
    def __call__(self, y):
        nvp = NF_compressor(dim)(y)
        return nvp


nf = hk.without_apply_rng(
    hk.transform(lambda theta, y: Flow_nd_Compressor()(y).log_prob(theta).squeeze())
)


# compressor
class CompressorCNN2D(hk.Module):
    def __init__(self, output_dim):
        super().__init__()
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


compressor = hk.transform_with_state(lambda y: CompressorCNN2D(dim)(y))
compressor_eval = hk.transform_with_state(lambda y: CompressorCNN2D(dim)(y))


print("######## TRAIN ########")

# init compressor
parameters_resnet, opt_state_resnet = compressor.init(
    jax.random.PRNGKey(0), y=0.5 * jnp.ones([1, field_npix, field_npix, nbins])
)
# init nf
params_nf = nf.init(
    jax.random.PRNGKey(0), theta=0.5 * jnp.ones([1, dim]), y=0.5 * jnp.ones([1, dim])
)

parameters_compressor = hk.data_structures.merge(parameters_resnet, params_nf)

del parameters_resnet, params_nf

# define optimizer
total_steps = args.total_steps - args.total_steps // 3
lr_scheduler = optax.piecewise_constant_schedule(
    init_value=0.0005,
    boundaries_and_scales={
        int(total_steps * 0.1): 0.7,
        int(total_steps * 0.2): 0.7,
        int(total_steps * 0.3): 0.7,
        int(total_steps * 0.4): 0.7,
        int(total_steps * 0.5): 0.7,
        int(total_steps * 0.6): 0.7,
        int(total_steps * 0.7): 0.7,
        int(total_steps * 0.8): 0.7,
        int(total_steps * 0.9): 0.7,
    },
)

optimizer_c = optax.adam(learning_rate=lr_scheduler)
opt_state_c = optimizer_c.init(parameters_compressor)

model_compressor = TrainModel(
    compressor=compressor,
    nf=nf,
    optimizer=optimizer_c,
    loss_name=loss_name,
)


# train dataset
# ds_tr = tfds.load("NbodyCosmogridDatasetLarge/grid", split="train")
ds_tr = tfds.load("NbodyCosmogridDatasetTomo/grid", split="train")

ds_tr = ds_tr.repeat()
ds_tr = ds_tr.shuffle(800)
# ds_tr = ds_tr.map(augmentation)
ds_tr = ds_tr.map(augmentation, num_parallel_calls=tf.data.AUTOTUNE)
ds_tr = ds_tr.batch(128)
ds_tr = ds_tr.prefetch(tf.data.experimental.AUTOTUNE)
ds_train = iter(tfds.as_numpy(ds_tr))

# test dataset
ds_te = tfds.load("NbodyCosmogridDatasetTomo/grid", split="test")

ds_te = ds_te.repeat()
ds_te = ds_te.shuffle(200)
ds_te = ds_te.map(augmentation, num_parallel_calls=tf.data.AUTOTUNE)
ds_te = ds_te.batch(128)
ds_te = ds_te.prefetch(tf.data.experimental.AUTOTUNE)
ds_test = iter(tfds.as_numpy(ds_te))

update = jax.jit(model_compressor.update)

store_loss = []
loss_train = []
loss_test = []
for batch in tqdm(range(1, args.total_steps + 1)):
    ex = next(ds_train)
    if not jnp.isnan(ex["maps"]).any():
        b_loss, parameters_compressor, opt_state_c, opt_state_resnet = update(
            model_params=parameters_compressor,
            opt_state=opt_state_c,
            theta=ex["theta"],
            x=ex["maps"],
            state_resnet=opt_state_resnet,
        )

        store_loss.append(b_loss)

        if jnp.isnan(b_loss):
            print("NaN Loss")
            break

    if batch % 2000 == 0:
        # save params
        save_dir_params = f"./save_params/{args.loss}/{args.map_kind}/sigma_{sigma_e}/gal_density_{int(galaxy_density*4)}/BNT"
        save_dir_fig = f"./fig/{args.loss}/{args.map_kind}/sigma_{sigma_e}/gal_density_{int(galaxy_density*4)}/BNT"
        os.makedirs(save_dir_params, exist_ok=True)
        os.makedirs(save_dir_fig, exist_ok=True)

        with open(
            f"{save_dir_params}/params_nd_compressor_BNT_batch{batch}.pkl",
            "wb",
        ) as fp:
            pickle.dump(parameters_compressor, fp)

        with open(
            f"{save_dir_params}/opt_state_resnet_BNT_batch{batch}.pkl",
            "wb",
        ) as fp:
            pickle.dump(opt_state_resnet, fp)

        # save plot losses
        plt.figure()
        plt.plot(store_loss[1000:])
        plt.title("Batch Loss")
        plt.savefig(f"{save_dir_fig}/loss_compressor_BNT")
        plt.close()

        ex_test = next(ds_test)

        b_loss_test, _, _, _ = update(
            model_params=parameters_compressor,
            opt_state=opt_state_c,
            theta=ex_test["theta"],
            x=ex_test["maps"],
            state_resnet=opt_state_resnet,
        )

        loss_train.append(b_loss)
        loss_test.append(b_loss_test)

        jnp.save(
            f"{save_dir_params}/loss_train_BNT.npy", loss_train
        )
        jnp.save(f"{save_dir_params}/loss_test_BNT.npy", loss_test)

        plt.figure()
        plt.plot(loss_train, label="train loss")
        plt.plot(loss_test, label="test loss")
        plt.legend()
        plt.title("Batch Loss")
        plt.savefig(f"{save_dir_fig}/loss_compressor_train_test_BNT")
        plt.close()

        # save contour plot
        y, _ = compressor_eval.apply(
            parameters_compressor,
            opt_state_resnet,
            None,
            m_data.reshape([1, field_npix, field_npix, nbins]),
        )

        nvp_sample_nd = hk.transform(
            lambda x: Flow_nd_Compressor()(x).sample(100000, seed=hk.next_rng_key())
        )
        sample_nd = nvp_sample_nd.apply(
            parameters_compressor,
            rng=jax.random.PRNGKey(43),
            x=y * jnp.ones([100000, dim]),
        )
        idx = jnp.where(jnp.isnan(sample_nd))[0]
        sample_nd = jnp.delete(sample_nd, idx, axis=0)

        truth_arr = np.array(truth)
        theta = dict(zip(params_name, truth_arr))

        plt.figure()
        param_limits = {}
        for i, name in enumerate(params_name):
            s_min, s_max = sample_nd[:, i].min(), sample_nd[:, i].max()
            truth_val = truth[i]

            lower = min(s_min, truth_val) - 0.05 * abs(truth_val)
            upper = max(s_max, truth_val) + 0.05 * abs(truth_val)
            
            param_limits[name] = (lower, upper)

        # Convert truth values to dictionary
        truth_arr = np.array(truth)
        theta_dict = dict(zip(params_name, truth_arr))

        # Create MCSamples object for GetDist
        samples = MCSamples(samples=sample_nd, names=params_name, labels=params_name)

        # Create plotter
        g = gplot.get_subplot_plotter(subplot_size=1.5)
        g.triangle_plot(samples, filled=True, markers=truth_arr, marker_args={"color": "red", "lw": 1.2}, param_limits=param_limits)

        plt.savefig(
            f"{save_dir_fig}/contour_plot_compressor_BNT_batch{batch}"
        )
        plt.close()
