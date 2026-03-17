from functools import partial

import h5py
import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils import gcs_utils

# disable internet connection
gcs_utils.gcs_dataset_info_files = lambda *args, **kwargs: None
gcs_utils.is_dataset_on_gcs = lambda *args, **kwargs: False

_CITATION = """
"""

_DESCRIPTION = """
Dataset of N-body cosmological simulations for parameter inference.
"""


class DatasetConfig(tfds.core.BuilderConfig):
    def __init__(
        self,
        *,
        xsize,
        size,
        **kwargs,
    ):
        v1 = tfds.core.Version("0.0.2")
        super().__init__(description=("N-body cosmological simulations."), version=v1, **kwargs)
        self.xsize = xsize
        self.size = size


class NbodyCosmogridDatasetTomo(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for N-body cosmological simulations."""

    VERSION = tfds.core.Version("0.0.2")
    RELEASE_NOTES = {
        "0.0.1": "Initial release.",
        "0.0.2": "Tomographic 4-channel maps (bins 1-4).",
    }
    BUILDER_CONFIGS = [
        DatasetConfig(
            name="grid",
            xsize=80,
            size=10,
        ),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "map_nbody": tfds.features.Tensor(
                        shape=[
                            self.builder_config.xsize,
                            self.builder_config.xsize,
                            4,
                        ],
                        dtype=tf.float32,
                    ),
                    "theta": tfds.features.Tensor(shape=[6], dtype=tf.float32),
                }
            ),
            supervised_keys=None,
            homepage="https://dataset-homepage/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "start": 1,
                    "end": 900,
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "start": 900,
                    "end": 1300,
                },
            ),
        ]

    def _generate_examples(self, start, end):
        """Yields examples."""

        filename = "/home/tersenov/CosmoGridV1/CosmoGridV1_metainfo.h5"
        f = h5py.File(filename, "r")
        dataset_grid = f["parameters"]["grid"]

        nb_of_projected_map = 25
        cosmo_parameters = jnp.array(
            [
                dataset_grid["Om"],
                dataset_grid["s8"],
                dataset_grid["w0"],
                dataset_grid["H0"],
                dataset_grid["ns"],
                dataset_grid["Ob"],
            ]
        ).T

        nside = 512
        xsize = self.builder_config.xsize  # width of figure in pixels
        size = self.builder_config.size  # Size of square in degrees
        reso = size * 60 / xsize
        master_key = jax.random.PRNGKey(0)

        for i in range(start, end):
            key, master_key = jax.random.split(master_key)
            params = cosmo_parameters[i]
            path_string = "/home/tersenov/" + dataset_grid["path_par"][i].decode(
                "utf-8"
            ).replace("CosmoGrid", "CosmoGridV1").replace("raw", "stage3_forecast").replace("grid", "new_grid")
            
            for j in range(7):
                filename = path_string + "perm_000" + str(j)
                filename_nbody = filename + "/projected_probes_maps_nobaryons512.h5"
                sim_nbody = h5py.File(filename_nbody, "r")

                # load 4 tomographic bins
                kg = sim_nbody["kg"]
                tomo_maps = [
                    np.array(kg[f"stage3_lensing{b}"]) for b in (1, 2, 3, 4)
                ]
                
                # projection
                key1, key2 = jax.random.split(key)
                lon = jax.random.randint(key1, (nb_of_projected_map,), -180, 180)
                lat = jax.random.randint(key2, (nb_of_projected_map,), -90, 90)
                
                for k in range(nb_of_projected_map):
                    proj = hp.projector.GnomonicProj(
                        rot=[lon[k], lat[k], 0], xsize=xsize, ysize=xsize, reso=reso
                    )
                    # Project each tomographic bin with the same projector
                    projected_bins = [
                        proj.projmap(bin_map, vec2pix_func=partial(hp.vec2pix, nside))
                        for bin_map in tomo_maps
                    ]
                    projection_nbody = np.stack(projected_bins, axis=-1).astype(np.float32)
                    
                    yield f"{i}-{j}-{k}", {
                        "map_nbody": jnp.array(projection_nbody, dtype=jnp.float32),
                        "theta": jnp.array(params, dtype=jnp.float32),
                    }