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


def _angular_distance_deg(
    lon_deg: float,
    lat_deg: float,
    lon_ref_deg: np.ndarray,
    lat_ref_deg: np.ndarray,
) -> np.ndarray:
    """Great-circle distance (degrees) between one point and reference arrays."""
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    lon_ref = np.deg2rad(lon_ref_deg)
    lat_ref = np.deg2rad(lat_ref_deg)
    cos_dist = (
        np.sin(lat) * np.sin(lat_ref)
        + np.cos(lat) * np.cos(lat_ref) * np.cos(lon - lon_ref)
    )
    cos_dist = np.clip(cos_dist, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_dist))


def _build_non_overlapping_centers(
    n_centers: int,
    min_separation_deg: float,
    center_nside: int,
) -> np.ndarray:
    """Build deterministic patch centers separated by at least min_separation_deg."""
    npix = hp.nside2npix(center_nside)
    theta, phi = hp.pix2ang(center_nside, np.arange(npix))
    lon = np.degrees(phi) - 180.0
    lat = 90.0 - np.degrees(theta)

    # Deterministic candidate orders; keep best if requested count is too high.
    candidate_orders = [
        np.argsort(np.abs(lat)),  # favor moderate latitudes first
        np.arange(npix),
        np.argsort(lon),
    ]

    best_selected: list[tuple[float, float]] = []
    for order in candidate_orders:
        selected: list[tuple[float, float]] = []
        for idx in order:
            cand_lon = float(lon[idx])
            cand_lat = float(lat[idx])
            if not selected:
                selected.append((cand_lon, cand_lat))
                if len(selected) >= n_centers:
                    break
                continue
            sel_lon = np.array([p[0] for p in selected], dtype=np.float64)
            sel_lat = np.array([p[1] for p in selected], dtype=np.float64)
            min_dist = float(
                np.min(_angular_distance_deg(cand_lon, cand_lat, sel_lon, sel_lat))
            )
            if min_dist >= min_separation_deg:
                selected.append((cand_lon, cand_lat))
                if len(selected) >= n_centers:
                    break
        if len(selected) > len(best_selected):
            best_selected = selected
        if len(selected) >= n_centers:
            return np.array(selected[:n_centers], dtype=np.float32)

    raise ValueError(
        "Unable to generate enough non-overlapping centers with "
        f"center_nside={center_nside}, min_separation_deg={min_separation_deg}. "
        f"Requested {n_centers}, best achieved {len(best_selected)}."
    )


class DatasetConfig(tfds.core.BuilderConfig):
    def __init__(
        self,
        *,
        xsize,
        size,
        projections_mode="random",
        nb_of_projected_map=25,
        min_separation_deg=0.0,
        center_nside=16,
        **kwargs,
    ):
        v1 = tfds.core.Version("0.0.2")
        super().__init__(description=("N-body cosmological simulations."), version=v1, **kwargs)
        self.xsize = xsize
        self.size = size
        self.projections_mode = projections_mode
        self.nb_of_projected_map = nb_of_projected_map
        self.min_separation_deg = min_separation_deg
        self.center_nside = center_nside


class NbodyCosmogridDatasetTomo(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for N-body cosmological simulations."""

    VERSION = tfds.core.Version("0.0.2")
    RELEASE_NOTES = {
        "0.0.1": "Initial release.",
        "0.0.2": "Tomographic 4-channel maps (bins 1-4).",
        "0.0.3": "Added deterministic non-overlapping projection config.",
    }
    BUILDER_CONFIGS = [
        DatasetConfig(
            name="grid",
            xsize=80,
            size=10,
        ),
        DatasetConfig(
            name="grid_20deg_160px",
            xsize=160,
            size=20,
        ),
        DatasetConfig(
            name="grid_20deg_160px_nonoverlap48",
            xsize=160,
            size=20,
            projections_mode="non_overlap",
            nb_of_projected_map=48,
            min_separation_deg=28.5,
            center_nside=32,
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

        nb_of_projected_map = int(self.builder_config.nb_of_projected_map)
        projections_mode = str(self.builder_config.projections_mode)
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
        fixed_centers: np.ndarray | None = None
        if projections_mode == "non_overlap":
            fixed_centers = _build_non_overlapping_centers(
                n_centers=nb_of_projected_map,
                min_separation_deg=float(self.builder_config.min_separation_deg),
                center_nside=int(self.builder_config.center_nside),
            )
        elif projections_mode != "random":
            raise ValueError(
                f"Unknown projections_mode='{projections_mode}'. "
                "Allowed: 'random', 'non_overlap'."
            )

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
                if fixed_centers is not None:
                    lon = fixed_centers[:, 0]
                    lat = fixed_centers[:, 1]
                else:
                    key, subkey = jax.random.split(key)
                    key1, key2 = jax.random.split(subkey)
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
