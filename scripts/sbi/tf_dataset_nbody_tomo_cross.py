"""TFDS builder for the 10-channel auto+cross harmonic maps (nobnt).

Produces a proper TFDS dataset (ArrayRecord or TFRecord) of the 4 auto + 6 cross
channels, so the auto+cross CNN can be trained via the same standard, fast loader the
auto-only path uses — instead of the slow/fragile custom `.npz` + hand-rolled
tf.data + DLPack path (see HANDOFF_CNN_LOADER_REBUILD.md).

Data source: this builder **reserializes the already-validated `.npz` cross cache**
(`results/exploratory/cross_maps_campaign/full_sphere_cache_grid`), one TFDS example per
patch. The `.npz` patches are bit-exact to `build_full_sphere_cross_cache.compute_cross_patches`
(the single-source-of-truth compute), verified by `tests/validate_cross_compute_refactor.py`;
reserializing avoids recomputing ~9k SHTs (a serial TFDS `GeneratorBasedBuilder` cannot
parallelize them). The maps are float32, identical to the cache.

Build (one-time, CPU/IO only — touches no GPU; ~1 h I/O-bound):
  cd scripts/sbi
  # ArrayRecord (random-access, for Grain):
  conda run -n jaxili tfds build tf_dataset_nbody_tomo_cross.py \
    --config grid_20deg_160px_nonoverlap48 --file_format=array_record \
    --data_dir /nas/tersenov/tfds_cross_arrayrecord
  # TFRecord (streaming, for the vanilla auto-only tf.data path):
  conda run -n jaxili tfds build tf_dataset_nbody_tomo_cross.py \
    --config grid_20deg_160px_nonoverlap48 --file_format=tfrecord \
    --data_dir /nas/tersenov/tfds_cross_tfrecord

Fast validation build on a subset: CROSS_TFDS_COSMO_LIMIT=N (distinct cosmologies/split).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils import gcs_utils

# Disable any network access (mirrors the auto-only builder).
gcs_utils.gcs_dataset_info_files = lambda *a, **k: None
gcs_utils.is_dataset_on_gcs = lambda *a, **k: False

_HERE = Path(__file__).resolve().parent
# Validated .npz cross cache (the build source).
_NPZ_CACHE = _HERE / "results" / "exploratory" / "cross_maps_campaign" / "full_sphere_cache_grid"
_REGIME = "nobnt"  # Phase 1 scope: no-BNT only.
N_AUTO, N_CROSS = 4, 6
N_CHANNELS = N_AUTO + N_CROSS  # 10
# TFDS split -> .npz cache subdir (grid[1:900]=train, grid[900:1300]=val, fiducial=obs).
_SPLIT_SUBDIR = {tfds.Split.TRAIN: "train", tfds.Split.TEST: "val", "obs": "obs"}


def _cosmo_limit() -> int | None:
    """Optional per-split distinct-cosmology cap for fast validation builds."""
    v = int(os.environ.get("CROSS_TFDS_COSMO_LIMIT", "0"))
    return v if v > 0 else None


def _process_npz(npz_path: str):
    """Pool worker: read one zlib `.npz`, return (cosmo_idx, perm, patches, theta).

    Module-level (picklable) so a multiprocessing.Pool can ship it to workers.
    """
    d = np.load(npz_path, allow_pickle=True)
    return (
        int(d["cosmo_idx"]),
        int(d["perm"]),
        np.asarray(d["patches"]),                  # (48, H, W, 10) float32
        d["theta"].astype(np.float32),             # (6,) float32
    )


class CrossDatasetConfig(tfds.core.BuilderConfig):
    def __init__(self, *, xsize, size, **kwargs):
        super().__init__(version=tfds.core.Version("0.0.1"), **kwargs)
        self.xsize = xsize
        self.size = size


class NbodyCosmogridDatasetTomoCross(tfds.core.GeneratorBasedBuilder):
    """10-channel (4 auto + 6 cross) harmonic maps for N-body parameter inference."""

    VERSION = tfds.core.Version("0.0.1")
    RELEASE_NOTES = {"0.0.1": "10-channel auto+cross harmonic maps (nobnt), 48 non-overlap patches."}
    BUILDER_CONFIGS = [
        CrossDatasetConfig(name="grid_20deg_160px_nonoverlap48", xsize=160, size=20),
    ]
    # Skip TFDS's build-time shuffle. The training loader (tfds.load + tf.data .shuffle)
    # reshuffles every epoch, so the build-time shuffle is wasted I/O. Skipping it also
    # removes the second slow serial pass.
    DISABLE_SHUFFLING = True

    def _info(self) -> tfds.core.DatasetInfo:
        c = self.builder_config
        return tfds.core.DatasetInfo(
            builder=self,
            description="N-body 10-channel auto+cross harmonic maps (4 auto + 6 cross), nobnt.",
            features=tfds.features.FeaturesDict(
                {
                    "map_nbody": tfds.features.Tensor(
                        shape=[c.xsize, c.xsize, N_CHANNELS], dtype=tf.float32
                    ),
                    "theta": tfds.features.Tensor(shape=[6], dtype=tf.float32),
                    # Provenance (cheap int64 scalars) — enables the bit-exact gate vs the
                    # .npz cache + split-overlap audits. The training loader selects only
                    # map_nbody + theta.
                    "cosmo_idx": tfds.features.Tensor(shape=[], dtype=tf.int64),
                    "perm": tfds.features.Tensor(shape=[], dtype=tf.int64),
                    "patch": tfds.features.Tensor(shape=[], dtype=tf.int64),
                }
            ),
            supervised_keys=None,
            homepage="https://dataset-homepage/",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return [
            tfds.core.SplitGenerator(name=split, gen_kwargs={"split_subdir": subdir})
            for split, subdir in _SPLIT_SUBDIR.items()
        ]

    def _generate_examples(self, split_subdir):
        cache_dir = _NPZ_CACHE / _REGIME / split_subdir
        files = sorted(cache_dir.glob("*.npz"))
        limit = _cosmo_limit()
        if limit is not None:
            seen: set[str] = set()
            kept = []
            for f in files:
                cid = f.name.split("_perm")[0]
                if cid not in seen:
                    if len(seen) >= limit:
                        continue
                    seen.add(cid)
                kept.append(f)
            files = kept

        # CROSS_TFDS_BUILD_WORKERS: parallel .npz decode (zlib decompress is the bottleneck).
        # Default 50 (the available CPU budget). 0/1 = serial fallback.
        n_workers = int(os.environ.get("CROSS_TFDS_BUILD_WORKERS", "50"))
        files_str = [str(f) for f in files]

        if n_workers > 1 and len(files_str) > 1:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")  # fresh interpreters, no fork-safety issues with TF
            with ctx.Pool(min(n_workers, len(files_str))) as pool:
                for ci, perm, patches, theta32 in pool.imap_unordered(
                    _process_npz, files_str, chunksize=1
                ):
                    for k in range(patches.shape[0]):
                        yield f"{ci}-{perm}-{k}", {
                            "map_nbody": patches[k],
                            "theta": theta32,
                            "cosmo_idx": np.int64(ci),
                            "perm": np.int64(perm),
                            "patch": np.int64(k),
                        }
        else:
            for f in files_str:
                ci, perm, patches, theta32 = _process_npz(f)
                for k in range(patches.shape[0]):
                    yield f"{ci}-{perm}-{k}", {
                        "map_nbody": patches[k],
                        "theta": theta32,
                        "cosmo_idx": np.int64(ci),
                        "perm": np.int64(perm),
                        "patch": np.int64(k),
                    }
