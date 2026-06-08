#!/usr/bin/env python3
"""Build the 10deg cross TFDS programmatically (bypasses the `tfds` CLI, which fails to import
apache_beam in this env). Reserializes the intact 10deg grid cache -> TFRecord. No new deps.

The build runs under `if __name__ == "__main__":` — REQUIRED, because the builder's spawn-based
multiprocessing Pool re-imports this module in each worker; without the guard it would re-run
download_and_prepare() recursively (the "freeze_support / Safe importing of main module" error).

Env: CROSS_TFDS_CACHE_DIR (the 10deg grid cache), TFDS_DATA_DIR (output, local XFS),
     CROSS_TFDS_BUILD_WORKERS, CROSS_TFDS_COSMO_LIMIT (optional per-split cosmology cap).
"""
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tf_dataset_nbody_tomo_cross  # noqa: F401  (registers the builder; reads CROSS_TFDS_CACHE_DIR)
import tensorflow_datasets as tfds

NAME = "nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180"


def main():
    data_dir = os.environ.get("TFDS_DATA_DIR", "/home/tersenov/tensorflow_datasets")
    print(f"building {NAME}\n  cache={os.environ.get('CROSS_TFDS_CACHE_DIR')}\n  data_dir={data_dir}",
          flush=True)
    b = tfds.builder(NAME, data_dir=data_dir, file_format="tfrecord")
    b.download_and_prepare()
    print("DONE splits:", {k: v.num_examples for k, v in b.info.splits.items()}, flush=True)


if __name__ == "__main__":
    main()
