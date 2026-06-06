#!/usr/bin/env python
"""Phase A3 -- independent bit-exactness oracle for the 10deg TFDS.

The build logged `bit-exact verified (ci=703,perm=3,patch=118)` against the
(now-deleted) grid cache. This re-derives that exact patch FRESH from CosmoGridV1
raw via build_full_sphere_cross_cache.py and bit-matches it against the TFDS
example -- an oracle independent of the build run's own cache+TFDS pipeline.

ci=703 -> grid row 703 -> cosmo_001814 ; noise seed = 12345 + 100*703 + 3.
CPU-only. Builds one (cosmo,perm) into a temp dir, compares, cleans up.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import glob
import json
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import tensorflow as tf

TFDS_DIR = (
    "/home/tersenov/tensorflow_datasets/nbody_cosmogrid_dataset_tomo_cross/"
    "grid_10deg_80px_nonoverlap180/0.0.1"
)
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
CI, PERM, PATCH, COSMO_ID = 703, 3, 118, "cosmo_001814"
OUT_DIR = "results/exploratory/cross_maps_campaign/validate_10deg"


def fetch_tfds_reference():
    """Pull the (ci,perm,patch) example's map from the TFDS train shards."""
    spec = {
        "cosmo_idx": tf.io.FixedLenFeature([], tf.int64),
        "perm": tf.io.FixedLenFeature([], tf.int64),
        "patch": tf.io.FixedLenFeature([], tf.int64),
        "map_nbody": tf.io.FixedLenFeature([80 * 80 * 10], tf.float32),
    }
    files = sorted(glob.glob(f"{TFDS_DIR}/*-train.tfrecord-*"))
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=8)
    for rec in ds:
        p = tf.io.parse_single_example(rec, spec)
        if int(p["cosmo_idx"]) == CI and int(p["perm"]) == PERM and int(p["patch"]) == PATCH:
            m = p["map_nbody"].numpy().reshape(80, 80, 10)
            return m
    raise SystemExit(f"triple (ci={CI},perm={PERM},patch={PATCH}) not found in TFDS")


def rederive_patch(tmp):
    cmd = [
        PY, "build_full_sphere_cross_cache.py",
        "--cosmo-subset", "grid", "--cosmo-id", COSMO_ID,
        "--realizations", str(PERM), "--regime", "nobnt",
        "--field-size", "10", "--field-npix", "80",
        "--n-centers", "180", "--center-nside", "64",
        "--min-separation-deg", "14.2", "--max-abs-lat", "75",
        "--num-workers", "1", "--out-dir", tmp,
    ]
    print("re-derive:", " ".join(cmd), flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(r.stdout[-2000:], flush=True)
    if r.returncode != 0:
        print(r.stderr[-3000:], flush=True)
        raise SystemExit(f"build failed rc={r.returncode}")
    print(f"  SHT re-derive took {time.time()-t0:.0f}s", flush=True)
    npz = os.path.join(tmp, "nobnt", "train", f"{COSMO_ID}_perm{PERM}.npz")
    z = np.load(npz, allow_pickle=True)
    assert int(z["cosmo_idx"]) == CI, f"rebuilt cosmo_idx {int(z['cosmo_idx'])} != {CI}"
    return np.asarray(z["patches"])[PATCH]  # (80,80,10)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("== A3 bit-match: fetch TFDS reference ==", flush=True)
    ref = fetch_tfds_reference()
    print(f"  TFDS map shape {ref.shape}, finite={np.isfinite(ref).all()}", flush=True)

    tmp = tempfile.mkdtemp(prefix="bitmatch10_", dir=".")
    try:
        got = rederive_patch(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    exact = bool(np.array_equal(ref, got))
    maxdiff = float(np.abs(ref.astype(np.float64) - got.astype(np.float64)).max())
    per_ch = [float(np.abs(ref[..., c].astype(np.float64) - got[..., c]).max()) for c in range(10)]
    report = {
        "triple": {"cosmo_idx": CI, "perm": PERM, "patch": PATCH, "cosmo_id": COSMO_ID},
        "bit_exact": exact,
        "max_abs_diff": maxdiff,
        "per_channel_max_abs_diff": per_ch,
        "ref_shape": list(ref.shape),
        "PASS": exact or maxdiff == 0.0,
    }
    with open(os.path.join(OUT_DIR, "report_bitmatch.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    print("\n=== A3 RESULT:", "BIT-EXACT PASS" if report["PASS"] else f"FAIL maxdiff={maxdiff}", "===", flush=True)


if __name__ == "__main__":
    main()
