#!/usr/bin/env python
"""Phase B-2 parity checks for the L1 tfds_cross loader (CPU-only).

Guards the known `auto_scalar` regression (feedback_l1_cross_must_use_harmonic_route):
A. σ_c from the TFDS (new route) is CHANNEL-AWARE (10 distinct values, auto≫cross) and
   in the same ballpark as σ_c from the proven fiducial-cache path.
B. Map parity: the new route reads the SAME maps as the cache path and applies channel_scale
   (MULTIPLY, L1 convention) identically — re-derive cosmo_001814/perm3 (=cosmo_idx 703) and
   bit-match the TFDS-read vs cache-read patches, raw and σ_c-scaled.

Run CPU-only: CUDA_VISIBLE_DEVICES= python validate_phase_b2_l1.py
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import glob
import json
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import npe_l1norm_cross_jaxili_nbody_tomo as L1  # noqa: E402

TFDS_NAME = "nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180"
DATA_DIR = "/home/tersenov/tensorflow_datasets"
RAW_DIR = f"{DATA_DIR}/{TFDS_NAME.split('/')[0]}/{TFDS_NAME.split('/')[1]}/0.0.1"
FID = "results/exploratory/cross_maps_campaign/full_sphere_cache_fiducial_10deg"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
CI, PERM, COSMO_ID = 703, 3, "cosmo_001814"
OUT = "results/exploratory/cross_maps_campaign/validate_10deg/report_phase_b2_l1.json"


def sigma_from_fiducial_cache():
    """σ_c (per-channel std) over the fiducial cache obs patches (proven path)."""
    accum_sq = accum_sum = None
    n = 0
    for maps_np, _theta, _p in L1.iter_harmonic_examples(
        cache_dir=__import__("pathlib").Path(FID), regime="nobnt", split="obs",
        rng=np.random.default_rng(0), flip=False, n_take=16,
    ):
        flat = maps_np.reshape(-1, maps_np.shape[-1]).astype(np.float64)
        accum_sq = (flat ** 2).sum(0) if accum_sq is None else accum_sq + (flat ** 2).sum(0)
        accum_sum = flat.sum(0) if accum_sum is None else accum_sum + flat.sum(0)
        n += flat.shape[0]
    mean = accum_sum / n
    return np.sqrt(np.maximum(accum_sq / n - mean ** 2, 0.0)).astype(np.float32)


def check_sigma():
    sig_tfds = L1.calibrate_channel_noise_sigma_from_cross_tfds(
        TFDS_NAME, DATA_DIR, n_calibration_examples=5760, seed=7717
    )
    sig_fid = sigma_from_fiducial_cache()
    auto_aware = float(sig_tfds[:4].min()) > float(sig_tfds[4:].max())
    distinct = len(set(np.round(sig_tfds, 12))) >= 8
    ratio = sig_tfds / np.maximum(sig_fid, 1e-30)
    ballpark = bool(np.all(ratio > 0.5) and np.all(ratio < 2.0))
    ok = bool(auto_aware and distinct and ballpark)
    print("[A] σ_c parity (TFDS grid vs fiducial cache):")
    for c in range(10):
        print(f"    ch{c}: tfds={sig_tfds[c]:.4e}  fid={sig_fid[c]:.4e}  ratio={ratio[c]:.2f}")
    print(f"    channel-aware(auto≫cross)={auto_aware}  distinct={distinct}  "
          f"ballpark(0.5–2×)={ballpark} -> {'PASS' if ok else 'FAIL'}")
    return ok, sig_tfds.tolist(), sig_fid.tolist()


def rederive(tmp):
    cmd = [
        PY, "build_full_sphere_cross_cache.py", "--cosmo-subset", "grid",
        "--cosmo-id", COSMO_ID, "--realizations", str(PERM), "--regime", "nobnt",
        "--field-size", "10", "--field-npix", "80", "--n-centers", "180",
        "--center-nside", "64", "--min-separation-deg", "14.2", "--max-abs-lat", "75",
        "--num-workers", "1", "--out-dir", tmp,
    ]
    r = subprocess.run(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:]); raise SystemExit("re-derive failed")
    return np.asarray(np.load(
        os.path.join(tmp, "nobnt", "train", f"{COSMO_ID}_perm{PERM}.npz"),
        allow_pickle=True)["patches"])  # (180,80,80,10)


def tfds_patches_for(ci, perm):
    """Read the (ci,perm) patches from the TFDS, ordered by patch index."""
    import tensorflow as tf
    import tensorflow_datasets as tfds
    spec = {
        "cosmo_idx": tf.io.FixedLenFeature([], tf.int64),
        "perm": tf.io.FixedLenFeature([], tf.int64),
        "patch": tf.io.FixedLenFeature([], tf.int64),
        "map_nbody": tf.io.FixedLenFeature([80 * 80 * 10], tf.float32),
    }
    files = sorted(glob.glob(f"{RAW_DIR}/*-train.tfrecord-*"))
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=8)
    out = {}
    for rec in ds:
        p = tf.io.parse_single_example(rec, spec)
        if int(p["cosmo_idx"]) == ci and int(p["perm"]) == perm:
            out[int(p["patch"])] = p["map_nbody"].numpy().reshape(80, 80, 10)
            if len(out) == 180:
                break
    return np.stack([out[i] for i in range(len(out))])


def check_map_parity(sigma_c):
    tmp = tempfile.mkdtemp(prefix="b2parity_", dir=".")
    try:
        cache_p = rederive(tmp)                 # (180,80,80,10)
    finally:
        pass
    t0 = time.time()
    tfds_p = tfds_patches_for(CI, PERM)
    shutil.rmtree(tmp, ignore_errors=True)
    raw_ok = bool(cache_p.shape == tfds_p.shape and np.array_equal(cache_p, tfds_p))
    # L1 channel_scale = noise_sigma/σ_c, capped to 1.0 on the 4 auto channels (MULTIPLY).
    noise_sigma = L1.pixel_noise_sigma(0.26, 10.0, 10.0, 80)
    scale = (float(noise_sigma) / np.maximum(np.asarray(sigma_c), 1e-30)).astype(np.float32)
    scale[:4] = 1.0
    scaled_ok = bool(np.array_equal((cache_p * scale).astype(np.float32),
                                    (tfds_p * scale).astype(np.float32)))
    print(f"[B] map parity (cosmo_idx {CI}=perm {PERM}, {tfds_p.shape[0]} patches, "
          f"tfds-read {time.time()-t0:.0f}s):")
    print(f"    raw bit-match={raw_ok}  σ_c-scaled bit-match={scaled_ok} "
          f"(maxdiff raw={np.abs(cache_p-tfds_p).max():.2e}) -> "
          f"{'PASS' if (raw_ok and scaled_ok) else 'FAIL'}")
    return bool(raw_ok and scaled_ok)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sig_ok, sig_tfds, sig_fid = check_sigma()
    map_ok = check_map_parity(sig_tfds)
    report = {"PASS_sigma": sig_ok, "PASS_map_parity": map_ok,
              "sigma_tfds": sig_tfds, "sigma_fiducial": sig_fid,
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    allpass = sig_ok and map_ok
    print(f"\nWrote {OUT}\n=== RESULT: {'ALL PASS' if allpass else 'FAIL'} ===")
    sys.exit(0 if allpass else 1)


if __name__ == "__main__":
    main()
