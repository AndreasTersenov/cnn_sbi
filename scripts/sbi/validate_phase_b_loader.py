#!/usr/bin/env python
"""CPU unit checks for the Phase B-1 `tfds_cross` loader (gate the GPU smoke).

A. channel-RMS from a TFDS sample is in the Phase-A bounds (auto >> cross).
B. perm-split: every perm 0-6 covers all 899 cosmos -> compressor(0-4)/NDE(5-6)
   are EXAMPLE-disjoint with full cosmology coverage (one full train scan).
C. finite loader contract: iter_cross_tfds_batches yields (B,80,80,10) maps scaled
   to ~unit RMS and theta with H0->h0 applied.

Run CPU-only: CUDA_VISIBLE_DEVICES= python validate_phase_b_loader.py
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import glob
import json
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TFDS_NAME = "nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180"
DATA_DIR = "/home/tersenov/tensorflow_datasets"
RAW_DIR = f"{DATA_DIR}/{TFDS_NAME.split('/')[0]}/{TFDS_NAME.split('/')[1]}/0.0.1"
COMP = (0, 4)
NDE = (5, 6)
OUT = "results/exploratory/cross_maps_campaign/validate_10deg/report_phase_b_loader.json"


def check_rms():
    from tfds_cross_tfdata_loader import compute_cross_tfds_channel_rms
    rms = compute_cross_tfds_channel_rms(TFDS_NAME, DATA_DIR, "train", n_sample=8000)
    auto, cross = rms[:4], rms[4:]
    ok = bool(
        3e-3 <= auto.min() and auto.max() <= 2e-2
        and 5e-8 <= cross.min() and cross.max() <= 2e-6
        and auto.min() > cross.max()
    )
    print("[A] channel-RMS (auto then cross): " + ", ".join(f"{x:.3e}" for x in rms))
    print(
        f"    auto [{auto.min():.2e},{auto.max():.2e}] cross "
        f"[{cross.min():.2e},{cross.max():.2e}] -> {'PASS' if ok else 'FAIL'}"
    )
    return ok, [float(x) for x in rms]


def check_perm_filter():
    import tensorflow as tf
    import tensorflow_datasets as tfds

    spec = {
        "cosmo_idx": tf.io.FixedLenFeature([], tf.int64),
        "perm": tf.io.FixedLenFeature([], tf.int64),
        "patch": tf.io.FixedLenFeature([], tf.int64),
    }
    files = sorted(glob.glob(f"{RAW_DIR}/*-train.tfrecord-*"))
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.batch(4096).map(
        lambda b: tf.io.parse_example(b, spec), num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(8)
    per_perm_cosmo = {p: set() for p in range(7)}
    per_perm_count = {p: 0 for p in range(7)}
    t0 = time.time()
    for b in tfds.as_numpy(ds):
        ci, pe = b["cosmo_idx"], b["perm"]
        for p in range(7):
            m = pe == p
            if m.any():
                per_perm_cosmo[p].update(np.unique(ci[m]).tolist())
                per_perm_count[p] += int(m.sum())
    comp_cosmo = set().union(*(per_perm_cosmo[p] for p in range(COMP[0], COMP[1] + 1)))
    nde_cosmo = set().union(*(per_perm_cosmo[p] for p in range(NDE[0], NDE[1] + 1)))
    comp_count = sum(per_perm_count[p] for p in range(COMP[0], COMP[1] + 1))
    nde_count = sum(per_perm_count[p] for p in range(NDE[0], NDE[1] + 1))
    full = set(range(1, 900))
    all_perms_full = all(per_perm_cosmo[p] == full for p in range(7))
    ok = bool(all_perms_full and comp_cosmo == full and nde_cosmo == full)
    print(f"[B] perm-split scan {time.time() - t0:.0f}s")
    print("    per-perm #cosmos: " + ", ".join(f"p{p}={len(per_perm_cosmo[p])}" for p in range(7)))
    print(f"    per-perm #ex: " + ", ".join(f"p{p}={per_perm_count[p]}" for p in range(7)))
    print(
        f"    compressor perms {COMP}: {len(comp_cosmo)} cosmos, {comp_count} ex | "
        f"NDE perms {NDE}: {len(nde_cosmo)} cosmos, {nde_count} ex"
    )
    print(
        f"    example-disjoint by construction; all 899 cosmos in both streams: "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok, {
        "per_perm_cosmos": {p: len(per_perm_cosmo[p]) for p in range(7)},
        "per_perm_count": per_perm_count,
        "compressor_cosmos": len(comp_cosmo),
        "nde_cosmos": len(nde_cosmo),
        "compressor_examples": comp_count,
        "nde_examples": nde_count,
    }


def check_loader_contract(rms):
    from tfds_cross_tfdata_loader import iter_cross_tfds_batches
    rms = np.asarray(rms, np.float32)
    it = iter_cross_tfds_batches(
        TFDS_NAME, DATA_DIR, "train", batch_size=64, seed=0, flip=False,
        channel_scale=rms, channel_slice=None, perm_lo=NDE[0], perm_hi=NDE[1],
    )
    maps, theta = next(it)
    per_ch_rms = np.sqrt((maps.astype(np.float64) ** 2).mean(axis=(0, 1, 2)))
    ok = bool(
        maps.shape[1:] == (80, 80, 10)
        and theta.shape[1] == 6
        and np.isfinite(maps).all()
        and 0.3 < per_ch_rms.mean() < 3.0          # scaled to ~unit RMS
        and float(theta[:, 3].max()) < 1.5         # H0/100 -> h0 applied (raw ~67)
        and 0.1 < float(theta[:, 0].mean()) < 0.6  # Om in a sane grid range
    )
    print(f"[C] loader contract: maps {maps.shape} theta {theta.shape}")
    print(
        f"    scaled per-ch RMS mean={per_ch_rms.mean():.3f} (expect ~1) | "
        f"theta[3](h0) max={float(theta[:, 3].max()):.3f} (<1.5) | "
        f"theta[0](Om) mean={float(theta[:, 0].mean()):.3f} -> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rms_ok, rms = check_rms()
    perm_ok, perm_info = check_perm_filter()
    contract_ok = check_loader_contract(rms)
    report = {
        "rms": rms,
        "PASS_rms": rms_ok,
        "perm_split": perm_info,
        "PASS_perm_split": perm_ok,
        "PASS_loader_contract": contract_ok,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    allpass = rms_ok and perm_ok and contract_ok
    print(f"\nWrote {OUT}")
    print(f"=== RESULT: {'ALL PASS' if allpass else 'FAIL'} ===")
    sys.exit(0 if allpass else 1)


if __name__ == "__main__":
    main()
