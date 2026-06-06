#!/usr/bin/env python
"""Phase A dataset validation for the 10deg L1-vs-CNN campaign (CPU-only).

A1 (scales)   : structure, finiteness, per-channel RMS on a shuffled sample.
A2 (disjoint) : full-scan cosmo_idx/perm/patch audit + train/test disjointness,
                parsing ONLY the int64 scalars (the 256 KB map bytes are read off
                disk but never decoded).

Run CPU-only:  CUDA_VISIBLE_DEVICES= python validate_10deg_dataset.py --check {scales,disjoint,all}
Writes a JSON report under results/exploratory/cross_maps_campaign/validate_10deg/.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import glob
import json
import time

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

TFDS_DIR = (
    "/home/tersenov/tensorflow_datasets/nbody_cosmogrid_dataset_tomo_cross/"
    "grid_10deg_80px_nonoverlap180/0.0.1"
)
OUT_DIR = "results/exploratory/cross_maps_campaign/validate_10deg"

# Expected structure (from the build manifest / 20deg conventions).
EXP = {
    "map_shape": [80, 80, 10],
    "train_count": 1132740,
    "test_count": 504000,
    "train_cosmo_range": [1, 899],
    "test_cosmo_range": [900, 1299],
    "perm_range": [0, 6],
    "patch_range": [0, 179],
    # cache-measured (fiducial, same pipeline): auto 7e-3..1e-2, cross 2e-7..6e-7
    "auto_rms_bounds": [3e-3, 2e-2],
    "cross_rms_bounds": [5e-8, 2e-6],
}


def check_scales(n_per_split=400):
    builder = tfds.builder_from_directory(TFDS_DIR)
    out = {}
    for split in ["train", "test"]:
        ds = builder.as_dataset(split=split, shuffle_files=True).take(n_per_split)
        maps, thetas, cidx = [], [], []
        for ex in tfds.as_numpy(ds):
            maps.append(ex["map_nbody"])
            thetas.append(ex["theta"])
            cidx.append(int(ex["cosmo_idx"]))
        M = np.stack(maps)  # (n, 80, 80, 10)
        rms = np.sqrt((M.astype(np.float64) ** 2).mean(axis=(0, 1, 2)))
        auto_lo, auto_hi = float(rms[:4].min()), float(rms[:4].max())
        cross_lo, cross_hi = float(rms[4:].min()), float(rms[4:].max())
        out[split] = {
            "n": int(M.shape[0]),
            "map_shape": list(M.shape[1:]),
            "dtype": str(M.dtype),
            "all_finite": bool(np.isfinite(M).all()),
            "theta_shape": list(np.asarray(thetas).shape[1:]),
            "per_channel_rms": [float(x) for x in rms],
            "auto_rms_range": [auto_lo, auto_hi],
            "cross_rms_range": [cross_lo, cross_hi],
            "cosmo_idx_sample_range": [min(cidx), max(cidx)],
            "PASS_shape": list(M.shape[1:]) == EXP["map_shape"],
            "PASS_finite": bool(np.isfinite(M).all()),
            "PASS_auto_scale": EXP["auto_rms_bounds"][0] <= auto_lo and auto_hi <= EXP["auto_rms_bounds"][1],
            "PASS_cross_scale": EXP["cross_rms_bounds"][0] <= cross_lo and cross_hi <= EXP["cross_rms_bounds"][1],
            "PASS_ordering": auto_lo > cross_hi,  # every auto channel >> every cross channel
        }
    return out


def check_disjoint(batch=4096, report_every=40):
    spec = {
        "cosmo_idx": tf.io.FixedLenFeature([], tf.int64),
        "perm": tf.io.FixedLenFeature([], tf.int64),
        "patch": tf.io.FixedLenFeature([], tf.int64),
    }
    out = {}
    for split in ["train", "test"]:
        files = sorted(glob.glob(f"{TFDS_DIR}/*-{split}.tfrecord-*"))
        assert files, f"no tfrecord files for split {split}"
        ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
        ds = ds.batch(batch).map(
            lambda b: tf.io.parse_example(b, spec),
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(8)
        cset = set()
        pmin, pmax = 10**9, -1
        patmin, patmax = 10**9, -1
        count = 0
        t0 = time.time()
        for i, p in enumerate(tfds.as_numpy(ds)):
            ci, pe, pa = p["cosmo_idx"], p["perm"], p["patch"]
            cset.update(np.unique(ci).tolist())
            pmin, pmax = min(pmin, int(pe.min())), max(pmax, int(pe.max()))
            patmin, patmax = min(patmin, int(pa.min())), max(patmax, int(pa.max()))
            count += int(ci.shape[0])
            if (i + 1) % report_every == 0:
                dt = time.time() - t0
                print(
                    f"  [{split}] {count:>9,} ex | {len(cset)} cosmos | "
                    f"{count/max(dt,1e-9):,.0f} ex/s | {dt:.0f}s",
                    flush=True,
                )
        out[split] = {
            "count": count,
            "n_cosmo": len(cset),
            "cosmo_min": min(cset),
            "cosmo_max": max(cset),
            "perm_range": [pmin, pmax],
            "patch_range": [patmin, patmax],
            "cosmo_set": sorted(cset),
            "PASS_count": count == EXP[f"{split}_count"],
            "PASS_cosmo_range": [min(cset), max(cset)] == EXP[f"{split}_cosmo_range"],
            "PASS_perm_range": [pmin, pmax] == EXP["perm_range"],
            "PASS_patch_range": [patmin, patmax] == EXP["patch_range"],
            "scan_seconds": round(time.time() - t0, 1),
        }
        print(f"  [{split}] DONE: {count:,} ex, cosmos [{min(cset)},{max(cset)}]", flush=True)
    tr, te = set(out["train"]["cosmo_set"]), set(out["test"]["cosmo_set"])
    inter = sorted(tr & te)
    out["disjoint"] = {"train_inter_test": inter, "PASS_disjoint": len(inter) == 0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", choices=["scales", "disjoint", "all"], default="all")
    ap.add_argument("--n-per-split", type=int, default=400)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    report = {"tfds_dir": TFDS_DIR, "expected": EXP, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    if args.check in ("scales", "all"):
        print("== A1 scales ==", flush=True)
        report["scales"] = check_scales(args.n_per_split)
        print(json.dumps(report["scales"], indent=2), flush=True)

    if args.check in ("disjoint", "all"):
        print("== A2 disjoint (full scan) ==", flush=True)
        report["disjoint"] = check_disjoint()

    out_path = os.path.join(OUT_DIR, f"report_{args.check}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)

    # Terminal PASS/FAIL banner
    fails = []
    for sp, d in report.get("scales", {}).items():
        for k, v in d.items():
            if k.startswith("PASS_") and not v:
                fails.append(f"scales.{sp}.{k}")
    dj = report.get("disjoint", {})
    for sp in ("train", "test"):
        for k, v in dj.get(sp, {}).items():
            if k.startswith("PASS_") and not v:
                fails.append(f"disjoint.{sp}.{k}")
    if "disjoint" in dj and not dj["disjoint"].get("PASS_disjoint", True):
        fails.append("disjoint.train_inter_test")
    print("\n=== RESULT:", "ALL PASS" if not fails else f"FAIL: {fails}", "===", flush=True)


if __name__ == "__main__":
    main()
