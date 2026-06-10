#!/usr/bin/env python3
"""Precompute the 'both' (16ch) flat-local L1 datavector for EVERY fiducial obs patch.

Population-sweep prep (the 9000-obs/arm headline). Reads the fiducial obs cache (200 perms x
180 patches = 36000), builds the patch-local conv+product cross on-device, and computes the
both L1 datavector with the FROZEN per-(channel,scale) sigma + the calibrated per-channel
ranges (flat_local_ranges.npy from the 'both' training build) — the EXACT inference datavector.
The population sweep then column-slices this per arm (none/conv/product/both) and evaluates
each trained posterior. Reads the small fiducial cache (NOT the 290GB TFDS) => no contention
with the training-datavector loader pass on the other GPU.

Run on GPU 2:  CUDA_VISIBLE_DEVICES=2 python precompute_fiducial_both_datavectors.py
"""
import os, sys, glob, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flatsky_cross as fx
import flatsky_cross_l1 as fxl

FC = (HERE + "/results/exploratory/cross_maps_campaign/"
      "full_sphere_cache_fiducial_10deg/nobnt/obs")
BOTH_CACHE = HERE + "/results/exploratory/flatsky_cross_2026_06/l1_matrix/l1_both_cache/flat_local_both"
SIG = HERE + "/results/exploratory/flatsky_cross_2026_06/flatsky_cross_noise_sigma.npz"
OUT = HERE + "/results/exploratory/flatsky_cross_2026_06/fiducial_both_datavectors.npz"
RESO, L1N, NB, NS = 7.5, 40, 4, 5


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--both-cache", default=BOTH_CACHE,
                    help="'both' training-build cache dir (flat_local_ranges.npy source).")
    ap.add_argument("--sigma", default=SIG,
                    help="frozen noise sigma npz (the BNT table when --bnt).")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--bnt", action="store_true",
                    help="BNT the obs autos on-device before the channel build (must match "
                         "the training build; the sigma table's bnt key is enforced).")
    a = ap.parse_args()

    import torch
    from wl_stats_torch import WLStatistics
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ranges = np.load(a.both_cache + "/flat_local_ranges.npy")        # (16, 2) calibrated, frozen
    assert ranges.shape == (16, 2), ranges.shape
    stats = WLStatistics(n_scales=NS, device=dev, pixel_arcmin=RESO, dtype=torch.float64)
    sig, names, _ = fxl.select_frozen_sigma(a.sigma, "both", NB, dev, expected_bnt=a.bnt)
    files = sorted(glob.glob(FC + "/cosmo_fiducial_perm*.npz"),
                   key=lambda f: int(f.split("perm")[-1].split(".")[0]))
    print(f"############ precompute fiducial 'both' datavectors (bnt={a.bnt}) ############")
    print(f"  {len(files)} perm files, ranges from {a.both_cache}")

    X, perms, patches = [], [], []
    truth = None
    t0 = time.time()
    for i, f in enumerate(files):
        z = np.load(f)
        autos = z["patches"][:, :, :, :4].astype(np.float32)        # (180,80,80,4)
        x = fxl.build_and_l1(autos, "both", sig, stats, L1N, ranges, clamp_overflow=True,
                             bnt=a.bnt)  # (180,3200)
        X.append(x)
        p = int(z["perm"]) if "perm" in z.files else int(f.split("perm")[-1].split(".")[0])
        perms.append(np.full(x.shape[0], p, np.int32))
        patches.append(np.arange(x.shape[0], dtype=np.int32))
        if truth is None:
            th = np.asarray(z["theta"], np.float64).copy(); th[3] /= 100.0  # h0
            truth = th
        if (i + 1) % 40 == 0:
            n = (i + 1) * 180
            print(f"    {i+1}/{len(files)} perms ({n} patches, {time.time()-t0:.0f}s, "
                  f"{n/(time.time()-t0):.0f}/s)")
    X = np.concatenate(X).astype(np.float32)                        # (36000, 3200)
    perms = np.concatenate(perms); patches = np.concatenate(patches)
    np.savez(a.out, x=X, perm=perms, patch=patches, truth=truth,
             channel_names=np.array(names), n_scales=NS, l1_nbins=L1N,
             ranges=ranges, bnt=bool(a.bnt),
             note="both(16ch) flat-local obs datavectors; slice per arm via op_feature_columns")
    print(f"  saved {X.shape} -> {a.out}  ({time.time()-t0:.0f}s); truth={truth}")


if __name__ == "__main__":
    main()
