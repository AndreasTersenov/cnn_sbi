#!/usr/bin/env python
"""Build the auto+product+product3 L1 arm (lane D of PLAN_OVERNIGHT_MENU_2.md).

Datavector = [cached flat-local auto+product blocks (2000 cols, the measured 2875 arm,
bit-identical) | fresh product3 block (5 channels x 200 cols: the 4 triple products
kappa_i*kappa_j*kappa_k, i<j<k, + the quadruple)]. Order-3 closure test: registered
decisive band = > +5% pooled FoM3 over 2875 with every science marginal <=; expectation
within seed noise (pair2d ~ l1+product and resolution-beats-order both point there).

sigma rows: EMPIRICAL from the product3 frozen table (freeze_flatsky_cross_noise.py
--op product3) — products of S+N have no analytic shortcut. The table's auto rows are
asserted to match the verified no-BNT table (same seed_base => same realizations).
Alignment asserted as in every concat builder.
"""
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flatsky_cross as fx
import flatsky_cross_l1 as fxl

BASE = HERE + "/results/exploratory/flatsky_cross_2026_06"
TFDS = "nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180"
DDIR = "/home/tersenov/tensorflow_datasets"
SIGMA_NOBNT = BASE + "/flatsky_cross_noise_sigma.npz"
SIGMA_P3 = BASE + "/flatsky_cross_noise_sigma_product3.npz"
PROD_CACHE = BASE + "/l1_matrix/l1_product_cache/flat_local_product"
PROD_FID = BASE + "/gate_c/lc2st/fiducial_summaries_product.npz"
FID_OBS = (HERE + "/results/exploratory/cross_maps_campaign/"
           "full_sphere_cache_fiducial_10deg/nobnt/obs")
RESO, L1N, NS = 7.5, 40, 5


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-cache", required=True)
    ap.add_argument("--out-fid", required=True)
    a = ap.parse_args()

    import torch
    from wl_stats_torch import WLStatistics

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stats = WLStatistics(n_scales=NS, device=dev, pixel_arcmin=RESO, dtype=torch.float64)

    zp = np.load(SIGMA_P3)
    assert str(zp["op"]) == "product3" and int(zp["n_scales"]) == NS, \
        f"need the product3 table (freeze --op product3), got op={zp.get('op')}"
    sigma9 = np.asarray(zp["sigma"], np.float64)               # (9, NS)
    names9 = [str(x) for x in zp["channel_names"]]
    zn = np.load(SIGMA_NOBNT)
    assert np.allclose(sigma9[:4], np.asarray(zn["sigma"], np.float64)[:4], rtol=1e-3), \
        "product3 table auto rows disagree with the verified no-BNT table"
    print("############ auto+product+product3 arm build (lane D) ############")
    print("  [check] product3 table auto rows match the no-BNT table OK")
    for nm, row in zip(names9[4:], sigma9[4:]):
        print(f"    {nm:10s}: " + " ".join(f"{s:.4e}" for s in row))
    sig_t = torch.from_numpy(sigma9).to(dev, dtype=torch.float64)

    ranges9 = fxl.calibrate_snr_range_flat_local(
        tfds_name=TFDS, data_dir=DDIR, op="product3", frozen_sigma=sig_t, stats=stats,
        nbins=4, channel_names=names9, n_calibration_examples=20 * 180,
        perm_lo=5, perm_hi=6, subtract_coarse_mean=True, margin=0.05,
        q_lo=0.5, q_hi=99.5, seed=0)

    p3_cols = np.arange(4 * NS * L1N, 9 * NS * L1N)            # the 5 product3 channels

    ds_tr = fxl.compute_l1_dataset_flat_local(
        TFDS, DDIR, "train", "product3", sig_t, stats, L1N, ranges9,
        perm_lo=5, perm_hi=6, flip=True, seed=1001, batch_size=512,
        subtract_coarse_mean=True, clamp_overflow=True)
    ds_va = fxl.compute_l1_dataset_flat_local(
        TFDS, DDIR, "test", "product3", sig_t, stats, L1N, ranges9,
        perm_lo=0, perm_hi=1, flip=False, seed=2001, batch_size=512,
        subtract_coarse_mean=True, clamp_overflow=True)

    base_tr = np.load(PROD_CACHE + "/l1_train.npz")
    base_va = np.load(PROD_CACHE + "/l1_val.npz")
    for nm, mine, theirs in (("train", ds_tr, base_tr), ("val", ds_va, base_va)):
        th_m = np.asarray(mine["theta"], np.float64)
        th_b = np.asarray(theirs["theta"], np.float64)
        assert th_m.shape == th_b.shape and np.array_equal(th_m, th_b), \
            f"{nm} theta NOT bit-equal — row alignment broken"
        print(f"  [align] {nm} theta bit-equal over {th_m.shape} OK")
    x_tr = np.concatenate([base_tr["x"], ds_tr["x"][:, p3_cols]], axis=1).astype(np.float32)
    x_va = np.concatenate([base_va["x"], ds_va["x"][:, p3_cols]], axis=1).astype(np.float32)
    print(f"  concatenated train {x_tr.shape}, val {x_va.shape} (= [product 2000 | p3 1000])")

    os.makedirs(a.out_cache, exist_ok=True)
    np.savez(a.out_cache + "/l1_train.npz", theta=base_tr["theta"], x=x_tr)
    np.savez(a.out_cache + "/l1_val.npz", theta=base_va["theta"], x=x_va)
    np.savez(a.out_cache + "/l1_cache_meta.npz",
             parents=np.array([PROD_CACHE, SIGMA_P3]), sigma9=sigma9, ranges9=ranges9,
             channel_names=np.array(names9),
             note="x = [flat_local_product 2000 | product3 block 1000]; "
                  "PLAN_OVERNIGHT_MENU_2.md lane D")

    files = sorted(glob.glob(FID_OBS + "/cosmo_fiducial_perm*.npz"),
                   key=lambda f: int(f.split("perm")[-1].split(".")[0]))
    print(f"  fiducial product3 pass over {len(files)} perm files ...")
    X, perms, patches = [], [], []
    t0 = time.time()
    for i, f in enumerate(files):
        zf = np.load(f)
        autos = zf["patches"][:, :, :, :4].astype(np.float32)
        x = fxl.build_and_l1(autos, "product3", sig_t, stats, L1N, ranges9,
                             clamp_overflow=True)
        X.append(x[:, p3_cols])
        p = int(zf["perm"]) if "perm" in zf.files else int(f.split("perm")[-1].split(".")[0])
        perms.append(np.full(x.shape[0], p, np.int32))
        patches.append(np.arange(x.shape[0], dtype=np.int32))
        if (i + 1) % 40 == 0:
            print(f"    {i+1}/{len(files)} ({time.time()-t0:.0f}s)")
    X = np.concatenate(X).astype(np.float32)
    perms = np.concatenate(perms); patches = np.concatenate(patches)

    fz = np.load(PROD_FID)
    assert np.array_equal(fz["perm"], perms), "fiducial perm arrays differ"
    assert np.array_equal(fz["patch"], patches), "fiducial patch arrays differ"
    S = np.concatenate([fz["S"], X], axis=1).astype(np.float32)
    out = {"S": S, "perm": perms, "patch": patches}
    for k in ("truth", "theta"):
        if k in fz.files:
            out[k] = fz[k]
    os.makedirs(os.path.dirname(a.out_fid), exist_ok=True)
    np.savez(a.out_fid, **out)
    print(f"  saved fiducial {S.shape} -> {a.out_fid}")
    print("BUILD OK")


if __name__ == "__main__":
    main()
