#!/usr/bin/env python
"""Build the bnt+deep (5-channel) L1 arm — the §5.4 one-extra-deep-channel test.

PLAN_BNTDEEP_TEST.md. The 5-channel datavector = [cached BNT-auto blocks (800 cols, the
measured 0.15x arm, bit-identical) | fresh deep-channel block (200 cols)]: the L1 datavector
is per-channel blocks by construction, so concatenation IS the 5-channel arm, ceteris
paribus. Deep channel = plain bin average of the ORIGINAL noisy demeaned autos (mix mode
'deep' = (1/4,1/4,1/4,1/4)).

Deep-channel frozen sigma is DERIVED from the verified no-BNT table:
  sigma_deep(s) = (1/4) * sqrt(sum_j sigma_j(s)^2)
exact by wavelet linearity + verified inter-bin noise independence (table GATE, corr +0.003).

Alignment is ASSERTED, not assumed: train/val theta must be bit-equal to the BNT cache's
(NaN-batch skipping makes row order parameter-dependent => identical loader parameters:
train split/perms 5-6/flip/seed 1001/batch 512; val test/0-1/noflip/2001); fiducial
perm/patch arrays must be bit-equal to fiducial_summaries_l1_none.npz.
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
BNT_CACHE = BASE + "/bnt_campaign/l1_matrix/l1_none_cache/flat_local_none_bnt"
BNT_FID = BASE + "/bnt_campaign/fiducial_summaries/fiducial_summaries_l1_none.npz"
FID_OBS = (HERE + "/results/exploratory/cross_maps_campaign/"
           "full_sphere_cache_fiducial_10deg/nobnt/obs")
OUT_CACHE = BASE + "/bntdeep_campaign/l1_matrix/l1_none_cache/flat_local_none_bntdeep"
OUT_FID = BASE + "/bntdeep_campaign/fiducial_summaries/fiducial_summaries_l1_none.npz"
RESO, L1N, NS = 7.5, 40, 5


def main():
    import torch
    from wl_stats_torch import WLStatistics

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stats = WLStatistics(n_scales=NS, device=dev, pixel_arcmin=RESO, dtype=torch.float64)

    # ---- deep-channel sigma, derived from the verified no-BNT table ----
    z = np.load(SIGMA_NOBNT)
    table_mode = str(z["mode"]) if "mode" in z.files else "none"
    assert table_mode == "none", f"need the no-BNT table, got mode={table_mode}"
    sig_auto = np.asarray(z["sigma"], np.float64)[:4]              # (4, NS)
    assert int(z["n_scales"]) == NS
    sigma_deep = 0.25 * np.sqrt((sig_auto ** 2).sum(axis=0))       # (NS,)
    print("############ bnt+deep arm build (§5.4 test) ############")
    print("  derived deep-channel sigma(s) = 1/4*sqrt(sum_j sigma_j^2):")
    print("    " + " ".join(f"{s:.4e}" for s in sigma_deep))
    print("    (per-bin auto sigmas near-equal; deep ≈ sigma_auto/2: "
          + " ".join(f"{r:.4f}" for r in sigma_deep / sig_auto.mean(axis=0)) + ")")
    sig_t = torch.from_numpy(sigma_deep[None]).to(dev, dtype=torch.float64)  # (1, NS)

    # ---- per-channel SNR range, entry-script protocol ----
    ranges_deep = fxl.calibrate_snr_range_flat_local(
        tfds_name=TFDS, data_dir=DDIR, op="none", frozen_sigma=sig_t, stats=stats,
        nbins=4, channel_names=["deep"], n_calibration_examples=20 * 180,
        perm_lo=5, perm_hi=6, subtract_coarse_mean=True, margin=0.05,
        q_lo=0.5, q_hi=99.5, seed=0, bnt="deep",
    )                                                               # (1, 2)

    # ---- train/val deep-only passes (BNT-cache loader parameters EXACTLY) ----
    ds_tr = fxl.compute_l1_dataset_flat_local(
        TFDS, DDIR, "train", "none", sig_t, stats, L1N, ranges_deep,
        perm_lo=5, perm_hi=6, flip=True, seed=1001, batch_size=512,
        subtract_coarse_mean=True, clamp_overflow=True, bnt="deep")
    ds_va = fxl.compute_l1_dataset_flat_local(
        TFDS, DDIR, "test", "none", sig_t, stats, L1N, ranges_deep,
        perm_lo=0, perm_hi=1, flip=False, seed=2001, batch_size=512,
        subtract_coarse_mean=True, clamp_overflow=True, bnt="deep")

    # ---- concat with the cached BNT arm; alignment asserted ----
    bnt_tr = np.load(BNT_CACHE + "/l1_train.npz")
    bnt_va = np.load(BNT_CACHE + "/l1_val.npz")
    for name, mine, theirs in (("train", ds_tr, bnt_tr), ("val", ds_va, bnt_va)):
        a = np.asarray(mine["theta"], np.float64)
        b = np.asarray(theirs["theta"], np.float64)
        assert a.shape == b.shape, f"{name} theta shape {a.shape} != {b.shape}"
        assert np.array_equal(a, b), f"{name} theta NOT bit-equal — row alignment broken"
        print(f"  [align] {name} theta bit-equal over {a.shape} OK")
    x_tr = np.concatenate([bnt_tr["x"], ds_tr["x"]], axis=1).astype(np.float32)
    x_va = np.concatenate([bnt_va["x"], ds_va["x"]], axis=1).astype(np.float32)
    print(f"  concatenated train {x_tr.shape}, val {x_va.shape}  (= [bnt 800 | deep 200])")

    os.makedirs(OUT_CACHE, exist_ok=True)
    np.savez(OUT_CACHE + "/l1_train.npz", theta=bnt_tr["theta"], x=x_tr)
    np.savez(OUT_CACHE + "/l1_val.npz", theta=bnt_va["theta"], x=x_va)
    # ranges: BNT rows are implicit in the cached columns; persist [bnt(4); deep(1)] for
    # provenance (BNT rows sliced from the bnt both-cache ranges, as the arm build did).
    bnt_ranges = np.load(BASE + "/bnt_campaign/l1_matrix/l1_both_cache/"
                         "flat_local_both_bnt/flat_local_ranges.npy")[:4]
    np.save(OUT_CACHE + "/flat_local_ranges.npy",
            np.concatenate([bnt_ranges, ranges_deep], axis=0))
    np.savez(OUT_CACHE + "/l1_cache_meta.npz",
             mode="bnt_deep_concat", parents=np.array([BNT_CACHE, SIGMA_NOBNT]),
             sigma_deep=sigma_deep, ranges_deep=ranges_deep,
             n_l1_channels=5, l1_nbins=L1N, n_scales=NS,
             note="x = [flat_local_none_bnt 800 cols | deep-channel 200 cols]; "
                  "PLAN_BNTDEEP_TEST.md; theta bit-equality asserted at build")

    # ---- fiducial: deep block for all 36000 obs, concat with the BNT fid summaries ----
    files = sorted(glob.glob(FID_OBS + "/cosmo_fiducial_perm*.npz"),
                   key=lambda f: int(f.split("perm")[-1].split(".")[0]))
    print(f"  fiducial deep pass over {len(files)} perm files ...")
    X, perms, patches = [], [], []
    t0 = time.time()
    for i, f in enumerate(files):
        zf = np.load(f)
        autos = zf["patches"][:, :, :, :4].astype(np.float32)
        x = fxl.build_and_l1(autos, "none", sig_t, stats, L1N, ranges_deep,
                             clamp_overflow=True, bnt="deep")        # (180, 200)
        X.append(x)
        p = int(zf["perm"]) if "perm" in zf.files else int(f.split("perm")[-1].split(".")[0])
        perms.append(np.full(x.shape[0], p, np.int32))
        patches.append(np.arange(x.shape[0], dtype=np.int32))
        if (i + 1) % 40 == 0:
            print(f"    {i+1}/{len(files)} ({time.time()-t0:.0f}s)")
    X = np.concatenate(X).astype(np.float32)
    perms = np.concatenate(perms); patches = np.concatenate(patches)

    fz = np.load(BNT_FID)
    assert np.array_equal(fz["perm"], perms), "fiducial perm arrays differ — order broken"
    assert np.array_equal(fz["patch"], patches), "fiducial patch arrays differ — order broken"
    print(f"  [align] fiducial perm/patch bit-equal over {perms.shape} OK")
    S5 = np.concatenate([fz["S"], X], axis=1).astype(np.float32)     # (36000, 1000)
    truth = next(fz[k] for k in ("truth", "theta") if k in fz.files)
    os.makedirs(os.path.dirname(OUT_FID), exist_ok=True)
    np.savez(OUT_FID, S=S5, perm=perms, patch=patches, truth=truth,
             mode="bnt_deep", note="[bnt none 800 | deep 200] per obs")
    print(f"  saved fiducial {S5.shape} -> {OUT_FID}")
    print("BUILD OK")


if __name__ == "__main__":
    main()
