#!/usr/bin/env python
"""Build a 2D-1D Haar SCATTERING ℓ1 arm (Phase 2 / Approach B of TOMO_2D1D_WAVELET_RESEARCH.md).

2D starlet → |·| → 1D Haar across the 4 (pre-mixed) channels → S/N-binned ℓ1, with the EMPIRICAL
per-(mode,scale) noise σ from freeze_haar_scatter_noise.py. Same loader params as build_flatsky_haar_arm.py
(theta bit-aligned with flat_none) and the same downstream common-MAF sweep + TARP/SBC gate.

  --pre-basis none  P=I (no-BNT scattering)      --pre-basis bnt  P=B (scattering in BNT space)
--smoke: calibrate small + build on 1 fiducial file, print shape/NaN, exit before the full pass.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flatsky_cross as fx
import flatsky_haar_scatter as hs

BASE = HERE + "/results/exploratory/flatsky_cross_2026_06"
TFDS = "nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180"
DDIR = "/home/tersenov/tensorflow_datasets"
FID_OBS = (HERE + "/results/exploratory/cross_maps_campaign/"
           "full_sphere_cache_fiducial_10deg/nobnt/obs")
RESO, L1N, NS = 7.5, 40, 5
MODE_NAMES = ["scat_deep", "scat_coarse", "scat_d12", "scat_d34"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-basis", choices=("none", "bnt"), required=True)
    ap.add_argument("--out-cache", required=True)
    ap.add_argument("--out-fid", required=True)
    ap.add_argument("--base-fid", default=BASE + "/gate_c/lc2st/fiducial_summaries_none.npz")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    import torch
    from wl_stats_torch import WLStatistics

    pre = (a.pre_basis if a.pre_basis == "bnt" else False)
    sig_npz = BASE + f"/flatsky_haar_scatter_sigma_{a.pre_basis}.npz"
    z = np.load(sig_npz)
    sigma_modes = np.asarray(z["sigma"], np.float64)             # (4, NS)
    assert int(z["n_scales"]) == NS and sigma_modes.shape == (4, NS), sigma_modes.shape
    print(f"############ Haar-scatter arm build: pre_basis={a.pre_basis} ############", flush=True)
    for m in range(4):
        print(f"    sigma {MODE_NAMES[m]:12s}: " + " ".join(f"{s:.4e}" for s in sigma_modes[m]), flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stats = WLStatistics(n_scales=NS, device=dev, pixel_arcmin=RESO, dtype=torch.float64)
    H = torch.from_numpy(hs.haar4()).to(dev, dtype=torch.float64)
    sig_t = torch.from_numpy(sigma_modes).to(dev, dtype=torch.float64)

    ncal = (2 * 180) if a.smoke else (20 * 180)
    ranges = hs.calibrate_scatter_snr_range(
        TFDS, DDIR, pre, H, sig_t, stats, n_calibration_examples=ncal,
        perm_lo=5, perm_hi=6, subtract_coarse_mean=True, margin=0.05,
        q_lo=0.5, q_hi=99.5, seed=0, mode_names=MODE_NAMES)

    if a.smoke:
        f = sorted(glob.glob(FID_OBS + "/cosmo_fiducial_perm*.npz"))[0]
        autos = np.load(f)["patches"][:, :, :, :4].astype(np.float32)
        x = hs.build_and_scatter_l1(autos, pre, H, sig_t, stats, L1N, ranges, clamp_overflow=True)
        exp = 4 * NS * L1N
        print(f"  [smoke] x{x.shape} (expect (*, {exp})) NaN={np.isnan(x).any()} "
              f"finite={np.isfinite(x).mean():.4f}", flush=True)
        assert x.shape[1] == exp and not np.isnan(x).any(), "smoke FAIL"
        print("SMOKE OK", flush=True); return

    ds_tr = hs.compute_scatter_dataset(TFDS, DDIR, "train", pre, H, sig_t, stats, L1N, ranges,
                                       perm_lo=5, perm_hi=6, flip=True, seed=1001, batch_size=512)
    ds_va = hs.compute_scatter_dataset(TFDS, DDIR, "test", pre, H, sig_t, stats, L1N, ranges,
                                       perm_lo=0, perm_hi=1, flip=False, seed=2001, batch_size=512)

    fn = BASE + "/l1_matrix/l1_none_cache/flat_local_none"
    for nm, mine, base in (("train", ds_tr, fn + "/l1_train.npz"), ("val", ds_va, fn + "/l1_val.npz")):
        tb = np.load(base)["theta"].astype(np.float64); tm = np.asarray(mine["theta"], np.float64)
        assert tm.shape == tb.shape and np.array_equal(tm, tb), f"{nm} theta NOT bit-equal vs flat_none"
        print(f"  [align] {nm} theta bit-equal vs flat_none {tm.shape} OK", flush=True)

    os.makedirs(a.out_cache, exist_ok=True)
    np.savez(a.out_cache + "/l1_train.npz", theta=ds_tr["theta"], x=ds_tr["x"].astype(np.float32))
    np.savez(a.out_cache + "/l1_val.npz", theta=ds_va["theta"], x=ds_va["x"].astype(np.float32))
    np.savez(a.out_cache + "/l1_cache_meta.npz", pre_basis=a.pre_basis, sigma_modes=sigma_modes,
             mode_names=np.array(MODE_NAMES), ranges=ranges, note="2D-1D Haar scattering ℓ1 (Phase 2)")
    print(f"  saved cache train x{ds_tr['x'].shape}, val x{ds_va['x'].shape}", flush=True)

    files = sorted(glob.glob(FID_OBS + "/cosmo_fiducial_perm*.npz"),
                   key=lambda f: int(f.split("perm")[-1].split(".")[0]))
    print(f"  fiducial pass over {len(files)} files ...", flush=True)
    X, perms, patches = [], [], []
    t0 = time.time()
    for i, f in enumerate(files):
        zf = np.load(f); autos = zf["patches"][:, :, :, :4].astype(np.float32)
        x = hs.build_and_scatter_l1(autos, pre, H, sig_t, stats, L1N, ranges, clamp_overflow=True)
        X.append(x)
        p = int(zf["perm"]) if "perm" in zf.files else int(f.split("perm")[-1].split(".")[0])
        perms.append(np.full(x.shape[0], p, np.int32)); patches.append(np.arange(x.shape[0], dtype=np.int32))
        if (i + 1) % 40 == 0:
            print(f"    {i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    X = np.concatenate(X).astype(np.float32); perms = np.concatenate(perms); patches = np.concatenate(patches)
    fz = np.load(a.base_fid)
    assert np.array_equal(fz["perm"], perms) and np.array_equal(fz["patch"], patches), "fiducial align"
    out = {"S": X, "perm": perms, "patch": patches}
    for k in ("truth", "theta"):
        if k in fz.files:
            out[k] = fz[k]
    os.makedirs(os.path.dirname(a.out_fid), exist_ok=True)
    np.savez(a.out_fid, **out)
    print(f"  saved fiducial S{X.shape} -> {a.out_fid}\nBUILD OK", flush=True)


if __name__ == "__main__":
    main()
