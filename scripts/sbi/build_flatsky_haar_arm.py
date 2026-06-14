#!/usr/bin/env python
"""Build a 2D-1D Haar wavelet-ℓ1 arm (Approach A of TOMO_2D1D_WAVELET_RESEARCH.md).

A fixed mix matrix M over the 4 autos (op='none'), then the frozen-σ wavelet ℓ1. Because the
spatial starlet and the bin-mix are both linear, this IS the 2D-1D wavelet ℓ1-norm with a Haar
along the bin axis (m0 deep mode ¼Σκ + m1 coarse diff + m2/m3 fine diffs). Per-channel σ by
quadrature from the verified no-BNT frozen table (σ²_m = Σ_b M[m,b]² σ²_auto,b — independent
per-bin shape noise). Reuses the EXACT flatsky_cross_l1 calls of build_flatsky_postcut_arm.py
(same loader params ⇒ theta bit-aligned with flat_none), minus the BNT cut/assemble machinery.

  --mix haar      M = orthonormal 4×4 Haar over autos (no BNT).               [A1]
  --mix haar_bnt  M = Haar · B  (Haar across uncut BNT channels).             [A2 uncut]

Output: <out-cache>/l1_{train,val}.npz (theta + Haar-channel ℓ1) + <out-fid> fiducial summaries.
(autohaar arm = concatenate the flat_none cache with this one afterwards — separate trivial step.)

--smoke: calibrate on a tiny set + run build_and_l1 on ONE fiducial file, print shape/NaN/σ, exit
BEFORE the full 324k train pass (catches arg/import/shape bugs cheaply).
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
import flatsky_cross_l1 as fxl

BASE = HERE + "/results/exploratory/flatsky_cross_2026_06"
TFDS = "nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180"
DDIR = "/home/tersenov/tensorflow_datasets"
SIGMA_NOBNT = BASE + "/flatsky_cross_noise_sigma.npz"
FID_OBS = (HERE + "/results/exploratory/cross_maps_campaign/"
           "full_sphere_cache_fiducial_10deg/nobnt/obs")
RESO, L1N, NS = 7.5, 40, 5


def haar_4() -> np.ndarray:
    """Orthonormal 2-level Haar over the 4 autos; row 0 = deep mode ¼Σκ (×2)."""
    s = 1.0 / np.sqrt(2.0)
    return np.array([[0.5, 0.5, 0.5, 0.5],
                     [0.5, 0.5, -0.5, -0.5],
                     [s, -s, 0.0, 0.0],
                     [0.0, 0.0, s, -s]], dtype=np.float64)


def mix_for(name: str) -> tuple[np.ndarray, list[str]]:
    H = haar_4()
    names = ["haar_deep", "haar_coarse", "haar_d12", "haar_d34"]
    if name == "haar":
        return H, names
    if name == "haar_bnt":
        B = fx.bnt_matrix_np().astype(np.float64)            # (4,4) BNT over autos
        return (H @ B), [f"{n}_bnt" for n in names]          # Haar across BNT channels
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", choices=("haar", "haar_bnt"), required=True)
    ap.add_argument("--out-cache", required=True)
    ap.add_argument("--out-fid", required=True)
    ap.add_argument("--base-fid", default=BASE + "/gate_c/lc2st/fiducial_summaries_none.npz",
                    help="fiducial summaries to copy perm/patch/truth from (alignment check)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    import torch
    from wl_stats_torch import WLStatistics

    M, uname = mix_for(a.mix)
    print(f"############ 2D-1D Haar arm build: mix={a.mix} ############", flush=True)
    print(f"  mix matrix M (rows=output channels over the 4 autos):\n{M}", flush=True)
    # orthonormality / structure sanity (Haar itself is orthonormal; Haar·B is not, that's fine)
    if a.mix == "haar":
        assert np.allclose(M @ M.T, np.eye(4), atol=1e-12), "Haar not orthonormal"
        print("  [check] Haar orthonormal OK", flush=True)

    z = np.load(SIGMA_NOBNT)
    assert (str(z["mode"]) if "mode" in z.files else "none") == "none", "expected no-BNT σ table"
    sig_auto = np.asarray(z["sigma"], np.float64)[:4]        # (4, NS)
    assert int(z["n_scales"]) == NS
    sigma_u = np.sqrt((M ** 2) @ (sig_auto ** 2))            # (C, NS) quadrature over autos
    assert np.all(np.isfinite(sigma_u)) and np.all(sigma_u > 0), "bad sigma_u"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sig_t = torch.from_numpy(sigma_u).to(dev, dtype=torch.float64)
    for k, row in enumerate(sigma_u):
        print(f"    sigma {uname[k]:14s}: " + " ".join(f"{s:.4e}" for s in row), flush=True)

    stats = WLStatistics(n_scales=NS, device=dev, pixel_arcmin=RESO, dtype=torch.float64)
    Mf = M.astype(np.float64)

    ncal = (2 * 180) if a.smoke else (20 * 180)
    ranges = fxl.calibrate_snr_range_flat_local(
        tfds_name=TFDS, data_dir=DDIR, op="none", frozen_sigma=sig_t, stats=stats,
        nbins=4, channel_names=uname, n_calibration_examples=ncal,
        perm_lo=5, perm_hi=6, subtract_coarse_mean=True, margin=0.05,
        q_lo=0.5, q_hi=99.5, seed=0, bnt=Mf)

    if a.smoke:
        files = sorted(glob.glob(FID_OBS + "/cosmo_fiducial_perm*.npz"))[:1]
        zf = np.load(files[0]); autos = zf["patches"][:, :, :, :4].astype(np.float32)
        xu = fxl.build_and_l1(autos, "none", sig_t, stats, L1N, ranges,
                              clamp_overflow=True, bnt=Mf)
        exp = M.shape[0] * NS * L1N
        print(f"  [smoke] build_and_l1 -> x{xu.shape} (expect (*, {exp})), "
              f"NaN={np.isnan(xu).any()}, finite_frac={np.isfinite(xu).mean():.4f}", flush=True)
        assert xu.shape[1] == exp and not np.isnan(xu).any(), "smoke shape/NaN FAIL"
        print("SMOKE OK", flush=True)
        return

    ds_tr = fxl.compute_l1_dataset_flat_local(
        TFDS, DDIR, "train", "none", sig_t, stats, L1N, ranges,
        perm_lo=5, perm_hi=6, flip=True, seed=1001, batch_size=512,
        subtract_coarse_mean=True, clamp_overflow=True, bnt=Mf)
    ds_va = fxl.compute_l1_dataset_flat_local(
        TFDS, DDIR, "test", "none", sig_t, stats, L1N, ranges,
        perm_lo=0, perm_hi=1, flip=False, seed=2001, batch_size=512,
        subtract_coarse_mean=True, clamp_overflow=True, bnt=Mf)

    # alignment vs flat_none (same loader params ⇒ identical theta)
    fn = BASE + "/l1_matrix/l1_none_cache/flat_local_none"
    for nm, mine, base_npz in (("train", ds_tr, fn + "/l1_train.npz"),
                               ("val", ds_va, fn + "/l1_val.npz")):
        tb = np.load(base_npz)["theta"].astype(np.float64)
        tm = np.asarray(mine["theta"], np.float64)
        assert tm.shape == tb.shape and np.array_equal(tm, tb), f"{nm} theta NOT bit-equal vs flat_none"
        print(f"  [align] {nm} theta bit-equal vs flat_none {tm.shape} OK", flush=True)

    os.makedirs(a.out_cache, exist_ok=True)
    np.savez(a.out_cache + "/l1_train.npz", theta=ds_tr["theta"], x=ds_tr["x"].astype(np.float32))
    np.savez(a.out_cache + "/l1_val.npz", theta=ds_va["theta"], x=ds_va["x"].astype(np.float32))
    np.savez(a.out_cache + "/l1_cache_meta.npz", mix=a.mix, M=M, sigma_u=sigma_u,
             channel_names=np.array(uname), ranges=ranges,
             note="2D-1D Haar wavelet ℓ1 arm; PLAN_2D1D_PHASE_1_2.md Phase 1")
    print(f"  saved cache train x{ds_tr['x'].shape}, val x{ds_va['x'].shape}", flush=True)

    # ---- fiducial pass (mirror build_flatsky_postcut_arm.py:184-213) ----
    files = sorted(glob.glob(FID_OBS + "/cosmo_fiducial_perm*.npz"),
                   key=lambda f: int(f.split("perm")[-1].split(".")[0]))
    print(f"  fiducial pass over {len(files)} perm files ...", flush=True)
    X, perms, patches = [], [], []
    t0 = time.time()
    for i, f in enumerate(files):
        zf = np.load(f)
        autos = zf["patches"][:, :, :, :4].astype(np.float32)
        xu = fxl.build_and_l1(autos, "none", sig_t, stats, L1N, ranges,
                              clamp_overflow=True, bnt=Mf)
        X.append(xu)
        p = int(zf["perm"]) if "perm" in zf.files else int(f.split("perm")[-1].split(".")[0])
        perms.append(np.full(xu.shape[0], p, np.int32))
        patches.append(np.arange(xu.shape[0], dtype=np.int32))
        if (i + 1) % 40 == 0:
            print(f"    {i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    X = np.concatenate(X).astype(np.float32)
    perms = np.concatenate(perms); patches = np.concatenate(patches)

    fz = np.load(a.base_fid)
    assert np.array_equal(fz["perm"], perms), "fiducial perm arrays differ from base_fid"
    assert np.array_equal(fz["patch"], patches), "fiducial patch arrays differ from base_fid"
    out = {"S": X, "perm": perms, "patch": patches}
    for k in ("truth", "theta"):
        if k in fz.files:
            out[k] = fz[k]
    os.makedirs(os.path.dirname(a.out_fid), exist_ok=True)
    np.savez(a.out_fid, **out)
    print(f"  saved fiducial S{X.shape} -> {a.out_fid}", flush=True)
    print("BUILD OK", flush=True)


if __name__ == "__main__":
    main()
