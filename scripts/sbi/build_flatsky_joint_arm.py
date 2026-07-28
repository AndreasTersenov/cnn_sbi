#!/usr/bin/env python
"""Build a joint-statistic arm (overnight menu): cov / pair2d / full4d / jointl1, in either
basis, optionally APPENDED to an existing cache (A1 = cov appended to the BNT-L1 cache).

PLAN_OVERNIGHT_MENU.md. Loader parameters mirror every campaign arm (train perms 5-6 /
flip / seed 1001 / batch 512; val test 0-1 / noflip / 2001; same NaN-batch skip), so
append-mode row alignment can be HARD-ASSERTED against the parent cache. Frozen sigma table
of the arm's basis (nobnt / bnt, both GATE A1b-passed). Fiducial = the standard 36000-obs
pass, appended likewise when --append-fid is given.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flatsky_joint_stats as fjs

# RECOVERY PATCH (2026-07-28): the four constants below pointed at the dead Titan machine
# (DDIR at /home/tersenov, BASE/FID_OBS inside the damaged tree). Made env-overridable, with
# the originals as defaults so the file still behaves identically where those paths exist.
# Nothing else in this file is changed. See recovery/HANDOFF_STATE.md sections 3 and 7a.
BASE = os.environ.get("JOINT_BASE", HERE + "/results/exploratory/flatsky_cross_2026_06")
TFDS = os.environ.get(
    "JOINT_TFDS", "nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180")
DDIR = os.environ.get("JOINT_TFDS_DIR", "/home/tersenov/tensorflow_datasets")
FID_OBS = os.environ.get(
    "JOINT_FID_OBS",
    HERE + "/results/exploratory/cross_maps_campaign/"
           "full_sphere_cache_fiducial_10deg/nobnt/obs")
SIGMA = {"nobnt": os.environ.get("JOINT_SIGMA_NOBNT",
                                 BASE + "/flatsky_cross_noise_sigma.npz"),
         "bnt": os.environ.get("JOINT_SIGMA_BNT",
                               BASE + "/flatsky_cross_noise_sigma_bnt.npz")}
RESO, NS = 7.5, 5


def load_sigma(basis, dev):
    import torch
    z = np.load(SIGMA[basis])
    mode = str(z["mode"]) if "mode" in z.files else ("bnt" if bool(z.get("bnt", False)) else "none")
    want = "none" if basis == "nobnt" else "bnt"
    assert mode == want, f"sigma table {SIGMA[basis]} mode={mode}, want {want}"
    sig = np.asarray(z["sigma"], np.float64)[:4]                   # auto rows
    assert int(z["n_scales"]) == NS
    return torch.from_numpy(sig).to(dev, dtype=torch.float64)


def dataset_pass(split, perm_lo, perm_hi, flip, seed, stat, basis_mix, sigma, stats, k,
                 dq_gen=None, ranges=None, rotation=None):
    from tfds_cross_tfdata_loader import iter_cross_tfds_batches
    print(f"  [{stat}] features from cross TFDS [{split} perms {perm_lo}-{perm_hi} "
          f"flip={flip} basis={basis_mix}] ...", flush=True)
    xs, ths = [], []
    n, t0, nxt = 0, time.time(), 40000
    for autos_np, theta_np in iter_cross_tfds_batches(
            TFDS, DDIR, split, 512, flip=flip, channel_scale=None,
            channel_slice=slice(0, 4), perm_lo=perm_lo, perm_hi=perm_hi, seed=seed):
        if np.isnan(autos_np).any():
            print("    [!] skipped batch with NaN autos"); continue
        xs.append(fjs.compute_features(autos_np, stat, basis_mix, sigma, stats, k,
                                       dequant_gen=dq_gen, ranges=ranges, rotation=rotation))
        th = theta_np.copy(); th[:, 3] = th[:, 3] / 100.0
        ths.append(th)
        n += autos_np.shape[0]
        if n >= nxt:
            el = time.time() - t0
            print(f"    {n} patches ({el:.1f}s, {n/max(el, 1e-9):.0f}/s)", flush=True)
            nxt += 40000
    print(f"  [{stat}] done: {n} patches in {time.time()-t0:.1f}s", flush=True)
    return {"theta": np.concatenate(ths, 0), "x": np.concatenate(xs, 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stat", choices=("cov", "pair2d", "full4d", "jointl1"), required=True)
    ap.add_argument("--basis", choices=("nobnt", "bnt"), required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out-cache", required=True)
    ap.add_argument("--out-fid", required=True)
    ap.add_argument("--append-to", default=None,
                    help="parent cache dir; new features are CONCATENATED after its x "
                         "(theta bit-equality asserted)")
    ap.add_argument("--append-fid", default=None,
                    help="parent fiducial npz (S concatenated; perm/patch asserted)")
    ap.add_argument("--dequantize", action="store_true",
                    help="add seeded U(0,1) dequantization noise to full4d counts")
    ap.add_argument("--adaptive-ranges", action="store_true",
                    help="per-(channel,scale) percentile SNR grid instead of fixed [-5,5] "
                         "(the transported-binning variant; calibrated on ~3600 train maps)")
    ap.add_argument("--rotated-binning", action="store_true",
                    help="pair2d/jointl1 only: per-(pair,scale) 2-D PCA-whitened grid (rotates "
                         "the grid onto the cloud's eigen-axes = the shear-aware transport, P4c). "
                         "Supersedes --adaptive-ranges for the pairwise binning.")
    a = ap.parse_args()

    import torch
    from wl_stats_torch import WLStatistics
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stats = WLStatistics(n_scales=NS, device=dev, pixel_arcmin=RESO, dtype=torch.float64)
    sigma = load_sigma(a.basis, dev)
    basis_mix = False if a.basis == "nobnt" else "bnt"
    dq_gen = None
    if a.dequantize:
        dq_gen = torch.Generator(device=dev); dq_gen.manual_seed(0xDEC0DE)

    print(f"############ joint arm build: stat={a.stat} basis={a.basis} k={a.k} "
          f"dequantize={a.dequantize} adaptive={a.adaptive_ranges} ############", flush=True)
    ranges = None
    rotation = None

    def _autos_iter():
        from tfds_cross_tfdata_loader import iter_cross_tfds_batches
        for autos_np, _ in iter_cross_tfds_batches(
                TFDS, DDIR, "train", 512, flip=False, channel_scale=None,
                channel_slice=slice(0, 4), perm_lo=5, perm_hi=6, seed=0):
            if not np.isnan(autos_np).any():
                yield autos_np

    if a.rotated_binning:
        rotation = fjs.calibrate_joint_rotation(_autos_iter(), basis_mix, sigma, stats, a.k)
        print(f"  rotated (2-D PCA-whitened) pairwise binning: {len(rotation['pairs'])} pairs "
              f"x {rotation['mu'].shape[1]} scales calibrated (shear-aware transport)", flush=True)
    elif a.adaptive_ranges:
        ranges = fjs.calibrate_joint_ranges(_autos_iter(), basis_mix, sigma, stats, a.k)
        print("  adaptive per-(channel,scale) SNR ranges:", flush=True)
        for c in range(4):
            r = ranges[c].cpu().numpy()
            print("    ch%d: " % c + " ".join(f"[{x[0]:.1f},{x[1]:.1f}]" for x in r),
                  flush=True)
    ds_tr = dataset_pass("train", 5, 6, True, 1001, a.stat, basis_mix, sigma, stats, a.k,
                         dq_gen=dq_gen, ranges=ranges, rotation=rotation)
    ds_va = dataset_pass("test", 0, 1, False, 2001, a.stat, basis_mix, sigma, stats, a.k,
                         dq_gen=dq_gen, ranges=ranges, rotation=rotation)
    x_tr, th_tr = ds_tr["x"], ds_tr["theta"]
    x_va, th_va = ds_va["x"], ds_va["theta"]

    if a.append_to:
        p_tr = np.load(a.append_to + "/l1_train.npz")
        p_va = np.load(a.append_to + "/l1_val.npz")
        for nm, mine, theirs in (("train", th_tr, p_tr["theta"]), ("val", th_va, p_va["theta"])):
            assert np.array_equal(np.asarray(mine, np.float64), np.asarray(theirs, np.float64)), \
                f"{nm} theta NOT bit-equal vs parent — alignment broken"
            print(f"  [align] {nm} theta bit-equal OK", flush=True)
        x_tr = np.concatenate([p_tr["x"], x_tr], axis=1)
        x_va = np.concatenate([p_va["x"], x_va], axis=1)
        th_tr, th_va = p_tr["theta"], p_va["theta"]
    x_tr = x_tr.astype(np.float32); x_va = x_va.astype(np.float32)
    print(f"  train {x_tr.shape}, val {x_va.shape}", flush=True)
    os.makedirs(a.out_cache, exist_ok=True)
    np.savez(a.out_cache + "/l1_train.npz", theta=th_tr, x=x_tr)
    np.savez(a.out_cache + "/l1_val.npz", theta=th_va, x=x_va)
    np.savez(a.out_cache + "/l1_cache_meta.npz", stat=a.stat, basis=a.basis, k=a.k,
             dequantize=a.dequantize, adaptive_ranges=a.adaptive_ranges,
             rotated_binning=a.rotated_binning,
             snr_range=fjs.SNR_RANGE, append_to=str(a.append_to),
             note="overnight menu arm; PLAN_OVERNIGHT_MENU.md")

    files = sorted(glob.glob(FID_OBS + "/cosmo_fiducial_perm*.npz"),
                   key=lambda f: int(f.split("perm")[-1].split(".")[0]))
    print(f"  fiducial pass over {len(files)} perm files ...", flush=True)
    X, perms, patches = [], [], []
    truth = None
    t0 = time.time()
    for i, f in enumerate(files):
        zf = np.load(f)
        autos = zf["patches"][:, :, :, :4].astype(np.float32)
        X.append(fjs.compute_features(autos, a.stat, basis_mix, sigma, stats, a.k,
                                      dequant_gen=dq_gen, ranges=ranges, rotation=rotation))
        p = int(zf["perm"]) if "perm" in zf.files else int(f.split("perm")[-1].split(".")[0])
        perms.append(np.full(X[-1].shape[0], p, np.int32))
        patches.append(np.arange(X[-1].shape[0], dtype=np.int32))
        if truth is None:
            th = np.asarray(zf["theta"], np.float64).copy(); th[3] /= 100.0
            truth = th
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    X = np.concatenate(X)
    perms = np.concatenate(perms); patches = np.concatenate(patches)
    if a.append_fid:
        fz = np.load(a.append_fid)
        assert np.array_equal(fz["perm"], perms) and np.array_equal(fz["patch"], patches), \
            "fiducial perm/patch differ vs parent — alignment broken"
        print("  [align] fiducial perm/patch bit-equal OK", flush=True)
        X = np.concatenate([fz["S"], X], axis=1)
        truth = next(fz[kk] for kk in ("truth", "theta") if kk in fz.files)
    X = X.astype(np.float32)
    os.makedirs(os.path.dirname(a.out_fid), exist_ok=True)
    np.savez(a.out_fid, S=X, perm=perms, patch=patches, truth=truth,
             stat=a.stat, basis=a.basis, k=a.k)
    print(f"  saved fiducial {X.shape} -> {a.out_fid}", flush=True)
    print("BUILD OK", flush=True)


if __name__ == "__main__":
    main()
