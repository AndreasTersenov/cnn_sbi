#!/usr/bin/env python
"""Build the post-cut recombination arms (lane B of PLAN_OVERNIGHT_MENU_2.md).

Cut-then-mix: linear recombinations of the KEPT, SCALE-CUT BNT channels, appended to the
masked BNT-l1 base (B0). By wavelet linearity, W_s(mix of cut maps) = sum over the BNT
channels KEPT at scale s of the mix coefficients times W_s(channel) — so each (new
channel, scale) feature block equals the scale-s block of a plain mix channel whose mix
row is the kept-masked row. Implementation: collect the UNIQUE nonzero masked rows
(expressed over the ORIGINAL autos via c^T B — any linear combo of BNT channels is a
linear combo of autos), run ONE multi-channel build with those rows (exact sigma rows
sqrt(U^2 @ sigma_auto^2) from the verified no-BNT table), then assemble the (channel,
scale) blocks by lookup.

Variants:
  cutsum6  — pairwise sums of cut BNT channels: rows_bnt = 0.5(e_i + e_j), i<j  (B1)
  cutdeep2 — reconstructed-deep from kept content: rows_bnt = [(1/4)1^T B^-1; e_4^T B^-1]
             (uncut limit IS deep2 — asserted structurally at full-kept scales)      (B2)

Alignment asserted (theta / fiducial perm+patch bit-equality vs the base cache), same
loader parameters as every concat builder (train perms 5-6/flip/seed 1001/batch 512;
val test/0-1/noflip/2001).
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


def parse_keep(spec: str, n_scales: int) -> np.ndarray:
    """(n_channels, n_scales) 0/1 kept indicator from 'sc,sc;...' (index 0 = finest)."""
    rows = []
    for part in spec.split(";"):
        kept = np.zeros(n_scales)
        for s in part.split(","):
            kept[int(s)] = 1.0
        rows.append(kept)
    return np.stack(rows)  # (C_bnt, NS)


def variant_rows_bnt(variant: str) -> tuple[np.ndarray, list[str]]:
    """Mix rows over the BNT channels (float64) + row names."""
    if variant == "cutsum6":
        rows, names = [], []
        for i, j in fx.cross_pairs(4):
            r = np.zeros(4); r[i] = r[j] = 0.5
            rows.append(r); names.append(f"sum{i+1}{j+1}")
        return np.stack(rows), names
    if variant == "cutdeep2":
        Binv = np.linalg.inv(fx.bnt_matrix_np().astype(np.float64))
        return np.stack([0.25 * np.ones(4) @ Binv, Binv[3]]), ["avg_rec", "bin4_rec"]
    raise ValueError(variant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("cutsum6", "cutdeep2"), required=True)
    ap.add_argument("--keep", required=True,
                    help="kept scales per BNT channel, e.g. '3,4;2,3,4;1,2,3,4;0,1,2,3,4'")
    ap.add_argument("--base-cache", required=True, help="the masked B0 cache dir")
    ap.add_argument("--base-fid", required=True)
    ap.add_argument("--out-cache", required=True)
    ap.add_argument("--out-fid", required=True)
    a = ap.parse_args()

    import torch
    from wl_stats_torch import WLStatistics

    B = fx.bnt_matrix_np().astype(np.float64)                  # (4,4) BNT over autos
    kept = parse_keep(a.keep, NS)                              # (4, NS) over BNT channels
    rows_bnt, row_names = variant_rows_bnt(a.variant)          # (R,4), R names
    R = rows_bnt.shape[0]
    print(f"############ post-cut arm build: {a.variant} ############")
    print(f"  keep (BNT channels x scales, 0=finest):\n{kept.astype(int)}")

    # per-(row, scale) masked rows over autos; collect uniques
    unique_rows: list[np.ndarray] = []
    key_of: dict[tuple, int] = {}
    slot = {}                                                  # (r, s) -> uidx or None
    for r in range(R):
        for s in range(NS):
            c = rows_bnt[r] * kept[:, s]                       # masked over BNT channels
            if not np.any(np.abs(c) > 0):
                slot[(r, s)] = None
                continue
            row_auto = c @ B                                   # over original autos
            key = tuple(np.round(row_auto, 12))
            if key not in key_of:
                key_of[key] = len(unique_rows)
                unique_rows.append(row_auto)
            slot[(r, s)] = key_of[key]
    U = np.stack(unique_rows)                                  # (u, 4) over autos
    print(f"  {R} rows x {NS} scales -> {U.shape[0]} unique nonzero masked rows")

    # uncut-limit structural check: at fully-kept scales cutdeep2 rows == deep2 rows
    full_s = [s for s in range(NS) if kept[:, s].all()]
    if a.variant == "cutdeep2" and full_s:
        d2 = fx.deep2_matrix_np().astype(np.float64)
        for r in range(R):
            got = rows_bnt[r] * kept[:, full_s[0]] @ B
            assert np.allclose(got, d2[r], atol=1e-6), (r, got, d2[r])
        print(f"  [check] full-kept scales reproduce deep2 rows exactly OK")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stats = WLStatistics(n_scales=NS, device=dev, pixel_arcmin=RESO, dtype=torch.float64)

    # exact sigma rows from the verified no-BNT table (inter-bin noise independence)
    z = np.load(SIGMA_NOBNT)
    assert (str(z["mode"]) if "mode" in z.files else "none") == "none"
    sig_auto = np.asarray(z["sigma"], np.float64)[:4]          # (4, NS)
    assert int(z["n_scales"]) == NS
    sigma_u = np.sqrt((U ** 2) @ (sig_auto ** 2))              # (u, NS)
    sig_t = torch.from_numpy(sigma_u).to(dev, dtype=torch.float64)
    uname = [f"u{k}" for k in range(U.shape[0])]
    for k, row in enumerate(sigma_u):
        print(f"    sigma u{k}: " + " ".join(f"{s:.4e}" for s in row))

    Uf = U.astype(np.float64)
    ranges_u = fxl.calibrate_snr_range_flat_local(
        tfds_name=TFDS, data_dir=DDIR, op="none", frozen_sigma=sig_t, stats=stats,
        nbins=4, channel_names=uname, n_calibration_examples=20 * 180,
        perm_lo=5, perm_hi=6, subtract_coarse_mean=True, margin=0.05,
        q_lo=0.5, q_hi=99.5, seed=0, bnt=Uf)

    ds_tr = fxl.compute_l1_dataset_flat_local(
        TFDS, DDIR, "train", "none", sig_t, stats, L1N, ranges_u,
        perm_lo=5, perm_hi=6, flip=True, seed=1001, batch_size=512,
        subtract_coarse_mean=True, clamp_overflow=True, bnt=Uf)
    ds_va = fxl.compute_l1_dataset_flat_local(
        TFDS, DDIR, "test", "none", sig_t, stats, L1N, ranges_u,
        perm_lo=0, perm_hi=1, flip=False, seed=2001, batch_size=512,
        subtract_coarse_mean=True, clamp_overflow=True, bnt=Uf)

    # ---- assemble the (row, scale) blocks from the unique-row build ----
    layout = [(r, s) for r in range(R) for s in range(NS) if slot[(r, s)] is not None]

    def assemble(x_u: np.ndarray) -> np.ndarray:
        blocks = []
        for r, s in layout:
            u = slot[(r, s)]
            base = u * NS * L1N + s * L1N
            blocks.append(x_u[:, base:base + L1N])
        return np.concatenate(blocks, axis=1)

    base_tr = np.load(a.base_cache + "/l1_train.npz")
    base_va = np.load(a.base_cache + "/l1_val.npz")
    for nm, mine, theirs in (("train", ds_tr, base_tr), ("val", ds_va, base_va)):
        th_m = np.asarray(mine["theta"], np.float64)
        th_b = np.asarray(theirs["theta"], np.float64)
        assert th_m.shape == th_b.shape and np.array_equal(th_m, th_b), \
            f"{nm} theta NOT bit-equal — row alignment broken"
        print(f"  [align] {nm} theta bit-equal over {th_m.shape} OK")
    new_tr, new_va = assemble(ds_tr["x"]), assemble(ds_va["x"])
    x_tr = np.concatenate([base_tr["x"], new_tr], axis=1).astype(np.float32)
    x_va = np.concatenate([base_va["x"], new_va], axis=1).astype(np.float32)
    print(f"  concatenated train {x_tr.shape}, val {x_va.shape} "
          f"(= [base {base_tr['x'].shape[1]} | {a.variant} {new_tr.shape[1]}])")

    os.makedirs(a.out_cache, exist_ok=True)
    np.savez(a.out_cache + "/l1_train.npz", theta=base_tr["theta"], x=x_tr)
    np.savez(a.out_cache + "/l1_val.npz", theta=base_va["theta"], x=x_va)
    np.savez(a.out_cache + "/l1_cache_meta.npz",
             variant=a.variant, keep=np.array(a.keep), base=np.array(a.base_cache),
             rows_bnt=rows_bnt, row_names=np.array(row_names), U=U, sigma_u=sigma_u,
             ranges_u=ranges_u, layout=np.array(layout, np.int64),
             note="post-cut recombination arm; PLAN_OVERNIGHT_MENU_2.md lane B")

    # ---- fiducial pass ----
    files = sorted(glob.glob(FID_OBS + "/cosmo_fiducial_perm*.npz"),
                   key=lambda f: int(f.split("perm")[-1].split(".")[0]))
    print(f"  fiducial pass over {len(files)} perm files ...")
    X, perms, patches = [], [], []
    t0 = time.time()
    for i, f in enumerate(files):
        zf = np.load(f)
        autos = zf["patches"][:, :, :, :4].astype(np.float32)
        xu = fxl.build_and_l1(autos, "none", sig_t, stats, L1N, ranges_u,
                              clamp_overflow=True, bnt=Uf)
        X.append(assemble(xu))
        p = int(zf["perm"]) if "perm" in zf.files else int(f.split("perm")[-1].split(".")[0])
        perms.append(np.full(xu.shape[0], p, np.int32))
        patches.append(np.arange(xu.shape[0], dtype=np.int32))
        if (i + 1) % 40 == 0:
            print(f"    {i+1}/{len(files)} ({time.time()-t0:.0f}s)")
    X = np.concatenate(X).astype(np.float32)
    perms = np.concatenate(perms); patches = np.concatenate(patches)

    fz = np.load(a.base_fid)
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
