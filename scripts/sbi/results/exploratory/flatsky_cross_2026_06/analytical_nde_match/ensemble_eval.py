#!/usr/bin/env python3
"""Compressor-ENSEMBLE per-obs FoM3 over the 9000-fiducial population (the properly-calibrated
joint ℓ1 number). Pools the posteriors of the 3 compressor-seed arms PER physical obs (each
compressor sees the same obs through its own compression), exactly the ensemble that calibrated
the gate (SBC 0.31->0.298 noBNT). Reuses train_sbilens_realnvp + the FoM3 machinery from the
production pipeline. Writes <out>/per_patch_metrics.npz + median_summary.json.

Usage: ensemble_eval.py --mode {nobnt,bnt} --out <dir>
"""
import argparse
import json
import sys
import time
from types import SimpleNamespace
from pathlib import Path
import numpy as np

import os as _os
# Titan paths (dead machine); overridable so this runs on Jean-Zay.
REPO = _os.environ.get("CNN_SBI_REPO", "/lustre/fswork/projects/rech/prk/ulx34io/recovery/rescued_scripts")
SBI = _os.environ.get("CNN_SBI_SBI", REPO)
A = _os.environ.get("CNN_SBI_ARMS", f"{SBI}/results/exploratory/flatsky_cross_2026_06/analytical_nde_match")
ARMS = {
    "nobnt": [f"{A}/jointl1_nobnt", f"{A}/jointl1_nobnt_s42", f"{A}/jointl1_nobnt_s43"],
    "bnt":   [f"{A}/jointl1_bnt",   f"{A}/jointl1_bnt_s42",   f"{A}/jointl1_bnt_s43"],
    # BNT auto/+product 3-compressor ensembles (calibration fix 2026-06-26)
    "product_bnt": [f"{A}/l1product_bnt_vmim_s41", f"{A}/ens_bnt_product_s42", f"{A}/ens_bnt_product_s43"],
    "auto_bnt":    [f"{A}/bnt_auto",               f"{A}/ens_bnt_auto_s42",    f"{A}/ens_bnt_auto_s43"],
    # no-BNT counterparts (uniform-analytical-family check 2026-06-27; CNN stays single)
    "product_nobnt": [f"{A}/l1product_vmim_s41", f"{A}/l1product_vmim_s42", f"{A}/l1product_vmim_s43"],
    "auto_nobnt":    [f"{A}/l1none_vmim_s41",   f"{A}/ens_nobnt_auto_s42", f"{A}/ens_nobnt_auto_s43"],
}
FLOW = SimpleNamespace(flow_total_steps=50000, flow_batch_size=128, flow_lr_init=1e-3,
                       flow_lr_end=1e-5, flow_save_every=2000, flow_patience=20,
                       flow_grad_clip=1.0, flow_weight_decay=1e-4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(ARMS), default="nobnt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cuda-visible-devices", default="2")
    ap.add_argument("--m-samples", type=int, default=2000)
    ap.add_argument("--n-obs", type=int, default=9000)
    ap.add_argument("--max-perm", type=int, default=50)
    ap.add_argument("--seeds", default="41,42,43")
    ap.add_argument("--save-samples", type=int, default=2000,
                    help="thinned pooled samples to persist per obs (0 = off). "
                         "2000 x 9000 obs x 3 params float32 = ~216 MB.")
    a = ap.parse_args()
    sys.path.insert(0, REPO); sys.path.insert(0, SBI)
    from train_jaxili_from_compressed import (setup_env, compute_fom3, marginal_stats,
                                               fom2d, posterior_moments)
    setup_env(a.cuda_visible_devices)
    import jax, jax.numpy as jnp
    from npe_l1norm_cross_jaxili_nbody_tomo import preprocess_summaries, filter_zero_variance_bins
    from train_nde_from_compressed import train_sbilens_realnvp

    seeds = [int(s) for s in a.seeds.split(",")]
    arms = []
    sel_ref = perm_ref = patch_ref = truth = None
    for adir in ARMS[a.mode]:
        cdir = Path(adir) / "cache"
        tr = np.load(cdir / "l1_train.npz"); va = np.load(cdir / "l1_val.npz")
        theta_tr = tr["theta"].astype(np.float32)
        x_tr_raw = tr["x"].astype(np.float64); x_va_raw = va["x"].astype(np.float64)
        tr_p, _, _, mean, std = preprocess_summaries(
            x_tr_raw, x_va_raw[:1], x_va_raw[:1], summary_transform="none", clip_value=None)
        mask, _ = filter_zero_variance_bins(tr_p, min_variance=1e-12, verbose=False)
        x_tr = tr_p[:, mask].astype(np.float32); dim = x_tr.shape[1]
        fz = np.load(str(Path(adir) / "fiducial_summaries.npz"))
        S = fz["S"].astype(np.float64); perm = fz["perm"]
        sel = np.where(perm < a.max_perm)[0][:a.n_obs]
        _, _, fid_p, _, _ = preprocess_summaries(
            x_tr_raw, x_va_raw[:1], S[sel], summary_transform="none", clip_value=None, mean=mean, std=std)
        x_obs = fid_p[:, mask].astype(np.float32)
        if sel_ref is None:
            sel_ref, perm_ref, patch_ref = sel, perm[sel], fz["patch"][sel]
            truth = next((fz[k] for k in ("truth", "theta") if k in fz.files), None)
        else:
            assert np.array_equal(sel, sel_ref), f"{adir}: sel misaligned vs first arm"
        out = Path(adir) / "_tmp_ens_flow"; out.mkdir(parents=True, exist_ok=True)
        print(f"[{Path(adir).name}] train dim {dim} ...", flush=True)
        samplers = train_sbilens_realnvp(4, 128, theta_tr, x_tr, seeds, a.m_samples, dim, out, FLOW)
        arms.append((samplers, jnp.asarray(x_obs)))

    N = arms[0][1].shape[0]
    print(f"[ensemble] pooling {len(arms)} compressors x {len(seeds)} seeds over {N} obs", flush=True)
    sig = np.full((N, 3), np.nan); pair = np.full((N, 3), np.nan); fom3 = np.full(N, np.nan)
    # --- THE FIX -------------------------------------------------------------------
    # The original pipeline stored only fom3 + marginal sigma. The ensemble jackknife
    # needs, per observation, the posterior MEAN vector and the FULL 3x3 covariance
    # (a 50/50 pooled pair has cov = 0.5*(C1+C2) + 0.25*(m1-m2)(m1-m2)^T, so the means
    # are essential, and the off-diagonal rho~0.93 is most of the information). That
    # omission is exactly why the original bars were unrecoverable after the disk
    # failure. Persist mean + cov always -- it costs under 1 MB for 9000 obs -- and a
    # thinned copy of the raw pooled samples so anything can be recomputed from scratch.
    post_mean = np.full((N, 3), np.nan)   # NB: 'mean' above is the preprocessing mean
    cov = np.full((N, 3, 3), np.nan)
    n_pooled = np.zeros(N, dtype=np.int64)
    keep = int(a.save_samples)
    samples_out = np.full((N, keep, 3), np.nan, dtype=np.float32) if keep else None
    # PER-MEMBER moments as well. The leave-one-out jackknife forms 1/2-1/2 mixtures of
    # PAIRS of members, and a mixture covariance is
    #     C_mix = 0.5*(C_i + C_j) + 0.25*(m_i - m_j)(m_i - m_j)^T
    # which cannot be reconstructed from the POOLED moments alone. Saving only the
    # pooled ensemble would leave the ensemble rows blocked all over again.
    n_arms = len(arms)
    arm_mean = np.full((n_arms, N, 3), np.nan)
    arm_cov = np.full((n_arms, N, 3, 3), np.nan)
    # -------------------------------------------------------------------------------
    t0 = time.time()
    for i in range(N):
        pooled, per_arm = [], []
        for samplers, x_dev in arms:
            arm_s = []
            for seed, fn in samplers:
                k = jax.random.PRNGKey(seed * 100003 + int(sel_ref[i]))
                arm_s.append(np.asarray(fn(x_dev[i], k)))
            aps = np.concatenate(arm_s, 0)
            aps = aps[np.all(np.isfinite(aps), 1)]
            per_arm.append(aps)
            pooled.append(aps)
        ps = np.concatenate(pooled, 0)
        if ps.shape[0] < 100:
            continue
        ms = marginal_stats(ps); f2 = fom2d(ps)
        sig[i] = [ms["sigma"][q] for q in ("Omega_m", "sigma_8", "w_0")]
        pair[i] = [f2["fom2d_Omega_m_sigma_8"], f2["fom2d_Omega_m_w_0"], f2["fom2d_sigma_8_w_0"]]
        fom3[i] = compute_fom3(ps)["fom3"]
        post_mean[i], cov[i] = posterior_moments(ps)
        n_pooled[i] = ps.shape[0]
        for ai, aps in enumerate(per_arm):
            if aps.shape[0] >= 100:
                arm_mean[ai, i], arm_cov[ai, i] = posterior_moments(aps)
        if keep:
            idx = (np.linspace(0, ps.shape[0] - 1, keep).astype(int) if ps.shape[0] >= keep
                   else np.arange(ps.shape[0]))
            samples_out[i, :len(idx)] = ps[idx, :3].astype(np.float32)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{N} ({time.time()-t0:.0f}s)", flush=True)
    outd = Path(a.out); outd.mkdir(parents=True, exist_ok=True)
    np.savez(outd / "per_patch_metrics.npz", sigma=sig, fom2d=pair, fom3=fom3,
             perm=perm_ref, patch=patch_ref, sel=sel_ref,
             truth=(truth if truth is not None else np.array([])),
             # added by the recovery: what makes the jackknife possible
             mean=post_mean, cov=cov, n_pooled=n_pooled,
             arm_mean=arm_mean, arm_cov=arm_cov,
             arm_dirs=np.array([str(d) for d in ARMS[a.mode]]),
             param_order=np.array(["Omega_m", "sigma_8", "w_0"]))
    if samples_out is not None:
        np.save(outd / "per_obs_samples.npy", samples_out)
        print(f"  wrote per_obs_samples.npy {samples_out.shape} float32 "
              f"({samples_out.nbytes/1e6:.0f} MB)", flush=True)
    g0 = np.isfinite(cov[:, 0, 0])
    if g0.any():
        rho = cov[g0, 0, 1] / np.sqrt(cov[g0, 0, 0] * cov[g0, 1, 1])
        print(f"  persisted mean+cov for {int(g0.sum())}/{N} obs; "
              f"median rho(Om,s8) = {np.median(rho):.4f}", flush=True)
    g = np.isfinite(fom3)
    med = dict(arm=f"{a.mode}_ensemble", n=int(g.sum()),
               sigma_Om=float(np.median(sig[g, 0])), sigma_s8=float(np.median(sig[g, 1])),
               sigma_w0=float(np.median(sig[g, 2])), fom3=float(np.median(fom3[g])))
    json.dump(med, open(outd / "median_summary.json", "w"), indent=2)
    print(f"\n[ENSEMBLE {a.mode}] N={med['n']} median FoM3={med['fom3']:.0f} "
          f"sigma={med['sigma_Om']:.3f}/{med['sigma_s8']:.3f}/{med['sigma_w0']:.3f}", flush=True)


if __name__ == "__main__":
    main()
