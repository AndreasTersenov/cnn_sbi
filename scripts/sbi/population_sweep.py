#!/usr/bin/env python3
"""Population sweep over 9000 fiducial obs for VMIM-compressed arms -> FoM3 + moments.

Replacement for population_sweep_flatsky.py (destroyed: 8617 bytes of zeros). Behaviour and
knobs recovered from the surviving drivers run_flatsky_population_sweep.py /
run_flatsky_cnn_population_sweep.py:

    per arm: train 3 NDE seeds in-process on the COMPRESSED cache, sample posteriors at
    9000 fiducial obs (perm<50 x 180 patches), pool 3 seeds/obs, per-patch sigma/2D/FoM3,
    report the MEDIAN.  preproc none / clip 0 / min-var 1e-12 (compression already applied
    log1p-zscore / clip 5 / min-var 1e-5).

ARCHITECTURE (confirmed by the author): every arm's readout is
    VMIM(compressor seed) -> RealNVP 4x128, pooled over 3 flow seeds.
A single-quoted row = one compressor, NDE-pooled; its error bar is the SPREAD across the
three compressor seeds. An ensemble row = the SAME three compressors POOLED (9 flows).
Pass one --arm-dir for a single, three for an ensemble.

Do not feed a raw L1 vector to the flow: sbi_lens RealNVP craters on it
(ESTIMATOR_OPTIMIZATION_RECORD: "the same RealNVP craters on the 2000-D L1 vector"). The
compression stage is not optional.

PERSISTS what the original omitted: per observation the posterior MEAN and FULL 3x3
COVARIANCE, and the same per pooled member (arm_mean/arm_cov) -- what the ensemble
jackknife in final_bars.py needs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SRC = os.environ.get("CNN_SBI_SRC",
                     "/lustre/fswork/projects/rech/prk/ulx34io/recovery/rescued_scripts")
sys.path.insert(0, SRC)

FLOW = SimpleNamespace(flow_total_steps=50000, flow_batch_size=128, flow_lr_init=1e-3,
                       flow_lr_end=1e-5, flow_save_every=2000, flow_patience=20,
                       flow_grad_clip=1.0, flow_weight_decay=1e-4)


def arm_paths(d: Path):
    """Locate an arm's compressed train/val caches and its fiducial summaries.

    Accepts <arm>/cache/<p>_train.npz and <arm>/<p>_train.npz, with prefix p in
    {l1, cnn}: the VMIM-from-cache arms write l1_*, the CNN driver writes cnn_*.
    Returns (train_npz, val_npz, fiducial_npz)."""
    for parent in (d / "cache", d):
        for prefix in ("l1", "cnn"):
            tr, va = parent / f"{prefix}_train.npz", parent / f"{prefix}_val.npz"
            if tr.exists() and va.exists():
                fid = d / "fiducial_summaries.npz"
                if not fid.exists():
                    fid = parent / "fiducial_summaries.npz"
                if not fid.exists():
                    raise SystemExit(f"missing fiducial_summaries.npz for {d}")
                return tr, va, fid
    raise SystemExit(f"no {{l1,cnn}}_train.npz + _val.npz under {d} or {d}/cache")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", required=True, action="append",
                    help="VMIM-compressed arm dir; repeat to POOL members (ensemble row)")
    ap.add_argument("--arm-label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="41,42,43", help="NDE (flow) seeds pooled per arm")
    ap.add_argument("--n-obs", type=int, default=9000)
    ap.add_argument("--max-perm", type=int, default=50)
    ap.add_argument("--m-samples", type=int, default=2000)
    ap.add_argument("--preproc-transform", default="none")
    ap.add_argument("--clip-value", type=float, default=0.0)
    ap.add_argument("--min-feature-variance", type=float, default=1e-12)
    ap.add_argument("--save-samples", type=int, default=2000)
    ap.add_argument("--cuda-visible-devices", default="0")
    a = ap.parse_args()

    from train_jaxili_from_compressed import (setup_env, compute_fom3, marginal_stats,
                                              fom2d, posterior_moments)
    setup_env(a.cuda_visible_devices)
    import jax, jax.numpy as jnp
    from npe_l1norm_cross_jaxili_nbody_tomo import (preprocess_summaries,
                                                    filter_zero_variance_bins)
    from train_nde_from_compressed import train_sbilens_realnvp

    seeds = [int(s) for s in a.seeds.split(",")]
    outd = Path(a.out); outd.mkdir(parents=True, exist_ok=True)
    clip = None if a.clip_value in (0, 0.0) else a.clip_value
    t0 = time.time()

    members, sel_ref, perm_ref, patch_ref, truth = [], None, None, None, None
    for adir in a.arm_dir:
        d = Path(adir); trp, vap, fidp = arm_paths(d)
        tr = np.load(trp); va = np.load(vap)
        theta_tr = tr["theta"].astype(np.float32)
        x_tr_raw = tr["x"].astype(np.float64); x_va_raw = va["x"].astype(np.float64)
        tr_p, _, _, mean, std = preprocess_summaries(
            x_tr_raw, x_va_raw[:1], x_va_raw[:1],
            summary_transform=a.preproc_transform, clip_value=clip)
        mask, _ = filter_zero_variance_bins(tr_p, min_variance=a.min_feature_variance,
                                            verbose=False)
        x_tr = tr_p[:, mask].astype(np.float32); dim = x_tr.shape[1]

        fz = np.load(fidp)
        S = fz["S"].astype(np.float64); perm = fz["perm"]
        sel = np.where(perm < a.max_perm)[0][:a.n_obs]
        _, _, fid_p, _, _ = preprocess_summaries(
            x_tr_raw, x_va_raw[:1], S[sel], summary_transform=a.preproc_transform,
            clip_value=clip, mean=mean, std=std)
        x_obs = fid_p[:, mask].astype(np.float32)
        if sel_ref is None:
            sel_ref, perm_ref, patch_ref = sel, perm[sel], fz["patch"][sel]
            truth = next((fz[k] for k in ("truth", "theta") if k in fz.files), None)
        elif not np.array_equal(sel, sel_ref):
            raise SystemExit(f"{adir}: observation selection differs from the first member")

        flow_out = outd / f"_flow_{d.name}"; flow_out.mkdir(parents=True, exist_ok=True)
        print(f"[{d.name}] train {x_tr.shape} dim={dim} -> {len(seeds)} flow seeds", flush=True)
        samplers = train_sbilens_realnvp(4, 128, theta_tr, x_tr, seeds, a.m_samples,
                                         dim, flow_out, FLOW)
        members.append((d.name, samplers, jnp.asarray(x_obs)))

    N, K = len(sel_ref), len(members)
    print(f"[{a.arm_label}] {K} compressor member(s) x {len(seeds)} flow seeds "
          f"over {N} obs ({time.time()-t0:.0f}s)", flush=True)

    sig = np.full((N, 3), np.nan); pair = np.full((N, 3), np.nan); fom3 = np.full(N, np.nan)
    post_mean = np.full((N, 3), np.nan); cov = np.full((N, 3, 3), np.nan)
    n_pooled = np.zeros(N, dtype=np.int64)
    arm_mean = np.full((K, N, 3), np.nan); arm_cov = np.full((K, N, 3, 3), np.nan)
    keep = int(a.save_samples)
    samples_out = np.full((N, keep, 3), np.nan, np.float32) if keep else None

    for i in range(N):
        per_member = []
        for nm, samplers, x_dev in members:
            s_list = []
            for seed, fn in samplers:
                k = jax.random.PRNGKey(seed * 100003 + int(sel_ref[i]))
                s_list.append(np.asarray(fn(x_dev[i], k)))
            s = np.concatenate(s_list, 0)
            per_member.append(s[np.all(np.isfinite(s), 1)])
        ps = np.concatenate(per_member, 0)
        if ps.shape[0] < 100:
            continue
        ms = marginal_stats(ps); f2 = fom2d(ps)
        sig[i] = [ms["sigma"][q] for q in ("Omega_m", "sigma_8", "w_0")]
        pair[i] = [f2["fom2d_Omega_m_sigma_8"], f2["fom2d_Omega_m_w_0"], f2["fom2d_sigma_8_w_0"]]
        fom3[i] = compute_fom3(ps)["fom3"]
        post_mean[i], cov[i] = posterior_moments(ps)
        n_pooled[i] = ps.shape[0]
        for mi, s in enumerate(per_member):
            if s.shape[0] >= 100:
                arm_mean[mi, i], arm_cov[mi, i] = posterior_moments(s)
        if keep:
            idx = (np.linspace(0, ps.shape[0] - 1, keep).astype(int)
                   if ps.shape[0] >= keep else np.arange(ps.shape[0]))
            samples_out[i, :len(idx)] = ps[idx, :3].astype(np.float32)
        if (i + 1) % 1000 == 0:
            g = np.isfinite(fom3[:i + 1])
            print(f"  {i+1}/{N} ({time.time()-t0:.0f}s) running median FoM3="
                  f"{np.median(fom3[:i+1][g]):.1f}", flush=True)

    np.savez(outd / "per_patch_metrics.npz",
             sigma=sig, fom2d=pair, fom3=fom3, perm=perm_ref, patch=patch_ref, sel=sel_ref,
             truth=(truth if truth is not None else np.array([])),
             mean=post_mean, cov=cov, n_pooled=n_pooled,
             arm_mean=arm_mean, arm_cov=arm_cov,
             arm_dirs=np.array([m[0] for m in members]),
             param_order=np.array(["Omega_m", "sigma_8", "w_0"]),
             arm_label=a.arm_label, nde_seeds=np.array(seeds))
    if samples_out is not None:
        np.save(outd / "per_obs_samples.npy", samples_out)

    g = np.isfinite(fom3)
    med = dict(arm=a.arm_label, n=int(g.sum()), n_members=K, nde_seeds=seeds,
               fom3=float(np.median(fom3[g])),
               sigma_Om=float(np.median(sig[g, 0])), sigma_s8=float(np.median(sig[g, 1])),
               sigma_w0=float(np.median(sig[g, 2])),
               m_samples=a.m_samples, elapsed_s=round(time.time() - t0, 1))
    json.dump(med, open(outd / "median_summary.json", "w"), indent=2)
    rho = cov[g, 0, 1] / np.sqrt(cov[g, 0, 0] * cov[g, 1, 1])
    print(f"\n[{a.arm_label}] N={med['n']}  MEDIAN FoM3 = {med['fom3']:.1f}", flush=True)
    print(f"  sigma = {med['sigma_Om']:.4f}/{med['sigma_s8']:.4f}/{med['sigma_w0']:.4f}", flush=True)
    print(f"  median rho(Om,s8) = {np.median(rho):.4f}", flush=True)


if __name__ == "__main__":
    main()
