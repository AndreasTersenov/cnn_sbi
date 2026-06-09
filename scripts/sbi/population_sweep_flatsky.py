#!/usr/bin/env python3
"""Population sweep for one flat-local L1 arm: per-patch sigma/2D/FoM3 over 9000 fiducial obs.

Mirrors tarp_stratified_val.py's retrain+preprocess (sidesteps the high-dim jaxili reload
truncation), then samples posteriors at N fiducial obs (sliced per-arm summaries), POOLING the
3 seeds per obs, and records per-patch marginal sigma, per-pair 2D FoM, and FoM3. The headline
is the MEDIAN over patches (robust to single-realization scatter). Fiducial obs = perm<50 x 180
patches = 9000 (matches the prior SUMMARY_PHASE_D 9000-obs/arm for an apples-to-apples comparison).
"""
import argparse, sys, time
from pathlib import Path
import numpy as np

REPO = "/mnt/home/tersenov/software/cnn_sbi"
SBI = f"{REPO}/scripts/sbi"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache-dir", required=True)
    p.add_argument("--cache-prefix", default="l1")
    p.add_argument("--arm-label", required=True)
    p.add_argument("--fiducial-summaries-npz", required=True)   # per-arm sliced (key 'S')
    p.add_argument("--output-dir", required=True)
    p.add_argument("--preproc-transform", default="log1p-zscore")
    p.add_argument("--clip-value", type=float, default=5.0)
    p.add_argument("--min-feature-variance", type=float, default=1e-5)
    p.add_argument("--seeds", default="41,42,43")
    p.add_argument("--n-obs", type=int, default=9000)
    p.add_argument("--max-perm", type=int, default=50)          # perm<50 x 180 = 9000
    p.add_argument("--m-samples", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--decay-steps", type=int, default=10000)
    p.add_argument("--cuda-visible-devices", default="1")
    return p.parse_args()


def main():
    a = parse_args()
    sys.path.insert(0, str(REPO)); sys.path.insert(0, SBI)
    from train_jaxili_from_compressed import setup_env, compute_fom3, fom2d, marginal_stats
    setup_env(a.cuda_visible_devices)
    import jax, jax.numpy as jnp
    from jaxili.inference import NPE
    from npe_l1norm_cross_jaxili_nbody_tomo import (
        train_with_nan_retry, preprocess_summaries, filter_zero_variance_bins)

    cdir = Path(a.train_cache_dir)
    tr = np.load(cdir / f"{a.cache_prefix}_train.npz"); va = np.load(cdir / f"{a.cache_prefix}_val.npz")
    theta_tr = tr["theta"].astype(np.float32)
    x_tr_raw = tr["x"].astype(np.float64); x_va_raw = va["x"].astype(np.float64)
    clip = a.clip_value if a.clip_value > 0 else None
    tr_p, _, _, mean, std = preprocess_summaries(
        x_tr_raw, x_va_raw[:1], x_va_raw[:1], summary_transform=a.preproc_transform, clip_value=clip)
    mask, _ = filter_zero_variance_bins(tr_p, min_variance=a.min_feature_variance, verbose=False)
    x_tr = tr_p[:, mask].astype(np.float32)
    print(f"[{a.arm_label}] train{x_tr_raw.shape} -> dim {x_tr.shape[1]}", flush=True)

    # fiducial obs (sliced per-arm summaries), preprocessed with the TRAIN mean/std + mask
    fz = np.load(a.fiducial_summaries_npz)
    S = fz["S"].astype(np.float64); perm = fz["perm"]
    sel = np.where(perm < a.max_perm)[0][:a.n_obs]
    _, _, fid_p, _, _ = preprocess_summaries(
        x_tr_raw, x_va_raw[:1], S[sel], summary_transform=a.preproc_transform,
        clip_value=clip, mean=mean, std=std)
    x_obs = fid_p[:, mask].astype(np.float32)
    truth = fz["truth"] if "truth" in fz.files else None
    N, M = x_obs.shape[0], a.m_samples
    print(f"[{a.arm_label}] {N} fiducial obs (perm<{a.max_perm})", flush=True)

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    posteriors = []
    params = jnp.asarray(theta_tr); data = jnp.asarray(x_tr)
    tmp = Path(a.output_dir) / "ckpts"
    for seed in seeds:
        t0 = time.time(); sk = jax.random.PRNGKey(seed + 1)
        inf = NPE().append_simulations(params, data, key=sk)
        inf, _m, _d = train_with_nan_retry(inf, str((tmp / f"s{seed}").resolve()), a.epochs,
                                           a.learning_rate, a.batch_size, a.warmup_steps,
                                           a.decay_steps, params, data, sk)
        posteriors.append((seed, inf.build_posterior()))
        print(f"  NDE seed {seed} {time.time()-t0:.0f}s", flush=True)

    # per-patch metrics (3-seed pooled posterior per obs)
    sig = np.full((N, 3), np.nan); pair = np.full((N, 3), np.nan); fom3 = np.full(N, np.nan)
    t0 = time.time()
    for i in range(N):
        pooled = []
        for seed, post in posteriors:
            k = jax.random.PRNGKey(seed * 100003 + int(sel[i]))
            pooled.append(np.asarray(post.sample(x=jnp.asarray(x_obs[i]), num_samples=M, key=k)))
        ps = np.concatenate(pooled, 0); ps = ps[np.all(np.isfinite(ps), 1)]
        if ps.shape[0] < 100:
            continue
        ms = marginal_stats(ps); f2 = fom2d(ps)
        sig[i] = list(ms["sigma"].values())[:3]      # Om, s8, w0
        pair[i] = list(f2.values())                  # Om_s8, Om_w0, s8_w0
        fom3[i] = compute_fom3(ps)["fom3"]
        if (i + 1) % 500 == 0:
            print(f"  sampled {i+1}/{N} ({time.time()-t0:.0f}s)", flush=True)

    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "per_patch_metrics.npz", sigma=sig, fom2d=pair, fom3=fom3,
             perm=perm[sel], patch=fz["patch"][sel], sel=sel,
             truth=(truth if truth is not None else np.array([])))
    g = np.isfinite(fom3)
    med = dict(arm=a.arm_label, n=int(g.sum()),
               sigma_Om=float(np.median(sig[g, 0])), sigma_s8=float(np.median(sig[g, 1])),
               sigma_w0=float(np.median(sig[g, 2])),
               fom2d_Om_s8=float(np.median(pair[g, 0])), fom2d_Om_w0=float(np.median(pair[g, 1])),
               fom2d_s8_w0=float(np.median(pair[g, 2])), fom3=float(np.median(fom3[g])))
    import json
    json.dump(med, open(out / "median_summary.json", "w"), indent=2)
    print(f"[{a.arm_label}] MEDIAN over {med['n']} obs: sig(Om,s8,w0)="
          f"{med['sigma_Om']:.3f},{med['sigma_s8']:.3f},{med['sigma_w0']:.3f}  FoM3={med['fom3']:.0f}",
          flush=True)


if __name__ == "__main__":
    main()
