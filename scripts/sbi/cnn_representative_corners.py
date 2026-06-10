#!/usr/bin/env python3
"""3-seed pooled CNN posteriors at the representative obs (typical + favorable) for one arm.

CNN analog of representative_corner_flatsky.py: trains 3 MAF seeds once on the arm's CNN compressed
cache (preproc none), samples the pooled posterior at the typical (perm16/patch23) and favorable
(perm0/patch90) fiducial obs, and saves corner_samples.npz with keys 'typical'/'favorable' (same
layout as the L1 representative corners, so overlays are trivial). Mirrors population_sweep's train.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np

REPO = "/mnt/home/tersenov/software/cnn_sbi"; SBI = f"{REPO}/scripts/sbi"
OBS = [(16, 23, "typical"), (0, 90, "favorable")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm-label", required=True)
    p.add_argument("--train-cache-dir", required=True)
    p.add_argument("--cache-prefix", default="cnn")
    p.add_argument("--fiducial-summaries-npz", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seeds", default="41,42,43")
    p.add_argument("--m-samples", type=int, default=4000)
    p.add_argument("--preproc-transform", default="none")
    p.add_argument("--clip-value", type=float, default=0.0)
    p.add_argument("--min-feature-variance", type=float, default=1e-12)
    p.add_argument("--epochs", type=int, default=50000)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--decay-steps", type=int, default=10000)
    p.add_argument("--cuda-visible-devices", default="1")
    a = p.parse_args()

    sys.path.insert(0, REPO); sys.path.insert(0, SBI)
    from train_jaxili_from_compressed import setup_env, compute_fom3, marginal_stats
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

    fz = np.load(a.fiducial_summaries_npz)
    S = fz["S"].astype(np.float64); perm = fz["perm"]; patch = fz["patch"]
    rows = {lab: int(np.where((perm == pm) & (patch == pa))[0][0]) for pm, pa, lab in OBS}
    _, _, fid_p, _, _ = preprocess_summaries(
        x_tr_raw, x_va_raw[:1], S[[rows[l] for _, _, l in OBS]],
        summary_transform=a.preproc_transform, clip_value=clip, mean=mean, std=std)
    x_obs = {lab: fid_p[i, mask].astype(np.float32) for i, (_, _, lab) in enumerate(OBS)}

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    params = jnp.asarray(theta_tr); data = jnp.asarray(x_tr)
    tmp = Path(a.output_dir) / "ckpts"
    posteriors = []
    for seed in seeds:
        t0 = time.time(); sk = jax.random.PRNGKey(seed + 1)
        inf = NPE().append_simulations(params, data, key=sk)
        inf, _m, _d = train_with_nan_retry(inf, str((tmp / f"s{seed}").resolve()), a.epochs,
                                           a.learning_rate, a.batch_size, a.warmup_steps,
                                           a.decay_steps, params, data, sk)
        posteriors.append((seed, inf.build_posterior()))
        print(f"[{a.arm_label}] NDE seed {seed} {time.time()-t0:.0f}s", flush=True)

    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    saved = {}
    for _, _, lab in OBS:
        pooled = []
        for seed, post in posteriors:
            k = jax.random.PRNGKey(seed * 100003 + rows[lab])
            pooled.append(np.asarray(post.sample(x=jnp.asarray(x_obs[lab]), num_samples=a.m_samples, key=k)))
        ps = np.concatenate(pooled, 0); ps = ps[np.all(np.isfinite(ps), 1)]
        saved[lab] = ps.astype(np.float32)
        f3 = compute_fom3(ps)["fom3"]; sg = list(marginal_stats(ps)["sigma"].values())
        print(f"[{a.arm_label}] {lab} (perm{rows[lab]} row): FoM3={f3:.0f} "
              f"sig(Om,s8,w0)={sg[0]:.4f},{sg[1]:.4f},{sg[2]:.4f}  N={ps.shape[0]}", flush=True)
    np.savez(out / "corner_samples.npz", **saved)
    print(f"[{a.arm_label}] wrote {out / 'corner_samples.npz'}", flush=True)


if __name__ == "__main__":
    main()
