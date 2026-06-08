#!/usr/bin/env python3
"""#2: proper varied-θ TARP-DRP, STRATIFIED by per-point FoM3, to test whether the
TIGHT (high-FoM3) L1 posteriors are calibrated or over-confident.

Uses the held-out VAL ensemble (varied cosmologies + their patches) — so θ is drawn
from the prior and TARP-DRP is VALID (unlike the fixed-θ fiducial test). For one arm:
  - train the NDE (3 seeds), reusing the campaign preprocessing,
  - sample posteriors at N val points, compute per-point FoM3 (3-seed pooled),
  - split into FoM3 terciles (LOW / MID / HIGH),
  - dump per (tercile, seed) in run_tarp_coverage format (theta = the val truths),
  - report the θ-spread within each tercile (TARP-validity check) + FoM3 ranges.
Then `run_tarp_coverage.py` on the dumps gives a DRP curve per tercile: if the HIGH
tercile sits on the diagonal, the tight posteriors are calibrated (FoM3 real); if it
dives below, they are over-confident (FoM3 inflated).
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache-dir", required=True)
    p.add_argument("--cache-prefix", choices=["cnn", "l1"], required=True)
    p.add_argument("--arm-label", required=True)
    p.add_argument("--dumps-root", required=True)
    p.add_argument("--preproc-transform", default="none")
    p.add_argument("--clip-value", type=float, default=0.0)
    p.add_argument("--min-feature-variance", type=float, default=1e-12)
    p.add_argument("--seeds", default="41,42,43")
    p.add_argument("--n-points", type=int, default=600)
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
    sys.path.insert(0, str(REPO))
    from train_jaxili_from_compressed import setup_env, compute_fom3
    setup_env(a.cuda_visible_devices)
    import jax, jax.numpy as jnp
    from jaxili.inference import NPE
    from npe_l1norm_cross_jaxili_nbody_tomo import (
        train_with_nan_retry, preprocess_summaries, filter_zero_variance_bins)

    cdir = Path(a.train_cache_dir)
    tr = np.load(cdir / f"{a.cache_prefix}_train.npz")
    va = np.load(cdir / f"{a.cache_prefix}_val.npz")
    theta_tr = tr["theta"].astype(np.float32); x_tr_raw = tr["x"].astype(np.float64)
    theta_va = va["theta"].astype(np.float64); x_va_raw = va["x"].astype(np.float64)
    print(f"[{a.arm_label}] train{x_tr_raw.shape} val{x_va_raw.shape}", flush=True)

    clip = a.clip_value if a.clip_value > 0 else None
    tr_p, va_p, _, mean, std = preprocess_summaries(
        x_tr_raw, x_va_raw, x_va_raw[:1], summary_transform=a.preproc_transform, clip_value=clip)
    mask, _ = filter_zero_variance_bins(tr_p, min_variance=a.min_feature_variance, verbose=False)
    x_tr = tr_p[:, mask].astype(np.float32); x_va = va_p[:, mask].astype(np.float32)

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    posteriors = []
    params = jnp.asarray(theta_tr); data = jnp.asarray(x_tr)
    tmp = Path(a.dumps_root).parent / "ckpts" / a.arm_label
    for seed in seeds:
        t0 = time.time(); sk = jax.random.PRNGKey(seed + 1)
        inf = NPE().append_simulations(params, data, key=sk)
        inf, _m, _d = train_with_nan_retry(inf, str((tmp / f"s{seed}").resolve()), a.epochs,
                                           a.learning_rate, a.batch_size, a.warmup_steps,
                                           a.decay_steps, params, data, sk)
        posteriors.append((seed, inf.build_posterior()))
        print(f"  NDE seed {seed} {time.time()-t0:.0f}s", flush=True)

    rng = np.random.default_rng(3)
    N = int(min(a.n_points, x_va.shape[0])); M = a.m_samples
    idx = rng.choice(x_va.shape[0], size=N, replace=False)
    # per-seed samples (N,M,6) + pooled FoM3
    samp = {s: np.empty((N, M, 6), np.float32) for s, _ in posteriors}
    fom3 = np.empty(N)
    for i, gi in enumerate(idx):
        pooled = []
        for seed, post in posteriors:
            k = jax.random.PRNGKey(seed * 100003 + int(gi))
            s = np.asarray(post.sample(x=jnp.asarray(x_va[gi]), num_samples=M, key=k))
            samp[seed][i] = s; pooled.append(s)
        ps = np.concatenate(pooled, 0); ps = ps[np.all(np.isfinite(ps), 1)]
        fom3[i] = compute_fom3(ps)["fom3"]
        if (i + 1) % 100 == 0:
            print(f"  sampled {i+1}/{N}", flush=True)

    th = theta_va[idx].astype(np.float32)
    q1, q2 = np.percentile(fom3, [33.33, 66.67])
    strata = {"LOW": fom3 <= q1, "MID": (fom3 > q1) & (fom3 < q2), "HIGH": fom3 >= q2}
    print(f"\n  FoM3 terciles: LOW<= {q1:.0f} <MID< {q2:.0f} <=HIGH (range {fom3.min():.0f}-{fom3.max():.0f})", flush=True)
    for name, msk in strata.items():
        thm = th[msk]
        print(f"  {a.arm_label} {name}: n={msk.sum()} FoM3 med {np.median(fom3[msk]):.0f} | "
              f"theta-spread Om {thm[:,0].std():.3f} s8 {thm[:,1].std():.3f} w0 {thm[:,2].std():.3f}", flush=True)
        for seed in samp:
            dd = Path(a.dumps_root) / f"{a.arm_label}_{name}" / f"seed_{seed}" / f"n{int(msk.sum())}_m{M}"
            dd.mkdir(parents=True, exist_ok=True)
            np.savez(dd / "posterior_samples.npz", samples=samp[seed][msk], theta=th[msk])
    print(f"[{a.arm_label}] dumps -> {a.dumps_root}", flush=True)


if __name__ == "__main__":
    main()
