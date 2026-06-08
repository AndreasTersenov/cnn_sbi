#!/usr/bin/env python3
"""Coverage test for the per-patch fiducial posteriors (the full-200 step-2 question):
is the high, variable per-patch L1 FoM3 REAL information or OVER-TIGHT / under-covered?

The high FoM3 is at the FIXED fiducial cosmology over many patches, so the right test is
**coverage of the fiducial truth over the patch population**. Two complementary measures:

  1. Mahalanobis-χ²₃ coverage (PRIMARY, tied to FoM3): per patch, pool seeds, take the
     (Ωm,σ8,w0) covariance Σ (the SAME Σ FoM3=1/√detΣ uses) and mean μ; D²=(θfid-μ)ᵀΣ⁻¹(θfid-μ).
     If calibrated, D² ~ χ²₃ across patches → empirical coverage at 68/95% matches. If
     over-tight (Σ too small ⟹ FoM3 inflated), D² is systematically large → under-coverage.
  2. A per-seed posterior dump in run_tarp_coverage format (theta=fiducial) for the standard
     TARP/DRP figure (non-parametric cross-check).

Also reports patch_idx==0 (the campaign's fixed obs) vs the patch population, to test the
"patch-0 atypically low" finding robustly.

Reuses the campaign preprocessing (preprocess_summaries + filter_zero_variance_bins) and the
exact NDE (train_with_nan_retry + NPE), matching the arm's posteriors.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
FIDUCIAL = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache-dir", required=True)
    p.add_argument("--cache-prefix", choices=["cnn", "l1"], required=True)
    p.add_argument("--summaries-npz", required=True)
    p.add_argument("--arm-label", required=True)
    p.add_argument("--dumps-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--preproc-transform", default="none")
    p.add_argument("--clip-value", type=float, default=0.0)
    p.add_argument("--min-feature-variance", type=float, default=1e-12)
    p.add_argument("--seeds", default="41,42,43")
    p.add_argument("--n-patches", type=int, default=200)
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
    from train_jaxili_from_compressed import setup_env
    setup_env(a.cuda_visible_devices)
    import jax, jax.numpy as jnp
    from jaxili.inference import NPE
    from npe_l1norm_cross_jaxili_nbody_tomo import (
        train_with_nan_retry, preprocess_summaries, filter_zero_variance_bins)
    from scipy.stats import chi2

    cdir = Path(a.train_cache_dir)
    tr = np.load(cdir / f"{a.cache_prefix}_train.npz")
    theta_tr = tr["theta"].astype(np.float32); x_tr_raw = tr["x"].astype(np.float64)
    z = np.load(a.summaries_npz)
    S_raw = z["S"].astype(np.float64); perm = z["perm"]; patch = z["patch"]

    clip = a.clip_value if a.clip_value > 0 else None
    tr_proc, _, S_proc, mean, std = preprocess_summaries(
        x_tr_raw, x_tr_raw[:1], S_raw, summary_transform=a.preproc_transform, clip_value=clip)
    mask, _ = filter_zero_variance_bins(tr_proc, min_variance=a.min_feature_variance, verbose=False)
    x_tr = tr_proc[:, mask].astype(np.float32); S = S_proc[:, mask].astype(np.float32)
    print(f"[{a.arm_label}] train{x_tr.shape} S{S.shape}", flush=True)

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    posteriors = []
    params = jnp.asarray(theta_tr); data = jnp.asarray(x_tr)
    out_dir = Path(a.output_dir) / a.arm_label; out_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        t0 = time.time(); sk = jax.random.PRNGKey(seed + 1)
        inf = NPE().append_simulations(params, data, key=sk)
        ckpt = str((out_dir / f"ckpt_{seed}").resolve())
        inf, _m, _d = train_with_nan_retry(inf, ckpt, a.epochs, a.learning_rate, a.batch_size,
                                           a.warmup_steps, a.decay_steps, params, data, sk)
        posteriors.append((seed, inf.build_posterior()))
        print(f"  NDE seed {seed} {time.time()-t0:.0f}s", flush=True)

    # choose N patches: random population + force-include up to 60 patch_idx==0 instances
    rng = np.random.default_rng(7)
    p0_idx = np.where(patch == 0)[0]
    p0_take = p0_idx[rng.choice(len(p0_idx), size=min(60, len(p0_idx)), replace=False)]
    rest = np.setdiff1d(np.arange(S.shape[0]), p0_take)
    pop_take = rest[rng.choice(len(rest), size=min(a.n_patches, len(rest)), replace=False)]
    sel = np.concatenate([pop_take, p0_take])
    N = len(sel); M = a.m_samples
    print(f"  N={N} patches ({len(pop_take)} population + {len(p0_take)} patch0)", flush=True)

    # sample: per-seed dump array (N,M,6) + pooled per patch for Mahalanobis
    dump = {seed: np.empty((N, M, 6), np.float32) for seed, _ in posteriors}
    D2 = np.empty(N); fom3 = np.empty(N)
    for i, gi in enumerate(sel):
        pooled = []
        for seed, post in posteriors:
            k = jax.random.PRNGKey(seed * 100003 + int(gi))
            s = np.asarray(post.sample(x=jnp.asarray(S[gi]), num_samples=M, key=k))
            dump[seed][i] = s; pooled.append(s)
        ps = np.concatenate(pooled, 0); ps = ps[np.all(np.isfinite(ps), 1)]
        c3 = np.cov(ps[:, :3], rowvar=False); mu = ps[:, :3].mean(0)
        d = FIDUCIAL[:3] - mu
        D2[i] = float(d @ np.linalg.solve(c3, d))
        fom3[i] = float(1.0 / np.sqrt(np.linalg.det(c3)))
        if (i + 1) % 50 == 0:
            print(f"  sampled {i+1}/{N}", flush=True)

    # ---- Mahalanobis-χ²₃ coverage ----
    is_p0 = np.array([patch[gi] == 0 for gi in sel])
    def cov_at(level):  # fraction with D2 <= chi2_3 quantile(level)
        thr = chi2.ppf(level, df=3)
        return float(np.mean(D2 <= thr))
    levels = np.array([0.5, 0.68, 0.90, 0.95, 0.99])
    emp = {f"{int(l*100)}%": cov_at(l) for l in levels}
    # full ECP curve: empirical CDF of chi2_3.cdf(D2) should be uniform
    u = chi2.cdf(D2, df=3)
    aa = np.linspace(0, 1, 51); ecp = np.array([np.mean(u <= x) for x in aa])
    res = {
        "arm": a.arm_label, "n_patches": int(N),
        "coverage_vs_nominal": emp,
        "fom3_pop_median": float(np.median(fom3[~is_p0])),
        "fom3_patch0_median": float(np.median(fom3[is_p0])) if is_p0.any() else None,
        "D2_median": float(np.median(D2)), "chi2_3_median": float(chi2.ppf(0.5, df=3)),
        "verdict": ("OVER-CONFIDENT (under-covers -> FoM3 inflated)"
                    if cov_at(0.68) < 0.60 else
                    "calibrated/conservative (coverage ok -> FoM3 trustworthy)"),
    }
    (out_dir / "coverage.json").write_text(json.dumps(res, indent=2))
    np.savez(out_dir / "coverage_arrays.npz", D2=D2, fom3=fom3, is_p0=is_p0,
             alpha=aa, ecp=ecp)
    print(f"  [coverage] 68%-cred contains truth in {emp['68%']*100:.0f}% of patches "
          f"(nominal 68); 95% -> {emp['95%']*100:.0f}%. D2 median {res['D2_median']:.2f} "
          f"(χ²₃ median {res['chi2_3_median']:.2f}). VERDICT: {res['verdict']}", flush=True)
    if is_p0.any():
        print(f"  [patch0] FoM3 median patch0 {res['fom3_patch0_median']:.0f} vs "
              f"population {res['fom3_pop_median']:.0f}", flush=True)

    # ---- dump per-seed for run_tarp_coverage (theta = fiducial repeated) ----
    th = np.tile(FIDUCIAL.astype(np.float32), (N, 1))
    for seed in dump:
        dd = Path(a.dumps_root) / a.arm_label / f"seed_{seed}" / f"n{N}_m{M}"
        dd.mkdir(parents=True, exist_ok=True)
        np.savez(dd / "posterior_samples.npz", samples=dump[seed], theta=th)
    print(f"[{a.arm_label}] dumps -> {Path(a.dumps_root)/a.arm_label}", flush=True)


if __name__ == "__main__":
    main()
