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
    p.add_argument("--sample-eager", action="store_true",
                   help="Legacy un-jitted per-obs sampling (reproduces pre-2026-06-10 "
                        "sweeps bit-for-bit; ~100x slower than the default jitted path).")
    p.add_argument("--epochs", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    # NB: currently INERT — jaxili's NPE.train does not forward decay_steps to the
    # optimizer (cosine horizon defaults to ~num_epochs/2), so the effective LR is
    # constant 1e-4 after warmup. Kept for CLI compatibility; do not tune via this.
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
    # PRNG keys below are seed*100003 + sel[i]; obs indices >= 100003 would alias
    # seed s / obs (100003+k) with seed s+1 / obs k.
    assert sel.size > 0 and int(sel.max()) < 100003, \
        f"obs index {int(sel.max())} >= 100003 would collide PRNG keys across seeds"
    _, _, fid_p, _, _ = preprocess_summaries(
        x_tr_raw, x_va_raw[:1], S[sel], summary_transform=a.preproc_transform,
        clip_value=clip, mean=mean, std=std)
    x_obs = fid_p[:, mask].astype(np.float32)
    # fidsumm files written before 2026-06-10 stored the key as `theta`
    truth = next((fz[k] for k in ("truth", "theta") if k in fz.files), None)
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

    # Per-posterior samplers. Default: jitted closure — jaxili's eager
    # DirectPosterior.sample is un-jitted (~600 tiny dispatches/call, ~180 ms/obs
    # measured); jit collapses it to ~1 ms/obs (~90x sweep-phase speedup). Same
    # per-obs PRNG keys; samples differ only at the TF32 kernel level (bench:
    # max|dev|=3.4e-3; full-arm 9000-obs median FoM3 dev -0.39%, see
    # multiseed/population_sweep/none_s42/jit_validation.json). Adopted 2026-06-10;
    # --sample-eager reproduces the legacy path bit-for-bit.
    samplers = []
    for seed, post in posteriors:
        if a.sample_eager:
            samplers.append((seed, lambda x, k, _p=post: _p.sample(x=x, num_samples=M, key=k)))
        else:
            _params, _model = post.state.params, post.model
            samplers.append((seed, jax.jit(
                lambda x, k, _m=_model, _p=_params: _m.apply(
                    {"params": _p}, x, M, k, method="sample"))))
    x_dev = jnp.asarray(x_obs)

    # per-patch metrics (3-seed pooled posterior per obs)
    sig = np.full((N, 3), np.nan); pair = np.full((N, 3), np.nan); fom3 = np.full(N, np.nan)
    n_finite = np.zeros(N, dtype=np.int32)   # surviving pooled samples per obs (of 3*M)
    t0 = time.time()
    for i in range(N):
        pooled = []
        for seed, fn in samplers:
            k = jax.random.PRNGKey(seed * 100003 + int(sel[i]))
            pooled.append(np.asarray(fn(x_dev[i], k)))
        ps = np.concatenate(pooled, 0); ps = ps[np.all(np.isfinite(ps), 1)]
        n_finite[i] = ps.shape[0]
        if ps.shape[0] < 100:
            continue
        ms = marginal_stats(ps); f2 = fom2d(ps)
        sig[i] = [ms["sigma"][k] for k in ("Omega_m", "sigma_8", "w_0")]
        pair[i] = [f2["fom2d_Omega_m_sigma_8"], f2["fom2d_Omega_m_w_0"],
                   f2["fom2d_sigma_8_w_0"]]
        fom3[i] = compute_fom3(ps)["fom3"]
        if (i + 1) % 500 == 0:
            print(f"  sampled {i+1}/{N} ({time.time()-t0:.0f}s)", flush=True)
    n_skipped = int((n_finite < 100).sum())
    n_partial = int(((n_finite >= 100) & (n_finite < 3 * M)).sum())
    if n_skipped or n_partial:
        print(f"  [warn] {n_skipped} obs skipped (<100 finite), {n_partial} obs with "
              f"partial sample loss (<{3*M} finite) — see n_finite in per_patch_metrics.npz",
              flush=True)

    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "per_patch_metrics.npz", sigma=sig, fom2d=pair, fom3=fom3,
             perm=perm[sel], patch=fz["patch"][sel], sel=sel, n_finite=n_finite,
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
