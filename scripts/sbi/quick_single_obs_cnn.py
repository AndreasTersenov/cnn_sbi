#!/usr/bin/env python3
"""QUICK single-obs sanity look for one flat-local CNN arm (NOT the population result).

Trains ONE MAF seed on the arm's CNN compressed cache (faithful to population_sweep_flatsky's
preprocess + train), samples the pooled posterior at ONE fiducial obs, computes FoM3 / marginal
sigma / 2D area, and saves a getdist corner. Single-seed + single-obs => a reasonable-ness check
and a rough L1 comparison, NOT the calibrated 9000-obs median. Label it as such.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np

REPO = "/mnt/home/tersenov/software/cnn_sbi"; SBI = f"{REPO}/scripts/sbi"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm-label", required=True)
    p.add_argument("--train-cache-dir", required=True)
    p.add_argument("--cache-prefix", default="cnn")
    p.add_argument("--fiducial-summaries-npz", required=True)
    p.add_argument("--obs-perm", type=int, default=0)
    p.add_argument("--obs-patch", type=int, default=90)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seeds", default="41")           # 1 seed for speed; can pass 41,42,43
    p.add_argument("--preproc-transform", default="none")
    p.add_argument("--clip-value", type=float, default=0.0)
    p.add_argument("--min-feature-variance", type=float, default=1e-12)
    p.add_argument("--epochs", type=int, default=50000)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--decay-steps", type=int, default=10000)
    p.add_argument("--m-samples", type=int, default=4000)
    p.add_argument("--cuda-visible-devices", default="0")
    a = p.parse_args()

    sys.path.insert(0, REPO); sys.path.insert(0, SBI)
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

    fz = np.load(a.fiducial_summaries_npz)
    S = fz["S"].astype(np.float64); perm = fz["perm"]; patch = fz["patch"]
    row = int(np.where((perm == a.obs_perm) & (patch == a.obs_patch))[0][0])
    _, _, fid_p, _, _ = preprocess_summaries(
        x_tr_raw, x_va_raw[:1], S[row:row+1], summary_transform=a.preproc_transform,
        clip_value=clip, mean=mean, std=std)
    x_obs = fid_p[:, mask].astype(np.float32)[0]
    truth = fz["theta"][row] if "theta" in fz.files else None
    print(f"[{a.arm_label}] obs = perm{a.obs_perm}/patch{a.obs_patch}; truth={truth}", flush=True)

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    params = jnp.asarray(theta_tr); data = jnp.asarray(x_tr)
    tmp = Path(a.output_dir) / "ckpts"
    pooled = []
    for seed in seeds:
        t0 = time.time(); sk = jax.random.PRNGKey(seed + 1)
        inf = NPE().append_simulations(params, data, key=sk)
        inf, _m, _d = train_with_nan_retry(inf, str((tmp / f"s{seed}").resolve()), a.epochs,
                                           a.learning_rate, a.batch_size, a.warmup_steps,
                                           a.decay_steps, params, data, sk)
        post = inf.build_posterior()
        k = jax.random.PRNGKey(seed * 100003 + row)
        pooled.append(np.asarray(post.sample(x=jnp.asarray(x_obs), num_samples=a.m_samples, key=k)))
        print(f"  NDE seed {seed} {time.time()-t0:.0f}s", flush=True)

    ps = np.concatenate(pooled, 0); ps = ps[np.all(np.isfinite(ps), 1)]
    ms = marginal_stats(ps); f2 = fom2d(ps); f3 = compute_fom3(ps)["fom3"]
    sig = list(ms["sigma"].values())
    print(f"\n[{a.arm_label}] SINGLE-OBS (perm{a.obs_perm}/patch{a.obs_patch}, "
          f"{len(seeds)}-seed, {ps.shape[0]} samples):", flush=True)
    print(f"  sigma(Om,s8,w0) = {sig[0]:.4f}, {sig[1]:.4f}, {sig[2]:.4f}", flush=True)
    print(f"  2D(Om,s8)={list(f2.values())[0]:.0f}  FoM3 = {f3:.0f}", flush=True)

    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"single_obs_{a.arm_label}.npz", samples=ps, truth=(truth if truth is not None else []),
             sigma=np.array(sig), fom3=np.float64(f3))

    # --- getdist corner (3 science params) ---
    labels = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]
    names = ["Om", "s8", "w0", "h0", "ns", "Ob"]
    try:
        from getdist import MCSamples, plots
        import matplotlib; matplotlib.use("Agg")
        idx = [0, 1, 2]   # Om, s8, w0
        smp = MCSamples(samples=ps[:, idx], names=[names[i] for i in idx],
                        labels=[labels[i] for i in idx], label=f"CNN {a.arm_label}")
        markers = (None if truth is None
                   else {names[idx[j]]: float(truth[idx[j]]) for j in range(len(idx))})
        g = plots.get_subplot_plotter()
        g.triangle_plot([smp], filled=True, title_limit=1, markers=markers)
        png = out / f"single_obs_corner_{a.arm_label}.png"
        g.export(str(png))
        print(f"  wrote {png}", flush=True)
    except Exception as e:
        print(f"  [warn] getdist plot failed ({e}); samples saved for offline plotting.", flush=True)


if __name__ == "__main__":
    main()
