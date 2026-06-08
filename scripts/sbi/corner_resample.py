#!/usr/bin/env python3
"""Headline-corner re-sampling: re-train the campaign-EXACT common jaxili MAF
(3 seeds) for ONE arm and draw a large posterior at ONE observed patch, saving
the RAW per-seed sample arrays (so the corner can be plotted with no smoothing,
per-seed or pooled).

Same NDE + preprocessing path as geometry_resample.py / fiducial_analyze.py
(G3-validated). Unlike the TARP dumps (2000 samples/seed) this draws
--samples-per-seed (default 30000) at a single (patch, perm) obs, which is what
the headline corner needs to judge whether marginal "bumpiness" is real summary
structure (seed-consistent) or sampling noise (seed-varying).

Writes <output-dir>/<arm-label>/corner_samples.npz with:
  samples_seed{41,42,43} : (n_kept, 6) raw posterior draws per seed
  pooled                 : (sum n_kept, 6) all seeds concatenated
  obs_patch, obs_perm, fom3_per_seed, fom3_pooled, truth
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache-dir", required=True)
    p.add_argument("--cache-prefix", choices=["cnn", "l1"], required=True)
    p.add_argument("--summaries-npz", required=True)
    p.add_argument("--arm-label", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--preproc-transform", default="none",
                   choices=["none", "zscore", "log1p-zscore", "log10p-zscore"])
    p.add_argument("--clip-value", type=float, default=0.0, help="0 => no clip")
    p.add_argument("--min-feature-variance", type=float, default=1e-12)
    p.add_argument("--seeds", default="41,42,43")
    p.add_argument("--obs-patch", type=int, required=True)
    p.add_argument("--obs-perm", type=int, required=True)
    p.add_argument("--samples-per-seed", type=int, default=30000)
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
    from train_jaxili_from_compressed import setup_env, compute_fom3, FIDUCIAL
    setup_env(a.cuda_visible_devices)
    import jax, jax.numpy as jnp
    from jaxili.inference import NPE
    from npe_l1norm_cross_jaxili_nbody_tomo import (
        train_with_nan_retry, preprocess_summaries, filter_zero_variance_bins,
    )

    cdir = Path(a.train_cache_dir)
    tr = np.load(cdir / f"{a.cache_prefix}_train.npz")
    theta_tr = tr["theta"].astype(np.float32)
    x_tr_raw = tr["x"].astype(np.float64)

    z = np.load(a.summaries_npz)
    S_raw = z["S"].astype(np.float64)
    perm = z["perm"].astype(int); patch = z["patch"].astype(int)
    pos = {(int(perm[k]), int(patch[k])): k for k in range(len(perm))}
    key = (a.obs_perm, a.obs_patch)
    if key not in pos:
        raise SystemExit(f"obs (perm={a.obs_perm}, patch={a.obs_patch}) not in summaries")
    assert S_raw.shape[1] == x_tr_raw.shape[1], \
        f"dim mismatch S={S_raw.shape[1]} vs train={x_tr_raw.shape[1]}"

    # ---- preprocessing fit on TRAIN, applied to S (campaign-exact) ----
    clip = a.clip_value if a.clip_value > 0 else None
    tr_proc, _, S_proc, mean, std = preprocess_summaries(
        x_tr_raw, x_tr_raw[:1], S_raw, summary_transform=a.preproc_transform, clip_value=clip)
    mask, n_removed = filter_zero_variance_bins(tr_proc, min_variance=a.min_feature_variance, verbose=False)
    x_tr = tr_proc[:, mask].astype(np.float32)
    S = S_proc[:, mask].astype(np.float32)
    x_obs = S[pos[key]]
    print(f"[{a.arm_label}] train x{x_tr.shape}; masked dim {x_tr.shape[1]}/{mask.size} "
          f"(removed {n_removed}); obs (patch {a.obs_patch}, perm {a.obs_perm})", flush=True)

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    out_dir = Path(a.output_dir) / a.arm_label
    out_dir.mkdir(parents=True, exist_ok=True)
    params = jnp.asarray(theta_tr); data = jnp.asarray(x_tr)

    saved = {}
    fom3_per_seed = {}
    pooled = []
    for seed in seeds:
        t0 = time.time()
        sk = jax.random.PRNGKey(int(seed) + 1)
        inf = NPE().append_simulations(params, data, key=sk)
        ckpt = str((out_dir / f"ckpt_{a.arm_label}_s{seed}").resolve())
        inf, _m, _d = train_with_nan_retry(inf, ckpt, a.epochs, a.learning_rate,
                                           a.batch_size, a.warmup_steps, a.decay_steps,
                                           params, data, sk)
        post = inf.build_posterior()
        ks = jax.random.PRNGKey(int(seed) + 7)
        s = np.asarray(post.sample(x=jnp.asarray(x_obs), num_samples=a.samples_per_seed, key=ks))
        s = s[np.all(np.isfinite(s), axis=1)]
        saved[f"samples_seed{seed}"] = s
        pooled.append(s)
        fom3_per_seed[seed] = compute_fom3(s)["fom3"]
        print(f"  seed {seed}: {s.shape[0]} kept, FoM3={fom3_per_seed[seed]:.0f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    pooled = np.concatenate(pooled, 0)
    np.savez(out_dir / "corner_samples.npz",
             pooled=pooled, obs_patch=a.obs_patch, obs_perm=a.obs_perm,
             truth=np.asarray(FIDUCIAL, float),
             fom3_pooled=compute_fom3(pooled)["fom3"],
             fom3_per_seed=np.array([fom3_per_seed[s] for s in seeds]),
             seeds=np.array(seeds), **saved)
    (out_dir / "corner_meta.json").write_text(json.dumps({
        "arm": a.arm_label, "obs_patch": a.obs_patch, "obs_perm": a.obs_perm,
        "seeds": seeds, "samples_per_seed": a.samples_per_seed,
        "fom3_per_seed": {str(k): v for k, v in fom3_per_seed.items()},
        "fom3_pooled": compute_fom3(pooled)["fom3"],
        "preproc": {"transform": a.preproc_transform, "clip": a.clip_value,
                    "min_var": a.min_feature_variance},
    }, indent=2))
    print(f"[{a.arm_label}] DONE -> {out_dir/'corner_samples.npz'} "
          f"(pooled {pooled.shape[0]}, FoM3={compute_fom3(pooled)['fom3']:.0f})", flush=True)


if __name__ == "__main__":
    main()
