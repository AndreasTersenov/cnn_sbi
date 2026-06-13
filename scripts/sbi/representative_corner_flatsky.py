#!/usr/bin/env python3
"""Re-sample one flat-local arm's pooled-3-seed posterior at a few specific fiducial obs (for
REPRESENTATIVE corner plots). Retrains in-process (high-dim reload truncates), samples at the
typical patch (perm16/patch23, auto FoM3=median) and the earlier favorable one (perm0/patch90),
saves the pooled samples. Mirrors population_sweep_flatsky.py."""
import argparse, sys, time
from pathlib import Path
import numpy as np
REPO = "/mnt/home/tersenov/software/cnn_sbi"; SBI = f"{REPO}/scripts/sbi"
OBS = [(16, 23, "typical"), (0, 90, "favorable")]


def main():
    p = argparse.ArgumentParser()
    for k in ("train-cache-dir", "arm-label", "fiducial-summaries-npz", "output-dir"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--seeds", default="41,42,43"); p.add_argument("--m-samples", type=int, default=4000)
    p.add_argument("--cuda-visible-devices", default="1")
    # preproc flags (default = the historical log1p-zscore/clip5/min-var1e-5 behaviour;
    # A1's compressed cache needs none/0/1e-12 — compressed features can be negative)
    p.add_argument("--preproc-transform", default="log1p-zscore")
    p.add_argument("--clip-value", type=float, default=5.0)
    p.add_argument("--min-feature-variance", type=float, default=1e-5)
    a = p.parse_args()
    sys.path.insert(0, REPO); sys.path.insert(0, SBI)
    from train_jaxili_from_compressed import setup_env
    setup_env(a.cuda_visible_devices)
    import jax, jax.numpy as jnp
    from jaxili.inference import NPE
    from npe_l1norm_cross_jaxili_nbody_tomo import (train_with_nan_retry, preprocess_summaries,
                                                    filter_zero_variance_bins)
    cdir = Path(a.train_cache_dir)
    tr = np.load(cdir / "l1_train.npz"); va = np.load(cdir / "l1_val.npz")
    theta_tr = tr["theta"].astype(np.float32); x_tr_raw = tr["x"].astype(np.float64)
    clipv = a.clip_value if a.clip_value > 0 else None
    tr_p, _, _, mean, std = preprocess_summaries(x_tr_raw, va["x"][:1].astype(np.float64),
                                                 va["x"][:1].astype(np.float64),
                                                 summary_transform=a.preproc_transform,
                                                 clip_value=clipv)
    mask, _ = filter_zero_variance_bins(tr_p, min_variance=a.min_feature_variance, verbose=False)
    x_tr = tr_p[:, mask].astype(np.float32)
    fz = np.load(a.fiducial_summaries_npz); S = fz["S"].astype(np.float64)
    rows = []
    for pm, pa, _ in OBS:
        rows.append(int(np.where((fz["perm"] == pm) & (fz["patch"] == pa))[0][0]))
    _, _, fid_p, _, _ = preprocess_summaries(x_tr_raw, va["x"][:1].astype(np.float64), S[rows],
                                             summary_transform=a.preproc_transform,
                                             clip_value=clipv, mean=mean, std=std)
    x_obs = fid_p[:, mask].astype(np.float32)
    params = jnp.asarray(theta_tr); data = jnp.asarray(x_tr)
    posts = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        t0 = time.time(); sk = jax.random.PRNGKey(seed + 1)
        inf = NPE().append_simulations(params, data, key=sk)
        inf, _m, _d = train_with_nan_retry(inf, str((Path(a.output_dir) / f"ck{seed}").resolve()),
                                           50000, 1e-4, 256, 100, 10000, params, data, sk)
        posts.append((seed, inf.build_posterior())); print(f"  seed {seed} {time.time()-t0:.0f}s", flush=True)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    samples = {}
    for r, (pm, pa, tag) in zip(range(len(rows)), OBS):
        pooled = [np.asarray(post.sample(x=jnp.asarray(x_obs[r]), num_samples=a.m_samples,
                  key=jax.random.PRNGKey(seed * 7 + r))) for seed, post in posts]
        ps = np.concatenate(pooled, 0); samples[tag] = ps[np.all(np.isfinite(ps), 1)]
    np.savez(out / "corner_samples.npz", **samples)
    print(f"[{a.arm_label}] saved corner_samples.npz "
          f"({ {t: samples[t].shape for t in samples} })", flush=True)


if __name__ == "__main__":
    main()
