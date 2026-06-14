#!/usr/bin/env python3
"""GATE C (varied-θ TARP-DRP, FoM3-tercile-stratified) for ARBITRARY NDE families.

Sibling of tarp_stratified_val.py, but tests the ACTUAL NDE under question (e.g. the sbi_lens
RealNVP that produced FoM3 3139) rather than the common jaxili MAF — because here the NDE family
IS what's under test (does its tight posterior reflect real information or over-confidence?).

Reuses the family-dispatch training from train_nde_from_compressed.py (so the trained estimator
is bit-identical to the sweep arm) and the dump format of tarp_stratified_val.py (so
run_tarp_coverage.py consumes it unchanged). Varied-θ set = cnn_val.npz (val cosmologies, disjoint
from train). Stratify the val posteriors by per-point FoM3; the HIGH tercile is where
over-confidence shows. SBC ranks computable from the same dumps.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import numpy as np

REPO = "/mnt/home/tersenov/software/cnn_sbi"
SBI = f"{REPO}/scripts/sbi"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache-dir", required=True)
    p.add_argument("--cache-prefix", default="cnn")
    p.add_argument("--arm-label", required=True)
    p.add_argument("--dumps-root", required=True)
    p.add_argument("--nde-family", required=True)
    p.add_argument("--nde-layers", type=int, default=-1)
    p.add_argument("--nde-hidden", type=int, default=-1)
    p.add_argument("--preproc-transform", default="none")
    p.add_argument("--clip-value", type=float, default=0.0)
    p.add_argument("--min-feature-variance", type=float, default=1e-12)
    p.add_argument("--seeds", default="41,42,43")
    p.add_argument("--n-points", type=int, default=600)
    p.add_argument("--m-samples", type=int, default=2000)
    # jaxili knobs
    p.add_argument("--epochs", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--decay-steps", type=int, default=10000)
    # sbi_lens flow knobs (match npe_cnn defaults)
    p.add_argument("--flow-total-steps", type=int, default=50000)
    p.add_argument("--flow-batch-size", type=int, default=128)
    p.add_argument("--flow-lr-init", type=float, default=1e-3)
    p.add_argument("--flow-lr-end", type=float, default=1e-5)
    p.add_argument("--flow-save-every", type=int, default=2000)
    p.add_argument("--flow-patience", type=int, default=20)
    p.add_argument("--flow-grad-clip", type=float, default=1.0)
    p.add_argument("--flow-weight-decay", type=float, default=1e-4)
    p.add_argument("--cuda-visible-devices", default="0")
    return p.parse_args()


def main():
    a = parse_args()
    sys.path.insert(0, REPO); sys.path.insert(0, SBI)
    from train_jaxili_from_compressed import setup_env, compute_fom3
    setup_env(a.cuda_visible_devices)
    import jax
    import jax.numpy as jnp
    from npe_l1norm_cross_jaxili_nbody_tomo import preprocess_summaries, filter_zero_variance_bins
    from train_nde_from_compressed import (resolve_capacity, train_sbilens_realnvp,
                                           train_jaxili_family)

    layers, hidden = resolve_capacity(a.nde_family, a.nde_layers, a.nde_hidden)
    cdir = Path(a.train_cache_dir)
    tr = np.load(cdir / f"{a.cache_prefix}_train.npz")
    va = np.load(cdir / f"{a.cache_prefix}_val.npz")
    theta_tr = tr["theta"].astype(np.float32); x_tr_raw = tr["x"].astype(np.float64)
    theta_va = va["theta"].astype(np.float64); x_va_raw = va["x"].astype(np.float64)
    print(f"[{a.arm_label}] family={a.nde_family} {layers}x{hidden} train{x_tr_raw.shape} val{x_va_raw.shape}", flush=True)

    clip = a.clip_value if a.clip_value > 0 else None
    tr_p, va_p, _, mean, std = preprocess_summaries(
        x_tr_raw, x_va_raw, x_va_raw[:1], summary_transform=a.preproc_transform, clip_value=clip)
    mask, _ = filter_zero_variance_bins(tr_p, min_variance=a.min_feature_variance, verbose=False)
    x_tr = tr_p[:, mask].astype(np.float32); x_va = va_p[:, mask].astype(np.float32)
    dim = x_tr.shape[1]

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    M = a.m_samples
    out_ck = Path(a.dumps_root).parent / "ckpts" / a.arm_label
    out_ck.mkdir(parents=True, exist_ok=True)
    if a.nde_family == "sbilens_realnvp":
        samplers = train_sbilens_realnvp(layers, hidden, theta_tr, x_tr, seeds, M, dim, out_ck, a)
    else:
        samplers = train_jaxili_family(a.nde_family, layers, hidden, theta_tr, x_tr, seeds, M, out_ck, a)

    # sample posteriors at N val points (varied θ) — reuse sampler_fn(x_single, key) -> (M,6)
    rng = np.random.default_rng(3)
    N = int(min(a.n_points, x_va.shape[0]))
    idx = rng.choice(x_va.shape[0], size=N, replace=False)
    x_dev = jnp.asarray(x_va)
    samp = {s: np.empty((N, M, 6), np.float32) for s, _ in samplers}
    fom3 = np.empty(N)
    t0 = time.time()
    for i, gi in enumerate(idx):
        pooled = []
        for seed, fn in samplers:
            k = jax.random.PRNGKey(seed * 100003 + int(gi))
            s = np.asarray(fn(x_dev[gi], k))
            samp[seed][i] = s; pooled.append(s)
        ps = np.concatenate(pooled, 0); ps = ps[np.all(np.isfinite(ps), 1)]
        fom3[i] = compute_fom3(ps)["fom3"]
        if (i + 1) % 100 == 0:
            print(f"  sampled {i+1}/{N} ({time.time()-t0:.0f}s)", flush=True)

    th = theta_va[idx].astype(np.float32)
    q1, q2 = np.percentile(fom3, [33.33, 66.67])
    strata = {"LOW": fom3 <= q1, "MID": (fom3 > q1) & (fom3 < q2), "HIGH": fom3 >= q2}
    print(f"\n  FoM3 terciles: LOW<= {q1:.0f} <MID< {q2:.0f} <=HIGH "
          f"(range {fom3.min():.0f}-{fom3.max():.0f})", flush=True)
    for name, msk in strata.items():
        thm = th[msk]
        print(f"  {a.arm_label} {name}: n={int(msk.sum())} FoM3 med {np.median(fom3[msk]):.0f} | "
              f"theta-spread Om {thm[:,0].std():.3f} s8 {thm[:,1].std():.3f} w0 {thm[:,2].std():.3f}",
              flush=True)
        for seed in samp:
            dd = Path(a.dumps_root) / f"{a.arm_label}_{name}" / f"seed_{seed}" / f"n{int(msk.sum())}_m{M}"
            dd.mkdir(parents=True, exist_ok=True)
            np.savez(dd / "posterior_samples.npz", samples=samp[seed][msk], theta=th[msk])
    print(f"[{a.arm_label}] dumps -> {a.dumps_root}", flush=True)


if __name__ == "__main__":
    main()
