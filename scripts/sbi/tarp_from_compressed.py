#!/usr/bin/env python3
"""Dump posterior samples for TARP from a compressed-cache NDE arm.

Produces the dump format the existing `run_tarp_coverage.py` consumes, so we
reuse the repo's established TARP pipeline (3-D + 6-D curves, seed bands,
overlays) instead of reinventing it:

    <dumps-root>/<arm-label>/seed_<seed>/n<N>_m<M>/posterior_samples.npz
        samples : (N, M, 6)   # N test sims, M posterior samples each
        theta   : (N, 6)      # the true theta for each test sim

It trains the SAME jaxili NDE as `train_jaxili_from_compressed.py` (imports its
helpers; separate file so the running MAF waiter is untouched), then samples the
posterior at N held-out test points — the test ensemble is the on-disk
`cnn_val.npz` (θ,x), no re-simulation.

Then:  python run_tarp_coverage.py --dumps-root <dumps-root> --outdir <dumps-root>
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--compressed-dir", required=True)
    p.add_argument("--arm-label", required=True)
    p.add_argument("--dumps-root", required=True,
                   help="Root dir; dump lands at <root>/<arm>/seed_<seed>/n<N>_m<M>/posterior_samples.npz")
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--n-sims", type=int, default=200)
    p.add_argument("--m-samples", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--decay-steps", type=int, default=10000)
    p.add_argument("--standardize-summary", action="store_true")
    p.add_argument("--min-feature-variance", type=float, default=1e-12)
    p.add_argument("--cuda-visible-devices", default="1")
    return p.parse_args()


def main():
    args = parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train_jaxili_from_compressed import setup_env
    setup_env(args.cuda_visible_devices)
    import jax, jax.numpy as jnp
    from npe_l1norm_cross_jaxili_nbody_tomo import train_with_nan_retry
    from jaxili.inference import NPE

    cdir = Path(args.compressed_dir).resolve()

    def _load(which):  # cache-agnostic: CNN (cnn_*.npz) or L1 (l1_*.npz)
        for pref in ("cnn", "l1"):
            f = cdir / f"{pref}_{which}.npz"
            if f.exists():
                z = np.load(f)
                return z["theta"].astype(np.float32), z["x"].astype(np.float32)
        raise FileNotFoundError(f"no cnn/l1 _{which}.npz in {cdir}")

    theta_tr, x_tr_raw = _load("train")
    theta_val, x_val_raw = _load("val")
    # mask + optional z-score, fit on TRAIN (identical to standardize_and_mask,
    # but applied column-wise to the 2-D val ensemble).
    var = x_tr_raw.var(axis=0)
    mask = var > args.min_feature_variance
    x_tr = x_tr_raw[:, mask]; x_val = x_val_raw[:, mask]
    if args.standardize_summary:
        mu, sd = x_tr.mean(0), x_tr.std(0); sd = np.where(sd > 0, sd, 1.0)
        x_tr = (x_tr - mu) / sd; x_val = (x_val - mu) / sd
    x_tr = x_tr.astype(np.float32); x_val = x_val.astype(np.float32)
    print(f"  train theta{theta_tr.shape} x{x_tr.shape}; val ensemble x{x_val.shape} "
          f"(masked {int(mask.sum())}/{mask.size} features)", flush=True)

    N = int(min(args.n_sims, x_val.shape[0]))
    M = int(args.m_samples)
    dump_dir = (Path(args.dumps_root).resolve() / args.arm_label
                / f"seed_{args.seed}" / f"n{N}_m{M}")
    dump_dir.mkdir(parents=True, exist_ok=True)

    # --- train the NDE (identical recipe to train_jaxili_from_compressed) ---
    t0 = time.time()
    split_key = jax.random.PRNGKey(int(args.seed) + 1)
    params = jnp.asarray(theta_tr); data = jnp.asarray(x_tr)
    inference = NPE().append_simulations(params, data, key=split_key)
    ckpt = str(dump_dir / "ckpt")   # absolute (dumps-root resolved) -> orbax-safe
    inference, _m, _de = train_with_nan_retry(
        inference, ckpt, args.epochs, args.learning_rate, args.batch_size,
        args.warmup_steps, args.decay_steps, params, data, split_key)
    posterior = inference.build_posterior()
    print(f"  NDE trained in {time.time()-t0:.0f}s", flush=True)

    # --- sample posterior at N test points -> samples (N, M, 6) ---
    rng = np.random.default_rng(int(args.seed))
    idx = rng.choice(x_val.shape[0], size=N, replace=False)
    x_te = x_val[idx]; th_te = theta_val[idx].astype(np.float32)
    D = th_te.shape[1]
    t1 = time.time()
    samp = np.empty((N, M, D), dtype=np.float32)
    for i in range(N):
        k = jax.random.PRNGKey(int(args.seed) * 100003 + i)
        s = np.asarray(posterior.sample(
            x=jnp.asarray(x_te[i]), num_samples=M, key=k))
        samp[i] = s
        if (i + 1) % 50 == 0:
            print(f"  sampled {i+1}/{N} ({(time.time()-t1)/(i+1):.2f}s/pt)", flush=True)

    out = dump_dir / "posterior_samples.npz"
    np.savez(out, samples=samp, theta=th_te)
    print(f"[dump] wrote {out}  samples{samp.shape} theta{th_te.shape} "
          f"(sampling {time.time()-t1:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
