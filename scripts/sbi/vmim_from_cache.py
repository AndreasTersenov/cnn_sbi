#!/usr/bin/env python
"""A1 (PLAN_OVERNIGHT_MENU_2.md): VMIM-compress a (theta, x) cache to a low-dim summary.

Question it serves: is the joint-stat miscalibration a high-dimensional count-feature
artifact? Train an MLP compressor c(x) (VMIM objective: maximize E log q(theta | c(x))
with a ConditionalRealNVP companion — the npe_l1vmim pattern, minus the TFDS/wandb
plumbing), then write a COMPRESSED cache + fiducial npz in the standard flat-local
layout so population_sweep_flatsky.py and tarp_stratified_val.py run on it UNCHANGED
(downstream preproc: none / clip 0 / min-var 1e-12 — preprocessing happens HERE, fitted
on train, mirroring the quoted sweeps: log1p-zscore / clip 5 / min-var 1e-5).

Compressor seed: single (41), like the CNN arms; statistical rigor comes from the 3 NDE
seeds pooled in the downstream sweep. --max-minutes implements the registered time-box
(keeps best-so-far on expiry; exits nonzero only if NO usable checkpoint exists).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--fid-npz", required=True)
    p.add_argument("--out-cache", required=True)
    p.add_argument("--out-fid", required=True)
    p.add_argument("--summary-dim", type=int, default=10)
    p.add_argument("--hidden", default="256,256")
    p.add_argument("--nf-layers", type=int, default=4)
    p.add_argument("--nf-hidden", type=int, default=128)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--val-subsample", type=int, default=20000)
    p.add_argument("--max-minutes", type=float, default=110.0)
    p.add_argument("--preproc-transform", default="log1p-zscore")
    p.add_argument("--clip-value", type=float, default=5.0)
    p.add_argument("--min-feature-variance", type=float, default=1e-5)
    p.add_argument("--cuda-visible-devices", default="1")
    return p.parse_args()


def main():
    a = parse_args()
    from train_jaxili_from_compressed import setup_env
    setup_env(a.cuda_visible_devices)
    import jax
    import jax.numpy as jnp
    import haiku as hk
    import optax
    from functools import partial
    import tensorflow_probability.substrates.jax as _tfpj  # noqa: F401 — materializes the
    # tfp lazy loader BEFORE sbi_lens touches tfp.substrates (AttributeError otherwise)
    from sbi_lens.normflow.models import AffineCoupling, ConditionalRealNVP
    from npe_l1norm_cross_jaxili_nbody_tomo import (
        preprocess_summaries, filter_zero_variance_bins)

    t_start = time.time()
    deadline = t_start + a.max_minutes * 60.0

    cdir = Path(a.cache_dir)
    tr = np.load(cdir / "l1_train.npz")
    va = np.load(cdir / "l1_val.npz")
    fz = np.load(a.fid_npz)
    theta_tr = tr["theta"].astype(np.float32)
    theta_va = va["theta"].astype(np.float32)
    x_tr_raw = tr["x"].astype(np.float64)
    x_va_raw = va["x"].astype(np.float64)
    S_raw = fz["S"].astype(np.float64)
    print(f"[vmim] train {x_tr_raw.shape} val {x_va_raw.shape} fid {S_raw.shape}", flush=True)

    # ---- preprocessing exactly as the quoted sweeps (fit on train) ----
    clip = a.clip_value if a.clip_value > 0 else None
    tr_p, va_p, S_p, mean, std = preprocess_summaries(
        x_tr_raw, x_va_raw, S_raw, summary_transform=a.preproc_transform, clip_value=clip)
    mask, _ = filter_zero_variance_bins(tr_p, min_variance=a.min_feature_variance,
                                        verbose=False)
    x_tr = tr_p[:, mask].astype(np.float32)
    x_va = va_p[:, mask].astype(np.float32)
    S_x = S_p[:, mask].astype(np.float32)
    in_dim = x_tr.shape[1]
    print(f"[vmim] preprocessed dim {in_dim} (mask kept {mask.sum()})", flush=True)

    # ---- compressor MLP (npe_l1vmim CompressorMLP, stateless) + RealNVP companion ----
    hidden = tuple(int(h) for h in a.hidden.split(","))

    def mlp_fn(x):
        net = x
        for width in hidden:
            net = hk.Linear(width)(net)
            net = jax.nn.leaky_relu(net)
        return hk.Linear(a.summary_dim)(net)

    mlp = hk.without_apply_rng(hk.transform(mlp_fn))
    bijector_fn = partial(AffineCoupling, layers=[a.nf_hidden, a.nf_hidden],
                          activation=jax.nn.silu)
    nf_factory = partial(ConditionalRealNVP, n_layers=a.nf_layers, bijector_fn=bijector_fn)
    n_cosmo = theta_tr.shape[1]
    nf = hk.without_apply_rng(hk.transform(
        lambda theta, y: nf_factory(n_cosmo)(y).log_prob(theta).squeeze()))

    key = jax.random.PRNGKey(a.seed)
    params = {"c": mlp.init(key, jnp.zeros((1, in_dim), jnp.float32)),
              "nf": nf.init(key, jnp.zeros((1, n_cosmo), jnp.float32),
                            jnp.zeros((1, a.summary_dim), jnp.float32))}
    n_par = sum(x.size for x in jax.tree.leaves(params))
    print(f"[vmim] compressor+NF params: {n_par:,}", flush=True)

    schedule_steps = a.steps - a.steps // 3
    lr_schedule = optax.piecewise_constant_schedule(
        init_value=a.lr,
        boundaries_and_scales={int(schedule_steps * f): 0.7
                               for f in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)})
    optimizer = optax.adam(learning_rate=lr_schedule)
    opt_state = optimizer.init(params)

    def loss_fn(p, theta, x):
        y = mlp.apply(p["c"], x)
        return -jnp.mean(nf.apply(p["nf"], theta, y))

    @jax.jit
    def update(p, o, theta, x):
        loss, grads = jax.value_and_grad(loss_fn)(p, theta, x)
        updates, o = optimizer.update(grads, o)
        return loss, optax.apply_updates(p, updates), o

    @jax.jit
    def eval_loss(p, theta, x):
        return loss_fn(p, theta, x)

    rng = np.random.default_rng(a.seed)
    vsel = rng.choice(x_va.shape[0], size=min(a.val_subsample, x_va.shape[0]),
                      replace=False)
    xv = jnp.asarray(x_va[vsel]); tv = jnp.asarray(theta_va[vsel])
    xt = jnp.asarray(x_tr); tt = jnp.asarray(theta_tr)

    best_val, best_params, best_step = np.inf, None, 0
    nonfinite = 0
    hist = []
    n_train = x_tr.shape[0]
    for step in range(1, a.steps + 1):
        idx = rng.integers(0, n_train, size=a.batch_size)
        loss, params, opt_state = update(params, opt_state, tt[idx], xt[idx])
        lf = float(loss)
        if not np.isfinite(lf):
            nonfinite += 1
            if nonfinite > 20:
                print(f"[vmim] too many non-finite losses ({nonfinite}) — stopping", flush=True)
                break
        if step % a.val_every == 0 or step == a.steps:
            vchunks = [float(eval_loss(params, tv[i:i + 4096], xv[i:i + 4096]))
                       for i in range(0, xv.shape[0], 4096)]
            vl = float(np.mean(vchunks))
            hist.append((step, lf, vl))
            flag = ""
            if np.isfinite(vl) and vl < best_val:
                best_val, best_params, best_step = vl, jax.tree.map(np.asarray, params), step
                flag = " *best"
            print(f"[vmim] step {step}/{a.steps} train {lf:.4f} val {vl:.4f}"
                  f" ({time.time()-t_start:.0f}s){flag}", flush=True)
            if time.time() > deadline:
                print(f"[vmim] TIME-BOX reached at step {step} — keeping best-so-far", flush=True)
                break

    if best_params is None:
        print("[vmim] NO usable checkpoint (no finite val eval) — FAIL", flush=True)
        sys.exit(3)
    print(f"[vmim] best val {best_val:.4f} @ step {best_step}", flush=True)

    @jax.jit
    def compress(x):
        return mlp.apply(best_params["c"], x)

    def compress_np(x, bs=8192):
        return np.concatenate([np.asarray(compress(jnp.asarray(x[i:i + bs])))
                               for i in range(0, x.shape[0], bs)]).astype(np.float32)

    y_tr, y_va, y_S = compress_np(x_tr), compress_np(x_va), compress_np(S_x)
    os.makedirs(a.out_cache, exist_ok=True)
    np.savez(f"{a.out_cache}/l1_train.npz", theta=tr["theta"], x=y_tr)
    np.savez(f"{a.out_cache}/l1_val.npz", theta=va["theta"], x=y_va)
    out = {"S": y_S, "perm": fz["perm"], "patch": fz["patch"]}
    for k in ("truth", "theta"):
        if k in fz.files:
            out[k] = fz[k]
    os.makedirs(os.path.dirname(a.out_fid), exist_ok=True)
    np.savez(a.out_fid, **out)
    np.savez(f"{a.out_cache}/l1_cache_meta.npz",
             parent=np.array(a.cache_dir), summary_dim=a.summary_dim,
             hidden=np.array(hidden), nf_layers=a.nf_layers, nf_hidden=a.nf_hidden,
             steps_run=hist[-1][0] if hist else 0, best_step=best_step,
             best_val=best_val, seed=a.seed,
             preproc=np.array(f"{a.preproc_transform}/clip{a.clip_value}/"
                              f"minvar{a.min_feature_variance} APPLIED HERE — downstream "
                              "must use none/0/1e-12"),
             note="VMIM-compressed cache; PLAN_OVERNIGHT_MENU_2.md lane A1")
    with open(f"{a.out_cache}/vmim_history.json", "w") as fh:
        json.dump({"hist": hist, "best_val": best_val, "best_step": best_step,
                   "wall_s": time.time() - t_start}, fh, indent=2)
    print(f"[vmim] wrote compressed cache {y_tr.shape}/{y_va.shape} + fid {y_S.shape}")
    print("BUILD OK")


if __name__ == "__main__":
    main()
