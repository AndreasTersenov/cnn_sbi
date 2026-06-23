#!/usr/bin/env python3
"""Unified NDE-family sweep on FROZEN CNN-VMIM summaries (Phase 1, CNN-optimization 2026-06).

Holds the compressor's summaries fixed and varies ONLY the density estimator, so any FoM3
difference is the NDE. Reuses population_sweep_flatsky.py's metric machinery VERBATIM
(preprocess_summaries / filter_zero_variance_bins / compute_fom3 / fom2d / marginal_stats) and
the same 9000-obs pooled-median protocol, so every family is apples-to-apples with the existing
L1 and CNN baselines.

Families (--nde-family):
  sbilens_realnvp : the PRODUCTION CNN NDE (sbi_lens ConditionalRealNVP via build_flow/train_flow
                    in npe_cnn_nbody_tomo.py). Capacity = --nde-layers / --nde-hidden.
  jaxili_maf      : jaxili ConditionalMAF (the common-metric NDE; reproduces the B1 baseline).
  jaxili_realnvp  : jaxili ConditionalRealNVP (framework control vs sbilens_realnvp).
  jaxili_mdn      : jaxili MixtureDensityNetwork (a different family).

Per-family defaults match the canonical config; override with --nde-layers / --nde-hidden.
Metric = median FoM3 over N fiducial obs (default 9000; screen with --n-obs 1000). 2 seeds to
screen, 3 for finalists. See PLAN_CNN_NDE_SWEEP_2026-06-13.md.

NOTE on jaxili NaN-retry: the shared train_with_nan_retry() rebuilds NPE() with DEFAULTS on a NaN
(npe_l1norm_cross_jaxili_nbody_tomo.py:2430), which would silently revert a non-MAF family to MAF.
This script uses a FAMILY-PRESERVING retry instead.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = "/mnt/home/tersenov/software/cnn_sbi"
SBI = f"{REPO}/scripts/sbi"

# Per-family (n_layers/transforms, hidden-width) defaults — used when --nde-layers/-hidden < 0.
FAMILY_DEFAULTS = {
    "sbilens_realnvp": (4, 128),   # production CNN flow (npe_cnn --nvp-layers/--nvp-hidden)
    "sbilens_nsf":     (4, 128),   # A2: rational-quadratic spline couplings (more expressive family)
    "jaxili_maf":      (5, 50),    # jaxili default (the B1 common-metric baseline)
    "jaxili_realnvp":  (5, 50),    # MAF-equivalent capacity (framework control)
    "jaxili_mdn":      (10, 50),   # n_components=10, hidden=50
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-cache-dir", required=True)
    p.add_argument("--cache-prefix", default="cnn")
    p.add_argument("--arm-label", required=True)
    p.add_argument("--fiducial-summaries-npz", required=True)   # per-arm sliced (key 'S')
    p.add_argument("--output-dir", required=True)
    p.add_argument("--nde-family", required=True, choices=list(FAMILY_DEFAULTS))
    p.add_argument("--nde-layers", type=int, default=-1, help="<0 => family default")
    p.add_argument("--nde-hidden", type=int, default=-1, help="<0 => family default")
    # preprocessing — MUST match the existing CNN sweep (none / clip 0 / min-var 1e-12)
    p.add_argument("--preproc-transform", default="none")
    p.add_argument("--clip-value", type=float, default=0.0)
    p.add_argument("--min-feature-variance", type=float, default=1e-12)
    # population
    p.add_argument("--seeds", default="41,42")
    p.add_argument("--n-obs", type=int, default=9000)
    p.add_argument("--max-perm", type=int, default=50)
    p.add_argument("--m-samples", type=int, default=2000)
    # jaxili training knobs (match population_sweep_flatsky / train_jaxili_from_compressed)
    p.add_argument("--epochs", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--decay-steps", type=int, default=10000)
    # sbi_lens flow knobs (match npe_cnn_nbody_tomo defaults)
    p.add_argument("--flow-total-steps", type=int, default=50000)
    p.add_argument("--flow-batch-size", type=int, default=128)
    p.add_argument("--flow-lr-init", type=float, default=1e-3)
    p.add_argument("--flow-lr-end", type=float, default=1e-5)
    p.add_argument("--flow-save-every", type=int, default=2000)
    p.add_argument("--flow-patience", type=int, default=20)
    p.add_argument("--flow-grad-clip", type=float, default=1.0)
    p.add_argument("--flow-weight-decay", type=float, default=1e-4)
    p.add_argument("--cuda-visible-devices", default="2")
    return p.parse_args()


def resolve_capacity(family, layers, hidden):
    dl, dh = FAMILY_DEFAULTS[family]
    return (dl if layers < 0 else layers), (dh if hidden < 0 else hidden)


# ----------------------------------------------------------------------------------
# Family-specific training: each returns a list of (seed, sampler_fn) where
# sampler_fn(x_single[dim], key) -> samples[M, 6]. Sampler is jitted (~1 ms/obs).
# ----------------------------------------------------------------------------------
def train_jaxili_family(family, layers, hidden, theta_tr, x_tr, seeds, M, out_dir, args):
    """jaxili NPE families with FAMILY-PRESERVING NaN retry."""
    import inspect
    import jax
    import jax.numpy as jnp
    from jaxili.inference import NPE
    from jaxili.model import ConditionalMAF, ConditionalRealNVP, MixtureDensityNetwork

    # Reuse jaxili's OWN default activation object: jax.nn.silu resolves to a PjitFunction
    # without __name__ in the full import stack, which crashes create_trainer's
    # `activation.__name__` (npe.py:455). The default works, so reuse its exact object.
    _ACT = inspect.signature(NPE.__init__).parameters["model_hparams"].default["activation"]

    if family == "jaxili_maf":
        model_class = ConditionalMAF
        hparams = {"n_layers": layers, "layers": [hidden, hidden],
                   "activation": _ACT, "use_reverse": True, "seed": 42}
    elif family == "jaxili_realnvp":
        model_class = ConditionalRealNVP
        hparams = {"n_layers": layers, "layers": [hidden, hidden], "activation": _ACT}
    elif family == "jaxili_mdn":
        model_class = MixtureDensityNetwork
        hparams = {"n_components": layers, "layers": [hidden, hidden], "activation": _ACT}
    else:
        raise ValueError(family)

    params = jnp.asarray(theta_tr)
    data = jnp.asarray(x_tr)
    samplers = []
    for seed in seeds:
        t0 = time.time()
        split_key = jax.random.PRNGKey(seed + 1)

        def make_inf():
            inf = NPE(model_class=model_class, model_hparams=hparams)
            return inf.append_simulations(params, data, key=split_key)

        ckpt = str((out_dir / "ckpts" / f"{family}_s{seed}").resolve())
        inf = make_inf()
        for attempt in range(1, 11):                    # family-preserving retry
            print(f"######## {family} s{seed} TRAIN ATTEMPT {attempt}/10 ########", flush=True)
            metrics, _de = inf.train(checkpoint_path=ckpt, num_epochs=args.epochs,
                                     learning_rate=args.learning_rate,
                                     training_batch_size=args.batch_size,
                                     warmup=args.warmup_steps, decay_steps=args.decay_steps)
            bad = any(isinstance(metrics, dict) and k in metrics
                      and not np.all(np.isfinite(np.asarray(metrics[k])))
                      for k in ("train/loss", "val/loss"))
            if bad:
                print(f"  NaN in {family} s{seed}; re-init SAME family.", flush=True)
                inf = make_inf()
                continue
            break
        post = inf.build_posterior()
        _p, _m = post.state.params, post.model
        sampler = jax.jit(lambda x, k, _m=_m, _p=_p: _m.apply({"params": _p}, x, M, k,
                                                              method="sample"))
        samplers.append((seed, sampler))
        print(f"  NDE seed {seed} {time.time()-t0:.0f}s", flush=True)
    return samplers


def train_sbilens_realnvp(layers, hidden, theta_tr, x_tr, seeds, M, dim, out_dir, args):
    """Production sbi_lens ConditionalRealNVP via npe_cnn build_flow / train_flow (wandb off)."""
    import jax
    import jax.numpy as jnp
    import wandb
    wandb.init(mode="disabled", project="nde-sweep", reinit=True)
    sys.path.insert(0, SBI)
    from npe_cnn_nbody_tomo import build_flow, train_flow

    # deterministic 90/10 train/val split for the flow's early stopping
    n = len(theta_tr)
    pidx = np.random.RandomState(0).permutation(n)
    nval = max(1, n // 10)
    val_idx, tr_idx = pidx[:nval], pidx[nval:]
    dtr = {"theta": theta_tr[tr_idx].astype(np.float32), "x": x_tr[tr_idx].astype(np.float32)}
    dva = {"theta": theta_tr[val_idx].astype(np.float32), "x": x_tr[val_idx].astype(np.float32)}

    samplers = []
    for seed in seeds:
        t0 = time.time()
        rng = jax.random.PRNGKey(seed)
        nf_logp, nf_sample = build_flow(n_cosmo_params=6, n_layers=layers, hidden=hidden)
        best_params = train_flow(
            rng, nf_logp, dtr, dva, n_cosmo=6, summary_dim=dim,
            total_steps=args.flow_total_steps, batch_size=args.flow_batch_size,
            save_every=args.flow_save_every, save_dir=out_dir / "ckpts" / f"sbilens_s{seed}",
            lr_init=args.flow_lr_init, end_lr=args.flow_lr_end,
            grad_clip=args.flow_grad_clip, weight_decay=args.flow_weight_decay,
            patience=args.flow_patience, lr_schedule_fn=None)

        def sampler(x, k, _p=best_params, _ns=nf_sample):
            y = jnp.broadcast_to(jnp.asarray(x).reshape(1, dim), (M, dim))
            return _ns.apply(_p, k, y, M)
        samplers.append((seed, jax.jit(sampler)))
        print(f"  NDE seed {seed} {time.time()-t0:.0f}s", flush=True)
    return samplers


def train_sbilens_nsf(layers, hidden, theta_tr, x_tr, seeds, M, dim, out_dir, args):
    """A2: Neural Spline Flow (RQS couplings) in the sbi_lens RealNVP chain. theta is standardized to
    the base scale (mean 0.5, std 0.05) so the spline models shape/correlations near [-B,B]; samples
    are un-standardized. Same train_flow / pooling / metric machinery as train_sbilens_realnvp."""
    import jax
    import jax.numpy as jnp
    import wandb
    wandb.init(mode="disabled", project="nde-sweep", reinit=True)
    sys.path.insert(0, SBI)
    from npe_cnn_nbody_tomo import build_spline_flow, train_flow

    mu = theta_tr.mean(0).astype(np.float32)
    sd = (theta_tr.std(0) + 1e-8).astype(np.float32)
    stdize = lambda t: ((t - mu) / sd).astype(np.float32)            # z-score -> N(0,1) base

    n = len(theta_tr)
    pidx = np.random.RandomState(0).permutation(n)
    nval = max(1, n // 10)
    val_idx, tr_idx = pidx[:nval], pidx[nval:]
    dtr = {"theta": stdize(theta_tr[tr_idx]), "x": x_tr[tr_idx].astype(np.float32)}
    dva = {"theta": stdize(theta_tr[val_idx]), "x": x_tr[val_idx].astype(np.float32)}

    samplers = []
    for seed in seeds:
        t0 = time.time()
        rng = jax.random.PRNGKey(seed)
        nf_logp, nf_sample = build_spline_flow(n_cosmo_params=6, n_layers=layers, hidden=hidden)
        best_params = train_flow(
            rng, nf_logp, dtr, dva, n_cosmo=6, summary_dim=dim,
            total_steps=args.flow_total_steps, batch_size=args.flow_batch_size,
            save_every=args.flow_save_every, save_dir=out_dir / "ckpts" / f"nsf_s{seed}",
            lr_init=args.flow_lr_init, end_lr=args.flow_lr_end,
            grad_clip=args.flow_grad_clip, weight_decay=args.flow_weight_decay,
            patience=args.flow_patience, lr_schedule_fn=None)

        mu_j, sd_j = jnp.asarray(mu), jnp.asarray(sd)

        def sampler(x, k, _p=best_params, _ns=nf_sample, _mu=mu_j, _sd=sd_j):
            y = jnp.broadcast_to(jnp.asarray(x).reshape(1, dim), (M, dim))
            raw = _ns.apply(_p, k, y, M)
            return _mu + _sd * raw                           # un-standardize (z-score) to raw theta
        samplers.append((seed, jax.jit(sampler)))
        print(f"  NDE seed {seed} {time.time()-t0:.0f}s", flush=True)
    return samplers


def main():
    a = parse_args()
    sys.path.insert(0, str(REPO)); sys.path.insert(0, SBI)
    from train_jaxili_from_compressed import setup_env, compute_fom3, fom2d, marginal_stats
    setup_env(a.cuda_visible_devices)
    import jax
    import jax.numpy as jnp
    from npe_l1norm_cross_jaxili_nbody_tomo import preprocess_summaries, filter_zero_variance_bins

    layers, hidden = resolve_capacity(a.nde_family, a.nde_layers, a.nde_hidden)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"[{a.arm_label}] family={a.nde_family} layers={layers} hidden={hidden}", flush=True)

    # ---- training data + preprocessing (identical to population_sweep_flatsky.py) ----
    cdir = Path(a.train_cache_dir)
    tr = np.load(cdir / f"{a.cache_prefix}_train.npz")
    va = np.load(cdir / f"{a.cache_prefix}_val.npz")
    theta_tr = tr["theta"].astype(np.float32)
    x_tr_raw = tr["x"].astype(np.float64); x_va_raw = va["x"].astype(np.float64)
    clip = a.clip_value if a.clip_value > 0 else None
    tr_p, _, _, mean, std = preprocess_summaries(
        x_tr_raw, x_va_raw[:1], x_va_raw[:1], summary_transform=a.preproc_transform, clip_value=clip)
    mask, _ = filter_zero_variance_bins(tr_p, min_variance=a.min_feature_variance, verbose=False)
    x_tr = tr_p[:, mask].astype(np.float32)
    dim = x_tr.shape[1]
    print(f"[{a.arm_label}] train{x_tr_raw.shape} -> dim {dim}", flush=True)

    # ---- fiducial obs (preprocessed with TRAIN mean/std + mask) ----
    fz = np.load(a.fiducial_summaries_npz)
    S = fz["S"].astype(np.float64); perm = fz["perm"]
    sel = np.where(perm < a.max_perm)[0][:a.n_obs]
    assert sel.size > 0 and int(sel.max()) < 100003, \
        f"obs index {int(sel.max())} >= 100003 collides PRNG keys across seeds"
    _, _, fid_p, _, _ = preprocess_summaries(
        x_tr_raw, x_va_raw[:1], S[sel], summary_transform=a.preproc_transform,
        clip_value=clip, mean=mean, std=std)
    x_obs = fid_p[:, mask].astype(np.float32)
    truth = next((fz[k] for k in ("truth", "theta") if k in fz.files), None)
    N, M = x_obs.shape[0], a.m_samples
    print(f"[{a.arm_label}] {N} fiducial obs (perm<{a.max_perm})", flush=True)

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    if a.nde_family == "sbilens_realnvp":
        samplers = train_sbilens_realnvp(layers, hidden, theta_tr, x_tr, seeds, M, dim, out, a)
    elif a.nde_family == "sbilens_nsf":
        samplers = train_sbilens_nsf(layers, hidden, theta_tr, x_tr, seeds, M, dim, out, a)
    else:
        samplers = train_jaxili_family(a.nde_family, layers, hidden, theta_tr, x_tr, seeds, M, out, a)

    # ---- per-patch metrics (seed-pooled posterior per obs); identical to population_sweep ----
    x_dev = jnp.asarray(x_obs)
    sig = np.full((N, 3), np.nan); pair = np.full((N, 3), np.nan); fom3 = np.full(N, np.nan)
    n_finite = np.zeros(N, dtype=np.int32)
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
        pair[i] = [f2["fom2d_Omega_m_sigma_8"], f2["fom2d_Omega_m_w_0"], f2["fom2d_sigma_8_w_0"]]
        fom3[i] = compute_fom3(ps)["fom3"]
        if (i + 1) % 500 == 0:
            print(f"  sampled {i+1}/{N} ({time.time()-t0:.0f}s)", flush=True)
    n_skipped = int((n_finite < 100).sum())
    if n_skipped:
        print(f"  [warn] {n_skipped} obs skipped (<100 finite samples)", flush=True)

    np.savez(out / "per_patch_metrics.npz", sigma=sig, fom2d=pair, fom3=fom3,
             perm=perm[sel], patch=fz["patch"][sel], sel=sel, n_finite=n_finite,
             truth=(truth if truth is not None else np.array([])))
    g = np.isfinite(fom3)
    med = dict(arm=a.arm_label, nde_family=a.nde_family, layers=int(layers), hidden=int(hidden),
               n=int(g.sum()),
               sigma_Om=float(np.median(sig[g, 0])), sigma_s8=float(np.median(sig[g, 1])),
               sigma_w0=float(np.median(sig[g, 2])),
               fom2d_Om_s8=float(np.median(pair[g, 0])), fom2d_Om_w0=float(np.median(pair[g, 1])),
               fom2d_s8_w0=float(np.median(pair[g, 2])), fom3=float(np.median(fom3[g])))
    json.dump(med, open(out / "median_summary.json", "w"), indent=2)
    print(f"[{a.arm_label}] MEDIAN over {med['n']} obs: "
          f"sig(Om,s8,w0)={med['sigma_Om']:.3f},{med['sigma_s8']:.3f},{med['sigma_w0']:.3f}  "
          f"FoM3={med['fom3']:.0f}", flush=True)


if __name__ == "__main__":
    main()
