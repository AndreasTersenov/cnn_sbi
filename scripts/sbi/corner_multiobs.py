#!/usr/bin/env python3
"""Sample the corner_resample MAFs (3 seeds/arm) at SEVERAL representative patches.

CNN: reloads the already-trained jaxili checkpoints (bit-exact, ~1 min).
L1 : retrains the 3 seeds (cheap, ~4 min total) — its checkpoint is NOT reloadable
     because jaxili stores the 2000-dim Standardizer mean/std as a *truncated* numpy
     repr in hparams.json (`[a, b, c, ...]`), so the regex reload recovers only 3
     values (shape (3,)). Retraining is deterministic and validated below.

Validation gate per arm: reproduce patch 76 / perm 1 pooled FoM3 vs the value saved
by corner_resample (L1 9755, CNN 17192) within --tol; abort on mismatch.

Writes <output>/multiobs_samples_<arm>.npz: pooled_<patch>_<perm> (n,6) per obs.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
DC = REPO / "results/exploratory/definitive_comparison_10deg/phase_c"
RS = DC / "analysis/corner_resample"

ARMS = {
    "l1_auto_cross": dict(
        cache=DC / "l1_auto_cross_cache", prefix="l1",
        summ=DC / "fiducial_summaries/l1_auto_cross.npz",
        transform="log1p-zscore", clip=5.0, min_var=1e-5, expect=9755.0, reload=False),
    "cnn_auto_cross": dict(
        cache=DC / "cnn_auto_cross_s41/cache", prefix="cnn",
        summ=DC / "fiducial_summaries/cnn_auto_cross.npz",
        transform="none", clip=0.0, min_var=1e-12, expect=17192.0, reload=True),
}


def parse_obs(spec):
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            p, m = tok.split(":")
            out.append((int(p), int(m)))
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--obs-list", default="66:1,123:1,164:1,35:1,0:1")
    p.add_argument("--seeds", default="41,42,43")
    p.add_argument("--samples-per-seed", type=int, default=30000)
    p.add_argument("--tol", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--decay-steps", type=int, default=10000)
    p.add_argument("--cuda-visible-devices", default="1")
    p.add_argument("--output-dir", default=str(RS))
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
        _resolve_latest_jaxili_checkpoint_dir, _normalize_jaxili_hparams_embedding_arrays,
    )

    obs_list = parse_obs(a.obs_list)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    out = Path(a.output_dir)
    print(f"obs (patch:perm): {obs_list}; seeds {seeds}; {a.samples_per_seed}/seed", flush=True)

    for arm, cfg in ARMS.items():
        tr = np.load(cfg["cache"] / f"{cfg['prefix']}_train.npz")
        theta_tr = tr["theta"].astype(np.float32)
        x_tr_raw = tr["x"].astype(np.float64)
        z = np.load(cfg["summ"])
        S_raw = z["S"].astype(np.float64)
        perm = z["perm"].astype(int); patch = z["patch"].astype(int)
        pos = {(int(perm[k]), int(patch[k])): k for k in range(len(perm))}

        clip = cfg["clip"] if cfg["clip"] > 0 else None
        tr_proc, _, S_proc, _, _ = preprocess_summaries(
            x_tr_raw, x_tr_raw[:1], S_raw, summary_transform=cfg["transform"], clip_value=clip)
        mask, _ = filter_zero_variance_bins(tr_proc, min_variance=cfg["min_var"], verbose=False)
        x_tr = tr_proc[:, mask].astype(np.float32)
        S = S_proc[:, mask].astype(np.float32)
        D = int(mask.sum())
        print(f"\n[{arm}] masked dim {D}; mode={'reload' if cfg['reload'] else 'retrain'}", flush=True)

        posteriors = []
        if cfg["reload"]:
            for seed in seeds:
                ck = (RS / arm / f"ckpt_{arm}_s{seed}").resolve()
                vdir = _resolve_latest_jaxili_checkpoint_dir(ck)
                _normalize_jaxili_hparams_embedding_arrays(vdir)
                exmp = (jnp.zeros((1, 6), jnp.float32), jnp.zeros((1, D), jnp.float32))
                inf = NPE.load_from_checkpoints(checkpoint=str(vdir), exmp_input=exmp)
                posteriors.append((seed, inf.build_posterior()))
                print(f"  reloaded seed {seed}", flush=True)
        else:
            params = jnp.asarray(theta_tr); data = jnp.asarray(x_tr)
            for seed in seeds:
                sk = jax.random.PRNGKey(int(seed) + 1)
                inf = NPE().append_simulations(params, data, key=sk)
                ckpt = str((out / arm / f"ckpt_retrain_{arm}_s{seed}").resolve())
                inf, _m, _d = train_with_nan_retry(inf, ckpt, a.epochs, a.learning_rate,
                                                   a.batch_size, a.warmup_steps, a.decay_steps,
                                                   params, data, sk)
                posteriors.append((seed, inf.build_posterior()))
                print(f"  retrained seed {seed}", flush=True)

        def sample_pooled(x_obs):
            o = []
            for seed, post in posteriors:
                k = jax.random.PRNGKey(int(seed) + 7)
                o.append(np.asarray(post.sample(x=jnp.asarray(x_obs), num_samples=a.samples_per_seed, key=k)))
            s = np.concatenate(o, 0)
            return s[np.all(np.isfinite(s), 1)]

        v = sample_pooled(S[pos[(1, 76)]])
        fv = compute_fom3(v)["fom3"]
        rel = abs(fv - cfg["expect"]) / cfg["expect"]
        ok = rel <= a.tol
        print(f"  [validate] (76,1) pooled FoM3={fv:.0f} vs saved {cfg['expect']:.0f} "
              f"(rel {rel*100:.1f}%) -> {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            raise SystemExit(f"[{arm}] validation FAILED; aborting.")

        saved, fom = {}, {}
        for (pp, mm) in obs_list:
            key = (mm, pp)
            if key not in pos:
                print(f"  WARN obs (patch {pp}, perm {mm}) missing; skip"); continue
            s = sample_pooled(S[pos[key]])
            saved[f"pooled_{pp}_{mm}"] = s
            fom[f"{pp}_{mm}"] = compute_fom3(s)["fom3"]
            print(f"  obs patch {pp} perm {mm}: {s.shape[0]} kept, FoM3={fom[f'{pp}_{mm}']:.0f}", flush=True)

        np.savez(out / f"multiobs_samples_{arm}.npz",
                 truth=np.asarray(FIDUCIAL, float), seeds=np.array(seeds),
                 obs=np.array(obs_list), **saved)
        (out / f"multiobs_fom_{arm}.json").write_text(json.dumps(fom, indent=2))
        print(f"[{arm}] DONE -> {out/('multiobs_samples_'+arm+'.npz')}", flush=True)


if __name__ == "__main__":
    main()
