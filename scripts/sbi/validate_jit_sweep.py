#!/usr/bin/env python3
"""Adoption gate for the jitted sampling path: re-derive one arm's FULL 9000-obs
pooled median with jit-sampling from the arm's saved NDE checkpoints and compare to
the production (eager-loop) median_summary.json. Bit-identity already failed at the
TF32 level (bench_sample_jit.py: max|Δ|=3.4e-3, FoM3 dev ≤0.67%/obs); this checks the
HEADLINE metric is reproduced within noise before wiring jit into the sweep.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

REPO = "/mnt/home/tersenov/software/cnn_sbi"
SBI = f"{REPO}/scripts/sbi"
CNNP = Path(f"{SBI}/results/exploratory/flatsky_cross_2026_06/cnn_phase")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="none_s42", help="multiseed arm: {none,product}_s{42,43}")
    ap.add_argument("--m-samples", type=int, default=2000)
    ap.add_argument("--cuda-visible-devices", default="2")
    a = ap.parse_args()
    op, _, cseed = a.arm.partition("_s")
    sys.path.insert(0, REPO); sys.path.insert(0, SBI)
    from train_jaxili_from_compressed import setup_env, compute_fom3, fom2d, marginal_stats
    setup_env(a.cuda_visible_devices)
    import jax, jax.numpy as jnp
    from jaxili.inference import NPE
    from npe_l1norm_cross_jaxili_nbody_tomo import (
        preprocess_summaries, filter_zero_variance_bins,
        _resolve_latest_jaxili_checkpoint_dir, _normalize_jaxili_hparams_embedding_arrays)

    arm_dir = CNNP / "multiseed" / "population_sweep" / a.arm
    cache = CNNP / f"cnn_{op}_s{cseed}" / "cache"
    fid = CNNP / "multiseed" / "fiducial_summaries" / f"fiducial_summaries_{op}_s{cseed}.npz"
    ref = json.load(open(arm_dir / "median_summary.json"))

    tr = np.load(cache / "cnn_train.npz"); va = np.load(cache / "cnn_val.npz")
    x_tr_raw = tr["x"].astype(np.float64); x_va_raw = va["x"].astype(np.float64)
    tr_p, _, _, mean, std = preprocess_summaries(
        x_tr_raw, x_va_raw[:1], x_va_raw[:1], summary_transform="none", clip_value=None)
    mask, _ = filter_zero_variance_bins(tr_p, min_variance=1e-12, verbose=False)
    fz = np.load(fid)
    sel = np.where(fz["perm"] < 50)[0][:9000]
    _, _, fid_p, _, _ = preprocess_summaries(
        x_tr_raw, x_va_raw[:1], fz["S"].astype(np.float64)[sel],
        summary_transform="none", clip_value=None, mean=mean, std=std)
    x_obs = fid_p[:, mask].astype(np.float32)
    N, M, xdim = x_obs.shape[0], a.m_samples, int(mask.sum())
    print(f"[{a.arm}] {N} obs, dim {xdim}; reference FoM3 {ref['fom3']:.2f}", flush=True)

    fns = []
    for seed in (41, 42, 43):
        vdir = _resolve_latest_jaxili_checkpoint_dir(arm_dir / "ckpts" / f"s{seed}")
        _normalize_jaxili_hparams_embedding_arrays(vdir)
        exmp = (jnp.zeros((1, 6), jnp.float32), jnp.zeros((1, xdim), jnp.float32))
        inf = NPE.load_from_checkpoints(checkpoint=str(vdir), exmp_input=exmp)
        post = inf.build_posterior()
        params, model = post.state.params, post.model
        fns.append((seed, jax.jit(
            lambda x, k, _m=model, _p=params: _m.apply({"params": _p}, x, M, k, method="sample"))))
        print(f"  seed {seed} reloaded + jitted", flush=True)

    x_dev = jnp.asarray(x_obs)
    sig = np.full((N, 3), np.nan); pair = np.full((N, 3), np.nan); fom3 = np.full(N, np.nan)
    t0 = time.time()
    for i in range(N):
        pooled = [np.asarray(fn(x_dev[i], jax.random.PRNGKey(seed * 100003 + int(sel[i]))))
                  for seed, fn in fns]
        ps = np.concatenate(pooled, 0); ps = ps[np.all(np.isfinite(ps), 1)]
        if ps.shape[0] < 100:
            continue
        ms = marginal_stats(ps); f2 = fom2d(ps)
        sig[i] = [ms["sigma"][k] for k in ("Omega_m", "sigma_8", "w_0")]
        pair[i] = [f2["fom2d_Omega_m_sigma_8"], f2["fom2d_Omega_m_w_0"], f2["fom2d_sigma_8_w_0"]]
        fom3[i] = compute_fom3(ps)["fom3"]
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{N} ({time.time()-t0:.0f}s)", flush=True)
    g = np.isfinite(fom3)
    out = dict(fom3=float(np.median(fom3[g])), sigma_Om=float(np.median(sig[g, 0])),
               sigma_s8=float(np.median(sig[g, 1])), sigma_w0=float(np.median(sig[g, 2])),
               fom2d_Om_s8=float(np.median(pair[g, 0])), n=int(g.sum()),
               wall_s=time.time() - t0)
    print(f"\n[{a.arm}] JIT sweep: {out['wall_s']:.0f}s for {N} obs "
          f"(production eager sampling phase was ~4100-4800s)")
    print(f"{'metric':<14}{'eager (prod)':>14}{'jit':>14}{'rel Δ':>10}")
    for k in ("fom3", "sigma_Om", "sigma_s8", "sigma_w0", "fom2d_Om_s8"):
        r = ref[k]; v = out[k]
        print(f"{k:<14}{r:>14.4f}{v:>14.4f}{(v-r)/r:>+10.2%}")
    json.dump({"arm": a.arm, "reference": ref, "jit": out},
              open(arm_dir / "jit_validation.json", "w"), indent=2)
    print(f"wrote {arm_dir/'jit_validation.json'}", flush=True)


if __name__ == "__main__":
    main()
