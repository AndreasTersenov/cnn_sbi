#!/usr/bin/env python3
"""Full-200 fiducial analysis for ONE arm: step-1 (mean datavector) + step-2
(per-patch FoM distribution). Reuses the campaign's EXACT NDE + preprocessing so
the numbers are comparable to the Phase C posteriors.

Preprocessing per arm (matches how that arm's campaign posteriors were made):
  - CNN non-std : transform=none,  clip=None, min_var=1e-12   (train_jaxili mask-only)
  - CNN std     : transform=zscore,clip=None, min_var=1e-12   (train_jaxili --standardize)
  - L1          : transform=log1p-zscore, clip=5.0, min_var=1e-5 (npe_l1 preprocess_summaries)
All via the campaign functions preprocess_summaries + filter_zero_variance_bins.

G3 end-to-end gate: at (perm0,patch0) the 3-seed-pooled FoM3 must reproduce the
campaign arm's known perm-0 value within --g3-tol; else ABORT (no garbage). This
validates the whole chain (summary extraction + preprocessing + NDE + sampling).

The "mean datavector" = mean over the 9600 RAW per-patch summaries, then pushed
through the SAME preprocessing as an obs (average the l1s/summaries, then infer).
"""
from __future__ import annotations
import argparse, csv, json, sys, time
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
    p.add_argument("--n-step2-patches", type=int, default=300)
    p.add_argument("--step1-samples", type=int, default=100000)
    p.add_argument("--step2-samples", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--decay-steps", type=int, default=10000)
    p.add_argument("--expected-fom3", type=float, default=0.0, help="G3 target (perm0); 0 => skip")
    p.add_argument("--g3-tol", type=float, default=0.20, help="relative tolerance for G3")
    p.add_argument("--cuda-visible-devices", default="1")
    return p.parse_args()


def main():
    a = parse_args()
    sys.path.insert(0, str(REPO))
    from train_jaxili_from_compressed import (
        setup_env, compute_fom3, fom2d, marginal_stats, PARAM_KEYS,
    )
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
    perm = z["perm"]; patch = z["patch"]
    assert S_raw.shape[1] == x_tr_raw.shape[1], \
        f"dim mismatch S={S_raw.shape[1]} vs train={x_tr_raw.shape[1]}"
    mean_dv_raw = S_raw.mean(axis=0, keepdims=True)
    print(f"[{a.arm_label}] train x{x_tr_raw.shape} ; S{S_raw.shape} "
          f"({len(np.unique(perm))} perms); transform={a.preproc_transform} "
          f"clip={a.clip_value} min_var={a.min_feature_variance}", flush=True)

    # ---- preprocessing fit on TRAIN, applied to S and mean_dv (campaign functions) ----
    clip = a.clip_value if a.clip_value > 0 else None
    tr_proc, _, S_proc, mean, std = preprocess_summaries(
        x_tr_raw, x_tr_raw[:1], S_raw, summary_transform=a.preproc_transform, clip_value=clip)
    _, _, mean_dv_proc, _, _ = preprocess_summaries(
        x_tr_raw, x_tr_raw[:1], mean_dv_raw, summary_transform=a.preproc_transform,
        clip_value=clip, mean=mean, std=std)
    mask, n_removed = filter_zero_variance_bins(tr_proc, min_variance=a.min_feature_variance, verbose=False)
    x_tr = tr_proc[:, mask].astype(np.float32)
    S = S_proc[:, mask].astype(np.float32)
    mean_dv = mean_dv_proc[:, mask][0].astype(np.float32)
    print(f"  masked dim {x_tr.shape[1]}/{mask.size} (removed {n_removed})", flush=True)

    # ---- train one NDE per seed ----
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    out_dir = Path(a.output_dir) / a.arm_label
    out_dir.mkdir(parents=True, exist_ok=True)
    posteriors = []
    params = jnp.asarray(theta_tr); data = jnp.asarray(x_tr)
    for seed in seeds:
        t0 = time.time()
        sk = jax.random.PRNGKey(int(seed) + 1)
        inf = NPE().append_simulations(params, data, key=sk)
        ckpt = str((out_dir / f"ckpt_{a.arm_label}_s{seed}").resolve())
        inf, _m, _d = train_with_nan_retry(inf, ckpt, a.epochs, a.learning_rate,
                                           a.batch_size, a.warmup_steps, a.decay_steps,
                                           params, data, sk)
        posteriors.append((seed, inf.build_posterior()))
        print(f"  NDE seed {seed} trained in {time.time()-t0:.0f}s", flush=True)

    def sample_pooled(x_obs, n_per_seed):
        out = []
        for seed, post in posteriors:
            k = jax.random.PRNGKey(int(seed) + 7)
            out.append(np.asarray(post.sample(x=jnp.asarray(x_obs), num_samples=n_per_seed, key=k)))
        s = np.concatenate(out, 0)
        return s[np.all(np.isfinite(s), axis=1)]

    # ---- G3: reproduce campaign perm-0 FoM3 before trusting anything ----
    if a.expected_fom3 > 0:
        sel0 = (perm == 0) & (patch == 0)
        s0 = sample_pooled(S[sel0][0], a.step1_samples)
        f0 = compute_fom3(s0)["fom3"]
        rel = abs(f0 - a.expected_fom3) / a.expected_fom3
        ok = rel <= a.g3_tol
        print(f"  [G3] perm0/patch0 FoM3={f0:.0f} vs campaign {a.expected_fom3:.0f} "
              f"(rel {rel*100:.1f}%) -> {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            raise SystemExit(f"[G3] FAILED for {a.arm_label}: {f0:.0f} vs {a.expected_fom3:.0f} "
                             f"(rel {rel*100:.1f}% > {a.g3_tol*100:.0f}%); aborting (no garbage).")

    # ---- STEP 1: mean datavector ----
    print("######## STEP 1 — mean datavector ########", flush=True)
    s1 = sample_pooled(mean_dv, a.step1_samples)
    np.save(out_dir / "mean_dv_posterior.npy", s1)
    m1 = {**compute_fom3(s1), **fom2d(s1), **marginal_stats(s1),
          "n_samples": int(s1.shape[0]), "n_perms_averaged": int(len(np.unique(perm))),
          "n_patches_averaged": int(S.shape[0])}
    (out_dir / "mean_dv.fom.json").write_text(json.dumps(m1, indent=2))
    print(f"  [step1] FoM3={m1['fom3']:.0f} sig(w0)={m1['sigma']['w_0']:.4f} "
          f"sig(Om)={m1['sigma']['Omega_m']:.4f}", flush=True)

    # ---- STEP 2: per-patch distribution ----
    print("######## STEP 2 — per-patch FoM distribution ########", flush=True)
    rng = np.random.default_rng(1234)
    n2 = int(min(a.n_step2_patches, S.shape[0]))
    idx = rng.choice(S.shape[0], size=n2, replace=False)
    rows = []
    t0 = time.time()
    for j, i in enumerate(idx):
        sp = sample_pooled(S[i], a.step2_samples)
        f3 = compute_fom3(sp); f2 = fom2d(sp); mg = marginal_stats(sp)
        rows.append({"patch_global_idx": int(i), "perm": int(perm[i]), "patch": int(patch[i]),
                     "fom3": f3["fom3"], "valid_fom3": f3["valid_fom3"],
                     **f2, **{f"sig_{p}": mg["sigma"][p] for p in PARAM_KEYS}})
        if (j + 1) % 50 == 0:
            print(f"  [step2] {j+1}/{n2} ({(time.time()-t0)/(j+1):.2f}s/patch)", flush=True)
    with open(out_dir / "per_patch_fom.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    f3s = np.array([r["fom3"] for r in rows if r["valid_fom3"]], float)
    sw = np.array([r["sig_w_0"] for r in rows], float)
    som = np.array([r["sig_Omega_m"] for r in rows], float)
    summ = {"n_patches": n2, "fom3_mean": float(np.mean(f3s)), "fom3_std": float(np.std(f3s)),
            "fom3_median": float(np.median(f3s)), "fom3_p16": float(np.percentile(f3s, 16)),
            "fom3_p84": float(np.percentile(f3s, 84)),
            "sig_w0_mean": float(np.mean(sw)), "sig_w0_std": float(np.std(sw)),
            "sig_Om_mean": float(np.mean(som)), "sig_Om_std": float(np.std(som))}
    (out_dir / "step2_distribution_summary.json").write_text(json.dumps(summ, indent=2))
    np.savez(out_dir / "step2_fom3.npz", fom3=f3s, sig_w0=sw, sig_Om=som)
    print(f"  [step2] FoM3 mean {summ['fom3_mean']:.0f}±{summ['fom3_std']:.0f} "
          f"median {summ['fom3_median']:.0f} [{summ['fom3_p16']:.0f},{summ['fom3_p84']:.0f}]; "
          f"sig(w0) {summ['sig_w0_mean']:.4f}±{summ['sig_w0_std']:.4f}", flush=True)
    print(f"[{a.arm_label}] DONE.", flush=True)


if __name__ == "__main__":
    main()
