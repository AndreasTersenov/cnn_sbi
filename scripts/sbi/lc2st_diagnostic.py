#!/usr/bin/env python3
"""L-C2ST (plain, from-scratch, sklearn) — LOCAL calibration of q(theta|x) at the fiducial.

Tests whether the estimated posterior equals the true posterior LOCALLY at an observation x0
(Linhart et al. 2023, verified against github.com/JuliaLinhart/lc2st). This is the test SBC is
blind to by construction: SBC averages over the prior, so a miscalibration localized at the
fiducial (e.g. L1's -0.37sigma w0 offset that CANCELS globally) is invisible to SBC but visible
to L-C2ST.

CONSTRUCTION (plain L-C2ST):
  Calibration set: N pairs (theta_i, x_i) from the joint (val). Draw theta_tilde_i ~ q(.|x_i).
    class TRUE (label 0): features [theta_i,       x_i]
    class EST  (label 1): features [theta_tilde_i, x_i]   (SAME x_i in both classes)
  Train classifier d([theta,x]) -> P(label=1). Under H0 (q=p) the joints are identical -> d==0.5.
  LOCAL statistic at x0:  draw theta_k ~ q(.|x0);  T(x0) = mean_k (d([theta_k, x0]) - 0.5)^2.
  NULL: permute the 2N labels, retrain, recompute T(x0). p = (1+#{T_perm >= T_obs})/(n_null+1).
    (the permutation null absorbs finite-sample classifier overfitting -> no holdout needed.)

CORRECTNESS GATE (runs first; aborts on fail):
  ST_H0  (calibrated by construction): BOTH classes drawn from q(.|x_i) -> must give p>0.05
         (no false positive). Validates the test does not reject a calibrated q.
  ST_H1  (known miscalibration): class EST = q-draws shifted +shift*sigma on w0; eval theta also
         shifted -> must give p<0.05. Validates POWER (the test detects a known local offset).
         For L1 (high-dim x, no-PCA) this self-test also REPORTS power at L1's dimensionality.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
PARAM_KEYS = ["Omega_m", "sigma_8", "w_0", "h_0", "n_s", "Omega_b"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache-dir", required=True)
    p.add_argument("--cache-prefix", choices=["cnn", "l1"], required=True)
    p.add_argument("--arm-label", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--fiducial-summaries-npz", required=True, help="per-patch fiducial summaries (S, perm, patch)")
    p.add_argument("--preproc-transform", default="none",
                   choices=["none", "zscore", "log1p-zscore", "log10p-zscore"])
    p.add_argument("--clip-value", type=float, default=0.0)
    p.add_argument("--min-feature-variance", type=float, default=1e-12)
    p.add_argument("--seeds", default="41,42,43")
    p.add_argument("--n-cal", type=int, default=2000, help="calibration pairs per class")
    p.add_argument("--n-eval", type=int, default=1000, help="theta draws at each x0 for the statistic")
    p.add_argument("--n-x0", type=int, default=30, help="number of typical fiducial obs to test at")
    p.add_argument("--n-null", type=int, default=100, help="permutation null trials")
    p.add_argument("--clf-kind", default="logreg", choices=["logreg", "mlp"],
                   help="logreg = validated powered+valid config (synthetic power study 2026-06-04); "
                        "mlp overfits at our N and is underpowered")
    p.add_argument("--clf-hidden", default="64,64")
    p.add_argument("--st-shift", type=float, default=0.5, help="ST_H1 known w0 shift in sigma")
    p.add_argument("--st-ncal", type=int, default=1000, help="smaller N for the self-test gate")
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
    from train_jaxili_from_compressed import setup_env
    setup_env(a.cuda_visible_devices)
    import jax, jax.numpy as jnp
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from jaxili.inference import NPE
    from npe_l1norm_cross_jaxili_nbody_tomo import (
        train_with_nan_retry, preprocess_summaries, filter_zero_variance_bins,
    )
    hidden = tuple(int(h) for h in a.clf_hidden.split(","))
    w_idx = PARAM_KEYS.index("w_0")
    rng = np.random.default_rng(0)

    # ---- load train + val + fiducial; preprocess val & fiducial with train-fit pipeline ----
    cdir = Path(a.train_cache_dir)
    tr = np.load(cdir / f"{a.cache_prefix}_train.npz")
    theta_tr = tr["theta"].astype(np.float32); x_tr_raw = tr["x"].astype(np.float64)
    va = np.load(cdir / f"{a.cache_prefix}_val.npz")
    theta_va_all = va["theta"].astype(np.float64)
    # calibration: N_cal random val pairs
    n_cal_max = max(a.n_cal, a.st_ncal)
    cal_idx = rng.choice(theta_va_all.shape[0], size=min(n_cal_max, theta_va_all.shape[0]), replace=False)
    theta_cal = theta_va_all[cal_idx]
    x_cal_raw = va["x"][cal_idx].astype(np.float64)

    z = np.load(a.fiducial_summaries_npz)
    S_fid_raw = z["S"].astype(np.float64); fid_patch = z["patch"].astype(int)

    clip = a.clip_value if a.clip_value > 0 else None
    tr_proc, _, cal_proc, mean, std = preprocess_summaries(
        x_tr_raw, x_tr_raw[:1], x_cal_raw, summary_transform=a.preproc_transform, clip_value=clip)
    _, _, fid_proc, _, _ = preprocess_summaries(
        x_tr_raw, x_tr_raw[:1], S_fid_raw, summary_transform=a.preproc_transform, clip_value=clip, mean=mean, std=std)
    mask, _ = filter_zero_variance_bins(tr_proc, min_variance=a.min_feature_variance, verbose=False)
    x_tr = tr_proc[:, mask].astype(np.float32)
    x_cal = cal_proc[:, mask].astype(np.float32)
    x_fid = fid_proc[:, mask].astype(np.float32)
    print(f"[{a.arm_label}] train x{x_tr.shape}; cal x{x_cal.shape}; fiducial x{x_fid.shape}", flush=True)

    # ---- typical (non-polar) fiducial obs as x0 set: exclude patch 0 (polar) ----
    typ = np.where(fid_patch != 0)[0]
    x0_idx = rng.choice(typ, size=min(a.n_x0, len(typ)), replace=False)
    X0 = x_fid[x0_idx]  # (n_x0, dim)

    # ---- train 3-seed NDE (pooled q) ----
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    out_dir = Path(a.output_dir) / a.arm_label; out_dir.mkdir(parents=True, exist_ok=True)
    posteriors = []
    params = jnp.asarray(theta_tr); data = jnp.asarray(x_tr)
    for seed in seeds:
        sk = jax.random.PRNGKey(int(seed) + 1)
        inf = NPE().append_simulations(params, data, key=sk)
        ckpt = str((out_dir / f"ckpt_{a.arm_label}_s{seed}").resolve())
        inf, _m, _d = train_with_nan_retry(inf, ckpt, a.epochs, a.learning_rate, a.batch_size,
                                           a.warmup_steps, a.decay_steps, params, data, sk)
        posteriors.append((seed, inf.build_posterior()))
        print(f"  NDE seed {seed} trained", flush=True)

    def sample_one_per_x(x_arr, salt=0):
        """one posterior draw per row of x_arr -> (len, 6). Stratified over the 3 seeds (i%3),
        which marginally = a pooled-q draw. `salt` makes repeated calls INDEPENDENT (needed so the
        two self-test classes are distinct draws, not identical points)."""
        out = np.empty((len(x_arr), 6), np.float32)
        for i, xo in enumerate(x_arr):
            seed, post = posteriors[i % len(posteriors)]
            k = jax.random.PRNGKey(int(seed) * 100003 + i * 7 + int(salt) * 1000003)
            s = np.asarray(post.sample(x=jnp.asarray(xo), num_samples=1, key=k))[0]
            out[i] = s
        return out

    def sample_many(x0, n):
        out = []
        for seed, post in posteriors:
            k = jax.random.PRNGKey(int(seed) + 7)
            out.append(np.asarray(post.sample(x=jnp.asarray(x0), num_samples=n // len(posteriors) + 1, key=k)))
        s = np.concatenate(out, 0)
        s = s[np.all(np.isfinite(s), axis=1)]
        return s[:n]

    # ---- L-C2ST core (classifier + permutation null, eval at a set of x0) ----
    def run_lc2st(theta_c0, theta_c1, x_c, X0_set, theta_eval_list, n_null, tag):
        """theta_eval_list[i] = (K,6) PRE-sampled draws used in the local statistic at X0_set[i].
        The SAME draws are used for the observed classifier and every permutation-null classifier,
        so the null isolates the label-permutation effect (not sampling noise). Returns (T_obs,p,T_null)."""
        Xfeat = np.vstack([np.hstack([theta_c0, x_c]), np.hstack([theta_c1, x_c])]).astype(np.float32)
        y = np.concatenate([np.zeros(len(x_c)), np.ones(len(x_c))]).astype(int)
        scaler = StandardScaler().fit(Xfeat); Xs = scaler.transform(Xfeat)
        # precompute the evaluation feature matrix at each x0 ONCE (shared across all classifiers)
        feat_by_x0 = [scaler.transform(np.hstack([te, np.tile(x0, (len(te), 1))]).astype(np.float32))
                      for te, x0 in zip(theta_eval_list, X0_set)]

        def stat(clf, feat):
            d = clf.predict_proba(feat)[:, 1]
            return float(np.mean((d - 0.5) ** 2))

        def fit_clf(yy, seed):
            if a.clf_kind == "logreg":   # validated config (power study 2026-06-04): valid + powered
                return LogisticRegression(C=1.0, max_iter=500).fit(Xs, yy)
            return MLPClassifier(hidden_layer_sizes=hidden, max_iter=300, random_state=seed,
                                 alpha=1e-3).fit(Xs, yy)

        t0 = time.time()
        clf = fit_clf(y, 0)
        T_obs = np.array([stat(clf, f) for f in feat_by_x0])           # (n_x0,)
        T_null = np.empty((n_null, len(X0_set)))
        for b in range(n_null):
            yp = rng.permutation(y)
            clf_p = fit_clf(yp, 1000 + b)
            T_null[b] = [stat(clf_p, f) for f in feat_by_x0]
        # p per x0: fraction of null >= obs (1-sided; T>=0, large T => miscalibrated)
        p = (1 + (T_null >= T_obs[None, :]).sum(0)) / (n_null + 1)
        print(f"  [{tag}] clf+{n_null}null in {time.time()-t0:.0f}s | "
              f"median T_obs {np.median(T_obs):.4e}, median p {np.median(p):.3f}, "
              f"frac p<0.05 = {np.mean(p<0.05):.2f}", flush=True)
        return T_obs, p, T_null

    # ============ CORRECTNESS GATE ============
    print("\n## L-C2ST SELF-TESTS (gate)", flush=True)
    sc = min(a.st_ncal, len(x_cal))
    xc = x_cal[:sc]
    X0g = X0[:8]
    n_null_gate = min(50, a.n_null)
    # ST_H0: both classes from q(.|x_i) -> calibrated by construction -> expect p>0.05
    eval_h0 = [sample_many(x0, a.n_eval) for x0 in X0g]
    th_q_a = sample_one_per_x(xc, salt=1); th_q_b = sample_one_per_x(xc, salt=2)  # INDEPENDENT q-draws
    _, p_h0, _ = run_lc2st(th_q_a, th_q_b, xc, X0g, eval_h0, n_null_gate, "ST_H0 calibrated")
    h0_ok = np.median(p_h0) > 0.05
    # ST_H1: REUSE the H0 draws (no extra sampling). class1 = th_q_b shifted +shift*sigma on w0;
    # class0 = th_q_a (unshifted); eval draws = H0 eval shifted -> a KNOWN miscalibration -> expect p<0.05.
    sig_w = float(np.std(np.concatenate([e[:, w_idx] for e in eval_h0])))
    th_c0 = th_q_a
    th_c1 = th_q_b.copy(); th_c1[:, w_idx] += a.st_shift * sig_w
    eval_h1 = [e.copy() for e in eval_h0]
    for e in eval_h1:
        e[:, w_idx] += a.st_shift * sig_w
    _, p_h1, _ = run_lc2st(th_c0, th_c1, xc, X0g, eval_h1, n_null_gate, "ST_H1 +shift")
    h1_ok = np.median(p_h1) < 0.05
    print(f"  ST_H0 (calibrated -> want median p>0.05): median p={np.median(p_h0):.3f} -> {'OK' if h0_ok else 'FAIL'}")
    print(f"  ST_H1 (+{a.st_shift}sig w0 -> want median p<0.05): median p={np.median(p_h1):.3f} -> {'OK' if h1_ok else 'FAIL'}")
    if not (h0_ok and h1_ok):
        raise SystemExit(f"[GATE] L-C2ST self-tests FAILED (H0 ok={h0_ok}, H1 ok={h1_ok}); "
                         f"H1 fail at this x-dim = UNDERPOWERED. Aborting (no misleading result).")
    print("  >>> L-C2ST GATE PASSED <<<\n", flush=True)

    # ============ REAL L-C2ST at the fiducial ============
    print("## REAL L-C2ST (true vs estimator joint), at typical fiducial obs", flush=True)
    nc = min(a.n_cal, len(x_cal))
    xc = x_cal[:nc]
    theta_true = theta_cal[:nc].astype(np.float32)            # class TRUE
    theta_est = sample_one_per_x(xc, salt=5)                  # class EST ~ q(.|x_i)
    eval_real = [sample_many(x0, a.n_eval) for x0 in X0]      # pre-sampled once, shared
    T_obs, p, T_null = run_lc2st(theta_true, theta_est, xc, X0, eval_real, a.n_null, "REAL")
    summary = {"arm": a.arm_label, "n_cal": nc, "n_x0": len(X0), "n_null": a.n_null,
               "median_T_obs": float(np.median(T_obs)), "median_p": float(np.median(p)),
               "frac_reject_p05": float(np.mean(p < 0.05)), "frac_reject_p01": float(np.mean(p < 0.01)),
               "gate": {"st_h0_median_p": float(np.median(p_h0)), "st_h1_median_p": float(np.median(p_h1))}}
    (out_dir / "lc2st_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(out_dir / "lc2st_results.npz", T_obs=T_obs, p=p, T_null=T_null, x0_patch=fid_patch[x0_idx])
    print(f"\n## RESULT [{a.arm_label}]: at {len(X0)} typical fiducial obs, median p={np.median(p):.3f}, "
          f"reject(p<0.05) {np.mean(p<0.05)*100:.0f}% of obs", flush=True)
    print(f"   interpretation: high reject-frac => q is LOCALLY miscalibrated at the fiducial.", flush=True)
    print(f"[{a.arm_label}] DONE -> {out_dir}/lc2st_summary.json", flush=True)


if __name__ == "__main__":
    main()
