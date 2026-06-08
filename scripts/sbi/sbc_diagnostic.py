#!/usr/bin/env python3
"""Simulation-Based Calibration (SBC) for ONE arm — global calibration via rank statistics.

Draw (theta_i, x_i) from the joint (held-out val pairs; theta from the grid prior, x its
realization). Sample M posterior draws from q(theta|x_i) (3-seed pool, the campaign posterior).
rank_p = #{posterior samples < theta_i[p]} in {0..M}. Over N cosmologies the per-param rank
histogram is UNIFORM iff q is calibrated. Direction convention (asserted by self-test 2):
posterior biased LOW (mean<truth) -> truth in upper tail -> ranks skew HIGH.

This is the global counterpart to the local fiducial offset: if w0 ranks are uniform, the
-0.37sigma fiducial w0 offset CANCELS across the prior (globally calibrated, locally offset).

CORRECTNESS GATE (runs before the real SBC; aborts on fail):
  ST1 (calibrated): rank a fresh posterior draw among other draws from the SAME q -> must be
       uniform (mean norm-rank ~0.5, KS p>0.01). Validates the rank/sampler code is unbiased.
  ST2 (known shift): shifting w0 samples by +0.3*sigma must move the mean norm-rank DOWN
       (more samples above truth -> fewer below -> lower rank), monotonically in {-0.3,0,+0.3}.
       Validates the code DETECTS a known miscalibration in the right direction.

Reuses the campaign-exact preprocessing + NDE (same as geometry_resample/fiducial_analyze).
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
    p.add_argument("--preproc-transform", default="none",
                   choices=["none", "zscore", "log1p-zscore", "log10p-zscore"])
    p.add_argument("--clip-value", type=float, default=0.0)
    p.add_argument("--min-feature-variance", type=float, default=1e-12)
    p.add_argument("--seeds", default="41,42,43")
    p.add_argument("--n-cosmologies", type=int, default=0, help="0 => one realization per unique val theta")
    p.add_argument("--m-per-seed", type=int, default=1000, help="posterior samples per seed (pooled = 3x)")
    p.add_argument("--epochs", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--decay-steps", type=int, default=10000)
    p.add_argument("--st-shift", type=float, default=0.3, help="self-test 2 shift in units of sigma")
    p.add_argument("--cuda-visible-devices", default="1")
    return p.parse_args()


def main():
    a = parse_args()
    sys.path.insert(0, str(REPO))
    from train_jaxili_from_compressed import setup_env
    setup_env(a.cuda_visible_devices)
    import jax, jax.numpy as jnp
    from scipy import stats as sstats
    from jaxili.inference import NPE
    from npe_l1norm_cross_jaxili_nbody_tomo import (
        train_with_nan_retry, preprocess_summaries, filter_zero_variance_bins,
    )

    cdir = Path(a.train_cache_dir)
    tr = np.load(cdir / f"{a.cache_prefix}_train.npz")
    theta_tr = tr["theta"].astype(np.float32); x_tr_raw = tr["x"].astype(np.float64)
    va = np.load(cdir / f"{a.cache_prefix}_val.npz")
    theta_va_full = va["theta"].astype(np.float64)

    # ---- pick N test cosmologies: one random realization per unique val theta (BEFORE preprocess) ----
    rng = np.random.default_rng(7)
    uniq = np.unique(theta_va_full, axis=0)
    sel = []
    for u in uniq:
        rows = np.where((theta_va_full == u).all(axis=1))[0]
        sel.append(rng.choice(rows))
    sel = np.array(sel)
    if a.n_cosmologies > 0 and a.n_cosmologies < len(sel):
        sel = rng.choice(sel, size=a.n_cosmologies, replace=False)
    theta_va = theta_va_full[sel]                 # (N,6), aligned with x_va below
    x_va_raw = va["x"][sel].astype(np.float64)    # only the N selected realizations

    # ---- preprocessing fit on TRAIN, applied to the selected val rows (campaign NDE input) ----
    clip = a.clip_value if a.clip_value > 0 else None
    tr_proc, _, va_proc, mean, std = preprocess_summaries(
        x_tr_raw, x_tr_raw[:1], x_va_raw, summary_transform=a.preproc_transform, clip_value=clip)
    mask, _ = filter_zero_variance_bins(tr_proc, min_variance=a.min_feature_variance, verbose=False)
    x_tr = tr_proc[:, mask].astype(np.float32)
    x_va = va_proc[:, mask].astype(np.float32)     # (N, masked_dim), row-aligned with theta_va
    N = len(sel)
    print(f"[{a.arm_label}] train x{x_tr.shape}; SBC over N={N} cosmologies "
          f"(of {uniq.shape[0]} unique val theta)", flush=True)

    # ---- train 3-seed NDE (campaign-exact) ----
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
                                           a.batch_size, a.warmup_steps, a.decay_steps, params, data, sk)
        posteriors.append((seed, inf.build_posterior()))
        print(f"  NDE seed {seed} trained in {time.time()-t0:.0f}s", flush=True)

    def sample_pooled(x_obs, n_per_seed):
        out = []
        for seed, post in posteriors:
            k = jax.random.PRNGKey(int(seed) + 7)
            out.append(np.asarray(post.sample(x=jnp.asarray(x_obs), num_samples=n_per_seed, key=k)))
        s = np.concatenate(out, 0)
        return s[np.all(np.isfinite(s), axis=1)]

    def norm_ranks(theta_true, S):
        """normalized rank per param: #(samples < truth)/M, in [0,1]."""
        return (S < theta_true[None, :]).sum(0) / S.shape[0]

    # ========================= CORRECTNESS GATE =========================
    print("\n## SELF-TEST 1 (calibrated control: rank a posterior draw among draws from the SAME q)", flush=True)
    st1 = []
    for i in range(min(120, N)):
        S = sample_pooled(x_va[i], a.m_per_seed)
        if S.shape[0] < 50:
            continue
        # pseudo-truth must be a draw from the SAME (pooled 3-seed mixture) distribution as the
        # comparison set, else it's not exchangeable. S is ordered by seed-block, so SHUFFLE first
        # -> Sp[0] is a uniformly-random element of the mixture, exchangeable with the rest.
        Sp = S[np.random.default_rng(1000 + i).permutation(S.shape[0])]
        pseudo_truth = Sp[0]; rest = Sp[1:]
        st1.append((rest < pseudo_truth[None, :]).sum(0) / rest.shape[0])
    st1 = np.array(st1)  # (n,6)
    st1_ok = True
    for p in range(3):
        ks = sstats.kstest(st1[:, p], "uniform")
        mean_r = st1[:, p].mean()
        ok = (ks.pvalue > 0.01) and (abs(mean_r - 0.5) < 0.05)
        st1_ok &= ok
        print(f"  {PARAM_KEYS[p]:9s} mean-rank {mean_r:.3f} (want 0.5)  KS p={ks.pvalue:.3f}  -> {'OK' if ok else 'FAIL'}")
    if not st1_ok:
        raise SystemExit("[ST1] calibrated control NOT uniform -> rank/sampler code is wrong; aborting.")

    print(f"\n## SELF-TEST 2 (known +/-{a.st_shift}sigma shift on w_0 must move mean-rank monotonically DOWN)", flush=True)
    w = PARAM_KEYS.index("w_0")
    means = {}
    for shift in (-a.st_shift, 0.0, a.st_shift):
        rr = []
        for i in range(min(120, N)):
            S = sample_pooled(x_va[i], a.m_per_seed)
            if S.shape[0] < 50:
                continue
            Sa = S.copy(); Sa[:, w] = Sa[:, w] + shift * S[:, w].std()
            rr.append((Sa[:, w] < theta_va[i, w]).sum() / Sa.shape[0])
        means[shift] = float(np.mean(rr))
        print(f"  shift {shift:+.2f}sigma -> mean w0 norm-rank {means[shift]:.3f}")
    mono = means[-a.st_shift] > means[0.0] > means[a.st_shift]
    detect = (means[0.0] - means[a.st_shift]) > 0.05
    print(f"  monotonic down? {mono} ; +shift detectably lowers rank? {detect} -> {'OK' if (mono and detect) else 'FAIL'}")
    if not (mono and detect):
        raise SystemExit("[ST2] code does not detect a known shift in the right direction; aborting.")
    print("\n  >>> CORRECTNESS GATE PASSED <<<\n", flush=True)

    # ========================= REAL SBC =========================
    print("## REAL SBC", flush=True)
    R = []
    t0 = time.time()
    for i in range(N):
        S = sample_pooled(x_va[i], a.m_per_seed)
        R.append(norm_ranks(theta_va[i], S))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{N} ({(time.time()-t0)/(i+1):.2f}s/cosmo)", flush=True)
    R = np.array(R)  # (N,6)
    np.savez(out_dir / "sbc_ranks.npz", ranks=R, theta=theta_va, params=PARAM_KEYS)

    summary = {"arm": a.arm_label, "N": int(R.shape[0]), "M": int(3 * a.m_per_seed)}
    print("\n## SBC RESULT (mean norm-rank: 0.5=unbiased; >0.5 => posterior biased LOW; KS vs uniform)")
    for p in range(6):
        ks = sstats.kstest(R[:, p], "uniform")
        mean_r = float(R[:, p].mean()); std_r = float(R[:, p].std())
        summary[PARAM_KEYS[p]] = {"mean_rank": mean_r, "std_rank": std_r,
                                  "ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue)}
        tag = "UNIFORM" if ks.pvalue > 0.05 else "NON-UNIFORM"
        bias = "low" if mean_r > 0.52 else ("high" if mean_r < 0.48 else "centered")
        print(f"  {PARAM_KEYS[p]:9s} mean-rank {mean_r:.3f} std {std_r:.3f} | KS p={ks.pvalue:.3g} [{tag}] bias={bias}")
    (out_dir / "sbc_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- rank histograms ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    for p in range(6):
        a_ = ax[p // 3, p % 3]
        a_.hist(R[:, p], bins=20, range=(0, 1), color="C0", alpha=0.8, density=True)
        a_.axhline(1.0, color="gray", lw=1, ls="--")
        a_.set_title(f"{PARAM_KEYS[p]} (mean {R[:,p].mean():.3f})")
        a_.set_xlabel("normalized rank"); a_.set_ylim(0, 2.2)
    fig.suptitle(f"SBC rank histograms — {a.arm_label} (N={R.shape[0]}, M={3*a.m_per_seed}); flat=calibrated")
    fig.tight_layout(); fig.savefig(out_dir / "sbc_histograms.png", dpi=130); plt.close(fig)
    print(f"\n[{a.arm_label}] DONE -> {out_dir}/sbc_summary.json + sbc_histograms.png", flush=True)


if __name__ == "__main__":
    main()
