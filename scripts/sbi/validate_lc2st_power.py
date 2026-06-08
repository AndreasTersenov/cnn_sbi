#!/usr/bin/env python3
"""L-C2ST power calibration on SYNTHETIC data (no NDE / no GPU sampling).

Isolates the test's operating characteristics vs (x-dim, N_cal, classifier) so we pick a config
that (a) does NOT false-positive under H0 and (b) HAS power to detect a known ~0.4sigma local
offset -- BEFORE spending NDE sampling. Also directly answers the L1 feasibility question
(does the test retain power when x is 2000-dim and we cannot PCA it?).

Synthetic model (mimics an SBI posterior):
  x_i ~ N(0, I_d);  true:  theta|x ~ N(M x, Sigma);  est: theta|x ~ N(M x + bias, Sigma)
  bias=0 => H0 (calibrated);  bias on w0 => H1 (known local offset).
Reuses the SAME L-C2ST core (classifier + label-permutation null + local MSE-from-0.5 statistic).
"""
from __future__ import annotations
import argparse, warnings
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*max_iter.*")

SIG = np.array([0.025, 0.038, 0.13, 0.05, 0.04, 0.01])  # ~campaign posterior widths


def make_clf(kind, seed):
    if kind == "mlp64x64":
        return MLPClassifier((64, 64), alpha=1e-3, max_iter=300, random_state=seed)
    if kind == "mlp32_es":
        return MLPClassifier((32,), alpha=1e-2, max_iter=300, random_state=seed,
                             early_stopping=True, n_iter_no_change=8)
    if kind == "mlp64_a1":
        return MLPClassifier((64,), alpha=1.0, max_iter=300, random_state=seed)
    if kind == "logreg":
        return LogisticRegression(C=1.0, max_iter=500)
    raise ValueError(kind)


def run_lc2st(theta0, theta1, xc, X0, eval_list, n_null, clf_kind, rng):
    Xf = np.vstack([np.hstack([theta0, xc]), np.hstack([theta1, xc])]).astype(np.float32)
    y = np.concatenate([np.zeros(len(xc)), np.ones(len(xc))]).astype(int)
    sc = StandardScaler().fit(Xf); Xs = sc.transform(Xf)
    feats = [sc.transform(np.hstack([te, np.tile(x0, (len(te), 1))]).astype(np.float32))
             for te, x0 in zip(eval_list, X0)]
    def stat(clf, f): d = clf.predict_proba(f)[:, 1]; return float(np.mean((d - 0.5) ** 2))
    clf = make_clf(clf_kind, 0).fit(Xs, y)
    T_obs = np.array([stat(clf, f) for f in feats])
    T_null = np.empty((n_null, len(X0)))
    for b in range(n_null):
        T_null[b] = [stat(make_clf(clf_kind, 1000 + b).fit(Xs, rng.permutation(y)), f) for f in feats]
    p = (1 + (T_null >= T_obs[None, :]).sum(0)) / (n_null + 1)
    return T_obs, p


def scenario(d, n_cal, clf_kind, bias_w0, n_x0=8, n_eval=400, n_null=60, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.normal(0, 1.0 / np.sqrt(d), size=(6, d)) * SIG[:, None]   # x->theta mean map, scaled
    L = np.diag(SIG)                                                   # posterior chol (diag)
    bias = np.zeros(6); bias[2] = bias_w0 * SIG[2]                     # w0 = index 2
    xc = rng.normal(0, 1, size=(n_cal, d))
    mu = xc @ M.T
    theta_true = mu + rng.normal(0, 1, (n_cal, 6)) @ L
    theta_est = mu + bias + rng.normal(0, 1, (n_cal, 6)) @ L
    X0 = rng.normal(0, 1, size=(n_x0, d))
    mu0 = X0 @ M.T
    eval_list = [mu0[i] + bias + rng.normal(0, 1, (n_eval, 6)) @ L for i in range(n_x0)]
    _, p = run_lc2st(theta_true, theta_est, xc, X0, eval_list, n_null, clf_kind, rng)
    return float(np.median(p)), float(np.mean(p < 0.05))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="10,2000")
    ap.add_argument("--ncals", default="500,2000,5000")
    ap.add_argument("--clfs", default="mlp64x64,mlp32_es,mlp64_a1,logreg")
    ap.add_argument("--bias", type=float, default=0.4, help="H1 w0 shift in sigma")
    args = ap.parse_args()
    print(f"H1 shift = {args.bias}sigma on w0. Want: H0 median-p>0.05 AND H1 median-p<0.05 "
          f"(ideally H1 reject-frac high).\n")
    print(f"{'dim':>5} {'n_cal':>6} {'clf':>10} | {'H0 med-p':>8} {'H0 rej':>6} | {'H1 med-p':>8} {'H1 rej':>6}  verdict")
    for d in [int(x) for x in args.dims.split(",")]:
        for nc in [int(x) for x in args.ncals.split(",")]:
            for clf in args.clfs.split(","):
                h0_p, h0_rej = scenario(d, nc, clf, 0.0, seed=1)
                h1_p, h1_rej = scenario(d, nc, clf, args.bias, seed=2)
                ok = (h0_p > 0.05) and (h1_p < 0.05)
                v = "GOOD" if ok else ("no-power" if h1_p >= 0.05 else "false-pos")
                print(f"{d:>5} {nc:>6} {clf:>10} | {h0_p:8.3f} {h0_rej:6.2f} | {h1_p:8.3f} {h1_rej:6.2f}  {v}")
        print()


if __name__ == "__main__":
    main()
