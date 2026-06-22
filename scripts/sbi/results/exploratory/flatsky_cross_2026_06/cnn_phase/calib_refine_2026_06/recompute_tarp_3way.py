#!/usr/bin/env python3
"""3-way un-stratified TARP-DRP with proper 1-sigma bands: CNN vs L1(auto+product) vs joint-L1 (ensemble).

CNN and L1+product read their un-stratified dumps_all (600 obs x 3 NDE seeds pooled). The joint-L1
"reported" posterior is the 3-compressor-seed ENSEMBLE x 3 NDE seeds = 9 flows pooled; it has only
stratified dumps, so we reconstruct each (compressor, NDE-seed) 600-obs set by concatenating its
LOW/MID/HIGH terciles and aligning across combos on theta (tercile membership differs per arm, so a
row-position pool would mismatch obs). Then pool all 9 along the sample axis and subsample to 6000 for
comparability. Band = bootstrap of the 600 sightlines (16-84 pct = 1 sigma). CPU only.
Convention matches run_tarp_coverage.py: first 3 params; (M,N,3); references='random'; norm=True.
"""
from pathlib import Path
import glob
import numpy as np
import tarp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/flatsky_cross_2026_06"
AM = f"{ROOT}/analytical_nde_match"
OUT = Path(f"{ROOT}/cnn_phase/calib_refine_2026_06/figs")
ALPHA = np.linspace(0.0, 1.0, 61)
NBOOT = 200
RNG = np.random.default_rng(0)
JOINTL1_ARMS = ["jointl1_nobnt", "jointl1_nobnt_s42", "jointl1_nobnt_s43"]


def ecp_once(samp_NMD, theta_ND):
    s = np.transpose(samp_NMD, (1, 0, 2))
    ecp, alpha = tarp.get_tarp_coverage(s, theta_ND, references="random", norm=True)
    ecp = np.asarray(ecp); ecp = ecp.mean(0) if ecp.ndim == 2 else ecp
    return np.interp(ALPHA, np.asarray(alpha), ecp)


def tarp_band(samp, theta):
    N = theta.shape[0]
    boot = np.array([ecp_once(samp[i := RNG.integers(0, N, N)], theta[i]) for _ in range(NBOOT)])
    mean = boot.mean(0); lo, hi = np.percentile(boot, [16, 84], 0)
    return mean, lo, hi, float(np.trapz(mean - ALPHA, ALPHA) * 2), float(boot[:, 30].std())


def load_dumps_all(dumps_all):
    dd = sorted(glob.glob(f"{dumps_all}/seed_*/n*_m*/posterior_samples.npz"))
    arrs = [np.load(d) for d in dd]
    theta = arrs[0]["theta"][:, :3].astype(np.float32)
    samp = np.concatenate([a["samples"][:, :, :3] for a in arrs], axis=1).astype(np.float32)
    return samp, theta


def load_jointl1_ensemble():
    """theta-aligned pool over 3 compressor arms x 3 NDE seeds -> (600, 9*M, 3); subsample to 6000."""
    canon_theta, mats = None, []
    for arm in JOINTL1_ARMS:
        for seed in (41, 42, 43):
            S, T = [], []
            for f in sorted(glob.glob(f"{AM}/{arm}/gate/tarp_drp/dumps/*/seed_{seed}/n*_m*/posterior_samples.npz")):
                z = np.load(f); S.append(z["samples"][:, :, :3]); T.append(z["theta"][:, :3])
            S = np.concatenate(S, 0).astype(np.float32); T = np.concatenate(T, 0).astype(np.float32)
            order = np.lexsort(T.T[::-1])               # canonical order by theta
            S, T = S[order], T[order]
            if canon_theta is None:
                canon_theta = T
            assert np.allclose(T, canon_theta, atol=1e-5), f"theta mismatch for {arm} seed {seed}"
            mats.append(S)
    pooled = np.concatenate(mats, axis=1)               # (600, 9*2000, 3)
    idx = RNG.choice(pooled.shape[1], 6000, replace=False)
    return pooled[:, idx, :], canon_theta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arms = {}
    sC, tC = load_dumps_all(f"{ROOT}/cnn_phase/arch_sweep_2026_06_13/gate_c_resnet18/dumps_all/resnet18_all")
    arms["CNN"] = (sC, tC, "#1f77b4", "CNN auto-only (resnet18+RealNVP)")
    sL, tL = load_dumps_all(f"{ROOT}/gate_c/tarp_drp/dumps_all/flat_product_all")
    arms["L1product"] = (sL, tL, "#d62728", "L1 auto+product (MAF)")
    sJ, tJ = load_jointl1_ensemble()
    arms["jointL1"] = (sJ, tJ, "#2ca02c", "joint L1 (3-seed ensemble)")

    res = {}
    for name, (samp, theta, color, label) in arms.items():
        mean, lo, hi, net, se = tarp_band(samp, theta)
        res[name] = dict(mean=mean, lo=lo, hi=hi, net=net, se=se, color=color, label=label, n=theta.shape[0])
        print(f"{name:10s} | TARP net {net:+.4f} (SE@0.5 {se:.4f}, N={theta.shape[0]}, M={samp.shape[1]})", flush=True)

    np.savez(OUT.parent / "tarp_3way.npz", alpha=ALPHA,
             **{f"{k}_{q}": res[k][q] for k in res for q in ("mean", "lo", "hi", "net")})

    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7, label="ideal (calibrated)")
    for name in ("L1product", "jointL1", "CNN"):
        r = res[name]
        ax.fill_between(ALPHA, r["lo"], r["hi"], color=r["color"], alpha=0.20)
        ax.plot(ALPHA, r["mean"], color=r["color"], lw=2.2, label=f"{r['label']}  (net {r['net']:+.3f})")
    ax.set_xlabel("credibility level"); ax.set_ylabel("expected coverage probability")
    ax.set_title(r"TARP-DRP, un-stratified, 1$\sigma$ bands"
                 "\n(+ = conservative / over-covers;  − = over-confident)", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper left"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"tarp_cnn_l1_jointl1_unstratified_1sigma.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/tarp_cnn_l1_jointl1_unstratified_1sigma.{{pdf,png}}")


if __name__ == "__main__":
    main()
