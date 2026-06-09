#!/usr/bin/env python3
"""SBC (simulation-based calibration) rank test, reusing the GATE-C TARP dump posteriors.

The TARP dumps already hold posteriors at the held-out val ensemble (samples (N,M,6) + theta).
SBC rank for (val point i, param p) = fraction of the M posterior samples below the truth.
Calibrated => ranks UNIFORM on [0,1] (mean 0.5, std ~0.289, flat histogram, KS p large).
Pools all FoM3 terciles + 3 seeds per arm. Global test (a local bias that cancels over the prior
is invisible here — that's L-C2ST's job). CPU-only.
"""
import os, glob, numpy as np
from scipy import stats
from scipy.stats import binom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
G = HERE + "/results/exploratory/flatsky_cross_2026_06/gate_c/tarp_drp"
OUT = HERE + "/results/exploratory/flatsky_cross_2026_06/gate_c/sbc"
os.makedirs(OUT, exist_ok=True)
ARMS = [("flat_none", "auto-only", "#555555"), ("flat_conv", "+conv", "#1f77b4"),
        ("flat_product", "+product", "#2ca02c"), ("flat_both", "+both", "#d62728")]
PNAMES = ["Om", "s8", "w0", "h0", "ns", "Ob"]
SCI = [0, 1, 2]


def ranks_by_seed(arm):
    """Return {seed: ranks (N,6)} — terciles pooled WITHIN a seed (independent 600-pt
    ensemble). KS must be per-seed; pooling seeds correlates the same val points and
    over-rejects. The figure pools for visualization only."""
    out = {}
    for seed in (41, 42, 43):
        rs = []
        for f in sorted(glob.glob(f"{G}/dumps/{arm}_*/seed_{seed}/n*_m*/posterior_samples.npz")):
            z = np.load(f)
            rs.append((z["samples"] < z["theta"][:, None, :]).mean(axis=1))
        if rs:
            out[seed] = np.concatenate(rs, axis=0)
    return out


def ranks_for_arm(arm):
    bs = ranks_by_seed(arm)
    return np.concatenate(list(bs.values()), axis=0) if bs else None


def main():
    fig, axes = plt.subplots(len(ARMS), 3, figsize=(11, 11), sharex=True)
    print(f"{'arm':10s} {'param':5s}  {'mean':>6s} {'std':>6s}  {'KS p (per-seed mean[range])':>28s}  verdict")
    summary = {}
    for ai, (arm, lab, col) in enumerate(ARMS):
        bs = ranks_by_seed(arm)
        R = np.concatenate(list(bs.values()), axis=0)   # pooled, for the histogram only
        summary[arm] = {}
        for k, pi in enumerate(SCI):
            r = R[:, pi]
            ks_seed = [stats.kstest(bs[s][:, pi], "uniform").pvalue for s in bs]   # INDEPENDENT
            ksm = float(np.mean(ks_seed))
            mean, std = r.mean(), r.std()
            ok = ksm > 0.05 and abs(mean - 0.5) < 0.05
            verdict = "UNIFORM (calibrated)" if ok else ("mild" if ksm > 0.02 else "NON-UNIFORM")
            print(f"{lab:10s} {PNAMES[pi]:5s}  {mean:6.3f} {std:6.3f}  "
                  f"{ksm:6.2f} [{min(ks_seed):.2f},{max(ks_seed):.2f}]{'':10s}  {verdict}")
            summary[arm][PNAMES[pi]] = dict(mean=float(mean), std=float(std),
                                            ks_p_per_seed_mean=ksm,
                                            ks_p_per_seed=[float(x) for x in ks_seed])
            ax = axes[ai, k]
            # bars = mean over seeds of the per-seed density histogram (each seed = 600
            # independent val obs). band = 99% binomial uniform null for N=600 (Talts 2018),
            # converted to density (count/(N*binwidth)). N_indep=600, not the 1800 pooled.
            nb = 20; N_indep = min(len(v) for v in bs.values()); bw = 1.0 / nb
            dens = np.mean([np.histogram(bs[s][:, pi], bins=nb, range=(0, 1), density=True)[0]
                            for s in bs], axis=0)
            edges = np.linspace(0, 1, nb + 1)
            lo, hi = binom.ppf(0.005, N_indep, 1 / nb) / (N_indep * bw), \
                     binom.ppf(0.995, N_indep, 1 / nb) / (N_indep * bw)
            ax.axhspan(lo, hi, color="0.85", zorder=0)          # 99% uniform band
            ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.6, zorder=1)
            ax.stairs(dens, edges, color=col, fill=True, alpha=0.75, zorder=2)
            n_out = int(((dens < lo) | (dens > hi)).sum())
            summary[arm][PNAMES[pi]]["bins_outside_99band"] = n_out
            if ai == 0:
                ax.set_title(PNAMES[pi])
            if k == 0:
                ax.set_ylabel(lab, fontsize=10)
            ax.set_ylim(0, 2.0)
        print()
    for ax in axes[-1]:
        ax.set_xlabel("SBC rank")
    fig.suptitle("GATE C — SBC rank histograms (flat within grey 99% uniform band = calibrated; bars=mean of 3 seeds, band N=600)",
                 fontsize=13)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/sbc_rank_histograms.{ext}", bbox_inches="tight", dpi=130)
    import json
    json.dump(summary, open(f"{OUT}/sbc_summary.json", "w"), indent=2)
    print(f"wrote {OUT}/sbc_rank_histograms.{{png,pdf}} + sbc_summary.json")


if __name__ == "__main__":
    main()
