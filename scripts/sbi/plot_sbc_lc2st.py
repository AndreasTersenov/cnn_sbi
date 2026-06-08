#!/usr/bin/env python
"""Standard SBC rank-histogram + L-C2ST calibration plots (Phase D, 10°). CPU-only."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binom

AN = "results/exploratory/definitive_comparison_10deg/phase_c/analysis"
OUT = f"{AN}/figs"
os.makedirs(OUT, exist_ok=True)
ARMS = ["l1_auto_cross", "cnn_auto_cross", "l1_auto_only", "cnn_auto_only"]
LBL = {"l1_auto_cross": "L1 a+c", "cnn_auto_cross": "CNN a+c",
       "l1_auto_only": "L1 auto", "cnn_auto_only": "CNN auto"}
PLBL = [r"$\Omega_m$", r"$\sigma_8$", r"$w_0$", r"$h_0$", r"$n_s$", r"$\Omega_b$"]


def sbc():
    import json
    nb = 20
    fig, ax = plt.subplots(len(ARMS), 6, figsize=(15, 9), sharex=True)
    for r, arm in enumerate(ARMS):
        d = np.load(f"{AN}/sbc/{arm}/sbc_ranks.npz")
        ranks = d["ranks"]
        summ = json.load(open(f"{AN}/sbc/{arm}/sbc_summary.json"))
        N = ranks.shape[0]
        lo, hi = binom.ppf(0.005, N, 1 / nb), binom.ppf(0.995, N, 1 / nb)  # 99% uniform band
        for c in range(6):
            a = ax[r, c]
            a.axhspan(lo, hi, color="0.85", zorder=0)        # uniform 99% band
            a.axhline(N / nb, color="0.5", lw=0.8, ls="--")
            a.hist(ranks[:, c], bins=nb, range=(0, 1), color="#1f77b4" if "cnn" in arm else "#2ca02c",
                   alpha=0.8, edgecolor="k", lw=0.3)
            pk = ["Omega_m", "sigma_8", "w_0", "h_0", "n_s", "Omega_b"][c]
            ksp = summ[pk]["ks_pvalue"]
            a.set_title(f"{PLBL[c]}  (KS p={ksp:.2f})", fontsize=8.5,
                        color="darkred" if ksp < 0.05 else "black")
            a.tick_params(labelsize=6)
            if c == 0:
                a.set_ylabel(f"{LBL[arm]}\ncount", fontsize=9)
            if r == len(ARMS) - 1:
                a.set_xlabel("rank", fontsize=8)
    fig.suptitle("SBC rank histograms (400 val cosmologies, M=3000) — flat within the grey 99% "
                 "uniform band = calibrated", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/D5_sbc_ranks.png", dpi=130, bbox_inches="tight")
    fig.savefig(f"{OUT}/D5_sbc_ranks.pdf", bbox_inches="tight")
    print("wrote D5_sbc_ranks")


def lc2st():
    arms = ["cnn_auto_cross", "cnn_auto_only"]
    col = {"cnn_auto_cross": "#1f77b4", "cnn_auto_only": "#aec7e8"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for arm in arms:
        d = np.load(f"{AN}/lc2st/{arm}/lc2st_results.npz")
        p = d["p"]; T = d["T_obs"]; Tn = d["T_null"]
        # left: per-obs p-values (none below 0.05 = locally calibrated)
        ax[0].hist(p, bins=15, range=(0, 1), histtype="step", lw=2, color=col[arm],
                   label=f"{LBL[arm]} (median p={np.median(p):.2f}, {int((p<0.05).sum())}/30 reject)")
        # right: T_obs vs the permutation null (pooled), per arm
        ax[1].scatter(np.full(len(T), arm == "cnn_auto_only") + np.random.uniform(-0.08, 0.08, len(T)),
                      T, s=14, color=col[arm], alpha=0.7, zorder=3, label=LBL[arm])
        nperc = np.percentile(Tn.ravel(), [5, 50, 95])
        x = float(arm == "cnn_auto_only")
        ax[1].plot([x - 0.2, x + 0.2], [nperc[1], nperc[1]], color="k", lw=1.5)
        ax[1].add_patch(plt.Rectangle((x - 0.2, nperc[0]), 0.4, nperc[2] - nperc[0],
                                      color="0.8", zorder=1))
    ax[0].axvline(0.05, color="red", ls="--", lw=1, label="p=0.05")
    ax[0].set_xlabel("local-C2ST p-value (per fiducial obs)"); ax[0].set_ylabel("count")
    ax[0].set_title("L-C2ST p-values (CNN) — all > 0.05 → locally calibrated"); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.2)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["CNN a+c", "CNN auto"])
    ax[1].set_yscale("log")
    ax[1].set_ylabel("L-C2ST statistic T(x0)")
    ax[1].set_title("T_obs (points) vs permutation null (grey 5–95%, line=median)")
    ax[1].grid(alpha=0.2)
    fig.suptitle("L-C2ST local calibration at the fiducial (logreg) — 0/30 obs reject (vs 87% at 20°)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/D6_lc2st.png", dpi=130, bbox_inches="tight")
    fig.savefig(f"{OUT}/D6_lc2st.pdf", bbox_inches="tight")
    print("wrote D6_lc2st")


if __name__ == "__main__":
    sbc(); lc2st()
