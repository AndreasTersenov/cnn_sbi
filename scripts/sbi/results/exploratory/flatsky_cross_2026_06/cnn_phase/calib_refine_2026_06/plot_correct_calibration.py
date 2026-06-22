#!/usr/bin/env python3
"""The *correct* (FoM3-stratified) calibration plots for the best CNN arm (resnet18 + sbi_lens RealNVP).

Why "correct": the repo's standing TARP figure (tarp_resnet18_rnvp_dim3) is built from the UN-stratified
curve (all 600 val points pooled), which mixes posteriors whose size varies ~12x and reads a spurious
net +0.06 over-coverage (a known TARP heterogeneity effect — exactly what FoM3-stratification controls).
This script plots the like-with-like stratified gate instead:
  (1) TARP-DRP, the three FoM3 terciles (LOW/MID/HIGH) each averaged over NDE seeds 41/42/43, vs the
      diagonal, with the un-stratified pooled curve kept as a faint dashed reference (nothing hidden).
  (2) SBC rank histograms (Om, s8, w0), per-seed density averaged over seeds, with the 99% binomial band.
All read from the existing GATE-C dumps/curves — NO GPU, no retraining.
"""
from pathlib import Path
import glob
import numpy as np
from scipy.stats import binom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/flatsky_cross_2026_06/cnn_phase/arch_sweep_2026_06_13/gate_c_resnet18"
OUT = Path("/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/flatsky_cross_2026_06/cnn_phase/calib_refine_2026_06/figs")
SEEDS = [41, 42, 43]
PARAMS = [r"$\Omega_m$", r"$\sigma_8$", r"$w_0$"]
NB = 20
N_INDEP = 600  # points per seed (200 per tercile x 3 terciles)
TCOL = {"LOW": "#1f77b4", "MID": "#ff7f0e", "HIGH": "#2ca02c"}
TLAB = {"LOW": "LOW FoM3 (widest)", "MID": "MID FoM3", "HIGH": "HIGH FoM3 (tightest)"}


def net_of(alpha, ecp_mean):
    return float(np.trapz(ecp_mean - alpha, alpha) * 2)


def load_tercile(terc):
    """Stack ecp_bootstrap over the 3 seed curves -> mean + 16/84 band on a shared alpha grid."""
    ecps, alpha = [], None
    for f in sorted(glob.glob(f"{G}/curves/tarp_curve_resnet18_rnvp_{terc}_seed*_dim3.npz")):
        d = np.load(f)
        alpha = d["alpha"]
        ecps.append(np.asarray(d["ecp_bootstrap"]))   # (n_boot, n_alpha)
    stack = np.concatenate(ecps, 0)                    # (3*n_boot, n_alpha)
    return alpha, stack.mean(0), np.percentile(stack, 16, 0), np.percentile(stack, 84, 0)


def load_unstrat():
    ecps, alpha = [], None
    for f in sorted(glob.glob(f"{G}/all/curves/*resnet18_all*dim3*.npz")):
        d = np.load(f)
        alpha = d["alpha"]
        ecps.append(np.asarray(d["ecp_bootstrap"]).mean(0))
    e = np.array(ecps)
    return alpha, e.mean(0)


def load_unstrat_boot():
    """Pool the full bootstrap arrays over the 3 seed curves -> proper band (bootstrap + seed variance)."""
    ecps, alpha = [], None
    for f in sorted(glob.glob(f"{G}/all/curves/*resnet18_all*dim3*.npz")):
        d = np.load(f)
        alpha = d["alpha"]
        ecps.append(np.asarray(d["ecp_bootstrap"]))     # (n_boot, n_alpha)
    stack = np.concatenate(ecps, 0)                      # (3*n_boot, n_alpha)
    return (alpha, stack.mean(0),
            np.percentile(stack, 16, 0), np.percentile(stack, 84, 0),     # 68%
            np.percentile(stack, 2.5, 0), np.percentile(stack, 97.5, 0))  # 95%


def sbc_density_per_seed():
    """Per-seed rank density (pool LOW/MID/HIGH within a seed), then average density over seeds."""
    per_seed = {p: [] for p in range(3)}
    for s in SEEDS:
        samp, th = [], []
        for t in ["LOW", "MID", "HIGH"]:
            g = glob.glob(f"{G}/dumps/resnet18_rnvp_{t}/seed_{s}/n*_m*/posterior_samples.npz")
            if g:
                z = np.load(g[0]); samp.append(z["samples"]); th.append(z["theta"])
        S = np.concatenate(samp, 0); T = np.concatenate(th, 0)
        ranks = np.stack([(S[:, :, p] < T[:, p, None]).mean(1) for p in range(3)], 1)  # (Npts,3)
        for p in range(3):
            per_seed[p].append(ranks[:, p])
    return per_seed


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ----- (1) TARP-DRP, FoM3-stratified -----
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7, label="ideal (calibrated)")
    nets = {}
    for terc in ["LOW", "MID", "HIGH"]:
        a, m, lo, hi = load_tercile(terc)
        nets[terc] = net_of(a, m)
        ax.fill_between(a, lo, hi, color=TCOL[terc], alpha=0.18)
        ax.plot(a, m, color=TCOL[terc], lw=2.2, label=f"{TLAB[terc]} (net {nets[terc]:+.3f})")
    au, mu = load_unstrat()
    net_u = net_of(au, mu)
    ax.plot(au, mu, color="0.45", lw=1.6, ls=":", label=f"un-stratified pooled (net {net_u:+.3f})")
    strat_net = float(np.mean(list(nets.values())))
    ax.set_xlabel("credibility level"); ax.set_ylabel("expected coverage probability")
    ax.set_title("CNN TARP-DRP, FoM3-stratified (like-with-like)\n"
                 f"stratified net {strat_net:+.3f}  ·  resnet18 + sbi_lens RealNVP", fontsize=10)
    ax.legend(fontsize=8, loc="upper left"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"tarp_resnet18_stratified.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ----- (1b) TARP-DRP, UN-stratified (all 600 pts) with proper bootstrap band -----
    ab, mb, lo68, hi68, lo95, hi95 = load_unstrat_boot()
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7, label="ideal (calibrated)")
    ax.fill_between(ab, lo95, hi95, color="#1f77b4", alpha=0.15, label="95% bootstrap band")
    ax.fill_between(ab, lo68, hi68, color="#1f77b4", alpha=0.30, label="68% bootstrap band")
    ax.plot(ab, mb, color="#1f77b4", lw=2.2, label=f"CNN, all val obs (net {net_of(ab, mb):+.3f})")
    ax.set_xlabel("credibility level"); ax.set_ylabel("expected coverage probability")
    ax.set_title("CNN TARP-DRP, un-stratified (all val obs)\n"
                 "resnet18 + sbi_lens RealNVP  ·  band = pooled bootstrap (3 NDE seeds)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"tarp_resnet18_unstratified.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ----- (2) SBC rank histograms with 99% band -----
    per_seed = sbc_density_per_seed()
    bw = 1.0 / NB
    band_lo = binom.ppf(0.005, N_INDEP, 1 / NB) / (N_INDEP * bw)
    band_hi = binom.ppf(0.995, N_INDEP, 1 / NB) / (N_INDEP * bw)
    centers = (np.arange(NB) + 0.5) / NB
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    stds = []
    for p in range(3):
        ax = axes[p]
        dens = np.mean([np.histogram(r, bins=NB, range=(0, 1), density=True)[0]
                        for r in per_seed[p]], 0)
        std_p = float(np.mean([np.std(r) for r in per_seed[p]]))
        stds.append(std_p)
        nout = int(np.sum((dens < band_lo) | (dens > band_hi)))
        ax.axhspan(band_lo, band_hi, color="0.85", zorder=0,
                   label="99% uniform band" if p == 0 else None)
        ax.bar(centers, dens, width=bw, color="#1f77b4", alpha=0.78, edgecolor="white", linewidth=0.4)
        ax.set_title(f"SBC {PARAMS[p]}  (std {std_p:.3f}; {nout}/{NB} out)", fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, max(dens.max() * 1.15, band_hi * 1.3))
        ax.set_xlabel("posterior rank of truth")
        if p == 0:
            ax.set_ylabel("density"); ax.legend(fontsize=8)
    fig.suptitle("CNN SBC rank histograms (resnet18 + sbi_lens RealNVP) — flat in band = marginals calibrated; "
                 "ideal std 0.289", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"sbc_resnet18_correct.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("TARP per-tercile net:", {k: round(v, 4) for k, v in nets.items()},
          "| stratified mean", round(strat_net, 4), "| un-stratified", round(net_u, 4))
    print("SBC rank-std (Om/s8/w0):", [round(s, 4) for s in stds])
    print(f"wrote {OUT}/tarp_resnet18_stratified.{{pdf,png}} and sbc_resnet18_correct.{{pdf,png}}")


if __name__ == "__main__":
    main()
