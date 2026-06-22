#!/usr/bin/env python3
"""Stratified (FoM3-tercile) TARP-DRP for the CNN with PROPER per-tercile 1-sigma bands.

Replaces the bogus shaded bands in tarp_resnet18_stratified (they came from the pipeline's degenerate
`ecp_bootstrap`, which resamples reference points only). Here, per tercile, we pool the 3 NDE seeds
(the reported posterior) and bootstrap that tercile's ~200 sightlines (16-84 pct = 1 sigma). Mean curve
is the bootstrap mean. CPU only. Same TARP convention as run_tarp_coverage.py.
"""
from pathlib import Path
import glob
import numpy as np
import tarp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/flatsky_cross_2026_06/cnn_phase/arch_sweep_2026_06_13/gate_c_resnet18"
OUT = Path("/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/flatsky_cross_2026_06/cnn_phase/calib_refine_2026_06/figs")
ALPHA = np.linspace(0.0, 1.0, 61)
NBOOT = 200
RNG = np.random.default_rng(0)
TCOL = {"LOW": "#1f77b4", "MID": "#ff7f0e", "HIGH": "#2ca02c"}
TLAB = {"LOW": "LOW FoM3 (widest)", "MID": "MID FoM3", "HIGH": "HIGH FoM3 (tightest)"}


def ecp_once(samp_NMD, theta_ND):
    s = np.transpose(samp_NMD, (1, 0, 2))
    ecp, alpha = tarp.get_tarp_coverage(s, theta_ND, references="random", norm=True)
    ecp = np.asarray(ecp); ecp = ecp.mean(0) if ecp.ndim == 2 else ecp
    return np.interp(ALPHA, np.asarray(alpha), ecp)


def tercile_pool(terc):
    S = []
    for seed in (41, 42, 43):
        g = glob.glob(f"{G}/dumps/resnet18_rnvp_{terc}/seed_{seed}/n*_m*/posterior_samples.npz")
        z = np.load(g[0]); S.append(z["samples"][:, :, :3])
    theta = np.load(g[0])["theta"][:, :3].astype(np.float32)   # same theta across seeds in a tercile
    samp = np.concatenate(S, axis=1).astype(np.float32)
    return samp, theta


def band(samp, theta):
    N = theta.shape[0]
    boot = np.array([ecp_once(samp[i := RNG.integers(0, N, N)], theta[i]) for _ in range(NBOOT)])
    mean = boot.mean(0); lo, hi = np.percentile(boot, [16, 84], 0)
    return mean, lo, hi, float(np.trapz(mean - ALPHA, ALPHA) * 2), N


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7, label="ideal (calibrated)")
    nets = {}
    for terc in ("LOW", "MID", "HIGH"):
        samp, theta = tercile_pool(terc)
        mean, lo, hi, net, N = band(samp, theta)
        nets[terc] = net
        ax.fill_between(ALPHA, lo, hi, color=TCOL[terc], alpha=0.18)
        ax.plot(ALPHA, mean, color=TCOL[terc], lw=2.2, label=f"{TLAB[terc]} (net {net:+.3f}, N={N})")
        print(f"{terc}: net {net:+.4f} N={N}", flush=True)
    ax.set_xlabel("credibility level"); ax.set_ylabel("expected coverage probability")
    ax.set_title("CNN TARP-DRP, FoM3-stratified (like-with-like)\n"
                 r"resnet18 + sbi_lens RealNVP  ·  proper 1$\sigma$ bands per tercile", fontsize=10)
    ax.legend(fontsize=8, loc="upper left"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"tarp_resnet18_stratified.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"stratified mean net {np.mean(list(nets.values())):+.4f}  -> {OUT}/tarp_resnet18_stratified.{{pdf,png}}")


if __name__ == "__main__":
    main()
