#!/usr/bin/env python3
"""Honest CNN vs L1(auto+product) calibration: un-stratified TARP-DRP with a PROPER 1-sigma band,
plus SBC rank-std, both recomputed consistently from the saved posterior dumps.

The pipeline's saved `ecp_bootstrap` resamples only the random reference points (per-bin std ~1e-4),
so its band is ~200x too small. Here, for each arm we pool the 3 NDE seeds (the reported posterior),
recompute the un-stratified ECP, and bootstrap the 600 validation sightlines ourselves (B resamples,
fresh random references each), reporting the 16-84 percentile = 1-sigma band. SBC std is computed by
pooling each arm's stratified dumps (LOW+MID+HIGH) per seed and averaging std over seeds. CPU only.
Convention matches run_tarp_coverage.py: first 3 params; samples -> (M,N,3); references='random'; norm=True.
"""
from pathlib import Path
import glob
import numpy as np
import tarp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = f"/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/flatsky_cross_2026_06"
OUT = Path(f"{B}/cnn_phase/calib_refine_2026_06/figs")
ALPHA = np.linspace(0.0, 1.0, 61)
NBOOT = 200
RNG = np.random.default_rng(0)

ARMS = {
    "CNN": dict(
        color="#1f77b4", label="CNN auto-only (resnet18+RealNVP)",
        dumps_all=f"{B}/cnn_phase/arch_sweep_2026_06_13/gate_c_resnet18/dumps_all/resnet18_all",
        strat=f"{B}/cnn_phase/arch_sweep_2026_06_13/gate_c_resnet18/dumps/resnet18_rnvp"),
    "L1product": dict(
        color="#d62728", label="L1 auto+product (MAF)",
        dumps_all=f"{B}/gate_c/tarp_drp/dumps_all/flat_product_all",
        strat=f"{B}/gate_c/tarp_drp/dumps/flat_product"),
}


def ecp_once(samp_NMD, theta_ND):
    s = np.transpose(samp_NMD, (1, 0, 2))
    ecp, alpha = tarp.get_tarp_coverage(s, theta_ND, references="random", norm=True)
    ecp = np.asarray(ecp); ecp = ecp.mean(0) if ecp.ndim == 2 else ecp
    return np.interp(ALPHA, np.asarray(alpha), ecp)


def pooled_unstrat(dumps_all):
    dd = sorted(glob.glob(f"{dumps_all}/seed_*/n*_m*/posterior_samples.npz"))
    arrs = [np.load(d) for d in dd]
    theta = arrs[0]["theta"][:, :3].astype(np.float32)
    samp = np.concatenate([a["samples"][:, :, :3] for a in arrs], axis=1).astype(np.float32)
    return samp, theta, len(arrs)


def tarp_band(samp, theta):
    N = theta.shape[0]
    boot = np.empty((NBOOT, ALPHA.size))
    for b in range(NBOOT):
        idx = RNG.integers(0, N, size=N)
        boot[b] = ecp_once(samp[idx], theta[idx])
    mean = boot.mean(0)
    lo, hi = np.percentile(boot, [16, 84], 0)              # 1-sigma
    net = float(np.trapz(mean - ALPHA, ALPHA) * 2)
    return mean, lo, hi, net, float(boot[:, np.argmin(np.abs(ALPHA - 0.5))].std())


def sbc_std(strat_prefix):
    """Pool LOW/MID/HIGH per seed -> ranks -> std per param, averaged over seeds."""
    per = {p: [] for p in range(3)}
    for s in (41, 42, 43):
        S, T = [], []
        for t in ("LOW", "MID", "HIGH"):
            g = glob.glob(f"{strat_prefix}_{t}/seed_{s}/n*_m*/posterior_samples.npz")
            if g:
                z = np.load(g[0]); S.append(z["samples"]); T.append(z["theta"])
        S = np.concatenate(S, 0); T = np.concatenate(T, 0)
        for p in range(3):
            r = (S[:, :, p] < T[:, p, None]).mean(1)
            per[p].append(float(np.std(r)))
    return [float(np.mean(per[p])) for p in range(3)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, cfg in ARMS.items():
        samp, theta, nseed = pooled_unstrat(cfg["dumps_all"])
        mean, lo, hi, net, se = tarp_band(samp, theta)
        std = sbc_std(cfg["strat"])
        res[name] = dict(mean=mean, lo=lo, hi=hi, net=net, se=se, std=std,
                         color=cfg["color"], label=cfg["label"], n=theta.shape[0])
        print(f"{name:10s} | TARP net {net:+.4f} (SE@0.5 {se:.4f}, N={theta.shape[0]}, {nseed} seeds) | "
              f"SBC std {std[0]:.3f}/{std[1]:.3f}/{std[2]:.3f}")

    np.savez(OUT.parent / "tarp_compare_cnn_l1.npz",
             alpha=ALPHA, **{f"{k}_{q}": res[k][q] for k in res for q in ("mean", "lo", "hi")},
             cnn_net=res["CNN"]["net"], l1_net=res["L1product"]["net"])

    # individual 1-sigma plots
    for name in res:
        r = res[name]
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7, label="ideal (calibrated)")
        ax.fill_between(ALPHA, r["lo"], r["hi"], color=r["color"], alpha=0.28, label=r"1$\sigma$ (sightline bootstrap)")
        ax.plot(ALPHA, r["mean"], color=r["color"], lw=2.2, label=f"{r['label']} (net {r['net']:+.3f})")
        ax.set_xlabel("credibility level"); ax.set_ylabel("expected coverage probability")
        ax.set_title(f"TARP-DRP, un-stratified (N={r['n']} sightlines)", fontsize=10)
        ax.legend(fontsize=8, loc="upper left"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal"); fig.tight_layout()
        tag = "resnet18" if name == "CNN" else "l1product"
        for e in ("pdf", "png"):
            fig.savefig(OUT / f"tarp_{tag}_unstratified_1sigma.{e}", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # overlay
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7, label="ideal (calibrated)")
    for name in ("L1product", "CNN"):
        r = res[name]
        ax.fill_between(ALPHA, r["lo"], r["hi"], color=r["color"], alpha=0.22)
        ax.plot(ALPHA, r["mean"], color=r["color"], lw=2.2,
                label=f"{r['label']}  (net {r['net']:+.3f})")
    ax.set_xlabel("credibility level"); ax.set_ylabel("expected coverage probability")
    ax.set_title(r"CNN vs L1+product TARP-DRP, un-stratified, 1$\sigma$ bands"
                 "\n(+ = conservative / over-covers; − = over-confident)", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper left"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"tarp_cnn_vs_l1_unstratified_1sigma.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/tarp_cnn_vs_l1_unstratified_1sigma.{{pdf,png}} (+ per-arm)")


if __name__ == "__main__":
    main()
