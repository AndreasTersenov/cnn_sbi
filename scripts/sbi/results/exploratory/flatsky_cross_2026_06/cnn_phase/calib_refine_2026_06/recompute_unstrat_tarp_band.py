#!/usr/bin/env python3
"""Un-stratified TARP-DRP for the best CNN arm with a PROPER uncertainty band.

The saved gate curves' `ecp_bootstrap` resamples the random reference points only (per-bin std ~1e-4),
so it badly understates the real uncertainty, which is set by the finite number of validation
sightlines (N=600 -> binomial SE ~0.02 at credibility 0.5). Here we recompute the coverage on the
*reported* posterior (3 NDE seeds pooled, 6000 samples/obs) and bootstrap the 600 sightlines ourselves
(B resamples, fresh random references each), interpolating each onto a common alpha grid. CPU only.

Convention matches run_tarp_coverage.py: first 3 params (Om, s8, w0); samples transposed to (M,N,D);
references='random'; norm=True.
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
B = 200
RNG = np.random.default_rng(0)


def ecp_once(samples_NMD, theta_ND):
    """samples (N,M,3), theta (N,3) -> ecp interpolated onto ALPHA (one random-reference draw)."""
    s_tarp = np.transpose(samples_NMD, (1, 0, 2))           # (M,N,3) as tarp expects
    ecp, alpha = tarp.get_tarp_coverage(s_tarp, theta_ND, references="random", norm=True)
    ecp = np.asarray(ecp)
    if ecp.ndim == 2:
        ecp = ecp.mean(0)
    return np.interp(ALPHA, np.asarray(alpha), ecp)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dumps = sorted(glob.glob(f"{G}/dumps_all/resnet18_all/seed_*/n*_m*/posterior_samples.npz"))
    arrs = [np.load(d) for d in dumps]
    theta = arrs[0]["theta"][:, :3].astype(np.float32)       # (600,3); same θ across seeds
    samp = np.concatenate([a["samples"][:, :, :3] for a in arrs], axis=1).astype(np.float32)  # (600, 3*M, 3)
    N = theta.shape[0]
    print(f"pooled posterior: {samp.shape[1]} samples/obs over {len(arrs)} seeds, N={N} sightlines", flush=True)

    boot = np.empty((B, ALPHA.size))
    for b in range(B):
        idx = RNG.integers(0, N, size=N)                     # resample sightlines w/ replacement
        boot[b] = ecp_once(samp[idx], theta[idx])
        if (b + 1) % 50 == 0:
            print(f"  bootstrap {b+1}/{B}", flush=True)
    mean = boot.mean(0)
    lo68, hi68 = np.percentile(boot, [16, 84], 0)
    lo95, hi95 = np.percentile(boot, [2.5, 97.5], 0)
    net = float(np.trapz(mean - ALPHA, ALPHA) * 2)
    se_mid = float(boot[:, np.argmin(np.abs(ALPHA - 0.5))].std())
    print(f"net {net:+.4f} | recomputed SE@0.5 = {se_mid:.4f} (binomial expectation ~{np.sqrt(0.25/N):.4f})")

    np.savez(OUT.parent / "unstrat_tarp_proper_band.npz", alpha=ALPHA, mean=mean,
             lo68=lo68, hi68=hi68, lo95=lo95, hi95=hi95, net=net)

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7, label="ideal (calibrated)")
    ax.fill_between(ALPHA, lo95, hi95, color="#1f77b4", alpha=0.16, label="95% band (sightline bootstrap)")
    ax.fill_between(ALPHA, lo68, hi68, color="#1f77b4", alpha=0.32, label="68% band (sightline bootstrap)")
    ax.plot(ALPHA, mean, color="#1f77b4", lw=2.2, label=f"CNN, all val obs (net {net:+.3f})")
    ax.set_xlabel("credibility level"); ax.set_ylabel("expected coverage probability")
    ax.set_title("CNN TARP-DRP, un-stratified (all val obs)\n"
                 f"resnet18 + sbi_lens RealNVP  ·  proper band, N={N} sightlines", fontsize=10)
    ax.legend(fontsize=8, loc="upper left"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"tarp_resnet18_unstratified_properband.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/tarp_resnet18_unstratified_properband.{{pdf,png}}")


if __name__ == "__main__":
    main()
