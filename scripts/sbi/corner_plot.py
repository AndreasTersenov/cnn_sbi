#!/usr/bin/env python3
"""Headline corner from corner_resample.py output, with NO smoothing (raw
histograms via the `corner` package), at one typical patch.

Outputs (under <analysis>/corner_resample/figs):
  C1_corner_pooled        : L1 a+c vs CNN a+c, 6 params, pooled 3 seeds (no smoothing)
  C2_corner_perseed_<arm> : per-arm 3-seed overlay (tests whether marginal
                            "bumpiness" is seed-consistent = real structure, or
                            seed-varying = sampling noise)
  C3_nuisance_marginals   : 1D marginals of h0, ns, Ωb — L1 vs CNN, per seed
                            (the crisp side-by-side for "why is only CNN bumpy?")
CPU-only.
"""
import os, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

AN = "results/exploratory/definitive_comparison_10deg/phase_c/analysis"
RS = f"{AN}/corner_resample"
OUT = f"{RS}/figs"
os.makedirs(OUT, exist_ok=True)

LABELS = [r"$\Omega_m$", r"$\sigma_8$", r"$w_0$", r"$h_0$", r"$n_s$", r"$\Omega_b$"]
FIDUCIAL = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493])
LEVELS = (0.393, 0.864)        # 1σ, 2σ of the 2-D mass (Gaussian-equivalent)
SEEDS = [41, 42, 43]
SEEDCOL = {41: "#1b9e77", 42: "#d95f02", 43: "#7570b3"}
ARMCOL = {"l1_auto_cross": "#2ca02c", "cnn_auto_cross": "#1f77b4"}
ARMLBL = {"l1_auto_cross": "L1 a+c", "cnn_auto_cross": "CNN a+c"}


def load(arm):
    return np.load(f"{RS}/{arm}/corner_samples.npz")


def common_range(arrs, pad=0.05):
    """Union [0.5, 99.5]-pct range across all arrays, padded — so overlaid
    corners share identical panel limits."""
    stack = np.concatenate(arrs, 0)
    lo = np.percentile(stack, 0.5, axis=0); hi = np.percentile(stack, 99.5, axis=0)
    span = hi - lo
    return list(zip(lo - pad * span, hi + pad * span))


def corner_pooled():
    d1, d2 = load("l1_auto_cross"), load("cnn_auto_cross")
    p1, p2 = d1["pooled"], d2["pooled"]
    rng = common_range([p1, p2])
    fig = corner.corner(p1, labels=LABELS, color=ARMCOL["l1_auto_cross"], bins=45,
                        smooth=None, smooth1d=None, levels=LEVELS, range=rng,
                        plot_datapoints=False, plot_density=False, fill_contours=True,
                        hist_kwargs={"density": True}, contour_kwargs={"linewidths": 1.4})
    corner.corner(p2, fig=fig, color=ARMCOL["cnn_auto_cross"], bins=45,
                  smooth=None, smooth1d=None, levels=LEVELS, range=rng,
                  plot_datapoints=False, plot_density=False, fill_contours=True,
                  truths=FIDUCIAL, truth_color="k",
                  hist_kwargs={"density": True}, contour_kwargs={"linewidths": 1.4})
    # legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=ARMCOL[a], lw=3,
               label=f"{ARMLBL[a]} (FoM3={load(a)['fom3_pooled']:.0f})")
               for a in ("l1_auto_cross", "cnn_auto_cross")]
    fig.legend(handles=handles, loc="upper right", fontsize=13, frameon=False,
               bbox_to_anchor=(0.98, 0.98))
    fig.suptitle("Headline corner — typical patch (patch 76, perm 1; lat −25°), "
                 "3 seeds pooled, NO smoothing", fontsize=13, y=1.0)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/C1_corner_pooled.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig); print("wrote C1_corner_pooled")


def corner_perseed(arm):
    d = load(arm)
    arrs = [d[f"samples_seed{s}"] for s in SEEDS]
    rng = common_range(arrs)
    fig = None
    for s, a in zip(SEEDS, arrs):
        fig = corner.corner(a, fig=fig, labels=LABELS, color=SEEDCOL[s], bins=45,
                            smooth=None, smooth1d=None, levels=LEVELS, range=rng,
                            plot_datapoints=False, plot_density=False, fill_contours=False,
                            hist_kwargs={"density": True},
                            contour_kwargs={"linewidths": 1.3})
    # truth markers on the last
    corner.overplot_lines(fig, FIDUCIAL, color="k", lw=0.8, ls="--")
    from matplotlib.lines import Line2D
    fom = d["fom3_per_seed"]
    handles = [Line2D([0], [0], color=SEEDCOL[s], lw=3, label=f"seed {s} (FoM3={fom[i]:.0f})")
               for i, s in enumerate(SEEDS)]
    fig.legend(handles=handles, loc="upper right", fontsize=13, frameon=False,
               bbox_to_anchor=(0.98, 0.98))
    fig.suptitle(f"{ARMLBL[arm]} — per-seed posteriors (patch 76, perm 1), NO smoothing\n"
                 "bumps in same place across seeds = real structure; bumps that move = sampling noise",
                 fontsize=12, y=1.0)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/C2_corner_perseed_{arm}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig); print(f"wrote C2_corner_perseed_{arm}")


def nuisance_marginals():
    """1D marginals for h0, ns, Ωb (indices 3,4,5): rows=seed, cols=param,
    L1 vs CNN overlaid — the crisp test for 'only CNN is bumpy here'."""
    d1, d2 = load("l1_auto_cross"), load("cnn_auto_cross")
    idx = [3, 4, 5]
    fig, ax = plt.subplots(len(SEEDS), 3, figsize=(12, 9), sharex="col")
    for r, s in enumerate(SEEDS):
        a1, a2 = d1[f"samples_seed{s}"], d2[f"samples_seed{s}"]
        for c, j in enumerate(idx):
            x = ax[r, c]
            lo = min(a1[:, j].min(), a2[:, j].min()); hi = max(a1[:, j].max(), a2[:, j].max())
            bins = np.linspace(lo, hi, 60)
            x.hist(a1[:, j], bins=bins, density=True, histtype="step", lw=1.8,
                   color=ARMCOL["l1_auto_cross"], label="L1 a+c")
            x.hist(a2[:, j], bins=bins, density=True, histtype="step", lw=1.8,
                   color=ARMCOL["cnn_auto_cross"], label="CNN a+c")
            x.axvline(FIDUCIAL[j], color="k", lw=0.8, ls="--")
            x.grid(alpha=0.2)
            if r == 0:
                x.set_title(LABELS[c + 3] if False else LABELS[j])
            if c == 0:
                x.set_ylabel(f"seed {s}\ndensity", fontsize=10)
            if r == 0 and c == 2:
                x.legend(fontsize=9)
    fig.suptitle("Nuisance-parameter 1D marginals (raw, 60 bins): h0, ns, Ωb — "
                 "L1 vs CNN per seed.\nFlat priors → broad, ragged marginals; "
                 "compare whether ragged structure repeats across seeds.", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/C3_nuisance_marginals.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig); print("wrote C3_nuisance_marginals")


if __name__ == "__main__":
    corner_pooled()
    corner_perseed("l1_auto_cross")
    corner_perseed("cnn_auto_cross")
    nuisance_marginals()
