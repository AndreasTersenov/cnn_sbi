#!/usr/bin/env python3
"""Per-patch L1-vs-CNN corners (no smoothing) from corner_multiobs.py output, plus
a patch-variation overlay (all patches, one method) for Ωm/σ8/w0. CPU-only."""
import os, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import corner

RS = "results/exploratory/definitive_comparison_10deg/phase_c/analysis/corner_resample"
OUT = f"{RS}/figs_multiobs"
os.makedirs(OUT, exist_ok=True)
LABELS = [r"$\Omega_m$", r"$\sigma_8$", r"$w_0$", r"$h_0$", r"$n_s$", r"$\Omega_b$"]
FIDUCIAL = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493])
LEVELS = (0.393, 0.864)
ARMCOL = {"l1_auto_cross": "#2ca02c", "cnn_auto_cross": "#1f77b4"}
ARMLBL = {"l1_auto_cross": "L1 a+c", "cnn_auto_cross": "CNN a+c"}
# patch -> latitude (from the geometry grid, perm 1) for captions
import csv
LAT = {}
for r in csv.DictReader(open(f"{RS}/../geometry/cnn_auto_cross/per_patch_grid.csv")):
    if int(r["perm"]) == 1:
        LAT[int(r["patch"])] = float(r["lat"])


def load(arm):
    d = np.load(f"{RS}/multiobs_samples_{arm}.npz")
    fom = json.load(open(f"{RS}/multiobs_fom_{arm}.json"))
    obs = [tuple(x) for x in d["obs"]]
    return d, fom, obs


def common_range(arrs, idx=None, pad=0.05):
    stack = np.concatenate(arrs, 0)
    if idx is not None:
        stack = stack[:, idx]
    lo = np.percentile(stack, 0.5, 0); hi = np.percentile(stack, 99.5, 0)
    span = hi - lo
    return list(zip(lo - pad * span, hi + pad * span))


def per_obs():
    dl, foml, obs = load("l1_auto_cross")
    dc, fomc, _ = load("cnn_auto_cross")
    for (pp, mm) in obs:
        p1 = dl[f"pooled_{pp}_{mm}"]; p2 = dc[f"pooled_{pp}_{mm}"]
        rng = common_range([p1, p2])
        fig = corner.corner(p1, labels=LABELS, color=ARMCOL["l1_auto_cross"], bins=45,
                            smooth=None, smooth1d=None, levels=LEVELS, range=rng,
                            plot_datapoints=False, plot_density=False, fill_contours=True,
                            hist_kwargs={"density": True})
        corner.corner(p2, fig=fig, color=ARMCOL["cnn_auto_cross"], bins=45,
                      smooth=None, smooth1d=None, levels=LEVELS, range=rng,
                      plot_datapoints=False, plot_density=False, fill_contours=True,
                      truths=FIDUCIAL, truth_color="k", hist_kwargs={"density": True})
        handles = [Line2D([0], [0], color=ARMCOL["l1_auto_cross"], lw=3,
                          label=f"L1 a+c (FoM3={foml[f'{pp}_{mm}']:.0f})"),
                   Line2D([0], [0], color=ARMCOL["cnn_auto_cross"], lw=3,
                          label=f"CNN a+c (FoM3={fomc[f'{pp}_{mm}']:.0f})")]
        fig.legend(handles=handles, loc="upper right", fontsize=13, frameon=False,
                   bbox_to_anchor=(0.98, 0.98))
        fig.suptitle(f"patch {pp}, perm {mm} (lat {LAT.get(pp, float('nan')):+.0f}°) — "
                     "3 seeds pooled, NO smoothing", fontsize=13, y=1.0)
        for ext in ("png", "pdf"):
            fig.savefig(f"{OUT}/corner_patch{pp}_perm{mm}.{ext}", dpi=140, bbox_inches="tight")
        plt.close(fig); print(f"wrote corner_patch{pp}_perm{mm}")


def variation_overlay():
    """All patches overlaid for one method, Ωm/σ8/w0 only — shows patch-to-patch
    drift of the posterior (center + size)."""
    idx = [0, 1, 2]
    L3 = [LABELS[i] for i in idx]
    cmap = plt.cm.viridis
    for arm in ("cnn_auto_cross", "l1_auto_cross"):
        d, fom, obs = load(arm)
        # add patch 76 (sampled separately by corner_resample.py)
        d76 = np.load(f"{RS}/{arm}/corner_samples.npz")
        samp = {(pp, mm): d[f"pooled_{pp}_{mm}"] for (pp, mm) in obs}
        f3 = dict(fom)
        samp[(76, 1)] = d76["pooled"]; f3["76_1"] = float(d76["fom3_pooled"])
        obs = sorted(samp.keys(), key=lambda k: f3[f"{k[0]}_{k[1]}"])  # loose -> tight
        arrs = [samp[(pp, mm)][:, idx] for (pp, mm) in obs]
        rng = common_range(arrs)
        fig = None
        cols = [cmap(t) for t in np.linspace(0.1, 0.9, len(obs))]
        for c, (pp, mm) in zip(cols, obs):
            fig = corner.corner(samp[(pp, mm)][:, idx], fig=fig, labels=L3, color=c,
                                bins=40, smooth=None, smooth1d=None, levels=(0.864,), range=rng,
                                plot_datapoints=False, plot_density=False, fill_contours=False,
                                hist_kwargs={"density": True})
        corner.overplot_lines(fig, FIDUCIAL[idx], color="k", lw=0.8, ls="--")
        handles = [Line2D([0], [0], color=c, lw=3,
                          label=f"patch {pp} (lat {LAT.get(pp, float('nan')):+.0f}°, FoM3={f3[f'{pp}_{mm}']:.0f})")
                   for c, (pp, mm) in zip(cols, obs)]
        fig.legend(handles=handles, loc="upper right", fontsize=11, frameon=False,
                   bbox_to_anchor=(0.98, 0.98))
        fig.suptitle(f"{ARMLBL[arm]} — Ωm/σ8/w0 across patches (2σ contour, NO smoothing)\n"
                     "shows realization/geometry scatter of the posterior", fontsize=12, y=1.0)
        for ext in ("png", "pdf"):
            fig.savefig(f"{OUT}/variation_{arm}.{ext}", dpi=140, bbox_inches="tight")
        plt.close(fig); print(f"wrote variation_{arm}")


if __name__ == "__main__":
    per_obs()
    variation_overlay()
