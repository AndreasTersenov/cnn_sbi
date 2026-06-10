#!/usr/bin/env python3
"""Stitched paper figure: the de-leaked flat-local L1-vs-CNN result in one panel (A&A double-col).

Main: 3-param (Om,s8,w0) filled-contour overlay of CNN vs L1 +product at the typical patch
(the money plot — L1 tighter ⇒ it reads the physical cross, CNN doesn't).
Inset (upper-right empty triangle): FoM3 bars CNN vs L1, all 4 arms (auto-only tie; cross
arms = L1 rises, CNN flat). Grayscale-safe: contour linestyles (L1 solid / CNN dashed) +
bar hatching (L1 hatched). Output: vector PDF + PNG.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from getdist import MCSamples, plots

STYLE = "/home/tersenov/.claude/skills/figure-polish/style/aa.mplstyle"
FC = Path("/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/flatsky_cross_2026_06")
OUT = FC / "cnn_phase/figs"
AA_DOUBLE = 7.087   # A&A double-column width (180 mm), inches
ARMS = ["none", "conv", "product", "both"]
ARMLAB = {"none": "auto", "conv": "+conv", "product": "+prod", "both": "+both"}
C_CNN, C_L1 = "#0072B2", "#D55E00"   # Wong colourblind-safe (blue / vermillion)


def med(base, op):
    f = FC / base / f"flat_{op}" / "median_summary.json"
    return json.load(open(f)) if f.exists() else None


def main():
    plt.style.use(STYLE)
    plt.rcParams["figure.constrained_layout.use"] = False   # getdist + manual inset use fixed positions
    plt.rcParams["savefig.bbox"] = None   # keep the full 7.087in (180mm) canvas; tight crop undershoots

    idx = [0, 1, 2]
    cnn = np.load(OUT.parent / "representative_corner/flat_product/corner_samples.npz")["typical"][:, idx]
    l1 = np.load(FC / "representative_corner/flat_product/corner_samples.npz")["typical"][:, idx]
    names = ["Om", "s8", "w0"]
    labels = [r"\Omega_m", r"\sigma_8", "w_0"]
    truth = {"Om": 0.26, "s8": 0.84, "w0": -1.0}
    mc_l1 = MCSamples(samples=l1, names=names, labels=labels)
    mc_cnn = MCSamples(samples=cnn, names=names, labels=labels)

    g = plots.get_subplot_plotter(width_inch=AA_DOUBLE)
    g.settings.axes_fontsize = 11
    g.settings.axes_labelsize = 15
    g.settings.linewidth_contour = 1.5
    # Draw CNN first (underneath), L1 last (on top) so the tighter L1 contour sits over the
    # larger CNN one. L1 solid / CNN dashed -> distinguishable in grayscale too.
    g.triangle_plot([mc_cnn, mc_l1], params=names, filled=True,
                    contour_colors=[C_CNN, C_L1], contour_ls=["--", "-"], contour_lws=[1.5, 1.5],
                    markers=truth)
    fig = g.fig

    # Suppress getdist's auto legend (samples0/samples1) — the inset legend below is descriptive
    # and shares the contour colours, so it serves both the bars and the contours.
    for _leg in list(fig.legends):
        _leg.remove()
    for _ax in fig.axes:
        _l = _ax.get_legend()
        if _l is not None:
            _l.remove()

    # --- FoM3 bars inset in the empty upper-right triangle ---
    cnn_f = [med("cnn_phase/population_sweep", op)["fom3"] for op in ARMS]
    l1_f = [med("population_sweep", op)["fom3"] for op in ARMS]
    ax = fig.add_axes([0.605, 0.715, 0.355, 0.200])   # high in the upper-right, clear of the triangle
    x = np.arange(len(ARMS)); w = 0.4
    ax.bar(x - w/2, cnn_f, w, color=C_CNN, edgecolor="black", linewidth=0.6,
           label=r"CNN (VMIM)")
    ax.bar(x + w/2, l1_f, w, color=C_L1, edgecolor="black", linewidth=0.6, hatch="///",
           label=r"L1 (wavelet $\ell_1$)")
    ax.axhline(cnn_f[0], color="0.45", ls=":", lw=0.9, zorder=0)   # CNN auto-only reference
    ax.set_xticks(x); ax.set_xticklabels([ARMLAB[o] for o in ARMS], fontsize=8.5)
    ax.set_ylabel(r"FoM$_3$", fontsize=10); ax.set_ylim(0, 3300)
    ax.tick_params(labelsize=8, top=False, right=False)
    ax.set_title(r"pooled 9000-obs median", fontsize=9, pad=3)
    # legend in the empty (1,2) cell below the inset, clear of all panels
    ax.legend(fontsize=9, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.78), handlelength=1.3, handletextpad=0.5)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"stitched_cnn_vs_l1.{ext}")
    print(f"wrote {OUT}/stitched_cnn_vs_l1.{{pdf,png}}")


if __name__ == "__main__":
    main()
