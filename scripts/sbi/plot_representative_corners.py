#!/usr/bin/env python3
"""Representative flat-local L1 corners: overlay the 4 arms at the TYPICAL fiducial patch
(perm16/patch23, auto FoM3 = median) and, for contrast, the earlier favorable one (perm0/patch90)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
from getdist import MCSamples, plots
HERE = "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi"
RC = HERE + "/results/exploratory/flatsky_cross_2026_06/representative_corner"
OUT = HERE + "/results/exploratory/flatsky_cross_2026_06/figs"
ARMS = [("flat_none", "auto-only", "#555555"), ("flat_conv", "+conv", "#1f77b4"),
        ("flat_product", "+product", "#2ca02c"), ("flat_both", "+both", "#d62728")]
NAMES = ["Om", "s8", "w0"]; LAB = [r"\Omega_m", r"\sigma_8", "w_0"]
TRUTH = {"Om": 0.26, "s8": 0.84, "w0": -1.0}
RANGES = {"Om": (0.1, 0.5), "s8": (0.5, 1.25), "w0": (-1.8, -0.2)}

for tag, title in [("typical", "TYPICAL patch (perm16/patch23, auto FoM3=median 2405)"),
                   ("favorable", "favorable patch (perm0/patch90) — what the single-obs corner used")]:
    mcs, leg, cols = [], [], []
    for arm, lab, col in ARMS:
        z = np.load(f"{RC}/{arm}/corner_samples.npz")
        s = z[tag][:, :3]
        mcs.append(MCSamples(samples=s, names=NAMES, labels=LAB, ranges=RANGES, label=lab,
                             settings={"smooth_scale_2D": 0.35, "smooth_scale_1D": 0.35}))
        leg.append(lab); cols.append(col)
    g = plots.get_subplot_plotter(width_inch=7.5)
    g.settings.legend_fontsize = 12; g.settings.axes_labelsize = 14
    g.triangle_plot(mcs, NAMES, filled=True, legend_labels=leg, colors=cols, contour_colors=cols,
                    markers=TRUTH)
    g.fig.suptitle(f"Flat-local L1 — {title}", y=1.02, fontsize=12)
    g.export(f"{OUT}/representative_corner_{tag}.png")
    g.export(f"{OUT}/representative_corner_{tag}.pdf")
    print(f"wrote representative_corner_{tag}.png")
