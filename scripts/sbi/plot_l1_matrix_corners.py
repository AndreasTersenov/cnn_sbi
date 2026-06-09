#!/usr/bin/env python3
"""Overlay corner plots for the flat-local L1 matrix (single-obs SANITY CHECK, pre-GATE-C).

Pools the 3 NDE seeds per arm and overlays auto-only / conv / product / both on the science
params (Omega_m, sigma8, w0), truth marked. These are SINGLE-OBS posteriors (obs perm0 patch90)
— a sanity check that contours are sensible and sit near truth, NOT the headline (that needs the
9000-obs population sweep + GATE C calibration). h0/Ob nuisances omitted (h0 has a known weak-
nuisance scaling quirk, consistent across arms).
"""
import os, glob, numpy as np
import matplotlib
matplotlib.use("Agg")
from getdist import MCSamples, plots

HERE = os.path.dirname(os.path.abspath(__file__))
D = HERE + "/results/exploratory/flatsky_cross_2026_06/l1_matrix"
OUT = HERE + "/results/exploratory/flatsky_cross_2026_06/figs"
os.makedirs(OUT, exist_ok=True)
TRUTH = {"Om": 0.26, "s8": 0.84, "w0": -1.0}
ARMS = [("none", "auto-only", "#444444"), ("conv", "+conv", "#1f77b4"),
        ("product", "+product", "#2ca02c"), ("both", "+both", "#d62728")]
NAMES = ["Om", "s8", "w0"]
LABELS = [r"\Omega_m", r"\sigma_8", "w_0"]
RANGES = {"Om": (0.1, 0.5), "s8": (0.5, 1.25), "w0": (-1.8, -0.2)}


def pooled(arm):
    fs = sorted(glob.glob(f"{D}/l1_{arm}_s*/posterior.npy"))
    s = np.concatenate([np.load(f)[:, :3] for f in fs], axis=0)  # Om, s8, w0
    return s, len(fs)


def main():
    mcs, leg, colors = [], [], []
    print("pooled posterior summaries (science params, 3 seeds):")
    for arm, lab, col in ARMS:
        s, n = pooled(arm)
        mcs.append(MCSamples(samples=s, names=NAMES, labels=LABELS,
                             ranges=RANGES, label=lab,
                             settings={"smooth_scale_2D": 0.3, "smooth_scale_1D": 0.3}))
        leg.append(lab); colors.append(col)
        sig = s.std(0)
        print(f"  {lab:10s} (n={n} seeds): "
              f"Om {s[:,0].mean():.3f}±{sig[0]:.3f} | s8 {s[:,1].mean():.3f}±{sig[1]:.3f} | "
              f"w0 {s[:,2].mean():.3f}±{sig[2]:.3f}")

    g = plots.get_subplot_plotter(width_inch=8)
    g.settings.legend_fontsize = 13
    g.settings.axes_fontsize = 11
    g.settings.axes_labelsize = 14
    g.triangle_plot(mcs, NAMES, filled=True, legend_labels=leg, colors=colors,
                    contour_colors=colors,
                    markers={"Om": TRUTH["Om"], "s8": TRUTH["s8"], "w0": TRUTH["w0"]})
    for ext in ("png", "pdf"):
        g.export(f"{OUT}/l1_matrix_corner_science.{ext}")
    print(f"\nwrote {OUT}/l1_matrix_corner_science.{{png,pdf}}")


if __name__ == "__main__":
    main()
