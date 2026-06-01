#!/usr/bin/env python3
"""L1 vs CNN overlay corners (split-matched 70/30, perm 0, 3 seeds pooled).

CNN = jaxili MAF NDE on CNN-VMIM (RealNVP companion) summaries — tf.data route,
      LEAKAGE-INFLATED absolute (~1.6x).
L1  = jaxili MAF NDE on wavelet-L1 datavector — CLEAN (disjoint 70/30).
Same NDE for both; only the compressor differs. Absolute comparison is therefore
apples-to-oranges (CNN inflated, L1 clean); the honest comparison needs the CNN
disjoint rerun. Shown because Andreas asked.
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
from getdist import MCSamples, plots

PB = "results/exploratory/definitive_comparison/phaseB_tfdata_2026_05_30"
POST = "results/exploratory/definitive_comparison/posteriors"
FIG = os.path.join(PB, "figures"); os.makedirs(FIG, exist_ok=True)
NAMES = ["Omega_m", "sigma_8", "w_0", "h_0", "n_s", "Omega_b"]
LABELS = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]
TRUTH = [0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493]
SEEDS = [41, 42, 43]


def pool_cnn(arm):  # arm in {autocross, autoonly}
    fs = [f"{PB}/posteriors/{arm}/{arm}_cs41/{arm}_cs41_s{s}_p0.npy" for s in SEEDS]
    fs = [f for f in fs if os.path.exists(f)]
    return np.concatenate([np.load(f) for f in fs], 0), len(fs)


def pool_l1(arm):  # arm in {autocross, autoonly}
    fs = [f"{POST}/l1_{arm}_split70/l1_{arm}_split70_s{s}_p0.npy" for s in SEEDS]
    fs = [f for f in fs if os.path.exists(f)]
    return np.concatenate([np.load(f) for f in fs], 0), len(fs)


def fom3(x):  # 1/sqrt(det C) on (Om, s8, w0)
    c = np.cov(x[:, :3], rowvar=False)
    return 1.0 / np.sqrt(np.linalg.det(c))


def overlay(arm, pretty):
    cnn, ncnn = pool_cnn(arm)
    l1, nl1 = pool_l1(arm)
    print(f"{arm}: CNN {cnn.shape} (n={ncnn}, FoM3={fom3(cnn):.0f})  "
          f"L1 {l1.shape} (n={nl1}, FoM3={fom3(l1):.0f})")
    s_l1 = MCSamples(samples=l1, names=NAMES, labels=LABELS, label="L1 (clean)")
    s_cnn = MCSamples(samples=cnn, names=NAMES, labels=LABELS, label="CNN (leak-inflated)")
    g = plots.get_subplot_plotter(width_inch=10)
    g.settings.legend_fontsize = 15
    g.settings.axes_fontsize = 11
    g.settings.axes_labelsize = 15
    g.triangle_plot(
        [s_l1, s_cnn], filled=True,
        contour_colors=["#1d3557", "#c1121f"],
        legend_labels=[f"L1 split70 (clean, n={nl1})",
                       f"CNN split70 (tf.data, leak-inflated, n={ncnn})"],
        markers={n: t for n, t in zip(NAMES, TRUTH)},
    )
    g.fig.suptitle(
        f"L1 vs CNN — {pretty}  |  jaxili MAF NDE (same for both), 70/30, perm0, 3 seeds\n"
        f"CNN FoM3={fom3(cnn):.0f} (inflated ~1.6x) vs L1 FoM3={fom3(l1):.0f} (clean) "
        "— absolute is apples-to-oranges",
        fontsize=11, y=1.02)
    out = os.path.join(FIG, f"l1_vs_cnn_{arm}.png")
    g.export(out)
    print("wrote", out)
    return out


overlay("autocross", "auto+cross (10ch)")
overlay("autoonly", "auto-only (4ch)")
