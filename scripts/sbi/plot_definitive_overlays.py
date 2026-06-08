#!/usr/bin/env python3
"""Grouped getdist corner overlays for the definitive L1-vs-CNN comparison.

Each group is a small (<=3) set of posteriors chosen to make one comparison
legible. Each arm = the 3 NDE seeds POOLED at a fixed perm (the declared
"3-seed pooled" unit). Default params = the 3 science params (Ωm, σ8, w0);
the headline group also gets a full 6-param version.
"""
import glob, os
import numpy as np
from getdist import MCSamples, plots

DC = "results/exploratory/definitive_comparison"
OUT = os.path.join(DC, "PHASE_C_2026_05_31", "overlays"); os.makedirs(OUT, exist_ok=True)

NAMES6 = ["Om", "s8", "w0", "h0", "ns", "Ob"]
LABS6 = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]
TRUTH6 = [0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493]

# key -> glob of per-(seed) posteriors at a fixed perm
G = {
    "l1_ac_p0": f"{DC}/posteriors/l1_autocross_split70/l1_autocross_split70_s*_p0.npy",
    "l1_ac_p1": f"{DC}/posteriors/l1_autocross_split70/l1_autocross_split70_s*_p1.npy",
    "l1_ac_p2": f"{DC}/posteriors/l1_autocross_split70/l1_autocross_split70_s*_p2.npy",
    "l1_ao_p0": f"{DC}/posteriors/l1_autoonly_split70/l1_autoonly_split70_s*_p0.npy",
    "cnn_ac_p0": f"{DC}/phaseB_tfdata_2026_05_30/posteriors/autocross/autocross_cs41/autocross_cs41_s*_p0.npy",
    "cnn_ac_p1": f"{DC}/phaseB_multiperm_2026_05_31/posteriors/autocross_multiperm/autocross_multiperm_s*_p1.npy",
    "cnn_ac_p2": f"{DC}/phaseB_multiperm_2026_05_31/posteriors/autocross_multiperm/autocross_multiperm_s*_p2.npy",
    "cnn_ao_harm_p0": f"{DC}/phaseB_tfdata_2026_05_30/posteriors/autoonly/autoonly_cs41/autoonly_cs41_s*_p0.npy",
    "cnn_ao_native_p0": f"{DC}/phaseB_nativeauto_2026_05_31/posteriors/cnn_auto_native_rnvp/cnn_auto_native_rnvp_s*_p0.npy",
    "cnn_ac_std_p0": f"{DC}/phaseB_std_2026_05_31/posteriors/cnn_autocross_rnvp_std/cnn_autocross_rnvp_std_s*_p0.npy",
    "cnn_ac_maf_p0": f"{DC}/phaseB_maf_2026_05_31/posteriors/autocross/autocross_cs41/autocross_cs41_s*_p0.npy",
}
LAB = {
    "l1_ac_p0": "L1 auto+cross", "l1_ac_p1": "L1 a+c (perm 1)", "l1_ac_p2": "L1 a+c (perm 2)",
    "l1_ao_p0": "L1 auto-only",
    "cnn_ac_p0": "CNN auto+cross (RealNVP)", "cnn_ac_p1": "CNN a+c (perm 1)", "cnn_ac_p2": "CNN a+c (perm 2)",
    "cnn_ao_harm_p0": "CNN auto-only (harmonic)", "cnn_ao_native_p0": "CNN auto-only (native-TFDS)",
    "cnn_ac_std_p0": "CNN a+c (standardized)", "cnn_ac_maf_p0": "CNN a+c (MAF companion)",
}
# perm-labelled keys want the base name in legend (group title carries "perm 0")
LAB_PERM = dict(LAB); LAB_PERM["l1_ac_p0"] = "L1 a+c (perm 0)"; LAB_PERM["cnn_ac_p0"] = "CNN a+c (perm 0)"

COLORS = ["#C0392B", "#2471A3", "#27AE60", "#8E44AD"]  # red, blue, green, purple


def load_pool(key):
    fs = sorted(f for f in glob.glob(G[key]) if "fom" not in os.path.basename(f).lower())
    if not fs:
        raise FileNotFoundError(G[key])
    return np.concatenate([np.load(f) for f in fs], 0), len(fs)


def mcs(key, idx, labperm=False):
    x, n = load_pool(key)
    x = x[:, idx]
    lab = (LAB_PERM if labperm else LAB)[key]
    return MCSamples(samples=x, names=[NAMES6[i] for i in idx],
                     labels=[LABS6[i] for i in idx], label=lab,
                     settings={"smooth_scale_2D": 0.3, "fine_bins_2D": 256})


def make(group_keys, fname, title, idx=(0, 1, 2), labperm=False):
    samples = [mcs(k, list(idx), labperm) for k in group_keys]
    g = plots.get_subplot_plotter(width_inch=2.6 * len(idx))
    g.settings.legend_fontsize = 13
    g.settings.axes_fontsize = 11
    g.settings.axes_labelsize = 15
    g.triangle_plot(
        samples, filled=True,
        contour_colors=COLORS[:len(group_keys)],
        legend_labels=[(LAB_PERM if labperm else LAB)[k] for k in group_keys],
        markers={NAMES6[i]: TRUTH6[i] for i in idx},
    )
    g.fig.suptitle(title, y=1.02, fontsize=14)
    out = os.path.join(OUT, fname)
    g.export(out)
    print(f"  wrote {out}  ({', '.join(group_keys)})")


def main():
    # 1. headline — the money plot (3 science params + full 6)
    make(["l1_ac_p0", "cnn_ac_p0"], "01_headline_l1_vs_cnn_autocross.png",
         "Headline: L1 vs CNN — auto+cross (perm 0, 3-seed pooled)")
    make(["l1_ac_p0", "cnn_ac_p0"], "01b_headline_l1_vs_cnn_autocross_6param.png",
         "Headline: L1 vs CNN — auto+cross (all 6 params)", idx=(0, 1, 2, 3, 4, 5))
    # 2-3. perm sensitivity — why perm-averaging flipped the headline
    make(["l1_ac_p0", "l1_ac_p1", "l1_ac_p2"], "02_perm_sensitivity_L1_autocross.png",
         "L1 auto+cross across 3 fiducial realizations (FoM3 spread 27%)", labperm=True)
    make(["cnn_ac_p0", "cnn_ac_p1", "cnn_ac_p2"], "03_perm_sensitivity_CNN_autocross.png",
         "CNN auto+cross across 3 fiducial realizations (FoM3 spread 12%)", labperm=True)
    # 4. cross-map gain + the G8 harmonic-route loss
    make(["cnn_ac_p0", "cnn_ao_native_p0", "cnn_ao_harm_p0"], "04_cross_gain_and_G8.png",
         "CNN: cross-map gain + the harmonic-route loss (G8)")
    # 5. auto-only, matched route
    make(["l1_ao_p0", "cnn_ao_harm_p0"], "05_auto_only_l1_vs_cnn.png",
         "L1 vs CNN — auto-only (harmonic route, perm 0)")
    # 6. CNN flavors: companion + standardization
    make(["cnn_ac_p0", "cnn_ac_maf_p0", "cnn_ac_std_p0"], "06_cnn_flavors_autocross.png",
         "CNN auto+cross — RealNVP vs MAF companion vs standardized")
    print(f"[overlays] -> {OUT}")


if __name__ == "__main__":
    main()
