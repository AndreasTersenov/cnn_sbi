#!/usr/bin/env python3
"""Posterior-contour visuals for the auto+cross reversal, from the per-patch fiducial
dumps (samples are real per-patch posteriors; L1 and CNN row i = the SAME sky patch).
  A: L1 vs CNN at a representative (population-median-FoM3) patch  -> the reversal.
  B: L1 at the polar patch-0 vs a typical patch                    -> the patch-0 anomaly.
  C: L1 across several individual patches                          -> patch-to-patch spread.
"""
import glob
import numpy as np
from getdist import MCSamples, plots

TPP = "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/definitive_comparison/fiducial_full200/tarp_per_patch"
OUT = TPP + "/figures"
NAMES = ["Om", "s8", "w0"]; LABS = [r"\Omega_m", r"\sigma_8", r"w_0"]; TRUTH = [0.26, 0.84, -1.0]


def load(arm):
    fs = sorted(glob.glob(f"{TPP}/dumps/{arm}/seed_*/*/posterior_samples.npz"))
    samp = [np.load(f)["samples"] for f in fs]          # each (260,2000,6)
    pooled = np.concatenate(samp, axis=1)               # (260, 6000, 6)
    ca = np.load(f"{TPP}/coverage/{arm}/coverage_arrays.npz")
    return pooled, ca["fom3"], ca["is_p0"].astype(bool)


def mcs(x, label):
    return MCSamples(samples=x[:, :3], names=NAMES, labels=LABS, label=label,
                     settings={"smooth_scale_2D": 0.35, "fine_bins_2D": 200})


def corner(samples_list, labels, colors, fname, title):
    g = plots.get_subplot_plotter(width_inch=8)
    g.settings.legend_fontsize = 13; g.settings.axes_labelsize = 15
    g.triangle_plot([mcs(s, l) for s, l in zip(samples_list, labels)], filled=True,
                    contour_colors=colors, legend_labels=labels,
                    markers={NAMES[i]: TRUTH[i] for i in range(3)})
    g.fig.suptitle(title, y=1.02, fontsize=13); g.export(fname); print("wrote", fname)


def main():
    L, Lf, Lp0 = load("l1_autocross")
    C, Cf, Cp0 = load("cnn_autocross")
    pop = ~Lp0
    # representative population patch: closest to L1 population-median FoM3
    med = np.median(Lf[pop])
    i_typ = np.arange(len(Lf))[pop][np.argmin(np.abs(Lf[pop] - med))]
    # a typical patch-0 (polar): closest to patch-0 median FoM3
    p0_med = np.median(Lf[Lp0])
    i_p0 = np.arange(len(Lf))[Lp0][np.argmin(np.abs(Lf[Lp0] - p0_med))]
    print(f"i_typ={i_typ} (L1 FoM3 {Lf[i_typ]:.0f}, CNN {Cf[i_typ]:.0f}); "
          f"i_p0={i_p0} (L1 FoM3 {Lf[i_p0]:.0f})")

    # A: reversal at the same representative patch
    corner([L[i_typ], C[i_typ]], ["L1 auto+cross", "CNN auto+cross"], ["#C0392B", "#2471A3"],
           f"{OUT}/reversal_A_l1_vs_cnn_typical_patch.png",
           f"L1 vs CNN auto+cross — same representative patch (FoM3 L1 {Lf[i_typ]:.0f} / CNN {Cf[i_typ]:.0f})")

    # B: patch-0 (polar) vs typical, for L1
    corner([L[i_p0], L[i_typ]], [f"L1 patch-0 (polar, FoM3 {Lf[i_p0]:.0f})",
                                 f"L1 typical patch (FoM3 {Lf[i_typ]:.0f})"], ["#E67E22", "#C0392B"],
           f"{OUT}/reversal_B_l1_polar_vs_typical.png",
           "L1 auto+cross — the campaign's polar patch-0 vs a typical patch")

    # C: L1 patch-to-patch spread (4 population patches spanning the FoM3 range)
    qs = [np.percentile(Lf[pop], q) for q in (15, 40, 65, 90)]
    idxs = [np.arange(len(Lf))[pop][np.argmin(np.abs(Lf[pop] - q))] for q in qs]
    corner([L[i] for i in idxs], [f"FoM3 {Lf[i]:.0f}" for i in idxs],
           ["#FAD7A0", "#E67E22", "#C0392B", "#7B241C"],
           f"{OUT}/reversal_C_l1_patch_spread.png",
           "L1 auto+cross — patch-to-patch variability (4 individual patches)")


if __name__ == "__main__":
    main()
