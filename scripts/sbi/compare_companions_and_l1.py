#!/usr/bin/env python3
"""3-way comparison: L1 vs CNN-RealNVP vs CNN-MAF companion (auto+cross & auto-only).

Pools compressor-seed-41 posteriors over NDE seeds (perm 0). Writes a FoM table
(SUMMARY_COMPANION_COMPARISON.md) + 3-way corner overlays. Robust to a method's
results not existing yet (skips it). CNN FoM are tf.data-route leak-inflated
(~1.6x); read CNN-vs-CNN (companion) deltas and the relative gains, not absolute.
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
from getdist import MCSamples, plots

ROOT = "results/exploratory/definitive_comparison"
OUTDIR = os.path.join(ROOT, "companion_comparison_2026_05_31"); os.makedirs(OUTDIR, exist_ok=True)
NAMES = ["Omega_m", "sigma_8", "w_0", "h_0", "n_s", "Omega_b"]
LABELS = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]
TRUTH = [0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493]
SEEDS = [41, 42, 43]

# (method, color, path-template for cs41 pooled npy given arm in {autocross,autoonly})
SOURCES = {
    "L1": ("#1d3557", lambda arm: [f"{ROOT}/posteriors/l1_{arm}_split70/l1_{arm}_split70_s{s}_p0.npy" for s in SEEDS]),
    "CNN-RealNVP": ("#c1121f", lambda arm: [f"{ROOT}/phaseB_tfdata_2026_05_30/posteriors/{arm}/{arm}_cs41/{arm}_cs41_s{s}_p0.npy" for s in SEEDS]),
    "CNN-MAF": ("#2a9d8f", lambda arm: [f"{ROOT}/phaseB_maf_2026_05_31/posteriors/{arm}/{arm}_cs41/{arm}_cs41_s{s}_p0.npy" for s in SEEDS]),
}


def pool(paths):
    a = [np.load(p) for p in paths if os.path.exists(p)]
    return (np.concatenate(a, 0), len(a)) if a else (None, 0)


def fom3(x):
    c = np.cov(x[:, :3], rowvar=False)
    return float(1.0 / np.sqrt(np.linalg.det(c)))


def sigmas(x):
    return {n: float(np.std(x[:, i])) for i, n in enumerate(NAMES)}


def main():
    lines = ["# Companion comparison — L1 vs CNN-RealNVP vs CNN-MAF (perm0, cs41, NDE seeds pooled)",
             "",
             "⚠️ CNN FoM are tf.data-route leak-inflated (~1.6×). Compare **CNN-MAF vs "
             "CNN-RealNVP** (same leakage → companion delta is clean) and relative gains.", ""]
    table = {}
    for arm, pretty in [("autocross", "auto+cross"), ("autoonly", "auto-only")]:
        samp = {}
        for method, (color, tmpl) in SOURCES.items():
            x, n = pool(tmpl(arm))
            if x is not None:
                samp[method] = (x, n)
                table[(arm, method)] = (fom3(x), sigmas(x), n)
        # corner overlay for this arm (methods present)
        if len(samp) >= 2:
            mc = [MCSamples(samples=x, names=NAMES, labels=LABELS, label=m)
                  for m, (x, n) in samp.items()]
            g = plots.get_subplot_plotter(width_inch=10)
            g.settings.legend_fontsize = 14; g.settings.axes_labelsize = 14
            g.triangle_plot(mc, filled=True,
                            contour_colors=[SOURCES[m][0] for m in samp],
                            legend_labels=[f"{m} (FoM3={table[(arm,m)][0]:.0f}, n={table[(arm,m)][2]})" for m in samp],
                            markers={n_: t for n_, t in zip(NAMES, TRUTH)})
            g.fig.suptitle(f"L1 vs CNN-RealNVP vs CNN-MAF — {pretty} (jaxili NDE, 70/30, perm0)\n"
                           "CNN absolute leak-inflated; MAF-vs-RealNVP delta is the clean read",
                           fontsize=11, y=1.02)
            out = os.path.join(OUTDIR, f"compare_{arm}.png"); g.export(out)
            print("wrote", out)
    # table
    lines += ["## FoM3 (pooled cs41)", "", "| input | method | FoM3 | σ(Ωm) | σ(σ8) | σ(w0) | n |",
              "|---|---|---|---|---|---|---|"]
    for arm in ("autocross", "autoonly"):
        for method in SOURCES:
            if (arm, method) in table:
                f3, sg, n = table[(arm, method)]
                lines.append(f"| {arm} | {method} | {f3:.0f} | {sg['Omega_m']:.4f} | "
                             f"{sg['sigma_8']:.4f} | {sg['w_0']:.4f} | {n} |")
    # companion delta
    lines += ["", "## Companion effect (CNN-MAF / CNN-RealNVP FoM3)", ""]
    for arm in ("autocross", "autoonly"):
        if (arm, "CNN-MAF") in table and (arm, "CNN-RealNVP") in table:
            r = table[(arm, "CNN-MAF")][0] / table[(arm, "CNN-RealNVP")][0]
            lines.append(f"- {arm}: MAF/RealNVP = **{r:.2f}×** "
                         f"(MAF {table[(arm,'CNN-MAF')][0]:.0f} vs RealNVP {table[(arm,'CNN-RealNVP')][0]:.0f})")
        else:
            lines.append(f"- {arm}: _MAF results not complete yet_")
    open(os.path.join(OUTDIR, "SUMMARY_COMPANION_COMPARISON.md"), "w").write("\n".join(lines) + "\n")
    print("wrote", os.path.join(OUTDIR, "SUMMARY_COMPANION_COMPARISON.md"))


if __name__ == "__main__":
    main()
