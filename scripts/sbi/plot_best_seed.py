#!/usr/bin/env python3
"""Plots for the CNN best-single-seed (un-pooled) check: FoM3 bars + best-seed corners.

Reads cnn_phase/best_seed/{per_seed.json, best_seed_samples_typical.npz} and the L1 / CNN pooled
representative corner samples. Emits a FoM3 bar chart (best-seed vs CNN-pooled vs L1-pooled) and a
product corner overlay (CNN best seed vs CNN pooled vs L1 pooled) at the typical obs.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = "/home/tersenov/.claude/skills/figure-polish/style/aa.mplstyle"
FC = Path("/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/flatsky_cross_2026_06")
CNNP = FC / "cnn_phase"
OUT = CNNP / "best_seed"
ARMS = [("none", "auto-only"), ("conv", "+conv"), ("product", "+product"), ("both", "+both")]
C_BEST, C_POOL, C_L1 = "#0072B2", "#56B4E9", "#D55E00"


def med(base, op):
    return json.load(open(FC / base / f"flat_{op}" / "median_summary.json"))["fom3"]


def main():
    plt.style.use(STYLE)
    per = json.load(open(OUT / "per_seed.json"))
    best = {op: per[op]["best_fom3"] for op, _ in ARMS}
    cnn_pool = {op: med("cnn_phase/population_sweep", op) for op, _ in ARMS}
    l1_pool = {op: med("population_sweep", op) for op, _ in ARMS}

    # --- FoM3 bars: best-seed vs CNN pooled vs L1 pooled ---
    fig, ax = plt.subplots(figsize=(7.087, 3.6))
    x = np.arange(len(ARMS)); w = 0.27
    ax.bar(x - w, [best[o] for o, _ in ARMS], w, color=C_BEST, edgecolor="k", lw=0.5,
           label="CNN best single seed (un-pooled)")
    ax.bar(x, [cnn_pool[o] for o, _ in ARMS], w, color=C_POOL, edgecolor="k", lw=0.5,
           label="CNN pooled-3-seed median")
    ax.bar(x + w, [l1_pool[o] for o, _ in ARMS], w, color=C_L1, edgecolor="k", lw=0.5, hatch="///",
           label="L1 pooled-3-seed median")
    ax.axhline(best["none"], color=C_BEST, ls=":", lw=0.9)   # CNN best auto-only reference
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in ARMS])
    ax.set_ylabel("FoM$_3$"); ax.legend(fontsize=8, frameon=False, loc="upper left")
    no_gain = all(best[o] <= best["none"] for o, _ in ARMS if o != "none")
    ax.set_title("CNN best single seed still shows no cross gain (cross arms $\\leq$ auto-only)"
                 if no_gain else
                 "CNN best single seed: at least one cross arm EXCEEDS auto-only", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fom3_best_seed.png", dpi=200); fig.savefig(OUT / "fom3_best_seed.pdf")
    print(f"wrote {OUT}/fom3_best_seed.{{png,pdf}}")

    # --- product corner: CNN best seed vs CNN pooled vs L1 pooled (typical obs) ---
    try:
        from getdist import MCSamples, plots
        idx = [0, 1, 2]; names = ["Om", "s8", "w0"]; labs = [r"\Omega_m", r"\sigma_8", "w_0"]
        truth = {"Om": 0.26, "s8": 0.84, "w0": -1.0}
        bs = np.load(OUT / "best_seed_samples_typical.npz")["product"][:, idx]
        cp = np.load(CNNP / "representative_corner/flat_product/corner_samples.npz")["typical"][:, idx]
        lp = np.load(FC / "representative_corner/flat_product/corner_samples.npz")["typical"][:, idx]
        mc = [MCSamples(samples=s, names=names, labels=labs) for s in (cp, lp, bs)]
        g = plots.get_subplot_plotter(width_inch=7.087)
        g.settings.axes_labelsize = 14
        g.triangle_plot(mc, params=names, filled=[False, True, False],
                        contour_colors=[C_POOL, C_L1, C_BEST],
                        contour_ls=["-", "-", "--"], contour_lws=[1.3, 1.3, 1.8], markers=truth)
        for lg in list(g.fig.legends): lg.remove()
        for axx in g.fig.axes:
            if axx.get_legend(): axx.get_legend().remove()
        import matplotlib.lines as mlines
        handles = [mlines.Line2D([], [], color=C_BEST, ls="--", lw=1.8, label="CNN best seed (un-pooled)"),
                   mlines.Line2D([], [], color=C_POOL, ls="-", lw=1.3, label="CNN pooled median"),
                   mlines.Line2D([], [], color=C_L1, ls="-", lw=1.3, label="L1 pooled median")]
        g.fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.97, 0.97),
                     fontsize=11, frameon=False)
        g.fig.savefig(OUT / "corner_best_seed_product.png", dpi=200)
        g.fig.savefig(OUT / "corner_best_seed_product.pdf")
        print(f"wrote {OUT}/corner_best_seed_product.{{png,pdf}}")

        # --- per-arm 2-contour overlay: CNN best seed vs L1 pooled (typical obs) ---
        import matplotlib.lines as mlines
        bs_all = np.load(OUT / "best_seed_samples_typical.npz")
        for op, lab in ARMS:
            cb = bs_all[op][:, idx]
            l1s = np.load(FC / f"representative_corner/flat_{op}/corner_samples.npz")["typical"][:, idx]
            mc_l1 = MCSamples(samples=l1s, names=names, labels=labs)
            mc_cb = MCSamples(samples=cb, names=names, labels=labs)
            gg = plots.get_subplot_plotter(width_inch=7.087)
            gg.settings.axes_labelsize = 15
            gg.triangle_plot([mc_cb, mc_l1], params=names, filled=True,
                             contour_colors=[C_BEST, C_L1], contour_ls=["--", "-"],
                             contour_lws=[1.6, 1.6], markers=truth)
            for lg in list(gg.fig.legends): lg.remove()
            for axx in gg.fig.axes:
                if axx.get_legend(): axx.get_legend().remove()
            h = [mlines.Line2D([], [], color=C_L1, ls="-", lw=1.6, label=f"L1 {lab} (pooled median)"),
                 mlines.Line2D([], [], color=C_BEST, ls="--", lw=1.6, label=f"CNN {lab} (best single seed)")]
            gg.fig.legend(handles=h, loc="upper right", bbox_to_anchor=(0.97, 0.97), fontsize=12, frameon=False)
            gg.fig.savefig(OUT / f"corner_best_seed_vs_l1_{op}.png", dpi=200)
            gg.fig.savefig(OUT / f"corner_best_seed_vs_l1_{op}.pdf")
            print(f"wrote {OUT}/corner_best_seed_vs_l1_{op}.{{png,pdf}}")
    except Exception as e:
        print(f"[warn] corner failed: {e}")


if __name__ == "__main__":
    main()
