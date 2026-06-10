#!/usr/bin/env python3
"""Consolidate the de-leaked flat-local L1-vs-CNN comparison: table + overlays + writeup.

Reads the CNN and L1 population-sweep median_summary.json (pooled 9000-obs median) and the
representative corner samples, emits:
  - FLATSKY_CNN_RESULT.md   (repo root): CNN FoM3/sigma/2D table vs L1 + vs full-sphere leaky
  - cnn_phase/figs/overlay_cnn_vs_l1_<arm>_typical.png  (per-arm contour overlay, typical patch)
  - cnn_phase/figs/fom3_bars_cnn_vs_l1.png
Defensive: missing inputs are reported, not fatal. CPU-only.
"""
import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SBI = Path("/mnt/home/tersenov/software/cnn_sbi/scripts/sbi")
ROOT = Path("/mnt/home/tersenov/software/cnn_sbi")
FC = SBI / "results/exploratory/flatsky_cross_2026_06"
CNNP = FC / "cnn_phase"
ARMS = ["none", "conv", "product", "both"]
ARMLAB = {"none": "auto-only", "conv": "+conv", "product": "+product", "both": "+both"}
FIGS = CNNP / "figs"; FIGS.mkdir(parents=True, exist_ok=True)
# full-sphere leaky reference (from FLATSKY_CROSS_RESULT.md / SUMMARY_PHASE_D)
FULLSPHERE = {"l1_auto": 2200, "l1_autocross": 8530, "cnn_autocross_leaky": 17251}


def load_med(base, op):
    f = Path(base) / f"flat_{op}" / "median_summary.json"
    return json.load(open(f)) if f.exists() else None


def main():
    cnn = {op: load_med(CNNP / "population_sweep", op) for op in ARMS}
    l1 = {op: load_med(FC / "population_sweep", op) for op in ARMS}

    # ---- markdown table ----
    lines = ["# Flat-sky (patch-local) cross — de-leaked L1-vs-CNN (2026-06-09)\n",
             "**Pooled 3-seed 9000-obs median, common jaxili MAF (NDE confound removed).** "
             "CNN-VMIM compressor on the same de-leaked patch-local cross as the L1 side "
             "(`FLATSKY_CROSS_RESULT.md`). Calibrated: TARP ✓ / SBC / L-C2ST (see cnn_phase/gate_c).\n",
             "## FoM3 (pooled median)\n",
             "| arm | CNN FoM3 | CNN vs auto | L1 FoM3 | L1 vs auto | CNN/L1 |",
             "|---|---|---|---|---|---|"]
    cnn_auto = cnn["none"]["fom3"] if cnn["none"] else None
    l1_auto = l1["none"]["fom3"] if l1["none"] else None
    for op in ARMS:
        c = cnn[op]; L = l1[op]
        cf = c["fom3"] if c else float("nan"); lf = L["fom3"] if L else float("nan")
        cva = f"{cf/cnn_auto:.2f}×" if (c and cnn_auto) else "—"
        lva = f"{lf/l1_auto:.2f}×" if (L and l1_auto) else "—"
        ratio = f"{cf/lf:.2f}×" if (c and L and lf) else "—"
        lines.append(f"| {ARMLAB[op]} | {cf:.0f} | {cva} | {lf:.0f} | {lva} | {ratio} |")
    lines += ["",
              f"*Full-sphere (leaky) reference: L1 auto {FULLSPHERE['l1_auto']}, L1 auto+cross "
              f"{FULLSPHERE['l1_autocross']} (3.88×), CNN auto+cross ~{FULLSPHERE['cnn_autocross_leaky']} "
              f"(~7.4×). The leaky CNN crushed L1; de-leaked they should be comparable.*\n",
              "## Marginal sigma + 2D(Om,s8) (pooled median)\n",
              "| arm | CNN sig(Om,s8,w0) | CNN 2D(Om,s8) | L1 sig(Om,s8,w0) | L1 2D(Om,s8) |",
              "|---|---|---|---|---|"]
    for op in ARMS:
        c = cnn[op]; L = l1[op]
        cs = (f"{c['sigma_Om']:.3f},{c['sigma_s8']:.3f},{c['sigma_w0']:.3f}" if c else "—")
        ls = (f"{L['sigma_Om']:.3f},{L['sigma_s8']:.3f},{L['sigma_w0']:.3f}" if L else "—")
        c2 = f"{c['fom2d_Om_s8']:.0f}" if c else "—"
        l2 = f"{L['fom2d_Om_s8']:.0f}" if L else "—"
        lines.append(f"| {ARMLAB[op]} | {cs} | {c2} | {ls} | {l2} |")
    lines += ["", "## GATE C", "- TARP-DRP: see `cnn_phase/gate_c/tarp_drp/` (per-arm/tercile).",
              "- SBC: see `cnn_phase/gate_c/sbc/sbc_summary.json`.",
              "- L-C2ST: see `cnn_phase/gate_c/lc2st/` (works for CNN 10-dim).", ""]
    (ROOT / "FLATSKY_CNN_RESULT.md").write_text("\n".join(lines))
    print(f"wrote {ROOT/'FLATSKY_CNN_RESULT.md'}")

    # ---- FoM3 bar chart ----
    x = np.arange(len(ARMS)); w = 0.38
    cf = [cnn[op]["fom3"] if cnn[op] else 0 for op in ARMS]
    lf = [l1[op]["fom3"] if l1[op] else 0 for op in ARMS]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w/2, cf, w, label="CNN", color="#1f77b4")
    ax.bar(x + w/2, lf, w, label="L1", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels([ARMLAB[o] for o in ARMS])
    ax.set_ylabel("FoM3 (pooled 9000-obs median)"); ax.legend()
    ax.set_title("De-leaked flat-local cross: CNN vs L1 FoM3")
    fig.tight_layout(); fig.savefig(FIGS / "fom3_bars_cnn_vs_l1.png", dpi=130)
    print(f"wrote {FIGS/'fom3_bars_cnn_vs_l1.png'}")

    # ---- per-arm contour overlays at the typical patch ----
    try:
        from getdist import MCSamples, plots
        idx = [0, 1, 2]; names = ["Om", "s8", "w0"]; labels = [r"\Omega_m", r"\sigma_8", "w_0"]
        truth = {"Om": 0.26, "s8": 0.84, "w0": -1.0}
        for op in ARMS:
            cnn_f = CNNP / "representative_corner" / f"flat_{op}" / "corner_samples.npz"
            l1_f = FC / "representative_corner" / f"flat_{op}" / "corner_samples.npz"
            if not (cnn_f.exists() and l1_f.exists()):
                print(f"  [skip overlay {op}] missing {'CNN' if not cnn_f.exists() else 'L1'} samples")
                continue
            cs = np.load(cnn_f)["typical"][:, idx]; ls = np.load(l1_f)["typical"][:, idx]
            mc_c = MCSamples(samples=cs, names=names, labels=labels, label=f"CNN {ARMLAB[op]}")
            mc_l = MCSamples(samples=ls, names=names, labels=labels, label=f"L1 {ARMLAB[op]}")
            g = plots.get_subplot_plotter(width_inch=6.5)
            g.settings.alpha_filled_add = 0.55
            # CNN first (underneath), L1 last (on top) so the tighter L1 contour sits over CNN's
            g.triangle_plot([mc_c, mc_l], filled=True, contour_colors=["#1f77b4", "#d62728"],
                            markers=truth, legend_labels=[f"CNN {ARMLAB[op]}", f"L1 {ARMLAB[op]}"])
            g.export(str(FIGS / f"overlay_cnn_vs_l1_{op}_typical.png"))
            print(f"  wrote {FIGS/f'overlay_cnn_vs_l1_{op}_typical.png'}")
    except Exception as e:
        print(f"  [warn] overlays failed: {e}")


if __name__ == "__main__":
    main()
