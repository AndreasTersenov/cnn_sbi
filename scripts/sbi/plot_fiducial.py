#!/usr/bin/env python3
"""Plots + summary for the full-200 fiducial study (defensive: skips missing arms).

Reads <root>/<arm>/{mean_dv_posterior.npy, mean_dv.fom.json, step2_fom3.npz,
step2_distribution_summary.json} and writes to <out>:
  - overlays/meandv_l1_vs_cnn_autocross.png (+ auto-only)  [getdist corner]
  - overlays/fom3_distribution.png                          [step-2 hist, matplotlib]
  - FIDUCIAL_FULL200_SUMMARY.md
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = ["Om", "s8", "w0"]; LABS = [r"\Omega_m", r"\sigma_8", r"w_0"]
TRUTH = [0.26, 0.84, -1.0]
DISPLAY = {
    "l1_autocross": "L1 auto+cross", "cnn_autocross": "CNN auto+cross",
    "l1_autoonly": "L1 auto-only", "cnn_autoonly": "CNN auto-only",
    "cnn_autocross_std": "CNN auto+cross (std)", "cnn_maf_autocross": "CNN auto+cross (MAF)",
}


def _load(root, arm):
    d = Path(root) / arm
    out = {}
    if (d / "mean_dv_posterior.npy").exists():
        out["post"] = np.load(d / "mean_dv_posterior.npy")
    if (d / "mean_dv.fom.json").exists():
        out["m1"] = json.load(open(d / "mean_dv.fom.json"))
    if (d / "step2_distribution_summary.json").exists():
        out["s2"] = json.load(open(d / "step2_distribution_summary.json"))
    if (d / "step2_fom3.npz").exists():
        out["s2arr"] = np.load(d / "step2_fom3.npz")
    return out


def corner(pairs, data, fname, title):
    try:
        from getdist import MCSamples, plots
    except Exception as e:
        print(f"  [corner] getdist unavailable: {e}"); return
    mcs = []
    for arm, color in pairs:
        if arm not in data or "post" not in data[arm]:
            continue
        x = data[arm]["post"][:, :3]
        mcs.append(MCSamples(samples=x, names=NAMES, labels=LABS,
                             label=DISPLAY.get(arm, arm),
                             settings={"smooth_scale_2D": 0.3, "fine_bins_2D": 256}))
    if len(mcs) < 1:
        print(f"  [corner] no arms for {fname}"); return
    g = plots.get_subplot_plotter(width_inch=8)
    g.settings.legend_fontsize = 13
    cols = [c for a, c in pairs if a in data and "post" in data[a]]
    g.triangle_plot(mcs, filled=True, contour_colors=cols,
                    markers={NAMES[i]: TRUTH[i] for i in range(3)})
    g.fig.suptitle(title, y=1.02, fontsize=13)
    g.export(fname); print(f"  wrote {fname}")


def fom_dist(arms, data, fname):
    present = [(a, c) for a, c in arms if a in data and "s2arr" in data[a]]
    if not present:
        print("  [fom_dist] nothing to plot"); return
    fig, ax = plt.subplots(figsize=(8, 5))
    for arm, color in present:
        f3 = data[arm]["s2arr"]["fom3"]; f3 = f3[np.isfinite(f3)]
        ax.hist(f3, bins=40, histtype="step", lw=2, color=color, density=True,
                label=f"{DISPLAY.get(arm, arm)} (med {np.median(f3):.0f})")
        ax.axvline(np.median(f3), color=color, ls="--", alpha=0.6)
    ax.set_xlabel("per-patch FoM3 (single 20 deg² obs)"); ax.set_ylabel("density")
    ax.set_title("Step-2: per-patch FoM3 distribution (full-200 fiducial)")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)
    print(f"  wrote {fname}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    od = Path(a.out) / "overlays"; od.mkdir(parents=True, exist_ok=True)
    arms = ["l1_autocross", "cnn_autocross", "l1_autoonly", "cnn_autoonly",
            "cnn_autocross_std", "cnn_maf_autocross"]
    data = {arm: _load(a.root, arm) for arm in arms}

    corner([("l1_autocross", "#C0392B"), ("cnn_autocross", "#2471A3")],
           data, str(od / "meandv_l1_vs_cnn_autocross.png"),
           "Mean-datavector posterior (full-200): L1 vs CNN — auto+cross")
    corner([("l1_autoonly", "#C0392B"), ("cnn_autoonly", "#2471A3")],
           data, str(od / "meandv_l1_vs_cnn_autoonly.png"),
           "Mean-datavector posterior (full-200): L1 vs CNN — auto-only")
    fom_dist([("l1_autocross", "#C0392B"), ("cnn_autocross", "#2471A3"),
              ("l1_autoonly", "#E67E22"), ("cnn_autoonly", "#5DADE2")],
             data, str(od / "fom3_distribution.png"))

    # ---- summary markdown ----
    L = ["# Full-200 fiducial study — mean datavector + per-patch distribution", "",
         "Each arm: NDE trained on the arm's cache (3 seeds pooled). **Step 1** =",
         "posterior at the mean of all 9600 per-patch summaries (de-noised single-survey",
         "contour; NOT a 200×-tighter constraint — same per-patch noise model). **Step 2** =",
         "FoM3/σ over ~300 individual 20 deg² patches (real which-sky scatter). Each arm",
         "passed the G3 gate (reproduces its campaign perm-0 FoM3 within 20%).", "",
         "## Step 1 — mean datavector", "",
         "| arm | FoM3 | σ(Ωm) | σ(σ8) | σ(w0) |", "|---|---|---|---|---|"]
    for arm in arms:
        d = data.get(arm, {})
        if "m1" not in d:
            L.append(f"| {DISPLAY.get(arm, arm)} | (pending/failed) | | | |"); continue
        m = d["m1"]; sg = m["sigma"]
        L.append(f"| {DISPLAY.get(arm, arm)} | {m['fom3']:.0f} | {sg['Omega_m']:.4f} "
                 f"| {sg['sigma_8']:.4f} | {sg['w_0']:.4f} |")
    L += ["", "## Step 2 — per-patch FoM3 distribution (mean ± std; median [16,84])", "",
          "| arm | FoM3 mean±std | median [16,84] | σ(w0) mean±std |",
          "|---|---|---|---|"]
    for arm in arms:
        d = data.get(arm, {})
        if "s2" not in d:
            L.append(f"| {DISPLAY.get(arm, arm)} | (pending/failed) | | |"); continue
        s = d["s2"]
        L.append(f"| {DISPLAY.get(arm, arm)} | {s['fom3_mean']:.0f}±{s['fom3_std']:.0f} "
                 f"| {s['fom3_median']:.0f} [{s['fom3_p16']:.0f},{s['fom3_p84']:.0f}] "
                 f"| {s['sig_w0_mean']:.4f}±{s['sig_w0_std']:.4f} |")
    L += ["", "Figures: `overlays/meandv_l1_vs_cnn_{autocross,autoonly}.png`, "
          "`overlays/fom3_distribution.png`.", "",
          "_Auto-generated by plot_fiducial.py._"]
    (Path(a.out) / "FIDUCIAL_FULL200_SUMMARY.md").write_text("\n".join(L) + "\n")
    print(f"  wrote {Path(a.out) / 'FIDUCIAL_FULL200_SUMMARY.md'}")


if __name__ == "__main__":
    main()
