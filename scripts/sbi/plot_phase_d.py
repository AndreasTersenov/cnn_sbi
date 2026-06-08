#!/usr/bin/env python
"""Phase-D result figures: per-patch constraining power, w0 offset, TARP coverage,
and the headline L1-vs-CNN corner. CPU-only."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import csv, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AN = "results/exploratory/definitive_comparison_10deg/phase_c/analysis"
OUT = f"{AN}/figs"
os.makedirs(OUT, exist_ok=True)
FIDUCIAL = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493])
ARMS = ["l1_auto_cross", "cnn_auto_cross", "l1_auto_only", "cnn_auto_only"]
COL = {"l1_auto_cross": "#2ca02c", "cnn_auto_cross": "#1f77b4",
       "l1_auto_only": "#98df8a", "cnn_auto_only": "#aec7e8"}
LBL = {"l1_auto_cross": "L1 a+c", "cnn_auto_cross": "CNN a+c",
       "l1_auto_only": "L1 auto", "cnn_auto_only": "CNN auto"}


def grid(arm):
    rows = list(csv.DictReader(open(f"{AN}/geometry/{arm}/per_patch_grid.csv")))
    g = lambda k: np.array([float(r[k]) for r in rows])
    d = {k: g(k) for k in ["sig_w_0", "fom2d_Omega_m_sigma_8", "fom3", "pull_w_0"]}
    return {k: v[np.isfinite(v)] for k, v in d.items()}


def fig_constraining():
    D = {a: grid(a) for a in ARMS}
    metrics = [("sig_w_0", "σ(w0)", False), ("fom2d_Omega_m_sigma_8", "2D FoM (Ωm,σ8)", True),
               ("fom3", "FoM3", True)]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    for j, (key, lab, logy) in enumerate(metrics):
        a = ax[j]
        data = [D[arm][key] for arm in ARMS]
        vp = a.violinplot(data, showmedians=True, widths=0.8)
        for i, b in enumerate(vp["bodies"]):
            b.set_facecolor(COL[ARMS[i]]); b.set_alpha(0.6)
        a.set_xticks(range(1, 5)); a.set_xticklabels([LBL[x] for x in ARMS], rotation=20, fontsize=9)
        a.set_title(lab); a.grid(alpha=0.2)
        if logy: a.set_yscale("log")
        for i, arm in enumerate(ARMS):
            m = np.median(D[arm][key]); a.text(i + 1, m, f" {m:.3g}", fontsize=7, va="center")
    fig.suptitle("Phase D — per-patch constraining power (9000 obs/arm; CNN tighter on auto+cross)", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{OUT}/D1_constraining_power.png", dpi=130, bbox_inches="tight")
    fig.savefig(f"{OUT}/D1_constraining_power.pdf", bbox_inches="tight"); print("wrote D1")


def fig_offset():
    D = {a: grid(a) for a in ARMS}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for k, (probe, arms) in enumerate([("auto+cross", ["l1_auto_cross", "cnn_auto_cross"]),
                                       ("auto-only", ["l1_auto_only", "cnn_auto_only"])]):
        a = ax[k]
        for arm in arms:
            p = D[arm]["pull_w_0"]
            a.hist(p, bins=60, range=(-2, 2), histtype="step", lw=2, color=COL[arm],
                   label=f"{LBL[arm]} (mean {p.mean():+.2f}σ)", density=True)
        a.axvline(0, color="k", lw=0.8)
        a.axvline(-0.37, color="red", ls="--", lw=1.2, label="20° L1 a+c (−0.37σ)")
        a.set_xlabel("w0 pull (bias/σ)"); a.set_title(probe); a.legend(fontsize=8); a.grid(alpha=0.2)
    ax[0].set_ylabel("density")
    fig.suptitle("Phase D — fiducial w0 offset: L1's 20° −0.37σ shrinks to −0.10σ at 10° (and matches CNN)", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{OUT}/D2_w0_offset.png", dpi=130, bbox_inches="tight")
    fig.savefig(f"{OUT}/D2_w0_offset.pdf", bbox_inches="tight"); print("wrote D2")


def fig_coverage():
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for arm in ARMS:
        d = np.load(f"{AN}/tarp/{arm}/coverage_arrays.npz")
        ax.plot(d["alpha"], d["ecp"], lw=2, color=COL[arm], label=LBL[arm])
    ax.set_xlabel("nominal credibility"); ax.set_ylabel("empirical coverage (Mahalanobis χ²₃)")
    ax.set_title("Phase D — TARP coverage\n(above diagonal = conservative; below = over-confident)")
    ax.legend(fontsize=9); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(f"{OUT}/D3_tarp_coverage.png", dpi=130, bbox_inches="tight")
    fig.savefig(f"{OUT}/D3_tarp_coverage.pdf", bbox_inches="tight"); print("wrote D3")


def fig_corner():
    from getdist import MCSamples, plots
    names = ["Om", "s8", "w0", "h0", "ns", "Ob"]
    labels = [r"\Omega_m", r"\sigma_8", "w_0", "h_0", "n_s", r"\Omega_b"]
    k_obs = 50  # a typical population obs (same patch across arms; 0-189 are population)
    samps = []
    for arm, col in [("l1_auto_cross", "#2ca02c"), ("cnn_auto_cross", "#1f77b4")]:
        pooled = []
        for f in sorted(glob.glob(f"{AN}/tarp/dumps/{arm}/seed_*/n*/posterior_samples.npz")):
            pooled.append(np.load(f)["samples"][k_obs])
        p = np.concatenate(pooled, 0); p = p[np.all(np.isfinite(p), 1)]
        samps.append(MCSamples(samples=p, names=names, labels=labels, label=LBL[arm]))
    g = plots.get_subplot_plotter(width_inch=8)
    g.triangle_plot(samps, filled=True, contour_colors=["#2ca02c", "#1f77b4"],
                    markers={n: FIDUCIAL[i] for i, n in enumerate(names)})
    g.export(f"{OUT}/D4_corner_autocross.png"); g.export(f"{OUT}/D4_corner_autocross.pdf")
    print("wrote D4")


if __name__ == "__main__":
    fig_constraining(); fig_offset(); fig_coverage(); fig_corner()
