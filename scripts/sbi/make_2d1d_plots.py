#!/usr/bin/env python
"""Plots for the 2D-1D Haar wavelet-ℓ1 Phase-1 result (RESULT_2D1D_PHASE1.md).

(1) DATAVECTORS:
    1a sensitivity: the haar_nobnt 2D-1D Haar ℓ1 datavector, cosmology-colored by σ8
       (4 Haar channels × 5 wavelet scales grid) — shows the statistic AND its constraining power.
    1b BNT collapse: mean datavector per Haar channel, no-BNT (solid) vs BNT-space (dashed) — the
       goal-2 story (deep channel collapses moving to BNT space).
(2) CONTOURS: GetDist (Ωm,σ8,w0) corner overlaying flat_none, flat_product, haar_nobnt,
    autohaar_nobnt, haar_bnt_uncut at a single MATCHED truth (same val point, no retraining),
    pooled over the 3 NDE seeds. Illustrative single realization; population medians in caption.
CPU-only (numpy + matplotlib + getdist).
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors

HERE = os.path.dirname(os.path.abspath(__file__))
FS = HERE + "/results/exploratory/flatsky_cross_2026_06"
OM2 = FS + "/overnight_menu_2"
GC2 = OM2 + "/gate_c_2d1d/tarp_drp/dumps"     # Haar arms
GC0 = FS + "/gate_c/tarp_drp/dumps"           # baselines
OUT = FS + "/plots_2d1d"
NSC, NBIN = 5, 40
SEEDS = (41, 42, 43)
SCI = [0, 1, 2]
PN = ["Omega_m", "sigma_8", "w_0"]
PLAB = [r"\Omega_m", r"\sigma_8", r"w_0"]


def load_cache_x_theta(arm):
    z = np.load(f"{OM2}/{arm}/cache/l1_train.npz")
    return z["x"].astype(np.float64), z["theta"].astype(np.float64)


def channel_ranges(arm):
    m = np.load(f"{OM2}/{arm}/cache/l1_cache_meta.npz", allow_pickle=True)
    return np.asarray(m["ranges"], float), [str(s) for s in m["channel_names"]]


def mean_datavector(arm):
    """Mean over the 36000 fiducial patches → (C, NSC, NBIN)."""
    S = np.load(f"{FS}/gate_c/lc2st/fiducial_summaries_{arm}.npz")["S"].astype(np.float64)
    C = S.shape[1] // (NSC * NBIN)
    return S.mean(0).reshape(C, NSC, NBIN), C


# ---------------- Fig 1a: cosmology-colored datavector (sensitivity) ----------------
def fig_sensitivity(arm="haar_nobnt", n_curves=500):
    x, th = load_cache_x_theta(arm)
    rng = np.random.default_rng(0)
    idx = rng.choice(x.shape[0], size=min(n_curves, x.shape[0]), replace=False)
    C = x.shape[1] // (NSC * NBIN)
    Xc = x[idx].reshape(len(idx), C, NSC, NBIN)
    s8 = th[idx, 1]
    rng_ch, names = channel_ranges(arm)
    norm = mcolors.Normalize(vmin=s8.min(), vmax=s8.max())
    cmap = cm.viridis
    fig, axes = plt.subplots(C, NSC, figsize=(3.0 * NSC, 2.4 * C), squeeze=False)
    order = np.argsort(s8)
    for c in range(C):
        snr = np.linspace(rng_ch[c, 0], rng_ch[c, 1], NBIN)
        for s in range(NSC):
            ax = axes[c, s]
            for i in order:
                ax.plot(snr, Xc[i, c, s, :], color=cmap(norm(s8[i])), alpha=0.10, lw=0.4)
            if c == 0:
                ax.set_title(f"scale {s}", fontsize=9)
            if s == 0:
                ax.set_ylabel(f"{names[c]}\nmean L1", fontsize=8)
            if c == C - 1:
                ax.set_xlabel("S/N", fontsize=8)
            ax.tick_params(labelsize=7)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
    cb.set_label(r"$\sigma_8$", fontsize=10)
    fig.suptitle(f"2D-1D Haar wavelet-$\\ell_1$ datavector ({arm}), colored by $\\sigma_8$ "
                 f"— {len(idx)} cosmologies", fontsize=11)
    p = f"{OUT}/datavector_sensitivity_{arm}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


# ---------------- Fig 1b: BNT collapse (no-BNT vs BNT-space mean datavector) ----------------
def fig_bnt_collapse():
    dv_n, _ = mean_datavector("haar_nobnt")        # (4, NSC, NBIN)
    dv_b, _ = mean_datavector("haar_bnt_uncut")
    rng_n, names = channel_ranges("haar_nobnt")
    rng_b, _ = channel_ranges("haar_bnt_uncut")
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), squeeze=False)
    sc_colors = cm.plasma(np.linspace(0.1, 0.85, NSC))
    for c in range(4):
        ax = axes[0, c]
        snr_n = np.linspace(rng_n[c, 0], rng_n[c, 1], NBIN)
        snr_b = np.linspace(rng_b[c, 0], rng_b[c, 1], NBIN)
        for s in range(NSC):
            ax.plot(snr_n, dv_n[c, s], color=sc_colors[s], lw=1.6,
                    label=("no-BNT" if s == 0 else None))
            ax.plot(snr_b, dv_b[c, s], color=sc_colors[s], lw=1.3, ls="--",
                    label=("BNT space" if s == 0 else None))
        ax.set_title(names[c].replace("_", " "), fontsize=10)
        ax.set_xlabel("S/N", fontsize=9); ax.set_yscale("log")
        if c == 0:
            ax.set_ylabel("mean L1 (log)", fontsize=9); ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
    fig.suptitle("2D-1D Haar ℓ1 datavector — no-BNT (solid) vs BNT space (dashed); "
                 "color = wavelet scale. Deep channel collapses under BNT (FoM3 2676→885).",
                 fontsize=11)
    p = f"{OUT}/datavector_bnt_collapse.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


# ---------------- Fig 2: contours at a matched truth ----------------
def gather_points(arm, root):
    """truth-key -> list of (M,6) sample arrays over seeds/terciles."""
    pts = {}
    for seed in SEEDS:
        for terc in ("LOW", "MID", "HIGH"):
            for f in glob.glob(f"{root}/{arm}_{terc}/seed_{seed}/n*_m*/posterior_samples.npz"):
                z = np.load(f); s, th = z["samples"], z["theta"]
                for i in range(th.shape[0]):
                    pts.setdefault(tuple(np.round(th[i].astype(np.float64), 8)), []).append(s[i])
    return pts


def fig_contours():
    from getdist import MCSamples, plots as gdplt
    arms = [("flat_none", GC0, "auto-only (2405)"),
            ("flat_product", GC0, "L1+product (2875)"),
            ("haar_nobnt", GC2, "2D-1D Haar (2676)"),
            ("autohaar_nobnt", GC2, "autos⊕Haar (2954)"),
            ("haar_bnt_uncut", GC2, "Haar in BNT space (885)")]
    P = {a: gather_points(a, r) for a, r, _ in arms}
    common = set(P[arms[0][0]])
    for a, _, _ in arms[1:]:
        common &= set(P[a])
    print(f"  truths common to all {len(arms)} arms: {len(common)}")
    if not common:
        print("  [!] no common truth across all arms — falling back to the 3 Haar arms only")
        arms = arms[2:]; common = set(P["haar_nobnt"]) & set(P["autohaar_nobnt"]) & set(P["haar_bnt_uncut"])
    keys = np.array(sorted(common))
    center = np.median(keys, axis=0)
    truth = keys[np.argmin(((keys - center) ** 2).sum(1))]    # most central matched cosmology
    print(f"  matched truth (Om,s8,w0,h0,ns,Ob) = {np.round(truth,4)}")
    mcs, cols = [], ["#888888", "#1f77b4", "#2ca02c", "#d62728", "#9467bd"][-len(arms):]
    for a, _, lab in arms:
        s = np.concatenate(P[a][tuple(truth)], 0)[:, SCI]
        s = s[np.all(np.isfinite(s), 1)]
        mcs.append(MCSamples(samples=s, names=PN, labels=PLAB, label=lab))
    g = gdplt.get_subplot_plotter(width_inch=8.0)
    g.triangle_plot(mcs, PN, filled=True, colors=cols, legend_labels=[a[2] for a in arms],
                    markers={PN[i]: float(truth[i]) for i in range(3)},
                    marker_args={"color": "k", "lw": 0.9, "ls": ":"})
    for ext in ("png", "pdf"):
        p = f"{OUT}/contours_2d1d.{ext}"; g.export(p); print("wrote", p)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_sensitivity("haar_nobnt")
    fig_bnt_collapse()
    fig_contours()
    print("ALL PLOTS DONE ->", OUT)
