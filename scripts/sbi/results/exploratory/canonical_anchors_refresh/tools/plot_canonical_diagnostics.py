#!/usr/bin/env python3
"""plot_canonical_diagnostics.py — comprehensive interpretability suite for
the canonical-anchors-refresh campaign.

Andreas's ask: detailed per-run plots so any single seed of any arm can be
diagnosed (NDE convergence, compressor convergence, L1 datavector content,
SNR coverage, feature mask, per-seed posterior shape, bias-vs-truth).

Output structure:
    <campaign>/canonical_diagnostics.pdf          — bound PDF
    <campaign>/canonical_diagnostics_png/
        00_overview/                              — cross-arm summaries
            00_fom3_bars.png
            01_bias_bars.png
            02_cross_arm_overlay_auto.png
            03_cross_arm_overlay_autocross.png
            04_sanity_tfds_vs_cache.png
        <arm>/
            01_compressor_loss_3seed.png          — CNN only
            02_nde_loss_3seed.png
            03_corner_3seed_overlay.png
            04_bias_per_seed.png
            05_l1_datavector_per_seed.png         — L1 only
            06_l1_feature_mask.png                — L1 only
            07_l1_summary_health.png              — L1 only
            seed_{41,42,43}/
                A_compressor_loss.png             — CNN only
                B_nde_loss.png
                C_corner.png
                D_l1_datavector.png               — L1 only
                E_summary_stats.png

Robust to missing files (skips panels with a warning).

Usage:
    python plot_canonical_diagnostics.py <campaign_dir>
"""

from __future__ import annotations
import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

warnings.filterwarnings("ignore")

TRUTH = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493])
PARAM_NAMES = ["Omega_m", "sigma_8", "w0", "h0", "n_s", "Omega_b"]
PARAM_LABELS_PLAIN = [r"$\Omega_m$", r"$\sigma_8$", r"$w_0$",
                       r"$h_0$", r"$n_s$", r"$\Omega_b$"]

ARMS = {
    "cnn_auto":              dict(label="CNN auto-only (TFDS)",        color="#1f77b4"),
    "cnn_cross":             dict(label="CNN auto+cross",              color="#d62728"),
    "l1_auto":               dict(label="L1 auto-only",                color="#2ca02c"),
    "l1_cross":              dict(label="L1 auto+cross",               color="#ff7f0e"),
    "cnn_cache_auto_sanity": dict(label="CNN cache-auto (sanity)",     color="#9467bd"),
}
SEEDS = [41, 42, 43]


def fom3(s): return 1.0 / np.sqrt(np.linalg.det(np.cov(s[:, :3].T)))


def safe_load(p: Path):
    try:
        s = np.load(p)
        if np.isnan(s).any():
            print(f"  [warn] NaN in {p.name}, skipping")
            return None
        return s
    except FileNotFoundError:
        return None


def find_arm_runs(campaign_dir: Path) -> dict:
    out = {arm: {} for arm in ARMS}
    pd = campaign_dir / "posteriors"
    td = campaign_dir / "train"
    for arm in ARMS:
        for seed in SEEDS:
            stem = f"{arm}_canon_s{seed}"
            post = pd / f"{stem}.npy"
            meta = pd / f"{stem}.meta.json"
            tdir = td / stem
            if post.exists():
                out[arm][seed] = dict(
                    posterior=post, meta=meta,
                    train_dir=tdir if tdir.exists() else None,
                )
    return out


# ============================================================
# Loaders for per-run artifacts
# ============================================================
def load_cnn_compressor_curve(tdir: Path):
    tp = list(tdir.rglob("loss_compressor_train.npy"))
    vp = list(tdir.rglob("loss_compressor_test.npy"))
    if not tp or not vp: return None
    return dict(train=np.load(tp[0]), val=np.load(vp[0]))


def load_cnn_nde_curve(tdir: Path):
    vp = list(tdir.rglob("loss_val_cnn.npy"))
    sp = list(tdir.rglob("loss_val_steps.npy"))
    if not vp: return None
    val = np.load(vp[0])
    steps = np.load(sp[0]) if sp else np.arange(len(val))
    return dict(val=val, steps=steps)


def load_l1_nde_curve(tdir: Path):
    md = list(tdir.rglob("NDE_w_Standardization/version_*/metrics"))
    if not md: return None
    metric_dir = sorted(md)[-1]
    efiles = sorted(metric_dir.glob("eval_epoch_*.json"))
    if not efiles: return None
    epochs, train_loss, val_loss = [], [], []
    for ef in efiles:
        try:
            j = json.loads(ef.read_text())
            n = int(ef.stem.split("_")[-1])
            epochs.append(n)
            val_loss.append(float(j.get("val/loss", np.nan)))
            train_loss.append(float(j.get("train/loss", np.nan)))
        except Exception:
            continue
    return dict(epochs=np.array(epochs),
                train=np.array(train_loss),
                val=np.array(val_loss))


def load_l1_standardization(tdir: Path):
    sp = list(tdir.rglob("l1_*_standardization.npz"))
    if not sp: return None
    return np.load(sp[0])


def load_l1_feature_mask(tdir: Path):
    mp = list(tdir.rglob("l1_*_feature_mask.npz"))
    if not mp: return None
    return np.load(mp[0])


def load_meta(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


# ============================================================
# Per-run panels
# ============================================================
def panel_cnn_compressor(ax, tdir, seed_label):
    d = load_cnn_compressor_curve(tdir)
    if d is None:
        ax.text(0.5, 0.5, "(no compressor loss arrays)", ha="center", transform=ax.transAxes); return
    tr, val = d["train"], d["val"]
    n = len(tr)
    steps_val = np.linspace(0, n, len(val)) if len(val) else np.array([])
    ax.plot(np.arange(n), tr, alpha=0.4, lw=0.7, label="train")
    if len(val):
        finite = np.isfinite(val) & (np.abs(val) < 50)
        ax.plot(steps_val[finite], val[finite], color="C1", lw=1.5, label="val")
        best = np.nanmin(val[finite]) if finite.any() else np.nan
        best_step = steps_val[finite][np.nanargmin(val[finite])] if finite.any() else 0
        ax.axhline(best, color="C1", lw=0.5, linestyle=":", alpha=0.7)
        ax.annotate(f"best={best:.2f} @ step {int(best_step)}",
                     xy=(best_step, best), xytext=(0.6, 0.95), textcoords="axes fraction",
                     fontsize=8, color="C1")
    ax.set_title(f"Compressor VMIM loss — {seed_label}")
    ax.set_xlabel("step (batch idx)"); ax.set_ylabel("VMIM loss")
    ax.set_ylim(-20, 0); ax.grid(alpha=0.3); ax.legend(fontsize=8)


def panel_nde_loss(ax, tdir, seed_label, is_l1: bool):
    if is_l1:
        d = load_l1_nde_curve(tdir)
        if d is None:
            ax.text(0.5, 0.5, "(no L1 NDE metrics found)", ha="center", transform=ax.transAxes); return
        ep, tr, val = d["epochs"], d["train"], d["val"]
        ax.plot(ep, tr, alpha=0.5, lw=1.0, label="train")
        ax.plot(ep, val, lw=1.5, color="C1", label="val")
        finite = np.isfinite(val)
        if finite.any():
            best = np.nanmin(val[finite])
            best_ep = ep[finite][np.nanargmin(val[finite])]
            ax.axhline(best, color="C1", lw=0.5, linestyle=":", alpha=0.7)
            ax.annotate(f"best val={best:.2f} @ epoch {int(best_ep)}",
                         xy=(best_ep, best), xytext=(0.55, 0.95), textcoords="axes fraction",
                         fontsize=8, color="C1")
        ax.set_xlabel("epoch")
    else:
        d = load_cnn_nde_curve(tdir)
        if d is None:
            ax.text(0.5, 0.5, "(no CNN NDE loss found)", ha="center", transform=ax.transAxes); return
        val, steps = d["val"], d["steps"]
        bad = ~np.isfinite(val) | (np.abs(val) > 50)
        good = ~bad
        ax.plot(steps[good], val[good], lw=1.5, color="C1", label="val")
        if bad.any():
            ax.scatter(steps[bad], np.full(bad.sum(), 4.5),
                        marker="x", s=50, color="red", label=f"NaN/blow ({int(bad.sum())})")
        if good.any():
            best = np.nanmin(val[good])
            best_step = steps[good][np.nanargmin(val[good])]
            ax.axhline(best, color="C1", lw=0.5, linestyle=":", alpha=0.7)
            ax.annotate(f"best val={best:.2f} @ step {int(best_step)}",
                         xy=(best_step, best), xytext=(0.55, 0.95), textcoords="axes fraction",
                         fontsize=8, color="C1")
        ax.set_xlabel("NDE step")
    ax.set_title(f"NDE val-loss — {seed_label}")
    ax.set_ylabel("val loss")
    ax.set_ylim(-15, 5); ax.axhline(0, color="gray", lw=0.5, linestyle=":"); ax.grid(alpha=0.3); ax.legend(fontsize=8)


def panel_posterior_corner(ax, samples, seed_label):
    """Simple 1D posterior projection in a 3-panel grid; pass ax as a single ax that we sub-grid."""
    if samples is None:
        ax.text(0.5, 0.5, "(no posterior)", ha="center", transform=ax.transAxes); return
    # 3 stacked sub-axes for Omega_m, sigma_8, w0 1D marginals
    fig = ax.figure
    pos = ax.get_position()
    ax.remove()
    h = pos.height / 3
    sub_axes = []
    for i in range(3):
        sax = fig.add_axes([pos.x0, pos.y0 + (2-i)*h, pos.width, h * 0.9])
        sub_axes.append(sax)
    for i, (sax, pi, plab) in enumerate(zip(sub_axes, range(3), [r"$\Omega_m$", r"$\sigma_8$", r"$w_0$"])):
        sax.hist(samples[:, pi], bins=80, density=True, alpha=0.65)
        sax.axvline(TRUTH[pi], color="red", lw=1.0, label="truth")
        m = samples[:, pi].mean(); s = samples[:, pi].std()
        sax.axvline(m, color="C0", lw=1.0, linestyle=":", label=f"mean={m:.4f}")
        sax.set_ylabel(plab, rotation=0, ha="right")
        sax.set_xlabel(f"{plab} (bias = {(m-TRUTH[pi])/s:+.2f}σ)" if i == 0 else "")
        if i == 0: sax.set_title(f"1D marginals — {seed_label}")
        sax.legend(fontsize=7); sax.grid(alpha=0.3)


def panel_l1_datavector(ax, tdir, seed_label):
    """Plot the prior-averaged L1 datavector (mean) ± per-feature dispersion."""
    d = load_l1_standardization(tdir)
    if d is None:
        ax.text(0.5, 0.5, "(no L1 standardization NPZ)", ha="center", transform=ax.transAxes); return
    mean = d["mean"]; std = d["std"]
    idx = np.arange(len(mean))
    ax.plot(idx, mean, color="C0", lw=0.8, label="prior-mean")
    ax.fill_between(idx, mean - std, mean + std, alpha=0.25, color="C0", label="±1 std (cosmo dispersion)")
    # Channel boundary markers: divide len(mean) by N channels (will guess from arm)
    # heuristic: 4 channels → 200 each; 10 channels → 80 each
    n_features = len(mean)
    if n_features % 4 == 0 and n_features // 4 in (200, 100, 50):
        nch = 4
    elif n_features % 10 == 0:
        nch = 10
    else:
        nch = None
    if nch:
        for c in range(1, nch):
            ax.axvline(c * (n_features // nch), color="gray", lw=0.5, linestyle="--", alpha=0.5)
        ax.text(0.02, 0.97, f"{nch} channels × {n_features // nch} feats", transform=ax.transAxes,
                fontsize=7, va="top")
    ax.set_title(f"L1 datavector (prior-mean ± cosmo dispersion) — {seed_label}")
    ax.set_xlabel("feature index"); ax.set_ylabel("L1 value (post log1p, pre z-score)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


def panel_l1_feature_mask(ax, tdir, seed_label):
    d = load_l1_feature_mask(tdir)
    if d is None:
        ax.text(0.5, 0.5, "(no L1 feature mask)", ha="center", transform=ax.transAxes); return
    mask = np.asarray(d["valid_mask"]).astype(int) if "valid_mask" in d.files else np.asarray(d[d.files[0]]).astype(int)
    kept = int(mask.sum()); total = len(mask)
    ax.bar(np.arange(total), mask, width=1.0, color="gray")
    ax.set_title(f"L1 feature mask (variance gate, threshold {float(d.get('min_variance', d.get(d.files[1], 'n/a')))}) — kept {kept}/{total} ({100*kept/total:.0f}%)")
    ax.set_xlabel("feature index"); ax.set_ylabel("kept (1/0)")
    ax.set_ylim(-0.1, 1.1); ax.grid(alpha=0.3)


def panel_l1_summary_health(ax, tdir, seed_label):
    """Per-feature mean (the actual conditioning quantity after standardization)."""
    d = load_l1_standardization(tdir)
    if d is None:
        ax.text(0.5, 0.5, "(no L1 standardization NPZ)", ha="center", transform=ax.transAxes); return
    mean = d["mean"]; std = d["std"]
    # Histogram of mean values + histogram of std values
    ax.hist(mean, bins=40, alpha=0.6, label=f"per-feature mean (range {mean.min():.2f}—{mean.max():.2f})", color="C0")
    ax2 = ax.twinx()
    ax2.hist(std, bins=40, alpha=0.4, color="C3", label=f"per-feature std (range {std.min():.3f}—{std.max():.3f})")
    ax.set_title(f"L1 per-feature mean / std distribution — {seed_label}")
    ax.set_xlabel("value"); ax.set_ylabel("# features (mean, blue)")
    ax2.set_ylabel("# features (std, red)", color="C3")
    ax.legend(loc="upper left", fontsize=7); ax2.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.3)


def panel_bias_per_seed(ax, samples_by_seed, arm_label):
    """3 seeds × 3 params bias-vs-truth bar chart."""
    if not samples_by_seed:
        ax.text(0.5, 0.5, "(no posteriors)", ha="center", transform=ax.transAxes); return
    seeds = sorted(samples_by_seed)
    x = np.arange(len(seeds)); width = 0.27
    for j, (pi, plab) in enumerate([(0, r"$\Omega_m$"), (1, r"$\sigma_8$"), (2, r"$w_0$")]):
        bs = []
        for s in seeds:
            samp = samples_by_seed[s]
            m = samp[:, pi].mean(); st = samp[:, pi].std()
            bs.append((m - TRUTH[pi]) / st)
        ax.bar(x + (j-1)*width, bs, width, label=plab)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(0.5, color="gray", lw=0.5, linestyle=":")
    ax.axhline(-0.5, color="gray", lw=0.5, linestyle=":")
    ax.axhline(1.0, color="orange", lw=0.5, linestyle="--", alpha=0.6)
    ax.axhline(-1.0, color="orange", lw=0.5, linestyle="--", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels([f"s{s}" for s in seeds])
    ax.set_ylabel("bias (σ-units)")
    ax.set_title(f"Bias-vs-truth per seed — {arm_label}")
    ax.legend(fontsize=8, ncol=3); ax.grid(axis="y", alpha=0.3)


def panel_3seed_corner_overlay(ax, samples_by_seed, arm_label):
    """Just leave this as a placeholder we'll handle via getdist in a dedicated function."""
    ax.text(0.5, 0.5, "(see contour overlay panel)", ha="center", transform=ax.transAxes)


# ============================================================
# getdist contour helpers (separate figures)
# ============================================================
def make_3seed_contour_overlay(samples_by_seed, arm_label, color):
    try:
        from getdist import MCSamples, plots
    except ImportError:
        return None
    if not samples_by_seed:
        return None
    mcs = []
    for s in sorted(samples_by_seed):
        samp = samples_by_seed[s]
        f = fom3(samp)
        mcs.append(MCSamples(samples=samp, names=PARAM_NAMES,
                              labels=[l.strip("$") for l in PARAM_LABELS_PLAIN],
                              label=f"s{s} (FoM3 = {f:,.0f})",
                              settings={"smooth_scale_2D": 0.35, "smooth_scale_1D": 0.35}))
    g = plots.get_subplot_plotter(subplot_size=2.0)
    g.triangle_plot(mcs, params=["Omega_m", "sigma_8", "w0"], filled=True,
                     markers={"Omega_m": 0.26, "sigma_8": 0.84, "w0": -1.0})
    g.fig.suptitle(f"3-seed overlay — {arm_label}", y=1.02)
    return g.fig


def make_cross_arm_overlay(runs, arm_a, arm_b, title):
    try:
        from getdist import MCSamples, plots
    except ImportError:
        return None
    mc_list = []; col_list = []
    for arm in (arm_a, arm_b):
        if not runs[arm]: continue
        samples_list = [safe_load(runs[arm][s]["posterior"]) for s in sorted(runs[arm])]
        samples_list = [s for s in samples_list if s is not None]
        if not samples_list: continue
        pool = np.concatenate(samples_list, 0)
        f = fom3(pool)
        mc_list.append(MCSamples(samples=pool, names=PARAM_NAMES,
                                  labels=[l.strip("$") for l in PARAM_LABELS_PLAIN],
                                  label=f"{ARMS[arm]['label']} (FoM3 = {f:,.0f})",
                                  settings={"smooth_scale_2D": 0.35, "smooth_scale_1D": 0.35}))
        col_list.append(ARMS[arm]["color"])
    if len(mc_list) < 2:
        return None
    g = plots.get_subplot_plotter(subplot_size=2.2)
    g.triangle_plot(mc_list, params=["Omega_m", "sigma_8", "w0"], filled=True,
                     contour_colors=col_list,
                     markers={"Omega_m": 0.26, "sigma_8": 0.84, "w0": -1.0})
    g.fig.suptitle(title, y=1.02)
    return g.fig


def make_fom3_bars(runs):
    arms_present = [a for a in ARMS if runs[a]]
    if not arms_present:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(arms_present))
    pools, mosses, per_seeds = [], [], []
    for arm in arms_present:
        ps = [fom3(s) for s in (safe_load(runs[arm][k]["posterior"]) for k in sorted(runs[arm])) if s is not None]
        per_seeds.append(ps); mosses.append(np.mean(ps) if ps else 0)
        all_samps = [safe_load(runs[arm][k]["posterior"]) for k in sorted(runs[arm])]
        all_samps = [s for s in all_samps if s is not None]
        pools.append(fom3(np.concatenate(all_samps, 0)) if all_samps else 0)
    ax.bar(x, pools, color=[ARMS[a]["color"] for a in arms_present], alpha=0.55, label="3-seed pool")
    for i, ps in enumerate(per_seeds):
        ax.scatter([i]*len(ps), ps, color="k", s=40, zorder=3, label="per-seed" if i==0 else "")
    for i, (pool, mos) in enumerate(zip(pools, mosses)):
        ax.text(i, pool + max(pools)*0.02, f"pool={pool:,.0f}\nMoS={mos:,.0f}\nhc={pool/mos:.2f}",
                ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([ARMS[a]["label"] for a in arms_present], rotation=20, ha="right")
    ax.set_ylabel("FoM3 = 1/√det Cov₃")
    ax.set_title("Headline FoM3 — 3-seed pool, MoS, and per-seed; pool/MoS haircut annotated")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def make_bias_bars_summary(runs):
    arms_present = [a for a in ARMS if runs[a]]
    if not arms_present:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(arms_present)); width = 0.27
    for j, (pi, plab) in enumerate([(0, r"$\Omega_m$"), (1, r"$\sigma_8$"), (2, r"$w_0$")]):
        meds = []
        for arm in arms_present:
            bs = []
            for s in sorted(runs[arm]):
                samp = safe_load(runs[arm][s]["posterior"])
                if samp is None: continue
                m = samp[:, pi].mean(); st = samp[:, pi].std()
                bs.append((m - TRUTH[pi]) / st)
            meds.append(np.median(bs) if bs else 0)
        ax.bar(x + (j-1)*width, meds, width, label=plab)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(0.5, color="gray", lw=0.5, linestyle=":")
    ax.axhline(-0.5, color="gray", lw=0.5, linestyle=":")
    ax.set_xticks(x); ax.set_xticklabels([ARMS[a]["label"] for a in arms_present], rotation=20, ha="right")
    ax.set_ylabel("median bias (σ-units)")
    ax.set_title("Bias-vs-truth (median over seeds) per arm")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


# ============================================================
# Main: per-arm + per-seed page assembly
# ============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign_dir", type=Path)
    ap.add_argument("--out-prefix", type=str, default="canonical_diagnostics")
    args = ap.parse_args()

    runs = find_arm_runs(args.campaign_dir)
    n_runs = sum(len(seeds) for seeds in runs.values())
    print(f"\nFound {n_runs} posteriors across {sum(1 for v in runs.values() if v)} arms")
    for arm, seeds in runs.items():
        if seeds:
            print(f"  {ARMS[arm]['label']:<40}  seeds={sorted(seeds)}")
    if n_runs == 0:
        print("Nothing to plot."); return 1

    out_pdf = args.campaign_dir / f"{args.out_prefix}.pdf"
    png_root = args.campaign_dir / f"{args.out_prefix}_png"
    png_root.mkdir(exist_ok=True)
    overview_dir = png_root / "00_overview"; overview_dir.mkdir(exist_ok=True)

    with PdfPages(out_pdf) as pdf:
        # ====================== OVERVIEW SECTION ======================
        f = make_fom3_bars(runs)
        if f:
            pdf.savefig(f); f.savefig(overview_dir / "00_fom3_bars.png", dpi=140); plt.close(f)
        f = make_bias_bars_summary(runs)
        if f:
            pdf.savefig(f); f.savefig(overview_dir / "01_bias_bars.png", dpi=140); plt.close(f)
        # Cross-arm contour overlays
        for arm_a, arm_b, title, fname in [
            ("cnn_auto",  "l1_auto",                 "CNN vs L1 — auto-only",  "02_cross_arm_overlay_auto.png"),
            ("cnn_cross", "l1_cross",                "CNN vs L1 — auto+cross", "03_cross_arm_overlay_autocross.png"),
            ("cnn_auto",  "cnn_cross",               "CNN auto vs auto+cross", "04_cnn_auto_vs_cross.png"),
            ("l1_auto",   "l1_cross",                "L1 auto vs auto+cross",  "05_l1_auto_vs_cross.png"),
            ("cnn_auto",  "cnn_cache_auto_sanity",   "Sanity: CNN-TFDS-auto vs CNN-cache-auto", "06_sanity_tfds_vs_cache.png"),
        ]:
            if not (runs.get(arm_a) and runs.get(arm_b)): continue
            f = make_cross_arm_overlay(runs, arm_a, arm_b, title)
            if f:
                pdf.savefig(f, bbox_inches="tight"); f.savefig(overview_dir / fname, dpi=140, bbox_inches="tight"); plt.close(f)

        # ====================== PER-ARM SECTION ======================
        for arm, seeds_dict in runs.items():
            if not seeds_dict: continue
            arm_dir = png_root / arm; arm_dir.mkdir(exist_ok=True)
            is_l1 = arm.startswith("l1")
            seeds = sorted(seeds_dict)
            samples_by_seed = {s: safe_load(seeds_dict[s]["posterior"]) for s in seeds}
            samples_by_seed = {k: v for k, v in samples_by_seed.items() if v is not None}

            # === Per-arm 3-seed overlay panels ===
            if not is_l1:
                # CNN: compressor curve overlay
                fig, ax = plt.subplots(figsize=(10, 5))
                for s in seeds:
                    tdir = seeds_dict[s]["train_dir"]
                    if tdir is None: continue
                    d = load_cnn_compressor_curve(tdir)
                    if d is None: continue
                    val = d["val"]; tr = d["train"]
                    steps_v = np.linspace(0, len(tr), len(val))
                    finite = np.isfinite(val) & (np.abs(val) < 50)
                    ax.plot(steps_v[finite], val[finite], lw=1.2, label=f"s{s} val")
                ax.set_title(f"Compressor val-loss (3 seeds) — {ARMS[arm]['label']}")
                ax.set_xlabel("step"); ax.set_ylabel("VMIM loss"); ax.set_ylim(-20, 0)
                ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
                pdf.savefig(fig); fig.savefig(arm_dir / "01_compressor_loss_3seed.png", dpi=140); plt.close(fig)

            # NDE loss overlay (CNN or L1)
            fig, ax = plt.subplots(figsize=(10, 5))
            for s in seeds:
                tdir = seeds_dict[s]["train_dir"]
                if tdir is None: continue
                if is_l1:
                    d = load_l1_nde_curve(tdir)
                    if d is None: continue
                    ax.plot(d["epochs"], d["val"], lw=1.2, label=f"s{s} val")
                else:
                    d = load_cnn_nde_curve(tdir)
                    if d is None: continue
                    val, steps = d["val"], d["steps"]
                    good = np.isfinite(val) & (np.abs(val) < 50)
                    ax.plot(steps[good], val[good], lw=1.2, label=f"s{s} val")
            ax.set_title(f"NDE val-loss (3 seeds) — {ARMS[arm]['label']}")
            ax.set_xlabel("epoch" if is_l1 else "step"); ax.set_ylabel("val loss"); ax.set_ylim(-15, 5)
            ax.axhline(0, color="gray", lw=0.5, linestyle=":")
            ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
            pdf.savefig(fig); fig.savefig(arm_dir / "02_nde_loss_3seed.png", dpi=140); plt.close(fig)

            # 3-seed corner overlay (getdist)
            f = make_3seed_contour_overlay(samples_by_seed, ARMS[arm]["label"], ARMS[arm]["color"])
            if f:
                pdf.savefig(f, bbox_inches="tight"); f.savefig(arm_dir / "03_corner_3seed_overlay.png", dpi=140, bbox_inches="tight"); plt.close(f)

            # Bias per seed
            fig, ax = plt.subplots(figsize=(10, 4.5))
            panel_bias_per_seed(ax, samples_by_seed, ARMS[arm]["label"])
            plt.tight_layout(); pdf.savefig(fig); fig.savefig(arm_dir / "04_bias_per_seed.png", dpi=140); plt.close(fig)

            # L1-specific arm-level panels
            if is_l1:
                # 3-seed datavector overlay
                fig, axes = plt.subplots(len(seeds), 1, figsize=(11, 3*len(seeds)), squeeze=False, sharex=True)
                for ax, s in zip(axes[:, 0], seeds):
                    tdir = seeds_dict[s]["train_dir"]
                    if tdir is not None:
                        panel_l1_datavector(ax, tdir, f"s{s}")
                plt.tight_layout(); pdf.savefig(fig); fig.savefig(arm_dir / "05_l1_datavector_per_seed.png", dpi=140); plt.close(fig)

                # Feature mask (one seed is enough — same for all)
                fig, ax = plt.subplots(figsize=(10, 3))
                tdir = seeds_dict[seeds[0]]["train_dir"]
                if tdir is not None: panel_l1_feature_mask(ax, tdir, f"s{seeds[0]}")
                plt.tight_layout(); pdf.savefig(fig); fig.savefig(arm_dir / "06_l1_feature_mask.png", dpi=140); plt.close(fig)

                # Per-feature mean/std distribution
                fig, ax = plt.subplots(figsize=(10, 4))
                tdir = seeds_dict[seeds[0]]["train_dir"]
                if tdir is not None: panel_l1_summary_health(ax, tdir, f"s{seeds[0]}")
                plt.tight_layout(); pdf.savefig(fig); fig.savefig(arm_dir / "07_l1_summary_health.png", dpi=140); plt.close(fig)

            # === Per-seed pages ===
            for s in seeds:
                seed_dir = arm_dir / f"seed_{s}"; seed_dir.mkdir(exist_ok=True)
                tdir = seeds_dict[s]["train_dir"]
                seed_label = f"{ARMS[arm]['label']} — seed {s}"

                # A: compressor (CNN) or skip (L1)
                if not is_l1 and tdir is not None:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    panel_cnn_compressor(ax, tdir, seed_label)
                    plt.tight_layout(); pdf.savefig(fig); fig.savefig(seed_dir / "A_compressor_loss.png", dpi=140); plt.close(fig)

                # B: NDE
                if tdir is not None:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    panel_nde_loss(ax, tdir, seed_label, is_l1)
                    plt.tight_layout(); pdf.savefig(fig); fig.savefig(seed_dir / "B_nde_loss.png", dpi=140); plt.close(fig)

                # C: posterior corner
                samp = samples_by_seed.get(s)
                if samp is not None:
                    fig = plt.figure(figsize=(8, 8))
                    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
                    panel_posterior_corner(ax, samp, seed_label)
                    pdf.savefig(fig); fig.savefig(seed_dir / "C_corner_1d.png", dpi=140); plt.close(fig)

                # D: L1 datavector (L1 only)
                if is_l1 and tdir is not None:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    panel_l1_datavector(ax, tdir, seed_label)
                    plt.tight_layout(); pdf.savefig(fig); fig.savefig(seed_dir / "D_l1_datavector.png", dpi=140); plt.close(fig)

                # E: summary stats text card
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.axis("off")
                if samp is not None:
                    m = samp[:, :3].mean(0); st = samp[:, :3].std(0); b = (m - TRUTH[:3]) / st
                    f3 = fom3(samp)
                    meta = load_meta(seeds_dict[s]["meta"])
                    txt = [
                        f"=== {seed_label} ===", "",
                        f"FoM3 = {f3:,.0f}",
                        f"Ωₘ mean = {m[0]:.4f}  std = {st[0]:.4f}  bias = {b[0]:+.2f}σ",
                        f"σ₈ mean = {m[1]:.4f}  std = {st[1]:.4f}  bias = {b[1]:+.2f}σ",
                        f"w₀ mean = {m[2]:+.4f}  std = {st[2]:.4f}  bias = {b[2]:+.2f}σ",
                        "",
                        f"NaN samples: {int(np.isnan(samp).any())}",
                        f"npe_split_seed: {meta.get('npe_split_seed', 'n/a')}",
                        f"compressor_train_split: {meta.get('compressor_train_split', 'n/a')}",
                        f"nde_train_split: {meta.get('nde_train_split', 'n/a')}",
                        f"npe_epochs: {meta.get('npe_epochs', 'n/a')}",
                    ]
                    ax.text(0.05, 0.95, "\n".join(txt), transform=ax.transAxes,
                             fontsize=10, family="monospace", va="top")
                plt.tight_layout(); pdf.savefig(fig); fig.savefig(seed_dir / "E_summary_stats.png", dpi=140); plt.close(fig)

    print(f"\nWrote {out_pdf}")
    print(f"PNGs under {png_root}/00_overview/ and {png_root}/<arm>/seed_<N>/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
