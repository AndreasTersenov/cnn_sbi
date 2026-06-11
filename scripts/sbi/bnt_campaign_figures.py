#!/usr/bin/env python3
"""Remaining BNT-campaign figures (CPU): inflation bars, marginal-sigma dumbbells,
SBC rank histograms, L-C2ST panel. Complements the corner overlays
(bnt_corner_overlays.py) and the TARP figures (run_tarp_coverage). All numbers read
from artifacts on disk — nothing hardcoded."""
import glob, json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = "/home/tersenov/.claude/skills/figure-polish/style/aa.mplstyle"
SBI = Path("/mnt/home/tersenov/software/cnn_sbi/scripts/sbi")
FC = SBI / "results/exploratory/flatsky_cross_2026_06"
BNT = FC / "bnt_campaign"
FIGS = BNT / "figures"
CNN_SEEDS = (41, 42, 43)
C_NOBNT, C_BNT = "#0072B2", "#D55E00"


def med(path):
    f = Path(path) / "median_summary.json"
    return json.load(open(f)) if f.exists() else None


def collect():
    """{(probe, op): {'nobnt': median-dict or per-seed list, 'bnt': ...}}"""
    out = {}
    for op in ("none", "product"):
        out[("l1", op)] = {
            "nobnt": [med(FC / f"population_sweep/flat_{op}")],
            "bnt": [med(BNT / f"population_sweep/l1_{op}")],
        }
        nob = [med(FC / f"cnn_phase/population_sweep/flat_{op}")] + \
              [med(FC / f"cnn_phase/multiseed/population_sweep/{op}_s{s}") for s in (42, 43)]
        bnt = [med(BNT / f"population_sweep/cnn_{op}_s{s}") for s in CNN_SEEDS]
        out[("cnn", op)] = {"nobnt": nob, "bnt": bnt}
    return out


def fig_inflation_bars(data):
    labels, no_v, b_v, no_err, b_err = [], [], [], [], []
    for probe in ("l1", "cnn"):
        for op in ("none", "product"):
            d = data[(probe, op)]
            nv = [x["fom3"] for x in d["nobnt"] if x]; bv = [x["fom3"] for x in d["bnt"] if x]
            labels.append(f"{probe.upper()}\n{'auto' if op == 'none' else '+product'}")
            no_v.append(np.mean(nv)); b_v.append(np.mean(bv))
            no_err.append(np.std(nv) if len(nv) > 1 else 0)
            b_err.append(np.std(bv) if len(bv) > 1 else 0)
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.087, 3.8))
    ax.bar(x - w/2, no_v, w, yerr=no_err, color=C_NOBNT, edgecolor="k", lw=0.5,
           capsize=3, label="no BNT")
    ax.bar(x + w/2, b_v, w, yerr=b_err, color=C_BNT, edgecolor="k", lw=0.5,
           capsize=3, label="BNT")
    for xi, (n, b) in enumerate(zip(no_v, b_v)):
        ax.text(xi + w/2, b * 1.25, f"{b/n:.2f}×", ha="center", fontsize=9, color=C_BNT)
    ax.set_yscale("log"); ax.set_ylabel(r"FoM$_3$ (pooled 9000-obs median)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("BNT inflation: per-channel L1 collapses, the CNN is (near-)lossless",
                 fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"fom3_bnt_inflation.{ext}", dpi=200)
    plt.close(fig); print("  wrote fom3_bnt_inflation.{png,pdf}")


def fig_sigma_dumbbell(data):
    pars = [("sigma_Om", r"$\sigma(\Omega_m)$"), ("sigma_s8", r"$\sigma(\sigma_8)$"),
            ("sigma_w0", r"$\sigma(w_0)$")]
    arms = [("l1", "none", "L1 auto"), ("l1", "product", "L1 +product"),
            ("cnn", "none", "CNN auto"), ("cnn", "product", "CNN +product")]
    fig, axes = plt.subplots(1, 3, figsize=(7.087, 2.9), sharey=True)
    for ax, (key, lab) in zip(axes, pars):
        for yi, (probe, op, alab) in enumerate(arms):
            d = data[(probe, op)]
            nv = np.mean([x[key] for x in d["nobnt"] if x])
            bv = np.mean([x[key] for x in d["bnt"] if x])
            ax.plot([nv, bv], [yi, yi], "-", color="0.6", lw=1.2, zorder=1)
            ax.plot(nv, yi, "o", color=C_NOBNT, ms=6, zorder=2)
            ax.plot(bv, yi, "o", color=C_BNT, ms=6, zorder=2)
        ax.set_yticks(range(len(arms))); ax.set_yticklabels([a[2] for a in arms])
        ax.set_xlabel(lab)
        ax.invert_yaxis()
    axes[0].plot([], [], "o", color=C_NOBNT, label="no BNT")
    axes[0].plot([], [], "o", color=C_BNT, label="BNT")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Marginal widths under BNT (pooled 9000-obs medians)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"sigma_bnt_dumbbell.{ext}", dpi=200)
    plt.close(fig); print("  wrote sigma_bnt_dumbbell.{png,pdf}")


def fig_sbc(data):
    arms = [("cnn", "none", "CNN auto"), ("cnn", "product", "CNN +product"),
            ("l1", "none", "L1 auto"), ("l1", "product", "L1 +product")]
    PN = [r"$\Omega_m$", r"$\sigma_8$", r"$w_0$"]
    fig, axes = plt.subplots(4, 3, figsize=(7.087, 7.5), sharex=True)
    nbins = 20
    for r, (probe, op, lab) in enumerate(arms):
        rs = []
        for f in sorted(glob.glob(str(BNT / f"gate_c/tarp_drp/dumps/bnt_{probe}_{op}_*/"
                                        "seed_*/n*_m*/posterior_samples.npz"))):
            z = np.load(f)
            rs.append((z["samples"] < z["theta"][:, None, :]).mean(axis=1))
        ranks = np.concatenate(rs, axis=0)
        n = ranks.shape[0]
        lo, hi = (np.array(
            [np.percentile(np.random.binomial(n, 1/nbins, 5000), q) for q in (0.5, 99.5)])
            / n * nbins)
        for c in range(3):
            ax = axes[r, c]
            ax.axhspan(lo, hi, color="0.92", zorder=0)
            ax.hist(ranks[:, c], bins=nbins, range=(0, 1), density=True,
                    color=C_BNT, edgecolor="k", lw=0.4, zorder=2)
            ax.axhline(1.0, color="k", ls=":", lw=0.8)
            if r == 0:
                ax.set_title(PN[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(lab, fontsize=9)
            ax.set_yticks([])
    axes[-1, 1].set_xlabel("SBC rank (uniform = calibrated; grey = 99% binomial band)")
    fig.suptitle("GATE C (BNT arms) — SBC rank histograms, science parameters", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"sbc_bnt_ranks.{ext}", dpi=200)
    plt.close(fig); print("  wrote sbc_bnt_ranks.{png,pdf}")


def fig_lc2st():
    arms = [("none", "CNN auto-only (BNT)"), ("product", "CNN auto+product (BNT)")]
    fig, axes = plt.subplots(1, 2, figsize=(7.087, 3.0))
    for ax, (op, lab) in zip(axes, arms):
        base = BNT / "gate_c/lc2st" / f"bnt_cnn_{op}" / f"bnt_cnn_{op}"
        d = np.load(base / "lc2st_results.npz")
        s = json.load(open(base / "lc2st_summary.json"))
        T_obs, T_null = d["T_obs"], d["T_null"].ravel()
        thr = np.percentile(T_null, 95)
        ax.hist(T_null, bins=40, density=True, color="0.80", label="permutation null")
        y = ax.get_ylim()[1]
        ax.plot(T_obs, np.full_like(T_obs, 0.04 * y), "|", color=C_BNT, ms=11, mew=1.4,
                label=r"observed $T(x_0)$")
        ax.axvline(np.median(T_obs), color=C_BNT, lw=1.6)
        ax.axvline(thr, color="k", ls="--", lw=1.0, label="p=0.05 threshold")
        ax.set_title(lab, fontsize=10)
        ax.text(0.97, 0.95,
                f"reject@p<0.05: {s['frac_reject_p05']*100:.0f}%\nmedian p = {s['median_p']:.2f}\n"
                f"self-test ST$_{{H1}}$ p = {s['gate']['st_h1_median_p']:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.6))
        ax.set_yticks([]); ax.set_xlabel("L-C2ST statistic $T$")
        ax.legend(fontsize=7, loc="center right", frameon=False)
    fig.suptitle("GATE C (BNT CNN arms) — L-C2ST local calibration at the fiducial", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"lc2st_bnt_cnn.{ext}", dpi=200)
    plt.close(fig); print("  wrote lc2st_bnt_cnn.{png,pdf}")


def main():
    try:
        plt.style.use(STYLE)
    except OSError:
        pass
    FIGS.mkdir(parents=True, exist_ok=True)
    data = collect()
    fig_inflation_bars(data)
    fig_sigma_dumbbell(data)
    fig_sbc(data)
    fig_lc2st()
    print("done")


if __name__ == "__main__":
    main()
