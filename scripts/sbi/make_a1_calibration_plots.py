#!/usr/bin/env python
"""A1 calibration, shown honestly: TARP-DRP (joint coverage) for the 3 VMIM compressor
seeds + the clean l1+product reference, AND SBC rank histograms (marginal calibration) for
A1 vs l1+product. From existing gate dumps/curves (CPU).

SBC ranks: for each val point, rank_p = fraction of posterior samples with theta_p < truth_p.
Calibrated -> uniform on [0,1] (flat); over-confident -> U-shaped (piles at 0,1);
conservative -> dome (piles at 0.5).
"""
import glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FC = ("/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/"
      "flatsky_cross_2026_06")
OUT = f"{FC}/overnight_menu_2/lane_a_plots"
PN = [r"$\Omega_m$", r"$\sigma_8$", r"$w_0$"]
SEEDS = (41, 42, 43)
# arm -> (gate_root, arm_key, color)
A1_SEEDS = {
    "A1 seed41 (net +0.021)": (f"{FC}/overnight_menu_2/gate_c", "A1_pair2d_vmim", "tab:red"),
    "A1 seed42 (+0.009)": (f"{FC}/overnight_menu_2/gate_c", "A1_vmim_s42", "tab:orange"),
    "A1 seed43 (−0.022)": (f"{FC}/overnight_menu_2/gate_c", "A1_vmim_s43", "tab:brown"),
}
REF = ("l1+product (clean, −0.015)", f"{FC}/gate_c", "flat_product", "tab:purple")


def tarp_pooled(gr, arm, terc=None):
    """Mean ECP over seeds (+terciles if terc=None) on a common alpha grid."""
    tercs = [terc] if terc else ["HIGH", "MID", "LOW"]
    curves = []
    for t in tercs:
        for s in SEEDS:
            f = f"{gr}/tarp_drp/curves/tarp_curve_{arm}_{t}_seed{s}_dim3.npz"
            if Path(f).exists():
                z = np.load(f)
                curves.append((np.asarray(z["alpha"]), np.asarray(z["ecp_bootstrap"]).mean(0)))
    if not curves:
        return None, None
    a0 = curves[0][0]
    return a0, np.mean([np.interp(a0, a, e) for a, e in curves], axis=0)


def sbc_ranks(gr, arm):
    rs = []
    for f in glob.glob(f"{gr}/tarp_drp/dumps/{arm}_*/seed_*/n*_m*/posterior_samples.npz"):
        z = np.load(f)
        rs.append((z["samples"] < z["theta"][:, None, :]).mean(1))
    return np.concatenate(rs, 0) if rs else None


def main():
    Path(OUT).mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13, 8))
    # --- row 1: TARP-DRP (HIGH tercile + all pooled) ---
    for col, (terc, title) in enumerate([("HIGH", "TARP HIGH-FoM3 tercile (tightest)"),
                                         (None, "TARP all terciles pooled")]):
        ax = fig.add_subplot(2, 3, col + 1)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="calibrated")
        for lab, (gr, arm, c) in A1_SEEDS.items():
            a, e = tarp_pooled(gr, arm, terc)
            if a is not None:
                ax.plot(a, e, color=c, lw=1.7, label=lab)
        a, e = tarp_pooled(REF[1], REF[2], terc)
        ax.plot(a, e, color=REF[3], lw=2.2, ls=(0, (4, 2)), label=REF[0])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("ECP"); ax.set_title(title, fontsize=9)
        ax.text(0.96, 0.05, "below diag =\nover-confident", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color="gray")
        if col == 0:
            ax.legend(fontsize=6.5, loc="upper left")
    # third top panel: net-bias-vs-seed summary
    ax = fig.add_subplot(2, 3, 3)
    labs = ["s41", "s42", "s43", "l1+prod"]
    vals = [0.021, 0.009, -0.022, -0.015]
    errs = [0.028, 0.022, 0.015, 0.035]
    cols = ["tab:red", "tab:orange", "tab:brown", "tab:purple"]
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.errorbar(range(4), vals, yerr=errs, fmt="o", color="k", ecolor="gray", capsize=3)
    for i, (v, c) in enumerate(zip(vals, cols)):
        ax.plot(i, v, "o", color=c, ms=9)
    ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("net TARP bias  ⟨ECP−α⟩"); ax.set_title("net coverage vs seed", fontsize=9)
    ax.text(0.5, 0.95, "+ conservative / − over-confident", transform=ax.transAxes,
            ha="center", va="top", fontsize=7, color="gray")
    # --- row 2: SBC rank histograms per science param (A1 pooled-3-seeds vs l1+product) ---
    a1_ranks = np.concatenate([sbc_ranks(gr, arm) for _, (gr, arm, _) in A1_SEEDS.items()], 0)
    ref_ranks = sbc_ranks(REF[1], REF[2])
    for p in range(3):
        ax = fig.add_subplot(2, 3, 4 + p)
        bins = np.linspace(0, 1, 21)
        ax.hist(a1_ranks[:, p], bins=bins, density=True, histtype="step", color="tab:red",
                lw=1.8, label=f"A1 (std {a1_ranks[:, p].std():.3f})")
        ax.hist(ref_ranks[:, p], bins=bins, density=True, histtype="step", color="tab:purple",
                lw=1.8, ls=(0, (4, 2)), label=f"l1+prod (std {ref_ranks[:, p].std():.3f})")
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        ax.set_title(f"SBC ranks {PN[p]}", fontsize=9); ax.set_xlabel("rank")
        ax.set_ylim(0, 1.8)
        ax.legend(fontsize=6.5, loc="lower center")
        if p == 0:
            ax.text(0.02, 1.7, "uniform=calibrated, U=over-confident, dome=conservative",
                    fontsize=6.5, color="gray")
    fig.suptitle("A1 (VMIM joint PDF) calibration — joint coverage (top) + marginal SBC "
                 "(bottom); uniform=0.289", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/a1_calibration.{ext}", dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}/a1_calibration.png/pdf", flush=True)
    print(f"A1 pooled SBC std (Om,s8,w0) = {np.round(a1_ranks.std(0)[:3], 3)}", flush=True)
    print(f"l1+product SBC std            = {np.round(ref_ranks.std(0)[:3], 3)}", flush=True)


if __name__ == "__main__":
    main()
