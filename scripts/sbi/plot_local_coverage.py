#!/usr/bin/env python3
"""D7 — per-patch local calibration vs latitude (auto+cross).

For each patch (180 indices), across its 50 perms (realizations), compute the std
of the marginal pull z = (posterior_mean - truth)/posterior_sigma, per parameter.
For a calibrated posterior z-std = 1 (the nominal-68% interval covers 68%); z-std<1
means the posterior over-covers (conservative). Plotting z-std vs latitude shows
(a) whether calibration depends on sky geometry (flat = no), and (b) that CNN sits
below the calibrated line (conservative) while L1 sits on it. CPU-only.
"""
import os, csv
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AN = "results/exploratory/definitive_comparison_10deg/phase_c/analysis"
OUT = f"{AN}/figs"
os.makedirs(OUT, exist_ok=True)
PARAMS = ["Omega_m", "sigma_8", "w_0"]
PLAB = [r"$\Omega_m$", r"$\sigma_8$", r"$w_0$"]
ARMC = {"l1_auto_cross": "#2ca02c", "cnn_auto_cross": "#1f77b4"}
ARML = {"l1_auto_cross": "L1 a+c", "cnn_auto_cross": "CNN a+c"}


def per_patch(arm):
    rows = list(csv.DictReader(open(f"{AN}/geometry/{arm}/per_patch_grid.csv")))
    patch = np.array([int(r["patch"]) for r in rows])
    lat = np.array([float(r["lat"]) for r in rows])
    pull = {p: np.array([float(r[f"pull_{p}"]) for r in rows]) for p in PARAMS}
    ids = np.unique(patch)
    plat = np.array([lat[patch == i][0] for i in ids])
    zstd = {p: np.array([np.nanstd(pull[p][patch == i]) for i in ids]) for p in PARAMS}
    return plat, zstd


def main():
    D = {a: per_patch(a) for a in ARMC}
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for j, p in enumerate(PARAMS):
        a = ax[j]
        a.axhline(1.0, color="k", lw=1.2, ls="--", zorder=1, label="calibrated (z-std=1)")
        a.axhspan(0.0, 1.0, color="0.92", zorder=0)  # conservative region
        for arm in ARMC:
            plat, zstd = D[arm]
            r = np.corrcoef(plat, zstd[p])[0, 1]
            a.scatter(plat, zstd[p], s=16, color=ARMC[arm], alpha=0.6, zorder=3,
                      label=f"{ARML[arm]} (mean {zstd[p].mean():.2f}, corr$_{{lat}}$={r:+.2f})")
            a.axhline(zstd[p].mean(), color=ARMC[arm], lw=1.4, ls=":", zorder=2)
        a.set_xlabel("patch latitude [deg]"); a.set_title(PLAB[j])
        a.grid(alpha=0.2); a.set_ylim(0.4, 1.4)
        a.legend(fontsize=8, loc="upper center")
        if j == 0:
            a.set_ylabel("per-patch z-std  (across 50 perms)")
    fig.suptitle("D7 — per-patch local calibration vs latitude (auto+cross): flat in latitude "
                 "= no geometry dependence;\nCNN sits in the grey (conservative) band, L1 on the "
                 "calibrated line. Tightness ≠ over-confidence.", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/D7_local_coverage_vs_latitude.{ext}", dpi=140, bbox_inches="tight")
    print("wrote D7_local_coverage_vs_latitude")


if __name__ == "__main__":
    main()
