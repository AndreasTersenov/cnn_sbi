#!/usr/bin/env python3
"""D8 — L1's fiducial 'bias' is prior shrinkage, scaling with information content.

Left : per-arm mean bias vs (prior_mean - truth) for Om/s8/w0. Pure shrinkage =>
       points fall on a line through the origin with slope (1-r), r=information
       fraction (bias = (1-r)(prior_mean - truth)).
Right: |Om/s8 mean bias| vs median FoM3 across the 4 arms => bias is monotonic in
       information, NOT a property of L1 (CNN auto-only is more biased than L1 a+c).
CPU-only.
"""
import os, csv
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DC = "results/exploratory/definitive_comparison_10deg/phase_c"
GEO = f"{DC}/analysis/geometry"
OUT = f"{DC}/analysis/figs"
os.makedirs(OUT, exist_ok=True)
P = ["Omega_m", "sigma_8", "w_0"]
PLAB = {"Omega_m": r"$\Omega_m$", "sigma_8": r"$\sigma_8$", "w_0": r"$w_0$"}
FID = {"Omega_m": 0.26, "sigma_8": 0.84, "w_0": -1.0}
ARMS = ["cnn_auto_cross", "l1_auto_cross", "cnn_auto_only", "l1_auto_only"]
ARMC = {"cnn_auto_cross": "#1f77b4", "l1_auto_cross": "#2ca02c",
        "cnn_auto_only": "#aec7e8", "l1_auto_only": "#98df8a"}
ARML = {"cnn_auto_cross": "CNN a+c", "l1_auto_cross": "L1 a+c",
        "cnn_auto_only": "CNN auto", "l1_auto_only": "L1 auto"}
PMARK = {"Omega_m": "o", "sigma_8": "s", "w_0": "^"}


def prior_means():
    th = np.load(f"{DC}/cnn_auto_cross_s41/cache/cnn_train.npz")["theta"][:, :3]
    return {p: float(th[:, i].mean()) for i, p in enumerate(P)}


def arm_bias(arm):
    rows = list(csv.DictReader(open(f"{GEO}/{arm}/per_patch_grid.csv")))
    f3 = np.array([float(r["fom3"]) for r in rows])
    b = {p: np.nanmean([float(r[f"bias_{p}"]) for r in rows]) for p in P}
    return b, float(np.nanmedian(f3))


def main():
    pm = prior_means()
    off = {p: pm[p] - FID[p] for p in P}   # prior_mean - truth
    B = {a: arm_bias(a) for a in ARMS}

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # ---- left: bias vs (prior_mean - truth), shrinkage lines ----
    # r is fit on Om,s8 ONLY (pure shrinkage); w0 also carries a flat-sky term and
    # is shown as an open marker (the exception that proves the rule).
    a = ax[0]
    SHR = ["Omega_m", "sigma_8"]
    xs_shr = np.array([off[p] for p in SHR])
    xs_all = np.array([off[p] for p in P])
    for arm in ARMS:
        bs = np.array([B[arm][0][p] for p in SHR])
        r_compl = float(np.dot(bs, xs_shr) / np.dot(xs_shr, xs_shr))   # (1-r) from Om,s8
        xx = np.linspace(min(xs_all) * 1.2, max(xs_all) * 1.2, 10)
        a.plot(xx, r_compl * xx, color=ARMC[arm], lw=1.3, alpha=0.7, zorder=1)
        for p in P:
            face = ARMC[arm] if p in SHR else "white"
            a.scatter(off[p], B[arm][0][p], facecolor=face, edgecolor=ARMC[arm] if p not in SHR else "k",
                      marker=PMARK[p], s=70, lw=1.0, zorder=3)
        a.plot([], [], color=ARMC[arm], lw=6, alpha=0.6,
               label=f"{ARML[arm]}  (r≈{1 - r_compl:.2f})")
    a.axhline(0, color="0.6", lw=0.8); a.axvline(0, color="0.6", lw=0.8)
    for p in P:  # param-marker legend (w0 open = excluded from r fit)
        a.scatter([], [], facecolor=("0.4" if p in SHR else "white"), edgecolor="k",
                  marker=PMARK[p], s=70, lw=0.6, label=PLAB[p] + ("" if p in SHR else " (flat-sky)"))
    a.set_xlabel(r"prior mean $-$ truth  [param units]")
    a.set_ylabel("mean posterior bias  [param units]")
    a.set_title("Bias points toward the prior mean\n(slope = 1$-$r; r = information fraction)")
    a.legend(fontsize=8, ncol=2); a.grid(alpha=0.2)

    # ---- right: |bias| vs median FoM3 (information axis) ----
    a = ax[1]
    for p in ("Omega_m", "sigma_8"):
        fom = [B[arm][1] for arm in ARMS]
        bb = [abs(B[arm][0][p]) for arm in ARMS]
        order = np.argsort(fom)
        a.plot(np.array(fom)[order], np.array(bb)[order], "-", color="0.6", lw=1, zorder=1)
        for arm in ARMS:
            a.scatter(B[arm][1], abs(B[arm][0][p]), color=ARMC[arm], marker=PMARK[p],
                      s=80, edgecolor="k", lw=0.4, zorder=3)
        a.plot([], [], color="0.4", marker=PMARK[p], lw=1, label=f"|bias| {PLAB[p]}")
    for arm in ARMS:
        a.annotate(ARML[arm], (B[arm][1], abs(B[arm][0]["Omega_m"])),
                   fontsize=7.5, xytext=(4, 4), textcoords="offset points")
    a.set_xscale("log")
    a.set_xlabel("median FoM3  (information content) →"); a.set_ylabel("|mean bias|  [param units]")
    a.set_title("Bias shrinks as information grows\n(CNN auto-only MORE biased than L1 a+c ⇒ not an L1 property)")
    a.legend(fontsize=9); a.grid(alpha=0.2, which="both")

    fig.suptitle("D8 — L1's fiducial 'bias' is prior shrinkage (regression to the prior mean), "
                 "set by information content, not by compressor", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/D8_shrinkage.{ext}", dpi=140, bbox_inches="tight")
    print("wrote D8_shrinkage")
    # also dump the numbers for the summary
    print("prior_mean-truth:", {p: round(off[p], 4) for p in P})
    for arm in ARMS:
        bs = np.array([B[arm][0][p] for p in SHR]); r_compl = float(np.dot(bs, xs_shr) / np.dot(xs_shr, xs_shr))
        print(f"  {arm:16} medFoM3={B[arm][1]:6.0f}  r(Om,s8)={1-r_compl:.2f}  bias={{Om:{B[arm][0]['Omega_m']:+.4f}, s8:{B[arm][0]['sigma_8']:+.4f}, w0:{B[arm][0]['w_0']:+.4f}}}")


if __name__ == "__main__":
    main()
