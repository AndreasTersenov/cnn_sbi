#!/usr/bin/env python
"""Diagnostics for the L1 wavelet-ℓ₁ datavectors (Phase-C auto+cross cache, 10°).

Datavector layout (compute_l1_batch): index = channel*200 + scale*40 + bin,
i.e. reshape (N, 10 channels, 5 scales, 40 SNR-bins). Channels 0-3 = auto κ1..κ4,
4-9 = cross (1,2),(1,3),(1,4),(2,3),(2,4),(3,4). Each (cosmo) datavector is averaged
over its perms×patches in the NDE-train cache.

Figures:
  A  L1(SNR) per wavelet scale × representative channel, one line per cosmology
     coloured by σ8  -> how the ℓ₁ statistic responds to σ8 at each scale.
  B  same, coloured by w0 (the headline parameter).
  C  sensitivity: corr(ℓ₁ bin, parameter) vs SNR-bin, per scale, auto vs cross
     -> WHERE in the datavector the Ωm/σ8/w0 information lives.
CPU-only.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

PC = "results/exploratory/definitive_comparison_10deg/phase_c"
OUT = f"{PC}/analysis/l1_diag"
os.makedirs(OUT, exist_ok=True)
NCH, NSC, NBIN = 10, 5, 40
CH_LABEL = [r"$\kappa_1$", r"$\kappa_2$", r"$\kappa_3$", r"$\kappa_4$",
            r"$\kappa_{1\times2}$", r"$\kappa_{1\times3}$", r"$\kappa_{1\times4}$",
            r"$\kappa_{2\times3}$", r"$\kappa_{2\times4}$", r"$\kappa_{3\times4}$"]
SHOW_CH = [0, 3, 9]  # κ1 (low-z auto), κ4 (high-z auto), κ3×4 (largest cross)


def load():
    d = np.load(f"{PC}/l1_auto_cross_cache/l1_train.npz")
    x, th = d["x"].astype(np.float64), d["theta"].astype(np.float64)
    s = np.load(f"{PC}/l1_auto_cross_cache/snr_calibration.npz")
    snr_auto = np.linspace(float(s["min_snr"]), float(s["max_snr"]), NBIN)
    snr_cross = np.linspace(float(s["min_snr_cross"]), float(s["max_snr_cross"]), NBIN)
    # average each cosmology over its perms×patches
    uniq, inv = np.unique(th, axis=0, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    inv_s, x_s = inv[order], x[order]
    bnd = np.searchsorted(inv_s, np.arange(len(uniq) + 1))
    Xc = np.stack([x_s[bnd[g]:bnd[g + 1]].mean(0) for g in range(len(uniq))])
    Xc = Xc.reshape(len(uniq), NCH, NSC, NBIN)
    print(f"  {len(uniq)} cosmologies; Xc {Xc.shape}")
    return Xc, uniq, snr_auto, snr_cross


def fig_colored(Xc, uniq, snr_auto, snr_cross, pidx, pname, fname):
    param = uniq[:, pidx]
    order = np.argsort(param)  # low values first (under), high on top
    norm = Normalize(param.min(), param.max())
    cmap = cm.viridis
    fig, ax = plt.subplots(NSC, len(SHOW_CH), figsize=(3.3 * len(SHOW_CH), 2.1 * NSC),
                           sharex="col")
    for r in range(NSC):
        for c, ch in enumerate(SHOW_CH):
            snr = snr_auto if ch < 4 else snr_cross
            a = ax[r, c]
            for i in order:
                a.plot(snr, Xc[i, ch, r, :], color=cmap(norm(param[i])), alpha=0.12, lw=0.5)
            a.set_yscale("log")
            a.grid(alpha=0.2)
            if r == 0:
                a.set_title(CH_LABEL[ch], fontsize=11)
            if c == 0:
                a.set_ylabel(f"scale j={r}\nℓ₁(SNR)", fontsize=9)
            if r == NSC - 1:
                a.set_xlabel("SNR")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, location="right", shrink=0.6, label=pname)
    fig.suptitle(f"L1 wavelet-ℓ₁ datavector per scale × channel — cosmologies coloured by {pname} "
                 f"(10°, per-cosmo mean over perms×patches)", fontsize=11)
    fig.savefig(f"{OUT}/{fname}.png", dpi=130, bbox_inches="tight")
    fig.savefig(f"{OUT}/{fname}.pdf", bbox_inches="tight")
    print(f"  wrote {OUT}/{fname}.png")


def fig_sensitivity(Xc, uniq, snr_auto, snr_cross):
    # Pearson corr across cosmologies between each ℓ₁ bin and each parameter.
    params = {"Omega_m": 0, "sigma_8": 1, "w_0": 2}
    Xz = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-30)  # [ncos,10,5,40]
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True, sharey=True)
    for c, (pname, pidx) in enumerate(params.items()):
        p = uniq[:, pidx]
        pz = (p - p.mean()) / (p.std() + 1e-30)
        corr = (Xz * pz[:, None, None, None]).mean(0)  # [10,5,40]
        for ri, (rows, lbl, snr) in enumerate(
                [(slice(0, 4), "auto (mean κ1-4)", snr_auto),
                 (slice(4, 10), "cross (mean 6)", snr_cross)]):
            cc = corr[rows].mean(0)  # [5,40]
            a = ax[ri, c]
            for j in range(NSC):
                a.plot(snr, cc[j], lw=1.3, label=f"j={j}")
            a.axhline(0, color="k", lw=0.5)
            a.grid(alpha=0.2)
            if ri == 0:
                a.set_title(pname, fontsize=12)
            if c == 0:
                a.set_ylabel(f"{lbl}\ncorr(ℓ₁, param)", fontsize=10)
            if ri == 1:
                a.set_xlabel("SNR")
    ax[0, 0].legend(fontsize=8, ncol=5, loc="upper center")
    fig.suptitle("L1 datavector sensitivity: corr(ℓ₁ bin, parameter) across cosmologies "
                 "— where the Ωm/σ8/w0 information lives", fontsize=12)
    fig.savefig(f"{OUT}/C_sensitivity.png", dpi=130, bbox_inches="tight")
    fig.savefig(f"{OUT}/C_sensitivity.pdf", bbox_inches="tight")
    print(f"  wrote {OUT}/C_sensitivity.png")


def fig_all_linear(Xc, uniq, snr_auto, snr_cross, pidx, pname, fname):
    """All 10 channels × 5 scales, LINEAR y, cosmologies coloured by `pname`."""
    param = uniq[:, pidx]
    order = np.argsort(param)
    norm = Normalize(param.min(), param.max())
    cmap = cm.viridis
    fig, ax = plt.subplots(NSC, NCH, figsize=(2.05 * NCH, 1.95 * NSC), sharex="col")
    for r in range(NSC):
        for ch in range(NCH):
            snr = snr_auto if ch < 4 else snr_cross
            a = ax[r, ch]
            for i in order:
                a.plot(snr, Xc[i, ch, r, :], color=cmap(norm(param[i])), alpha=0.12, lw=0.4)
            a.grid(alpha=0.2)
            a.tick_params(labelsize=6)
            if r == 0:
                a.set_title(CH_LABEL[ch], fontsize=10)
            if ch == 0:
                a.set_ylabel(f"scale j={r}\nℓ₁(SNR)", fontsize=8)
            if r == NSC - 1:
                a.set_xlabel("SNR", fontsize=8)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, location="right", shrink=0.5, label=pname)
    fig.suptitle(f"L1 wavelet-ℓ₁ datavector (LINEAR) — all 10 channels × 5 scales, "
                 f"cosmologies coloured by {pname} (10°, per-cosmo mean over perms×patches)",
                 fontsize=12)
    fig.savefig(f"{OUT}/{fname}.png", dpi=130, bbox_inches="tight")
    fig.savefig(f"{OUT}/{fname}.pdf", bbox_inches="tight")
    print(f"  wrote {OUT}/{fname}.pdf")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear10-only", action="store_true",
                    help="only the all-10-channel LINEAR σ8 figure")
    args = ap.parse_args()
    Xc, uniq, sa, sc = load()
    fig_all_linear(Xc, uniq, sa, sc, 1, r"$\sigma_8$", "A_all10_linear_by_sigma8")
    if args.linear10_only:
        return
    fig_colored(Xc, uniq, sa, sc, 1, r"$\sigma_8$", "A_datavector_by_sigma8")
    fig_colored(Xc, uniq, sa, sc, 2, r"$w_0$", "B_datavector_by_w0")
    fig_sensitivity(Xc, uniq, sa, sc)


if __name__ == "__main__":
    main()
