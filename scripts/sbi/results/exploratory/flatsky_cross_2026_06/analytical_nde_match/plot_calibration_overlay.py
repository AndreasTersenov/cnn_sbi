#!/usr/bin/env python3
"""Overlay GATE-C calibration (TARP-DRP + SBC) of the new best l1 arm vs the best CNN arm.
CPU-only: reads EXISTING gate curves + posterior dumps (no retraining).

l1  arm = l1product_rnvp  (l1+product VMIM->sbi_lens RealNVP; FoM3 ~3270; PASS-with-caveat)
CNN arm = resnet18_rnvp   (ResNet18 VMIM->sbi_lens RealNVP; FoM3 3293; PASS)

Outputs: tarp_overlay_l1_vs_cnn.{png,pdf}, sbc_overlay_l1_vs_cnn.{png,pdf}
TARP: ECP vs alpha per FoM3 tercile (mean over seeds; band=min/max over seeds); diagonal=perfect,
below=over-confident. SBC: rank histograms per param, pooled over terciles+seeds; flat=calibrated,
U-shaped=over-confident, hump=under-confident; uniform std = 1/sqrt(12)=0.289.
"""
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/flatsky_cross_2026_06"
HERE = Path(ROOT) / "analytical_nde_match"

ARMS = {
    "l1+product → RealNVP (FoM3 3270)": dict(
        color="#d62728",
        curves=f"{HERE}/gate_l1product_rnvp/tarp_drp/curves/tarp_curve_l1product_rnvp_{{T}}_seed*_dim3.npz",
        dumps=f"{HERE}/gate_l1product_rnvp/tarp_drp/dumps/l1product_rnvp_*/seed_*/n*_m*/posterior_samples.npz",
    ),
    "CNN ResNet18 → RealNVP (FoM3 3293)": dict(
        color="#1f77b4",
        curves=f"{ROOT}/cnn_phase/arch_sweep_2026_06_13/gate_c_resnet18/curves/tarp_curve_resnet18_rnvp_{{T}}_seed*_dim3.npz",
        dumps=f"{ROOT}/cnn_phase/arch_sweep_2026_06_13/gate_c_resnet18/dumps/resnet18_rnvp_*/seed_*/n*_m*/posterior_samples.npz",
    ),
}
TERCILES = ["LOW", "MID", "HIGH"]
PARAMS = [r"$\Omega_m$", r"$\sigma_8$", r"$w_0$"]


def load_tarp_curves(pattern_T):
    """Return list of (alpha, ecp_mean_over_bootstrap) per seed file."""
    out = []
    for f in sorted(glob.glob(pattern_T)):
        z = np.load(f)
        a = np.asarray(z["alpha"]); e = np.asarray(z["ecp_bootstrap"]).mean(0)
        out.append((a, e))
    return out


# ---------------- TARP overlay (3 panels, one per tercile) ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
for ax, T in zip(axes, TERCILES):
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect", zorder=1)
    ax.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05], color="grey", alpha=0.12, zorder=0)
    for name, cfg in ARMS.items():
        curves = load_tarp_curves(cfg["curves"].format(T=T))
        if not curves:
            continue
        # common-ish alpha: use each curve's own alpha; interpolate to a fine grid for band
        ag = np.linspace(0, 1, 101)
        E = np.array([np.interp(ag, a, e) for a, e in curves])
        m = E.mean(0)
        ax.plot(ag, m, color=cfg["color"], lw=2, label=name, zorder=3)
        ax.fill_between(ag, E.min(0), E.max(0), color=cfg["color"], alpha=0.20, zorder=2)
        wd = float(np.max(np.abs(m - ag)))
        ax.text(0.05, 0.92 - 0.07 * list(ARMS).index(name), f"worst dev {wd:+.3f}",
                transform=ax.transAxes, color=cfg["color"], fontsize=8)
    ax.set_title(f"{T} FoM3 tercile"); ax.set_xlabel(r"credibility level $\alpha$")
    ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
axes[0].set_ylabel("expected coverage probability (ECP)")
axes[0].legend(loc="lower right", fontsize=8)
fig.suptitle("TARP-DRP calibration: l1+product RealNVP vs CNN ResNet18 RealNVP\n"
             "(curve below diagonal = over-confident; grey band = ±0.05 PASS zone; "
             "HIGH tercile = tightest posteriors, where over-confidence shows)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"tarp_overlay_l1_vs_cnn.{ext}", dpi=150, bbox_inches="tight")
print("wrote tarp_overlay_l1_vs_cnn.png")
plt.close(fig)


# ---------------- SBC overlay (3 panels, one per param) ----------------
def pool_ranks(pattern):
    rs = []
    for f in sorted(glob.glob(pattern)):
        z = np.load(f)
        rs.append((z["samples"] < z["theta"][:, None, :]).mean(1))  # (n, 6) rank in [0,1]
    return np.concatenate(rs, 0) if rs else None


fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
nbins = 20
for d, (ax, pname) in enumerate(zip(axes, PARAMS)):
    ax.axhline(1.0, color="k", ls="--", lw=1, label="uniform (calibrated)")
    for name, cfg in ARMS.items():
        r = pool_ranks(cfg["dumps"])
        if r is None:
            continue
        ax.hist(r[:, d], bins=nbins, range=(0, 1), density=True, histtype="step",
                color=cfg["color"], lw=2,
                label=f"{name.split(' (')[0]} (std {r[:, d].std():.3f})")
    ax.set_title(pname); ax.set_xlabel("SBC rank"); ax.set_xlim(0, 1)
axes[0].set_ylabel("density")
axes[0].legend(loc="upper center", fontsize=7.5)
fig.suptitle("SBC rank histograms (pooled over terciles+seeds): l1+product vs CNN\n"
             "flat = calibrated · U-shape = over-confident · central hump = under-confident · "
             "uniform rank-std = 0.289", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.92])
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"sbc_overlay_l1_vs_cnn.{ext}", dpi=150, bbox_inches="tight")
print("wrote sbc_overlay_l1_vs_cnn.png")
