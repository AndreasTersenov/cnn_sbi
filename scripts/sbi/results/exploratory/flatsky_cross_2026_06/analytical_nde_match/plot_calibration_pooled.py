#!/usr/bin/env python3
"""POOLED calibration overlay (NO tercile split): l1+product RealNVP vs CNN ResNet18 RealNVP.
Pools all gate dumps (every tercile + seed) into one set per arm — same pooling the gate's verdict
uses — then: (a) one TARP-DRP coverage curve per arm; (b) SBC rank histogram per parameter.

TARP: ECP vs alpha; on diagonal = calibrated, below = over-confident. SBC: flat = calibrated,
U-shape = over-confident; uniform rank-std = 0.289. CPU-only (reads dumps).
Outputs: tarp_pooled_l1_vs_cnn.{png,pdf}, sbc_pooled_l1_vs_cnn.{png,pdf}
"""
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tarp
from pathlib import Path

ROOT = "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/flatsky_cross_2026_06"
HERE = Path(ROOT) / "analytical_nde_match"
PARAMS = [r"$\Omega_m$", r"$\sigma_8$", r"$w_0$"]

ARMS = {
    "l1+product → RealNVP (FoM3 3270)": dict(
        color="#d62728",
        dumps=f"{HERE}/gate_l1product_rnvp/tarp_drp/dumps/l1product_rnvp_*/seed_*/n*_m*/posterior_samples.npz"),
    "CNN ResNet18 → RealNVP (FoM3 3293)": dict(
        color="#1f77b4",
        dumps=f"{ROOT}/cnn_phase/arch_sweep_2026_06_13/gate_c_resnet18/dumps/resnet18_rnvp_*/seed_*/n*_m*/posterior_samples.npz"),
}


def pool(pattern):
    S, T = [], []
    for f in sorted(glob.glob(pattern)):
        z = np.load(f)
        S.append(np.asarray(z["samples"], np.float32)[:, :, :3])  # (n, M, 3)
        T.append(np.asarray(z["theta"], np.float32)[:, :3])       # (n, 3)
    if not S:
        return None, None
    return np.concatenate(S, 0), np.concatenate(T, 0)


# ---- TARP (one pooled curve per arm) ----
fig, ax = plt.subplots(figsize=(6.2, 6.0))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect", zorder=1)
ax.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05], color="grey", alpha=0.13, zorder=0,
                label="±0.05 PASS zone")
for i, (name, cfg) in enumerate(ARMS.items()):
    s, t = pool(cfg["dumps"])
    if s is None:
        print("no dumps for", name); continue
    ecp, alpha = tarp.get_tarp_coverage(np.transpose(s, (1, 0, 2)), t, references="random",
                                        num_bootstrap=200, norm=True, bootstrap=True)
    m = ecp.mean(0); lo, hi = np.percentile(ecp, [16, 84], axis=0)
    ax.plot(alpha, m, color=cfg["color"], lw=2.2, label=name, zorder=3)
    ax.fill_between(alpha, lo, hi, color=cfg["color"], alpha=0.22, zorder=2)
    worst = float(m[np.argmax(np.abs(m - alpha))] - alpha[np.argmax(np.abs(m - alpha))])
    net = float(np.trapz(m - alpha, alpha) * 2)
    ax.text(0.04, 0.93 - 0.06 * i, f"worst dev {worst:+.3f} | net {net:+.3f}",
            transform=ax.transAxes, color=cfg["color"], fontsize=9)
    print(f"{name}: N_sims={s.shape[0]} worst {worst:+.3f} net {net:+.3f}")
ax.set_xlabel(r"credibility level $\alpha$"); ax.set_ylabel("expected coverage probability (ECP)")
ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(loc="lower right", fontsize=8.5)
ax.set_title("TARP-DRP coverage (pooled over all terciles+seeds)\n"
             "on diagonal = calibrated · below = over-confident", fontsize=10.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"tarp_pooled_l1_vs_cnn.{ext}", dpi=150, bbox_inches="tight")
print("wrote tarp_pooled_l1_vs_cnn.png")
plt.close(fig)


# ---- SBC (pooled rank histograms, one panel per param) ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for d, (ax, pname) in enumerate(zip(axes, PARAMS)):
    ax.axhline(1.0, color="k", ls="--", lw=1, label="uniform (calibrated)")
    for name, cfg in ARMS.items():
        s, t = pool(cfg["dumps"])
        if s is None:
            continue
        ranks = (s < t[:, None, :]).mean(1)  # (N, 3)
        ax.hist(ranks[:, d], bins=20, range=(0, 1), density=True, histtype="step",
                color=cfg["color"], lw=2,
                label=f"{name.split(' (')[0]} (std {ranks[:, d].std():.3f})")
    ax.set_title(pname); ax.set_xlabel("SBC rank"); ax.set_xlim(0, 1)
axes[0].set_ylabel("density"); axes[0].legend(loc="upper center", fontsize=7.5)
fig.suptitle("SBC rank histograms (pooled over all terciles+seeds): l1+product vs CNN\n"
             "flat = calibrated · U-shape = over-confident · central hump = under-confident · "
             "uniform rank-std = 0.289", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.92])
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"sbc_pooled_l1_vs_cnn.{ext}", dpi=150, bbox_inches="tight")
print("wrote sbc_pooled_l1_vs_cnn.png")
