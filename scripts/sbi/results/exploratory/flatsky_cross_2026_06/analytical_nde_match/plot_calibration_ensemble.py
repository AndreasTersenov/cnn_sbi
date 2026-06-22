#!/usr/bin/env python3
"""Pooled TARP + SBC overlay with the COMPRESSOR-ENSEMBLE joint ℓ1 (the properly-calibrated arm).
l1+product (single) vs joint ℓ1 (3-compressor ensemble, pooled per-obs) vs CNN (single).
CPU-only (reads gate dumps). Outputs: tarp_pooled_ensemble_3arm.{png,pdf}, sbc_pooled_ensemble_3arm.{png,pdf}
"""
import glob
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tarp
from pathlib import Path

ROOT = "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/flatsky_cross_2026_06"
HERE = Path(ROOT) / "analytical_nde_match"
PARAMS = [r"$\Omega_m$", r"$\sigma_8$", r"$w_0$"]

SINGLE = {
    "l1+product → RealNVP": ("#d62728",
        f"{HERE}/gate_l1product_rnvp/tarp_drp/dumps/l1product_rnvp_*/seed_*/n*_m*/posterior_samples.npz"),
    "CNN ResNet18 → RealNVP": ("#1f77b4",
        f"{ROOT}/cnn_phase/arch_sweep_2026_06_13/gate_c_resnet18/dumps/resnet18_rnvp_*/seed_*/n*_m*/posterior_samples.npz"),
}
ENS_LABEL = "joint ℓ1 (3-compressor ensemble) → RealNVP"
ENS_COLOR = "#2ca02c"
ENS_ARMS = ["jointl1_nobnt", "jointl1_nobnt_s42", "jointl1_nobnt_s43"]
# draw order: loosest under, tightest on top
ORDER = ["l1+product → RealNVP", "CNN ResNet18 → RealNVP", ENS_LABEL]


def pool_single(pattern):
    S, T = [], []
    for f in sorted(glob.glob(pattern)):
        z = np.load(f)
        S.append(np.asarray(z["samples"], np.float32)[:, :, :3])
        T.append(np.asarray(z["theta"], np.float32)[:, :3])
    return (np.concatenate(S, 0), np.concatenate(T, 0)) if S else (None, None)


def _key(f):
    m = re.search(r"dumps/(.+?)_(LOW|MID|HIGH)/seed_(\d+)/", f)
    return (m.group(2), m.group(3)) if m else None


def pool_ensemble(arm_dirs):
    """Concatenate the arms' samples PER (tercile,seed,obs) -> compressor-ensemble posterior."""
    per_arm = []
    for adir in arm_dirs:
        d = {}
        for f in glob.glob(f"{HERE}/{adir}/gate/tarp_drp/dumps/*/seed_*/n*_m*/posterior_samples.npz"):
            k = _key(f)
            if k:
                z = np.load(f)
                d[k] = (np.asarray(z["samples"], np.float32)[:, :, :3], np.asarray(z["theta"], np.float32)[:, :3])
        per_arm.append(d)
    keys = sorted(set.intersection(*[set(d) for d in per_arm]))
    S, T = [], []
    for k in keys:
        S.append(np.concatenate([d[k][0] for d in per_arm], axis=1))   # pool samples across arms
        T.append(per_arm[0][k][1])
    return np.concatenate(S, 0), np.concatenate(T, 0)


def get(label):
    if label == ENS_LABEL:
        return pool_ensemble(ENS_ARMS), ENS_COLOR
    color, pat = SINGLE[label]
    return pool_single(pat), color


# ---- TARP ----
fig, ax = plt.subplots(figsize=(6.4, 6.2))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect", zorder=1)
ax.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05], color="grey", alpha=0.13, zorder=0, label="±0.05 PASS zone")
for i, label in enumerate(ORDER):
    (s, t), color = get(label)
    if s is None:
        print("no dumps", label); continue
    ecp, alpha = tarp.get_tarp_coverage(np.transpose(s, (1, 0, 2)), t, references="random",
                                        num_bootstrap=200, norm=True, bootstrap=True)
    m = ecp.mean(0); lo, hi = np.percentile(ecp, [16, 84], axis=0)
    ax.plot(alpha, m, color=color, lw=2.2, label=label, zorder=3)
    ax.fill_between(alpha, lo, hi, color=color, alpha=0.20, zorder=2)
    worst = float(m[np.argmax(np.abs(m - alpha))] - alpha[np.argmax(np.abs(m - alpha))])
    net = float(np.trapz(m - alpha, alpha) * 2)
    ax.text(0.04, 0.94 - 0.055 * i, f"worst {worst:+.3f} | net {net:+.3f}", transform=ax.transAxes,
            color=color, fontsize=9)
    print(f"{label}: N={s.shape[0]} M={s.shape[1]} worst {worst:+.3f} net {net:+.3f}")
ax.set_xlabel(r"credibility level $\alpha$"); ax.set_ylabel("expected coverage probability (ECP)")
ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(loc="lower right", fontsize=8)
ax.set_title("TARP-DRP (pooled): l1+product vs joint ℓ1 (compressor-ensemble) vs CNN\n"
             "on diagonal = calibrated", fontsize=10)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"tarp_pooled_ensemble_3arm.{ext}", dpi=150, bbox_inches="tight")
print("wrote tarp_pooled_ensemble_3arm.png")
plt.close(fig)

# ---- SBC ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
cache = {label: get(label) for label in ORDER}
for d, (ax, pname) in enumerate(zip(axes, PARAMS)):
    ax.axhline(1.0, color="k", ls="--", lw=1, label="uniform (calibrated)")
    for label in ORDER:
        (s, t), color = cache[label]
        if s is None:
            continue
        ranks = (s < t[:, None, :]).mean(1)
        ax.hist(ranks[:, d], bins=20, range=(0, 1), density=True, histtype="step", color=color, lw=2,
                label=f"{label.split(' →')[0]} (std {ranks[:, d].std():.3f})")
    ax.set_title(pname); ax.set_xlabel("SBC rank"); ax.set_xlim(0, 1)
axes[0].set_ylabel("density"); axes[0].legend(loc="upper center", fontsize=7)
fig.suptitle("SBC rank histograms (pooled): l1+product vs joint ℓ1 (compressor-ensemble) vs CNN\n"
             "flat = calibrated · uniform rank-std = 0.289", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.92])
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"sbc_pooled_ensemble_3arm.{ext}", dpi=150, bbox_inches="tight")
print("wrote sbc_pooled_ensemble_3arm.png")
