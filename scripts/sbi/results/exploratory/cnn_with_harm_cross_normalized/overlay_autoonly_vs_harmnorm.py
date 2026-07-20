#!/usr/bin/env python3
"""Overlay corner plot: auto-only CNN baseline vs harmonic 10-ch normalized CNN.

Three arms (seeds pooled within each):
  1. Auto-only plain CNN (5 seeds, zero-mean, run_b_advanced_plain, no-BNT)
  2. Harmonic 10-ch normalized plain CNN (3 seeds, A3-norm, no-BNT)
  3. A2 extended plain CNN (5 seeds, dense512, 240k steps, zero-mean, no-BNT)

Output: overlay_autoonly_vs_harmnorm.{pdf,png}
"""
import numpy as np
from pathlib import Path
from getdist import MCSamples, plots

REPO = Path(__file__).resolve().parents[5]
OUT  = Path(__file__).resolve().parent

PARAM_NAMES  = ["Omega_m", "sigma_8", "w0", "h0", "n_s", "Omega_b"]
PARAM_LABELS = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]
TRUTH        = [0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493]

def fom3(s):
    c = np.cov(s[:, :3].T)
    return 1.0 / np.sqrt(np.linalg.det(c))

# ── 1. Auto-only baseline (plain CNN, run_b, no-BNT, 5 seeds) ─────────────────
baseline_dir = (
    REPO / "scripts/sbi/results/exploratory"
    / "zero_mean_maps_parity_check/run_b_advanced_plain/posteriors"
)
baseline_files = sorted(baseline_dir.glob(
    "cnn_tomo4_20deg160_nobnt_advanced_arch64_dense256_nostd_long_zm_s4?.npy"
))
assert len(baseline_files) == 5, f"Expected 5 baseline files, got {len(baseline_files)}"
baseline_samples = np.concatenate([np.load(f) for f in baseline_files], axis=0)
fom_baseline = fom3(baseline_samples)

# ── 2. Harmonic 10-ch normalized plain CNN (A3-norm, 3 seeds) ─────────────────
harmnorm_dir = Path(__file__).resolve().parent / "posteriors"
harmnorm_files = sorted(harmnorm_dir.glob("cnn_harm_cross_norm_nobnt_s4?.npy"))
assert len(harmnorm_files) == 3, f"Expected 3 A3-norm files, got {len(harmnorm_files)}"
harmnorm_samples = np.concatenate([np.load(f) for f in harmnorm_files], axis=0)
fom_harmnorm = fom3(harmnorm_samples)

# ── 3. A2 extended plain CNN (5 seeds, dense512, 240k steps) ──────────────────
a2_dir = (
    REPO / "scripts/sbi/results/exploratory/cnn_extended_train_zm/posteriors"
)
a2_files = sorted(a2_dir.glob("cnn_tomo4_20deg160_nobnt_a2_plain_dense512_*_zm_s4?.npy"))
assert len(a2_files) == 5, f"Expected 5 A2 files, got {len(a2_files)}"
a2_samples = np.concatenate([np.load(f) for f in a2_files], axis=0)
fom_a2 = fom3(a2_samples)

# Per-seed FoM3 means (more meaningful than pooled)
fom_baseline_perseed  = np.mean([fom3(np.load(f)) for f in baseline_files])
fom_harmnorm_perseed  = np.mean([fom3(np.load(f)) for f in harmnorm_files])
fom_a2_perseed        = np.mean([fom3(np.load(f)) for f in a2_files])

print(f"Auto-only baseline   ({len(baseline_files)} seeds):  FoM3 pooled={fom_baseline:,.0f}  mean-per-seed={fom_baseline_perseed:,.0f}")
print(f"Harmonic 10-ch norm  ({len(harmnorm_files)} seeds):  FoM3 pooled={fom_harmnorm:,.0f}  mean-per-seed={fom_harmnorm_perseed:,.0f}")
print(f"A2 extended CNN      ({len(a2_files)} seeds):  FoM3 pooled={fom_a2:,.0f}  mean-per-seed={fom_a2_perseed:,.0f}")
print(f"Harmnorm / baseline (per-seed means): {fom_harmnorm_perseed / fom_baseline_perseed:.2f}×")

# ── GetDist objects ───────────────────────────────────────────────────────────
smooth = {"smooth_scale_2D": 0.35, "smooth_scale_1D": 0.35}

mc_baseline = MCSamples(
    samples=baseline_samples,
    names=PARAM_NAMES, labels=PARAM_LABELS,
    label=rf"CNN auto-only (5 seeds, $\langle$FoM3$\rangle$={fom_baseline_perseed:,.0f})",
    settings=smooth,
)
mc_harmnorm = MCSamples(
    samples=harmnorm_samples,
    names=PARAM_NAMES, labels=PARAM_LABELS,
    label=rf"CNN harm 10-ch norm (3 seeds, $\langle$FoM3$\rangle$={fom_harmnorm_perseed:,.0f})",
    settings=smooth,
)
mc_a2 = MCSamples(
    samples=a2_samples,
    names=PARAM_NAMES, labels=PARAM_LABELS,
    label=rf"CNN extended train (5 seeds, $\langle$FoM3$\rangle$={fom_a2_perseed:,.0f})",
    settings=smooth,
)

# ── Plot ──────────────────────────────────────────────────────────────────────
g = plots.get_subplot_plotter(subplot_size=2.0)
g.settings.axes_fontsize    = 11
g.settings.lab_fontsize      = 13
g.settings.legend_fontsize   = 10
g.settings.figure_legend_loc = "upper right"

g.triangle_plot(
    [mc_baseline, mc_a2, mc_harmnorm],
    filled=True,
    contour_colors=["steelblue", "seagreen", "darkorange"],
    contour_lws=[1.2, 1.2, 1.2],
    legend_loc="upper right",
)

# ── Truth markers ─────────────────────────────────────────────────────────────
n = len(PARAM_NAMES)
for row in range(n):
    for col in range(n):
        ax = g.subplots[row][col]
        if ax is None:
            continue
        if row == col:
            ax.axvline(TRUTH[row], color="black", ls="--", lw=1.0, alpha=0.8)
        elif row > col:
            ax.axvline(TRUTH[col], color="black", ls="--", lw=0.8, alpha=0.7)
            ax.axhline(TRUTH[row], color="black", ls="--", lw=0.8, alpha=0.7)

g.fig.suptitle(
    "CNN: auto-only (4 ch) vs extended training vs harmonic 10-ch normalized\n"
    "(no-BNT, fiducial cosmology, seeds pooled)",
    fontsize=12,
    y=1.01,
)

for fmt in ("pdf", "png"):
    out_path = OUT / f"overlay_autoonly_vs_harmnorm.{fmt}"
    g.export(str(out_path))
    print(f"Saved: {out_path}")
