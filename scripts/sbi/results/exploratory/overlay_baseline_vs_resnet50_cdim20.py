#!/usr/bin/env python3
"""Overlay corner plot: CNN baseline (plain cdim=10, 5 seeds) vs resnet50 cdim=20 (seed 42).

Uses getdist for contour plotting.  Output: overlay_baseline_vs_resnet50_cdim20.{pdf,png}
"""
import numpy as np
from pathlib import Path
from getdist import MCSamples, plots

REPO = Path(__file__).resolve().parents[4]
OUT  = Path(__file__).resolve().parent

PARAM_NAMES  = ["Omega_m", "sigma_8", "w0", "h0", "n_s", "Omega_b"]
PARAM_LABELS = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]

# ── Baseline CNN (plain arch, cdim=10, zero-mean, 5 seeds) ────────────────────
baseline_dir = (
    REPO / "scripts/sbi/results/exploratory"
    / "zero_mean_maps_parity_check/run_b_advanced_plain/posteriors"
)
baseline_files = sorted(baseline_dir.glob("cnn_tomo4_20deg160_nobnt_advanced_arch64_*_zm_s4?.npy"))
assert len(baseline_files) == 5, f"Expected 5 baseline files, got {len(baseline_files)}: {baseline_files}"
baseline_samples = np.concatenate([np.load(f) for f in baseline_files], axis=0)

def fom3(s):
    c = np.cov(s[:, :3].T)
    return 1.0 / np.sqrt(np.linalg.det(c))

fom_baseline  = fom3(baseline_samples)
fom_resnet50  = None  # filled below

# ── ResNet-50 cdim=20 (seed 42) ───────────────────────────────────────────────
r50_path = (
    REPO / "scripts/sbi/results/exploratory/cnn_resnet50_zm_sweep"
    / "posteriors/cnn_resnet50_zm_nobnt_cdim20_s42.npy"
)
r50_samples = np.load(r50_path)
fom_resnet50 = fom3(r50_samples)

print(f"Baseline  (5 seeds, {len(baseline_samples):,} samples):  FoM3 = {fom_baseline:,.0f}")
print(f"ResNet-50 cdim=20 s42 ({len(r50_samples):,} samples):    FoM3 = {fom_resnet50:,.0f}")
print(f"Ratio: {fom_resnet50 / fom_baseline:.2f}×")

# ── GetDist objects ───────────────────────────────────────────────────────────
mc_baseline = MCSamples(
    samples=baseline_samples,
    names=PARAM_NAMES,
    labels=PARAM_LABELS,
    label=rf"CNN plain cdim=10 (5 seeds, FoM3={fom_baseline:,.0f})",
    settings={"smooth_scale_2D": 0.35, "smooth_scale_1D": 0.35},
)
mc_r50 = MCSamples(
    samples=r50_samples,
    names=PARAM_NAMES,
    labels=PARAM_LABELS,
    label=rf"CNN resnet50 cdim=20 s42 (FoM3={fom_resnet50:,.0f})",
    settings={"smooth_scale_2D": 0.35, "smooth_scale_1D": 0.35},
)

# ── Plot ──────────────────────────────────────────────────────────────────────
g = plots.get_subplot_plotter(subplot_size=2.0)
g.settings.axes_fontsize   = 11
g.settings.lab_fontsize     = 13
g.settings.legend_fontsize  = 10
g.settings.figure_legend_loc = "upper right"

g.triangle_plot(
    [mc_baseline, mc_r50],
    filled=True,
    contour_colors=["steelblue", "darkorange"],
    contour_lws=[1.2, 1.2],
    legend_loc="upper right",
)

# ── Truth markers ─────────────────────────────────────────────────────────────
# Fiducial: [Omega_m, sigma_8, w0, h0, n_s, Omega_b]
TRUTH = [0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493]
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
    f"CNN capacity: plain cdim=10 vs resnet50 cdim=20\n"
    f"(no-BNT, zero-mean maps, fiducial cosmology)",
    fontsize=12,
    y=1.01,
)

for fmt in ("pdf", "png"):
    out_path = OUT / f"overlay_baseline_vs_resnet50_cdim20.{fmt}"
    g.export(str(out_path))
    print(f"Saved: {out_path}")
