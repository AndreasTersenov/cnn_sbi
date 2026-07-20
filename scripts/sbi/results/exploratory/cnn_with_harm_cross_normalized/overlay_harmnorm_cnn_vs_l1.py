#!/usr/bin/env python3
"""Overlay corner plot: harmonic 10-ch normalized CNN vs harmonic L1 cross (no-BNT).

Arms (seeds pooled within each):
  1. CNN harmonic 10-ch normalized, plain arch, no-BNT  (3 seeds, A3-norm)
  2. L1 harmonic auto+cross, no-BNT                     (3 seeds)

NOTE: The L1 harmonic no-BNT FoM3 is potentially artefactual — B2 truth-check
found sigma_8 response inverted for off-fiducial cosmologies (SBC pending).
Contours shown here are at the fiducial only.

Output: overlay_harmnorm_cnn_vs_l1.{pdf,png}
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

# ── 1. CNN harmonic 10-ch normalized (A3-norm, plain, 3 seeds) ────────────────
cnn_dir = Path(__file__).resolve().parent / "posteriors"
cnn_files = sorted(cnn_dir.glob("cnn_harm_cross_norm_nobnt_s4?.npy"))
assert len(cnn_files) == 3, f"Expected 3 A3-norm files, got {len(cnn_files)}"
cnn_samples = np.concatenate([np.load(f) for f in cnn_files], axis=0)
fom_cnn_perseed = np.mean([fom3(np.load(f)) for f in cnn_files])

# ── 2. L1 harmonic auto+cross no-BNT (3 seeds) ───────────────────────────────
l1_dir = (
    REPO / "scripts/sbi/results/exploratory"
    / "cross_maps_campaign/jaxili_harm_cross_nobnt/posteriors"
)
l1_files = sorted(l1_dir.glob("l1cross_tomo4_20deg160mp_harm_nobnt_p1_s4?.npy"))
assert len(l1_files) == 3, f"Expected 3 L1 files, got {len(l1_files)}"
l1_samples = np.concatenate([np.load(f) for f in l1_files], axis=0)
fom_l1_perseed = np.mean([fom3(np.load(f)) for f in l1_files])

print(f"CNN harm 10-ch norm  ({len(cnn_files)} seeds):  mean-per-seed FoM3 = {fom_cnn_perseed:,.0f}")
print(f"L1  harm cross nobnt ({len(l1_files)} seeds):  mean-per-seed FoM3 = {fom_l1_perseed:,.0f}  *** caution: B2 sigma_8 flip ***")
print(f"L1 / CNN ratio: {fom_l1_perseed / fom_cnn_perseed:.2f}×")

# ── GetDist objects ───────────────────────────────────────────────────────────
smooth = {"smooth_scale_2D": 0.35, "smooth_scale_1D": 0.35}

mc_cnn = MCSamples(
    samples=cnn_samples,
    names=PARAM_NAMES, labels=PARAM_LABELS,
    label=rf"CNN harm 10-ch norm (3 seeds, $\langle$FoM3$\rangle$={fom_cnn_perseed:,.0f})",
    settings=smooth,
)
mc_l1 = MCSamples(
    samples=l1_samples,
    names=PARAM_NAMES, labels=PARAM_LABELS,
    label=rf"L1 harm cross (3 seeds, $\langle$FoM3$\rangle$={fom_l1_perseed:,.0f}) [caution: $\sigma_8$ calib?]",
    settings=smooth,
)

# ── Plot ──────────────────────────────────────────────────────────────────────
g = plots.get_subplot_plotter(subplot_size=2.0)
g.settings.axes_fontsize    = 11
g.settings.lab_fontsize      = 13
g.settings.legend_fontsize   = 10
g.settings.figure_legend_loc = "upper right"

g.triangle_plot(
    [mc_cnn, mc_l1],
    filled=True,
    contour_colors=["darkorange", "steelblue"],
    contour_lws=[1.2, 1.2],
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
    "CNN harm 10-ch normalized  vs  L1 harmonic cross (no-BNT, fiducial, seeds pooled)\n"
    r"Note: L1 $\sigma_8$ calibration uncertain (B2 truth-check found off-fiducial flip)",
    fontsize=11,
    y=1.01,
)

for fmt in ("pdf", "png"):
    out_path = OUT / f"overlay_harmnorm_cnn_vs_l1.{fmt}"
    g.export(str(out_path))
    print(f"Saved: {out_path}")
