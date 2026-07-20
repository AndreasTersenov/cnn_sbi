#!/usr/bin/env python3
"""Pooled-contour overlay: iter-16 auto-only vs cross-push pooled-FoM3 best.

Two arms, seeds pooled within each (concat all draws):
  1. iter-16 auto-only — the cnn-auto-push-18-20-2026 certified best (3 seeds).
  2. iter-108-Q6ON-60k — the cnn-auto-cross-push-18-20-2026 pooled-FoM3 winner
     across all iterations with 3-seed posteriors on disk (pooled 23,986;
     beats harm-norm CB's pooled 23,280 by ~3%). Plain CNN, cdim=10,
     dense=256, conv=64,128,256, 60k compressor steps, standardize-summary
     on, harm-normalize on, no-BNT, zero-mean-maps.

Output: overlay_iter16_autoonly_vs_harmcross_best.{pdf,png}
"""
import numpy as np
from pathlib import Path
from getdist import MCSamples, plots

REPO = Path(__file__).resolve().parents[4]
OUT_DIR = Path(__file__).resolve().parent

PARAM_NAMES = ["Omega_m", "sigma_8", "w0", "h0", "n_s", "Omega_b"]
PARAM_LABELS = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]
TRUTH = [0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493]


def fom3(samples: np.ndarray) -> float:
    cov3 = np.cov(samples[:, :3].T)
    return 1.0 / np.sqrt(np.linalg.det(cov3))


auto_dir = Path("/nas/tersenov/claude-notes/runs/cnn-auto-push-18-20-2026/iter-16/posteriors")
auto_files = sorted(auto_dir.glob("cnn_auto_plain_step120000_s4?.npy"))
assert len(auto_files) == 3, f"Expected 3 iter-16 auto-only files, got {len(auto_files)}"
auto_samples = np.concatenate([np.load(f) for f in auto_files], axis=0)
fom_auto_pooled = fom3(auto_samples)
fom_auto_perseed = np.mean([fom3(np.load(f)) for f in auto_files])

cross_dir = Path("/nas/tersenov/claude-notes/runs/cnn-auto-cross-push-18-20-2026/iter-108-Q6ON-60k/posteriors")
cross_files = sorted(cross_dir.glob("cnn_autocross_plain_step60000_s4?.npy"))
assert len(cross_files) == 3, f"Expected 3 iter-108-Q6ON files, got {len(cross_files)}"
cross_samples = np.concatenate([np.load(f) for f in cross_files], axis=0)
fom_cross_pooled = fom3(cross_samples)
fom_cross_perseed = np.mean([fom3(np.load(f)) for f in cross_files])

print(f"AUTO-only iter-16            ({len(auto_files)} seeds):  pooled FoM3 = {fom_auto_pooled:,.0f}   <FoM3>_per-seed = {fom_auto_perseed:,.0f}")
print(f"AUTO+CROSS iter-108-Q6ON-60k ({len(cross_files)} seeds):  pooled FoM3 = {fom_cross_pooled:,.0f}   <FoM3>_per-seed = {fom_cross_perseed:,.0f}")
print(f"Cross / auto pooled FoM3 ratio:  {fom_cross_pooled / fom_auto_pooled:.2f}x")
print(f"Cross / auto per-seed FoM3 ratio: {fom_cross_perseed / fom_auto_perseed:.2f}x")

smooth = {"smooth_scale_2D": 0.35, "smooth_scale_1D": 0.35}

mc_auto = MCSamples(
    samples=auto_samples,
    names=PARAM_NAMES, labels=PARAM_LABELS,
    label=rf"auto-only iter-16 (pooled FoM3 $={fom_auto_pooled:,.0f}$)",
    settings=smooth,
)
mc_cross = MCSamples(
    samples=cross_samples,
    names=PARAM_NAMES, labels=PARAM_LABELS,
    label=rf"auto+cross iter-108-Q6ON (pooled FoM3 $={fom_cross_pooled:,.0f}$)",
    settings=smooth,
)

g = plots.get_subplot_plotter(subplot_size=1.5)
g.settings.alpha_filled_add = 0.55
g.settings.title_limit_fontsize = 10
g.settings.legend_fontsize = 11
g.settings.linewidth_contour = 1.5
g.triangle_plot(
    [mc_auto, mc_cross],
    filled=True,
    contour_colors=["#1f77b4", "#d62728"],
    markers={n: t for n, t in zip(PARAM_NAMES, TRUTH)},
)

out_pdf = OUT_DIR / "overlay_iter16_autoonly_vs_harmcross_best.pdf"
out_png = OUT_DIR / "overlay_iter16_autoonly_vs_harmcross_best.png"
g.export(str(out_pdf))
g.export(str(out_png), dpi=150)
print(f"\nWrote: {out_pdf}")
print(f"Wrote: {out_png}")
