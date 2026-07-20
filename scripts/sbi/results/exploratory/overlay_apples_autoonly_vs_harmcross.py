#!/usr/bin/env python3
"""Four-way overlay: CNN/L1 × auto-only/auto+cross, all 3-seed pooled.

  1. CNN auto-only      — 4 tomo auto, apples-to-apples at iter-108-Q6ON-60k config.
  2. CNN auto+cross     — 4 auto + 6 harm cross, iter-108-Q6ON-60k.
  3. L1  auto-only      — wavelet datavector on 4 tomo auto maps (20deg/160px, 3 seeds).
  4. L1  auto+cross     — wavelet on 4 auto + 6 harm cross, post-noise-fix v2_chsigma.

Output: overlay_apples_autoonly_vs_harmcross.{pdf,png}
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


auto_dir = REPO / "scripts/sbi/results/exploratory/apples_v_iter108_autoonly/posteriors"
auto_files = sorted(auto_dir.glob("cnn_auto_plain_cdim10_dense256_step60000_s4?.npy"))
assert len(auto_files) == 3, f"Expected 3 auto-only files, got {len(auto_files)}"
auto_samples = np.concatenate([np.load(f) for f in auto_files], axis=0)
fom_auto_pooled = fom3(auto_samples)
fom_auto_perseed = np.mean([fom3(np.load(f)) for f in auto_files])

cross_dir = Path("/nas/tersenov/claude-notes/runs/cnn-auto-cross-push-18-20-2026/iter-108-Q6ON-60k/posteriors")
cross_files = sorted(cross_dir.glob("cnn_autocross_plain_step60000_s4?.npy"))
assert len(cross_files) == 3, f"Expected 3 cross files, got {len(cross_files)}"
cross_samples = np.concatenate([np.load(f) for f in cross_files], axis=0)
fom_cross_pooled = fom3(cross_samples)
fom_cross_perseed = np.mean([fom3(np.load(f)) for f in cross_files])

l1_auto_dir = Path(
    "/mnt/home/tersenov/software/cnn_sbi/.worktrees/bnt_tomo_study/scripts/sbi/"
    "posteriors_archive/nobnt_tomo_bins_crosscorr_study_l1_jaxili_bestcfg/"
    "posteriors"
)
l1_auto_files = sorted(l1_auto_dir.glob("l1_tomo4_20deg160_nobnt_s4?.npy"))
assert len(l1_auto_files) == 3, f"Expected 3 L1 auto-only files, got {len(l1_auto_files)}"
l1_auto_samples = np.concatenate([np.load(f) for f in l1_auto_files], axis=0)
fom_l1auto_pooled = fom3(l1_auto_samples)
fom_l1auto_perseed = np.mean([fom3(np.load(f)) for f in l1_auto_files])

l1_cross_dir = REPO / "scripts/sbi/results/exploratory/auto_cross_v2_chsigma/l1_auto_cross/posteriors"
l1_cross_files = sorted(l1_cross_dir.glob("l1_auto_cross_s4?.npy"))
assert len(l1_cross_files) == 3, f"Expected 3 L1 auto+cross files, got {len(l1_cross_files)}"
l1_cross_samples = np.concatenate([np.load(f) for f in l1_cross_files], axis=0)
fom_l1cross_pooled = fom3(l1_cross_samples)
fom_l1cross_perseed = np.mean([fom3(np.load(f)) for f in l1_cross_files])

print(f"CNN auto-only    ({len(auto_files)} seeds):  pooled = {fom_auto_pooled:>7,.0f}   per-seed mean = {fom_auto_perseed:>7,.0f}")
print(f"CNN auto+cross   ({len(cross_files)} seeds):  pooled = {fom_cross_pooled:>7,.0f}   per-seed mean = {fom_cross_perseed:>7,.0f}")
print(f"L1  auto-only    ({len(l1_auto_files)} seeds):  pooled = {fom_l1auto_pooled:>7,.0f}   per-seed mean = {fom_l1auto_perseed:>7,.0f}")
print(f"L1  auto+cross   ({len(l1_cross_files)} seeds):  pooled = {fom_l1cross_pooled:>7,.0f}   per-seed mean = {fom_l1cross_perseed:>7,.0f}")
print(f"CNN cross/auto pooled ratio: {fom_cross_pooled / fom_auto_pooled:.2f}x")
print(f"L1  cross/auto pooled ratio: {fom_l1cross_pooled / fom_l1auto_pooled:.2f}x")
print(f"L1/CNN at auto-only        : {fom_l1auto_pooled / fom_auto_pooled:.2f}x")
print(f"L1/CNN at auto+cross       : {fom_l1cross_pooled / fom_cross_pooled:.2f}x")

smooth = {"smooth_scale_2D": 0.35, "smooth_scale_1D": 0.35}

mc_cnn_auto = MCSamples(
    samples=auto_samples,
    names=PARAM_NAMES, labels=PARAM_LABELS,
    label=rf"CNN auto-only (pooled FoM3 $={fom_auto_pooled:,.0f}$)",
    settings=smooth,
)
mc_cnn_cross = MCSamples(
    samples=cross_samples,
    names=PARAM_NAMES, labels=PARAM_LABELS,
    label=rf"CNN auto+cross (pooled FoM3 $={fom_cross_pooled:,.0f}$)",
    settings=smooth,
)
mc_l1_auto = MCSamples(
    samples=l1_auto_samples,
    names=PARAM_NAMES, labels=PARAM_LABELS,
    label=rf"L1 auto-only (pooled FoM3 $={fom_l1auto_pooled:,.0f}$)",
    settings=smooth,
)
mc_l1_cross = MCSamples(
    samples=l1_cross_samples,
    names=PARAM_NAMES, labels=PARAM_LABELS,
    label=rf"L1 auto+cross (pooled FoM3 $={fom_l1cross_pooled:,.0f}$)",
    settings=smooth,
)

g = plots.get_subplot_plotter(subplot_size=1.7)
# Order weakest-FoM3 first so tighter contours sit on top and stay visible.
g.triangle_plot(
    [mc_l1_auto, mc_cnn_auto, mc_cnn_cross, mc_l1_cross],
    filled=True,
    markers={n: t for n, t in zip(PARAM_NAMES, TRUTH)},
)

out_pdf = OUT_DIR / "overlay_apples_autoonly_vs_harmcross.pdf"
out_png = OUT_DIR / "overlay_apples_autoonly_vs_harmcross.png"
g.export(str(out_pdf))
g.export(str(out_png), dpi=150)
print(f"\nWrote: {out_pdf}")
print(f"Wrote: {out_png}")
