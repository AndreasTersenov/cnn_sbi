#!/usr/bin/env python3
"""Render a CNN iteration's posterior corner plot overlaid with the L1
auto+cross reference. Used for live visibility while Ralph runs the loop.

Usage:
    python render_overlay.py --iter-dir <NOTES_DIR>/runs/<tag>/iter-<n>

What it does:
    - Loads all 3-seed CNN posteriors from <iter-dir>/posteriors/*_s4?.npy
    - Loads the L1 auto+cross 3-seed reference (project default below)
    - Pools each (concatenates the per-seed samples)
    - Renders a GetDist triangle plot over (Ω_m, σ_8, w_0) with filled
      contours, fiducial markers, FoM3 in each label
    - Saves to <iter-dir>/overlay_vs_l1_autocross.pdf
    - Updates symlinks in <iter-dir>/.. (the run dir):
        latest_overlay.pdf  -> overlay of the just-rendered iter
        best_overlay.pdf    -> overlay of the iter listed as 'best'
                               (call with --is-best to update; or invoke
                               the convenience script update_best_overlay.sh)

The L1 reference is hard-coded to the project's headline reference:
    scripts/sbi/results/exploratory/auto_cross_v2_chsigma/l1_auto_cross/posteriors/
    (3 seeds; pooled FoM3 ~34 k mean; this is the cnn-auto-push fiber's
     target reference per the constitution).
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

# Fiducial cosmology used by the project
FIDUCIAL = {"omega_m": 0.26, "sigma_8": 0.84, "w_0": -1.0}

# Default L1 auto+cross reference
DEFAULT_L1_GLOB = (
    "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/"
    "auto_cross_v2_chsigma/l1_auto_cross/posteriors/l1_auto_cross_s4?.npy"
)


def fom3(samples: np.ndarray) -> float:
    C = np.cov(samples[:, :3].T)
    return 1.0 / np.sqrt(np.linalg.det(C))


def load_and_pool(glob_pattern: str, cap_per_seed: int = 100_000):
    """Return (pooled_samples, mean_of_seeds_fom3, pooled_fom3, n_seeds)."""
    paths = sorted(glob.glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"no posteriors matched {glob_pattern}")
    arrays = []
    per_seed_fom3 = []
    for p in paths:
        x = np.load(p, allow_pickle=False)
        if x.ndim != 2 or x.shape[1] < 3:
            print(f"  [skip] {p}: bad shape {x.shape}", file=sys.stderr)
            continue
        if x.shape[0] > cap_per_seed:
            x = x[:cap_per_seed]
        arrays.append(x[:, :3])
        per_seed_fom3.append(fom3(x))
    if not arrays:
        raise RuntimeError(f"no usable posteriors in {glob_pattern}")
    pooled = np.concatenate(arrays, axis=0)
    return pooled, float(np.mean(per_seed_fom3)), fom3(pooled), len(arrays)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter-dir", required=True, type=Path,
                    help="Path to <run-dir>/iter-<n>/")
    ap.add_argument("--l1-glob", default=DEFAULT_L1_GLOB,
                    help="Glob for L1 auto+cross reference posteriors")
    ap.add_argument("--is-best", action="store_true",
                    help="Also update <run-dir>/best_overlay.pdf symlink")
    ap.add_argument("--cnn-label", default=None,
                    help="Label for CNN samples (default derived from iter-dir)")
    ap.add_argument("--no-latest", action="store_true",
                    help="Skip updating the latest_overlay.pdf symlink")
    args = ap.parse_args()

    # getdist import deferred so the script can --help without the dep
    from getdist import MCSamples, plots

    iter_dir = args.iter_dir.resolve()
    cnn_glob = str(iter_dir / "posteriors" / "*_s4?.npy")
    cnn_samples, cnn_mos, cnn_pool, cnn_n = load_and_pool(cnn_glob)
    l1_samples, l1_mos, l1_pool, l1_n = load_and_pool(args.l1_glob)

    cnn_label = args.cnn_label or (
        f"CNN {iter_dir.name}  "
        f"FoM3 mean-of-{cnn_n}={cnn_mos:.0f} (pooled={cnn_pool:.0f})"
    )
    l1_label = (
        f"L1 auto+cross ref  "
        f"FoM3 mean-of-{l1_n}={l1_mos:.0f} (pooled={l1_pool:.0f})"
    )

    names = ["omega_m", "sigma_8", "w_0"]
    labels = [r"\Omega_m", r"\sigma_8", r"w_0"]

    mc_cnn = MCSamples(samples=cnn_samples, names=names, labels=labels,
                       label=cnn_label)
    mc_l1 = MCSamples(samples=l1_samples, names=names, labels=labels,
                      label=l1_label)

    g = plots.get_subplot_plotter(width_inch=8)
    g.settings.alpha_filled_add = 0.55
    g.triangle_plot(
        [mc_cnn, mc_l1],
        filled=True,
        markers=FIDUCIAL,
        marker_args={"color": "red", "lw": 1.0, "ls": "--"},
        contour_colors=["tab:blue", "tab:orange"],
        legend_loc="upper right",
    )

    out_pdf = iter_dir / "overlay_vs_l1_autocross.pdf"
    g.export(str(out_pdf))
    print(f"[ok] wrote {out_pdf}", flush=True)
    print(f"[fom3 mean-of-seeds] cnn={cnn_mos:.1f}  l1_ref={l1_mos:.1f}  "
          f"ratio={cnn_mos/l1_mos:.3f}  (target=1.00 to match L1 auto+cross)",
          flush=True)
    print(f"[fom3 pooled]        cnn={cnn_pool:.1f}  l1_ref={l1_pool:.1f}  "
          f"ratio={cnn_pool/l1_pool:.3f}  (covariance of plotted contour)",
          flush=True)

    if not args.no_latest:
        run_dir = iter_dir.parent
        latest = run_dir / "latest_overlay.pdf"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out_pdf.resolve())
        print(f"[ok] latest_overlay.pdf -> {out_pdf}", flush=True)

    if args.is_best:
        run_dir = iter_dir.parent
        best = run_dir / "best_overlay.pdf"
        if best.is_symlink() or best.exists():
            best.unlink()
        best.symlink_to(out_pdf.resolve())
        print(f"[ok] best_overlay.pdf -> {out_pdf}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
