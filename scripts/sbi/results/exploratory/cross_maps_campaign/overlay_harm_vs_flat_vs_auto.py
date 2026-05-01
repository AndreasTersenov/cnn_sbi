"""Overlay corner plots: harmonic-cache cross vs flat-sky cross (pct1) vs
auto-only zero-mean baseline. Three arms per regime, 3 seeds pooled per arm.

Output goes to cross_summary/overlay_harm_vs_flat_vs_auto_{regime}.pdf.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from getdist import MCSamples, plots as gplot

ROOT = Path("/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/cross_maps_campaign")
OUT = ROOT / "cross_summary"
OUT.mkdir(exist_ok=True)
SEEDS = (41, 42, 43)
TRUTH = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493], dtype=np.float64)
PARAM_NAMES = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]


def _pool(template: str) -> np.ndarray:
    return np.concatenate([np.load(template.format(s=s)) for s in SEEDS], axis=0)


def overlay(arms: list[tuple[str, str, str]], title: str, out: Path) -> None:
    chains = [
        MCSamples(
            samples=_pool(tpl),
            names=PARAM_NAMES,
            labels=PARAM_NAMES,
            label=label,
            settings={"smooth_scale_2D": 0.7, "smooth_scale_1D": 0.7},
        )
        for label, tpl, _ in arms
    ]
    colors = [c for _, _, c in arms]
    g = gplot.get_subplot_plotter(subplot_size=1.4)
    g.settings.alpha_filled_add = 0.45
    g.triangle_plot(
        chains,
        filled=True,
        contour_colors=colors,
        markers=TRUTH,
        marker_args={"color": "black", "lw": 1.0},
        legend_loc="upper right",
    )
    g.fig.suptitle(title, fontsize=13)
    g.fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(g.fig)
    print(f"wrote {out}")


def main() -> None:
    bnt_arms = [
        ("auto-only L1 (zero-mean)",
         str(ROOT / "jaxili_auto_zm_bnt" / "posteriors" /
             "l1_tomo4_20deg160mp_zm_bnt_s{s}.npy"),
         "#1f77b4"),
        ("flat-sky cross L1 (pct1)",
         str(ROOT / "jaxili_cross_bnt_pct1" / "posteriors" /
             "l1cross_tomo4_20deg160mp_bnt_p1_s{s}.npy"),
         "#d62728"),
        ("harmonic cross L1 (pct1)",
         str(ROOT / "jaxili_harm_cross_bnt" / "posteriors" /
             "l1cross_tomo4_20deg160mp_harm_bnt_p1_s{s}.npy"),
         "#2ca02c"),
    ]
    nobnt_arms = [
        ("auto-only L1 (zero-mean)",
         str(ROOT / "jaxili_auto_zm_nobnt" / "posteriors" /
             "l1_tomo4_20deg160mp_zm_nobnt_s{s}.npy"),
         "#1f77b4"),
        ("flat-sky cross L1 (pct1)",
         str(ROOT / "jaxili_cross_nobnt_pct1" / "posteriors" /
             "l1cross_tomo4_20deg160mp_nobnt_p1_s{s}.npy"),
         "#d62728"),
        ("harmonic cross L1 (pct1)",
         str(ROOT / "jaxili_harm_cross_nobnt" / "posteriors" /
             "l1cross_tomo4_20deg160mp_harm_nobnt_p1_s{s}.npy"),
         "#2ca02c"),
    ]
    overlay(bnt_arms,
            "BNT, zero-mean, 20deg/160px multipatch — 3 seeds pooled",
            OUT / "overlay_harm_vs_flat_vs_auto_bnt.pdf")
    overlay(nobnt_arms,
            "no-BNT, zero-mean, 20deg/160px multipatch — 3 seeds pooled",
            OUT / "overlay_harm_vs_flat_vs_auto_nobnt.pdf")


if __name__ == "__main__":
    main()
