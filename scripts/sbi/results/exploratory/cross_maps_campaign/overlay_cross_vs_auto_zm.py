"""Overlay corner plots: auto+cross (pct1) vs auto-only zero-mean baseline.

For each regime (BNT, no-BNT) we pool the 3-seed posteriors of the matched
arms and overlay 1D/2D contours via getdist. Output goes to cross_summary/.
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
    arrs = [np.load(template.format(s=s)) for s in SEEDS]
    return np.concatenate(arrs, axis=0)


def overlay(arm_cross_tpl: str, arm_auto_tpl: str, label_cross: str,
            label_auto: str, title: str, out: Path) -> None:
    samples_cross = _pool(arm_cross_tpl)
    samples_auto = _pool(arm_auto_tpl)
    chain_cross = MCSamples(
        samples=samples_cross,
        names=PARAM_NAMES,
        labels=PARAM_NAMES,
        label=label_cross,
        settings={"smooth_scale_2D": 0.7, "smooth_scale_1D": 0.7},
    )
    chain_auto = MCSamples(
        samples=samples_auto,
        names=PARAM_NAMES,
        labels=PARAM_NAMES,
        label=label_auto,
        settings={"smooth_scale_2D": 0.7, "smooth_scale_1D": 0.7},
    )
    g = gplot.get_subplot_plotter(subplot_size=1.4)
    g.settings.alpha_filled_add = 0.55
    g.triangle_plot(
        [chain_auto, chain_cross],
        filled=True,
        contour_colors=["#1f77b4", "#d62728"],
        markers=TRUTH,
        marker_args={"color": "black", "lw": 1.0},
        legend_loc="upper right",
    )
    g.fig.suptitle(title, fontsize=13)
    g.fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(g.fig)
    print(f"wrote {out}")


def main() -> None:
    overlay(
        arm_cross_tpl=str(ROOT / "jaxili_cross_bnt_pct1" / "posteriors" /
                          "l1cross_tomo4_20deg160mp_bnt_p1_s{s}.npy"),
        arm_auto_tpl=str(ROOT / "jaxili_auto_zm_bnt" / "posteriors" /
                         "l1_tomo4_20deg160mp_zm_bnt_s{s}.npy"),
        label_cross="auto+cross L1 (pct1)",
        label_auto="auto-only L1",
        title="BNT, zero-mean, 20deg/160px multipatch — 3 seeds pooled",
        out=OUT / "overlay_bnt_pct1_vs_auto_zm.pdf",
    )
    overlay(
        arm_cross_tpl=str(ROOT / "jaxili_cross_nobnt_pct1" / "posteriors" /
                          "l1cross_tomo4_20deg160mp_nobnt_p1_s{s}.npy"),
        arm_auto_tpl=str(ROOT / "jaxili_auto_zm_nobnt" / "posteriors" /
                         "l1_tomo4_20deg160mp_zm_nobnt_s{s}.npy"),
        label_cross="auto+cross L1 (pct1)",
        label_auto="auto-only L1",
        title="no-BNT, zero-mean, 20deg/160px multipatch — 3 seeds pooled",
        out=OUT / "overlay_nobnt_pct1_vs_auto_zm.pdf",
    )


if __name__ == "__main__":
    main()
