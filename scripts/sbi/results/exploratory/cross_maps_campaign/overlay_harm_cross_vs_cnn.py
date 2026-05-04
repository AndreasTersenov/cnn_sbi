"""Overlay corner plots for the harmonic-cross-maps headline result vs:
1. itself across regimes (harm_cross BNT vs harm_cross no-BNT)
2. the best CNN-demeaned reference (advanced_arch64_dense256_nostd_long, cdim=10)

CNN reference is `run_b_advanced_plain` from the 2026-04-22 zero-mean parity
check (PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md §13). It dominates the resnet18
config in both regimes (no-BNT FoM3 16763 vs 14947; BNT FoM3 15117 vs 9413).

All overlays pool seeds 41/42/43 (the 3-seed slice common to both campaigns).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from getdist import MCSamples, plots as gplot

CROSS_ROOT = Path("/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/cross_maps_campaign")
CNN_ROOT = Path("/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/zero_mean_maps_parity_check/run_b_advanced_plain/posteriors")
OUT = CROSS_ROOT / "cross_summary"
OUT.mkdir(exist_ok=True)
SEEDS = (41, 42, 43)
TRUTH = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493], dtype=np.float64)
PARAM_NAMES = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]

HARM_BNT = str(CROSS_ROOT / "jaxili_harm_cross_bnt" / "posteriors" /
               "l1cross_tomo4_20deg160mp_harm_bnt_p1_s{s}.npy")
HARM_NOBNT = str(CROSS_ROOT / "jaxili_harm_cross_nobnt" / "posteriors" /
                 "l1cross_tomo4_20deg160mp_harm_nobnt_p1_s{s}.npy")
CNN_BNT = str(CNN_ROOT / "cnn_tomo4_20deg160_bnt_advanced_arch64_dense256_nostd_long_zm_s{s}.npy")
CNN_NOBNT = str(CNN_ROOT / "cnn_tomo4_20deg160_nobnt_advanced_arch64_dense256_nostd_long_zm_s{s}.npy")


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
    overlay(
        [("harmonic cross L1, BNT",    HARM_BNT,   "#d62728"),
         ("harmonic cross L1, no-BNT", HARM_NOBNT, "#2ca02c")],
        "Harmonic auto+cross L1 — BNT vs no-BNT (3 seeds pooled)",
        OUT / "overlay_harm_cross_bnt_vs_nobnt.pdf",
    )
    overlay(
        [("best CNN-VMIM (demeaned), no-BNT", CNN_NOBNT,  "#1f77b4"),
         ("harmonic cross L1, no-BNT",        HARM_NOBNT, "#2ca02c")],
        "Harmonic auto+cross L1 vs best CNN — no-BNT (3 seeds pooled)",
        OUT / "overlay_harm_cross_vs_cnn_nobnt.pdf",
    )
    overlay(
        [("best CNN-VMIM (demeaned), BNT", CNN_BNT,  "#1f77b4"),
         ("harmonic cross L1, BNT",        HARM_BNT, "#d62728")],
        "Harmonic auto+cross L1 vs best CNN — BNT (3 seeds pooled)",
        OUT / "overlay_harm_cross_vs_cnn_bnt.pdf",
    )


if __name__ == "__main__":
    main()
