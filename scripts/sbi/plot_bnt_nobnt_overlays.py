#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


TRUTH = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493], dtype=np.float64)
PARAM_NAMES = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]
METHODS = ("cnn", "l1", "l1vmim")
SEEDS = (41, 42, 43)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot BNT vs no-BNT posterior contour overlays for tomo4 study"
    )
    p.add_argument(
        "--study-root",
        type=Path,
        default=Path("scripts/sbi/bnt_tomo4_study"),
        help="Path to bnt_tomo4_study directory",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scripts/sbi/bnt_tomo4_study/overlays"),
        help="Directory to write overlay figures",
    )
    p.add_argument(
        "--combined",
        action="store_true",
        help="Also write per-method seed-combined overlays",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI",
    )
    return p.parse_args()


def _import_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from getdist import MCSamples, plots as gplot

    return plt, MCSamples, gplot


def _posterior_path(root: Path, method: str, condition: str, seed: int) -> Path:
    return root / "posteriors" / f"{method}_tomo4_20deg160_{condition}_s{seed}.npy"


def _load_all(arr_paths: Iterable[Path]) -> np.ndarray:
    arrays = [np.load(p) for p in arr_paths]
    return np.concatenate(arrays, axis=0)


def _mcs(samples: np.ndarray, MCSamples, label: str):
    return MCSamples(
        samples=samples,
        names=PARAM_NAMES,
        labels=PARAM_NAMES,
        label=label,
        settings={"smooth_scale_2D": 0.7, "smooth_scale_1D": 0.7},
    )


def _plot_overlay(
    out_path: Path,
    samples_nobnt: np.ndarray,
    samples_bnt: np.ndarray,
    title: str,
    plt,
    MCSamples,
    gplot,
    dpi: int,
):
    chain_nobnt = _mcs(samples_nobnt, MCSamples, "no-BNT")
    chain_bnt = _mcs(samples_bnt, MCSamples, "BNT")

    g = gplot.get_subplot_plotter(subplot_size=1.4)
    g.settings.alpha_filled_add = 0.55
    g.triangle_plot(
        [chain_nobnt, chain_bnt],
        filled=True,
        contour_colors=["#1f77b4", "#d62728"],
        markers=TRUTH,
        marker_args={"color": "black", "lw": 1.0},
        legend_loc="upper right",
    )
    g.fig.suptitle(title, fontsize=12)
    g.fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(g.fig)


def _ensure_inputs(root: Path, methods: Sequence[str], seeds: Sequence[int]):
    missing: list[Path] = []
    for m in methods:
        for s in seeds:
            for c in ("nobnt", "bnt"):
                p = _posterior_path(root, m, c, s)
                if not p.exists():
                    missing.append(p)
    if missing:
        msg = "Missing posterior files:\n" + "\n".join(str(p) for p in missing)
        raise FileNotFoundError(msg)


def main() -> None:
    args = parse_args()
    root = args.study_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    _ensure_inputs(root, METHODS, SEEDS)
    plt, MCSamples, gplot = _import_plotting()

    written: list[Path] = []

    # Per-method, per-seed overlays
    for method in METHODS:
        for seed in SEEDS:
            p_nobnt = _posterior_path(root, method, "nobnt", seed)
            p_bnt = _posterior_path(root, method, "bnt", seed)
            s_nobnt = np.load(p_nobnt)
            s_bnt = np.load(p_bnt)
            out_path = out / f"overlay_{method}_seed{seed}_bnt_vs_nobnt.png"
            _plot_overlay(
                out_path,
                s_nobnt,
                s_bnt,
                f"{method.upper()} seed {seed}: BNT vs no-BNT",
                plt,
                MCSamples,
                gplot,
                args.dpi,
            )
            written.append(out_path)

    # Optional combined overlays per method (concat seeds)
    if args.combined:
        for method in METHODS:
            nobnt_paths = [_posterior_path(root, method, "nobnt", s) for s in SEEDS]
            bnt_paths = [_posterior_path(root, method, "bnt", s) for s in SEEDS]
            s_nobnt = _load_all(nobnt_paths)
            s_bnt = _load_all(bnt_paths)
            out_path = out / f"overlay_{method}_combined_bnt_vs_nobnt.png"
            _plot_overlay(
                out_path,
                s_nobnt,
                s_bnt,
                f"{method.upper()} combined seeds: BNT vs no-BNT",
                plt,
                MCSamples,
                gplot,
                args.dpi,
            )
            written.append(out_path)

    print(f"Wrote {len(written)} overlay figure(s) to: {out}")
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
