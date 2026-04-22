#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


TRUTH = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493], dtype=np.float64)
PARAM_NAMES = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]

DEFAULT_VARIANTS = "bin1_20deg160,bin2_20deg160,bin3_20deg160,bin4_20deg160,tomo4_20deg160"
DEFAULT_SEEDS = "41,42,43"
DEFAULT_COMPARISONS = "cnn:l1,cnn:l1vmim"
TRIO_METHODS = ("cnn", "l1", "l1vmim")
METHOD_LABELS = {"cnn": "CNN", "l1": "L1", "l1vmim": "L1+VMIM"}


def _csv_tokens(value: str) -> List[str]:
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def _csv_ints(value: str) -> List[int]:
    return [int(tok) for tok in _csv_tokens(value)]


def _parse_comparisons(spec: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for token in _csv_tokens(spec):
        if ":" not in token:
            raise ValueError(
                f"Invalid comparison token '{token}'. Expected format methodA:methodB."
            )
        left, right = token.split(":", 1)
        left = left.strip()
        right = right.strip()
        if not left or not right:
            raise ValueError(
                f"Invalid comparison token '{token}'. Empty method name is not allowed."
            )
        pairs.append((left, right))
    return pairs


def _import_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from getdist import MCSamples, plots as gplot

    return plt, MCSamples, gplot


def _posterior_path(root: Path, method: str, variant: str, seed: int) -> Path:
    return root / "posteriors" / f"{method}_{variant}_nobnt_s{seed}.npy"


def _load_all(paths: Iterable[Path]) -> np.ndarray:
    arrays = [np.load(p) for p in paths]
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
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    label_a: str,
    label_b: str,
    title: str,
    plt,
    MCSamples,
    gplot,
    dpi: int,
) -> None:
    chain_a = _mcs(samples_a, MCSamples, label_a)
    chain_b = _mcs(samples_b, MCSamples, label_b)

    g = gplot.get_subplot_plotter(subplot_size=1.4)
    g.settings.alpha_filled_add = 0.55
    g.triangle_plot(
        [chain_a, chain_b],
        filled=True,
        contour_colors=["#1f77b4", "#d62728"],
        markers=TRUTH,
        marker_args={"color": "black", "lw": 1.0},
        legend_loc="upper right",
    )
    g.fig.suptitle(title, fontsize=12)
    g.fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(g.fig)


def _plot_overlay_trio(
    out_path: Path,
    samples_cnn: np.ndarray,
    samples_l1: np.ndarray,
    samples_l1vmim: np.ndarray,
    title: str,
    plt,
    MCSamples,
    gplot,
    dpi: int,
) -> None:
    chain_cnn = _mcs(samples_cnn, MCSamples, METHOD_LABELS["cnn"])
    chain_l1 = _mcs(samples_l1, MCSamples, METHOD_LABELS["l1"])
    chain_l1vmim = _mcs(samples_l1vmim, MCSamples, METHOD_LABELS["l1vmim"])

    g = gplot.get_subplot_plotter(subplot_size=1.4)
    g.settings.alpha_filled_add = 0.55
    g.triangle_plot(
        [chain_cnn, chain_l1, chain_l1vmim],
        filled=True,
        contour_colors=["#1f77b4", "#d62728", "#2ca02c"],
        markers=TRUTH,
        marker_args={"color": "black", "lw": 1.0},
        legend_loc="upper right",
    )
    g.fig.suptitle(title, fontsize=12)
    g.fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(g.fig)


def _ensure_inputs_for_methods(
    root: Path,
    variants: Sequence[str],
    seeds: Sequence[int],
    methods: Sequence[str],
) -> None:
    missing: List[Path] = []
    for variant in variants:
        for seed in seeds:
            for method in methods:
                p = _posterior_path(root, method, variant, seed)
                if not p.exists():
                    missing.append(p)
    if missing:
        msg = "Missing posterior files:\n" + "\n".join(str(p) for p in missing)
        raise FileNotFoundError(msg)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot no-BNT tomographic-bin method comparison contour overlays."
    )
    p.add_argument(
        "--study-root",
        type=Path,
        default=Path("scripts/sbi/nobnt_tomo_bins_crosscorr_study"),
        help="Path to no-BNT cross-correlation study directory.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scripts/sbi/nobnt_tomo_bins_crosscorr_study/overlays"),
        help="Output directory for overlay figures.",
    )
    p.add_argument("--variants", type=str, default=DEFAULT_VARIANTS)
    p.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)
    p.add_argument(
        "--comparisons",
        type=str,
        default=DEFAULT_COMPARISONS,
        help="Comma-separated method pairs, e.g. 'cnn:l1,cnn:l1vmim'.",
    )
    p.add_argument(
        "--combined",
        action="store_true",
        help="Also write seed-combined overlays for each comparison and variant.",
    )
    p.add_argument(
        "--trio",
        "--three-way",
        dest="trio",
        action="store_true",
        help="Write three-method CNN/L1/L1+VMIM overlays instead of pairwise overlays.",
    )
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    study_root = args.study_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = _csv_tokens(args.variants)
    seeds = _csv_ints(args.seeds)

    if not variants:
        raise ValueError("--variants cannot be empty.")
    if not seeds:
        raise ValueError("--seeds cannot be empty.")

    plt, MCSamples, gplot = _import_plotting()

    written: List[Path] = []
    if args.trio:
        _ensure_inputs_for_methods(study_root, variants, seeds, list(TRIO_METHODS))
        for variant in variants:
            for seed in seeds:
                p_cnn = _posterior_path(study_root, "cnn", variant, seed)
                p_l1 = _posterior_path(study_root, "l1", variant, seed)
                p_l1vmim = _posterior_path(study_root, "l1vmim", variant, seed)
                out_path = (
                    output_dir
                    / f"overlay_{variant}_cnn_l1_l1vmim_seed{seed}_nobnt.png"
                )
                _plot_overlay_trio(
                    out_path=out_path,
                    samples_cnn=np.load(p_cnn),
                    samples_l1=np.load(p_l1),
                    samples_l1vmim=np.load(p_l1vmim),
                    title=f"{variant} seed {seed}: CNN vs L1 vs L1+VMIM (no-BNT)",
                    plt=plt,
                    MCSamples=MCSamples,
                    gplot=gplot,
                    dpi=args.dpi,
                )
                written.append(out_path)

            if args.combined:
                out_path = (
                    output_dir
                    / f"overlay_{variant}_cnn_l1_l1vmim_combined_nobnt.png"
                )
                _plot_overlay_trio(
                    out_path=out_path,
                    samples_cnn=_load_all(
                        [_posterior_path(study_root, "cnn", variant, s) for s in seeds]
                    ),
                    samples_l1=_load_all(
                        [_posterior_path(study_root, "l1", variant, s) for s in seeds]
                    ),
                    samples_l1vmim=_load_all(
                        [_posterior_path(study_root, "l1vmim", variant, s) for s in seeds]
                    ),
                    title=f"{variant}: CNN vs L1 vs L1+VMIM (combined seeds, no-BNT)",
                    plt=plt,
                    MCSamples=MCSamples,
                    gplot=gplot,
                    dpi=args.dpi,
                )
                written.append(out_path)
    else:
        comparisons = _parse_comparisons(args.comparisons)
        if not comparisons:
            raise ValueError("--comparisons cannot be empty.")

        pairwise_methods = sorted({method for pair in comparisons for method in pair})
        _ensure_inputs_for_methods(study_root, variants, seeds, pairwise_methods)
        for variant in variants:
            for left, right in comparisons:
                for seed in seeds:
                    p_left = _posterior_path(study_root, left, variant, seed)
                    p_right = _posterior_path(study_root, right, variant, seed)
                    out_path = (
                        output_dir
                        / f"overlay_{variant}_{left}_vs_{right}_seed{seed}_nobnt.png"
                    )
                    _plot_overlay(
                        out_path=out_path,
                        samples_a=np.load(p_left),
                        samples_b=np.load(p_right),
                        label_a=left.upper(),
                        label_b=right.upper(),
                        title=f"{variant} seed {seed}: {left.upper()} vs {right.upper()} (no-BNT)",
                        plt=plt,
                        MCSamples=MCSamples,
                        gplot=gplot,
                        dpi=args.dpi,
                    )
                    written.append(out_path)

                if args.combined:
                    left_paths = [
                        _posterior_path(study_root, left, variant, s) for s in seeds
                    ]
                    right_paths = [
                        _posterior_path(study_root, right, variant, s) for s in seeds
                    ]
                    out_path = (
                        output_dir
                        / f"overlay_{variant}_{left}_vs_{right}_combined_nobnt.png"
                    )
                    _plot_overlay(
                        out_path=out_path,
                        samples_a=_load_all(left_paths),
                        samples_b=_load_all(right_paths),
                        label_a=left.upper(),
                        label_b=right.upper(),
                        title=f"{variant}: {left.upper()} vs {right.upper()} (combined seeds, no-BNT)",
                        plt=plt,
                        MCSamples=MCSamples,
                        gplot=gplot,
                        dpi=args.dpi,
                    )
                    written.append(out_path)

    print(f"Wrote {len(written)} overlay figure(s) to: {output_dir}")
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
