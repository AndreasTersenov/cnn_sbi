#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


TRUTH = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493], dtype=np.float64)
PARAM_NAMES = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]
METHODS = ("cnn", "l1", "l1vmim")
METHOD_LABELS = {"cnn": "CNN", "l1": "L1 (jaxili)", "l1vmim": "L1+VMIM"}


def _default_perm_indices() -> str:
    return ",".join(str(i) for i in range(20))


def _csv_tokens(value: str) -> List[str]:
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def _csv_ints(value: str) -> List[int]:
    return [int(tok) for tok in _csv_tokens(value)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot contour overlays for tomo4 baryon-bias study."
    )
    p.add_argument(
        "--study-root",
        type=Path,
        default=Path("scripts/sbi/baryon_bias_tomo4_study"),
        help="Path to baryon bias study root.",
    )
    p.add_argument("--variant", type=str, default="tomo4_20deg160")
    p.add_argument("--seeds", type=str, default="41,42,43")
    p.add_argument("--perm-indices", type=str, default=_default_perm_indices())

    p.add_argument(
        "--cnn-baseline-root",
        type=Path,
        default=Path("scripts/sbi/nobnt_tomo_bins_crosscorr_study"),
    )
    p.add_argument(
        "--l1-baseline-root",
        type=Path,
        default=Path("scripts/sbi/nobnt_tomo_bins_crosscorr_study_l1_jaxili_bestcfg"),
    )
    p.add_argument(
        "--l1vmim-baseline-root",
        type=Path,
        default=Path("scripts/sbi/nobnt_tomo_bins_crosscorr_study"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <study-root>/overlays.",
    )
    p.add_argument(
        "--selected-perms",
        type=str,
        default="",
        help="Optional subset of perm indices for extra per-perm trio plots.",
    )
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def _import_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from getdist import MCSamples, plots as gplot

    return plt, MCSamples, gplot


def _bary_posterior_path(root: Path, method: str, variant: str, perm: int, seed: int) -> Path:
    return root / "posteriors" / f"{method}_{variant}_bary_perm{perm:04d}_s{seed}.npy"


def _nobary_posterior_path(root: Path, method: str, variant: str, seed: int) -> Path:
    return root / "posteriors" / f"{method}_{variant}_nobnt_s{seed}.npy"


def _baseline_root(method: str, args: argparse.Namespace) -> Path:
    if method == "cnn":
        return args.cnn_baseline_root.resolve()
    if method == "l1":
        return args.l1_baseline_root.resolve()
    if method == "l1vmim":
        return args.l1vmim_baseline_root.resolve()
    raise ValueError(f"Unsupported method '{method}'.")


def _load_many(paths: Iterable[Path]) -> np.ndarray:
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


def _plot_pair(
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


def _plot_trio(
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
    chain_vm = _mcs(samples_l1vmim, MCSamples, METHOD_LABELS["l1vmim"])

    g = gplot.get_subplot_plotter(subplot_size=1.4)
    g.settings.alpha_filled_add = 0.55
    g.triangle_plot(
        [chain_cnn, chain_l1, chain_vm],
        filled=True,
        contour_colors=["#1f77b4", "#d62728", "#2ca02c"],
        markers=TRUTH,
        marker_args={"color": "black", "lw": 1.0},
        legend_loc="upper right",
    )
    g.fig.suptitle(title, fontsize=12)
    g.fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(g.fig)


def _require_all(paths: Iterable[Path]) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        msg = "Missing required posterior files:\n" + "\n".join(str(p) for p in missing)
        raise FileNotFoundError(msg)


def main() -> None:
    args = parse_args()
    study_root = args.study_root.resolve()
    variant = args.variant
    seeds = _csv_ints(args.seeds)
    perms = sorted(set(_csv_ints(args.perm_indices)))
    selected_perms = sorted(set(_csv_ints(args.selected_perms))) if args.selected_perms else []

    if not seeds:
        raise ValueError("--seeds cannot be empty.")
    if not perms:
        raise ValueError("--perm-indices cannot be empty.")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (study_root / "overlays").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    plt, MCSamples, gplot = _import_plotting()
    written: List[Path] = []

    # Per-method combined overlays: no-bary baseline vs baryonified combined.
    for method in METHODS:
        baseline_root = _baseline_root(method, args)
        nobary_paths = [
            _nobary_posterior_path(baseline_root, method, variant, seed)
            for seed in seeds
        ]
        bary_paths = [
            _bary_posterior_path(study_root, method, variant, perm, seed)
            for perm in perms
            for seed in seeds
        ]
        _require_all(nobary_paths)
        _require_all(bary_paths)

        out_path = output_dir / f"overlay_{method}_{variant}_nobary_vs_bary_combined.png"
        _plot_pair(
            out_path=out_path,
            samples_a=_load_many(nobary_paths),
            samples_b=_load_many(bary_paths),
            label_a=f"{METHOD_LABELS[method]} no-bary",
            label_b=f"{METHOD_LABELS[method]} baryonified",
            title=f"{METHOD_LABELS[method]} {variant}: no-bary vs baryonified (combined)",
            plt=plt,
            MCSamples=MCSamples,
            gplot=gplot,
            dpi=args.dpi,
        )
        written.append(out_path)

    # Trio no-bary combined (reference).
    nobary_trio = {}
    for method in METHODS:
        baseline_root = _baseline_root(method, args)
        paths = [_nobary_posterior_path(baseline_root, method, variant, seed) for seed in seeds]
        _require_all(paths)
        nobary_trio[method] = _load_many(paths)

    out_nobary_trio = output_dir / f"overlay_trio_{variant}_nobary_combined.png"
    _plot_trio(
        out_path=out_nobary_trio,
        samples_cnn=nobary_trio["cnn"],
        samples_l1=nobary_trio["l1"],
        samples_l1vmim=nobary_trio["l1vmim"],
        title=f"{variant}: CNN vs L1 vs L1+VMIM (no-bary combined)",
        plt=plt,
        MCSamples=MCSamples,
        gplot=gplot,
        dpi=args.dpi,
    )
    written.append(out_nobary_trio)

    # Trio baryonified combined.
    bary_trio = {}
    for method in METHODS:
        paths = [
            _bary_posterior_path(study_root, method, variant, perm, seed)
            for perm in perms
            for seed in seeds
        ]
        _require_all(paths)
        bary_trio[method] = _load_many(paths)

    out_bary_trio = output_dir / f"overlay_trio_{variant}_baryonified_combined.png"
    _plot_trio(
        out_path=out_bary_trio,
        samples_cnn=bary_trio["cnn"],
        samples_l1=bary_trio["l1"],
        samples_l1vmim=bary_trio["l1vmim"],
        title=f"{variant}: CNN vs L1 vs L1+VMIM (baryonified combined)",
        plt=plt,
        MCSamples=MCSamples,
        gplot=gplot,
        dpi=args.dpi,
    )
    written.append(out_bary_trio)

    # Optional selected permutation trios (combined over seeds for each chosen perm).
    for perm in selected_perms:
        if perm not in perms:
            continue
        perm_paths = {
            method: [
                _bary_posterior_path(study_root, method, variant, perm, seed)
                for seed in seeds
            ]
            for method in METHODS
        }
        for paths in perm_paths.values():
            _require_all(paths)

        out_path = output_dir / f"overlay_trio_{variant}_bary_perm{perm:04d}_combined.png"
        _plot_trio(
            out_path=out_path,
            samples_cnn=_load_many(perm_paths["cnn"]),
            samples_l1=_load_many(perm_paths["l1"]),
            samples_l1vmim=_load_many(perm_paths["l1vmim"]),
            title=f"{variant}: CNN vs L1 vs L1+VMIM (baryonified perm {perm:04d}, combined seeds)",
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

