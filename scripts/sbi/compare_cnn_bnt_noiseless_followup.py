#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NOISELESS_ROOT = (
    ROOT
    / "scripts"
    / "sbi"
    / "results"
    / "final"
    / "paper_sbi_consolidation"
    / "cnn_noiseless_vs_noisy"
    / "posteriors"
)
DEFAULT_WORKING_NOISY_ROOT = (
    ROOT
    / "scripts"
    / "sbi"
    / "results"
    / "final"
    / "paper_sbi_consolidation"
    / "cnn_bnt_losslessness_campaign_cdim10"
    / "advanced_arch64_dense256_nostd"
    / "posteriors"
)
DEFAULT_OUT_ROOT = (
    ROOT
    / "scripts"
    / "sbi"
    / "results"
    / "final"
    / "paper_sbi_consolidation"
    / "cnn_noiseless_vs_noisy"
    / "followup_working_bnt_compare"
)

TRUTH = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493], dtype=np.float64)
PARAM_NAMES = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]


def _csv_ints(value: str) -> List[int]:
    return [int(tok.strip()) for tok in value.split(",") if tok.strip()]


def fom3(samples: np.ndarray) -> float:
    cov3 = np.cov(samples[:, :3], rowvar=False)
    sign, logdet = np.linalg.slogdet(cov3)
    return float(np.exp(-0.5 * logdet)) if sign > 0 else float("nan")


def om_s8_area(samples: np.ndarray) -> float:
    cov2 = np.cov(samples[:, :2], rowvar=False)
    det = float(np.linalg.det(cov2))
    return float(np.pi * np.sqrt(det)) if det > 0 else float("nan")


def ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or b == 0.0:
        return float("nan")
    return float(a / b)


def set_metrics(paths: Iterable[Path]) -> Dict[str, float]:
    std_sum_vals = []
    sigma8_vals = []
    fom_vals = []
    area_vals = []
    n = 0
    for path in paths:
        n += 1
        samples = np.load(path)
        std = np.std(samples, axis=0)
        std_sum_vals.append(float(np.sum(std)))
        sigma8_vals.append(float(std[1]))
        fom_vals.append(fom3(samples))
        area_vals.append(om_s8_area(samples))
    arr = lambda values: np.asarray(values, dtype=np.float64)
    std_sum_arr = arr(std_sum_vals)
    sigma8_arr = arr(sigma8_vals)
    fom_arr = arr(fom_vals)
    area_arr = arr(area_vals)
    return {
        "n": int(n),
        "std_sum_mean": float(np.mean(std_sum_arr)),
        "std_sum_std": float(np.std(std_sum_arr, ddof=1)) if n > 1 else 0.0,
        "sigma8_std_mean": float(np.mean(sigma8_arr)),
        "sigma8_std_std": float(np.std(sigma8_arr, ddof=1)) if n > 1 else 0.0,
        "fom3_mean": float(np.nanmean(fom_arr)),
        "fom3_std": float(np.nanstd(fom_arr, ddof=1)) if n > 1 else 0.0,
        "om_s8_area_mean": float(np.nanmean(area_arr)),
        "om_s8_area_std": float(np.nanstd(area_arr, ddof=1)) if n > 1 else 0.0,
    }


def _concat(paths: Iterable[Path]) -> np.ndarray:
    return np.concatenate([np.load(path) for path in paths], axis=0)


def _plot_overlay(
    out_path: Path,
    paths_a: List[Path],
    label_a: str,
    paths_b: List[Path],
    label_b: str,
    title: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from getdist import MCSamples, plots as gplot
    except Exception:
        return

    samples_a = _concat(paths_a)
    samples_b = _concat(paths_b)
    chain_a = MCSamples(
        samples=samples_a,
        names=PARAM_NAMES,
        labels=PARAM_NAMES,
        label=label_a,
        settings={"smooth_scale_2D": 0.7, "smooth_scale_1D": 0.7},
    )
    chain_b = MCSamples(
        samples=samples_b,
        names=PARAM_NAMES,
        labels=PARAM_NAMES,
        label=label_b,
        settings={"smooth_scale_2D": 0.7, "smooth_scale_1D": 0.7},
    )
    g = gplot.get_subplot_plotter(subplot_size=1.35)
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(g.fig)


def _fmt(v: float) -> str:
    return f"{v:.4f}" if np.isfinite(v) else "nan"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare noiseless BNT against working noisy BNT, and compare noiseless "
            "BNT vs noiseless no-BNT with overlays."
        )
    )
    p.add_argument("--noiseless-root", type=Path, default=DEFAULT_NOISELESS_ROOT)
    p.add_argument("--working-noisy-root", type=Path, default=DEFAULT_WORKING_NOISY_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--seeds", type=str, default="41,42,43")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _csv_ints(args.seeds)
    out_root = args.output_root.resolve()
    fig_root = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_root.mkdir(parents=True, exist_ok=True)

    noiseless_root = args.noiseless_root.resolve()
    noisy_root = args.working_noisy_root.resolve()

    paths_noiseless_bnt = [
        noiseless_root / f"cnn_tomo4_20deg160_bnt_noiseless_s{seed}.npy"
        for seed in seeds
    ]
    paths_noiseless_nobnt = [
        noiseless_root / f"cnn_tomo4_20deg160_nobnt_noiseless_s{seed}.npy"
        for seed in seeds
    ]
    paths_noisy_working_bnt = [
        noisy_root / f"cnn_tomo4_20deg160_bnt_advanced_arch64_dense256_nostd_s{seed}.npy"
        for seed in seeds
    ]
    paths_noisy_working_nobnt = [
        noisy_root / f"cnn_tomo4_20deg160_nobnt_advanced_arch64_dense256_nostd_s{seed}.npy"
        for seed in seeds
    ]

    for path in (
        paths_noiseless_bnt
        + paths_noiseless_nobnt
        + paths_noisy_working_bnt
        + paths_noisy_working_nobnt
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing posterior file: {path}")

    metrics_noiseless_bnt = set_metrics(paths_noiseless_bnt)
    metrics_noiseless_nobnt = set_metrics(paths_noiseless_nobnt)
    metrics_noisy_working_bnt = set_metrics(paths_noisy_working_bnt)
    metrics_noisy_working_nobnt = set_metrics(paths_noisy_working_nobnt)

    cmp_noiseless_bnt_vs_noisy_working_bnt = {
        "std_sum_ratio_noiseless_over_noisy_working": ratio(
            metrics_noiseless_bnt["std_sum_mean"],
            metrics_noisy_working_bnt["std_sum_mean"],
        ),
        "fom3_ratio_noiseless_over_noisy_working": ratio(
            metrics_noiseless_bnt["fom3_mean"],
            metrics_noisy_working_bnt["fom3_mean"],
        ),
        "sigma8_std_ratio_noiseless_over_noisy_working": ratio(
            metrics_noiseless_bnt["sigma8_std_mean"],
            metrics_noisy_working_bnt["sigma8_std_mean"],
        ),
        "om_s8_area_ratio_noiseless_over_noisy_working": ratio(
            metrics_noiseless_bnt["om_s8_area_mean"],
            metrics_noisy_working_bnt["om_s8_area_mean"],
        ),
    }
    cmp_noisy_working_bnt_vs_nobnt = {
        "std_sum_ratio_bnt_over_nobnt": ratio(
            metrics_noisy_working_bnt["std_sum_mean"],
            metrics_noisy_working_nobnt["std_sum_mean"],
        ),
        "fom3_ratio_bnt_over_nobnt": ratio(
            metrics_noisy_working_bnt["fom3_mean"],
            metrics_noisy_working_nobnt["fom3_mean"],
        ),
        "sigma8_std_ratio_bnt_over_nobnt": ratio(
            metrics_noisy_working_bnt["sigma8_std_mean"],
            metrics_noisy_working_nobnt["sigma8_std_mean"],
        ),
        "om_s8_area_ratio_bnt_over_nobnt": ratio(
            metrics_noisy_working_bnt["om_s8_area_mean"],
            metrics_noisy_working_nobnt["om_s8_area_mean"],
        ),
    }
    cmp_noiseless_bnt_vs_noiseless_nobnt = {
        "std_sum_ratio_bnt_over_nobnt": ratio(
            metrics_noiseless_bnt["std_sum_mean"],
            metrics_noiseless_nobnt["std_sum_mean"],
        ),
        "fom3_ratio_bnt_over_nobnt": ratio(
            metrics_noiseless_bnt["fom3_mean"],
            metrics_noiseless_nobnt["fom3_mean"],
        ),
        "sigma8_std_ratio_bnt_over_nobnt": ratio(
            metrics_noiseless_bnt["sigma8_std_mean"],
            metrics_noiseless_nobnt["sigma8_std_mean"],
        ),
        "om_s8_area_ratio_bnt_over_nobnt": ratio(
            metrics_noiseless_bnt["om_s8_area_mean"],
            metrics_noiseless_nobnt["om_s8_area_mean"],
        ),
    }

    payload = {
        "seeds": seeds,
        "paths": {
            "noiseless_bnt": [str(path) for path in paths_noiseless_bnt],
            "noiseless_nobnt": [str(path) for path in paths_noiseless_nobnt],
            "noisy_working_bnt": [str(path) for path in paths_noisy_working_bnt],
            "noisy_working_nobnt": [str(path) for path in paths_noisy_working_nobnt],
        },
        "metrics": {
            "noiseless_bnt": metrics_noiseless_bnt,
            "noiseless_nobnt": metrics_noiseless_nobnt,
            "noisy_working_bnt": metrics_noisy_working_bnt,
            "noisy_working_nobnt": metrics_noisy_working_nobnt,
        },
        "comparisons": {
            "noiseless_bnt_vs_noisy_working_bnt": cmp_noiseless_bnt_vs_noisy_working_bnt,
            "noisy_working_bnt_vs_noisy_working_nobnt": cmp_noisy_working_bnt_vs_nobnt,
            "noiseless_bnt_vs_noiseless_nobnt": cmp_noiseless_bnt_vs_noiseless_nobnt,
        },
    }
    (out_root / "comparison_working_noisy_bnt_vs_noiseless.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    with open(
        out_root / "comparison_working_noisy_bnt_vs_noiseless.csv",
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "comparison",
                "std_ratio",
                "fom_ratio",
                "sigma8_std_ratio",
                "om_s8_area_ratio",
            ]
        )
        c = cmp_noiseless_bnt_vs_noisy_working_bnt
        writer.writerow(
            [
                "noiseless_bnt_vs_noisy_working_bnt",
                c["std_sum_ratio_noiseless_over_noisy_working"],
                c["fom3_ratio_noiseless_over_noisy_working"],
                c["sigma8_std_ratio_noiseless_over_noisy_working"],
                c["om_s8_area_ratio_noiseless_over_noisy_working"],
            ]
        )
        c = cmp_noisy_working_bnt_vs_nobnt
        writer.writerow(
            [
                "noisy_working_bnt_vs_noisy_working_nobnt",
                c["std_sum_ratio_bnt_over_nobnt"],
                c["fom3_ratio_bnt_over_nobnt"],
                c["sigma8_std_ratio_bnt_over_nobnt"],
                c["om_s8_area_ratio_bnt_over_nobnt"],
            ]
        )
        c = cmp_noiseless_bnt_vs_noiseless_nobnt
        writer.writerow(
            [
                "noiseless_bnt_vs_noiseless_nobnt",
                c["std_sum_ratio_bnt_over_nobnt"],
                c["fom3_ratio_bnt_over_nobnt"],
                c["sigma8_std_ratio_bnt_over_nobnt"],
                c["om_s8_area_ratio_bnt_over_nobnt"],
            ]
        )

    _plot_overlay(
        out_path=fig_root / "overlay_noisy_working_bnt_vs_noiseless_bnt_combined.png",
        paths_a=paths_noisy_working_bnt,
        label_a="noisy BNT (working cdim10)",
        paths_b=paths_noiseless_bnt,
        label_b="noiseless BNT",
        title="CNN BNT: noisy working vs noiseless (seeds 41-43)",
    )
    _plot_overlay(
        out_path=fig_root / "overlay_noiseless_nobnt_vs_noiseless_bnt_combined.png",
        paths_a=paths_noiseless_nobnt,
        label_a="noiseless no-BNT",
        paths_b=paths_noiseless_bnt,
        label_b="noiseless BNT",
        title="CNN noiseless: no-BNT vs BNT (seeds 41-43)",
    )

    lines = [
        "# Working noisy-BNT vs noiseless comparisons",
        "",
        "- Working noisy BNT source: "
        "`cnn_bnt_losslessness_campaign_cdim10/advanced_arch64_dense256_nostd`",
        "- Seed subset used for fair comparison: `41,42,43`",
        "",
        "## Ratios",
        "",
        "| comparison | std ratio | fom ratio | sigma8 std ratio | Om-s8 area ratio |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    c = cmp_noiseless_bnt_vs_noisy_working_bnt
    lines.append(
        "| noiseless BNT / noisy working BNT | "
        f"{_fmt(c['std_sum_ratio_noiseless_over_noisy_working'])} | "
        f"{_fmt(c['fom3_ratio_noiseless_over_noisy_working'])} | "
        f"{_fmt(c['sigma8_std_ratio_noiseless_over_noisy_working'])} | "
        f"{_fmt(c['om_s8_area_ratio_noiseless_over_noisy_working'])} |"
    )
    c = cmp_noisy_working_bnt_vs_nobnt
    lines.append(
        "| noisy working BNT / noisy working no-BNT | "
        f"{_fmt(c['std_sum_ratio_bnt_over_nobnt'])} | "
        f"{_fmt(c['fom3_ratio_bnt_over_nobnt'])} | "
        f"{_fmt(c['sigma8_std_ratio_bnt_over_nobnt'])} | "
        f"{_fmt(c['om_s8_area_ratio_bnt_over_nobnt'])} |"
    )
    c = cmp_noiseless_bnt_vs_noiseless_nobnt
    lines.append(
        "| noiseless BNT / noiseless no-BNT | "
        f"{_fmt(c['std_sum_ratio_bnt_over_nobnt'])} | "
        f"{_fmt(c['fom3_ratio_bnt_over_nobnt'])} | "
        f"{_fmt(c['sigma8_std_ratio_bnt_over_nobnt'])} | "
        f"{_fmt(c['om_s8_area_ratio_bnt_over_nobnt'])} |"
    )
    lines += [
        "",
        "## Figures",
        "",
        "- `figures/overlay_noisy_working_bnt_vs_noiseless_bnt_combined.png`",
        "- `figures/overlay_noiseless_nobnt_vs_noiseless_bnt_combined.png`",
    ]
    (out_root / "WORKING_NOISY_BNT_VS_NOISELESS_REPORT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Saved follow-up outputs to: {out_root}")


if __name__ == "__main__":
    main()
