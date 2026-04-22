#!/usr/bin/env python3
"""Overlay plots: new (demeaned) CNN-VMIM posteriors vs canonical L1-norm.

For each CNN architecture from the zero-mean-maps parity run, concatenate
seeds 41-43 for each of {no-BNT, BNT} and overlay against the canonical L1-norm
posteriors from
  scripts/sbi/results/final/paper_sbi_consolidation/bnt_comparison_tomo4/posteriors/
(`l1_tomo4_20deg160_{bnt,nobnt}_s{41,42,43}.npy`).

Produces:
  overlays/run_a_resnet18_vs_l1_overlay.{png,pdf}
  overlays/run_b_advanced_plain_vs_l1_overlay.{png,pdf}
  metrics/comparison_cnn_vs_l1.{csv,json}
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[5]
TRUTH = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493], dtype=np.float64)
PARAM_NAMES = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]

CNN_COLOR_NOBNT = "#0173B2"
CNN_COLOR_BNT = "#CC78BC"
L1_COLOR_NOBNT = "#029E73"
L1_COLOR_BNT = "#D55E00"


@dataclass(frozen=True)
class OverlayJob:
    name: str
    title: str
    seeds: Tuple[int, ...]
    cnn_dir: Path
    cnn_label: str
    l1_dir: Path
    l1_label: str


def _import_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from getdist import MCSamples, plots as gplot
    return plt, MCSamples, gplot


def _posterior(dir_: Path, prefix: str, condition: str, label: str, seed: int) -> Path:
    # L1 paths are "l1_tomo4_20deg160_<cond>_s{seed}.npy"; CNN paths include label.
    if label:
        return dir_ / f"{prefix}_tomo4_20deg160_{condition}_{label}_s{seed}.npy"
    return dir_ / f"{prefix}_tomo4_20deg160_{condition}_s{seed}.npy"


def _concat(paths: List[Path]) -> np.ndarray:
    return np.concatenate([np.load(p) for p in paths], axis=0)


def _fom3(samples: np.ndarray) -> Tuple[float, bool]:
    cov3 = np.cov(samples[:, :3], rowvar=False)
    sign, logdet = np.linalg.slogdet(cov3)
    if sign <= 0:
        return float("nan"), False
    return float(np.exp(-0.5 * logdet)), True


def _per_seed_metrics(paths: List[Path]) -> Dict[str, float]:
    stds, sigma8s, foms = [], [], []
    for p in paths:
        s = np.load(p)
        stds.append(float(np.sum(np.std(s, axis=0))))
        sigma8s.append(float(np.std(s[:, 1])))
        fom, ok = _fom3(s)
        if ok:
            foms.append(fom)
    return {
        "std_sum_mean": float(np.mean(stds)) if stds else float("nan"),
        "sigma8_std_mean": float(np.mean(sigma8s)) if sigma8s else float("nan"),
        "fom3_mean": float(np.mean(foms)) if foms else float("nan"),
        "n_seeds": len(paths),
    }


def _plot_overlay_cnn_vs_l1(
    out_path: Path,
    cnn_nobnt: np.ndarray,
    cnn_bnt: np.ndarray,
    l1_nobnt: np.ndarray,
    l1_bnt: np.ndarray,
    title: str,
    dpi: int = 150,
) -> None:
    plt, MCSamples, gplot = _import_plotting()
    smooth = {"smooth_scale_2D": 0.7, "smooth_scale_1D": 0.7}
    # Draw widest → tightest so fully opaque fills don't hide smaller contours.
    chains = [
        MCSamples(samples=l1_bnt, names=PARAM_NAMES, labels=PARAM_NAMES,
                  label="L1 / BNT", settings=smooth),
        MCSamples(samples=l1_nobnt, names=PARAM_NAMES, labels=PARAM_NAMES,
                  label="L1 / no-BNT", settings=smooth),
        MCSamples(samples=cnn_bnt, names=PARAM_NAMES, labels=PARAM_NAMES,
                  label="CNN / BNT (demeaned)", settings=smooth),
        MCSamples(samples=cnn_nobnt, names=PARAM_NAMES, labels=PARAM_NAMES,
                  label="CNN / no-BNT (demeaned)", settings=smooth),
    ]
    g = gplot.get_subplot_plotter(subplot_size=1.4)
    g.settings.alpha_filled_add = 1.0
    g.settings.linewidth_contour = 1.6
    g.triangle_plot(
        chains,
        filled=True,
        markers=TRUTH,
        marker_args={"color": "black", "lw": 1.0},
        legend_loc="upper right",
    )
    g.fig.suptitle(title, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.fig.savefig(out_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    g.fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(g.fig)


def main() -> None:
    out_root = Path(__file__).resolve().parent
    overlays_dir = out_root / "overlays"
    metrics_dir = out_root / "metrics"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    l1_dir = (
        REPO_ROOT
        / "scripts/sbi/results/final/paper_sbi_consolidation/bnt_comparison_tomo4/posteriors"
    )
    seeds = (41, 42, 43)

    jobs: List[OverlayJob] = [
        OverlayJob(
            name="run_a_resnet18",
            title=(
                "resnet18 (zero-mean-maps, demeaned CNN) vs L1-norm "
                "[seeds 41-43, BNT & no-BNT]"
            ),
            seeds=seeds,
            cnn_dir=out_root / "run_a_resnet18" / "posteriors",
            cnn_label="resnet18_long15k_nostd6k_l8h256_zm",
            l1_dir=l1_dir,
            l1_label="",
        ),
        OverlayJob(
            name="run_b_advanced_plain",
            title=(
                "advanced_plain_cdim10 (zero-mean-maps, demeaned CNN) vs L1-norm "
                "[seeds 41-43, BNT & no-BNT]"
            ),
            seeds=seeds,
            cnn_dir=out_root / "run_b_advanced_plain" / "posteriors",
            cnn_label="advanced_arch64_dense256_nostd_long_zm",
            l1_dir=l1_dir,
            l1_label="",
        ),
    ]

    rows: List[Dict[str, object]] = []
    overlays_meta: List[Dict[str, str]] = []

    for job in jobs:
        cnn_nobnt_paths = [
            _posterior(job.cnn_dir, "cnn", "nobnt", job.cnn_label, s) for s in job.seeds
        ]
        cnn_bnt_paths = [
            _posterior(job.cnn_dir, "cnn", "bnt", job.cnn_label, s) for s in job.seeds
        ]
        l1_nobnt_paths = [
            _posterior(job.l1_dir, "l1", "nobnt", job.l1_label, s) for s in job.seeds
        ]
        l1_bnt_paths = [
            _posterior(job.l1_dir, "l1", "bnt", job.l1_label, s) for s in job.seeds
        ]

        missing = [
            p
            for p in cnn_nobnt_paths + cnn_bnt_paths + l1_nobnt_paths + l1_bnt_paths
            if not p.exists()
        ]
        if missing:
            print(f"[{job.name}] SKIPPING — {len(missing)} missing posterior file(s):")
            for p in missing:
                print(f"    MISSING: {p}")
            continue

        overlay_out = overlays_dir / f"{job.name}_vs_l1_overlay"
        _plot_overlay_cnn_vs_l1(
            overlay_out,
            _concat(cnn_nobnt_paths),
            _concat(cnn_bnt_paths),
            _concat(l1_nobnt_paths),
            _concat(l1_bnt_paths),
            job.title,
        )
        overlays_meta.append(
            {
                "config": job.name,
                "png": str(overlay_out.with_suffix(".png")),
                "pdf": str(overlay_out.with_suffix(".pdf")),
            }
        )
        print(f"[{job.name}] overlay -> {overlay_out}.{{png,pdf}}")

        for compressor, paths_by_cond in (
            ("cnn_demeaned", {"nobnt": cnn_nobnt_paths, "bnt": cnn_bnt_paths}),
            ("l1_norm", {"nobnt": l1_nobnt_paths, "bnt": l1_bnt_paths}),
        ):
            for regime, paths in paths_by_cond.items():
                m = _per_seed_metrics(paths)
                rows.append(
                    {
                        "config": job.name,
                        "compressor": compressor,
                        "regime": regime,
                        **m,
                    }
                )

    fieldnames = [
        "config",
        "compressor",
        "regime",
        "std_sum_mean",
        "sigma8_std_mean",
        "fom3_mean",
        "n_seeds",
    ]
    csv_path = metrics_dir / "comparison_cnn_vs_l1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fieldnames})
    json_path = metrics_dir / "comparison_cnn_vs_l1.json"
    json_path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "overlays": overlays_meta,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"metrics -> {csv_path}")
    print(f"metrics -> {json_path}")


if __name__ == "__main__":
    main()
