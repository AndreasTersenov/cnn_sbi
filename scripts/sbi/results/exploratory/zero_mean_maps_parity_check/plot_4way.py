#!/usr/bin/env python3
"""Produce 4-way overlay plots + old-vs-new metrics CSV/JSON.

Loads the two sets of reference posteriors (no-BNT & BNT, "old" = non-demeaned
training, "new" = demeaned training with `--zero-mean-maps`) for each of:
  - Run A: resnet18_long15k_nostd6k_l8h256 (seeds 41-43)
  - Run B: advanced_arch64_dense256_nostd_long (seeds 41-45)

Produces per-config overlay PNG+PDF via
`run_cnn_bnt_losslessness_campaign._plot_overlay_4way` and a metrics
comparison CSV/JSON via `run_cnn_bnt_losslessness_campaign._metrics_for_paths`.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[5]
CAMPAIGN_SCRIPT = REPO_ROOT / "scripts" / "sbi" / "run_cnn_bnt_losslessness_campaign.py"


def _import_campaign_helpers():
    spec = importlib.util.spec_from_file_location(
        "_losslessness_campaign", CAMPAIGN_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {CAMPAIGN_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass(frozen=True)
class OverlayConfig:
    name: str
    title: str
    seeds: Tuple[int, ...]
    old_dir: Path
    old_label: str
    new_dir: Path
    new_label: str


def _posterior_path(dir_: Path, condition: str, label: str, seed: int) -> Path:
    return dir_ / f"cnn_tomo4_20deg160_{condition}_{label}_s{seed}.npy"


def _concat(paths: List[Path]) -> np.ndarray:
    return np.concatenate([np.load(p) for p in paths], axis=0)


def _sigma8_std_mean(paths: List[Path]) -> float:
    if not paths:
        return float("nan")
    vals = np.array([float(np.std(np.load(p)[:, 1])) for p in paths], dtype=np.float64)
    return float(np.mean(vals))


def _expand_row(
    *,
    config_name: str,
    variant: str,
    regime: str,
    paths_by_cond: Dict[str, List[Path]],
    metrics_for_paths,
) -> Dict[str, object]:
    block = metrics_for_paths(paths_by_cond)
    agg = block[regime]
    return {
        "config": config_name,
        "variant": variant,
        "regime": regime,
        "std_sum_mean": float(agg["std_sum_mean"]),
        "fom3_mean": float(agg["fom3_mean"]),
        "sigma8_std_mean": _sigma8_std_mean(paths_by_cond[regime]),
        "fom3_ratio_bnt_over_nobnt": float(block["fom3_ratio_bnt_over_nobnt"]),
        "inflation_std_sum_bnt_over_nobnt": float(
            block["inflation_std_sum_bnt_over_nobnt"]
        ),
        "n_seeds": int(agg["n"]),
    }


def main() -> None:
    mod = _import_campaign_helpers()
    plot_4way = mod._plot_overlay_4way
    metrics_for_paths = mod._metrics_for_paths

    out_root = Path(__file__).resolve().parent
    overlays_dir = out_root / "overlays"
    metrics_dir = out_root / "metrics"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    configs: List[OverlayConfig] = [
        OverlayConfig(
            name="run_a_resnet18",
            title=(
                "resnet18_long15k_nostd6k_l8h256: "
                "demeaned (zero-mean-maps) vs original"
            ),
            seeds=(41, 42, 43),
            old_dir=(
                REPO_ROOT
                / "scripts/sbi/results/final/paper_sbi_consolidation"
                / "cnn_bnt_resnet_split_campaign/resnet_extended_tuning_v2"
                / "backbones/resnet18/resnet18_long15k_nostd6k_l8h256/posteriors"
            ),
            old_label="resnet18_long15k_nostd6k_l8h256",
            new_dir=out_root / "run_a_resnet18" / "posteriors",
            new_label="resnet18_long15k_nostd6k_l8h256_zm",
        ),
        OverlayConfig(
            name="run_b_advanced_plain",
            title=(
                "advanced_arch64_dense256_nostd_long: "
                "demeaned (zero-mean-maps) vs original"
            ),
            seeds=(41, 42, 43, 44, 45),
            old_dir=(
                REPO_ROOT
                / "scripts/sbi/results/final/paper_sbi_consolidation"
                / "cnn_bnt_losslessness_campaign_multipatch_advanced_cdim10_long120k_v1"
                / "advanced_arch64_dense256_nostd_long/posteriors"
            ),
            old_label="advanced_arch64_dense256_nostd_long",
            new_dir=out_root / "run_b_advanced_plain" / "posteriors",
            new_label="advanced_arch64_dense256_nostd_long_zm",
        ),
    ]

    all_rows: List[Dict[str, object]] = []
    overlays_meta: List[Dict[str, str]] = []

    for cfg in configs:
        old_nobnt = [
            _posterior_path(cfg.old_dir, "nobnt", cfg.old_label, s) for s in cfg.seeds
        ]
        old_bnt = [
            _posterior_path(cfg.old_dir, "bnt", cfg.old_label, s) for s in cfg.seeds
        ]
        new_nobnt = [
            _posterior_path(cfg.new_dir, "nobnt", cfg.new_label, s) for s in cfg.seeds
        ]
        new_bnt = [
            _posterior_path(cfg.new_dir, "bnt", cfg.new_label, s) for s in cfg.seeds
        ]

        missing = [p for p in old_nobnt + old_bnt + new_nobnt + new_bnt if not p.exists()]
        if missing:
            print(f"[{cfg.name}] SKIPPING — missing {len(missing)} posterior files:")
            for p in missing:
                print(f"    MISSING: {p}")
            continue

        overlay_out = overlays_dir / f"{cfg.name}_4way_overlay"
        plot_4way(
            overlay_out,
            _concat(old_nobnt),
            _concat(old_bnt),
            _concat(new_nobnt),
            _concat(new_bnt),
            cfg.title,
            dpi=150,
        )
        overlays_meta.append(
            {
                "config": cfg.name,
                "png": str(overlay_out.with_suffix(".png")),
                "pdf": str(overlay_out.with_suffix(".pdf")),
            }
        )
        print(f"[{cfg.name}] overlay -> {overlay_out}.png / .pdf")

        for regime in ("nobnt", "bnt"):
            all_rows.append(
                _expand_row(
                    config_name=cfg.name,
                    variant="old",
                    regime=regime,
                    paths_by_cond={"nobnt": old_nobnt, "bnt": old_bnt},
                    metrics_for_paths=metrics_for_paths,
                )
            )
            all_rows.append(
                _expand_row(
                    config_name=cfg.name,
                    variant="new",
                    regime=regime,
                    paths_by_cond={"nobnt": new_nobnt, "bnt": new_bnt},
                    metrics_for_paths=metrics_for_paths,
                )
            )

    fieldnames = [
        "config",
        "variant",
        "regime",
        "std_sum_mean",
        "fom3_mean",
        "sigma8_std_mean",
        "fom3_ratio_bnt_over_nobnt",
        "inflation_std_sum_bnt_over_nobnt",
        "n_seeds",
    ]
    csv_path = metrics_dir / "comparison_old_vs_new.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow({k: row[k] for k in fieldnames})

    json_path = metrics_dir / "comparison_old_vs_new.json"
    json_path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "overlays": overlays_meta,
                "rows": all_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"metrics -> {csv_path}")
    print(f"metrics -> {json_path}")


if __name__ == "__main__":
    main()
