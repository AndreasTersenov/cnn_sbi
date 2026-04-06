#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


DEFAULT_STUDY_ROOT = Path("scripts/sbi/nobnt_tomo_bins_crosscorr_study")
DEFAULT_VARIANTS = "bin1_20deg160,bin2_20deg160,bin3_20deg160,bin4_20deg160,tomo4_20deg160"
DEFAULT_METHODS = "cnn,l1,l1vmim"
DEFAULT_SEEDS = "41,42,43"


def _csv_tokens(value: str) -> List[str]:
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def _csv_ints(value: str) -> List[int]:
    return [int(tok) for tok in _csv_tokens(value)]


def _posterior_path(root: Path, method: str, variant: str, seed: int) -> Path:
    return root / "posteriors" / f"{method}_{variant}_nobnt_s{seed}.npy"


def _fom3_from_samples(samples: np.ndarray) -> Tuple[float, float, float, bool]:
    cov3 = np.cov(samples[:, :3], rowvar=False)
    sign, logdet = np.linalg.slogdet(cov3)
    if sign <= 0:
        return float("nan"), float("nan"), float(logdet), False
    det = float(np.exp(logdet))
    fom3 = float(np.exp(-0.5 * logdet))
    return fom3, det, float(logdet), True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute FoM on (Omega_m, sigma_8, w_0) for no-BNT tomographic-bin "
            "cross-correlation study."
        )
    )
    p.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    p.add_argument("--methods", type=str, default=DEFAULT_METHODS)
    p.add_argument("--variants", type=str, default=DEFAULT_VARIANTS)
    p.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Defaults to <study-root>/fom3_analysis.json",
    )
    p.add_argument(
        "--per-run-csv",
        type=Path,
        default=None,
        help="Defaults to <study-root>/fom3_per_run.csv",
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Defaults to <study-root>/fom3_summary.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    study_root = args.study_root.resolve()
    methods = _csv_tokens(args.methods)
    variants = _csv_tokens(args.variants)
    seeds = _csv_ints(args.seeds)

    if not methods:
        raise ValueError("--methods cannot be empty.")
    if not variants:
        raise ValueError("--variants cannot be empty.")
    if not seeds:
        raise ValueError("--seeds cannot be empty.")

    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else (study_root / "fom3_analysis.json")
    )
    per_run_csv = (
        args.per_run_csv.resolve()
        if args.per_run_csv is not None
        else (study_root / "fom3_per_run.csv")
    )
    summary_csv = (
        args.summary_csv.resolve()
        if args.summary_csv is not None
        else (study_root / "fom3_summary.csv")
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    per_run_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    missing_files: List[str] = []
    ndim_values: List[int] = []

    for method in methods:
        for variant in variants:
            for seed in seeds:
                p = _posterior_path(study_root, method, variant, seed)
                if not p.exists():
                    missing_files.append(str(p))
                    continue
                s = np.load(p)
                if s.ndim != 2:
                    raise ValueError(f"Posterior array must be 2D, got {s.ndim} for {p}")
                n_samples, n_dim = int(s.shape[0]), int(s.shape[1])
                if n_samples < 2:
                    raise ValueError(f"Posterior has too few samples ({n_samples}) for {p}")
                if n_dim < 3:
                    raise ValueError(f"Posterior has too few dimensions ({n_dim}) for {p}")
                ndim_values.append(n_dim)
                fom3, det_cov3, logdet_cov3, valid_fom = _fom3_from_samples(s)
                rows.append(
                    {
                        "method": method,
                        "variant": variant,
                        "seed": seed,
                        "file": p.name,
                        "n_samples": n_samples,
                        "n_dim": n_dim,
                        "fom3": fom3,
                        "det_cov3": det_cov3,
                        "logdet_cov3": logdet_cov3,
                        "valid_fom": bool(valid_fom),
                    }
                )

    by_group: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    by_group_total: Dict[Tuple[str, str], int] = defaultdict(int)
    for row in rows:
        key = (str(row["method"]), str(row["variant"]))
        by_group_total[key] += 1
        if bool(row["valid_fom"]):
            by_group[key].append(float(row["fom3"]))

    summary_rows: List[Dict[str, object]] = []
    for method in methods:
        for variant in variants:
            key = (method, variant)
            vals = by_group.get(key, [])
            n_valid = len(vals)
            n_total = by_group_total.get(key, 0)
            if n_valid > 0:
                vals_np = np.array(vals, dtype=np.float64)
                mean = float(np.mean(vals_np))
                std = float(np.std(vals_np, ddof=1)) if n_valid > 1 else 0.0
            else:
                mean, std = float("nan"), float("nan")
            summary_rows.append(
                {
                    "method": method,
                    "variant": variant,
                    "fom3_mean": mean,
                    "fom3_std": std,
                    "n_valid": n_valid,
                    "n_total": n_total,
                }
            )

    summary_lookup = {(r["method"], r["variant"]): r for r in summary_rows}
    bin_variants = [v for v in variants if v.startswith("bin")]
    if "tomo4_20deg160" in variants and "cnn" in methods:
        pass

    def _safe_ratio(num: float, den: float) -> float:
        if not np.isfinite(num) or not np.isfinite(den) or den == 0.0:
            return float("nan")
        return float(num / den)

    attribution_rows: List[Dict[str, object]] = []
    for comparator in ("l1", "l1vmim"):
        if comparator not in methods or "cnn" not in methods:
            continue
        cnn_tomo = summary_lookup.get(("cnn", "tomo4_20deg160"))
        cmp_tomo = summary_lookup.get((comparator, "tomo4_20deg160"))
        if cnn_tomo is None or cmp_tomo is None:
            continue
        r_full = _safe_ratio(
            float(cnn_tomo["fom3_mean"]),
            float(cmp_tomo["fom3_mean"]),
        )

        per_bin_ratios: List[float] = []
        for bv in bin_variants:
            cnn_bin = summary_lookup.get(("cnn", bv))
            cmp_bin = summary_lookup.get((comparator, bv))
            if cnn_bin is None or cmp_bin is None:
                continue
            ratio = _safe_ratio(
                float(cnn_bin["fom3_mean"]),
                float(cmp_bin["fom3_mean"]),
            )
            if np.isfinite(ratio):
                per_bin_ratios.append(ratio)
        r_bin_avg = (
            float(np.mean(np.array(per_bin_ratios, dtype=np.float64)))
            if per_bin_ratios
            else float("nan")
        )
        g_corr = _safe_ratio(r_full, r_bin_avg)
        attribution_rows.append(
            {
                "comparator": comparator,
                "r_full_tomo4": r_full,
                "r_bin_avg": r_bin_avg,
                "g_corr": g_corr,
                "n_bins_used": len(per_bin_ratios),
                "per_bin_ratios": per_bin_ratios,
            }
        )

    expected_count = len(methods) * len(variants) * len(seeds)
    present_count = len(rows)
    valid_fom_count = sum(1 for r in rows if bool(r["valid_fom"]))
    all_dim_six = bool(ndim_values) and all(d == 6 for d in ndim_values)
    all_expected_present = present_count == expected_count and len(missing_files) == 0
    all_fom_valid = valid_fom_count == present_count

    group_complete = True
    for method in methods:
        for variant in variants:
            if by_group_total.get((method, variant), 0) != len(seeds):
                group_complete = False
                break
        if not group_complete:
            break

    analysis = {
        "study_root": str(study_root),
        "assumptions": {
            "fom_subspace_param_indices": [0, 1, 2],
            "fom_subspace_param_names": ["Omega_m", "sigma_8", "w_0"],
            "parameter_order_assumption": (
                "Posterior columns follow repository-standard ordering: "
                "[Omega_m, sigma_8, w_0, h_0, n_s, Omega_b]."
            ),
        },
        "inputs": {
            "methods": methods,
            "variants": variants,
            "seeds": seeds,
            "expected_posterior_count": expected_count,
        },
        "sanity_checks": {
            "present_posterior_count": present_count,
            "missing_posterior_count": len(missing_files),
            "missing_posterior_files": missing_files,
            "all_expected_posteriors_present": all_expected_present,
            "all_method_variant_groups_complete": group_complete,
            "all_posteriors_have_dim6": all_dim_six,
            "valid_fom_count": valid_fom_count,
            "all_fom_valid": all_fom_valid,
        },
        "per_run_rows": rows,
        "summary_rows": summary_rows,
        "attribution": attribution_rows,
    }

    output_json.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    with open(per_run_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "variant",
                "seed",
                "file",
                "n_samples",
                "n_dim",
                "fom3",
                "det_cov3",
                "logdet_cov3",
                "valid_fom",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "variant",
                "fom3_mean",
                "fom3_std",
                "n_valid",
                "n_total",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote analysis JSON: {output_json}")
    print(f"Wrote per-run CSV:   {per_run_csv}")
    print(f"Wrote summary CSV:   {summary_csv}")
    if missing_files:
        print(f"WARNING: missing {len(missing_files)} posterior files.")
    if not all_fom_valid:
        invalid = present_count - valid_fom_count
        print(f"WARNING: {invalid} runs had non-positive covariance determinant in FoM3.")


if __name__ == "__main__":
    main()
