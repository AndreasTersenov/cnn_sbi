#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


TRUTH = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493], dtype=np.float64)
TRUTH3 = TRUTH[:3]
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_ROOT = DEFAULT_REPO_ROOT / "scripts" / "sbi" / "baryon_bias_tomo4_study"


def _default_perm_indices() -> str:
    return ",".join(str(i) for i in range(20))


def _csv_tokens(value: str) -> List[str]:
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def _csv_ints(value: str) -> List[int]:
    return [int(tok) for tok in _csv_tokens(value)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Analyze baryonified-observation bias for tomo4 inference "
            "(CNN / L1-jaxili / L1-VMIM) against no-bary baselines."
        )
    )
    p.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    p.add_argument("--methods", type=str, default="cnn,l1,l1vmim")
    p.add_argument("--seeds", type=str, default="41,42,43")
    p.add_argument("--perm-indices", type=str, default=_default_perm_indices())
    p.add_argument("--variant", type=str, default="tomo4_20deg160")

    p.add_argument(
        "--cnn-baseline-root",
        type=Path,
        default=DEFAULT_REPO_ROOT / "scripts" / "sbi" / "nobnt_tomo_bins_crosscorr_study",
    )
    p.add_argument(
        "--l1-baseline-root",
        type=Path,
        default=DEFAULT_REPO_ROOT / "scripts" / "sbi" / "nobnt_tomo_bins_crosscorr_study_l1_jaxili_bestcfg",
    )
    p.add_argument(
        "--l1vmim-baseline-root",
        type=Path,
        default=DEFAULT_REPO_ROOT / "scripts" / "sbi" / "nobnt_tomo_bins_crosscorr_study",
    )

    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--per-run-csv", type=Path, default=None)
    p.add_argument("--summary-csv", type=Path, default=None)
    return p.parse_args()


def _bary_posterior_path(root: Path, method: str, variant: str, perm: int, seed: int) -> Path:
    return root / "posteriors" / f"{method}_{variant}_bary_perm{perm:04d}_s{seed}.npy"


def _nobary_posterior_path(root: Path, method: str, variant: str, seed: int) -> Path:
    return root / "posteriors" / f"{method}_{variant}_nobnt_s{seed}.npy"


def _method_baseline_root(method: str, args: argparse.Namespace) -> Path:
    if method == "cnn":
        return args.cnn_baseline_root.resolve()
    if method == "l1":
        return args.l1_baseline_root.resolve()
    if method == "l1vmim":
        return args.l1vmim_baseline_root.resolve()
    raise ValueError(f"Unsupported method '{method}'.")


def _fom3(cov3: np.ndarray) -> Tuple[float, float, bool]:
    sign, logdet = np.linalg.slogdet(cov3)
    if sign <= 0:
        return float("nan"), float(logdet), False
    return float(np.exp(-0.5 * logdet)), float(logdet), True


def _mahalanobis(delta: np.ndarray, cov: np.ndarray) -> Tuple[float, bool]:
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return float("nan"), False
    value = float(np.sqrt(np.clip(float(delta.T @ inv_cov @ delta), 0.0, None)))
    return value, True


def _finite_stats(values: Iterable[float]) -> Dict[str, float]:
    arr = np.array([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p16": float("nan"),
            "p84": float("nan"),
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "p16": float(np.percentile(arr, 16)),
        "p84": float(np.percentile(arr, 84)),
    }


def main() -> None:
    args = parse_args()
    study_root = args.study_root.resolve()
    methods = _csv_tokens(args.methods)
    seeds = _csv_ints(args.seeds)
    perms = sorted(set(_csv_ints(args.perm_indices)))
    variant = args.variant

    if not methods:
        raise ValueError("--methods cannot be empty.")
    if not set(methods).issubset({"cnn", "l1", "l1vmim"}):
        raise ValueError("--methods must be subset of {cnn,l1,l1vmim}.")
    if not seeds:
        raise ValueError("--seeds cannot be empty.")
    if not perms:
        raise ValueError("--perm-indices cannot be empty.")

    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else study_root / "baryon_bias_analysis.json"
    )
    per_run_csv = (
        args.per_run_csv.resolve()
        if args.per_run_csv is not None
        else study_root / "baryon_bias_per_run.csv"
    )
    summary_csv = (
        args.summary_csv.resolve()
        if args.summary_csv is not None
        else study_root / "baryon_bias_summary.csv"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    per_run_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    missing_files: List[str] = []
    n_dim_values: List[int] = []
    baseline_cache: Dict[Tuple[str, int], Dict[str, object]] = {}

    for method in methods:
        baseline_root = _method_baseline_root(method, args)
        for seed in seeds:
            baseline_path = _nobary_posterior_path(baseline_root, method, variant, seed)
            if not baseline_path.exists():
                missing_files.append(str(baseline_path))
                continue
            s_nb = np.load(baseline_path)
            if s_nb.ndim != 2:
                raise ValueError(f"Baseline posterior must be 2D: {baseline_path}")
            if s_nb.shape[1] < 3:
                raise ValueError(f"Baseline posterior dim<3: {baseline_path}")
            n_dim_values.append(int(s_nb.shape[1]))

            cov_nb = np.cov(s_nb[:, :3], rowvar=False)
            fom_nb, logdet_nb, fom_nb_valid = _fom3(cov_nb)
            mu_nb = np.mean(s_nb[:, :3], axis=0)
            sigma_nb = np.sqrt(np.clip(np.diag(cov_nb), 0.0, None))
            baseline_cache[(method, seed)] = {
                "path": baseline_path,
                "samples": s_nb,
                "mu3": mu_nb,
                "cov3": cov_nb,
                "sigma3": sigma_nb,
                "fom3": fom_nb,
                "logdet": logdet_nb,
                "fom_valid": fom_nb_valid,
            }

    for method in methods:
        for seed in seeds:
            baseline = baseline_cache.get((method, seed))
            if baseline is None:
                continue
            for perm in perms:
                bary_path = _bary_posterior_path(study_root, method, variant, perm, seed)
                if not bary_path.exists():
                    missing_files.append(str(bary_path))
                    continue
                s_b = np.load(bary_path)
                if s_b.ndim != 2:
                    raise ValueError(f"Baryon posterior must be 2D: {bary_path}")
                if s_b.shape[1] < 3:
                    raise ValueError(f"Baryon posterior dim<3: {bary_path}")
                n_dim_values.append(int(s_b.shape[1]))

                cov_b = np.cov(s_b[:, :3], rowvar=False)
                fom_b, logdet_b, fom_b_valid = _fom3(cov_b)
                fom_nb = float(baseline["fom3"])
                fom_ratio = (
                    float(fom_b / fom_nb)
                    if np.isfinite(fom_b) and np.isfinite(fom_nb) and fom_nb > 0
                    else float("nan")
                )

                mu_b = np.mean(s_b[:, :3], axis=0)
                mu_nb = np.asarray(baseline["mu3"], dtype=np.float64)
                cov_nb = np.asarray(baseline["cov3"], dtype=np.float64)
                sigma_nb = np.asarray(baseline["sigma3"], dtype=np.float64)

                delta_truth = mu_b - TRUTH3
                delta_shift = mu_b - mu_nb

                d_truth, d_truth_valid = _mahalanobis(delta_truth, cov_nb)
                d_shift, d_shift_valid = _mahalanobis(delta_shift, cov_nb)

                norm_truth = np.divide(
                    delta_truth,
                    sigma_nb,
                    out=np.full_like(delta_truth, np.nan),
                    where=sigma_nb > 0,
                )
                norm_shift = np.divide(
                    delta_shift,
                    sigma_nb,
                    out=np.full_like(delta_shift, np.nan),
                    where=sigma_nb > 0,
                )

                rows.append(
                    {
                        "method": method,
                        "variant": variant,
                        "seed": seed,
                        "perm": perm,
                        "bary_file": bary_path.name,
                        "nobary_file": Path(str(baseline["path"])).name,
                        "n_samples_bary": int(s_b.shape[0]),
                        "n_samples_nobary": int(np.asarray(baseline["samples"]).shape[0]),
                        "n_dim": int(s_b.shape[1]),
                        "fom3_bary": fom_b,
                        "fom3_nobary": fom_nb,
                        "fom3_ratio_bary_over_nobary": fom_ratio,
                        "fom3_bary_valid": bool(fom_b_valid),
                        "fom3_nobary_valid": bool(baseline["fom_valid"]),
                        "logdet_cov3_bary": logdet_b,
                        "logdet_cov3_nobary": float(baseline["logdet"]),
                        "d_truth": d_truth,
                        "d_shift": d_shift,
                        "d_truth_valid": bool(d_truth_valid),
                        "d_shift_valid": bool(d_shift_valid),
                        "delta_truth_om": float(delta_truth[0]),
                        "delta_truth_s8": float(delta_truth[1]),
                        "delta_truth_w0": float(delta_truth[2]),
                        "delta_shift_om": float(delta_shift[0]),
                        "delta_shift_s8": float(delta_shift[1]),
                        "delta_shift_w0": float(delta_shift[2]),
                        "norm_truth_om": float(norm_truth[0]),
                        "norm_truth_s8": float(norm_truth[1]),
                        "norm_truth_w0": float(norm_truth[2]),
                        "norm_shift_om": float(norm_shift[0]),
                        "norm_shift_s8": float(norm_shift[1]),
                        "norm_shift_w0": float(norm_shift[2]),
                    }
                )

    summary_rows: List[Dict[str, object]] = []
    by_method: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)

    expected_runs = len(methods) * len(seeds) * len(perms)
    for method in methods:
        m_rows = by_method.get(method, [])
        fom_ratio_stats = _finite_stats([float(r["fom3_ratio_bary_over_nobary"]) for r in m_rows])
        d_truth_stats = _finite_stats([float(r["d_truth"]) for r in m_rows])
        d_shift_stats = _finite_stats([float(r["d_shift"]) for r in m_rows])
        abs_norm_shift_om = _finite_stats([abs(float(r["norm_shift_om"])) for r in m_rows])
        abs_norm_shift_s8 = _finite_stats([abs(float(r["norm_shift_s8"])) for r in m_rows])
        abs_norm_shift_w0 = _finite_stats([abs(float(r["norm_shift_w0"])) for r in m_rows])

        summary_rows.append(
            {
                "method": method,
                "n_runs": len(m_rows),
                "expected_runs_per_method": len(seeds) * len(perms),
                "fom3_ratio_mean": fom_ratio_stats["mean"],
                "fom3_ratio_std": fom_ratio_stats["std"],
                "fom3_ratio_median": fom_ratio_stats["median"],
                "fom3_ratio_p16": fom_ratio_stats["p16"],
                "fom3_ratio_p84": fom_ratio_stats["p84"],
                "d_truth_mean": d_truth_stats["mean"],
                "d_truth_std": d_truth_stats["std"],
                "d_truth_median": d_truth_stats["median"],
                "d_truth_p16": d_truth_stats["p16"],
                "d_truth_p84": d_truth_stats["p84"],
                "d_shift_mean": d_shift_stats["mean"],
                "d_shift_std": d_shift_stats["std"],
                "d_shift_median": d_shift_stats["median"],
                "d_shift_p16": d_shift_stats["p16"],
                "d_shift_p84": d_shift_stats["p84"],
                "abs_norm_shift_om_mean": abs_norm_shift_om["mean"],
                "abs_norm_shift_s8_mean": abs_norm_shift_s8["mean"],
                "abs_norm_shift_w0_mean": abs_norm_shift_w0["mean"],
            }
        )

    all_dim_six = bool(n_dim_values) and all(d == 6 for d in n_dim_values)
    present_runs = len(rows)
    missing_unique = sorted(set(missing_files))
    complete = (present_runs == expected_runs) and (len(missing_unique) == 0)

    analysis = {
        "study_root": str(study_root),
        "variant": variant,
        "inputs": {
            "methods": methods,
            "seeds": seeds,
            "perm_indices": perms,
            "expected_runs": expected_runs,
        },
        "assumptions": {
            "fom_subspace_param_indices": [0, 1, 2],
            "fom_subspace_param_names": ["Omega_m", "sigma_8", "w_0"],
            "parameter_order_assumption": (
                "Posterior columns follow repository standard ordering: "
                "[Omega_m, sigma_8, w_0, h_0, n_s, Omega_b]."
            ),
            "truth": TRUTH.tolist(),
            "metric_definitions": {
                "fom3": "1/sqrt(det(cov(theta[:,:3])))",
                "d_truth": "sqrt((mu_bary-truth)^T C_nobary^{-1} (mu_bary-truth))",
                "d_shift": "sqrt((mu_bary-mu_nobary)^T C_nobary^{-1} (mu_bary-mu_nobary))",
            },
        },
        "sanity_checks": {
            "present_runs": present_runs,
            "expected_runs": expected_runs,
            "all_expected_runs_present": complete,
            "missing_file_count": len(missing_unique),
            "missing_files": missing_unique,
            "all_posteriors_dim6": all_dim_six,
        },
        "baseline_roots": {
            "cnn": str(args.cnn_baseline_root.resolve()),
            "l1": str(args.l1_baseline_root.resolve()),
            "l1vmim": str(args.l1vmim_baseline_root.resolve()),
        },
        "per_run_rows": rows,
        "summary_rows": summary_rows,
    }

    output_json.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    with open(per_run_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "variant",
                "seed",
                "perm",
                "bary_file",
                "nobary_file",
                "n_samples_bary",
                "n_samples_nobary",
                "n_dim",
                "fom3_bary",
                "fom3_nobary",
                "fom3_ratio_bary_over_nobary",
                "fom3_bary_valid",
                "fom3_nobary_valid",
                "logdet_cov3_bary",
                "logdet_cov3_nobary",
                "d_truth",
                "d_shift",
                "d_truth_valid",
                "d_shift_valid",
                "delta_truth_om",
                "delta_truth_s8",
                "delta_truth_w0",
                "delta_shift_om",
                "delta_shift_s8",
                "delta_shift_w0",
                "norm_truth_om",
                "norm_truth_s8",
                "norm_truth_w0",
                "norm_shift_om",
                "norm_shift_s8",
                "norm_shift_w0",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "n_runs",
                "expected_runs_per_method",
                "fom3_ratio_mean",
                "fom3_ratio_std",
                "fom3_ratio_median",
                "fom3_ratio_p16",
                "fom3_ratio_p84",
                "d_truth_mean",
                "d_truth_std",
                "d_truth_median",
                "d_truth_p16",
                "d_truth_p84",
                "d_shift_mean",
                "d_shift_std",
                "d_shift_median",
                "d_shift_p16",
                "d_shift_p84",
                "abs_norm_shift_om_mean",
                "abs_norm_shift_s8_mean",
                "abs_norm_shift_w0_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote analysis JSON: {output_json}")
    print(f"Wrote per-run CSV:   {per_run_csv}")
    print(f"Wrote summary CSV:   {summary_csv}")
    if missing_unique:
        print(f"WARNING: missing {len(missing_unique)} required files.")


if __name__ == "__main__":
    main()
