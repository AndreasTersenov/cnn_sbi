#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PARAM_NAMES = ["Omega_m", "sigma_8", "w0", "h0", "n_s", "Omega_b"]
PAIR_INDICES = [(0, 1), (0, 2)]
PAIR_LABELS = {(0, 1): "Omega_m-sigma_8", (0, 2): "Omega_m-w0"}
POSTERIOR_FILE_RE = re.compile(r"^(cnn|l1)_(?P<variant>.+)_s(?P<seed>\d+)\.npy$")
TRUTH_THETA = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_cnn = repo_root / "scripts" / "sbi" / "systematic_runs_cnn_retrain_proper"
    default_l1 = repo_root / "scripts" / "sbi" / "systematic_runs_l1_rerun_proper"
    default_out = repo_root / "scripts" / "sbi" / "diagnostics" / "degeneracy_v2"

    p = argparse.ArgumentParser(
        description="Degeneracy diagnostics v2 for CNN/L1 rerun outputs and training caches."
    )
    p.add_argument("--cnn-run-dir", type=Path, default=default_cnn)
    p.add_argument("--l1-run-dir", type=Path, default=default_l1)
    p.add_argument("--output-dir", type=Path, default=default_out)
    p.add_argument(
        "--global-train-samples",
        type=int,
        default=20000,
        help="Max train rows used per variant for global ridge sensitivity.",
    )
    p.add_argument(
        "--global-val-samples",
        type=int,
        default=10000,
        help="Max validation rows used per variant for global ridge sensitivity.",
    )
    p.add_argument(
        "--local-neighbors",
        type=int,
        default=15000,
        help="Number of local train neighbors (in theta-space) for local Jacobian proxy.",
    )
    p.add_argument(
        "--ridge-alpha-cnn",
        type=float,
        default=1e-3,
        help="Ridge regularization for CNN summary->theta regression.",
    )
    p.add_argument(
        "--ridge-alpha-l1",
        type=float,
        default=1e-1,
        help="Ridge regularization for L1 summary->theta regression.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def parse_posterior_name(path: Path) -> Tuple[str, str, int]:
    m = POSTERIOR_FILE_RE.match(path.name)
    if not m:
        raise ValueError(f"Unexpected posterior filename format: {path.name}")
    method = path.name.split("_", 1)[0]
    return method, m.group("variant"), int(m.group("seed"))


def cache_prefix(method: str, variant: str) -> str:
    return f"cnn_{variant}" if method == "cnn" else f"l1_{variant}"


def cache_file_prefix(method: str) -> str:
    return "cnn" if method == "cnn" else "l1"


def load_cache(run_dir: Path, method: str, variant: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cprefix = cache_prefix(method, variant)
    fprefix = cache_file_prefix(method)
    train = np.load(run_dir / "cache" / cprefix / f"{fprefix}_train.npz")
    val = np.load(run_dir / "cache" / cprefix / f"{fprefix}_val.npz")
    return train["x"].astype(np.float64), train["theta"].astype(np.float64), val["x"].astype(np.float64), val["theta"].astype(np.float64)


def prior_range_from_theta(train_theta: np.ndarray, val_theta: np.ndarray) -> np.ndarray:
    theta_all = np.vstack([train_theta, val_theta])
    return theta_all.max(axis=0) - theta_all.min(axis=0)


def compute_pair_geometry(cov: np.ndarray, i: int, j: int) -> Dict[str, float]:
    sub = cov[np.ix_([i, j], [i, j])]
    evals, evecs = np.linalg.eigh(sub)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    denom = float(max(evals[-1], 1e-15))
    axis_ratio = float(math.sqrt(float(evals[0]) / denom))
    angle_deg = float(np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0])))
    return {
        "axis_ratio": axis_ratio,
        "angle_deg": angle_deg,
        "eigvals": [float(evals[0]), float(evals[1])],
    }


def compute_posterior_diagnostics(run_dirs: Dict[str, Path]) -> Dict[str, object]:
    per_run: List[Dict[str, object]] = []
    corr_by_variant_method: Dict[Tuple[str, str], List[np.ndarray]] = defaultdict(list)
    cov_by_variant_method: Dict[Tuple[str, str], List[np.ndarray]] = defaultdict(list)

    for method, run_dir in run_dirs.items():
        for posterior_path in sorted((run_dir / "posteriors").glob(f"{method}_*.npy")):
            _, variant, seed = parse_posterior_name(posterior_path)
            samples = np.load(posterior_path).astype(np.float64)
            cov = np.cov(samples, rowvar=False)
            corr = np.corrcoef(samples, rowvar=False)
            eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
            cond = float(eigvals[0] / max(eigvals[-1], 1e-15))

            cov_first3 = np.cov(samples[:, :3], rowvar=False)
            eig_first3 = np.sort(np.linalg.eigvalsh(cov_first3))[::-1]
            cond_first3 = float(eig_first3[0] / max(eig_first3[-1], 1e-15))

            cov_last3 = np.cov(samples[:, 3:], rowvar=False)
            eig_last3 = np.sort(np.linalg.eigvalsh(cov_last3))[::-1]
            cond_last3 = float(eig_last3[0] / max(eig_last3[-1], 1e-15))

            pair_metrics = {}
            for i, j in PAIR_INDICES:
                pair_key = f"{i}-{j}"
                geom = compute_pair_geometry(cov, i, j)
                pair_metrics[pair_key] = {
                    "corr": float(corr[i, j]),
                    "axis_ratio": geom["axis_ratio"],
                    "angle_deg": geom["angle_deg"],
                }

            row = {
                "method": method,
                "variant": variant,
                "seed": seed,
                "n_samples": int(samples.shape[0]),
                "covariance": cov.tolist(),
                "correlation": corr.tolist(),
                "eigvals_desc": eigvals.tolist(),
                "condition_number": cond,
                "condition_number_first3": cond_first3,
                "condition_number_last3": cond_last3,
                "pair_metrics": pair_metrics,
            }
            per_run.append(row)
            corr_by_variant_method[(method, variant)].append(corr)
            cov_by_variant_method[(method, variant)].append(cov)

    variant_avg: List[Dict[str, object]] = []
    method_avg_groups: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for (method, variant), corr_list in sorted(corr_by_variant_method.items()):
        cov_stack = np.stack(cov_by_variant_method[(method, variant)], axis=0)
        corr_stack = np.stack(corr_list, axis=0)
        avg_cov = cov_stack.mean(axis=0)
        avg_corr = corr_stack.mean(axis=0)
        eigvals = np.sort(np.linalg.eigvalsh(avg_cov))[::-1]

        pair_summary = {}
        for i, j in PAIR_INDICES:
            geom = compute_pair_geometry(avg_cov, i, j)
            pair_summary[f"{i}-{j}"] = {
                "corr": float(avg_corr[i, j]),
                "axis_ratio": geom["axis_ratio"],
                "angle_deg": geom["angle_deg"],
            }

        cond = float(eigvals[0] / max(eigvals[-1], 1e-15))
        cond_first3 = float(
            np.sort(np.linalg.eigvalsh(avg_cov[:3, :3]))[::-1][0]
            / max(np.sort(np.linalg.eigvalsh(avg_cov[:3, :3]))[::-1][-1], 1e-15)
        )
        cond_last3 = float(
            np.sort(np.linalg.eigvalsh(avg_cov[3:, 3:]))[::-1][0]
            / max(np.sort(np.linalg.eigvalsh(avg_cov[3:, 3:]))[::-1][-1], 1e-15)
        )

        out = {
            "method": method,
            "variant": variant,
            "n_seeds": int(corr_stack.shape[0]),
            "avg_covariance": avg_cov.tolist(),
            "avg_correlation": avg_corr.tolist(),
            "eigvals_desc": eigvals.tolist(),
            "condition_number": cond,
            "condition_number_first3": cond_first3,
            "condition_number_last3": cond_last3,
            "pair_metrics": pair_summary,
        }
        variant_avg.append(out)

        method_avg_groups[method]["condition_number"].append(cond)
        method_avg_groups[method]["condition_number_first3"].append(cond_first3)
        method_avg_groups[method]["condition_number_last3"].append(cond_last3)
        for key, data in pair_summary.items():
            method_avg_groups[method][f"pair_{key}_corr"].append(data["corr"])
            method_avg_groups[method][f"pair_{key}_axis_ratio"].append(data["axis_ratio"])

    method_avg: Dict[str, Dict[str, float]] = {}
    for method, metric_map in sorted(method_avg_groups.items()):
        method_avg[method] = {k: float(np.mean(v)) for k, v in sorted(metric_map.items())}

    return {
        "per_run": sorted(per_run, key=lambda r: (r["method"], r["variant"], r["seed"])),
        "variant_averages": sorted(variant_avg, key=lambda r: (r["method"], r["variant"])),
        "method_averages": method_avg,
    }


def sample_rows(x: np.ndarray, theta: np.ndarray, n_rows: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) <= n_rows:
        return x, theta
    idx = rng.choice(len(x), size=n_rows, replace=False)
    return x[idx], theta[idx]


def ridge_fit_eval(
    train_x: np.ndarray,
    train_theta: np.ndarray,
    val_x: np.ndarray,
    val_theta: np.ndarray,
    alpha: float,
) -> Dict[str, object]:
    x_mean = train_x.mean(axis=0)
    x_std = train_x.std(axis=0) + 1e-8
    y_mean = train_theta.mean(axis=0)
    y_std = train_theta.std(axis=0) + 1e-8

    x_train_n = (train_x - x_mean) / x_std
    y_train_n = (train_theta - y_mean) / y_std
    x_val_n = (val_x - x_mean) / x_std

    xtx = x_train_n.T @ x_train_n
    xty = x_train_n.T @ y_train_n
    w = np.linalg.solve(xtx + alpha * np.eye(xtx.shape[0]), xty)

    pred = x_val_n @ w
    pred = pred * y_std + y_mean

    residual = val_theta - pred
    ss_res = np.sum(residual**2, axis=0)
    ss_tot = np.sum((val_theta - val_theta.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)

    jacobian = w.T
    jac_svals = np.linalg.svd(jacobian, compute_uv=False)
    row_norms = np.linalg.norm(jacobian, axis=1)

    cov_x = np.cov(x_train_n, rowvar=False)
    jac_cov = jacobian @ cov_x @ jacobian.T
    jac_cov += np.eye(jac_cov.shape[0]) * 1e-12
    diag = np.sqrt(np.diag(jac_cov))
    jac_corr = jac_cov / np.maximum(diag[:, None] * diag[None, :], 1e-12)

    return {
        "r2_per_param": r2.tolist(),
        "r2_first3_mean": float(np.mean(r2[:3])),
        "r2_last3_mean": float(np.mean(r2[3:])),
        "r2_gap_first3_minus_last3": float(np.mean(r2[:3]) - np.mean(r2[3:])),
        "jacobian_singular_values": jac_svals.tolist(),
        "jacobian_condition_number": float(jac_svals[0] / max(jac_svals[-1], 1e-12)),
        "jacobian_row_norms": row_norms.tolist(),
        "jacobian_covariance": jac_cov.tolist(),
        "jacobian_correlation": jac_corr.tolist(),
        "theta_rmse_per_param": np.sqrt(np.mean(residual**2, axis=0)).tolist(),
        "alpha": float(alpha),
    }


def local_neighbor_indices(theta: np.ndarray, prior_range: np.ndarray, k: int) -> np.ndarray:
    scale = np.maximum(prior_range, 1e-12)
    dist = np.linalg.norm((theta - TRUTH_THETA) / scale, axis=1)
    k = min(k, len(theta))
    if k >= len(theta):
        return np.arange(len(theta))
    return np.argpartition(dist, kth=k - 1)[:k]


def compute_sensitivity_diagnostics(
    run_dirs: Dict[str, Path],
    variants: List[str],
    global_train_samples: int,
    global_val_samples: int,
    local_neighbors: int,
    ridge_alpha: Dict[str, float],
    seed: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []

    for method, run_dir in run_dirs.items():
        for variant in variants:
            train_x, train_theta, val_x, val_theta = load_cache(run_dir, method, variant)
            prior_range = prior_range_from_theta(train_theta, val_theta)

            global_train_x, global_train_theta = sample_rows(train_x, train_theta, global_train_samples, rng)
            global_val_x, global_val_theta = sample_rows(val_x, val_theta, global_val_samples, rng)
            global_fit = ridge_fit_eval(
                global_train_x,
                global_train_theta,
                global_val_x,
                global_val_theta,
                alpha=ridge_alpha[method],
            )

            local_idx = local_neighbor_indices(train_theta, prior_range, local_neighbors)
            local_x = train_x[local_idx]
            local_theta = train_theta[local_idx]
            local_perm = rng.permutation(len(local_x))
            local_x = local_x[local_perm]
            local_theta = local_theta[local_perm]
            split = max(int(0.8 * len(local_x)), 2)
            split = min(split, len(local_x) - 1)
            local_train_x = local_x[:split]
            local_train_theta = local_theta[:split]
            local_val_x = local_x[split:]
            local_val_theta = local_theta[split:]
            local_fit = ridge_fit_eval(
                local_train_x,
                local_train_theta,
                local_val_x,
                local_val_theta,
                alpha=ridge_alpha[method],
            )

            pair_alignment = {}
            for i, j in PAIR_INDICES:
                key = f"{i}-{j}"
                pair_alignment[key] = {
                    "global_jac_corr": float(global_fit["jacobian_correlation"][i][j]),
                    "local_jac_corr": float(local_fit["jacobian_correlation"][i][j]),
                }

            row = {
                "method": method,
                "variant": variant,
                "global": {
                    **global_fit,
                    "train_samples": int(len(global_train_x)),
                    "val_samples": int(len(global_val_x)),
                },
                "local": {
                    **local_fit,
                    "train_samples": int(len(local_train_x)),
                    "val_samples": int(len(local_val_x)),
                    "local_neighbors": int(len(local_idx)),
                },
                "pair_alignment": pair_alignment,
            }
            rows.append(row)

            del train_x, train_theta, val_x, val_theta
            del global_train_x, global_train_theta, global_val_x, global_val_theta
            del local_x, local_theta, local_train_x, local_train_theta, local_val_x, local_val_theta

    method_summary: Dict[str, Dict[str, float]] = {}
    for method in sorted(run_dirs.keys()):
        subset = [r for r in rows if r["method"] == method]
        method_summary[method] = {
            "global_r2_first3_mean": float(np.mean([r["global"]["r2_first3_mean"] for r in subset])),
            "global_r2_last3_mean": float(np.mean([r["global"]["r2_last3_mean"] for r in subset])),
            "local_r2_first3_mean": float(np.mean([r["local"]["r2_first3_mean"] for r in subset])),
            "local_r2_last3_mean": float(np.mean([r["local"]["r2_last3_mean"] for r in subset])),
            "global_jac_cond_mean": float(np.mean([r["global"]["jacobian_condition_number"] for r in subset])),
            "local_jac_cond_mean": float(np.mean([r["local"]["jacobian_condition_number"] for r in subset])),
            "local_pair01_corr_mean": float(np.mean([r["pair_alignment"]["0-1"]["local_jac_corr"] for r in subset])),
            "local_pair02_corr_mean": float(np.mean([r["pair_alignment"]["0-2"]["local_jac_corr"] for r in subset])),
        }

    return {
        "per_variant": sorted(rows, key=lambda r: (r["method"], r["variant"])),
        "method_averages": method_summary,
    }


def compare_expected_pairs(
    posterior_diag: Dict[str, object], sensitivity_diag: Dict[str, object]
) -> List[Dict[str, object]]:
    sens_map = {(r["method"], r["variant"]): r for r in sensitivity_diag["per_variant"]}
    rows: List[Dict[str, object]] = []

    for vrow in posterior_diag["variant_averages"]:
        method = vrow["method"]
        variant = vrow["variant"]
        srow = sens_map[(method, variant)]
        for i, j in PAIR_INDICES:
            key = f"{i}-{j}"
            pair_name = PAIR_LABELS[(i, j)]
            posterior_corr = float(vrow["pair_metrics"][key]["corr"])
            posterior_axis = float(vrow["pair_metrics"][key]["axis_ratio"])
            expected_corr = float(srow["pair_alignment"][key]["local_jac_corr"])
            corr_gap = posterior_corr - expected_corr
            sign_match = np.sign(posterior_corr) == np.sign(expected_corr) or abs(expected_corr) < 0.05
            rows.append(
                {
                    "method": method,
                    "variant": variant,
                    "pair": pair_name,
                    "posterior_corr": posterior_corr,
                    "expected_corr_local_jac": expected_corr,
                    "corr_gap": float(corr_gap),
                    "posterior_axis_ratio": posterior_axis,
                    "sign_match": bool(sign_match),
                }
            )
    return sorted(rows, key=lambda r: (r["method"], r["variant"], r["pair"]))


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_condition_numbers(per_run: List[Dict[str, object]], out_path: Path) -> None:
    labels = [f"{r['method']}:{r['variant']}:s{r['seed']}" for r in per_run]
    cond = [r["condition_number"] for r in per_run]
    cond_first3 = [r["condition_number_first3"] for r in per_run]
    cond_last3 = [r["condition_number_last3"] for r in per_run]

    x = np.arange(len(labels))
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(12, 0.65 * len(labels)), 5.5))
    ax.bar(x - width, cond, width=width, label="all 6 params")
    ax.bar(x, cond_first3, width=width, label="first3 (Omega_m,sigma_8,w0)")
    ax.bar(x + width, cond_last3, width=width, label="last3 (h0,n_s,Omega_b)")
    ax.set_yscale("log")
    ax.set_ylabel("Covariance condition number (log scale)")
    ax.set_title("Posterior covariance conditioning per run")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_eigenspectra(per_run: List[Dict[str, object]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in per_run:
        eig = np.asarray(r["eigvals_desc"], dtype=float)
        ax.plot(np.arange(1, len(eig) + 1), eig, marker="o", linewidth=1.3, alpha=0.75, label=f"{r['method']}:{r['variant']}:s{r['seed']}")
    ax.set_yscale("log")
    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title("Posterior covariance eigenvalue spectra")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_corr_heatmaps(variant_averages: List[Dict[str, object]], out_dir: Path) -> None:
    by_variant: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
    for row in variant_averages:
        by_variant[row["variant"]][row["method"]] = np.asarray(row["avg_correlation"], dtype=float)

    for variant, method_map in sorted(by_variant.items()):
        methods = [m for m in ["cnn", "l1"] if m in method_map]
        fig, axes = plt.subplots(1, len(methods), figsize=(5.0 * len(methods), 4.3), squeeze=False)
        for idx, method in enumerate(methods):
            corr = method_map[method]
            ax = axes[0, idx]
            im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
            ax.set_title(f"{method.upper()} avg corr\n{variant}")
            ax.set_xticks(np.arange(len(PARAM_NAMES)))
            ax.set_yticks(np.arange(len(PARAM_NAMES)))
            ax.set_xticklabels(PARAM_NAMES, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(PARAM_NAMES, fontsize=8)
            for i in range(corr.shape[0]):
                for j in range(corr.shape[1]):
                    ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=6, color="black")
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
        cbar.set_label("Correlation")
        fig.tight_layout()
        fig.savefig(out_dir / f"corr_heatmap_{variant}.png", dpi=170)
        plt.close(fig)


def plot_pair_structure(expected_rows: List[Dict[str, object]], out_path: Path) -> None:
    pair_names = ["Omega_m-sigma_8", "Omega_m-w0"]
    methods = ["cnn", "l1"]
    variants = sorted({r["variant"] for r in expected_rows})
    xpos = np.arange(len(variants))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col")

    for col, pair in enumerate(pair_names):
        for method in methods:
            vals_post = []
            vals_exp = []
            vals_axis = []
            for variant in variants:
                rows = [r for r in expected_rows if r["pair"] == pair and r["method"] == method and r["variant"] == variant]
                if rows:
                    vals_post.append(rows[0]["posterior_corr"])
                    vals_exp.append(rows[0]["expected_corr_local_jac"])
                    vals_axis.append(rows[0]["posterior_axis_ratio"])
                else:
                    vals_post.append(np.nan)
                    vals_exp.append(np.nan)
                    vals_axis.append(np.nan)
            marker = "o" if method == "cnn" else "s"
            ls = "-" if method == "cnn" else "--"
            axes[0, col].plot(xpos, vals_post, marker=marker, linestyle=ls, label=f"{method.upper()} posterior")
            axes[0, col].plot(xpos, vals_exp, marker=marker, linestyle=":", label=f"{method.upper()} local-Jac")
            axes[1, col].plot(xpos, vals_axis, marker=marker, linestyle=ls, label=f"{method.upper()} axis ratio")

        axes[0, col].axhline(0.0, color="k", linewidth=0.8, alpha=0.4)
        axes[0, col].set_ylim(-1.0, 1.0)
        axes[0, col].set_title(f"{pair}: correlation")
        axes[0, col].set_ylabel("corr")
        axes[0, col].grid(alpha=0.25)

        axes[1, col].set_title(f"{pair}: posterior ellipse axis ratio")
        axes[1, col].set_ylabel("axis ratio")
        axes[1, col].grid(alpha=0.25)

    axes[0, 0].legend(fontsize=8, loc="lower left")
    axes[1, 0].legend(fontsize=8, loc="upper left")

    for ax in axes[1, :]:
        ax.set_xticks(xpos)
        ax.set_xticklabels(variants, rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_sensitivity_r2(sensitivity_rows: List[Dict[str, object]], out_path: Path) -> None:
    labels = [f"{r['method']}:{r['variant']}" for r in sensitivity_rows]
    x = np.arange(len(labels))
    width = 0.22

    g_first3 = [r["global"]["r2_first3_mean"] for r in sensitivity_rows]
    g_last3 = [r["global"]["r2_last3_mean"] for r in sensitivity_rows]
    l_first3 = [r["local"]["r2_first3_mean"] for r in sensitivity_rows]
    l_last3 = [r["local"]["r2_last3_mean"] for r in sensitivity_rows]

    fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(labels)), 5.5))
    ax.bar(x - 1.5 * width, g_first3, width=width, label="global R² first3")
    ax.bar(x - 0.5 * width, g_last3, width=width, label="global R² last3")
    ax.bar(x + 0.5 * width, l_first3, width=width, label="local R² first3")
    ax.bar(x + 1.5 * width, l_last3, width=width, label="local R² last3")
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.4)
    ax.set_ylabel("R²")
    ax.set_title("Summary->theta linear sensitivity: first3 vs last3")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_sensitivity_singular_values(sensitivity_rows: List[Dict[str, object]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for r in sensitivity_rows:
        label = f"{r['method']}:{r['variant']}"
        g_sv = np.asarray(r["global"]["jacobian_singular_values"], dtype=float)
        l_sv = np.asarray(r["local"]["jacobian_singular_values"], dtype=float)
        axes[0].plot(np.arange(1, len(g_sv) + 1), g_sv, marker="o", alpha=0.75, label=label)
        axes[1].plot(np.arange(1, len(l_sv) + 1), l_sv, marker="o", alpha=0.75, label=label)

    axes[0].set_title("Global Jacobian proxy singular values")
    axes[1].set_title("Local Jacobian proxy singular values")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("Singular value rank")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("singular value (log scale)")
    axes[1].legend(fontsize=7, ncol=1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def build_findings(
    posterior_diag: Dict[str, object], sensitivity_diag: Dict[str, object], expected_rows: List[Dict[str, object]]
) -> Dict[str, object]:
    method_posterior = posterior_diag["method_averages"]
    method_sensitivity = sensitivity_diag["method_averages"]

    mismatch = [r for r in expected_rows if not r["sign_match"]]
    mismatch_count = len(mismatch)

    worst_gap = sorted(expected_rows, key=lambda r: abs(r["corr_gap"]), reverse=True)[:6]

    return {
        "posterior_method_averages": method_posterior,
        "sensitivity_method_averages": method_sensitivity,
        "pair_sign_mismatch_count": mismatch_count,
        "largest_pair_corr_gaps": worst_gap,
        "interpretation": {
            "circular_contours_indicator": "Low |corr| and axis-ratio near 1 imply circular/uninformative pair contours.",
            "lost_degeneracy_indicator": "Large posterior-vs-local-Jacobian corr gaps or sign flips indicate lost/rotated degeneracy directions.",
            "first3_vs_last3_indicator": "R²(first3) >> R²(last3) means summary features encode first3 much more strongly than last3.",
        },
    }


def flatten_posterior_rows(per_run: List[Dict[str, object]]) -> List[Dict[str, object]]:
    flat = []
    for r in per_run:
        row = {
            "method": r["method"],
            "variant": r["variant"],
            "seed": r["seed"],
            "condition_number": r["condition_number"],
            "condition_number_first3": r["condition_number_first3"],
            "condition_number_last3": r["condition_number_last3"],
            "eig1": r["eigvals_desc"][0],
            "eig2": r["eigvals_desc"][1],
            "eig3": r["eigvals_desc"][2],
            "eig4": r["eigvals_desc"][3],
            "eig5": r["eigvals_desc"][4],
            "eig6": r["eigvals_desc"][5],
            "corr_omega_m_sigma8": r["pair_metrics"]["0-1"]["corr"],
            "corr_omega_m_w0": r["pair_metrics"]["0-2"]["corr"],
            "axis_omega_m_sigma8": r["pair_metrics"]["0-1"]["axis_ratio"],
            "axis_omega_m_w0": r["pair_metrics"]["0-2"]["axis_ratio"],
        }
        flat.append(row)
    return flat


def flatten_sensitivity_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    flat = []
    for r in rows:
        row = {
            "method": r["method"],
            "variant": r["variant"],
            "global_r2_first3_mean": r["global"]["r2_first3_mean"],
            "global_r2_last3_mean": r["global"]["r2_last3_mean"],
            "global_r2_gap": r["global"]["r2_gap_first3_minus_last3"],
            "local_r2_first3_mean": r["local"]["r2_first3_mean"],
            "local_r2_last3_mean": r["local"]["r2_last3_mean"],
            "local_r2_gap": r["local"]["r2_gap_first3_minus_last3"],
            "global_jac_cond": r["global"]["jacobian_condition_number"],
            "local_jac_cond": r["local"]["jacobian_condition_number"],
            "local_corr_omega_m_sigma8": r["pair_alignment"]["0-1"]["local_jac_corr"],
            "local_corr_omega_m_w0": r["pair_alignment"]["0-2"]["local_jac_corr"],
        }
        flat.append(row)
    return flat


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    ensure_output_dir(output_dir)

    run_dirs = {
        "cnn": args.cnn_run_dir.resolve(),
        "l1": args.l1_run_dir.resolve(),
    }

    for method, path in run_dirs.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {method} run directory: {path}")

    variants = sorted(
        {
            parse_posterior_name(p)[1]
            for method, run_dir in run_dirs.items()
            for p in (run_dir / "posteriors").glob(f"{method}_*.npy")
        }
    )

    posterior_diag = compute_posterior_diagnostics(run_dirs)

    sensitivity_diag = compute_sensitivity_diagnostics(
        run_dirs=run_dirs,
        variants=variants,
        global_train_samples=args.global_train_samples,
        global_val_samples=args.global_val_samples,
        local_neighbors=args.local_neighbors,
        ridge_alpha={"cnn": args.ridge_alpha_cnn, "l1": args.ridge_alpha_l1},
        seed=args.seed,
    )

    expected_rows = compare_expected_pairs(posterior_diag, sensitivity_diag)
    findings = build_findings(posterior_diag, sensitivity_diag, expected_rows)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "cnn_run_dir": str(run_dirs["cnn"]),
            "l1_run_dir": str(run_dirs["l1"]),
            "output_dir": str(output_dir),
            "global_train_samples": args.global_train_samples,
            "global_val_samples": args.global_val_samples,
            "local_neighbors": args.local_neighbors,
            "ridge_alpha_cnn": args.ridge_alpha_cnn,
            "ridge_alpha_l1": args.ridge_alpha_l1,
            "seed": args.seed,
        },
        "param_names": PARAM_NAMES,
        "posterior": posterior_diag,
        "sensitivity": sensitivity_diag,
        "expected_pair_comparison": expected_rows,
        "findings": findings,
    }

    summary_json = output_dir / "degeneracy_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    posterior_flat = flatten_posterior_rows(posterior_diag["per_run"])
    write_csv(
        output_dir / "posterior_per_run_metrics.csv",
        posterior_flat,
        fieldnames=list(posterior_flat[0].keys()),
    )

    sensitivity_flat = flatten_sensitivity_rows(sensitivity_diag["per_variant"])
    write_csv(
        output_dir / "sensitivity_metrics.csv",
        sensitivity_flat,
        fieldnames=list(sensitivity_flat[0].keys()),
    )

    write_csv(
        output_dir / "expected_pair_comparison.csv",
        expected_rows,
        fieldnames=list(expected_rows[0].keys()),
    )

    plot_condition_numbers(posterior_diag["per_run"], output_dir / "posterior_condition_numbers.png")
    plot_eigenspectra(posterior_diag["per_run"], output_dir / "posterior_eigenspectra.png")
    plot_corr_heatmaps(posterior_diag["variant_averages"], output_dir)
    plot_pair_structure(expected_rows, output_dir / "pair_structure_comparison.png")
    plot_sensitivity_r2(sensitivity_diag["per_variant"], output_dir / "sensitivity_r2_first3_last3.png")
    plot_sensitivity_singular_values(
        sensitivity_diag["per_variant"], output_dir / "sensitivity_jacobian_singular_values.png"
    )

    print(f"Saved summary JSON: {summary_json}")
    print(f"Saved metrics CSVs and plots under: {output_dir}")


if __name__ == "__main__":
    main()
