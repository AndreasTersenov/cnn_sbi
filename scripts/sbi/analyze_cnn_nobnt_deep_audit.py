#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


TRUTH = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493], dtype=np.float64)
PARAM_NAMES = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]


def _fom3(samples: np.ndarray) -> Dict[str, float | bool]:
    cov3 = np.cov(samples[:, :3], rowvar=False)
    sign, logdet = np.linalg.slogdet(cov3)
    if sign <= 0:
        return {
            "fom3": float("nan"),
            "det_cov3": float("nan"),
            "logdet_cov3": float(logdet),
            "valid_fom3": False,
        }
    return {
        "fom3": float(np.exp(-0.5 * logdet)),
        "det_cov3": float(np.exp(logdet)),
        "logdet_cov3": float(logdet),
        "valid_fom3": True,
    }


def _om_s8_metrics(samples: np.ndarray) -> Dict[str, float]:
    pair = samples[:, :2]
    cov2 = np.cov(pair, rowvar=False)
    corr = np.corrcoef(pair[:, 0], pair[:, 1])[0, 1]
    det_cov2 = float(np.linalg.det(cov2))
    area_proxy = float(np.pi * np.sqrt(det_cov2)) if det_cov2 > 0 else float("nan")
    return {
        "om_s8_corr": float(corr),
        "om_s8_det_cov": det_cov2,
        "om_s8_area_proxy": area_proxy,
    }


def _metrics(samples: np.ndarray) -> Dict[str, float | bool]:
    std = np.std(samples, axis=0)
    out: Dict[str, float | bool] = {
        "std_sum": float(np.sum(std)),
        "bias_l2": float(np.linalg.norm(np.mean(samples, axis=0) - TRUTH)),
        "sigma8_std": float(std[1]),
        "w0_std": float(std[2]),
    }
    out.update(_fom3(samples))
    out.update(_om_s8_metrics(samples))
    return out


def _agg(rows: List[Dict[str, object]]) -> Dict[str, float]:
    def _mean_std(key: str) -> Dict[str, float]:
        vals = np.array([float(r[key]) for r in rows], dtype=np.float64)
        return {
            f"{key}_mean": float(np.mean(vals)),
            f"{key}_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        }

    out: Dict[str, float] = {"n": float(len(rows))}
    for key in (
        "std_sum",
        "bias_l2",
        "fom3",
        "sigma8_std",
        "w0_std",
        "om_s8_corr",
        "om_s8_area_proxy",
    ):
        out.update(_mean_std(key))
    return out


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(den) or den == 0.0:
        return float("nan")
    return float(num / den)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _concat(paths: List[Path]) -> np.ndarray:
    return np.concatenate([np.load(p) for p in paths], axis=0)


def _plot_overlay(
    out_path: Path,
    baseline_samples: np.ndarray,
    test_samples: np.ndarray,
    baseline_label: str,
    test_label: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from getdist import MCSamples, plots as gplot
    except Exception:
        return

    chain_base = MCSamples(
        samples=baseline_samples,
        names=PARAM_NAMES,
        labels=PARAM_NAMES,
        label=baseline_label,
        settings={"smooth_scale_2D": 0.7, "smooth_scale_1D": 0.7},
    )
    chain_test = MCSamples(
        samples=test_samples,
        names=PARAM_NAMES,
        labels=PARAM_NAMES,
        label=test_label,
        settings={"smooth_scale_2D": 0.7, "smooth_scale_1D": 0.7},
    )
    g = gplot.get_subplot_plotter(subplot_size=1.35)
    g.settings.alpha_filled_add = 0.55
    g.triangle_plot(
        [chain_base, chain_test],
        filled=True,
        contour_colors=["#1f77b4", "#d62728"],
        markers=TRUTH,
        marker_args={"color": "black", "lw": 1.0},
        legend_loc="upper right",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(g.fig)


def _fmt(v: float) -> str:
    if not np.isfinite(v):
        return "nan"
    return f"{v:.4f}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze no-BNT CNN deep-audit campaign outputs and build report."
    )
    p.add_argument("--campaign-root", type=Path, required=True)
    p.add_argument("--report-out", type=Path, default=None)
    p.add_argument("--summary-json", type=Path, default=None)
    p.add_argument("--summary-csv", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    manifest_path = campaign_root / "campaign_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing campaign manifest: {manifest_path}")
    manifest = _load_json(manifest_path)

    runs = manifest.get("runs", [])
    if not runs:
        raise ValueError(f"No runs found in manifest: {manifest_path}")

    run_names = [str(r["name"]) for r in runs]
    baseline_name = "baseline_fulltrain" if "baseline_fulltrain" in run_names else run_names[0]

    rows: List[Dict[str, object]] = []
    combined_samples: Dict[str, np.ndarray] = {}
    for run in runs:
        run_name = str(run["name"])
        seeds = [int(s) for s in run["seeds"]]
        sample_paths: List[Path] = []
        for seed in seeds:
            posterior_path = (
                campaign_root
                / "posteriors"
                / f"cnn_tomo4_20deg160_nobnt_{run_name}_s{seed}.npy"
            )
            if not posterior_path.exists():
                raise FileNotFoundError(f"Missing posterior: {posterior_path}")
            samples = np.load(posterior_path)
            sample_paths.append(posterior_path)
            metric = _metrics(samples)
            row = {"run_name": run_name, "seed": seed, "file": str(posterior_path)}
            row.update(metric)
            rows.append(row)
        combined_samples[run_name] = _concat(sample_paths)

    by_run: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_run.setdefault(str(row["run_name"]), []).append(row)

    summary_by_run: Dict[str, Dict[str, float]] = {
        run_name: _agg(run_rows) for run_name, run_rows in by_run.items()
    }
    baseline = summary_by_run[baseline_name]

    comparisons: Dict[str, Dict[str, float]] = {}
    for run_name, agg in summary_by_run.items():
        comparisons[run_name] = {
            "std_sum_ratio_vs_baseline": _safe_ratio(
                float(agg["std_sum_mean"]),
                float(baseline["std_sum_mean"]),
            ),
            "fom3_ratio_vs_baseline": _safe_ratio(
                float(agg["fom3_mean"]),
                float(baseline["fom3_mean"]),
            ),
            "sigma8_std_ratio_vs_baseline": _safe_ratio(
                float(agg["sigma8_std_mean"]),
                float(baseline["sigma8_std_mean"]),
            ),
            "om_s8_area_ratio_vs_baseline": _safe_ratio(
                float(agg["om_s8_area_proxy_mean"]),
                float(baseline["om_s8_area_proxy_mean"]),
            ),
            "om_s8_corr_delta_vs_baseline": float(
                agg["om_s8_corr_mean"] - baseline["om_s8_corr_mean"]
            ),
        }

    figures_dir = campaign_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    baseline_samples = combined_samples[baseline_name]
    for run_name in run_names:
        if run_name == baseline_name:
            continue
        _plot_overlay(
            out_path=figures_dir / f"overlay_{baseline_name}_vs_{run_name}_combined.png",
            baseline_samples=baseline_samples,
            test_samples=combined_samples[run_name],
            baseline_label=baseline_name,
            test_label=run_name,
        )

    summary_json_path = (
        args.summary_json.resolve()
        if args.summary_json is not None
        else campaign_root / "audit_summary.json"
    )
    summary_payload = {
        "baseline_run": baseline_name,
        "rows": rows,
        "summary_by_run": summary_by_run,
        "comparisons_vs_baseline": comparisons,
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    summary_csv_path = (
        args.summary_csv.resolve()
        if args.summary_csv is not None
        else campaign_root / "audit_summary.csv"
    )
    with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_name",
                "std_sum_mean",
                "fom3_mean",
                "sigma8_std_mean",
                "om_s8_corr_mean",
                "om_s8_area_proxy_mean",
                "std_sum_ratio_vs_baseline",
                "fom3_ratio_vs_baseline",
                "sigma8_std_ratio_vs_baseline",
                "om_s8_area_ratio_vs_baseline",
                "om_s8_corr_delta_vs_baseline",
            ]
        )
        for run_name in run_names:
            agg = summary_by_run[run_name]
            cmp = comparisons[run_name]
            writer.writerow(
                [
                    run_name,
                    agg["std_sum_mean"],
                    agg["fom3_mean"],
                    agg["sigma8_std_mean"],
                    agg["om_s8_corr_mean"],
                    agg["om_s8_area_proxy_mean"],
                    cmp["std_sum_ratio_vs_baseline"],
                    cmp["fom3_ratio_vs_baseline"],
                    cmp["sigma8_std_ratio_vs_baseline"],
                    cmp["om_s8_area_ratio_vs_baseline"],
                    cmp["om_s8_corr_delta_vs_baseline"],
                ]
            )

    data_pipeline_audit_path = campaign_root / "data_pipeline_audit.json"
    data_pipeline_audit = (
        _load_json(data_pipeline_audit_path) if data_pipeline_audit_path.exists() else None
    )

    report_out = (
        args.report_out.resolve()
        if args.report_out is not None
        else campaign_root / "CNN_NOBNT_DEEP_AUDIT_REPORT.md"
    )
    report_lines: List[str] = []
    report_lines.append("# CNN no-BNT deep audit report")
    report_lines.append("")
    report_lines.append(f"- Campaign root: `{campaign_root}`")
    report_lines.append(f"- Baseline run: `{baseline_name}`")
    report_lines.append(
        "- Metrics: width (`std_sum`), FoM3, `sigma8` std, and "
        "Omega_m-sigma_8 covariance proxies."
    )
    report_lines.append("")

    if data_pipeline_audit is not None:
        overlap = data_pipeline_audit.get("overlap", {})
        tv = overlap.get("train_vs_val", {})
        ab = overlap.get("split_a_vs_split_b", {})
        prep = data_pipeline_audit.get("preprocess_checks", {})
        report_lines.append("## Static pipeline checks")
        report_lines.append("")
        report_lines.append(
            f"- train/test tfds_id overlap: `{tv.get('tfds_id_overlap', 'n/a')}`; "
            f"theta overlap: `{tv.get('theta_overlap', 'n/a')}`."
        )
        report_lines.append(
            f"- split_a/split_b tfds_id overlap: `{ab.get('tfds_id_overlap', 'n/a')}`; "
            f"theta overlap: `{ab.get('theta_overlap', 'n/a')}`."
        )
        report_lines.append(
            f"- h0 rescale check passed: `{prep.get('theta_h0_rescaled_pass', 'n/a')}`; "
            f"augmentation stochastic check passed: "
            f"`{prep.get('augmentation_stochastic_pass', 'n/a')}`."
        )
        report_lines.append("")

    report_lines.append("## Dynamic control summary (mean across seeds)")
    report_lines.append("")
    report_lines.append(
        "| run | std_sum | FoM3 | sigma8_std | corr(Om,s8) | std_ratio | fom_ratio | sigma8_ratio |"
    )
    report_lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for run_name in run_names:
        agg = summary_by_run[run_name]
        cmp = comparisons[run_name]
        report_lines.append(
            f"| `{run_name}` | {_fmt(float(agg['std_sum_mean']))} | "
            f"{_fmt(float(agg['fom3_mean']))} | {_fmt(float(agg['sigma8_std_mean']))} | "
            f"{_fmt(float(agg['om_s8_corr_mean']))} | "
            f"{_fmt(float(cmp['std_sum_ratio_vs_baseline']))} | "
            f"{_fmt(float(cmp['fom3_ratio_vs_baseline']))} | "
            f"{_fmt(float(cmp['sigma8_std_ratio_vs_baseline']))} |"
        )
    report_lines.append("")

    report_lines.append("## Interpretation highlights")
    report_lines.append("")
    for control_name in ("baseline_fulltrain_shuffle", "split70_disjoint_shuffle"):
        if control_name in comparisons:
            cmp = comparisons[control_name]
            report_lines.append(
                f"- `{control_name}`: std_ratio={_fmt(float(cmp['std_sum_ratio_vs_baseline']))}, "
                f"fom_ratio={_fmt(float(cmp['fom3_ratio_vs_baseline']))}, "
                f"sigma8_ratio={_fmt(float(cmp['sigma8_std_ratio_vs_baseline']))}."
            )
    for run_name in ("split70_disjoint", "split70_small_nde10", "split70_long12000"):
        if run_name in comparisons:
            cmp = comparisons[run_name]
            report_lines.append(
                f"- `{run_name}`: std_ratio={_fmt(float(cmp['std_sum_ratio_vs_baseline']))}, "
                f"fom_ratio={_fmt(float(cmp['fom3_ratio_vs_baseline']))}, "
                f"corr_delta={_fmt(float(cmp['om_s8_corr_delta_vs_baseline']))}."
            )
    report_lines.append("")
    report_lines.append(
        "Overlay figures were written to `figures/overlay_<baseline>_vs_<run>_combined.png`."
    )
    report_out.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved summary JSON → {summary_json_path}")
    print(f"Saved summary CSV  → {summary_csv_path}")
    print(f"Saved report       → {report_out}")


if __name__ == "__main__":
    main()
