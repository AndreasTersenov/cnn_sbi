#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_ROOT = (
    DEFAULT_REPO_ROOT / "scripts" / "sbi" / "optimal_nobnt_crosscorr_benchmark"
)
DEFAULT_VARIANTS = "bin1_20deg160,bin2_20deg160,bin3_20deg160,bin4_20deg160,tomo4_20deg160"
DEFAULT_METHODS = "cnn,l1,l1vmim"
DEFAULT_L1_JAXILI = {
    "summary_transform": "log1p-zscore",
    "clip_value": 5.0,
    "learning_rate": 1e-4,
    "epochs": 5000,
}


def _csv_tokens(value: str) -> List[str]:
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def _format_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _build_paths(benchmark_root: Path) -> Dict[str, Path]:
    sweeps_root = benchmark_root / "sweeps"
    selected_root = benchmark_root / "selected_configs"
    matrix_root = benchmark_root / "final_matrix"
    reports_root = benchmark_root / "reports"

    return {
        "benchmark_root": benchmark_root,
        "sweeps_root": sweeps_root,
        "selected_root": selected_root,
        "matrix_root": matrix_root,
        "reports_root": reports_root,
        "sweep_run_root": sweeps_root / "l1_tomo4_opt_sweep",
        "selected_config": selected_root / "l1_tomo4_selected_config.json",
        "matrix_run_root": matrix_root / "nobnt_singlebin_vs_tomo4",
        "analysis_json": reports_root / "fom3_analysis.json",
        "per_run_csv": reports_root / "fom3_per_run.csv",
        "summary_csv": reports_root / "fom3_summary.csv",
        "overlays_dir": reports_root / "overlays",
    }


def _ensure_dirs(paths: Dict[str, Path]) -> None:
    for key in (
        "benchmark_root",
        "sweeps_root",
        "selected_root",
        "matrix_root",
        "reports_root",
        "overlays_dir",
    ):
        paths[key].mkdir(parents=True, exist_ok=True)


def _load_l1_config_from_selection(selection_path: Path) -> Dict[str, object]:
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    best = payload.get("best_config") if isinstance(payload, dict) else None
    if not isinstance(best, dict):
        raise ValueError(f"{selection_path} does not contain a 'best_config' object.")

    cfg: Dict[str, object] = dict(DEFAULT_L1_JAXILI)
    for key in cfg:
        if key in best:
            cfg[key] = best[key]

    return {
        "summary_transform": str(cfg["summary_transform"]),
        "clip_value": float(cfg["clip_value"]),
        "learning_rate": float(cfg["learning_rate"]),
        "epochs": int(cfg["epochs"]),
    }


def _run_phase(name: str, cmd: List[str], cwd: Path, dry_run: bool) -> None:
    print(f"[{name}] {_format_cmd(cmd)}")
    if dry_run:
        return
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode != 0:
        raise RuntimeError(f"Phase '{name}' failed with exit code {proc.returncode}.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Orchestrate a clean no-BNT cross-correlation benchmark for CNN vs L1 vs "
            "L1+VMIM across single-bin and tomo4."
        )
    )
    p.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    p.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)

    p.add_argument("--gpus", type=str, default="0,1,2")
    p.add_argument("--methods", type=str, default=DEFAULT_METHODS)
    p.add_argument("--variants", type=str, default=DEFAULT_VARIANTS)
    p.add_argument("--matrix-seeds", type=str, default="41,42,43")
    p.add_argument("--npe-samples", type=int, default=100000)
    p.add_argument("--conda-env", type=str, default="jaxili")

    p.add_argument("--sweep-seed41", type=int, default=41)
    p.add_argument("--sweep-robust-seeds", type=str, default="42,43")
    p.add_argument(
        "--selected-config-json",
        type=Path,
        default=None,
        help=(
            "Optional existing final_selection.json to source L1 settings from. "
            "If omitted, uses sweeps/l1_tomo4_opt_sweep/final_selection.json."
        ),
    )

    p.add_argument("--skip-sweep", action="store_true")
    p.add_argument("--skip-final-matrix", action="store_true")
    p.add_argument("--skip-analysis", action="store_true")
    p.add_argument("--skip-overlays", action="store_true")

    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Create benchmark directories and print planned commands without executing them.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    paths = _build_paths(args.benchmark_root.resolve())
    _ensure_dirs(paths)

    print("Benchmark root:")
    print(f"  {paths['benchmark_root']}")
    print("Created/ensured subdirectories:")
    for key in ("sweeps_root", "selected_root", "matrix_root", "reports_root"):
        print(f"  - {key.replace('_root', '')}: {paths[key]}")

    sweep_output_root = paths["sweep_run_root"]
    if args.selected_config_json is not None:
        selected_source = args.selected_config_json.resolve()
    else:
        selected_source = sweep_output_root / "final_selection.json"

    sweep_cmd = [
        "python",
        str(repo_root / "scripts" / "sbi" / "run_l1_jaxili_tomo4_opt_sweep.py"),
        "--repo-root",
        str(repo_root),
        "--output-root",
        str(sweep_output_root),
        "--gpus",
        args.gpus,
        "--seed41",
        str(args.sweep_seed41),
        "--robust-seeds",
        args.sweep_robust_seeds,
        "--npe-samples",
        str(args.npe_samples),
        "--conda-env",
        args.conda_env,
    ]

    l1_cfg: Dict[str, object] = dict(DEFAULT_L1_JAXILI)

    if not args.skip_sweep:
        _run_phase("phase1-sweep", sweep_cmd + (["--dry-run"] if args.dry_run else []), repo_root, args.dry_run)

    if selected_source.exists():
        l1_cfg = _load_l1_config_from_selection(selected_source)
        print(f"Using L1 config from: {selected_source}")
        print(
            "  L1 params: "
            f"summary_transform={l1_cfg['summary_transform']} "
            f"clip={l1_cfg['clip_value']} "
            f"lr={l1_cfg['learning_rate']} "
            f"epochs={l1_cfg['epochs']}"
        )
        if not args.dry_run and selected_source != paths["selected_config"]:
            shutil.copy2(selected_source, paths["selected_config"])
            print(f"Copied selected config to: {paths['selected_config']}")
    else:
        print(
            f"Selected config not found at {selected_source}. "
            "Using fallback L1 defaults for planning."
        )
        if not args.dry_run and not args.skip_final_matrix:
            raise FileNotFoundError(
                "Selected config is required for final matrix execution. "
                f"Missing: {selected_source}"
            )

    matrix_cmd = [
        "python",
        str(repo_root / "scripts" / "sbi" / "run_nobnt_tomo_bins_crosscorr_study.py"),
        "--repo-root",
        str(repo_root),
        "--output-root",
        str(paths["matrix_run_root"]),
        "--gpus",
        args.gpus,
        "--seeds",
        args.matrix_seeds,
        "--methods",
        args.methods,
        "--variants",
        args.variants,
        "--npe-samples",
        str(args.npe_samples),
        "--conda-env",
        args.conda_env,
        "--l1-estimator",
        "jaxili",
        "--l1-pca-components",
        "0",
        "--l1-summary-transform",
        str(l1_cfg["summary_transform"]),
        "--l1-clip-value",
        str(l1_cfg["clip_value"]),
        "--l1-learning-rate",
        str(l1_cfg["learning_rate"]),
        "--l1-epochs",
        str(l1_cfg["epochs"]),
    ]

    analysis_cmd = [
        "python",
        str(repo_root / "scripts" / "sbi" / "analyze_nobnt_tomo_bins_fom.py"),
        "--study-root",
        str(paths["matrix_run_root"]),
        "--methods",
        args.methods,
        "--variants",
        args.variants,
        "--seeds",
        args.matrix_seeds,
        "--output-json",
        str(paths["analysis_json"]),
        "--per-run-csv",
        str(paths["per_run_csv"]),
        "--summary-csv",
        str(paths["summary_csv"]),
    ]

    overlays_cmd = [
        "python",
        str(repo_root / "scripts" / "sbi" / "plot_tomo_bin_method_overlays.py"),
        "--study-root",
        str(paths["matrix_run_root"]),
        "--output-dir",
        str(paths["overlays_dir"]),
        "--variants",
        args.variants,
        "--seeds",
        args.matrix_seeds,
        "--trio",
        "--combined",
    ]

    print("Planned phases:")
    if args.skip_sweep:
        print("  - phase1-sweep: skipped")
    else:
        print(f"  - phase1-sweep: {_format_cmd(sweep_cmd + ['--dry-run']) if args.dry_run else _format_cmd(sweep_cmd)}")

    if args.skip_final_matrix:
        print("  - phase2-final-matrix: skipped")
    else:
        print(f"  - phase2-final-matrix: {_format_cmd(matrix_cmd + ['--dry-run']) if args.dry_run else _format_cmd(matrix_cmd)}")

    if args.skip_analysis:
        print("  - phase3-analysis: skipped")
    else:
        print(f"  - phase3-analysis: {_format_cmd(analysis_cmd)}")

    if args.skip_overlays:
        print("  - phase4-overlays: skipped")
    else:
        print(f"  - phase4-overlays: {_format_cmd(overlays_cmd)}")

    if args.dry_run:
        print("Dry-run mode enabled: commands were planned only; no heavy compute started.")
        return

    if not args.skip_final_matrix:
        _run_phase("phase2-final-matrix", matrix_cmd, repo_root, False)

    if not args.skip_analysis:
        _run_phase("phase3-analysis", analysis_cmd, repo_root, False)

    if not args.skip_overlays:
        _run_phase("phase4-overlays", overlays_cmd, repo_root, False)

    print("Benchmark orchestration complete.")


if __name__ == "__main__":
    main()
