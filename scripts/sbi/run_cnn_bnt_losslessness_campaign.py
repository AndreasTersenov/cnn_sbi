#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


DEFAULT_REPO_ROOT = str(Path(__file__).resolve().parents[2])
DEFAULT_OUTPUT_ROOT = (
    Path(DEFAULT_REPO_ROOT)
    / "scripts"
    / "sbi"
    / "results"
    / "final"
    / "paper_sbi_consolidation"
    / "cnn_bnt_losslessness_campaign"
)
DEFAULT_BASELINE_ROOT = (
    Path(DEFAULT_REPO_ROOT)
    / "scripts"
    / "sbi"
    / "results"
    / "final"
    / "paper_sbi_consolidation"
    / "bnt_comparison_tomo4"
)
TRUTH = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493], dtype=np.float64)
PARAM_NAMES = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]


@dataclass(frozen=True)
class Job:
    name: str
    command: List[str]
    log_path: Path


@dataclass(frozen=True)
class CampaignConfig:
    name: str
    seeds: Tuple[int, ...]
    compressor_steps: int
    compressor_conv_channels: str
    compressor_dense_width: int
    compressor_pool_window: int
    compressor_pool_stride: int
    standardize_summary: bool
    summary_clip_value: float
    flow_steps: int
    nvp_layers: int
    nvp_hidden: int
    batch_size: int
    patience: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Reproduce and optimize CNN BNT losslessness with matched BNT/noBNT "
            "compressor retraining and contour-level diagnostics."
        )
    )
    p.add_argument("--repo-root", type=str, default=DEFAULT_REPO_ROOT)
    p.add_argument("--gpus", type=str, default="0,1,2,3")
    p.add_argument("--map-kind", type=str, default="nbody")
    p.add_argument("--tfds-name", type=str, default="NbodyCosmogridDatasetTomo/grid_20deg_160px")
    p.add_argument(
        "--compressor-train-split",
        type=str,
        default="train",
        help="TFDS split used to train compressor checkpoints.",
    )
    p.add_argument(
        "--compressor-val-split",
        type=str,
        default="test",
        help="TFDS split used for compressor validation/test loss.",
    )
    p.add_argument(
        "--nde-train-split",
        type=str,
        default="train",
        help="TFDS split used to build NDE training summaries.",
    )
    p.add_argument(
        "--nde-val-split",
        type=str,
        default="test",
        help="TFDS split used to build NDE validation summaries.",
    )
    p.add_argument("--field-size", type=int, default=20)
    p.add_argument("--field-npix", type=int, default=160)
    p.add_argument("--nbins", type=int, default=4)
    p.add_argument("--tomo-bin-indices", type=str, default="1,2,3,4")
    p.add_argument("--npe-samples", type=int, default=100_000)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--compressor-save-every", type=int, default=2000)
    p.add_argument("--compressor-dim", type=int, default=6)
    p.add_argument("--compressor-batch-size", type=int, default=128)
    p.add_argument("--compressor-lr", type=float, default=5e-4)
    p.add_argument("--ds-batch-size", type=int, default=500)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--run-names",
        type=str,
        default="stagej_repro,advanced_arch96_nostd,advanced_arch64_dense256_nostd",
        help="Comma-separated subset of configs to run.",
    )
    return p.parse_args()


def _csv_tokens(value: str) -> List[str]:
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def _default_configs() -> Dict[str, CampaignConfig]:
    return {
        "stagej_repro": CampaignConfig(
            name="stagej_repro",
            seeds=(41, 42, 43, 44, 45),
            compressor_steps=60000,
            compressor_conv_channels="64,128,256",
            compressor_dense_width=128,
            compressor_pool_window=16,
            compressor_pool_stride=8,
            standardize_summary=False,
            summary_clip_value=5.0,
            flow_steps=5000,
            nvp_layers=4,
            nvp_hidden=128,
            batch_size=256,
            patience=30,
        ),
        "advanced_arch96_nostd": CampaignConfig(
            name="advanced_arch96_nostd",
            seeds=(41, 42, 43),
            compressor_steps=70000,
            compressor_conv_channels="96,192,384",
            compressor_dense_width=192,
            compressor_pool_window=16,
            compressor_pool_stride=8,
            standardize_summary=False,
            summary_clip_value=5.0,
            flow_steps=7000,
            nvp_layers=6,
            nvp_hidden=192,
            batch_size=256,
            patience=35,
        ),
        "advanced_arch64_dense256_nostd": CampaignConfig(
            name="advanced_arch64_dense256_nostd",
            seeds=(41, 42, 43, 44, 45),
            compressor_steps=80000,
            compressor_conv_channels="64,128,256",
            compressor_dense_width=256,
            compressor_pool_window=16,
            compressor_pool_stride=8,
            standardize_summary=False,
            summary_clip_value=5.0,
            flow_steps=8000,
            nvp_layers=8,
            nvp_hidden=256,
            batch_size=256,
            patience=40,
        ),
        "advanced_arch64_dense256_nostd_long": CampaignConfig(
            name="advanced_arch64_dense256_nostd_long",
            seeds=(41, 42, 43, 44, 45),
            compressor_steps=120000,
            compressor_conv_channels="64,128,256",
            compressor_dense_width=256,
            compressor_pool_window=16,
            compressor_pool_stride=8,
            standardize_summary=False,
            summary_clip_value=5.0,
            flow_steps=10000,
            nvp_layers=8,
            nvp_hidden=256,
            batch_size=256,
            patience=50,
        ),
    }


def run_jobs_parallel(
    jobs: List[Job], gpus: List[str], cwd: Path, dry_run: bool = False
) -> List[Dict[str, object]]:
    q: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        q.put(job)

    results: List[Dict[str, object]] = []
    lock = threading.Lock()

    def worker(gpu_id: str) -> None:
        while True:
            try:
                job = q.get_nowait()
            except queue.Empty:
                break

            job.log_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = list(job.command) + ["--cuda-visible-devices", gpu_id]
            t0 = time.time()
            if dry_run:
                rc = 0
                job.log_path.write_text(f"[dry-run] {' '.join(cmd)}\n", encoding="utf-8")
            else:
                with open(job.log_path, "w", encoding="utf-8") as logf:
                    proc = subprocess.run(
                        cmd,
                        cwd=str(cwd),
                        stdout=logf,
                        stderr=subprocess.STDOUT,
                    )
                    rc = proc.returncode
            dt = time.time() - t0
            with lock:
                results.append(
                    {
                        "name": job.name,
                        "gpu": gpu_id,
                        "returncode": int(rc),
                        "seconds": float(dt),
                        "log": str(job.log_path),
                        "cmd": cmd,
                    }
                )
            q.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def ensure_tfds_prepared(
    tfds_name: str,
    repo_root: Path,
    log_path: Path,
    dry_run: bool = False,
) -> None:
    """Prepare TFDS dataset once to avoid parallel builder races in worker jobs."""
    if dry_run:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"[dry-run] prepare tfds: {tfds_name}\n",
            encoding="utf-8",
        )
        return
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "jaxili",
        "python",
        "-c",
        (
            "import tensorflow_datasets as tfds; "
            "import scripts.sbi.tf_dataset_nbody_tomo as _; "
            f"b=tfds.builder('{tfds_name}'); "
            "b.download_and_prepare(); "
            "print('TFDS prepared:', b.name, b.builder_config.name, b.version)"
        ),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            "TFDS preparation failed for "
            f"{tfds_name}. See log: {log_path}"
        )


def require_success(results: List[Dict[str, object]], context: str) -> None:
    failed = [r for r in results if int(r.get("returncode", 1)) != 0]
    if failed:
        first = failed[0]
        raise RuntimeError(
            f"{context} failed for {len(failed)}/{len(results)} jobs. "
            f"First failure: {first.get('name')} (log: {first.get('log')})"
        )


def _compressor_paths(config_root: Path, condition: str, compressor_steps: int) -> Dict[str, Path]:
    base = (
        config_root
        / "compressor"
        / condition
        / "vmim"
        / "nbody"
        / "sigma_0.26"
        / "gal_density_30"
        / "bin_4"
    )
    return {
        "params": base / f"params_nd_compressor_batch{compressor_steps}.pkl",
        "state": base / f"opt_state_resnet_batch{compressor_steps}.pkl",
    }


def _posterior_path(config_root: Path, config_name: str, condition: str, seed: int) -> Path:
    return (
        config_root
        / "posteriors"
        / f"cnn_tomo4_20deg160_{condition}_{config_name}_s{seed}.npy"
    )


def _posterior_path_baseline(baseline_root: Path, condition: str, seed: int) -> Path:
    return baseline_root / "posteriors" / f"cnn_tomo4_20deg160_{condition}_s{seed}.npy"


def _fom3(samples: np.ndarray) -> Tuple[float, float, float, bool]:
    cov3 = np.cov(samples[:, :3], rowvar=False)
    sign, logdet = np.linalg.slogdet(cov3)
    if sign <= 0:
        return float("nan"), float("nan"), float(logdet), False
    det_cov3 = float(np.exp(logdet))
    return float(np.exp(-0.5 * logdet)), det_cov3, float(logdet), True


def _metrics_for_paths(paths_by_cond: Dict[str, List[Path]]) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for cond, paths in paths_by_cond.items():
        for path in paths:
            samples = np.load(path)
            std_sum = float(np.sum(np.std(samples, axis=0)))
            bias_l2 = float(np.linalg.norm(np.mean(samples, axis=0) - TRUTH))
            fom3, det_cov3, logdet_cov3, valid_fom3 = _fom3(samples)
            seed = int(path.stem.split("_s")[-1])
            rows.append(
                {
                    "condition": cond,
                    "seed": seed,
                    "file": str(path),
                    "std_sum": std_sum,
                    "bias_l2": bias_l2,
                    "fom3": fom3,
                    "det_cov3": det_cov3,
                    "logdet_cov3": logdet_cov3,
                    "valid_fom3": bool(valid_fom3),
                }
            )

    def _agg(cond: str) -> Dict[str, object]:
        vals = [r for r in rows if r["condition"] == cond]
        stds = np.array([float(v["std_sum"]) for v in vals], dtype=np.float64)
        biases = np.array([float(v["bias_l2"]) for v in vals], dtype=np.float64)
        foms = np.array([float(v["fom3"]) for v in vals], dtype=np.float64)
        valid_foms = foms[np.isfinite(foms)]
        return {
            "n": int(len(vals)),
            "std_sum_mean": float(np.mean(stds)) if len(stds) else float("nan"),
            "std_sum_std": float(np.std(stds, ddof=1)) if len(stds) > 1 else 0.0,
            "bias_l2_mean": float(np.mean(biases)) if len(biases) else float("nan"),
            "bias_l2_std": float(np.std(biases, ddof=1)) if len(biases) > 1 else 0.0,
            "fom3_mean": float(np.mean(valid_foms)) if len(valid_foms) else float("nan"),
            "fom3_std": (
                float(np.std(valid_foms, ddof=1)) if len(valid_foms) > 1 else 0.0
            ),
            "valid_fom3_count": int(len(valid_foms)),
        }

    nobnt = _agg("nobnt")
    bnt = _agg("bnt")
    std_inflation = (
        float(bnt["std_sum_mean"] / nobnt["std_sum_mean"])
        if np.isfinite(float(nobnt["std_sum_mean"])) and float(nobnt["std_sum_mean"]) != 0.0
        else float("nan")
    )
    fom_ratio = (
        float(bnt["fom3_mean"] / nobnt["fom3_mean"])
        if np.isfinite(float(nobnt["fom3_mean"])) and float(nobnt["fom3_mean"]) != 0.0
        else float("nan")
    )
    return {
        "rows": rows,
        "nobnt": nobnt,
        "bnt": bnt,
        "inflation_std_sum_bnt_over_nobnt": std_inflation,
        "fom3_ratio_bnt_over_nobnt": fom_ratio,
    }


def _rank_score(metric_block: Dict[str, object]) -> float:
    infl = float(metric_block.get("inflation_std_sum_bnt_over_nobnt", float("nan")))
    fomr = float(metric_block.get("fom3_ratio_bnt_over_nobnt", float("nan")))
    if not np.isfinite(infl) or not np.isfinite(fomr):
        return float("inf")
    return float(abs(infl - 1.0) + abs(fomr - 1.0))


def _import_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from getdist import MCSamples, plots as gplot

    return plt, MCSamples, gplot


def _plot_overlay(
    out_path: Path,
    samples_nobnt: np.ndarray,
    samples_bnt: np.ndarray,
    title: str,
    dpi: int,
) -> None:
    plt, MCSamples, gplot = _import_plotting()
    chain_nobnt = MCSamples(
        samples=samples_nobnt,
        names=PARAM_NAMES,
        labels=PARAM_NAMES,
        label="no-BNT",
        settings={"smooth_scale_2D": 0.7, "smooth_scale_1D": 0.7},
    )
    chain_bnt = MCSamples(
        samples=samples_bnt,
        names=PARAM_NAMES,
        labels=PARAM_NAMES,
        label="BNT",
        settings={"smooth_scale_2D": 0.7, "smooth_scale_1D": 0.7},
    )
    g = gplot.get_subplot_plotter(subplot_size=1.4)
    g.settings.alpha_filled_add = 0.55
    g.triangle_plot(
        [chain_nobnt, chain_bnt],
        filled=True,
        contour_colors=["#1f77b4", "#d62728"],
        markers=TRUTH,
        marker_args={"color": "black", "lw": 1.0},
        legend_loc="upper right",
    )
    g.fig.suptitle(title, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(g.fig)


def _concat(paths: Iterable[Path]) -> np.ndarray:
    return np.concatenate([np.load(p) for p in paths], axis=0)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    out_root = args.output_root.resolve()
    baseline_root = args.baseline_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    gpus = _csv_tokens(args.gpus)
    if not gpus:
        raise ValueError("--gpus cannot be empty.")

    configs_by_name = _default_configs()
    requested = _csv_tokens(args.run_names)
    if not requested:
        raise ValueError("--run-names cannot be empty.")
    unknown = [name for name in requested if name not in configs_by_name]
    if unknown:
        raise ValueError(f"Unknown run names: {unknown}.")
    configs = [configs_by_name[name] for name in requested]

    cnn_script = str(repo_root / "scripts" / "sbi" / "npe_cnn_nbody_tomo.py")

    manifest = {
        "repo_root": str(repo_root),
        "output_root": str(out_root),
        "baseline_root": str(baseline_root),
        "gpus": gpus,
        "tfds_name": args.tfds_name,
        "compressor_train_split": args.compressor_train_split,
        "compressor_val_split": args.compressor_val_split,
        "nde_train_split": args.nde_train_split,
        "nde_val_split": args.nde_val_split,
        "field_size": int(args.field_size),
        "field_npix": int(args.field_npix),
        "nbins": int(args.nbins),
        "tomo_bin_indices": str(args.tomo_bin_indices),
        "npe_samples": int(args.npe_samples),
        "compressor_dim": int(args.compressor_dim),
        "configs": [
            {
                "name": c.name,
                "seeds": list(c.seeds),
                "compressor_steps": c.compressor_steps,
                "compressor_conv_channels": c.compressor_conv_channels,
                "compressor_dense_width": c.compressor_dense_width,
                "compressor_pool_window": c.compressor_pool_window,
                "compressor_pool_stride": c.compressor_pool_stride,
                "standardize_summary": c.standardize_summary,
                "summary_clip_value": c.summary_clip_value,
                "flow_steps": c.flow_steps,
                "nvp_layers": c.nvp_layers,
                "nvp_hidden": c.nvp_hidden,
                "batch_size": c.batch_size,
                "patience": c.patience,
            }
            for c in configs
        ],
    }
    (out_root / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    ensure_tfds_prepared(
        args.tfds_name,
        repo_root=repo_root,
        log_path=out_root / "logs" / "tfds_prepare.log",
        dry_run=args.dry_run,
    )

    all_job_results: List[Dict[str, object]] = []
    per_config_metrics: List[Dict[str, object]] = []

    # Baseline reference from existing final-paper run.
    baseline_seeds = (41, 42, 43)
    baseline_paths = {
        "nobnt": [_posterior_path_baseline(baseline_root, "nobnt", s) for s in baseline_seeds],
        "bnt": [_posterior_path_baseline(baseline_root, "bnt", s) for s in baseline_seeds],
    }
    for cond in ("nobnt", "bnt"):
        for p in baseline_paths[cond]:
            if not p.exists():
                raise FileNotFoundError(f"Missing baseline posterior: {p}")
    baseline_metrics = _metrics_for_paths(baseline_paths)
    baseline_metrics["name"] = "baseline_final_paper"
    baseline_metrics["seeds"] = list(baseline_seeds)
    (out_root / "baseline_metrics.json").write_text(
        json.dumps(baseline_metrics, indent=2), encoding="utf-8"
    )
    if args.plot:
        _plot_overlay(
            out_root / "figures" / "overlay_baseline_finalpaper_combined_bnt_vs_nobnt.png",
            _concat(baseline_paths["nobnt"]),
            _concat(baseline_paths["bnt"]),
            "CNN baseline (final paper run): BNT vs no-BNT (combined seeds 41-43)",
            args.dpi,
        )

    for config in configs:
        cfg_root = out_root / config.name
        (cfg_root / "logs").mkdir(parents=True, exist_ok=True)
        (cfg_root / "posteriors").mkdir(parents=True, exist_ok=True)
        (cfg_root / "figures").mkdir(parents=True, exist_ok=True)

        train_jobs: List[Job] = []
        for cond in ("nobnt", "bnt"):
            comp_paths = _compressor_paths(cfg_root, cond, config.compressor_steps)
            if comp_paths["params"].exists() and comp_paths["state"].exists():
                continue
            cond_flag = ["--apply-bnt"] if cond == "bnt" else []
            train_jobs.append(
                Job(
                    name=f"train::{config.name}::{cond}",
                    log_path=cfg_root / "logs" / f"train_{cond}.log",
                    command=[
                        "conda",
                        "run",
                        "--no-capture-output",
                        "-n",
                        "jaxili",
                        "python",
                        cnn_script,
                        "--no-wandb",
                        "--map-kind",
                        args.map_kind,
                        "--tfds-name",
                        args.tfds_name,
                        "--compressor-train-split",
                        args.compressor_train_split,
                        "--compressor-val-split",
                        args.compressor_val_split,
                        "--nde-train-split",
                        args.nde_train_split,
                        "--nde-val-split",
                        args.nde_val_split,
                        "--field-size",
                        str(args.field_size),
                        "--field-npix",
                        str(args.field_npix),
                        "--nbins",
                        str(args.nbins),
                        "--tomo-bin-indices",
                        args.tomo_bin_indices,
                        "--cache-dir",
                        str(cfg_root / "cache" / f"{cond}_train"),
                        "--save-dir",
                        str(cfg_root / "compressor" / cond),
                        "--train-compressor",
                        "--compressor-dim",
                        str(args.compressor_dim),
                        "--compressor-conv-channels",
                        config.compressor_conv_channels,
                        "--compressor-dense-width",
                        str(config.compressor_dense_width),
                        "--compressor-pool-window",
                        str(config.compressor_pool_window),
                        "--compressor-pool-stride",
                        str(config.compressor_pool_stride),
                        "--compressor-steps",
                        str(config.compressor_steps),
                        "--compressor-save-every",
                        str(args.compressor_save_every),
                        "--compressor-batch-size",
                        str(args.compressor_batch_size),
                        "--compressor-lr",
                        str(args.compressor_lr),
                        "--total-steps",
                        "1",
                        "--save-every",
                        "1",
                        "--no-sample",
                    ]
                    + cond_flag,
                )
            )

        if train_jobs:
            train_results = run_jobs_parallel(train_jobs, gpus, repo_root, args.dry_run)
            all_job_results.extend(train_results)
            require_success(train_results, f"Compressor training ({config.name})")

        eval_jobs: List[Job] = []
        for cond in ("nobnt", "bnt"):
            comp_paths = _compressor_paths(cfg_root, cond, config.compressor_steps)
            if not comp_paths["params"].exists() or not comp_paths["state"].exists():
                raise FileNotFoundError(
                    f"Missing compressor checkpoint for {config.name}/{cond}: {comp_paths}"
                )
            for seed in config.seeds:
                posterior_out = _posterior_path(cfg_root, config.name, cond, seed)
                if posterior_out.exists():
                    continue
                cond_flag = ["--apply-bnt"] if cond == "bnt" else []
                std_flag = (
                    ["--standardize-summary"]
                    if config.standardize_summary
                    else ["--no-standardize-summary"]
                )
                cmd = [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "jaxili",
                    "python",
                    cnn_script,
                    "--no-wandb",
                    "--map-kind",
                    args.map_kind,
                    "--seed",
                    str(seed),
                    "--tfds-name",
                    args.tfds_name,
                    "--compressor-train-split",
                    args.compressor_train_split,
                    "--compressor-val-split",
                    args.compressor_val_split,
                    "--nde-train-split",
                    args.nde_train_split,
                    "--nde-val-split",
                    args.nde_val_split,
                    "--field-size",
                    str(args.field_size),
                    "--field-npix",
                    str(args.field_npix),
                    "--nbins",
                    str(args.nbins),
                    "--tomo-bin-indices",
                    args.tomo_bin_indices,
                    "--cache-dir",
                    str(cfg_root / "cache" / f"{cond}_eval"),
                    "--save-dir",
                    str(cfg_root / "eval" / cond / f"seed_{seed}"),
                    "--compressor-dim",
                    str(args.compressor_dim),
                    "--compressor-conv-channels",
                    config.compressor_conv_channels,
                    "--compressor-dense-width",
                    str(config.compressor_dense_width),
                    "--compressor-pool-window",
                    str(config.compressor_pool_window),
                    "--compressor-pool-stride",
                    str(config.compressor_pool_stride),
                    "--compressor-params",
                    str(comp_paths["params"]),
                    "--compressor-state",
                    str(comp_paths["state"]),
                    "--total-steps",
                    str(config.flow_steps),
                    "--save-every",
                    str(args.save_every),
                    "--patience",
                    str(config.patience),
                    "--batch-size",
                    str(config.batch_size),
                    "--nvp-layers",
                    str(config.nvp_layers),
                    "--nvp-hidden",
                    str(config.nvp_hidden),
                    "--summary-clip-value",
                    str(config.summary_clip_value),
                    "--npe-samples",
                    str(args.npe_samples),
                    "--posterior-out",
                    str(posterior_out),
                    "--ds-batch-size",
                    str(args.ds_batch_size),
                ] + std_flag + cond_flag
                if args.plot:
                    cmd.extend(
                        [
                            "--plot",
                            "--figure-out",
                            str(
                                cfg_root
                                / "figures"
                                / f"cnn_tomo4_20deg160_{cond}_{config.name}_s{seed}.png"
                            ),
                        ]
                    )
                eval_jobs.append(
                    Job(
                        name=f"eval::{config.name}::{cond}::s{seed}",
                        log_path=cfg_root / "logs" / f"eval_{cond}_s{seed}.log",
                        command=cmd,
                    )
                )

        if eval_jobs:
            eval_results = run_jobs_parallel(eval_jobs, gpus, repo_root, args.dry_run)
            all_job_results.extend(eval_results)
            require_success(eval_results, f"Posterior evaluation ({config.name})")

        paths = {
            "nobnt": [
                _posterior_path(cfg_root, config.name, "nobnt", seed) for seed in config.seeds
            ],
            "bnt": [
                _posterior_path(cfg_root, config.name, "bnt", seed) for seed in config.seeds
            ],
        }
        for cond in ("nobnt", "bnt"):
            for p in paths[cond]:
                if not p.exists():
                    raise FileNotFoundError(f"Missing posterior for {config.name}/{cond}: {p}")

        metric_block = _metrics_for_paths(paths)
        metric_block["name"] = config.name
        metric_block["seeds"] = list(config.seeds)
        metric_block["config"] = {
            "compressor_steps": config.compressor_steps,
            "compressor_conv_channels": config.compressor_conv_channels,
            "compressor_dense_width": config.compressor_dense_width,
            "compressor_pool_window": config.compressor_pool_window,
            "compressor_pool_stride": config.compressor_pool_stride,
            "standardize_summary": config.standardize_summary,
            "summary_clip_value": config.summary_clip_value,
            "flow_steps": config.flow_steps,
            "nvp_layers": config.nvp_layers,
            "nvp_hidden": config.nvp_hidden,
            "batch_size": config.batch_size,
            "patience": config.patience,
        }
        metric_block["rank_score"] = _rank_score(metric_block)
        per_config_metrics.append(metric_block)
        (cfg_root / "metrics.json").write_text(json.dumps(metric_block, indent=2), encoding="utf-8")

        if args.plot:
            _plot_overlay(
                cfg_root / "figures" / f"overlay_{config.name}_combined_bnt_vs_nobnt.png",
                _concat(paths["nobnt"]),
                _concat(paths["bnt"]),
                f"CNN {config.name}: BNT vs no-BNT (combined seeds)",
                args.dpi,
            )

    ranked = sorted(
        per_config_metrics,
        key=lambda row: (
            float(row.get("rank_score", float("inf"))),
            -float(row.get("nobnt", {}).get("fom3_mean", float("-inf"))),
        ),
    )

    summary = {
        "baseline": baseline_metrics,
        "configs": per_config_metrics,
        "ranked_configs": [
            {
                "name": row["name"],
                "rank_score": row["rank_score"],
                "inflation_std_sum_bnt_over_nobnt": row["inflation_std_sum_bnt_over_nobnt"],
                "fom3_ratio_bnt_over_nobnt": row["fom3_ratio_bnt_over_nobnt"],
                "nobnt_fom3_mean": row["nobnt"]["fom3_mean"],
                "bnt_fom3_mean": row["bnt"]["fom3_mean"],
                "seeds": row["seeds"],
            }
            for row in ranked
        ],
        "best_config": ranked[0]["name"] if ranked else None,
    }
    (out_root / "campaign_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_root / "job_results.json").write_text(json.dumps(all_job_results, indent=2), encoding="utf-8")

    print(f"Campaign complete. Outputs in: {out_root}")
    print(f"Best config: {summary['best_config']}")


if __name__ == "__main__":
    main()
