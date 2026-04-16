#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


TRUTH = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493], dtype=np.float64)


@dataclass(frozen=True)
class Job:
    name: str
    command: List[str]
    log_path: Path


def _pythonpath_with_repo(repo_root: Path) -> str:
    scripts_sbi = str((repo_root / "scripts" / "sbi").resolve())
    existing = os.environ.get("PYTHONPATH", "")
    return scripts_sbi if not existing else f"{scripts_sbi}:{existing}"


def _run_cmd(
    cmd: List[str],
    cwd: Path,
    log_path: Path,
    xla_mem_fraction: float,
    repo_root: Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(xla_mem_fraction)
    env["PYTHONPATH"] = _pythonpath_with_repo(repo_root)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    return int(proc.returncode)


def run_jobs_parallel(
    jobs: List[Job],
    gpus: List[str],
    cwd: Path,
    xla_mem_fraction: float,
    repo_root: Path,
) -> List[Dict[str, object]]:
    q: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        q.put(job)

    lock = threading.Lock()
    results: List[Dict[str, object]] = []

    def worker(gpu_id: str) -> None:
        while True:
            try:
                job = q.get_nowait()
            except queue.Empty:
                return
            cmd = list(job.command) + ["--cuda-visible-devices", gpu_id]
            t0 = time.time()
            rc = _run_cmd(
                cmd=cmd,
                cwd=cwd,
                log_path=job.log_path,
                xla_mem_fraction=xla_mem_fraction,
                repo_root=repo_root,
            )
            dt = time.time() - t0
            with lock:
                results.append(
                    {
                        "name": job.name,
                        "gpu": gpu_id,
                        "returncode": rc,
                        "seconds": float(dt),
                        "log": str(job.log_path),
                        "cmd": cmd,
                    }
                )
            q.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def require_success(results: Iterable[Dict[str, object]], context: str) -> None:
    failed = [r for r in results if int(r.get("returncode", 1)) != 0]
    if failed:
        first = failed[0]
        raise RuntimeError(
            f"{context} failed for {len(failed)} job(s). "
            f"First failure: {first['name']} ({first['log']})"
        )


def ensure_tfds_prepared(
    tfds_name: str,
    conda_env: str,
    repo_root: Path,
    log_path: Path,
    xla_mem_fraction: float,
) -> None:
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        conda_env,
        "python",
        "-c",
        (
            "import tensorflow_datasets as tfds; "
            "import tf_dataset_nbody_tomo as _; "
            f"b=tfds.builder('{tfds_name}'); "
            "b.download_and_prepare(); "
            "print('TFDS prepared:', b.name, b.builder_config.name, b.version)"
        ),
    ]
    rc = _run_cmd(
        cmd=cmd,
        cwd=repo_root,
        log_path=log_path,
        xla_mem_fraction=xla_mem_fraction,
        repo_root=repo_root,
    )
    if rc != 0:
        raise RuntimeError(f"TFDS prepare failed. See {log_path}")


def _compressor_ckpt_dir(output_root: Path, condition: str) -> Path:
    return (
        output_root
        / "compressor"
        / condition
        / "vmim"
        / "nbody"
        / "sigma_0.26"
        / "gal_density_30"
        / "bin_4"
    )


def _compressor_ckpt_paths(output_root: Path, condition: str, step: int) -> Dict[str, Path]:
    base = _compressor_ckpt_dir(output_root, condition)
    return {
        "params": base / f"params_nd_compressor_batch{step}.pkl",
        "state": base / f"opt_state_resnet_batch{step}.pkl",
    }


def _fom3(samples: np.ndarray) -> float:
    cov3 = np.cov(samples[:, :3], rowvar=False)
    sign, logdet = np.linalg.slogdet(cov3)
    if sign <= 0:
        return float("nan")
    return float(np.exp(-0.5 * logdet))


def _variant_metrics(variant_root: Path, variant_name: str, seeds: List[int]) -> Dict[str, object]:
    per_cond: Dict[str, List[Dict[str, float]]] = {"nobnt": [], "bnt": []}
    for cond in ("nobnt", "bnt"):
        for seed in seeds:
            post_path = (
                variant_root
                / "posteriors"
                / f"cnn_tomo4_20deg160_{cond}_{variant_name}_s{seed}.npy"
            )
            if not post_path.exists():
                raise FileNotFoundError(f"Missing posterior file: {post_path}")
            samples = np.load(post_path)
            std_sum = float(np.sum(np.std(samples, axis=0)))
            bias_l2 = float(np.linalg.norm(np.mean(samples, axis=0) - TRUTH))
            fom3 = _fom3(samples)
            per_cond[cond].append(
                {
                    "seed": float(seed),
                    "std_sum": std_sum,
                    "bias_l2": bias_l2,
                    "fom3": fom3,
                }
            )

    def _agg(rows: List[Dict[str, float]]) -> Dict[str, float]:
        std_vals = np.array([row["std_sum"] for row in rows], dtype=np.float64)
        bias_vals = np.array([row["bias_l2"] for row in rows], dtype=np.float64)
        fom_vals = np.array([row["fom3"] for row in rows], dtype=np.float64)
        valid_fom = fom_vals[np.isfinite(fom_vals)]
        return {
            "n": float(len(rows)),
            "std_sum_mean": float(np.mean(std_vals)) if len(std_vals) else float("nan"),
            "std_sum_std": float(np.std(std_vals, ddof=1)) if len(std_vals) > 1 else 0.0,
            "bias_l2_mean": float(np.mean(bias_vals)) if len(bias_vals) else float("nan"),
            "bias_l2_std": float(np.std(bias_vals, ddof=1)) if len(bias_vals) > 1 else 0.0,
            "fom3_mean": float(np.mean(valid_fom)) if len(valid_fom) else float("nan"),
            "fom3_std": float(np.std(valid_fom, ddof=1)) if len(valid_fom) > 1 else 0.0,
            "valid_fom3_count": float(len(valid_fom)),
        }

    nobnt = _agg(per_cond["nobnt"])
    bnt = _agg(per_cond["bnt"])

    infl = (
        float(bnt["std_sum_mean"] / nobnt["std_sum_mean"])
        if np.isfinite(float(nobnt["std_sum_mean"])) and float(nobnt["std_sum_mean"]) != 0.0
        else float("nan")
    )
    fom_ratio = (
        float(bnt["fom3_mean"] / nobnt["fom3_mean"])
        if np.isfinite(float(nobnt["fom3_mean"])) and float(nobnt["fom3_mean"]) != 0.0
        else float("nan")
    )
    rank_score = (
        float(abs(infl - 1.0) + abs(fom_ratio - 1.0))
        if np.isfinite(infl) and np.isfinite(fom_ratio)
        else float("inf")
    )

    metrics = {
        "variant": variant_name,
        "seeds": list(seeds),
        "inflation_std_sum_bnt_over_nobnt": infl,
        "fom3_ratio_bnt_over_nobnt": fom_ratio,
        "nobnt_fom3_mean": float(nobnt["fom3_mean"]),
        "bnt_fom3_mean": float(bnt["fom3_mean"]),
        "rank_score": rank_score,
        "nobnt": nobnt,
        "bnt": bnt,
        "per_seed": {
            "nobnt": per_cond["nobnt"],
            "bnt": per_cond["bnt"],
        },
        "rank_score_formula": "abs(inflation_std_sum_bnt_over_nobnt-1)+abs(fom3_ratio_bnt_over_nobnt-1)",
    }
    return metrics


def main() -> None:
    repo_root = Path.cwd().resolve()
    output_root = (
        repo_root
        / "scripts"
        / "sbi"
        / "results"
        / "final"
        / "paper_sbi_consolidation"
        / "cnn_bnt_resnet_split_campaign"
        / "resnet_extended_tuning_v2"
        / "resnet50"
    )
    snapshot_script = output_root / "tools" / "npe_cnn_nbody_tomo_snapshot.py"
    if not snapshot_script.exists():
        raise FileNotFoundError(f"Snapshot script missing: {snapshot_script}")

    conda_env = "jaxili"
    gpus = ["0", "1"]
    xla_mem_fraction = 0.5
    seeds = [41, 42, 43]
    compressor_steps = 30000

    variants = [
        {
            "name": "r50_long30k_nostd6k_l8h256",
            "total_steps": 6000,
            "nvp_layers": 8,
            "nvp_hidden": 256,
            "standardize": False,
        },
        {
            "name": "r50_long30k_std10k_l10h320",
            "total_steps": 10000,
            "nvp_layers": 10,
            "nvp_hidden": 320,
            "standardize": True,
        },
        {
            "name": "r50_long30k_nostd12k_l12h384",
            "total_steps": 12000,
            "nvp_layers": 12,
            "nvp_hidden": 384,
            "standardize": False,
        },
    ]

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)

    manifest = {
        "repo_root": str(repo_root),
        "output_root": str(output_root),
        "snapshot_script": str(snapshot_script),
        "conda_env": conda_env,
        "gpus": gpus,
        "xla_python_client_mem_fraction": xla_mem_fraction,
        "tfds_name": "NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48",
        "compressor_train_split": "train[:70%]",
        "compressor_val_split": "test",
        "nde_train_split": "train[70%:]",
        "nde_val_split": "test",
        "require_disjoint_train_examples": True,
        "map_kind": "nbody",
        "field_size": 20,
        "field_npix": 160,
        "nbins": 4,
        "tomo_bin_indices": "1,2,3,4",
        "compressor_dim": 6,
        "compressor_arch": "resnet50",
        "resnet_head_width": 256,
        "batch_size": 256,
        "patience": 35,
        "save_every": 500,
        "summary_clip_value": 5.0,
        "npe_samples": 100000,
        "ds_batch_size": 500,
        "compressor_steps": compressor_steps,
        "compressor_save_every": 3000,
        "seeds": seeds,
        "conditions": ["nobnt", "bnt"],
        "variants": variants,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_root / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    ensure_tfds_prepared(
        tfds_name="NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48",
        conda_env=conda_env,
        repo_root=repo_root,
        log_path=output_root / "logs" / "tfds_prepare.log",
        xla_mem_fraction=xla_mem_fraction,
    )

    common_base = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        conda_env,
        "python",
        str(snapshot_script),
        "--no-wandb",
        "--plot",
        "--map-kind",
        "nbody",
        "--tfds-name",
        "NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48",
        "--compressor-train-split",
        "train[:70%]",
        "--compressor-val-split",
        "test",
        "--nde-train-split",
        "train[70%:]",
        "--nde-val-split",
        "test",
        "--require-disjoint-train-examples",
        "--field-size",
        "20",
        "--field-npix",
        "160",
        "--nbins",
        "4",
        "--tomo-bin-indices",
        "1,2,3,4",
        "--compressor-dim",
        "6",
        "--compressor-arch",
        "resnet50",
        "--resnet-head-width",
        "256",
        "--batch-size",
        "256",
        "--patience",
        "35",
        "--save-every",
        "500",
        "--summary-clip-value",
        "5.0",
        "--npe-samples",
        "100000",
        "--ds-batch-size",
        "500",
    ]

    all_results: List[Dict[str, object]] = []

    # Step A: train long compressors once per condition.
    train_jobs: List[Job] = []
    for cond in ("nobnt", "bnt"):
        ckpts = _compressor_ckpt_paths(output_root, cond, compressor_steps)
        if ckpts["params"].exists() and ckpts["state"].exists():
            continue
        cond_flag = ["--apply-bnt"] if cond == "bnt" else []
        cmd = (
            list(common_base)
            + [
                "--cache-dir",
                str(output_root / "cache" / "compressor" / cond),
                "--save-dir",
                str(output_root / "compressor" / cond),
                "--train-compressor",
                "--compressor-steps",
                str(compressor_steps),
                "--compressor-save-every",
                "3000",
                "--total-steps",
                "1",
                "--save-every",
                "1",
                "--no-sample",
            ]
            + cond_flag
        )
        train_jobs.append(
            Job(
                name=f"train_compressor::{cond}",
                command=cmd,
                log_path=output_root / "logs" / f"train_compressor_{cond}.log",
            )
        )

    if train_jobs:
        train_results = run_jobs_parallel(
            jobs=train_jobs,
            gpus=gpus,
            cwd=repo_root,
            xla_mem_fraction=xla_mem_fraction,
            repo_root=repo_root,
        )
        all_results.extend(train_results)
        require_success(train_results, "Step A compressor training")

    for cond in ("nobnt", "bnt"):
        ckpts = _compressor_ckpt_paths(output_root, cond, compressor_steps)
        if not ckpts["params"].exists() or not ckpts["state"].exists():
            raise FileNotFoundError(
                "Missing long-compressor checkpoint files for "
                f"{cond}: {ckpts['params']} and/or {ckpts['state']}"
            )

    # Step B: evaluate NDE variants using long compressors.
    eval_jobs: List[Job] = []
    for variant in variants:
        variant_name = variant["name"]
        variant_root = output_root / variant_name
        (variant_root / "logs").mkdir(parents=True, exist_ok=True)
        (variant_root / "posteriors").mkdir(parents=True, exist_ok=True)
        (variant_root / "figures").mkdir(parents=True, exist_ok=True)

        for cond in ("nobnt", "bnt"):
            ckpts = _compressor_ckpt_paths(output_root, cond, compressor_steps)
            cond_flag = ["--apply-bnt"] if cond == "bnt" else []
            std_flag = ["--standardize-summary"] if variant["standardize"] else ["--no-standardize-summary"]

            for seed in seeds:
                post_path = (
                    variant_root
                    / "posteriors"
                    / f"cnn_tomo4_20deg160_{cond}_{variant_name}_s{seed}.npy"
                )
                fig_path = (
                    variant_root
                    / "figures"
                    / f"cnn_tomo4_20deg160_{cond}_{variant_name}_s{seed}.png"
                )
                if post_path.exists() and fig_path.exists():
                    continue

                cmd = (
                    list(common_base)
                    + [
                        "--seed",
                        str(seed),
                        "--cache-dir",
                        str(output_root / "cache" / "eval" / variant_name / cond / f"seed_{seed}"),
                        "--save-dir",
                        str(variant_root / "eval" / cond / f"seed_{seed}"),
                        "--compressor-params",
                        str(ckpts["params"]),
                        "--compressor-state",
                        str(ckpts["state"]),
                        "--total-steps",
                        str(variant["total_steps"]),
                        "--save-every",
                        "500",
                        "--nvp-layers",
                        str(variant["nvp_layers"]),
                        "--nvp-hidden",
                        str(variant["nvp_hidden"]),
                        "--posterior-out",
                        str(post_path),
                        "--figure-out",
                        str(fig_path),
                    ]
                    + std_flag
                    + cond_flag
                )
                eval_jobs.append(
                    Job(
                        name=f"eval::{variant_name}::{cond}::s{seed}",
                        command=cmd,
                        log_path=variant_root / "logs" / f"eval_{cond}_s{seed}.log",
                    )
                )

    if eval_jobs:
        eval_results = run_jobs_parallel(
            jobs=eval_jobs,
            gpus=gpus,
            cwd=repo_root,
            xla_mem_fraction=xla_mem_fraction,
            repo_root=repo_root,
        )
        all_results.extend(eval_results)
        require_success(eval_results, "Step B NDE evaluation")

    # Validate required outputs, compute per-variant metrics.
    variant_metrics: List[Dict[str, object]] = []
    for variant in variants:
        variant_name = variant["name"]
        variant_root = output_root / variant_name
        for cond in ("nobnt", "bnt"):
            for seed in seeds:
                post_path = (
                    variant_root
                    / "posteriors"
                    / f"cnn_tomo4_20deg160_{cond}_{variant_name}_s{seed}.npy"
                )
                fig_path = (
                    variant_root
                    / "figures"
                    / f"cnn_tomo4_20deg160_{cond}_{variant_name}_s{seed}.png"
                )
                log_path = variant_root / "logs" / f"eval_{cond}_s{seed}.log"
                if not post_path.exists():
                    raise FileNotFoundError(f"Missing required posterior: {post_path}")
                if not fig_path.exists():
                    raise FileNotFoundError(f"Missing required figure: {fig_path}")
                if not log_path.exists():
                    raise FileNotFoundError(f"Missing required log: {log_path}")

        metrics = _variant_metrics(variant_root, variant_name, seeds)
        metrics["config"] = {
            "compressor_steps": compressor_steps,
            "nvp_layers": int(variant["nvp_layers"]),
            "nvp_hidden": int(variant["nvp_hidden"]),
            "total_steps": int(variant["total_steps"]),
            "standardize_summary": bool(variant["standardize"]),
            "compressor_arch": "resnet50",
            "resnet_head_width": 256,
            "batch_size": 256,
            "patience": 35,
            "save_every": 500,
            "summary_clip_value": 5.0,
            "npe_samples": 100000,
            "ds_batch_size": 500,
        }
        (variant_root / "metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )
        variant_metrics.append(metrics)

    ranked = sorted(
        variant_metrics,
        key=lambda row: (
            float(row.get("rank_score", math.inf)),
            -float(row.get("nobnt_fom3_mean", float("-inf"))),
        ),
    )

    summary_json = {
        "output_root": str(output_root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rank_score_formula": "abs(inflation_std_sum_bnt_over_nobnt-1)+abs(fom3_ratio_bnt_over_nobnt-1)",
        "variants": [
            {
                "variant": row["variant"],
                "inflation_std_sum_bnt_over_nobnt": row["inflation_std_sum_bnt_over_nobnt"],
                "fom3_ratio_bnt_over_nobnt": row["fom3_ratio_bnt_over_nobnt"],
                "nobnt_fom3_mean": row["nobnt_fom3_mean"],
                "bnt_fom3_mean": row["bnt_fom3_mean"],
                "rank_score": row["rank_score"],
                "config": row.get("config", {}),
                "metrics_file": str(output_root / row["variant"] / "metrics.json"),
            }
            for row in ranked
        ],
        "best_variant_by_rank": ranked[0]["variant"] if ranked else None,
    }
    (output_root / "resnet50_extended_summary.json").write_text(
        json.dumps(summary_json, indent=2),
        encoding="utf-8",
    )

    csv_path = output_root / "resnet50_extended_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank_position",
                "variant",
                "inflation_std_sum_bnt_over_nobnt",
                "fom3_ratio_bnt_over_nobnt",
                "nobnt_fom3_mean",
                "bnt_fom3_mean",
                "rank_score",
                "metrics_file",
            ]
        )
        for idx, row in enumerate(ranked, start=1):
            writer.writerow(
                [
                    idx,
                    row["variant"],
                    row["inflation_std_sum_bnt_over_nobnt"],
                    row["fom3_ratio_bnt_over_nobnt"],
                    row["nobnt_fom3_mean"],
                    row["bnt_fom3_mean"],
                    row["rank_score"],
                    str(output_root / row["variant"] / "metrics.json"),
                ]
            )

    (output_root / "job_results.json").write_text(
        json.dumps(all_results, indent=2),
        encoding="utf-8",
    )

    print("Campaign complete.")
    print(f"Output root: {output_root}")
    print(f"Best variant by rank: {summary_json['best_variant_by_rank']}")


if __name__ == "__main__":
    main()
