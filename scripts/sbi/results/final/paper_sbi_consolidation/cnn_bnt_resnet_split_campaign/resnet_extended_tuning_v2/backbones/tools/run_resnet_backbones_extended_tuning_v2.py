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
from typing import Dict, Iterable, List, Tuple

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
    env.setdefault("WANDB_PROJECT", "cnn-bnt-resnet")
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


def _compressor_ckpt_dir(output_root: Path, arch: str, condition: str) -> Path:
    return (
        output_root
        / arch
        / "compressor"
        / condition
        / "vmim"
        / "nbody"
        / "sigma_0.26"
        / "gal_density_30"
        / "bin_4"
    )


def _compressor_ckpt_paths(output_root: Path, arch: str, condition: str, step: int) -> Dict[str, Path]:
    base = _compressor_ckpt_dir(output_root, arch, condition)
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
            fom3 = _fom3(samples)
            per_cond[cond].append(
                {
                    "seed": float(seed),
                    "std_sum": std_sum,
                    "fom3": fom3,
                }
            )

    def _agg(rows: List[Dict[str, float]]) -> Dict[str, float]:
        std_vals = np.array([row["std_sum"] for row in rows], dtype=np.float64)
        fom_vals = np.array([row["fom3"] for row in rows], dtype=np.float64)
        valid_fom = fom_vals[np.isfinite(fom_vals)]
        return {
            "n": float(len(rows)),
            "std_sum_mean": float(np.mean(std_vals)) if len(std_vals) else float("nan"),
            "std_sum_std": float(np.std(std_vals, ddof=1)) if len(std_vals) > 1 else 0.0,
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

    return {
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


def _variants_for_arch(arch: str) -> List[Dict[str, object]]:
    return [
        {
            "name": f"{arch}_long15k_nostd6k_l8h256",
            "total_steps": 6000,
            "nvp_layers": 8,
            "nvp_hidden": 256,
            "standardize": False,
        },
        {
            "name": f"{arch}_long15k_std10k_l10h320",
            "total_steps": 10000,
            "nvp_layers": 10,
            "nvp_hidden": 320,
            "standardize": True,
        },
    ]


def _validate_variant_outputs(
    output_root: Path,
    arch: str,
    variant_name: str,
    seeds: List[int],
) -> None:
    variant_root = output_root / arch / variant_name
    for cond in ("nobnt", "bnt"):
        for seed in seeds:
            post_path = variant_root / "posteriors" / f"cnn_tomo4_20deg160_{cond}_{variant_name}_s{seed}.npy"
            fig_path = variant_root / "figures" / f"cnn_tomo4_20deg160_{cond}_{variant_name}_s{seed}.png"
            log_path = variant_root / "logs" / f"eval_{cond}_s{seed}.log"
            if not post_path.exists():
                raise FileNotFoundError(f"Missing required posterior: {post_path}")
            if not fig_path.exists():
                raise FileNotFoundError(f"Missing required figure: {fig_path}")
            if not log_path.exists():
                raise FileNotFoundError(f"Missing required log: {log_path}")


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
        / "backbones"
    )
    snapshot_script = output_root / "tools" / "npe_cnn_nbody_tomo_snapshot.py"
    if not snapshot_script.exists():
        raise FileNotFoundError(f"Snapshot script missing: {snapshot_script}")

    conda_env = "jaxili"
    gpus = ["2", "3"]
    xla_mem_fraction = 0.5
    seeds = [41, 42, 43]
    conditions = ["nobnt", "bnt"]
    arches = ["resnet18", "resnet34"]
    compressor_steps = 15000

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
        "resnet_head_width": 256,
        "batch_size": 256,
        "patience": 35,
        "save_every": 500,
        "summary_clip_value": 5.0,
        "npe_samples": 100000,
        "ds_batch_size": 500,
        "compressor_steps": compressor_steps,
        "compressor_lr": 5e-4,
        "compressor_batch_size": 128,
        "compressor_save_every": 3000,
        "seeds": seeds,
        "conditions": conditions,
        "arches": arches,
        "variants": {arch: _variants_for_arch(arch) for arch in arches},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_root / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

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
        "--wandb-project",
        "cnn-bnt-resnet",
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

    train_jobs: List[Job] = []
    for arch in arches:
        arch_root = output_root / arch
        (arch_root / "logs").mkdir(parents=True, exist_ok=True)
        for cond in conditions:
            ckpts = _compressor_ckpt_paths(output_root, arch, cond, compressor_steps)
            if ckpts["params"].exists() and ckpts["state"].exists():
                continue
            cond_flag = ["--apply-bnt"] if cond == "bnt" else []
            cmd = (
                list(common_base)
                + [
                    "--compressor-arch",
                    arch,
                    "--cache-dir",
                    str(output_root / "cache" / arch / "compressor" / cond),
                    "--save-dir",
                    str(output_root / arch / "compressor" / cond),
                    "--train-compressor",
                    "--compressor-steps",
                    str(compressor_steps),
                    "--compressor-save-every",
                    "3000",
                    "--compressor-batch-size",
                    "128",
                    "--compressor-lr",
                    "5e-4",
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
                    name=f"train_compressor::{arch}::{cond}",
                    command=cmd,
                    log_path=arch_root / "logs" / f"train_compressor_{cond}.log",
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
        require_success(train_results, "Step 1 compressor training")

    for arch in arches:
        for cond in conditions:
            ckpts = _compressor_ckpt_paths(output_root, arch, cond, compressor_steps)
            if not ckpts["params"].exists() or not ckpts["state"].exists():
                raise FileNotFoundError(
                    f"Missing long-compressor checkpoint files for {arch}/{cond}: "
                    f"{ckpts['params']} and/or {ckpts['state']}"
                )

    eval_jobs: List[Job] = []
    for arch in arches:
        variants = _variants_for_arch(arch)
        for variant in variants:
            variant_name = str(variant["name"])
            variant_root = output_root / arch / variant_name
            (variant_root / "logs").mkdir(parents=True, exist_ok=True)
            (variant_root / "posteriors").mkdir(parents=True, exist_ok=True)
            (variant_root / "figures").mkdir(parents=True, exist_ok=True)

            for cond in conditions:
                ckpts = _compressor_ckpt_paths(output_root, arch, cond, compressor_steps)
                cond_flag = ["--apply-bnt"] if cond == "bnt" else []
                std_flag = ["--standardize-summary"] if variant["standardize"] else ["--no-standardize-summary"]

                for seed in seeds:
                    post_path = variant_root / "posteriors" / f"cnn_tomo4_20deg160_{cond}_{variant_name}_s{seed}.npy"
                    fig_path = variant_root / "figures" / f"cnn_tomo4_20deg160_{cond}_{variant_name}_s{seed}.png"
                    log_path = variant_root / "logs" / f"eval_{cond}_s{seed}.log"
                    if post_path.exists() and fig_path.exists() and log_path.exists():
                        continue

                    cmd = (
                        list(common_base)
                        + [
                            "--compressor-arch",
                            arch,
                            "--seed",
                            str(seed),
                            "--cache-dir",
                            str(output_root / "cache" / arch / "eval" / variant_name / cond / f"seed_{seed}"),
                            "--save-dir",
                            str(variant_root / "eval" / cond / f"seed_{seed}"),
                            "--compressor-params",
                            str(ckpts["params"]),
                            "--compressor-state",
                            str(ckpts["state"]),
                            "--total-steps",
                            str(int(variant["total_steps"])),
                            "--save-every",
                            "500",
                            "--nvp-layers",
                            str(int(variant["nvp_layers"])),
                            "--nvp-hidden",
                            str(int(variant["nvp_hidden"])),
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
                            name=f"eval::{arch}::{variant_name}::{cond}::s{seed}",
                            command=cmd,
                            log_path=log_path,
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
        require_success(eval_results, "Step 2 NDE evaluation")

    per_arch_ranked: Dict[str, List[Dict[str, object]]] = {}
    all_rows: List[Dict[str, object]] = []

    for arch in arches:
        variants = _variants_for_arch(arch)
        rows: List[Dict[str, object]] = []
        for variant in variants:
            variant_name = str(variant["name"])
            _validate_variant_outputs(output_root=output_root, arch=arch, variant_name=variant_name, seeds=seeds)
            variant_root = output_root / arch / variant_name
            metrics = _variant_metrics(variant_root, variant_name, seeds)
            metrics["arch"] = arch
            metrics["config"] = {
                "compressor_arch": arch,
                "compressor_steps": compressor_steps,
                "compressor_lr": 5e-4,
                "compressor_batch_size": 128,
                "compressor_save_every": 3000,
                "resnet_head_width": 256,
                "batch_size": 256,
                "patience": 35,
                "save_every": 500,
                "summary_clip_value": 5.0,
                "npe_samples": 100000,
                "ds_batch_size": 500,
                "total_steps": int(variant["total_steps"]),
                "nvp_layers": int(variant["nvp_layers"]),
                "nvp_hidden": int(variant["nvp_hidden"]),
                "standardize_summary": bool(variant["standardize"]),
            }
            (variant_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            rows.append(metrics)
            all_rows.append(metrics)

        ranked = sorted(
            rows,
            key=lambda row: (
                float(row.get("rank_score", math.inf)),
                -float(row.get("nobnt_fom3_mean", float("-inf"))),
            ),
        )
        per_arch_ranked[arch] = ranked

    global_ranked = sorted(
        all_rows,
        key=lambda row: (
            float(row.get("rank_score", math.inf)),
            -float(row.get("nobnt_fom3_mean", float("-inf"))),
        ),
    )

    summary_json = {
        "output_root": str(output_root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rank_score_formula": "abs(inflation_std_sum_bnt_over_nobnt-1)+abs(fom3_ratio_bnt_over_nobnt-1)",
        "best_per_arch": {
            arch: (ranked[0]["variant"] if ranked else None)
            for arch, ranked in per_arch_ranked.items()
        },
        "arch_variants": {
            arch: [
                {
                    "rank_within_arch": idx,
                    "variant": row["variant"],
                    "inflation_std_sum_bnt_over_nobnt": row["inflation_std_sum_bnt_over_nobnt"],
                    "fom3_ratio_bnt_over_nobnt": row["fom3_ratio_bnt_over_nobnt"],
                    "nobnt_fom3_mean": row["nobnt_fom3_mean"],
                    "bnt_fom3_mean": row["bnt_fom3_mean"],
                    "rank_score": row["rank_score"],
                    "metrics_file": str(output_root / arch / row["variant"] / "metrics.json"),
                }
                for idx, row in enumerate(ranked, start=1)
            ]
            for arch, ranked in per_arch_ranked.items()
        },
        "global_ranking": [
            {
                "rank_global": idx,
                "arch": row["arch"],
                "variant": row["variant"],
                "inflation_std_sum_bnt_over_nobnt": row["inflation_std_sum_bnt_over_nobnt"],
                "fom3_ratio_bnt_over_nobnt": row["fom3_ratio_bnt_over_nobnt"],
                "nobnt_fom3_mean": row["nobnt_fom3_mean"],
                "bnt_fom3_mean": row["bnt_fom3_mean"],
                "rank_score": row["rank_score"],
                "metrics_file": str(output_root / row["arch"] / row["variant"] / "metrics.json"),
            }
            for idx, row in enumerate(global_ranked, start=1)
        ],
    }
    (output_root / "backbone_summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    with open(output_root / "backbone_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "arch",
                "rank_within_arch",
                "rank_global",
                "variant",
                "inflation_std_sum_bnt_over_nobnt",
                "fom3_ratio_bnt_over_nobnt",
                "nobnt_fom3_mean",
                "bnt_fom3_mean",
                "rank_score",
                "best_for_arch",
                "metrics_file",
            ]
        )

        global_rank: Dict[Tuple[str, str], int] = {
            (row["arch"], row["variant"]): idx
            for idx, row in enumerate(global_ranked, start=1)
        }
        for arch in arches:
            ranked = per_arch_ranked.get(arch, [])
            best_variant = ranked[0]["variant"] if ranked else None
            for idx, row in enumerate(ranked, start=1):
                writer.writerow(
                    [
                        arch,
                        idx,
                        global_rank.get((arch, row["variant"]), ""),
                        row["variant"],
                        row["inflation_std_sum_bnt_over_nobnt"],
                        row["fom3_ratio_bnt_over_nobnt"],
                        row["nobnt_fom3_mean"],
                        row["bnt_fom3_mean"],
                        row["rank_score"],
                        str(row["variant"] == best_variant).lower(),
                        str(output_root / arch / row["variant"] / "metrics.json"),
                    ]
                )

    (output_root / "job_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    print("Backbone campaign complete.")
    print(f"Output root: {output_root}")
    for arch in arches:
        best = summary_json["best_per_arch"].get(arch)
        print(f"Best variant ({arch}): {best}")


if __name__ == "__main__":
    main()
