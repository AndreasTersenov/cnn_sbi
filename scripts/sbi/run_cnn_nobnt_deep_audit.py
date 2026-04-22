#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_REPO_ROOT
    / "scripts"
    / "sbi"
    / "results"
    / "final"
    / "paper_sbi_consolidation"
    / "cnn_nobnt_deep_audit"
)


@dataclass(frozen=True)
class Job:
    name: str
    command: List[str]
    log_path: Path


@dataclass(frozen=True)
class CompressorProfile:
    name: str
    train_split: str
    val_split: str
    steps: int
    conv_channels: str
    dense_width: int
    pool_window: int
    pool_stride: int


@dataclass(frozen=True)
class AuditRun:
    name: str
    seeds: Tuple[int, ...]
    compressor_profile: str
    npe_train_split: str
    npe_val_split: str
    shuffle_theta_train: bool
    epochs: int
    batch_size: int
    learning_rate: float
    standardize_summary: bool
    summary_clip_value: float


def _csv_tokens(value: str) -> List[str]:
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def _csv_ints(value: str) -> Tuple[int, ...]:
    return tuple(int(tok) for tok in _csv_tokens(value))


def _sanitize_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def _default_compressor_profiles() -> Dict[str, CompressorProfile]:
    return {
        "fulltrain": CompressorProfile(
            name="fulltrain",
            train_split="train",
            val_split="test",
            steps=60000,
            conv_channels="64,128,256",
            dense_width=128,
            pool_window=16,
            pool_stride=8,
        ),
        "split70": CompressorProfile(
            name="split70",
            train_split="train[:70%]",
            val_split="test",
            steps=60000,
            conv_channels="64,128,256",
            dense_width=128,
            pool_window=16,
            pool_stride=8,
        ),
    }


def _default_runs(seeds: Tuple[int, ...]) -> Dict[str, AuditRun]:
    return {
        "baseline_fulltrain": AuditRun(
            name="baseline_fulltrain",
            seeds=seeds,
            compressor_profile="fulltrain",
            npe_train_split="train",
            npe_val_split="test",
            shuffle_theta_train=False,
            epochs=5000,
            batch_size=256,
            learning_rate=1e-4,
            standardize_summary=False,
            summary_clip_value=0.0,
        ),
        "baseline_fulltrain_shuffle": AuditRun(
            name="baseline_fulltrain_shuffle",
            seeds=seeds,
            compressor_profile="fulltrain",
            npe_train_split="train",
            npe_val_split="test",
            shuffle_theta_train=True,
            epochs=5000,
            batch_size=256,
            learning_rate=1e-4,
            standardize_summary=False,
            summary_clip_value=0.0,
        ),
        "split70_disjoint": AuditRun(
            name="split70_disjoint",
            seeds=seeds,
            compressor_profile="split70",
            npe_train_split="train[70%:]",
            npe_val_split="test",
            shuffle_theta_train=False,
            epochs=5000,
            batch_size=256,
            learning_rate=1e-4,
            standardize_summary=False,
            summary_clip_value=0.0,
        ),
        "split70_disjoint_shuffle": AuditRun(
            name="split70_disjoint_shuffle",
            seeds=seeds,
            compressor_profile="split70",
            npe_train_split="train[70%:]",
            npe_val_split="test",
            shuffle_theta_train=True,
            epochs=5000,
            batch_size=256,
            learning_rate=1e-4,
            standardize_summary=False,
            summary_clip_value=0.0,
        ),
        "split70_small_nde10": AuditRun(
            name="split70_small_nde10",
            seeds=seeds,
            compressor_profile="split70",
            npe_train_split="train[90%:]",
            npe_val_split="test",
            shuffle_theta_train=False,
            epochs=5000,
            batch_size=256,
            learning_rate=1e-4,
            standardize_summary=False,
            summary_clip_value=0.0,
        ),
        "split70_long12000": AuditRun(
            name="split70_long12000",
            seeds=seeds,
            compressor_profile="split70",
            npe_train_split="train[70%:]",
            npe_val_split="test",
            shuffle_theta_train=False,
            epochs=12000,
            batch_size=256,
            learning_rate=1e-4,
            standardize_summary=False,
            summary_clip_value=0.0,
        ),
    }


def run_jobs_parallel(
    jobs: List[Job],
    gpus: List[str],
    cwd: Path,
    xla_mem_fraction: float,
    dry_run: bool = False,
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
                job.log_path.write_text(
                    "[dry-run] " + " ".join(cmd) + "\n",
                    encoding="utf-8",
                )
            else:
                env = dict(os.environ)
                env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(xla_mem_fraction)
                with open(job.log_path, "w", encoding="utf-8") as logf:
                    proc = subprocess.run(
                        cmd,
                        cwd=str(cwd),
                        env=env,
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
    conda_env: str,
    repo_root: Path,
    log_path: Path,
    dry_run: bool = False,
) -> None:
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
        conda_env,
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
            f"TFDS preparation failed for {tfds_name}. See log: {log_path}"
        )


def require_success(results: List[Dict[str, object]], context: str) -> None:
    failed = [r for r in results if int(r.get("returncode", 1)) != 0]
    if failed:
        first = failed[0]
        raise RuntimeError(
            f"{context} failed for {len(failed)}/{len(results)} jobs. "
            f"First failure: {first.get('name')} (log: {first.get('log')})"
        )


def _compressor_paths(output_root: Path, profile: CompressorProfile) -> Dict[str, Path]:
    base = (
        output_root
        / "compressors"
        / profile.name
        / "vmim"
        / "nbody"
        / "sigma_0.26"
        / "gal_density_30"
        / "bin_4"
    )
    return {
        "params": base / f"params_nd_compressor_batch{profile.steps}.pkl",
        "state": base / f"opt_state_resnet_batch{profile.steps}.pkl",
    }


def _posterior_path(output_root: Path, run_name: str, seed: int) -> Path:
    return output_root / "posteriors" / f"cnn_tomo4_20deg160_nobnt_{run_name}_s{seed}.npy"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run no-BNT CNN deep-audit controls (split/shuffle/data-starved/long-train) "
            "for suspiciously tight posterior investigation."
        )
    )
    p.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--conda-env", type=str, default="jaxili")
    p.add_argument("--gpus", type=str, default="0,1,2,3")
    p.add_argument("--xla-mem-fraction", type=float, default=0.5)
    p.add_argument("--map-kind", type=str, default="nbody")
    p.add_argument("--tfds-name", type=str, default="NbodyCosmogridDatasetTomo/grid_20deg_160px")
    p.add_argument("--field-size", type=int, default=20)
    p.add_argument("--field-npix", type=int, default=160)
    p.add_argument("--nbins", type=int, default=4)
    p.add_argument("--tomo-bin-indices", type=str, default="1,2,3,4")
    p.add_argument("--npe-samples", type=int, default=100_000)
    p.add_argument("--compressor-dim", type=int, default=6)
    p.add_argument("--compressor-save-every", type=int, default=2000)
    p.add_argument("--compressor-batch-size", type=int, default=128)
    p.add_argument("--compressor-lr", type=float, default=5e-4)
    p.add_argument("--ds-batch-size", type=int, default=500)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--seeds", type=str, default="41,42,43")
    p.add_argument(
        "--run-names",
        type=str,
        default=(
            "baseline_fulltrain,baseline_fulltrain_shuffle,"
            "split70_disjoint,split70_disjoint_shuffle,"
            "split70_small_nde10,split70_long12000"
        ),
        help="Comma-separated subset of audit runs to execute.",
    )
    p.add_argument("--skip-analysis", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    gpus = _csv_tokens(args.gpus)
    if not gpus:
        raise ValueError("--gpus cannot be empty.")
    if args.xla_mem_fraction <= 0.0 or args.xla_mem_fraction > 1.0:
        raise ValueError("--xla-mem-fraction must be in (0, 1].")
    seeds = _csv_ints(args.seeds)
    if not seeds:
        raise ValueError("--seeds cannot be empty.")

    compressor_profiles = _default_compressor_profiles()
    runs_by_name = _default_runs(seeds=seeds)
    requested_run_names = _csv_tokens(args.run_names)
    if not requested_run_names:
        raise ValueError("--run-names cannot be empty.")
    unknown = [name for name in requested_run_names if name not in runs_by_name]
    if unknown:
        raise ValueError(f"Unknown run names: {unknown}")
    runs = [runs_by_name[name] for name in requested_run_names]

    audit_script = repo_root / "scripts" / "sbi" / "audit_cnn_nobnt_data_pipeline.py"
    train_script = repo_root / "scripts" / "sbi" / "npe_cnn_nbody_tomo.py"
    eval_script = repo_root / "scripts" / "sbi" / "npe_cnn_jaxili_nbody_tomo.py"
    analysis_script = repo_root / "scripts" / "sbi" / "analyze_cnn_nobnt_deep_audit.py"

    manifest = {
        "repo_root": str(repo_root),
        "output_root": str(output_root),
        "conda_env": args.conda_env,
        "gpus": gpus,
        "xla_mem_fraction": float(args.xla_mem_fraction),
        "map_kind": args.map_kind,
        "tfds_name": args.tfds_name,
        "field_size": int(args.field_size),
        "field_npix": int(args.field_npix),
        "nbins": int(args.nbins),
        "tomo_bin_indices": args.tomo_bin_indices,
        "npe_samples": int(args.npe_samples),
        "compressor_dim": int(args.compressor_dim),
        "compressor_profiles": {
            name: {
                "train_split": prof.train_split,
                "val_split": prof.val_split,
                "steps": prof.steps,
                "conv_channels": prof.conv_channels,
                "dense_width": prof.dense_width,
                "pool_window": prof.pool_window,
                "pool_stride": prof.pool_stride,
            }
            for name, prof in compressor_profiles.items()
        },
        "runs": [
            {
                "name": run.name,
                "seeds": list(run.seeds),
                "compressor_profile": run.compressor_profile,
                "npe_train_split": run.npe_train_split,
                "npe_val_split": run.npe_val_split,
                "shuffle_theta_train": run.shuffle_theta_train,
                "epochs": run.epochs,
                "batch_size": run.batch_size,
                "learning_rate": run.learning_rate,
                "standardize_summary": run.standardize_summary,
                "summary_clip_value": run.summary_clip_value,
            }
            for run in runs
        ],
    }
    (output_root / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    ensure_tfds_prepared(
        tfds_name=args.tfds_name,
        conda_env=args.conda_env,
        repo_root=repo_root,
        log_path=output_root / "logs" / "tfds_prepare.log",
        dry_run=args.dry_run,
    )

    static_audit_cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "python",
        str(audit_script),
        "--tfds-name",
        args.tfds_name,
        "--train-split",
        "train",
        "--val-split",
        "test",
        "--split-a",
        "train[:70%]",
        "--split-b",
        "train[70%:]",
        "--map-kind",
        args.map_kind,
        "--field-size",
        str(args.field_size),
        "--field-npix",
        str(args.field_npix),
        "--sigma-e",
        "0.26",
        "--galaxy-density",
        str(30 / 4),
        "--nbins",
        str(args.nbins),
        "--tomo-bin-indices",
        args.tomo_bin_indices,
        "--output-json",
        str(output_root / "data_pipeline_audit.json"),
    ]
    static_log = output_root / "logs" / "data_pipeline_audit.log"
    if args.dry_run:
        static_log.parent.mkdir(parents=True, exist_ok=True)
        static_log.write_text(
            "[dry-run] " + " ".join(static_audit_cmd) + "\n",
            encoding="utf-8",
        )
    else:
        with open(static_log, "w", encoding="utf-8") as logf:
            proc = subprocess.run(
                static_audit_cmd,
                cwd=str(repo_root),
                stdout=logf,
                stderr=subprocess.STDOUT,
            )
        if proc.returncode != 0:
            raise RuntimeError(
                "Static data-pipeline audit failed. "
                f"See log: {static_log}"
            )

    all_job_results: List[Dict[str, object]] = []

    needed_profiles = sorted({run.compressor_profile for run in runs})
    train_jobs: List[Job] = []
    for profile_name in needed_profiles:
        profile = compressor_profiles[profile_name]
        comp_paths = _compressor_paths(output_root, profile)
        if comp_paths["params"].exists() and comp_paths["state"].exists():
            continue
        cmd = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            args.conda_env,
            "python",
            str(train_script),
            "--no-wandb",
            "--map-kind",
            args.map_kind,
            "--tfds-name",
            args.tfds_name,
            "--compressor-train-split",
            profile.train_split,
            "--compressor-val-split",
            profile.val_split,
            "--nde-train-split",
            profile.train_split,
            "--nde-val-split",
            profile.val_split,
            "--field-size",
            str(args.field_size),
            "--field-npix",
            str(args.field_npix),
            "--nbins",
            str(args.nbins),
            "--tomo-bin-indices",
            args.tomo_bin_indices,
            "--cache-dir",
            str(output_root / "cache" / f"compressor_{profile.name}"),
            "--save-dir",
            str(output_root / "compressors" / profile.name),
            "--train-compressor",
            "--compressor-dim",
            str(args.compressor_dim),
            "--compressor-conv-channels",
            profile.conv_channels,
            "--compressor-dense-width",
            str(profile.dense_width),
            "--compressor-pool-window",
            str(profile.pool_window),
            "--compressor-pool-stride",
            str(profile.pool_stride),
            "--compressor-steps",
            str(profile.steps),
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
            "--no-standardize-summary",
        ]
        train_jobs.append(
            Job(
                name=f"train_compressor::{profile.name}",
                command=cmd,
                log_path=output_root / "logs" / f"train_compressor_{profile.name}.log",
            )
        )

    if train_jobs:
        train_results = run_jobs_parallel(
            train_jobs,
            gpus=gpus,
            cwd=repo_root,
            xla_mem_fraction=args.xla_mem_fraction,
            dry_run=args.dry_run,
        )
        all_job_results.extend(train_results)
        require_success(train_results, "Compressor training")

    eval_jobs: List[Job] = []
    for run in runs:
        profile = compressor_profiles[run.compressor_profile]
        comp_paths = _compressor_paths(output_root, profile)
        if (
            not args.dry_run
            and (
                not comp_paths["params"].exists()
                or not comp_paths["state"].exists()
            )
        ):
            raise FileNotFoundError(
                f"Missing compressor checkpoint for profile={profile.name}: {comp_paths}"
            )
        split_tag = (
            f"{_sanitize_token(run.npe_train_split)}__"
            f"{_sanitize_token(run.npe_val_split)}"
        )
        cache_key = f"{profile.name}__{split_tag}"
        for seed in run.seeds:
            posterior_out = _posterior_path(output_root, run.name, seed)
            if posterior_out.exists():
                continue
            cmd = [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                args.conda_env,
                "python",
                str(eval_script),
                "--no-wandb",
                "--map-kind",
                args.map_kind,
                "--seed",
                str(seed),
                "--tfds-name",
                args.tfds_name,
                "--npe-train-split",
                run.npe_train_split,
                "--npe-val-split",
                run.npe_val_split,
                "--field-size",
                str(args.field_size),
                "--field-npix",
                str(args.field_npix),
                "--nbins",
                str(args.nbins),
                "--tomo-bin-indices",
                args.tomo_bin_indices,
                "--cache-dir",
                str(output_root / "cache" / f"eval_{cache_key}"),
                "--save-dir",
                str(output_root / "eval" / run.name / f"seed_{seed}"),
                "--compressor-dim",
                str(args.compressor_dim),
                "--compressor-conv-channels",
                profile.conv_channels,
                "--compressor-dense-width",
                str(profile.dense_width),
                "--compressor-pool-window",
                str(profile.pool_window),
                "--compressor-pool-stride",
                str(profile.pool_stride),
                "--compressor-params",
                str(comp_paths["params"]),
                "--compressor-state",
                str(comp_paths["state"]),
                "--epochs",
                str(run.epochs),
                "--batch-size",
                str(run.batch_size),
                "--learning-rate",
                str(run.learning_rate),
                "--npe-samples",
                str(args.npe_samples),
                "--posterior-out",
                str(posterior_out),
                "--ds-batch-size",
                str(args.ds_batch_size),
                "--summary-clip-value",
                str(run.summary_clip_value),
            ]
            if run.standardize_summary:
                cmd.append("--standardize-summary")
            else:
                cmd.append("--no-standardize-summary")
            if run.shuffle_theta_train:
                cmd.append("--shuffle-theta-train")
            if args.plot:
                cmd.extend(
                    [
                        "--plot",
                        "--figure-out",
                        str(output_root / "figures" / f"{run.name}_s{seed}.pdf"),
                    ]
                )
            eval_jobs.append(
                Job(
                    name=f"eval::{run.name}::s{seed}",
                    command=cmd,
                    log_path=output_root / "logs" / f"eval_{run.name}_s{seed}.log",
                )
            )

    if eval_jobs:
        eval_results = run_jobs_parallel(
            eval_jobs,
            gpus=gpus,
            cwd=repo_root,
            xla_mem_fraction=args.xla_mem_fraction,
            dry_run=args.dry_run,
        )
        all_job_results.extend(eval_results)
        require_success(eval_results, "No-BNT deep-audit runs")

    (output_root / "job_results.json").write_text(
        json.dumps(all_job_results, indent=2),
        encoding="utf-8",
    )

    if not args.skip_analysis:
        analysis_cmd = [
            "python",
            str(analysis_script),
            "--campaign-root",
            str(output_root),
            "--report-out",
            str(output_root / "CNN_NOBNT_DEEP_AUDIT_REPORT.md"),
        ]
        if args.dry_run:
            analysis_log = output_root / "logs" / "analysis.log"
            analysis_log.parent.mkdir(parents=True, exist_ok=True)
            analysis_log.write_text(
                "[dry-run] " + " ".join(analysis_cmd) + "\n",
                encoding="utf-8",
            )
        else:
            analysis_log = output_root / "logs" / "analysis.log"
            with open(analysis_log, "w", encoding="utf-8") as logf:
                proc = subprocess.run(
                    analysis_cmd,
                    cwd=str(repo_root),
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                )
            if proc.returncode != 0:
                raise RuntimeError(
                    "Analysis/report step failed. "
                    f"See log: {analysis_log}"
                )

    print(f"No-BNT deep audit complete. Outputs: {output_root}")


if __name__ == "__main__":
    main()
