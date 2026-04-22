#!/usr/bin/env python3
"""Zero-mean-maps parity campaign driver.

Re-runs the two best BNT/no-BNT reference configs (resnet18 "long15k_nostd6k_l8h256"
and advanced plain "arch64_dense256_nostd_long") with `--zero-mean-maps` enabled, so
we can compare posteriors against the originals and test whether the CNN compressor
was exploiting unphysical per-channel map mean (mass-sheet-degeneracy leak).

Mirrors the GPU-pool / FIFO-queue pattern from
`scripts/sbi/run_cnn_noise_curriculum_campaign.py` (per-GPU XLA memory fractions via
`--xla-mem-fraction-by-gpu`).
"""
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_REPO_ROOT = str(Path(__file__).resolve().parents[5])
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Job:
    name: str
    command: List[str]
    log_path: Path


@dataclass(frozen=True)
class CampaignConfig:
    name: str
    label: str
    out_subdir: str
    seeds: Tuple[int, ...]
    compressor_arch: str
    compressor_dim: int
    compressor_steps: int
    compressor_conv_channels: str
    compressor_dense_width: int
    compressor_pool_window: int
    compressor_pool_stride: int
    resnet_small_channels: str
    resnet_small_blocks: str
    resnet_head_width: int
    resnet_v2: bool
    standardize_summary: bool
    summary_clip_value: float
    flow_steps: int
    nvp_layers: int
    nvp_hidden: int
    batch_size: int
    patience: int
    compressor_train_split: str
    compressor_val_split: str
    nde_train_split: str
    nde_val_split: str
    require_disjoint_train_examples: bool


def _csv_tokens(value: str) -> List[str]:
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def _parse_gpu_mem_fraction_map(spec: str, gpus: List[str]) -> Dict[str, float]:
    if not spec.strip():
        return {}
    valid = set(gpus)
    parsed: Dict[str, float] = {}
    for token in _csv_tokens(spec):
        if ":" not in token:
            raise ValueError(
                "--xla-mem-fraction-by-gpu entries must be '<gpu>:<fraction>'."
            )
        gpu_id, frac_str = token.split(":", 1)
        gpu_id = gpu_id.strip()
        if gpu_id not in valid:
            raise ValueError(
                f"GPU '{gpu_id}' in --xla-mem-fraction-by-gpu is not listed in --gpus."
            )
        frac = float(frac_str)
        if not (0.0 < frac <= 1.0):
            raise ValueError(
                f"Invalid memory fraction {frac} for GPU '{gpu_id}'."
            )
        parsed[gpu_id] = frac
    return parsed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Zero-mean-maps parity campaign: rerun resnet18 and plain reference "
            "configs with --zero-mean-maps ON so posteriors can be compared "
            "against the originals."
        )
    )
    p.add_argument("--repo-root", type=str, default=DEFAULT_REPO_ROOT)
    p.add_argument("--conda-env", type=str, default="jaxili")
    p.add_argument("--gpus", type=str, default="0,1,2")
    p.add_argument("--xla-mem-fraction", type=float, default=0.45)
    p.add_argument(
        "--xla-mem-fraction-by-gpu",
        type=str,
        default="0:0.45,1:0.45,2:0.65",
    )
    p.add_argument(
        "--tfds-name",
        type=str,
        default="NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48",
    )
    p.add_argument("--map-kind", type=str, default="nbody")
    p.add_argument("--field-size", type=int, default=20)
    p.add_argument("--field-npix", type=int, default=160)
    p.add_argument("--nbins", type=int, default=4)
    p.add_argument("--tomo-bin-indices", type=str, default="1,2,3,4")
    p.add_argument("--npe-samples", type=int, default=100_000)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--compressor-save-every", type=int, default=3000)
    p.add_argument("--compressor-batch-size", type=int, default=128)
    p.add_argument("--compressor-lr", type=float, default=5e-4)
    p.add_argument("--ds-batch-size", type=int, default=500)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--run-names",
        type=str,
        default="run_a_resnet18,run_b_advanced_plain",
        help="Comma-separated subset of configs to run.",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Pass --plot to per-run flow inference (per-seed corner figure).",
    )
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def _default_configs() -> Dict[str, CampaignConfig]:
    run_a = CampaignConfig(
        name="run_a_resnet18",
        label="resnet18_long15k_nostd6k_l8h256_zm",
        out_subdir="run_a_resnet18",
        seeds=(41, 42, 43),
        compressor_arch="resnet18",
        compressor_dim=6,
        compressor_steps=15000,
        compressor_conv_channels="64,128,256",
        compressor_dense_width=256,
        compressor_pool_window=16,
        compressor_pool_stride=8,
        resnet_small_channels="64,128,256",
        resnet_small_blocks="2,2,2",
        resnet_head_width=256,
        resnet_v2=False,
        standardize_summary=False,
        summary_clip_value=5.0,
        flow_steps=6000,
        nvp_layers=8,
        nvp_hidden=256,
        batch_size=256,
        patience=35,
        compressor_train_split="train[:70%]",
        compressor_val_split="test",
        nde_train_split="train[70%:]",
        nde_val_split="test",
        require_disjoint_train_examples=True,
    )
    run_b = CampaignConfig(
        name="run_b_advanced_plain",
        label="advanced_arch64_dense256_nostd_long_zm",
        out_subdir="run_b_advanced_plain",
        seeds=(41, 42, 43, 44, 45),
        compressor_arch="plain",
        compressor_dim=10,
        compressor_steps=120000,
        compressor_conv_channels="64,128,256",
        compressor_dense_width=256,
        compressor_pool_window=16,
        compressor_pool_stride=8,
        resnet_small_channels="64,128,256",
        resnet_small_blocks="2,2,2",
        resnet_head_width=256,
        resnet_v2=False,
        standardize_summary=False,
        summary_clip_value=5.0,
        flow_steps=10000,
        nvp_layers=8,
        nvp_hidden=256,
        batch_size=256,
        patience=50,
        compressor_train_split="train",
        compressor_val_split="test",
        nde_train_split="train",
        nde_val_split="test",
        require_disjoint_train_examples=False,
    )
    return {c.name: c for c in (run_a, run_b)}


def run_jobs_parallel(
    jobs: List[Job],
    gpus: List[str],
    cwd: Path,
    xla_mem_fraction: float,
    xla_mem_fraction_by_gpu: Dict[str, float] | None = None,
    dry_run: bool = False,
) -> List[Dict[str, object]]:
    q: "queue.Queue[Job]" = queue.Queue()
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
            mem_fraction = (
                xla_mem_fraction_by_gpu.get(gpu_id, xla_mem_fraction)
                if xla_mem_fraction_by_gpu
                else xla_mem_fraction
            )
            if dry_run:
                rc = 0
                job.log_path.write_text(
                    "[dry-run] " + " ".join(cmd) + "\n",
                    encoding="utf-8",
                )
            else:
                env = dict(os.environ)
                env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)
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
                        "xla_mem_fraction": float(mem_fraction),
                        "cmd": cmd,
                    }
                )
            q.task_done()

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
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
        log_path.write_text(f"[dry-run] prepare tfds: {tfds_name}\n", encoding="utf-8")
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
        proc = subprocess.run(cmd, cwd=str(repo_root), stdout=logf, stderr=subprocess.STDOUT)
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


def _compressor_paths(cfg_root: Path, condition: str, compressor_steps: int) -> Dict[str, Path]:
    base = (
        cfg_root
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


def _resolve_compressor_paths(
    cfg_root: Path, condition: str, compressor_steps: int
) -> Dict[str, object]:
    requested = _compressor_paths(cfg_root, condition, compressor_steps)
    params = requested["params"]
    state = requested["state"]
    if params.exists() and state.exists():
        return {
            "params": params,
            "state": state,
            "resolved_step": int(compressor_steps),
            "requested_step": int(compressor_steps),
            "used_fallback": False,
        }
    base = params.parent
    pattern = re.compile(r"params_nd_compressor_batch(\d+)\.pkl$")
    candidates: List[Tuple[int, Path, Path]] = []
    if base.exists():
        for p in base.glob("params_nd_compressor_batch*.pkl"):
            m = pattern.match(p.name)
            if m is None:
                continue
            step = int(m.group(1))
            s = base / f"opt_state_resnet_batch{step}.pkl"
            if not s.exists():
                continue
            candidates.append((step, p, s))
    if not candidates:
        return {
            "params": params,
            "state": state,
            "resolved_step": int(compressor_steps),
            "requested_step": int(compressor_steps),
            "used_fallback": False,
        }
    within = [row for row in candidates if row[0] <= compressor_steps]
    chosen = max(within if within else candidates, key=lambda row: row[0])
    return {
        "params": chosen[1],
        "state": chosen[2],
        "resolved_step": int(chosen[0]),
        "requested_step": int(compressor_steps),
        "used_fallback": int(chosen[0]) != int(compressor_steps),
    }


def _posterior_path(cfg_root: Path, label: str, condition: str, seed: int) -> Path:
    return (
        cfg_root
        / "posteriors"
        / f"cnn_tomo4_20deg160_{condition}_{label}_s{seed}.npy"
    )


def _build_train_job(
    *,
    args: argparse.Namespace,
    config: CampaignConfig,
    cfg_root: Path,
    cnn_script: str,
    condition: str,
) -> Job:
    cond_flag = ["--apply-bnt"] if condition == "bnt" else []
    cache_dir = cfg_root / "cache" / f"{condition}_zeromean_train"
    std_flag = (
        ["--standardize-summary"]
        if config.standardize_summary
        else ["--no-standardize-summary"]
    )
    disjoint_flag = (
        ["--require-disjoint-train-examples"]
        if config.require_disjoint_train_examples
        else []
    )
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "python",
        cnn_script,
        "--no-wandb",
        "--zero-mean-maps",
        "--map-kind",
        args.map_kind,
        "--tfds-name",
        args.tfds_name,
        "--compressor-train-split",
        config.compressor_train_split,
        "--compressor-val-split",
        config.compressor_val_split,
        "--nde-train-split",
        config.nde_train_split,
        "--nde-val-split",
        config.nde_val_split,
        "--field-size",
        str(args.field_size),
        "--field-npix",
        str(args.field_npix),
        "--nbins",
        str(args.nbins),
        "--tomo-bin-indices",
        args.tomo_bin_indices,
        "--cache-dir",
        str(cache_dir),
        "--save-dir",
        str(cfg_root / "compressor" / condition),
        "--train-compressor",
        "--compressor-dim",
        str(config.compressor_dim),
        "--compressor-arch",
        config.compressor_arch,
        "--compressor-conv-channels",
        config.compressor_conv_channels,
        "--compressor-dense-width",
        str(config.compressor_dense_width),
        "--compressor-pool-window",
        str(config.compressor_pool_window),
        "--compressor-pool-stride",
        str(config.compressor_pool_stride),
        "--resnet-small-channels",
        config.resnet_small_channels,
        "--resnet-small-blocks",
        config.resnet_small_blocks,
        "--resnet-head-width",
        str(config.resnet_head_width),
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
        "--summary-clip-value",
        str(config.summary_clip_value),
        "--ds-batch-size",
        str(args.ds_batch_size),
    ] + std_flag + (["--resnet-v2"] if config.resnet_v2 else []) + disjoint_flag + cond_flag
    return Job(
        name=f"train::{config.name}::{condition}",
        log_path=cfg_root / "logs" / f"train_{condition}.log",
        command=cmd,
    )


def _build_eval_job(
    *,
    args: argparse.Namespace,
    config: CampaignConfig,
    cfg_root: Path,
    cnn_script: str,
    condition: str,
    seed: int,
    comp_paths: Dict[str, object],
) -> Job:
    cond_flag = ["--apply-bnt"] if condition == "bnt" else []
    std_flag = (
        ["--standardize-summary"]
        if config.standardize_summary
        else ["--no-standardize-summary"]
    )
    disjoint_flag = (
        ["--require-disjoint-train-examples"]
        if config.require_disjoint_train_examples
        else []
    )
    posterior_out = _posterior_path(cfg_root, config.label, condition, seed)
    cache_dir = cfg_root / "cache" / f"{condition}_zeromean_eval"
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "python",
        cnn_script,
        "--no-wandb",
        "--zero-mean-maps",
        "--map-kind",
        args.map_kind,
        "--seed",
        str(seed),
        "--tfds-name",
        args.tfds_name,
        "--compressor-train-split",
        config.compressor_train_split,
        "--compressor-val-split",
        config.compressor_val_split,
        "--nde-train-split",
        config.nde_train_split,
        "--nde-val-split",
        config.nde_val_split,
        "--field-size",
        str(args.field_size),
        "--field-npix",
        str(args.field_npix),
        "--nbins",
        str(args.nbins),
        "--tomo-bin-indices",
        args.tomo_bin_indices,
        "--cache-dir",
        str(cache_dir),
        "--save-dir",
        str(cfg_root / "eval" / condition / f"seed_{seed}"),
        "--compressor-dim",
        str(config.compressor_dim),
        "--compressor-arch",
        config.compressor_arch,
        "--compressor-conv-channels",
        config.compressor_conv_channels,
        "--compressor-dense-width",
        str(config.compressor_dense_width),
        "--compressor-pool-window",
        str(config.compressor_pool_window),
        "--compressor-pool-stride",
        str(config.compressor_pool_stride),
        "--resnet-small-channels",
        config.resnet_small_channels,
        "--resnet-small-blocks",
        config.resnet_small_blocks,
        "--resnet-head-width",
        str(config.resnet_head_width),
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
    ] + std_flag + (["--resnet-v2"] if config.resnet_v2 else []) + disjoint_flag + cond_flag
    if args.plot:
        cmd.extend(
            [
                "--plot",
                "--figure-out",
                str(
                    cfg_root
                    / "figures"
                    / f"cnn_tomo4_20deg160_{condition}_{config.label}_s{seed}.png"
                ),
            ]
        )
    return Job(
        name=f"eval::{config.name}::{condition}::s{seed}",
        log_path=cfg_root / "logs" / f"eval_{condition}_s{seed}.log",
        command=cmd,
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    gpus = _csv_tokens(args.gpus)
    if not gpus:
        raise ValueError("--gpus cannot be empty.")
    if not (0.0 < args.xla_mem_fraction <= 1.0):
        raise ValueError("--xla-mem-fraction must be in (0, 1].")
    xla_mem_fraction_by_gpu = _parse_gpu_mem_fraction_map(
        args.xla_mem_fraction_by_gpu, gpus
    )

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
        "conda_env": args.conda_env,
        "gpus": gpus,
        "xla_mem_fraction": float(args.xla_mem_fraction),
        "xla_mem_fraction_by_gpu": xla_mem_fraction_by_gpu,
        "tfds_name": args.tfds_name,
        "npe_samples": int(args.npe_samples),
        "zero_mean_maps": True,
        "configs": [
            {
                "name": c.name,
                "label": c.label,
                "seeds": list(c.seeds),
                "compressor_arch": c.compressor_arch,
                "compressor_dim": c.compressor_dim,
                "compressor_steps": c.compressor_steps,
                "standardize_summary": c.standardize_summary,
                "summary_clip_value": c.summary_clip_value,
                "flow_steps": c.flow_steps,
                "nvp_layers": c.nvp_layers,
                "nvp_hidden": c.nvp_hidden,
                "compressor_train_split": c.compressor_train_split,
                "nde_train_split": c.nde_train_split,
                "require_disjoint_train_examples": c.require_disjoint_train_examples,
            }
            for c in configs
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_root / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    ensure_tfds_prepared(
        args.tfds_name,
        args.conda_env,
        repo_root=repo_root,
        log_path=out_root / "logs" / "tfds_prepare.log",
        dry_run=args.dry_run,
    )

    all_job_results: List[Dict[str, object]] = []

    # Phase 1: build a single queue of compressor trainings across both configs
    # so GPUs pick up work as soon as they are free (FIFO).
    train_jobs: List[Job] = []
    for config in configs:
        cfg_root = out_root / config.out_subdir
        (cfg_root / "logs").mkdir(parents=True, exist_ok=True)
        (cfg_root / "posteriors").mkdir(parents=True, exist_ok=True)
        (cfg_root / "figures").mkdir(parents=True, exist_ok=True)
        for cond in ("nobnt", "bnt"):
            comp_paths = _compressor_paths(cfg_root, cond, config.compressor_steps)
            if comp_paths["params"].exists() and comp_paths["state"].exists():
                continue
            train_jobs.append(
                _build_train_job(
                    args=args,
                    config=config,
                    cfg_root=cfg_root,
                    cnn_script=cnn_script,
                    condition=cond,
                )
            )

    if train_jobs:
        print(f"Phase 1: launching {len(train_jobs)} compressor training job(s).")
        train_results = run_jobs_parallel(
            train_jobs,
            gpus,
            repo_root,
            xla_mem_fraction=args.xla_mem_fraction,
            xla_mem_fraction_by_gpu=xla_mem_fraction_by_gpu,
            dry_run=args.dry_run,
        )
        all_job_results.extend(train_results)
        require_success(train_results, "Compressor training")

    # Phase 2: flow inference jobs, one per (config, condition, seed).
    eval_jobs: List[Job] = []
    resolved_compressor_steps: Dict[str, Dict[str, int]] = {}
    for config in configs:
        cfg_root = out_root / config.out_subdir
        resolved_compressor_steps[config.name] = {}
        for cond in ("nobnt", "bnt"):
            comp_paths = _resolve_compressor_paths(cfg_root, cond, config.compressor_steps)
            resolved_compressor_steps[config.name][cond] = int(
                comp_paths["resolved_step"]
            )
            if (
                not args.dry_run
                and (
                    not Path(comp_paths["params"]).exists()
                    or not Path(comp_paths["state"]).exists()
                )
            ):
                raise FileNotFoundError(
                    f"Missing compressor checkpoint for {config.name}/{cond}: {comp_paths}"
                )
            if bool(comp_paths["used_fallback"]):
                print(
                    f"[{config.name}/{cond}] requested compressor step "
                    f"{comp_paths['requested_step']} not found; using step "
                    f"{comp_paths['resolved_step']} checkpoint."
                )
            for seed in config.seeds:
                posterior_out = _posterior_path(cfg_root, config.label, cond, seed)
                if posterior_out.exists():
                    continue
                eval_jobs.append(
                    _build_eval_job(
                        args=args,
                        config=config,
                        cfg_root=cfg_root,
                        cnn_script=cnn_script,
                        condition=cond,
                        seed=seed,
                        comp_paths=comp_paths,
                    )
                )

    if eval_jobs:
        print(f"Phase 2: launching {len(eval_jobs)} flow-eval job(s).")
        eval_results = run_jobs_parallel(
            eval_jobs,
            gpus,
            repo_root,
            xla_mem_fraction=args.xla_mem_fraction,
            xla_mem_fraction_by_gpu=xla_mem_fraction_by_gpu,
            dry_run=args.dry_run,
        )
        all_job_results.extend(eval_results)
        require_success(eval_results, "Flow-eval")

    (out_root / "job_results.json").write_text(
        json.dumps(all_job_results, indent=2),
        encoding="utf-8",
    )
    (out_root / "resolved_compressor_steps.json").write_text(
        json.dumps(resolved_compressor_steps, indent=2),
        encoding="utf-8",
    )
    print(f"Done. Outputs in: {out_root}")


if __name__ == "__main__":
    main()
