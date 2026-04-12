#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_REPO_ROOT
    / "scripts"
    / "sbi"
    / "optimal_nobnt_crosscorr_benchmark"
    / "sweeps"
    / "cnn_tomo4"
)


@dataclass(frozen=True)
class ArchConfig:
    conv_channels: str
    dense_width: int

    @property
    def conv_tag(self) -> str:
        return self.conv_channels.replace(",", "x")

    @property
    def arch_id(self) -> str:
        return f"conv_{self.conv_tag}__dw_{self.dense_width}"


@dataclass(frozen=True)
class SweepConfig:
    arch: ArchConfig
    standardize_summary: bool
    flow_steps: int
    batch_size: int

    @property
    def config_id(self) -> str:
        std_tag = "on" if self.standardize_summary else "off"
        return (
            f"{self.arch.arch_id}"
            f"__std_{std_tag}"
            f"__flow_{self.flow_steps}"
            f"__bs_{self.batch_size}"
        )


@dataclass
class Job:
    name: str
    stage: str
    config_id: str
    seed: Optional[int]
    command: List[str]
    log_path: Path
    expected_output: Optional[Path]
    gpu: str


def _csv_tokens(value: str) -> List[str]:
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def _csv_ints(value: str) -> List[int]:
    return [int(tok) for tok in _csv_tokens(value)]


def _csv_bool_tokens(value: str) -> List[bool]:
    out: List[bool] = []
    for tok in _csv_tokens(value):
        low = tok.lower()
        if low in {"1", "true", "t", "yes", "y", "on"}:
            out.append(True)
        elif low in {"0", "false", "f", "no", "n", "off"}:
            out.append(False)
        else:
            raise ValueError(
                f"Invalid token '{tok}' in boolean list. Use values from "
                "{on,off,true,false,1,0}."
            )
    return out


def _parse_conv_channel_options(value: str) -> List[str]:
    options: set[Tuple[int, ...]] = set()
    for token in [tok.strip() for tok in value.split(";") if tok.strip()]:
        channels = tuple(_csv_ints(token))
        if not channels:
            raise ValueError("Each conv-channel option must contain at least one integer.")
        if any(c <= 0 for c in channels):
            raise ValueError(f"Invalid conv-channel option '{token}'. Values must be > 0.")
        options.add(channels)

    if not options:
        raise ValueError("--conv-channel-options cannot be empty.")

    return [",".join(str(v) for v in ch) for ch in sorted(options)]


def _format_cmd(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _log_has_traceback(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Traceback" in line:
                return True
    return False


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _acquire_output_lock(out_root: Path) -> Path:
    lock_path = out_root / ".run.lock"
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "pid": os.getpid(),
                        "time": time.time(),
                        "script": str(Path(__file__).resolve()),
                    },
                    f,
                )
            return lock_path
        except FileExistsError:
            stale = True
            holder_pid = -1
            try:
                payload = json.loads(lock_path.read_text(encoding="utf-8"))
                holder_pid = int(payload.get("pid", -1))
                stale = not _pid_is_alive(holder_pid)
            except Exception:
                stale = True

            if stale:
                lock_path.unlink(missing_ok=True)
                continue

            raise RuntimeError(
                "Output root is locked by another running sweep process "
                f"(pid={holder_pid}). Lock file: {lock_path}"
            )


def _release_output_lock(lock_path: Path) -> None:
    lock_path.unlink(missing_ok=True)


def _fom3_from_posterior(path: Path) -> Tuple[float, float, float, bool, int]:
    samples = np.load(path)
    if samples.ndim != 2 or samples.shape[1] < 3 or samples.shape[0] < 2:
        return float("nan"), float("nan"), float("nan"), False, int(samples.shape[0])

    cov3 = np.cov(samples[:, :3], rowvar=False)
    sign, logdet = np.linalg.slogdet(cov3)
    if sign <= 0:
        return float("nan"), float("nan"), float(logdet), False, int(samples.shape[0])

    det = float(np.exp(logdet))
    fom3 = float(np.exp(-0.5 * logdet))
    return fom3, det, float(logdet), True, int(samples.shape[0])


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _cnn_comp_paths(save_dir: Path, map_kind: str, nbins: int, steps: int) -> Dict[str, Path]:
    base = (
        save_dir
        / "vmim"
        / map_kind
        / "sigma_0.26"
        / "gal_density_30"
        / f"bin_{nbins}"
    )
    return {
        "params": base / f"params_nd_compressor_batch{steps}.pkl",
        "state": base / f"opt_state_resnet_batch{steps}.pkl",
        "summary": base / "compressor_training_summary.json",
    }


def _build_architectures(args: argparse.Namespace) -> List[ArchConfig]:
    conv_options = _parse_conv_channel_options(args.conv_channel_options)
    dense_widths = _csv_ints(args.dense_width_options)
    if not dense_widths or any(v <= 0 for v in dense_widths):
        raise ValueError("--dense-width-options must contain positive integers.")

    dense_widths = sorted(set(dense_widths))
    archs = [
        ArchConfig(conv_channels=conv, dense_width=dw)
        for conv, dw in product(conv_options, dense_widths)
    ]
    archs = sorted(archs, key=lambda a: (tuple(_csv_ints(a.conv_channels)), a.dense_width))
    return archs


def _build_configs(args: argparse.Namespace, archs: List[ArchConfig]) -> List[SweepConfig]:
    standardize_options = _csv_bool_tokens(args.standardize_options)
    flow_steps_options = _csv_ints(args.flow_steps_options)
    batch_size_options = _csv_ints(args.batch_size_options)

    if not standardize_options:
        raise ValueError("--standardize-options cannot be empty.")
    if not flow_steps_options or any(v <= 0 for v in flow_steps_options):
        raise ValueError("--flow-steps-options must contain positive integers.")
    if not batch_size_options or any(v <= 0 for v in batch_size_options):
        raise ValueError("--batch-size-options must contain positive integers.")

    flow_steps_values = sorted(set(flow_steps_options))
    batch_sizes_values = sorted(set(batch_size_options))
    conv_options = sorted(
        {arch.conv_channels for arch in archs},
        key=lambda value: tuple(_csv_ints(value)),
    )
    dense_options = sorted({arch.dense_width for arch in archs})
    arch_lookup = {(arch.conv_channels, arch.dense_width): arch for arch in archs}

    baseline_conv = "64,128,256" if "64,128,256" in conv_options else conv_options[0]
    baseline_dense = 128 if 128 in dense_options else dense_options[0]
    baseline_std = False if False in standardize_options else standardize_options[0]
    baseline_flow = 5000 if 5000 in flow_steps_values else flow_steps_values[0]
    baseline_batch = 256 if 256 in batch_sizes_values else batch_sizes_values[0]

    def _arch(conv: str, dense: int) -> ArchConfig:
        return arch_lookup.get((conv, dense), ArchConfig(conv_channels=conv, dense_width=dense))

    if args.grid_mode == "cartesian":
        cfgs = [
            SweepConfig(
                arch=arch,
                standardize_summary=std,
                flow_steps=flow_steps,
                batch_size=batch_size,
            )
            for arch, std, flow_steps, batch_size in product(
                archs,
                standardize_options,
                flow_steps_values,
                batch_sizes_values,
            )
        ]
    else:
        cfg_set: set[SweepConfig] = set()
        # Block A: architecture family + summary-standardization scan.
        cfg_set.update(
            SweepConfig(
                arch=_arch(conv, baseline_dense),
                standardize_summary=std,
                flow_steps=baseline_flow,
                batch_size=baseline_batch,
            )
            for conv, std in product(conv_options, standardize_options)
        )
        # Block B: dense width and flow-steps scan on baseline conv.
        cfg_set.update(
            SweepConfig(
                arch=_arch(baseline_conv, dense),
                standardize_summary=baseline_std,
                flow_steps=flow_steps,
                batch_size=baseline_batch,
            )
            for dense, flow_steps in product(dense_options, flow_steps_values)
        )
        # Block C: batch-size check on baseline config.
        cfg_set.update(
            SweepConfig(
                arch=_arch(baseline_conv, baseline_dense),
                standardize_summary=baseline_std,
                flow_steps=baseline_flow,
                batch_size=batch_size,
            )
            for batch_size in batch_sizes_values
        )
        cfgs = list(cfg_set)

    cfgs = sorted(
        cfgs,
        key=lambda c: (
            tuple(_csv_ints(c.arch.conv_channels)),
            c.arch.dense_width,
            c.standardize_summary,
            c.flow_steps,
            c.batch_size,
        ),
    )
    return cfgs


def _maybe_run_job(job: Job, cwd: Path, dry_run: bool) -> Dict[str, object]:
    cmd = [str(x) for x in job.command] + ["--cuda-visible-devices", str(job.gpu)]
    job.log_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if dry_run:
        job.log_path.write_text(f"[dry-run] {_format_cmd(cmd)}\n", encoding="utf-8")
        rc = 0
        skipped = False
    else:
        already_done = (
            job.expected_output is not None
            and job.expected_output.exists()
            and job.log_path.exists()
            and not _log_has_traceback(job.log_path)
        )
        if already_done:
            rc = 0
            skipped = True
        else:
            skipped = False
            with job.log_path.open("w", encoding="utf-8") as logf:
                proc = subprocess.run(
                    cmd,
                    cwd=str(cwd),
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                )
            rc = int(proc.returncode)

    dt = float(time.time() - t0)
    print(
        f"[job] {job.name} stage={job.stage} gpu={job.gpu} rc={rc} "
        f"time={dt / 60.0:.2f}m skip={int(skipped)} log={job.log_path}"
    )

    return {
        "name": job.name,
        "stage": job.stage,
        "config_id": job.config_id,
        "seed": job.seed,
        "gpu": str(job.gpu),
        "returncode": int(rc),
        "seconds": dt,
        "skipped": bool(skipped),
        "log": str(job.log_path),
        "expected_output": str(job.expected_output) if job.expected_output else None,
        "cmd": cmd,
    }


def _build_train_job(
    arch: ArchConfig,
    args: argparse.Namespace,
    repo_root: Path,
    out_root: Path,
    cache_dir: Path,
    gpu: str,
) -> Job:
    cnn_script = repo_root / "scripts" / "sbi" / "npe_cnn_nbody_tomo.py"
    save_dir = out_root / "compressor" / arch.arch_id
    comp_paths = _cnn_comp_paths(
        save_dir=save_dir,
        map_kind=args.map_kind,
        nbins=4,
        steps=args.compressor_steps,
    )

    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "python",
        str(cnn_script),
        "--no-wandb",
        "--map-kind",
        args.map_kind,
        "--seed",
        str(args.compressor_seed),
        "--tfds-name",
        args.tfds_name,
        "--field-size",
        "20",
        "--field-npix",
        "160",
        "--nbins",
        "4",
        "--tomo-bin-indices",
        "1,2,3,4",
        "--cache-dir",
        str(cache_dir),
        "--save-dir",
        str(save_dir),
        "--train-compressor",
        "--compressor-dim",
        str(args.compressor_dim),
        "--compressor-steps",
        str(args.compressor_steps),
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

    return Job(
        name=f"train::{arch.arch_id}",
        stage="train_compressor",
        config_id=arch.arch_id,
        seed=None,
        command=cmd,
        log_path=out_root / "logs" / f"train_{arch.arch_id}.log",
        expected_output=comp_paths["params"],
        gpu=gpu,
    )


def _build_eval_job(
    cfg: SweepConfig,
    seed: int,
    args: argparse.Namespace,
    repo_root: Path,
    out_root: Path,
    cache_dir: Path,
    gpu: str,
) -> Job:
    cnn_script = repo_root / "scripts" / "sbi" / "npe_cnn_jaxili_nbody_tomo.py"
    comp_save_dir = out_root / "compressor" / cfg.arch.arch_id
    comp_paths = _cnn_comp_paths(
        save_dir=comp_save_dir,
        map_kind=args.map_kind,
        nbins=4,
        steps=args.compressor_steps,
    )
    posterior = out_root / "posteriors" / f"{cfg.config_id}_s{seed}.npy"

    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "python",
        str(cnn_script),
        "--no-wandb",
        "--map-kind",
        args.map_kind,
        "--seed",
        str(seed),
        "--tfds-name",
        args.tfds_name,
        "--field-size",
        "20",
        "--field-npix",
        "160",
        "--nbins",
        "4",
        "--tomo-bin-indices",
        "1,2,3,4",
        "--cache-dir",
        str(cache_dir),
        "--save-dir",
        str(out_root / "cnn_eval" / cfg.config_id / f"seed_{seed}"),
        "--compressor-params",
        str(comp_paths["params"]),
        "--compressor-state",
        str(comp_paths["state"]),
        "--compressor-dim",
        str(args.compressor_dim),
        "--total-steps",
        str(cfg.flow_steps),
        "--save-every",
        str(args.flow_save_every),
        "--patience",
        str(args.flow_patience),
        "--batch-size",
        str(cfg.batch_size),
        "--npe-samples",
        str(args.npe_samples),
        "--posterior-out",
        str(posterior),
        "--ds-batch-size",
        str(args.ds_batch_size),
        "--summary-clip-value",
        str(args.summary_clip_value),
    ]
    if cfg.standardize_summary:
        cmd.append("--standardize-summary")
    else:
        cmd.append("--no-standardize-summary")

    return Job(
        name=f"eval::{cfg.config_id}::s{seed}",
        stage="eval",
        config_id=cfg.config_id,
        seed=seed,
        command=cmd,
        log_path=out_root / "logs" / f"eval_{cfg.config_id}_s{seed}.log",
        expected_output=posterior,
        gpu=gpu,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compact no-BNT tomo4 CNN sweep with seed41 ranking and top-2 "
            "robustness reruns on seeds42/43 using FoM3(first 3 params)."
        )
    )
    p.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--gpus", type=str, default="0,1,2")
    p.add_argument("--conda-env", type=str, default="jaxili")

    p.add_argument("--seed41", type=int, default=41)
    p.add_argument("--robust-seeds", type=str, default="42,43")
    p.add_argument("--compressor-seed", type=int, default=41)

    p.add_argument("--map-kind", type=str, default="nbody")
    p.add_argument(
        "--tfds-name",
        type=str,
        default="NbodyCosmogridDatasetTomo/grid_20deg_160px",
    )

    p.add_argument(
        "--conv-channel-options",
        type=str,
        default="64,128,256;96,192,384",
        help="Semicolon-separated CNN conv stacks, each as comma-separated ints.",
    )
    p.add_argument("--dense-width-options", type=str, default="128,256")
    p.add_argument("--standardize-options", type=str, default="off,on")
    p.add_argument("--flow-steps-options", type=str, default="5000,10000")
    p.add_argument(
        "--batch-size-options",
        type=str,
        default="256",
        help="Comma-separated batch sizes for flow training.",
    )
    p.add_argument(
        "--grid-mode",
        type=str,
        choices=["focused", "cartesian"],
        default="focused",
        help=(
            "focused: compact architecture/flow/batch blocks; "
            "cartesian: full product over all options."
        ),
    )
    p.add_argument("--compressor-dim", type=int, default=6)
    p.add_argument("--compressor-steps", type=int, default=20000)
    p.add_argument("--compressor-save-every", type=int, default=2000)
    p.add_argument("--compressor-batch-size", type=int, default=128)
    p.add_argument("--compressor-lr", type=float, default=5e-4)
    p.add_argument("--compressor-pool-window", type=int, default=16)
    p.add_argument("--compressor-pool-stride", type=int, default=8)

    p.add_argument("--flow-save-every", type=int, default=500)
    p.add_argument("--flow-patience", type=int, default=30)
    p.add_argument("--nvp-layers", type=int, default=4)
    p.add_argument("--nvp-hidden", type=int, default=128)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--npe-samples", type=int, default=100000)
    p.add_argument("--ds-batch-size", type=int, default=500)
    p.add_argument("--summary-clip-value", type=float, default=5.0)

    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache dir. Defaults to <output-root>/cache/cnn_tomo4_20deg160_nobnt.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = args.repo_root.resolve()
    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    (out_root / "posteriors").mkdir(parents=True, exist_ok=True)

    lock_path = _acquire_output_lock(out_root)
    try:
        gpus = _csv_tokens(args.gpus)
        if not gpus:
            raise ValueError("--gpus cannot be empty.")

        robust_seeds = _csv_ints(args.robust_seeds)
        if not robust_seeds:
            raise ValueError("--robust-seeds cannot be empty.")
        if args.seed41 in robust_seeds:
            raise ValueError("--robust-seeds must not include --seed41.")

        if args.compressor_steps < args.compressor_save_every:
            raise ValueError("--compressor-steps must be >= --compressor-save-every.")

        archs = _build_architectures(args)
        configs = _build_configs(args, archs)
        if len(configs) < 2:
            raise ValueError("Need at least 2 configs for top-2 robustness protocol.")

        cache_root = (
            args.cache_dir.resolve()
            if args.cache_dir is not None
            else (out_root / "cache" / "cnn_tomo4_20deg160_nobnt")
        )
        cache_root.mkdir(parents=True, exist_ok=True)

        manifest = {
            "repo_root": str(repo_root),
            "output_root": str(out_root),
            "cache_dir": str(cache_root),
            "gpus": gpus,
            "seed41": int(args.seed41),
            "robust_seeds": robust_seeds,
            "protocol": "seed41 ranking -> top2 robust on seeds42,43 using FoM3(first 3 params)",
            "grid_mode": args.grid_mode,
            "fixed_data": {
                "apply_bnt": False,
                "variant": "tomo4_20deg160",
                "tfds_name": args.tfds_name,
                "field_size": 20,
                "field_npix": 160,
                "nbins": 4,
                "tomo_bins": "1,2,3,4",
            },
            "sweep_space": {
                "conv_channel_options": _parse_conv_channel_options(args.conv_channel_options),
                "dense_width_options": sorted(set(_csv_ints(args.dense_width_options))),
                "standardize_options": [
                    "on" if x else "off" for x in _csv_bool_tokens(args.standardize_options)
                ],
                "flow_steps_options": sorted(set(_csv_ints(args.flow_steps_options))),
                "batch_size_options": sorted(set(_csv_ints(args.batch_size_options))),
                "compressor_steps": int(args.compressor_steps),
            },
            "grid_size": len(configs),
            "architectures": [asdict(arch) for arch in archs],
            "configs": [
                {
                    "config_id": cfg.config_id,
                    "conv_channels": cfg.arch.conv_channels,
                    "dense_width": cfg.arch.dense_width,
                    "standardize_summary": cfg.standardize_summary,
                    "flow_steps": cfg.flow_steps,
                    "batch_size": cfg.batch_size,
                }
                for cfg in configs
            ],
        }
        (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[manifest] wrote {out_root / 'manifest.json'}")

        all_results: List[Dict[str, object]] = []
        eval_lookup: Dict[Tuple[str, int], Dict[str, object]] = {}

        gpu_index = 0

        used_archs = sorted(
            {cfg.arch for cfg in configs},
            key=lambda arch: (tuple(_csv_ints(arch.conv_channels)), arch.dense_width),
        )

        for arch in used_archs:
            gpu = gpus[gpu_index % len(gpus)]
            gpu_index += 1

            train_cache_dir = cache_root / f"arch_{arch.arch_id}"
            train_cache_dir.mkdir(parents=True, exist_ok=True)

            train_job = _build_train_job(
                arch=arch,
                args=args,
                repo_root=repo_root,
                out_root=out_root,
                cache_dir=train_cache_dir,
                gpu=gpu,
            )
            train_result = _maybe_run_job(train_job, repo_root, args.dry_run)
            all_results.append(train_result)

            if int(train_result["returncode"]) != 0:
                raise RuntimeError(
                    f"Compressor training failed for {arch.arch_id}. "
                    f"See log: {train_result['log']}"
                )

            comp_paths = _cnn_comp_paths(
                save_dir=out_root / "compressor" / arch.arch_id,
                map_kind=args.map_kind,
                nbins=4,
                steps=args.compressor_steps,
            )
            if not args.dry_run:
                for label in ("params", "state"):
                    if not comp_paths[label].exists():
                        raise FileNotFoundError(
                            f"Missing compressor {label} for {arch.arch_id}: {comp_paths[label]}"
                        )

        for cfg in configs:
            gpu = gpus[gpu_index % len(gpus)]
            gpu_index += 1

            eval_cache_dir = cache_root / f"arch_{cfg.arch.arch_id}"
            eval_cache_dir.mkdir(parents=True, exist_ok=True)

            eval_job = _build_eval_job(
                cfg=cfg,
                seed=int(args.seed41),
                args=args,
                repo_root=repo_root,
                out_root=out_root,
                cache_dir=eval_cache_dir,
                gpu=gpu,
            )
            eval_result = _maybe_run_job(eval_job, repo_root, args.dry_run)
            all_results.append(eval_result)
            eval_lookup[(cfg.config_id, int(args.seed41))] = eval_result

        if args.dry_run:
            (out_root / "job_results.json").write_text(
                json.dumps(all_results, indent=2),
                encoding="utf-8",
            )
            print("[dry-run] planned jobs only; skipping FoM3 ranking/selection outputs.")
            return

        seed41_rows: List[Dict[str, object]] = []
        for cfg in configs:
            seed = int(args.seed41)
            posterior = out_root / "posteriors" / f"{cfg.config_id}_s{seed}.npy"
            log_path = out_root / "logs" / f"eval_{cfg.config_id}_s{seed}.log"
            result = eval_lookup.get((cfg.config_id, seed), {})
            rc = int(result.get("returncode", 1 if not posterior.exists() else 0))
            traceback_found = _log_has_traceback(log_path)

            if posterior.exists() and rc == 0 and not traceback_found:
                fom3, det_cov3, logdet_cov3, valid_fom, n_samples = _fom3_from_posterior(
                    posterior
                )
            else:
                fom3, det_cov3, logdet_cov3, valid_fom, n_samples = (
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    False,
                    0,
                )

            seed41_rows.append(
                {
                    "config_id": cfg.config_id,
                    "seed": seed,
                    "conv_channels": cfg.arch.conv_channels,
                    "dense_width": cfg.arch.dense_width,
                    "standardize_summary": cfg.standardize_summary,
                    "flow_steps": cfg.flow_steps,
                    "batch_size": cfg.batch_size,
                    "returncode": rc,
                    "traceback_found": traceback_found,
                    "posterior_exists": posterior.exists(),
                    "posterior": str(posterior),
                    "log": str(log_path),
                    "n_samples": n_samples,
                    "fom3": fom3,
                    "det_cov3": det_cov3,
                    "logdet_cov3": logdet_cov3,
                    "valid_fom": valid_fom,
                }
            )

        _write_csv(
            out_root / "seed41_results.csv",
            seed41_rows,
            fieldnames=[
                "config_id",
                "seed",
                "conv_channels",
                "dense_width",
                "standardize_summary",
                "flow_steps",
                "batch_size",
                "returncode",
                "traceback_found",
                "posterior_exists",
                "posterior",
                "log",
                "n_samples",
                "fom3",
                "det_cov3",
                "logdet_cov3",
                "valid_fom",
            ],
        )
        (out_root / "seed41_results.json").write_text(
            json.dumps(seed41_rows, indent=2),
            encoding="utf-8",
        )

        ranked = [
            row
            for row in seed41_rows
            if bool(row["valid_fom"])
            and int(row["returncode"]) == 0
            and not bool(row["traceback_found"])
            and bool(row["posterior_exists"])
        ]
        ranked.sort(key=lambda r: float(r["fom3"]), reverse=True)
        for i, row in enumerate(ranked, start=1):
            row["rank_seed41"] = i

        _write_csv(
            out_root / "seed41_ranked.csv",
            ranked,
            fieldnames=[
                "rank_seed41",
                "config_id",
                "seed",
                "conv_channels",
                "dense_width",
                "standardize_summary",
                "flow_steps",
                "batch_size",
                "returncode",
                "traceback_found",
                "posterior_exists",
                "posterior",
                "log",
                "n_samples",
                "fom3",
                "det_cov3",
                "logdet_cov3",
                "valid_fom",
            ],
        )
        (out_root / "seed41_ranked.json").write_text(
            json.dumps(ranked, indent=2),
            encoding="utf-8",
        )

        if len(ranked) < 2:
            raise RuntimeError(
                f"Need at least 2 valid seed41 configs, got {len(ranked)}. "
                f"Inspect {out_root / 'seed41_results.json'}."
            )

        cfg_by_id = {cfg.config_id: cfg for cfg in configs}
        top2_ids = [str(ranked[0]["config_id"]), str(ranked[1]["config_id"])]
        top2_cfgs = [cfg_by_id[cfg_id] for cfg_id in top2_ids]
        print(f"[stage2] top2 seed41 configs: {top2_ids}")

        for cfg in top2_cfgs:
            for seed in robust_seeds:
                gpu = gpus[gpu_index % len(gpus)]
                gpu_index += 1

                eval_cache_dir = cache_root / f"arch_{cfg.arch.arch_id}"
                eval_cache_dir.mkdir(parents=True, exist_ok=True)

                eval_job = _build_eval_job(
                    cfg=cfg,
                    seed=seed,
                    args=args,
                    repo_root=repo_root,
                    out_root=out_root,
                    cache_dir=eval_cache_dir,
                    gpu=gpu,
                )
                eval_result = _maybe_run_job(eval_job, repo_root, args.dry_run)
                all_results.append(eval_result)
                eval_lookup[(cfg.config_id, int(seed))] = eval_result

        (out_root / "job_results.json").write_text(
            json.dumps(all_results, indent=2),
            encoding="utf-8",
        )

        robust_rows: List[Dict[str, object]] = []
        robust_seeds_full = [int(args.seed41)] + robust_seeds
        for cfg in top2_cfgs:
            per_seed: List[Dict[str, object]] = []
            fom_vals: List[float] = []
            clean_logs = True
            all_returncode_zero = True
            all_posteriors_present = True

            for seed in robust_seeds_full:
                posterior = out_root / "posteriors" / f"{cfg.config_id}_s{seed}.npy"
                log_path = out_root / "logs" / f"eval_{cfg.config_id}_s{seed}.log"
                result = eval_lookup.get((cfg.config_id, seed), {})

                traceback_found = _log_has_traceback(log_path)
                if traceback_found:
                    clean_logs = False

                rc = int(result.get("returncode", 1 if not posterior.exists() else 0))
                if rc != 0:
                    all_returncode_zero = False

                if posterior.exists() and rc == 0 and not traceback_found:
                    fom3, det_cov3, logdet_cov3, valid_fom, n_samples = _fom3_from_posterior(
                        posterior
                    )
                    if valid_fom:
                        fom_vals.append(fom3)
                else:
                    fom3, det_cov3, logdet_cov3, valid_fom, n_samples = (
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        False,
                        0,
                    )

                if not posterior.exists():
                    all_posteriors_present = False

                per_seed.append(
                    {
                        "seed": seed,
                        "returncode": rc,
                        "traceback_found": traceback_found,
                        "posterior_exists": posterior.exists(),
                        "posterior": str(posterior),
                        "log": str(log_path),
                        "n_samples": n_samples,
                        "fom3": fom3,
                        "det_cov3": det_cov3,
                        "logdet_cov3": logdet_cov3,
                        "valid_fom": valid_fom,
                    }
                )

            n_valid = len(fom_vals)
            mean_fom3 = (
                float(np.mean(np.array(fom_vals, dtype=np.float64)))
                if n_valid
                else float("nan")
            )
            std_fom3 = (
                float(np.std(np.array(fom_vals, dtype=np.float64), ddof=1))
                if n_valid > 1
                else 0.0 if n_valid == 1 else float("nan")
            )

            robust_rows.append(
                {
                    "config_id": cfg.config_id,
                    "conv_channels": cfg.arch.conv_channels,
                    "dense_width": cfg.arch.dense_width,
                    "standardize_summary": cfg.standardize_summary,
                    "flow_steps": cfg.flow_steps,
                    "batch_size": cfg.batch_size,
                    "seed41_fom3": next(
                        float(r["fom3"]) for r in per_seed if int(r["seed"]) == int(args.seed41)
                    ),
                    "fom3_mean": mean_fom3,
                    "fom3_std": std_fom3,
                    "n_valid": n_valid,
                    "n_total": len(robust_seeds_full),
                    "all_returncode_zero": all_returncode_zero,
                    "all_posteriors_present": all_posteriors_present,
                    "all_logs_clean_no_traceback": clean_logs,
                    "per_seed": per_seed,
                }
            )

        robust_rows.sort(key=lambda r: float(r["fom3_mean"]), reverse=True)
        for i, row in enumerate(robust_rows, start=1):
            row["rank_robust"] = i

        _write_csv(
            out_root / "top2_robustness.csv",
            robust_rows,
            fieldnames=[
                "rank_robust",
                "config_id",
                "conv_channels",
                "dense_width",
                "standardize_summary",
                "flow_steps",
                "batch_size",
                "seed41_fom3",
                "fom3_mean",
                "fom3_std",
                "n_valid",
                "n_total",
                "all_returncode_zero",
                "all_posteriors_present",
                "all_logs_clean_no_traceback",
            ],
        )
        robust_json = json.dumps(robust_rows, indent=2)
        (out_root / "top2_robustness.json").write_text(
            robust_json,
            encoding="utf-8",
        )
        (out_root / "robust_comparison.json").write_text(
            robust_json,
            encoding="utf-8",
        )
        _write_csv(
            out_root / "robust_comparison.csv",
            robust_rows,
            fieldnames=[
                "rank_robust",
                "config_id",
                "conv_channels",
                "dense_width",
                "standardize_summary",
                "flow_steps",
                "batch_size",
                "seed41_fom3",
                "fom3_mean",
                "fom3_std",
                "n_valid",
                "n_total",
                "all_returncode_zero",
                "all_posteriors_present",
                "all_logs_clean_no_traceback",
            ],
        )

        best = next(
            (
                row
                for row in robust_rows
                if bool(row["all_returncode_zero"])
                and bool(row["all_posteriors_present"])
                and bool(row["all_logs_clean_no_traceback"])
                and int(row["n_valid"]) == int(row["n_total"])
            ),
            None,
        )
        if best is None:
            raise RuntimeError(
                "No robust candidate passed clean-run requirements. "
                f"Inspect {out_root / 'top2_robustness.json'}."
            )

        final_summary = {
            "best_config": best,
            "protocol": {
                "stage1": "seed41 ranking by FoM3(first 3 params)",
                "stage2": f"top2 rerun on robust seeds {robust_seeds}",
            },
            "artifacts": {
                "manifest": str((out_root / "manifest.json").resolve()),
                "job_results": str((out_root / "job_results.json").resolve()),
                "seed41_results_json": str((out_root / "seed41_results.json").resolve()),
                "seed41_results_csv": str((out_root / "seed41_results.csv").resolve()),
                "seed41_ranked_json": str((out_root / "seed41_ranked.json").resolve()),
                "seed41_ranked_csv": str((out_root / "seed41_ranked.csv").resolve()),
                "top2_robustness_json": str((out_root / "top2_robustness.json").resolve()),
                "top2_robustness_csv": str((out_root / "top2_robustness.csv").resolve()),
                "robust_comparison_json": str((out_root / "robust_comparison.json").resolve()),
                "robust_comparison_csv": str((out_root / "robust_comparison.csv").resolve()),
            },
        }
        (out_root / "final_selection.json").write_text(
            json.dumps(final_summary, indent=2),
            encoding="utf-8",
        )
        _write_csv(
            out_root / "final_selection.csv",
            [best],
            fieldnames=[
                "rank_robust",
                "config_id",
                "conv_channels",
                "dense_width",
                "standardize_summary",
                "flow_steps",
                "batch_size",
                "seed41_fom3",
                "fom3_mean",
                "fom3_std",
                "n_valid",
                "n_total",
                "all_returncode_zero",
                "all_posteriors_present",
                "all_logs_clean_no_traceback",
            ],
        )

        print(
            f"[done] best={best['config_id']} "
            f"fom3_mean={float(best['fom3_mean']):.6f} "
            f"fom3_std={float(best['fom3_std']):.6f}"
        )
        print(f"[done] artifacts in {out_root}")
    finally:
        _release_output_lock(lock_path)


if __name__ == "__main__":
    main()
