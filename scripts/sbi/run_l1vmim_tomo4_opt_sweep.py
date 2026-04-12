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
    / "l1vmim_tomo4"
)


@dataclass(frozen=True)
class VmimArchitecture:
    compressor_hidden: str
    vmim_nf_layers: int
    vmim_nf_hidden: int

    @property
    def hidden_tag(self) -> str:
        return self.compressor_hidden.replace(",", "x")


@dataclass(frozen=True)
class SweepConfig:
    compressor_dim: int
    architecture: VmimArchitecture
    flow_steps: int
    batch_size: int

    @property
    def config_id(self) -> str:
        return (
            f"cdim{self.compressor_dim}"
            f"_h{self.architecture.hidden_tag}"
            f"_nf{self.architecture.vmim_nf_layers}x{self.architecture.vmim_nf_hidden}"
            f"_flow{self.flow_steps}"
            f"_bs{self.batch_size}"
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
    env: Optional[Dict[str, str]] = None


def _csv_tokens(value: str) -> List[str]:
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def _csv_ints(value: str) -> List[int]:
    return [int(tok) for tok in _csv_tokens(value)]


def _arch_bundle_tokens(value: str) -> List[str]:
    return [tok.strip() for tok in value.split(";") if tok.strip()]


def _parse_arch_bundles(value: str) -> List[VmimArchitecture]:
    bundles: List[VmimArchitecture] = []
    for token in _arch_bundle_tokens(value):
        parts = [p.strip() for p in token.split(":")]
        if len(parts) != 3:
            raise ValueError(
                f"Invalid arch bundle '{token}'. Expected format "
                "'hidden1,hidden2:nf_layers:nf_hidden'."
            )
        hidden, nf_layers, nf_hidden = parts
        hidden_vals = _csv_ints(hidden)
        if len(hidden_vals) < 1 or any(v <= 0 for v in hidden_vals):
            raise ValueError(f"Invalid compressor hidden widths in bundle '{token}'.")
        nf_layers_i = int(nf_layers)
        nf_hidden_i = int(nf_hidden)
        if nf_layers_i <= 0 or nf_hidden_i <= 0:
            raise ValueError(f"Invalid NF settings in bundle '{token}'.")
        bundles.append(
            VmimArchitecture(
                compressor_hidden=",".join(str(v) for v in hidden_vals),
                vmim_nf_layers=nf_layers_i,
                vmim_nf_hidden=nf_hidden_i,
            )
        )
    if not bundles:
        raise ValueError("At least one architecture bundle is required.")
    return bundles


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


def _vmim_comp_paths(save_dir: Path, map_kind: str, nbins: int) -> Dict[str, Path]:
    base = (
        save_dir
        / "vmim_l1"
        / map_kind
        / "sigma_0.26"
        / "gal_density_30"
        / f"bin_{nbins}"
    )
    return {
        "params": base / "params_nd_compressor_best.pkl",
        "state": base / "opt_state_resnet_best.pkl",
        "summary": base / "compressor_training_summary.json",
    }


def _build_configs(args: argparse.Namespace) -> List[SweepConfig]:
    dims = _csv_ints(args.compressor_dims)
    flow_steps = _csv_ints(args.flow_steps_options)
    batch_sizes = _csv_ints(args.batch_size_options)
    archs = _parse_arch_bundles(args.arch_bundles)

    if not dims or any(v <= 0 for v in dims):
        raise ValueError("--compressor-dims must be positive ints.")
    if not flow_steps or any(v <= 0 for v in flow_steps):
        raise ValueError("--flow-steps-options must be positive ints.")
    if not batch_sizes or any(v <= 0 for v in batch_sizes):
        raise ValueError("--batch-size-options must be positive ints.")

    cfgs = [
        SweepConfig(
            compressor_dim=int(dim),
            architecture=arch,
            flow_steps=int(flow),
            batch_size=int(bs),
        )
        for dim, arch, flow, bs in product(dims, archs, flow_steps, batch_sizes)
    ]
    cfgs = sorted(
        cfgs,
        key=lambda c: (
            c.compressor_dim,
            c.architecture.compressor_hidden,
            c.architecture.vmim_nf_layers,
            c.architecture.vmim_nf_hidden,
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
            and not _log_has_traceback(job.log_path)
        )
        if already_done:
            rc = 0
            skipped = True
        else:
            skipped = False
            run_env = os.environ.copy()
            if job.env:
                run_env.update(job.env)
            with job.log_path.open("w", encoding="utf-8") as logf:
                proc = subprocess.run(
                    cmd,
                    cwd=str(cwd),
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    env=run_env,
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
        "env": dict(job.env) if job.env else {},
        "cmd": cmd,
    }


def _build_train_job(
    cfg: SweepConfig,
    args: argparse.Namespace,
    repo_root: Path,
    out_root: Path,
    cache_dir: Path,
    gpu: str,
    extra_env: Optional[Dict[str, str]],
) -> Job:
    vmim_script = repo_root / "scripts" / "sbi" / "npe_l1vmim_jaxili_nbody_tomo.py"
    save_dir = out_root / "compressor" / cfg.config_id
    comp_paths = _vmim_comp_paths(save_dir=save_dir, map_kind=args.map_kind, nbins=4)

    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "python",
        str(vmim_script),
        "--no-wandb",
        "--map-kind",
        args.map_kind,
        "--seed",
        str(args.compressor_seed),
        "--tfds-name",
        args.tfds_name,
        "--field-size",
        str(args.field_size),
        "--field-npix",
        str(args.field_npix),
        "--nbins",
        "4",
        "--tomo-bin-indices",
        args.tomo_bins,
        "--n-scales",
        str(args.n_scales),
        "--l1-nbins",
        str(args.l1_nbins),
        "--l1-min-snr",
        str(args.l1_min_snr),
        "--l1-max-snr",
        str(args.l1_max_snr),
        "--cache-dir",
        str(cache_dir),
        "--save-dir",
        str(save_dir),
        "--train-compressor",
        "--compressor-log1p-input",
        "--compressor-input-standardize",
        "--compressor-input-clip",
        str(args.compressor_input_clip),
        "--compressor-dim",
        str(cfg.compressor_dim),
        "--compressor-hidden",
        cfg.architecture.compressor_hidden,
        "--compressor-vmim-nf-layers",
        str(cfg.architecture.vmim_nf_layers),
        "--compressor-vmim-nf-hidden",
        str(cfg.architecture.vmim_nf_hidden),
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
        "--epochs",
        "2",
        "--save-every",
        "1",
        "--no-sample",
    ]

    return Job(
        name=f"train::{cfg.config_id}",
        stage="train_compressor",
        config_id=cfg.config_id,
        seed=None,
        command=cmd,
        log_path=out_root / "logs" / f"train_{cfg.config_id}.log",
        expected_output=comp_paths["params"],
        gpu=gpu,
        env=extra_env,
    )


def _build_eval_job(
    cfg: SweepConfig,
    seed: int,
    args: argparse.Namespace,
    repo_root: Path,
    out_root: Path,
    cache_dir: Path,
    gpu: str,
    extra_env: Optional[Dict[str, str]],
) -> Job:
    vmim_script = repo_root / "scripts" / "sbi" / "npe_l1vmim_jaxili_nbody_tomo.py"
    save_dir = out_root / "compressor" / cfg.config_id
    comp_paths = _vmim_comp_paths(save_dir=save_dir, map_kind=args.map_kind, nbins=4)
    posterior = out_root / "posteriors" / f"{cfg.config_id}_s{seed}.npy"

    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "python",
        str(vmim_script),
        "--no-wandb",
        "--map-kind",
        args.map_kind,
        "--seed",
        str(seed),
        "--tfds-name",
        args.tfds_name,
        "--field-size",
        str(args.field_size),
        "--field-npix",
        str(args.field_npix),
        "--nbins",
        "4",
        "--tomo-bin-indices",
        args.tomo_bins,
        "--n-scales",
        str(args.n_scales),
        "--l1-nbins",
        str(args.l1_nbins),
        "--l1-min-snr",
        str(args.l1_min_snr),
        "--l1-max-snr",
        str(args.l1_max_snr),
        "--cache-dir",
        str(cache_dir),
        "--save-dir",
        str(out_root / "l1vmim_eval" / cfg.config_id / f"seed_{seed}"),
        "--compressor-log1p-input",
        "--compressor-input-standardize",
        "--compressor-input-clip",
        str(args.compressor_input_clip),
        "--compressor-dim",
        str(cfg.compressor_dim),
        "--compressor-hidden",
        cfg.architecture.compressor_hidden,
        "--compressor-vmim-nf-layers",
        str(cfg.architecture.vmim_nf_layers),
        "--compressor-vmim-nf-hidden",
        str(cfg.architecture.vmim_nf_hidden),
        "--compressor-params",
        str(comp_paths["params"]),
        "--compressor-state",
        str(comp_paths["state"]),
        "--total-steps",
        str(cfg.flow_steps),
        "--save-every",
        str(args.flow_save_every),
        "--patience",
        str(args.flow_patience),
        "--batch-size",
        str(cfg.batch_size),
        "--nvp-layers",
        str(args.nvp_layers),
        "--nvp-hidden",
        str(args.nvp_hidden),
        "--weight-decay",
        str(args.weight_decay),
        "--grad-clip",
        str(args.grad_clip),
        "--npe-samples",
        str(args.npe_samples),
        "--posterior-out",
        str(posterior),
        "--summary-clip-value",
        str(args.vmim_summary_clip_value),
    ]
    return Job(
        name=f"eval::{cfg.config_id}::s{seed}",
        stage="eval",
        config_id=cfg.config_id,
        seed=seed,
        command=cmd,
        log_path=out_root / "logs" / f"eval_{cfg.config_id}_s{seed}.log",
        expected_output=posterior,
        gpu=gpu,
        env=extra_env,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Focused no-BNT tomo4 L1+VMIM optimization sweep with seed41 ranking "
            "and top-2 robustness runs on seeds 42/43."
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
    p.add_argument("--field-size", type=int, default=20)
    p.add_argument("--field-npix", type=int, default=160)
    p.add_argument("--tomo-bins", type=str, default="1,2,3,4")

    p.add_argument("--n-scales", type=int, default=5)
    p.add_argument("--l1-nbins", type=int, default=40)
    p.add_argument("--l1-min-snr", type=float, default=-13.0)
    p.add_argument("--l1-max-snr", type=float, default=13.0)

    p.add_argument("--compressor-dims", type=str, default="48,64")
    p.add_argument(
        "--arch-bundles",
        type=str,
        default="512,512:8:256;768,768:10:384",
        help=(
            "Semicolon-separated architecture bundles in format "
            "'hidden1,hidden2:nf_layers:nf_hidden'."
        ),
    )
    p.add_argument("--flow-steps-options", type=str, default="12000")
    p.add_argument("--batch-size-options", type=str, default="256,384")

    p.add_argument("--compressor-steps", type=int, default=12000)
    p.add_argument("--compressor-save-every", type=int, default=500)
    p.add_argument("--compressor-batch-size", type=int, default=128)
    p.add_argument("--compressor-lr", type=float, default=3e-4)
    p.add_argument("--compressor-input-clip", type=float, default=6.0)

    p.add_argument("--flow-save-every", type=int, default=500)
    p.add_argument("--flow-patience", type=int, default=30)
    p.add_argument("--nvp-layers", type=int, default=4)
    p.add_argument("--nvp-hidden", type=int, default=128)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--vmim-summary-clip-value", type=float, default=0.0)
    p.add_argument("--npe-samples", type=int, default=100000)

    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional shared cache dir. Defaults to <output-root>/cache/l1vmim_tomo4_20deg160_nobnt.",
    )
    p.add_argument(
        "--disable-jax-preallocate",
        action="store_true",
        help="Set XLA_PYTHON_CLIENT_PREALLOCATE=false for child jobs.",
    )
    p.add_argument(
        "--xla-mem-fraction",
        type=float,
        default=0.0,
        help="Optional XLA_PYTHON_CLIENT_MEM_FRACTION for child jobs (>0 enables).",
    )
    p.add_argument(
        "--torch-expandable-segments",
        action="store_true",
        help="Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for child jobs.",
    )
    p.add_argument(
        "--skip-stage1-exec",
        action="store_true",
        help="Skip stage1 job execution; analyze existing seed41 artifacts only.",
    )
    p.add_argument(
        "--skip-stage2-exec",
        action="store_true",
        help="Skip stage2 top2 reruns; summarize using existing robust-seed artifacts.",
    )
    p.add_argument(
        "--stage1-only",
        action="store_true",
        help="Run seed41 sweep only (no top2 robust reruns).",
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

    gpus = _csv_tokens(args.gpus)
    if not gpus:
        raise ValueError("--gpus cannot be empty.")

    robust_seeds = _csv_ints(args.robust_seeds)
    if not robust_seeds:
        raise ValueError("--robust-seeds cannot be empty.")
    if args.seed41 in robust_seeds:
        raise ValueError("--robust-seeds must not include --seed41.")

    cache_dir = (
        args.cache_dir.resolve()
        if args.cache_dir is not None
        else (out_root / "cache" / "l1vmim_tomo4_20deg160_nobnt")
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    configs = _build_configs(args)
    if len(configs) < 2:
        raise ValueError("Need at least 2 configs for top-2 robustness protocol.")

    child_env: Dict[str, str] = {}
    if args.disable_jax_preallocate:
        child_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.xla_mem_fraction > 0:
        child_env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(float(args.xla_mem_fraction))
    if args.torch_expandable_segments:
        child_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    job_env = child_env if child_env else None

    manifest = {
        "repo_root": str(repo_root),
        "output_root": str(out_root),
        "cache_dir": str(cache_dir),
        "gpus": gpus,
        "seed41": int(args.seed41),
        "robust_seeds": robust_seeds,
        "protocol": "seed41 ranking -> top2 robust on seeds42,43 using FoM3(first 3 params)",
        "runtime_env_overrides": child_env,
        "fixed_extraction": {
            "n_scales": int(args.n_scales),
            "l1_nbins": int(args.l1_nbins),
            "l1_min_snr": float(args.l1_min_snr),
            "l1_max_snr": float(args.l1_max_snr),
            "apply_bnt": False,
            "variant": "tomo4_20deg160",
        },
        "sweep_space": {
            "compressor_dims": _csv_ints(args.compressor_dims),
            "arch_bundles": [asdict(b) for b in _parse_arch_bundles(args.arch_bundles)],
            "flow_steps_options": _csv_ints(args.flow_steps_options),
            "batch_size_options": _csv_ints(args.batch_size_options),
            "compressor_steps": int(args.compressor_steps),
        },
        "grid_size": len(configs),
        "configs": [asdict(c) for c in configs],
        "execution_flags": {
            "skip_stage1_exec": bool(args.skip_stage1_exec),
            "skip_stage2_exec": bool(args.skip_stage2_exec),
            "stage1_only": bool(args.stage1_only),
        },
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[manifest] wrote {out_root / 'manifest.json'}")

    all_results: List[Dict[str, object]] = []
    eval_lookup: Dict[Tuple[str, int], Dict[str, object]] = {}
    existing_results_path = out_root / "job_results.json"
    if (args.skip_stage1_exec or args.skip_stage2_exec) and existing_results_path.exists():
        try:
            existing_rows = json.loads(existing_results_path.read_text(encoding="utf-8"))
            if isinstance(existing_rows, list):
                all_results.extend(existing_rows)
                print(f"[resume] loaded {len(existing_rows)} existing job results")
        except Exception as exc:  # pragma: no cover
            print(f"[resume] warning: failed to load existing job_results.json: {exc}")

    gpu_index = 0

    # Stage 1: seed41 sweep (train compressor + eval on seed41 for each config).
    if args.skip_stage1_exec:
        print("[stage1] skipping execution; analyzing existing seed41 artifacts only.")
    else:
        for cfg in configs:
            gpu = gpus[gpu_index % len(gpus)]
            gpu_index += 1

            train_job = _build_train_job(
                cfg, args, repo_root, out_root, cache_dir, gpu, job_env
            )
            train_result = _maybe_run_job(train_job, repo_root, args.dry_run)
            all_results.append(train_result)
            if int(train_result["returncode"]) != 0:
                print(
                    f"[warn] compressor training failed for {cfg.config_id}; "
                    f"skipping seed41 eval. log={train_result['log']}"
                )
                continue

            eval_job = _build_eval_job(
                cfg=cfg,
                seed=int(args.seed41),
                args=args,
                repo_root=repo_root,
                out_root=out_root,
                cache_dir=cache_dir,
                gpu=gpu,
                extra_env=job_env,
            )
            eval_result = _maybe_run_job(eval_job, repo_root, args.dry_run)
            all_results.append(eval_result)
            eval_lookup[(cfg.config_id, int(args.seed41))] = eval_result

    # Analyze seed41 ranking.
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
                "compressor_dim": cfg.compressor_dim,
                "compressor_hidden": cfg.architecture.compressor_hidden,
                "vmim_nf_layers": cfg.architecture.vmim_nf_layers,
                "vmim_nf_hidden": cfg.architecture.vmim_nf_hidden,
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
            "compressor_dim",
            "compressor_hidden",
            "vmim_nf_layers",
            "vmim_nf_hidden",
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
        json.dumps(seed41_rows, indent=2), encoding="utf-8"
    )

    ranked = [
        row
        for row in seed41_rows
        if bool(row["valid_fom"])
        and int(row["returncode"]) == 0
        and not bool(row["traceback_found"])
    ]
    ranked.sort(key=lambda r: float(r["fom3"]), reverse=True)
    for i, row in enumerate(ranked, start=1):
        row["rank_seed41"] = i

    _write_csv(
        out_root / "ranking_seed41.csv",
        ranked,
        fieldnames=[
            "rank_seed41",
            "config_id",
            "seed",
            "compressor_dim",
            "compressor_hidden",
            "vmim_nf_layers",
            "vmim_nf_hidden",
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
    (out_root / "ranking_seed41.json").write_text(
        json.dumps(ranked, indent=2), encoding="utf-8"
    )
    _write_csv(
        out_root / "ranking.csv",
        ranked,
        fieldnames=[
            "rank_seed41",
            "config_id",
            "seed",
            "compressor_dim",
            "compressor_hidden",
            "vmim_nf_layers",
            "vmim_nf_hidden",
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
    (out_root / "ranking.json").write_text(
        json.dumps(ranked, indent=2), encoding="utf-8"
    )

    if args.stage1_only:
        (out_root / "job_results.json").write_text(
            json.dumps(all_results, indent=2), encoding="utf-8"
        )
        print("[done] stage1-only complete (ranking artifacts written).")
        return

    if len(ranked) < 2:
        raise RuntimeError(
            f"Need at least 2 valid seed41 configs, got {len(ranked)}. "
            f"Inspect {out_root / 'seed41_results.json'}."
        )

    top2_ids = [str(ranked[0]["config_id"]), str(ranked[1]["config_id"])]
    cfg_by_id = {cfg.config_id: cfg for cfg in configs}
    top2_cfgs = [cfg_by_id[cfg_id] for cfg_id in top2_ids]
    print(f"[stage2] top2 seed41 configs: {top2_ids}")

    # Stage 2: run top-2 on robust seeds.
    if args.skip_stage2_exec:
        print("[stage2] skipping execution; summarizing from existing robust-seed artifacts.")
    else:
        for cfg in top2_cfgs:
            for seed in robust_seeds:
                gpu = gpus[gpu_index % len(gpus)]
                gpu_index += 1
                eval_job = _build_eval_job(
                    cfg=cfg,
                    seed=seed,
                    args=args,
                    repo_root=repo_root,
                    out_root=out_root,
                    cache_dir=cache_dir,
                    gpu=gpu,
                    extra_env=job_env,
                )
                eval_result = _maybe_run_job(eval_job, repo_root, args.dry_run)
                all_results.append(eval_result)
                eval_lookup[(cfg.config_id, int(seed))] = eval_result

    (out_root / "job_results.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )

    # Robustness summary for top-2 over seeds 41/42/43.
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
        mean_fom3 = float(np.mean(np.array(fom_vals, dtype=np.float64))) if n_valid else float("nan")
        std_fom3 = (
            float(np.std(np.array(fom_vals, dtype=np.float64), ddof=1))
            if n_valid > 1
            else 0.0 if n_valid == 1 else float("nan")
        )
        robust_rows.append(
            {
                "config_id": cfg.config_id,
                "compressor_dim": cfg.compressor_dim,
                "compressor_hidden": cfg.architecture.compressor_hidden,
                "vmim_nf_layers": cfg.architecture.vmim_nf_layers,
                "vmim_nf_hidden": cfg.architecture.vmim_nf_hidden,
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
            "compressor_dim",
            "compressor_hidden",
            "vmim_nf_layers",
            "vmim_nf_hidden",
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
    (out_root / "top2_robustness.json").write_text(
        json.dumps(robust_rows, indent=2), encoding="utf-8"
    )

    best = robust_rows[0]
    if (
        not bool(best["all_returncode_zero"])
        or not bool(best["all_posteriors_present"])
        or not bool(best["all_logs_clean_no_traceback"])
        or int(best["n_valid"]) != int(best["n_total"])
    ):
        raise RuntimeError(
            "Best robust config has failed/incomplete runs. "
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
            "ranking_json": str((out_root / "ranking_seed41.json").resolve()),
            "ranking_csv": str((out_root / "ranking_seed41.csv").resolve()),
            "top2_robustness_json": str((out_root / "top2_robustness.json").resolve()),
            "top2_robustness_csv": str((out_root / "top2_robustness.csv").resolve()),
        },
    }
    (out_root / "final_selection.json").write_text(
        json.dumps(final_summary, indent=2), encoding="utf-8"
    )
    print(
        f"[done] best={best['config_id']} "
        f"fom3_mean={float(best['fom3_mean']):.6f} "
        f"fom3_std={float(best['fom3_std']):.6f}"
    )
    print(f"[done] artifacts in {out_root}")


if __name__ == "__main__":
    main()
