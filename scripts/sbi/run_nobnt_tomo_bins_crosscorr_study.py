#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


DEFAULT_REPO_ROOT = str(Path(__file__).resolve().parents[2])
DEFAULT_OUTPUT_ROOT = str(
    Path(DEFAULT_REPO_ROOT) / "scripts" / "sbi" / "nobnt_tomo_bins_crosscorr_study"
)
TRUTH = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493], dtype=np.float64)

ALL_VARIANTS: Dict[str, Dict[str, object]] = {
    "bin1_20deg160": {
        "tfds": "NbodyCosmogridDatasetTomo/grid_20deg_160px",
        "field_size": 20,
        "field_npix": 160,
        "nbins": 1,
        "tomo_bins": "1",
    },
    "bin2_20deg160": {
        "tfds": "NbodyCosmogridDatasetTomo/grid_20deg_160px",
        "field_size": 20,
        "field_npix": 160,
        "nbins": 1,
        "tomo_bins": "2",
    },
    "bin3_20deg160": {
        "tfds": "NbodyCosmogridDatasetTomo/grid_20deg_160px",
        "field_size": 20,
        "field_npix": 160,
        "nbins": 1,
        "tomo_bins": "3",
    },
    "bin4_20deg160": {
        "tfds": "NbodyCosmogridDatasetTomo/grid_20deg_160px",
        "field_size": 20,
        "field_npix": 160,
        "nbins": 1,
        "tomo_bins": "4",
    },
    "tomo4_20deg160": {
        "tfds": "NbodyCosmogridDatasetTomo/grid_20deg_160px",
        "field_size": 20,
        "field_npix": 160,
        "nbins": 4,
        "tomo_bins": "1,2,3,4",
    },
}


@dataclass
class Job:
    name: str
    command: List[str]
    log_path: Path


def _csv_tokens(value: str) -> List[str]:
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def _csv_ints(value: str) -> List[int]:
    return [int(tok) for tok in _csv_tokens(value)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "No-BNT tomographic bin cross-correlation attribution study runner "
            "(CNN vs L1 vs L1+VMIM)."
        )
    )
    p.add_argument("--repo-root", type=str, default=DEFAULT_REPO_ROOT)
    p.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--gpus", type=str, default="0,1,2")
    p.add_argument("--seeds", type=str, default="41,42,43")
    p.add_argument(
        "--methods",
        type=str,
        default="cnn,l1,l1vmim",
        help="Comma-separated subset of {cnn,l1,l1vmim}.",
    )
    p.add_argument(
        "--variants",
        type=str,
        default="bin1_20deg160,bin2_20deg160,bin3_20deg160,bin4_20deg160,tomo4_20deg160",
        help="Comma-separated subset of supported variants.",
    )
    p.add_argument("--map-kind", type=str, default="nbody")
    p.add_argument("--compressor-seed", type=int, default=41)
    p.add_argument("--flow-save-every", type=int, default=500)
    p.add_argument("--flow-patience", type=int, default=30)
    p.add_argument("--npe-samples", type=int, default=100000)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    # CNN defaults from best Stage-J no-standardize family.
    p.add_argument("--cnn-compressor-dim", type=int, default=6)
    p.add_argument("--cnn-compressor-steps", type=int, default=60000)
    p.add_argument("--cnn-compressor-save-every", type=int, default=2000)
    p.add_argument("--cnn-compressor-conv-channels", type=str, default="64,128,256")
    p.add_argument("--cnn-compressor-dense-width", type=int, default=128)
    p.add_argument("--cnn-compressor-pool-window", type=int, default=16)
    p.add_argument("--cnn-compressor-pool-stride", type=int, default=8)
    p.add_argument("--cnn-flow-steps", type=int, default=5000)
    p.add_argument("--cnn-batch-size", type=int, default=256)
    p.add_argument("--cnn-nvp-layers", type=int, default=4)
    p.add_argument("--cnn-nvp-hidden", type=int, default=128)
    p.add_argument("--cnn-weight-decay", type=float, default=1e-4)
    p.add_argument("--cnn-grad-clip", type=float, default=1.0)
    p.add_argument("--cnn-ds-batch-size", type=int, default=500)
    p.add_argument(
        "--cnn-standardize-summary",
        action="store_true",
        help="Use summary standardization in CNN flow stage (default: disabled).",
    )
    p.add_argument("--cnn-summary-clip-value", type=float, default=0.0)

    # L1 no-compression defaults.
    p.add_argument("--l1-flow-steps", type=int, default=5000)
    p.add_argument("--l1-batch-size", type=int, default=256)
    p.add_argument("--l1-nvp-layers", type=int, default=4)
    p.add_argument("--l1-nvp-hidden", type=int, default=128)
    p.add_argument("--l1-weight-decay", type=float, default=1e-4)
    p.add_argument("--l1-grad-clip", type=float, default=1.0)
    p.add_argument("--l1-ds-batch-size", type=int, default=96)
    p.add_argument("--n-scales", type=int, default=5)
    p.add_argument("--l1-nbins", type=int, default=40)
    p.add_argument("--l1-min-snr", type=float, default=-13.0)
    p.add_argument("--l1-max-snr", type=float, default=13.0)
    p.add_argument(
        "--l1-estimator",
        type=str,
        choices=("legacy", "jaxili"),
        default="jaxili",
        help="L1 backend script selector (default: jaxili).",
    )
    p.add_argument(
        "--l1-pca-components",
        type=int,
        default=0,
        help="PCA components passed to L1 eval script (0 disables PCA).",
    )
    p.add_argument(
        "--l1-summary-transform",
        type=str,
        default="log1p-zscore",
        choices=["log1p-zscore", "log10p-zscore", "zscore", "log1p", "log10p", "none"],
        help="Summary preprocessing transform for jaxili L1 (default: log1p-zscore).",
    )
    p.add_argument(
        "--l1-clip-value",
        type=float,
        default=5.0,
        help="Summary clip value for jaxili L1 (default: 5.0).",
    )
    p.add_argument(
        "--l1-learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for jaxili L1 training (default: 1e-4).",
    )
    p.add_argument(
        "--l1-epochs",
        type=int,
        default=None,
        help="Epochs for jaxili L1 training; if unset, uses --l1-flow-steps.",
    )

    # L1+VMIM defaults from best focused optimization config.
    p.add_argument("--conda-env", type=str, default="jaxili")
    p.add_argument("--vmim-compressor-dim", type=int, default=64)
    p.add_argument("--vmim-compressor-hidden", type=str, default="768,768")
    p.add_argument("--vmim-compressor-nf-layers", type=int, default=10)
    p.add_argument("--vmim-compressor-nf-hidden", type=int, default=384)
    p.add_argument("--vmim-compressor-input-clip", type=float, default=6.0)
    p.add_argument("--vmim-compressor-steps", type=int, default=12000)
    p.add_argument("--vmim-compressor-save-every", type=int, default=500)
    p.add_argument("--vmim-compressor-batch-size", type=int, default=128)
    p.add_argument("--vmim-compressor-lr", type=float, default=3e-4)
    p.add_argument("--vmim-flow-steps", type=int, default=12000)
    p.add_argument("--vmim-batch-size", type=int, default=256)
    p.add_argument("--vmim-nvp-layers", type=int, default=4)
    p.add_argument("--vmim-nvp-hidden", type=int, default=128)
    p.add_argument("--vmim-weight-decay", type=float, default=1e-4)
    p.add_argument("--vmim-grad-clip", type=float, default=1.0)
    p.add_argument("--vmim-summary-clip-value", type=float, default=0.0)

    return p.parse_args()


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
            cmd = [str(x) for x in job.command] + ["--cuda-visible-devices", str(gpu_id)]
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
                        "returncode": rc,
                        "seconds": dt,
                        "log": str(job.log_path),
                        "cmd": cmd,
                    }
                )
            q.task_done()

    workers = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    return results


def require_success(results: List[Dict[str, object]], context: str) -> None:
    failures = [r for r in results if int(r.get("returncode", 1)) != 0]
    if not failures:
        return
    first = failures[0]
    raise RuntimeError(
        f"{context} failed for {len(failures)}/{len(results)} jobs. "
        f"First failure: {first.get('name')} (log: {first.get('log')})"
    )


def summarize_posteriors(out_root: Path) -> None:
    rows: List[Dict[str, object]] = []
    for npy in sorted((out_root / "posteriors").glob("*.npy")):
        s = np.load(npy)
        rows.append(
            {
                "file": npy.name,
                "n_samples": int(s.shape[0]),
                "std_sum": float(np.sum(np.std(s, axis=0))),
                "bias_l2": float(np.linalg.norm(np.mean(s, axis=0) - TRUTH)),
            }
        )
    (out_root / "posterior_summary.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )


def _cnn_comp_paths(save_dir: Path, map_kind: str, nbins: int, steps: int) -> Dict[str, str]:
    base = (
        save_dir
        / "vmim"
        / map_kind
        / "sigma_0.26"
        / "gal_density_30"
        / f"bin_{nbins}"
    )
    return {
        "params": str(base / f"params_nd_compressor_batch{steps}.pkl"),
        "state": str(base / f"opt_state_resnet_batch{steps}.pkl"),
    }


def _vmim_comp_paths(save_dir: Path, map_kind: str, nbins: int) -> Dict[str, str]:
    base = (
        save_dir
        / "vmim_l1"
        / map_kind
        / "sigma_0.26"
        / "gal_density_30"
        / f"bin_{nbins}"
    )
    return {
        "params": str(base / "params_nd_compressor_best.pkl"),
        "state": str(base / "opt_state_resnet_best.pkl"),
    }


def _ensure_paths_exist(paths: Dict[str, str], label: str) -> None:
    for kind, pstr in paths.items():
        p = Path(pstr)
        if not p.exists():
            raise FileNotFoundError(f"Missing {label} {kind} file: {p}")


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    script_dir = repo_root / "scripts" / "sbi"
    cnn_train_script = str(script_dir / "npe_cnn_nbody_tomo.py")
    cnn_eval_script = str(script_dir / "npe_cnn_jaxili_nbody_tomo.py")
    l1_scripts = {
        "legacy": str(script_dir / "npe_l1norm_nbody_tomo.py"),
        "jaxili": str(script_dir / "npe_l1norm_jaxili_nbody_tomo.py"),
    }
    l1_script = l1_scripts[args.l1_estimator]
    vmim_script = str(script_dir / "npe_l1vmim_jaxili_nbody_tomo.py")

    gpus = _csv_tokens(args.gpus)
    if not gpus:
        raise ValueError("--gpus cannot be empty.")
    seeds = _csv_ints(args.seeds)
    if not seeds:
        raise ValueError("--seeds cannot be empty.")
    methods = _csv_tokens(args.methods)
    if not methods:
        raise ValueError("--methods cannot be empty.")
    if not set(methods).issubset({"cnn", "l1", "l1vmim"}):
        raise ValueError("--methods must be a subset of {cnn,l1,l1vmim}.")
    variants = _csv_tokens(args.variants)
    if not variants:
        raise ValueError("--variants cannot be empty.")
    invalid_variants = [v for v in variants if v not in ALL_VARIANTS]
    if invalid_variants:
        raise ValueError(
            f"Unknown variants: {invalid_variants}. Allowed: {sorted(ALL_VARIANTS)}"
        )

    manifest = {
        "repo_root": str(repo_root),
        "output_root": str(out_root),
        "mode": "nobnt_only",
        "methods": methods,
        "variants": variants,
        "seeds": seeds,
        "gpus": gpus,
        "dataset": "NbodyCosmogridDatasetTomo/grid_20deg_160px",
        "npe_samples": args.npe_samples,
        "cnn": {
            "compressor_dim": args.cnn_compressor_dim,
            "compressor_steps": args.cnn_compressor_steps,
            "compressor_save_every": args.cnn_compressor_save_every,
            "compressor_conv_channels": args.cnn_compressor_conv_channels,
            "compressor_dense_width": args.cnn_compressor_dense_width,
            "compressor_pool_window": args.cnn_compressor_pool_window,
            "compressor_pool_stride": args.cnn_compressor_pool_stride,
            "flow_steps": args.cnn_flow_steps,
            "batch_size": args.cnn_batch_size,
            "nvp_layers": args.cnn_nvp_layers,
            "nvp_hidden": args.cnn_nvp_hidden,
            "weight_decay": args.cnn_weight_decay,
            "grad_clip": args.cnn_grad_clip,
            "ds_batch_size": args.cnn_ds_batch_size,
            "standardize_summary": bool(args.cnn_standardize_summary),
            "summary_clip_value": args.cnn_summary_clip_value,
        },
        "l1": {
            "estimator": args.l1_estimator,
            "pca_components": args.l1_pca_components,
            "n_scales": args.n_scales,
            "l1_nbins": args.l1_nbins,
            "l1_min_snr": args.l1_min_snr,
            "l1_max_snr": args.l1_max_snr,
            "flow_steps": args.l1_flow_steps,
            "batch_size": args.l1_batch_size,
            "nvp_layers": args.l1_nvp_layers,
            "nvp_hidden": args.l1_nvp_hidden,
            "weight_decay": args.l1_weight_decay,
            "grad_clip": args.l1_grad_clip,
            "ds_batch_size": args.l1_ds_batch_size,
            "summary_transform": args.l1_summary_transform,
            "clip_value": args.l1_clip_value,
            "learning_rate": args.l1_learning_rate,
            "epochs": args.l1_epochs,
        },
        "l1vmim": {
            "conda_env": args.conda_env,
            "compressor_dim": args.vmim_compressor_dim,
            "compressor_hidden": args.vmim_compressor_hidden,
            "compressor_nf_layers": args.vmim_compressor_nf_layers,
            "compressor_nf_hidden": args.vmim_compressor_nf_hidden,
            "compressor_input_clip": args.vmim_compressor_input_clip,
            "compressor_steps": args.vmim_compressor_steps,
            "compressor_save_every": args.vmim_compressor_save_every,
            "compressor_batch_size": args.vmim_compressor_batch_size,
            "compressor_lr": args.vmim_compressor_lr,
            "flow_steps": args.vmim_flow_steps,
            "batch_size": args.vmim_batch_size,
            "nvp_layers": args.vmim_nvp_layers,
            "nvp_hidden": args.vmim_nvp_hidden,
            "weight_decay": args.vmim_weight_decay,
            "grad_clip": args.vmim_grad_clip,
            "summary_clip_value": args.vmim_summary_clip_value,
        },
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    all_results: List[Dict[str, object]] = []
    cnn_compressors: Dict[str, Dict[str, str]] = {}
    vmim_compressors: Dict[str, Dict[str, str]] = {}

    # Train CNN compressors once per variant.
    if "cnn" in methods:
        for variant in variants:
            cfg = ALL_VARIANTS[variant]
            cache_dir = out_root / "cache" / f"cnn_{variant}"
            save_dir = out_root / "cnn_compressor" / variant
            cmd = [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                args.conda_env,
                "python",
                cnn_train_script,
                "--no-wandb",
                "--map-kind",
                args.map_kind,
                "--seed",
                str(args.compressor_seed),
                "--tfds-name",
                str(cfg["tfds"]),
                "--field-size",
                str(cfg["field_size"]),
                "--field-npix",
                str(cfg["field_npix"]),
                "--nbins",
                str(cfg["nbins"]),
                "--tomo-bin-indices",
                str(cfg["tomo_bins"]),
                "--cache-dir",
                str(cache_dir),
                "--save-dir",
                str(save_dir),
                "--train-compressor",
                "--compressor-dim",
                str(args.cnn_compressor_dim),
                "--compressor-steps",
                str(args.cnn_compressor_steps),
                "--compressor-save-every",
                str(args.cnn_compressor_save_every),
                "--total-steps",
                "1",
                "--save-every",
                "1",
                "--no-sample",
            ]
            if not args.cnn_standardize_summary:
                cmd.append("--no-standardize-summary")
            train_job = Job(
                name=f"cnn_train::{variant}",
                command=cmd,
                log_path=out_root / "logs" / f"cnn_train_{variant}.log",
            )
            result = run_jobs_parallel([train_job], gpus, repo_root, args.dry_run)
            all_results.extend(result)
            require_success(result, f"CNN compressor training ({variant})")
            comp_paths = _cnn_comp_paths(
                save_dir=save_dir,
                map_kind=args.map_kind,
                nbins=int(cfg["nbins"]),
                steps=args.cnn_compressor_steps,
            )
            if not args.dry_run:
                _ensure_paths_exist(comp_paths, f"CNN compressor ({variant})")
            cnn_compressors[variant] = comp_paths

    # Train L1+VMIM compressors once per variant.
    if "l1vmim" in methods:
        for variant in variants:
            cfg = ALL_VARIANTS[variant]
            cache_dir = out_root / "cache" / f"l1vmim_{variant}"
            save_dir = out_root / "l1vmim_compressor" / variant
            cmd = [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                args.conda_env,
                "python",
                vmim_script,
                "--no-wandb",
                "--map-kind",
                args.map_kind,
                "--seed",
                str(args.compressor_seed),
                "--tfds-name",
                str(cfg["tfds"]),
                "--field-size",
                str(cfg["field_size"]),
                "--field-npix",
                str(cfg["field_npix"]),
                "--nbins",
                str(cfg["nbins"]),
                "--tomo-bin-indices",
                str(cfg["tomo_bins"]),
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
                str(args.vmim_compressor_input_clip),
                "--compressor-dim",
                str(args.vmim_compressor_dim),
                "--compressor-hidden",
                args.vmim_compressor_hidden,
                "--compressor-vmim-nf-layers",
                str(args.vmim_compressor_nf_layers),
                "--compressor-vmim-nf-hidden",
                str(args.vmim_compressor_nf_hidden),
                "--compressor-steps",
                str(args.vmim_compressor_steps),
                "--compressor-save-every",
                str(args.vmim_compressor_save_every),
                "--compressor-batch-size",
                str(args.vmim_compressor_batch_size),
                "--compressor-lr",
                str(args.vmim_compressor_lr),
                "--total-steps",
                "1",
                "--epochs",
                "2",
                "--save-every",
                "1",
                "--no-sample",
            ]
            train_job = Job(
                name=f"l1vmim_train::{variant}",
                command=cmd,
                log_path=out_root / "logs" / f"l1vmim_train_{variant}.log",
            )
            result = run_jobs_parallel([train_job], gpus, repo_root, args.dry_run)
            all_results.extend(result)
            require_success(result, f"L1-VMIM compressor training ({variant})")
            comp_paths = _vmim_comp_paths(
                save_dir=save_dir,
                map_kind=args.map_kind,
                nbins=int(cfg["nbins"]),
            )
            if not args.dry_run:
                _ensure_paths_exist(comp_paths, f"L1-VMIM compressor ({variant})")
            vmim_compressors[variant] = comp_paths

    eval_jobs: List[Job] = []
    for variant in variants:
        cfg = ALL_VARIANTS[variant]
        for seed in seeds:
            tag = f"{variant}_nobnt_s{seed}"

            if "cnn" in methods:
                posterior_out = out_root / "posteriors" / f"cnn_{tag}.npy"
                cmd = [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    args.conda_env,
                    "python",
                    cnn_eval_script,
                    "--no-wandb",
                    "--map-kind",
                    args.map_kind,
                    "--seed",
                    str(seed),
                    "--tfds-name",
                    str(cfg["tfds"]),
                    "--field-size",
                    str(cfg["field_size"]),
                    "--field-npix",
                    str(cfg["field_npix"]),
                    "--nbins",
                    str(cfg["nbins"]),
                    "--tomo-bin-indices",
                    str(cfg["tomo_bins"]),
                    "--cache-dir",
                    str(out_root / "cache" / f"cnn_{variant}"),
                    "--save-dir",
                    str(out_root / "cnn_eval" / variant / f"seed_{seed}"),
                    "--compressor-params",
                    cnn_compressors[variant]["params"],
                    "--compressor-state",
                    cnn_compressors[variant]["state"],
                    "--compressor-dim",
                    str(args.cnn_compressor_dim),
                    "--total-steps",
                    str(args.cnn_flow_steps),
                    "--save-every",
                    str(args.flow_save_every),
                    "--patience",
                    str(args.flow_patience),
                    "--batch-size",
                    str(args.cnn_batch_size),
                    "--npe-samples",
                    str(args.npe_samples),
                    "--posterior-out",
                    str(posterior_out),
                    "--ds-batch-size",
                    str(args.cnn_ds_batch_size),
                    "--summary-clip-value",
                    str(args.cnn_summary_clip_value),
                ]
                if args.cnn_standardize_summary:
                    cmd.append("--standardize-summary")
                else:
                    cmd.append("--no-standardize-summary")
                if args.plot:
                    cmd.extend(
                        [
                            "--plot",
                            "--figure-out",
                            str(out_root / "figures" / f"cnn_{tag}.pdf"),
                        ]
                    )
                eval_jobs.append(
                    Job(
                        name=f"cnn_eval::{tag}",
                        command=cmd,
                        log_path=out_root / "logs" / f"cnn_eval_{tag}.log",
                    )
                )

            if "l1" in methods:
                posterior_out = out_root / "posteriors" / f"l1_{tag}.npy"
                cmd = [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    args.conda_env,
                    "python",
                    l1_script,
                    "--no-wandb",
                    "--map-kind",
                    args.map_kind,
                    "--seed",
                    str(seed),
                    "--tfds-name",
                    str(cfg["tfds"]),
                    "--field-size",
                    str(cfg["field_size"]),
                    "--field-npix",
                    str(cfg["field_npix"]),
                    "--nbins",
                    str(cfg["nbins"]),
                    "--tomo-bin-indices",
                    str(cfg["tomo_bins"]),
                    "--cache-dir",
                    str(out_root / "cache" / f"l1_{variant}"),
                    "--save-dir",
                    str(out_root / "l1_eval" / variant / f"seed_{seed}"),
                    "--n-scales",
                    str(args.n_scales),
                    "--l1-nbins",
                    str(args.l1_nbins),
                    "--l1-min-snr",
                    str(args.l1_min_snr),
                    "--l1-max-snr",
                    str(args.l1_max_snr),
                    "--pca-components",
                    str(args.l1_pca_components),
                    "--total-steps",
                    str(args.l1_flow_steps),
                    "--save-every",
                    str(args.flow_save_every),
                    "--patience",
                    str(args.flow_patience),
                    "--batch-size",
                    str(args.l1_batch_size),
                    "--npe-samples",
                    str(args.npe_samples),
                    "--posterior-out",
                    str(posterior_out),
                    "--ds-batch-size",
                    str(args.l1_ds_batch_size),
                ]
                # NVP architecture args are only for legacy estimator, not jaxili
                if args.l1_estimator == "legacy":
                    cmd.extend([
                        "--nvp-layers",
                        str(args.l1_nvp_layers),
                        "--nvp-hidden",
                        str(args.l1_nvp_hidden),
                        "--weight-decay",
                        str(args.l1_weight_decay),
                        "--grad-clip",
                        str(args.l1_grad_clip),
                    ])
                elif args.l1_estimator == "jaxili":
                    # jaxili-specific hyperparameters
                    cmd.extend([
                        "--summary-transform",
                        str(args.l1_summary_transform),
                        "--clip-value",
                        str(args.l1_clip_value),
                        "--learning-rate",
                        str(args.l1_learning_rate),
                    ])
                    # Use --epochs if set, otherwise --total-steps will be used
                    if args.l1_epochs is not None:
                        cmd.extend([
                            "--epochs",
                            str(args.l1_epochs),
                        ])
                if args.plot:
                    cmd.extend(
                        [
                            "--plot",
                            "--figure-out",
                            str(out_root / "figures" / f"l1_{tag}.pdf"),
                        ]
                    )
                eval_jobs.append(
                    Job(
                        name=f"l1_eval::{tag}",
                        command=cmd,
                        log_path=out_root / "logs" / f"l1_eval_{tag}.log",
                    )
                )

            if "l1vmim" in methods:
                posterior_out = out_root / "posteriors" / f"l1vmim_{tag}.npy"
                cmd = [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    args.conda_env,
                    "python",
                    vmim_script,
                    "--no-wandb",
                    "--map-kind",
                    args.map_kind,
                    "--seed",
                    str(seed),
                    "--tfds-name",
                    str(cfg["tfds"]),
                    "--field-size",
                    str(cfg["field_size"]),
                    "--field-npix",
                    str(cfg["field_npix"]),
                    "--nbins",
                    str(cfg["nbins"]),
                    "--tomo-bin-indices",
                    str(cfg["tomo_bins"]),
                    "--n-scales",
                    str(args.n_scales),
                    "--l1-nbins",
                    str(args.l1_nbins),
                    "--l1-min-snr",
                    str(args.l1_min_snr),
                    "--l1-max-snr",
                    str(args.l1_max_snr),
                    "--cache-dir",
                    str(out_root / "cache" / f"l1vmim_{variant}"),
                    "--save-dir",
                    str(out_root / "l1vmim_eval" / variant / f"seed_{seed}"),
                    "--compressor-log1p-input",
                    "--compressor-input-standardize",
                    "--compressor-input-clip",
                    str(args.vmim_compressor_input_clip),
                    "--compressor-dim",
                    str(args.vmim_compressor_dim),
                    "--compressor-hidden",
                    args.vmim_compressor_hidden,
                    "--compressor-vmim-nf-layers",
                    str(args.vmim_compressor_nf_layers),
                    "--compressor-vmim-nf-hidden",
                    str(args.vmim_compressor_nf_hidden),
                    "--compressor-params",
                    vmim_compressors[variant]["params"],
                    "--compressor-state",
                    vmim_compressors[variant]["state"],
                    "--total-steps",
                    str(args.vmim_flow_steps),
                    "--save-every",
                    str(args.flow_save_every),
                    "--patience",
                    str(args.flow_patience),
                    "--batch-size",
                    str(args.vmim_batch_size),
                    "--nvp-layers",
                    str(args.vmim_nvp_layers),
                    "--nvp-hidden",
                    str(args.vmim_nvp_hidden),
                    "--weight-decay",
                    str(args.vmim_weight_decay),
                    "--grad-clip",
                    str(args.vmim_grad_clip),
                    "--npe-samples",
                    str(args.npe_samples),
                    "--posterior-out",
                    str(posterior_out),
                    "--summary-clip-value",
                    str(args.vmim_summary_clip_value),
                ]
                if args.plot:
                    cmd.extend(
                        [
                            "--plot",
                            "--figure-out",
                            str(out_root / "figures" / f"l1vmim_{tag}.pdf"),
                        ]
                    )
                eval_jobs.append(
                    Job(
                        name=f"l1vmim_eval::{tag}",
                        command=cmd,
                        log_path=out_root / "logs" / f"l1vmim_eval_{tag}.log",
                    )
                )

    if eval_jobs:
        eval_results = run_jobs_parallel(eval_jobs, gpus, repo_root, args.dry_run)
        all_results.extend(eval_results)
        require_success(eval_results, "No-BNT cross-correlation evaluation matrix")

    (out_root / "job_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    summarize_posteriors(out_root)
    print(f"Completed no-BNT cross-correlation study orchestration. Artifacts in: {out_root}")


if __name__ == "__main__":
    main()
