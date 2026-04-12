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


@dataclass
class Job:
    name: str
    command: List[str]
    log_path: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rigorous BNT tomo4 study runner")
    p.add_argument("--repo-root", type=str, default=DEFAULT_REPO_ROOT)
    p.add_argument("--gpus", type=str, default="0,1,2")
    p.add_argument("--seeds", type=str, default="41,42,43")
    p.add_argument("--map-kind", type=str, default="nbody")
    p.add_argument(
        "--conditions",
        type=str,
        default="nobnt,bnt",
        help="Comma-separated conditions from {nobnt,bnt}",
    )

    p.add_argument("--compressor-steps-cnn", type=int, default=20000)
    p.add_argument("--compressor-save-every-cnn", type=int, default=2000)
    p.add_argument("--compressor-steps-vmim", type=int, default=12000)
    p.add_argument("--compressor-save-every-vmim", type=int, default=500)
    p.add_argument("--compressor-dim-vmim", type=int, default=40)
    p.add_argument("--compressor-hidden-vmim", type=str, default="512,512")

    p.add_argument("--flow-steps-cnn-l1", type=int, default=5000)
    p.add_argument("--flow-steps-vmim", type=int, default=12000)
    p.add_argument("--npe-samples", type=int, default=100000)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--l1-ds-batch-size", type=int, default=96)

    p.add_argument("--n-scales", type=int, default=5)
    p.add_argument("--l1-nbins", type=int, default=40)
    p.add_argument("--l1-min-snr", type=float, default=-13.0)
    p.add_argument("--l1-max-snr", type=float, default=13.0)

    p.add_argument(
        "--output-root",
        type=str,
        default="/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/bnt_tomo4_study",
    )
    p.add_argument("--plot", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_jobs_parallel(
    jobs: List[Job], gpus: List[str], cwd: Path, dry_run: bool = False
) -> List[Dict[str, object]]:
    q: queue.Queue[Job] = queue.Queue()
    for j in jobs:
        q.put(j)

    results: List[Dict[str, object]] = []
    lock = threading.Lock()

    def worker(gpu_id: str):
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
                        cmd, cwd=str(cwd), stdout=logf, stderr=subprocess.STDOUT
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

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def require_success(results: List[Dict[str, object]], context: str) -> None:
    failures = [r for r in results if int(r.get("returncode", 1)) != 0]
    if failures:
        first = failures[0]
        raise RuntimeError(
            f"{context} failed for {len(failures)}/{len(results)} jobs. "
            f"First failure: {first.get('name')} (log: {first.get('log')})"
        )


def summarize_posteriors(out_root: Path) -> None:
    truth = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493], dtype=np.float64)
    rows = []
    for npy in sorted((out_root / "posteriors").glob("*.npy")):
        s = np.load(npy)
        rows.append(
            {
                "file": npy.name,
                "n_samples": int(s.shape[0]),
                "std_sum": float(np.sum(np.std(s, axis=0))),
                "bias_l2": float(np.linalg.norm(np.mean(s, axis=0) - truth)),
            }
        )
    (out_root / "posterior_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    script_dir = repo_root / "scripts" / "sbi"
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise ValueError("--gpus cannot be empty.")
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("--seeds cannot be empty.")
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    if not conditions or not set(conditions).issubset({"nobnt", "bnt"}):
        raise ValueError("--conditions must be subset of {nobnt,bnt}.")

    cnn_train_script = str(script_dir / "npe_cnn_nbody_tomo.py")
    cnn_eval_script = str(script_dir / "npe_cnn_jaxili_nbody_tomo.py")
    l1_script = str(script_dir / "npe_l1norm_jaxili_nbody_tomo.py")
    vmim_script = str(script_dir / "npe_l1vmim_jaxili_nbody_tomo.py")

    manifest = {
        "seeds": seeds,
        "conditions": conditions,
        "gpus": gpus,
        "dataset": "NbodyCosmogridDatasetTomo/grid_20deg_160px",
        "field_size": 20,
        "field_npix": 160,
        "tomo_bins": "1,2,3,4",
        "flow_steps_cnn_l1": args.flow_steps_cnn_l1,
        "flow_steps_vmim": args.flow_steps_vmim,
        "compressor_steps_cnn": args.compressor_steps_cnn,
        "compressor_steps_vmim": args.compressor_steps_vmim,
        "repo_root": str(repo_root),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    all_results: List[Dict[str, object]] = []
    cnn_compressors: Dict[str, Dict[str, str]] = {}
    vmim_compressors: Dict[str, Dict[str, str]] = {}

    # Compressor training once per condition for CNN and L1-VMIM.
    for cond in conditions:
        apply_bnt = cond == "bnt"
        cond_flag = ["--apply-bnt"] if apply_bnt else []

        cnn_save = out_root / "cnn_compressor" / cond
        cnn_cache = out_root / "cache" / f"cnn_tomo4_20deg160_{cond}"
        cnn_train_job = Job(
            name=f"cnn_train::{cond}",
            log_path=out_root / "logs" / f"cnn_train_{cond}.log",
            command=[
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                "jaxili",
                "python",
                cnn_train_script,
                "--no-wandb",
                "--map-kind",
                args.map_kind,
                "--tfds-name",
                "NbodyCosmogridDatasetTomo/grid_20deg_160px",
                "--field-size",
                "20",
                "--field-npix",
                "160",
                "--nbins",
                "4",
                "--tomo-bin-indices",
                "1,2,3,4",
                "--cache-dir",
                str(cnn_cache),
                "--save-dir",
                str(cnn_save),
                "--train-compressor",
                "--compressor-steps",
                str(args.compressor_steps_cnn),
                "--compressor-save-every",
                str(args.compressor_save_every_cnn),
                "--total-steps",
                "1",
                "--save-every",
                "1",
                "--no-sample",
            ]
            + cond_flag,
        )
        all_results.extend(run_jobs_parallel([cnn_train_job], gpus, repo_root, args.dry_run))
        require_success(all_results[-1:], f"CNN compressor training ({cond})")
        cnn_base = (
            cnn_save
            / "vmim"
            / args.map_kind
            / "sigma_0.26"
            / "gal_density_30"
            / "bin_4"
        )
        cnn_compressors[cond] = {
            "params": str(cnn_base / f"params_nd_compressor_batch{args.compressor_steps_cnn}.pkl"),
            "state": str(cnn_base / f"opt_state_resnet_batch{args.compressor_steps_cnn}.pkl"),
        }

        vmim_save = out_root / "l1_vmim_compressor" / cond
        vmim_cache = out_root / "cache" / f"l1vmim_tomo4_20deg160_{cond}"
        vmim_train_job = Job(
            name=f"l1vmim_train::{cond}",
            log_path=out_root / "logs" / f"l1vmim_train_{cond}.log",
            command=[
                "conda",
                "run",
                "-n",
                "jaxili",
                "python",
                vmim_script,
                "--no-wandb",
                "--map-kind",
                args.map_kind,
                "--tfds-name",
                "NbodyCosmogridDatasetTomo/grid_20deg_160px",
                "--field-size",
                "20",
                "--field-npix",
                "160",
                "--nbins",
                "4",
                "--tomo-bin-indices",
                "1,2,3,4",
                "--n-scales",
                str(args.n_scales),
                "--l1-nbins",
                str(args.l1_nbins),
                "--l1-min-snr",
                str(args.l1_min_snr),
                "--l1-max-snr",
                str(args.l1_max_snr),
                "--cache-dir",
                str(vmim_cache),
                "--save-dir",
                str(vmim_save),
                "--train-compressor",
                "--compressor-log1p-input",
                "--compressor-input-standardize",
                "--compressor-input-clip",
                "6",
                "--compressor-dim",
                str(args.compressor_dim_vmim),
                "--compressor-hidden",
                args.compressor_hidden_vmim,
                "--compressor-vmim-nf-layers",
                "8",
                "--compressor-vmim-nf-hidden",
                "256",
                "--compressor-steps",
                str(args.compressor_steps_vmim),
                "--compressor-save-every",
                str(args.compressor_save_every_vmim),
                "--compressor-batch-size",
                "128",
                "--compressor-lr",
                "3e-4",
                "--total-steps",
                "1",
                "--epochs",
                "2",
                "--save-every",
                "1",
                "--no-sample",
            ]
            + cond_flag,
        )
        all_results.extend(run_jobs_parallel([vmim_train_job], gpus, repo_root, args.dry_run))
        require_success(all_results[-1:], f"L1-VMIM compressor training ({cond})")
        vmim_base = (
            vmim_save
            / "vmim_l1"
            / args.map_kind
            / "sigma_0.26"
            / "gal_density_30"
            / "bin_4"
        )
        vmim_compressors[cond] = {
            "params": str(vmim_base / "params_nd_compressor_best.pkl"),
            "state": str(vmim_base / "opt_state_resnet_best.pkl"),
        }

    eval_jobs: List[Job] = []
    for cond in conditions:
        apply_bnt = cond == "bnt"
        cond_flag = ["--apply-bnt"] if apply_bnt else []
        for seed in seeds:
            tag = f"tomo4_20deg160_{cond}_s{seed}"

            cnn_eval_save = out_root / "cnn_eval" / cond / f"seed_{seed}"
            cnn_cache = out_root / "cache" / f"cnn_tomo4_20deg160_{cond}"
            cnn_post = out_root / "posteriors" / f"cnn_{tag}.npy"
            cnn_cmd = [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                "jaxili",
                "python",
                cnn_eval_script,
                "--no-wandb",
                "--map-kind",
                args.map_kind,
                "--seed",
                str(seed),
                "--tfds-name",
                "NbodyCosmogridDatasetTomo/grid_20deg_160px",
                "--field-size",
                "20",
                "--field-npix",
                "160",
                "--nbins",
                "4",
                "--tomo-bin-indices",
                "1,2,3,4",
                "--cache-dir",
                str(cnn_cache),
                "--save-dir",
                str(cnn_eval_save),
                "--compressor-params",
                cnn_compressors[cond]["params"],
                "--compressor-state",
                cnn_compressors[cond]["state"],
                "--total-steps",
                str(args.flow_steps_cnn_l1),
                "--save-every",
                str(args.save_every),
                "--patience",
                str(args.patience),
                "--batch-size",
                str(args.batch_size),
                "--npe-samples",
                str(args.npe_samples),
                "--posterior-out",
                str(cnn_post),
                "--ds-batch-size",
                "500",
            ] + cond_flag
            if args.plot:
                cnn_cmd.extend(
                    [
                        "--plot",
                        "--figure-out",
                        str(out_root / "figures" / f"cnn_{tag}.pdf"),
                    ]
                )
            eval_jobs.append(
                Job(
                    name=f"cnn_eval::{tag}",
                    log_path=out_root / "logs" / f"cnn_eval_{tag}.log",
                    command=cnn_cmd,
                )
            )

            l1_eval_save = out_root / "l1_eval" / cond / f"seed_{seed}"
            l1_cache = out_root / "cache" / f"l1_tomo4_20deg160_{cond}"
            l1_post = out_root / "posteriors" / f"l1_{tag}.npy"
            l1_cmd = [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                "jaxili",
                "python",
                l1_script,
                "--no-wandb",
                "--map-kind",
                args.map_kind,
                "--seed",
                str(seed),
                "--tfds-name",
                "NbodyCosmogridDatasetTomo/grid_20deg_160px",
                "--field-size",
                "20",
                "--field-npix",
                "160",
                "--nbins",
                "4",
                "--tomo-bin-indices",
                "1,2,3,4",
                "--cache-dir",
                str(l1_cache),
                "--save-dir",
                str(l1_eval_save),
                "--n-scales",
                str(args.n_scales),
                "--l1-nbins",
                str(args.l1_nbins),
                "--l1-min-snr",
                str(args.l1_min_snr),
                "--l1-max-snr",
                str(args.l1_max_snr),
                "--pca-components",
                "0",
                "--total-steps",
                str(args.flow_steps_cnn_l1),
                "--save-every",
                str(args.save_every),
                "--patience",
                str(args.patience),
                "--batch-size",
                str(args.batch_size),
                "--npe-samples",
                str(args.npe_samples),
                "--posterior-out",
                str(l1_post),
                "--ds-batch-size",
                str(args.l1_ds_batch_size),
            ] + cond_flag
            if args.plot:
                l1_cmd.extend(
                    [
                        "--plot",
                        "--figure-out",
                        str(out_root / "figures" / f"l1_{tag}.pdf"),
                    ]
                )
            eval_jobs.append(
                Job(
                    name=f"l1_eval::{tag}",
                    log_path=out_root / "logs" / f"l1_eval_{tag}.log",
                    command=l1_cmd,
                )
            )

            vmim_eval_save = out_root / "l1_vmim_eval" / cond / f"seed_{seed}"
            vmim_cache = out_root / "cache" / f"l1vmim_tomo4_20deg160_{cond}"
            vmim_post = out_root / "posteriors" / f"l1vmim_{tag}.npy"
            vmim_cmd = [
                "conda",
                "run",
                "-n",
                "jaxili",
                "python",
                vmim_script,
                "--no-wandb",
                "--map-kind",
                args.map_kind,
                "--seed",
                str(seed),
                "--tfds-name",
                "NbodyCosmogridDatasetTomo/grid_20deg_160px",
                "--field-size",
                "20",
                "--field-npix",
                "160",
                "--nbins",
                "4",
                "--tomo-bin-indices",
                "1,2,3,4",
                "--n-scales",
                str(args.n_scales),
                "--l1-nbins",
                str(args.l1_nbins),
                "--l1-min-snr",
                str(args.l1_min_snr),
                "--l1-max-snr",
                str(args.l1_max_snr),
                "--cache-dir",
                str(vmim_cache),
                "--save-dir",
                str(vmim_eval_save),
                "--compressor-log1p-input",
                "--compressor-input-standardize",
                "--compressor-input-clip",
                "6",
                "--compressor-dim",
                str(args.compressor_dim_vmim),
                "--compressor-hidden",
                args.compressor_hidden_vmim,
                "--compressor-vmim-nf-layers",
                "8",
                "--compressor-vmim-nf-hidden",
                "256",
                "--compressor-params",
                vmim_compressors[cond]["params"],
                "--compressor-state",
                vmim_compressors[cond]["state"],
                "--total-steps",
                str(args.flow_steps_vmim),
                "--save-every",
                str(args.save_every),
                "--patience",
                str(args.patience),
                "--batch-size",
                str(args.batch_size),
                "--npe-samples",
                str(args.npe_samples),
                "--posterior-out",
                str(vmim_post),
                "--summary-clip-value",
                "0",
            ] + cond_flag
            if args.plot:
                vmim_cmd.extend(
                    [
                        "--plot",
                        "--figure-out",
                        str(out_root / "figures" / f"l1vmim_{tag}.pdf"),
                    ]
                )
            eval_jobs.append(
                Job(
                    name=f"l1vmim_eval::{tag}",
                    log_path=out_root / "logs" / f"l1vmim_eval_{tag}.log",
                    command=vmim_cmd,
                )
            )

    all_results.extend(run_jobs_parallel(eval_jobs, gpus, repo_root, args.dry_run))
    require_success(all_results[-len(eval_jobs):], "BNT evaluation sweep")
    (out_root / "job_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    summarize_posteriors(out_root)
    print(f"Completed BNT tomo4 study orchestration. Artifacts in: {out_root}")


if __name__ == "__main__":
    main()
