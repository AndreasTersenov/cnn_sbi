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


@dataclass
class Job:
    name: str
    command: List[str]
    log_path: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Systematic L1-VMIM benchmark runner")
    p.add_argument("--repo-root", type=str, default="/mnt/home/tersenov/software/cnn_sbi")
    p.add_argument("--gpus", type=str, default="0,1,2")
    p.add_argument("--map-kind", type=str, default="nbody")
    p.add_argument(
        "--variants",
        type=str,
        default="bin3_20deg160,tomo4_20deg160",
        help="Comma-separated variants to run",
    )
    p.add_argument("--seeds", type=str, default="42")

    p.add_argument("--compressor-dim", type=int, default=6)
    p.add_argument("--compressor-hidden", type=str, default="256,256")
    p.add_argument("--compressor-steps", type=int, default=5000)
    p.add_argument("--compressor-save-every", type=int, default=500)
    p.add_argument("--compressor-batch-size", type=int, default=128)
    p.add_argument("--compressor-lr", type=float, default=1e-4)
    p.add_argument(
        "--compressor-log1p-input",
        dest="compressor_log1p_input",
        action="store_true",
        help="Use log1p on raw L1 vectors before VMIM compressor (recommended)",
    )
    p.add_argument(
        "--no-compressor-log1p-input",
        dest="compressor_log1p_input",
        action="store_false",
        help="Disable log1p transform before VMIM compressor",
    )
    p.set_defaults(compressor_log1p_input=True)

    p.add_argument("--flow-steps", type=int, default=500)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr-init", type=float, default=1e-3)
    p.add_argument("--lr-end", type=float, default=1e-5)
    p.add_argument("--npe-samples", type=int, default=50000)
    p.add_argument("--summary-clip-value", type=float, default=5.0)

    p.add_argument("--n-scales", type=int, default=5)
    p.add_argument("--l1-nbins", type=int, default=40)
    p.add_argument("--l1-min-snr", type=float, default=-13.0)
    p.add_argument("--l1-max-snr", type=float, default=13.0)

    p.add_argument(
        "--output-root",
        type=str,
        default="/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/l1_vmim_runs/systematic",
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
                output = f"[dry-run] {' '.join(cmd)}\n"
                job.log_path.write_text(output, encoding="utf-8")
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


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    script_path = repo_root / "scripts" / "sbi" / "npe_l1vmim_nbody_tomo.py"
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise ValueError("--gpus cannot be empty.")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("--seeds cannot be empty.")

    all_variants = {
        "tomo4_20deg160": {
            "tfds": "NbodyCosmogridDatasetTomo/grid_20deg_160px",
            "field_size": 20,
            "field_npix": 160,
            "nbins": 4,
            "tomo_bins": "1,2,3,4",
            "cache": str(repo_root / "scripts" / "sbi" / "l1_jax_runs" / "cache_tomo4_20deg160_snr13"),
        },
        "bin3_20deg160": {
            "tfds": "NbodyCosmogridDatasetTomo/grid_20deg_160px",
            "field_size": 20,
            "field_npix": 160,
            "nbins": 1,
            "tomo_bins": "3",
            "cache": str(repo_root / "scripts" / "sbi" / "l1_jax_runs" / "cache_bin3_20deg160_snr13"),
        },
        "bin1_20deg160": {
            "tfds": "NbodyCosmogridDatasetTomo/grid_20deg_160px",
            "field_size": 20,
            "field_npix": 160,
            "nbins": 1,
            "tomo_bins": "1",
            "cache": str(repo_root / "scripts" / "sbi" / "l1_jax_runs" / "cache_bin1_20deg160_snr13"),
        },
        "bin2_20deg160": {
            "tfds": "NbodyCosmogridDatasetTomo/grid_20deg_160px",
            "field_size": 20,
            "field_npix": 160,
            "nbins": 1,
            "tomo_bins": "2",
            "cache": str(repo_root / "scripts" / "sbi" / "l1_jax_runs" / "cache_bin2_20deg160_snr13"),
        },
    }

    selected = [v.strip() for v in args.variants.split(",") if v.strip()]
    if not selected:
        raise ValueError("--variants cannot be empty.")
    invalid = [v for v in selected if v not in all_variants]
    if invalid:
        raise ValueError(f"Unknown variants: {invalid}. Allowed: {sorted(all_variants)}")

    manifest = {
        "gpus": gpus,
        "seeds": seeds,
        "variants": selected,
        "compressor_dim": args.compressor_dim,
        "compressor_hidden": args.compressor_hidden,
        "compressor_steps": args.compressor_steps,
        "flow_steps": args.flow_steps,
        "npe_samples": args.npe_samples,
        "compressor_log1p_input": bool(args.compressor_log1p_input),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    jobs: List[Job] = []
    for variant_name in selected:
        v = all_variants[variant_name]
        for seed in seeds:
            tag = f"{variant_name}_s{seed}"
            save_dir = out_root / "save_params" / variant_name / f"seed_{seed}"
            posterior_out = out_root / "posteriors" / f"l1_vmim_{tag}.npy"

            cmd = [
                "conda",
                "run",
                "-n",
                "jaxili",
                "python",
                str(script_path),
                "--no-wandb",
                "--map-kind",
                args.map_kind,
                "--seed",
                str(seed),
                "--tfds-name",
                str(v["tfds"]),
                "--field-size",
                str(v["field_size"]),
                "--field-npix",
                str(v["field_npix"]),
                "--nbins",
                str(v["nbins"]),
                "--tomo-bin-indices",
                str(v["tomo_bins"]),
                "--n-scales",
                str(args.n_scales),
                "--l1-nbins",
                str(args.l1_nbins),
                "--l1-min-snr",
                str(args.l1_min_snr),
                "--l1-max-snr",
                str(args.l1_max_snr),
                "--cache-dir",
                str(v["cache"]),
                "--save-dir",
                str(save_dir),
                "--train-compressor",
                "--compressor-dim",
                str(args.compressor_dim),
                "--compressor-hidden",
                args.compressor_hidden,
                "--compressor-steps",
                str(args.compressor_steps),
                "--compressor-save-every",
                str(args.compressor_save_every),
                "--compressor-batch-size",
                str(args.compressor_batch_size),
                "--compressor-lr",
                str(args.compressor_lr),
                "--total-steps",
                str(args.flow_steps),
                "--save-every",
                str(args.save_every),
                "--patience",
                str(args.patience),
                "--batch-size",
                str(args.batch_size),
                "--lr-init",
                str(args.lr_init),
                "--lr-end",
                str(args.lr_end),
                "--npe-samples",
                str(args.npe_samples),
                "--summary-clip-value",
                str(args.summary_clip_value),
                "--posterior-out",
                str(posterior_out),
            ]
            if args.compressor_log1p_input:
                cmd.append("--compressor-log1p-input")
            else:
                cmd.append("--no-compressor-log1p-input")
            if args.plot:
                cmd.extend(
                    [
                        "--plot",
                        "--figure-out",
                        str(out_root / "figures" / f"l1_vmim_{tag}.png"),
                    ]
                )

            jobs.append(
                Job(
                    name=f"l1_vmim::{tag}",
                    log_path=out_root / "logs" / f"l1_vmim_{tag}.log",
                    command=cmd,
                )
            )

    results = run_jobs_parallel(jobs, gpus, repo_root, dry_run=args.dry_run)
    (out_root / "job_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    truth = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493], dtype=np.float64)
    rows = []
    post_dir = out_root / "posteriors"
    if post_dir.exists():
        for npy_path in sorted(post_dir.glob("*.npy")):
            s = np.load(npy_path)
            rows.append(
                {
                    "file": npy_path.name,
                    "n_samples": int(s.shape[0]),
                    "std_sum": float(np.sum(np.std(s, axis=0))),
                    "bias_l2": float(np.linalg.norm(np.mean(s, axis=0) - truth)),
                }
            )
    (out_root / "posterior_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Completed L1-VMIM sweep orchestration. Artifacts in: {out_root}")


if __name__ == "__main__":
    main()
