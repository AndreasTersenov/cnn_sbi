#!/usr/bin/env python3
"""A3: CNN with explicit harmonic cross-channel input (10-channel, no-BNT).

Feeds the 4-auto + 6-cross harmonic cache maps directly to the CNN compressor
(--cnn-map-route harmonic) instead of the standard 4-channel TFDS auto maps.
Sweeps seeds {41, 42, 43} in parallel across GPUs.

Compressor: plain, cdim=10, 150k steps, dense=256
Flow: 50k steps

Usage
-----
conda run -n jaxili python run_a3_cnn_harm_cross.py \
    --gpus 0,1,2 --xla-mem-fraction 0.45 \
    --xla-mem-fraction-by-gpu 0:0.30,1:0.45,2:0.45 [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[5]
OUTPUT_ROOT = Path(__file__).resolve().parent
CNN_SCRIPT = str(REPO_ROOT / "scripts" / "sbi" / "npe_cnn_nbody_tomo.py")
HARM_CACHE = str(
    REPO_ROOT / "scripts" / "sbi" / "results" / "exploratory"
    / "cross_maps_campaign" / "full_sphere_cache_grid"
)

SEEDS: Tuple[int, ...] = (41, 42, 43)


@dataclass(frozen=True)
class Job:
    name: str
    command: List[str]
    log_path: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A3: CNN with harmonic cross-channel input.")
    p.add_argument("--gpus", type=str, default="0,1,2")
    p.add_argument("--xla-mem-fraction", type=float, default=0.45)
    p.add_argument(
        "--xla-mem-fraction-by-gpu", type=str, default="",
        help="Per-GPU overrides, e.g. '0:0.30,1:0.45,2:0.45'.",
    )
    p.add_argument("--conda-env", type=str, default="jaxili")
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    p.add_argument("--compressor-steps", type=int, default=150000)
    p.add_argument("--compressor-dense-width", type=int, default=256)
    p.add_argument("--flow-steps", type=int, default=50000)
    p.add_argument("--npe-samples", type=int, default=100000)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_job(
    args: argparse.Namespace,
    seed: int,
    output_root: Path,
) -> Job:
    tag = f"cnn_harmcross_nobnt_s{seed}"
    run_dir = output_root / "nobnt" / f"full_seed_{seed}"
    posterior_out = output_root / "nobnt" / "posteriors" / f"cnn_harm_cross_nobnt_s{seed}.npy"
    figure_out = output_root / "nobnt" / "figures" / f"cnn_harm_cross_nobnt_s{seed}.pdf"

    cmd = [
        "conda", "run", "--no-capture-output", "-n", args.conda_env,
        "python", "-u", CNN_SCRIPT,
        "--no-wandb",
        "--map-kind", "nbody",
        "--seed", str(seed),
        "--field-size", "20",
        "--field-npix", "160",
        "--nbins", "4",
        "--tomo-bin-indices", "1,2,3,4",
        "--full-sphere-cross-cache", HARM_CACHE,
        "--cnn-map-route", "harmonic",
        "--harmonic-cache-regime", "nobnt",
        "--train-compressor",
        "--compressor-arch", "plain",
        "--compressor-dim", "10",
        "--compressor-dense-width", str(args.compressor_dense_width),
        "--compressor-steps", str(args.compressor_steps),
        "--compressor-save-every", "2000",
        "--compressor-batch-size", "128",
        "--compressor-train-split", "train",
        "--compressor-val-split", "test",
        "--nde-train-split", "train",
        "--nde-val-split", "test",
        "--total-steps", str(args.flow_steps),
        "--save-every", "2000",
        "--batch-size", "128",
        "--patience", "20",
        "--ds-batch-size", "500",
        "--npe-samples", str(args.npe_samples),
        "--save-dir", str(run_dir / "save_params"),
        "--cache-dir", str(run_dir / "cache"),
        "--posterior-out", str(posterior_out),
        "--figure-out", str(figure_out),
        "--plot",
    ]
    log_path = output_root / "nobnt" / "logs" / f"{tag}.log"
    return Job(name=tag, command=cmd, log_path=log_path)


def run_jobs_parallel(
    jobs: List[Job],
    gpus: List[str],
    xla_mem_fraction: float,
    dry_run: bool,
    gpu_fractions: Dict[str, float] | None = None,
) -> List[Dict]:
    q: "queue.Queue[Job]" = queue.Queue()
    for job in jobs:
        q.put(job)

    results: List[Dict] = []
    lock = threading.Lock()

    def worker(gpu_id: str) -> None:
        frac = (gpu_fractions or {}).get(gpu_id, xla_mem_fraction)
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
                job.log_path.write_text("[dry-run] " + " ".join(cmd) + "\n", encoding="utf-8")
            else:
                env = dict(os.environ)
                env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(frac)
                env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
                with open(job.log_path, "w", encoding="utf-8") as logf:
                    proc = subprocess.run(
                        cmd, env=env, stdout=logf, stderr=subprocess.STDOUT
                    )
                    rc = proc.returncode
            dt = time.time() - t0
            with lock:
                results.append({
                    "name": job.name,
                    "gpu": gpu_id,
                    "returncode": int(rc),
                    "seconds": float(dt),
                    "log": str(job.log_path),
                })
            q.task_done()

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    (args.output_root / "nobnt" / "posteriors").mkdir(parents=True, exist_ok=True)
    (args.output_root / "nobnt" / "figures").mkdir(parents=True, exist_ok=True)
    (args.output_root / "nobnt" / "logs").mkdir(parents=True, exist_ok=True)

    gpu_fractions: Dict[str, float] = {}
    if args.xla_mem_fraction_by_gpu:
        for item in args.xla_mem_fraction_by_gpu.split(","):
            gid, frac = item.split(":")
            gpu_fractions[gid.strip()] = float(frac)

    jobs = [build_job(args, seed, args.output_root) for seed in seeds]
    print(f"[A3] {len(jobs)} jobs across {len(gpus)} GPUs: {gpus}")
    for j in jobs:
        gpu_frac = gpu_fractions.get(gpus[jobs.index(j) % len(gpus)], args.xla_mem_fraction)
        print(f"  {j.name}  (assigned gpu tentative, xla_frac={gpu_frac})")

    results = run_jobs_parallel(
        jobs, gpus, args.xla_mem_fraction, args.dry_run, gpu_fractions
    )

    failed = [r for r in results if r["returncode"] != 0]
    status = "DONE" if not failed else f"PARTIAL ({len(failed)} failed)"
    print(f"\n[A3] {status}")
    for r in results:
        mark = "✓" if r["returncode"] == 0 else "✗"
        print(f"  {mark} {r['name']:40s}  gpu={r['gpu']}  {r['seconds']:.0f}s  rc={r['returncode']}")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "gpus": gpus,
        "xla_mem_fraction": args.xla_mem_fraction,
        "gpu_fractions": gpu_fractions,
        "compressor_steps": args.compressor_steps,
        "compressor_dense_width": args.compressor_dense_width,
        "flow_steps": args.flow_steps,
        "dry_run": args.dry_run,
        "results": results,
    }
    manifest_path = args.output_root / "nobnt" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[A3] Manifest at {manifest_path}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
