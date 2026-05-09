#!/usr/bin/env python3
"""A5: VMIM auxiliary-network width sweep (target stability check).

Tests whether the VMIM companion RealNVP is saturated at the default hidden
width of 128. Sweeps:
  vmim_nf_hidden ∈ {128, 256, 512}  (default × 1/2/4)
  × seeds {42, 43}
  × condition: no-BNT, zero-mean, plain CNN, cdim=10

If compressor quality (loss, posterior FoM3) stays flat across widths, the
VMIM bound is not limiting the compressor. If quality improves at larger
widths, the VMIM auxiliary was under-capacity and the compressor was
sub-optimal.

Usage
-----
conda run -n jaxili python run_a5_vmim_stability.py \\
    --gpus 0,1,2 --xla-mem-fraction 0.45 [--dry-run]
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

VMIM_HIDDENS: Tuple[int, ...] = (128, 256, 512)
SEEDS: Tuple[int, ...] = (42, 43)


@dataclass(frozen=True)
class Job:
    name: str
    command: List[str]
    log_path: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A5 VMIM stability sweep.")
    p.add_argument("--gpus", type=str, default="0,1,2")
    p.add_argument("--xla-mem-fraction", type=float, default=0.45)
    p.add_argument(
        "--xla-mem-fraction-by-gpu", type=str, default="",
        help="Per-GPU overrides, e.g. '0:0.30,1:0.45,2:0.45'. Overrides --xla-mem-fraction for listed GPUs.",
    )
    p.add_argument("--conda-env", type=str, default="jaxili")
    p.add_argument(
        "--tfds-name",
        type=str,
        default="NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48",
    )
    p.add_argument("--vmim-hiddens", type=str, default=",".join(str(h) for h in VMIM_HIDDENS))
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    p.add_argument("--compressor-steps", type=int, default=120000)
    p.add_argument("--compressor-dense-width", type=int, default=256)
    p.add_argument("--flow-steps", type=int, default=10000)
    p.add_argument("--npe-samples", type=int, default=100000)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_job(
    args: argparse.Namespace,
    vmim_hidden: int,
    seed: int,
    output_root: Path,
) -> Job:
    tag = f"plain_vmimh{vmim_hidden}_s{seed}"
    run_dir = output_root / tag
    posterior_out = output_root / "posteriors" / f"cnn_vmimstab_nobnt_vmimh{vmim_hidden}_s{seed}.npy"
    figure_out = output_root / "figures" / f"cnn_vmimstab_nobnt_vmimh{vmim_hidden}_s{seed}.pdf"

    cmd = [
        "conda", "run", "--no-capture-output", "-n", args.conda_env,
        "python", "-u", CNN_SCRIPT,
        "--no-wandb",
        "--train-compressor",
        "--zero-mean-maps",
        "--map-kind", "nbody",
        "--seed", str(seed),
        "--tfds-name", args.tfds_name,
        "--field-size", "20",
        "--field-npix", "160",
        "--nbins", "4",
        "--tomo-bin-indices", "1,2,3,4",
        "--compressor-arch", "plain",
        "--compressor-dim", "10",
        "--compressor-dense-width", str(args.compressor_dense_width),
        "--compressor-steps", str(args.compressor_steps),
        "--vmim-nf-hidden", str(vmim_hidden),
        "--compressor-train-split", "train",
        "--compressor-val-split", "test",
        "--nde-train-split", "train",
        "--nde-val-split", "test",
        "--total-steps", str(args.flow_steps),
        "--save-every", "500",
        "--npe-samples", str(args.npe_samples),
        "--ds-batch-size", "500",
        "--save-dir", str(run_dir),
        "--posterior-out", str(posterior_out),
        "--figure-out", str(figure_out),
        "--plot",
    ]
    return Job(
        name=tag,
        command=cmd,
        log_path=output_root / "logs" / f"{tag}.log",
    )


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
    vmim_hiddens = [int(h) for h in args.vmim_hiddens.split(",") if h.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    (args.output_root / "posteriors").mkdir(parents=True, exist_ok=True)
    (args.output_root / "figures").mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(parents=True, exist_ok=True)

    jobs = [
        build_job(args, vmim_hidden, seed, args.output_root)
        for vmim_hidden in vmim_hiddens
        for seed in seeds
    ]
    print(f"[A5] {len(jobs)} jobs across {len(gpus)} GPUs")
    for j in jobs:
        print(f"  {j.name}")

    gpu_fractions: Dict[str, float] = {}
    if args.xla_mem_fraction_by_gpu:
        for item in args.xla_mem_fraction_by_gpu.split(","):
            gid, frac = item.split(":")
            gpu_fractions[gid.strip()] = float(frac)
    results = run_jobs_parallel(jobs, gpus, args.xla_mem_fraction, args.dry_run, gpu_fractions)

    failed = [r for r in results if r["returncode"] != 0]
    status = "DONE" if not failed else f"PARTIAL ({len(failed)} failed)"
    print(f"\n[A5] {status}")
    for r in results:
        mark = "✓" if r["returncode"] == 0 else "✗"
        print(f"  {mark} {r['name']:40s}  gpu={r['gpu']}  {r['seconds']:.0f}s")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "vmim_hiddens": vmim_hiddens,
        "seeds": seeds,
        "gpus": gpus,
        "xla_mem_fraction": args.xla_mem_fraction,
        "compressor_steps": args.compressor_steps,
        "compressor_dense_width": args.compressor_dense_width,
        "flow_steps": args.flow_steps,
        "dry_run": args.dry_run,
        "results": results,
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[A5] Manifest at {manifest_path}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
