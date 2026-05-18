#!/usr/bin/env python3
"""Per-arm autoresearch runner for the cnn-auto-push fiber.

Two-phase pipeline that mirrors the existing
``scripts/sbi/results/exploratory/cnn_extended_train_zm/run_a2_cnn_wider_longer.py``
baseline:

* **Phase A** trains one shared compressor (``--total-steps``) on a single GPU
  with a fixed ``--train-seed``.
* **Phase B** trains an NDE flow per seed (``--seeds``) in parallel, one seed
  per GPU, reusing the shared compressor checkpoint from Phase A.

Auto-only inputs: no ``--channel-mode``, no ``--full-sphere-cross-cache``.
4 auto maps from TFDS ``NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48``.

Output layout under ``--out-dir``::

    compressor/...                                   # shared compressor checkpoint
    cache/                                           # train + eval caches
    eval/seed_<N>/                                   # per-seed NDE save_params
    posteriors/<stem>_s<N>.npy                       # matches *_s4?.npy glob
    logs/{compressor.log,nde_s<N>.log}
    run_manifest.json
    job_results.json

GPU policy: only GPUs 0,1,2 are accepted (Andreas's policy). Per-process
``XLA_PYTHON_CLIENT_MEM_FRACTION`` is set from ``--xla-mem-fraction`` (default
0.9 — appropriate for 1 process per GPU; lower if you co-pack).
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
CNN_SCRIPT = REPO_ROOT / "scripts" / "sbi" / "npe_cnn_nbody_tomo.py"

# Fixed dataset / geometry — auto-only, 4-bin tomo, 20° / 160 px, zero-mean maps
DATASET_FLAGS: List[str] = [
    "--map-kind", "nbody",
    "--tfds-name", "NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48",
    "--field-size", "20",
    "--field-npix", "160",
    "--nbins", "4",
    "--tomo-bin-indices", "1,2,3,4",
    "--zero-mean-maps",
    "--no-wandb",
]

# Split conventions (match run_a2 baseline)
SPLIT_FLAGS: List[str] = [
    "--compressor-train-split", "train",
    "--compressor-val-split", "test",
    "--nde-train-split", "train",
    "--nde-val-split", "test",
]


def _csv_int(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _csv_str(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--arm", required=True, choices=["plain", "resnet50_gn"])
    p.add_argument("--total-steps", type=int, required=True,
                   help="Compressor training steps (Phase A).")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output directory for posteriors, logs, manifest.")
    p.add_argument("--gpus", required=True,
                   help="Comma-separated GPU IDs (e.g. 0,1,2). Phase A uses gpus[0].")
    p.add_argument("--seeds", required=True,
                   help="Comma-separated NDE eval seeds (e.g. 41,42,43).")
    p.add_argument("--train-seed", type=int, default=42,
                   help="Seed for Phase A compressor training.")
    p.add_argument("--flow-steps", type=int, default=10000,
                   help="NDE flow training steps (Phase B).")
    p.add_argument("--compressor-save-every", type=int, default=3000)
    p.add_argument("--save-every", type=int, default=500,
                   help="NDE flow save-every (val-check cadence).")
    p.add_argument("--patience", type=int, default=50,
                   help="NDE flow early-stop patience (in val checks).")
    p.add_argument("--batch-size", type=int, default=256,
                   help="NDE flow batch size.")
    p.add_argument("--compressor-batch-size", type=int, default=128)
    p.add_argument("--compressor-lr", type=float, default=5e-4)
    p.add_argument("--compressor-dim", type=int, default=20)
    p.add_argument("--compressor-dense-width", type=int, default=512,
                   help="(plain arm only)")
    p.add_argument("--compressor-conv-channels", type=str, default="64,128,256",
                   help="(plain arm only)")
    p.add_argument("--npe-samples", type=int, default=100_000)
    p.add_argument("--nvp-layers", type=int, default=8)
    p.add_argument("--nvp-hidden", type=int, default=256)
    p.add_argument("--xla-mem-fraction", type=float, default=0.9,
                   help="Per-process XLA_PYTHON_CLIENT_MEM_FRACTION.")
    p.add_argument("--name-stem", type=str, default=None,
                   help="Posterior filename stem. Default: cnn_auto_<arm>_step<N>.")
    p.add_argument("--skip-compressor", action="store_true",
                   help="Skip Phase A (compressor checkpoint must already exist).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing.")
    return p.parse_args()


def _arch_flags(arm: str, args: argparse.Namespace) -> List[str]:
    if arm == "plain":
        return [
            "--compressor-arch", "plain",
            "--compressor-dim", str(args.compressor_dim),
            "--compressor-conv-channels", args.compressor_conv_channels,
            "--compressor-dense-width", str(args.compressor_dense_width),
            "--compressor-pool-window", "16",
            "--compressor-pool-stride", "8",
        ]
    if arm == "resnet50_gn":
        return [
            "--compressor-arch", "resnet50_gn",
            "--compressor-dim", str(args.compressor_dim),
        ]
    raise ValueError(f"unknown arm '{arm}'")


def _compressor_paths(out_dir: Path, total_steps: int) -> Dict[str, Path]:
    base = (
        out_dir
        / "compressor"
        / "nobnt"
        / "vmim"
        / "nbody"
        / "sigma_0.26"
        / "gal_density_30"
        / "bin_4"
    )
    return {
        "params": base / f"params_nd_compressor_batch{total_steps}.pkl",
        "state": base / f"opt_state_resnet_batch{total_steps}.pkl",
    }


def _build_compressor_cmd(args: argparse.Namespace, out_dir: Path) -> List[str]:
    return [
        "conda", "run", "--no-capture-output", "-n", "jaxili", "python",
        str(CNN_SCRIPT),
        "--seed", str(args.train_seed),
        "--cache-dir", str(out_dir / "cache" / "nobnt_train"),
        "--save-dir", str(out_dir / "compressor" / "nobnt"),
        "--train-compressor",
        "--compressor-steps", str(args.total_steps),
        "--compressor-save-every", str(args.compressor_save_every),
        "--compressor-batch-size", str(args.compressor_batch_size),
        "--compressor-lr", str(args.compressor_lr),
        # Phase A doesn't train the NDE — keep it at the minimum that the
        # inner script accepts (it still iterates the NDE loop briefly).
        "--total-steps", "1",
        "--save-every", "1",
        "--no-sample",
        "--summary-clip-value", "5.0",
        "--ds-batch-size", "500",
        "--no-standardize-summary",
        *DATASET_FLAGS,
        *SPLIT_FLAGS,
        *_arch_flags(args.arm, args),
    ]


def _build_nde_cmd(
    args: argparse.Namespace,
    out_dir: Path,
    seed: int,
    comp_paths: Dict[str, Path],
    posterior_out: Path,
) -> List[str]:
    return [
        "conda", "run", "--no-capture-output", "-n", "jaxili", "python",
        str(CNN_SCRIPT),
        "--seed", str(seed),
        "--cache-dir", str(out_dir / "cache" / "nobnt_eval"),
        "--save-dir", str(out_dir / "eval" / f"seed_{seed}"),
        "--compressor-params", str(comp_paths["params"]),
        "--compressor-state", str(comp_paths["state"]),
        "--total-steps", str(args.flow_steps),
        "--save-every", str(args.save_every),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--nvp-layers", str(args.nvp_layers),
        "--nvp-hidden", str(args.nvp_hidden),
        "--summary-clip-value", "5.0",
        "--npe-samples", str(args.npe_samples),
        "--posterior-out", str(posterior_out),
        "--ds-batch-size", "500",
        "--no-standardize-summary",
        *DATASET_FLAGS,
        *SPLIT_FLAGS,
        *_arch_flags(args.arm, args),
    ]


JobSpec = Tuple[str, List[str], Path]  # (name, cmd_without_gpu, log_path)


def _run_jobs_parallel(
    jobs: List[JobSpec],
    gpus: List[str],
    xla_mem_fraction: float,
    dry_run: bool,
) -> List[Dict[str, object]]:
    q: "queue.Queue[JobSpec]" = queue.Queue()
    for j in jobs:
        q.put(j)

    results: List[Dict[str, object]] = []
    lock = threading.Lock()

    def worker(gpu_id: str) -> None:
        while True:
            try:
                name, cmd, log_path = q.get_nowait()
            except queue.Empty:
                return
            log_path.parent.mkdir(parents=True, exist_ok=True)
            full_cmd = list(cmd) + ["--cuda-visible-devices", gpu_id]
            t0 = time.time()
            if dry_run:
                rc = 0
                log_path.write_text(
                    "[dry-run] " + " ".join(full_cmd) + "\n",
                    encoding="utf-8",
                )
            else:
                env = dict(os.environ)
                env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(xla_mem_fraction)
                with open(log_path, "w", encoding="utf-8") as logf:
                    proc = subprocess.run(
                        full_cmd, env=env,
                        stdout=logf, stderr=subprocess.STDOUT,
                    )
                    rc = proc.returncode
            dt = time.time() - t0
            with lock:
                results.append({
                    "name": name,
                    "gpu": gpu_id,
                    "returncode": int(rc),
                    "seconds": float(dt),
                    "log": str(log_path),
                })
            q.task_done()

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def _write_results(out_dir: Path, results: List[Dict[str, object]]) -> None:
    (out_dir / "job_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "posteriors").mkdir(exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    gpus = _csv_str(args.gpus)
    if not gpus:
        raise ValueError("--gpus cannot be empty")
    if any(g not in {"0", "1", "2"} for g in gpus):
        raise ValueError("GPU policy lock: only GPUs 0,1,2 are allowed.")
    seeds = _csv_int(args.seeds)
    if not seeds:
        raise ValueError("--seeds cannot be empty")
    if not (0.0 < args.xla_mem_fraction <= 1.0):
        raise ValueError("--xla-mem-fraction must be in (0, 1].")

    stem = args.name_stem or f"cnn_auto_{args.arm}_step{args.total_steps}"
    comp_paths = _compressor_paths(out_dir, args.total_steps)

    manifest = {
        "arm": args.arm,
        "compressor_steps": int(args.total_steps),
        "flow_steps": int(args.flow_steps),
        "seeds": seeds,
        "train_seed": int(args.train_seed),
        "gpus": gpus,
        "xla_mem_fraction": float(args.xla_mem_fraction),
        "name_stem": stem,
        "compressor_dim": int(args.compressor_dim),
        "compressor_dense_width": int(args.compressor_dense_width),
        "compressor_conv_channels": args.compressor_conv_channels,
        "compressor_batch_size": int(args.compressor_batch_size),
        "compressor_lr": float(args.compressor_lr),
        "save_every": int(args.save_every),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "nvp_layers": int(args.nvp_layers),
        "nvp_hidden": int(args.nvp_hidden),
        "npe_samples": int(args.npe_samples),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )

    all_results: List[Dict[str, object]] = []

    # Phase A — shared compressor
    if not args.skip_compressor:
        if comp_paths["params"].exists() and comp_paths["state"].exists():
            print(f"[skip] compressor exists at {comp_paths['params']}", flush=True)
        else:
            print(
                f"[phase A] training compressor ({args.total_steps} steps) "
                f"on GPU {gpus[0]}",
                flush=True,
            )
            phase_a: List[JobSpec] = [(
                "compressor",
                _build_compressor_cmd(args, out_dir),
                out_dir / "logs" / "compressor.log",
            )]
            res = _run_jobs_parallel(
                phase_a, gpus=[gpus[0]],
                xla_mem_fraction=args.xla_mem_fraction,
                dry_run=args.dry_run,
            )
            all_results.extend(res)
            failed = [r for r in res if r["returncode"] != 0]
            if failed:
                print(
                    f"[FAIL] compressor: rc={failed[0]['returncode']} "
                    f"(log: {failed[0]['log']})",
                    flush=True,
                )
                _write_results(out_dir, all_results)
                return 1

    if not args.dry_run:
        for key, p in comp_paths.items():
            if not p.exists():
                print(f"[FAIL] compressor {key} missing at {p}", flush=True)
                _write_results(out_dir, all_results)
                return 1

    # Phase B — NDE per seed, one seed per GPU
    print(
        f"[phase B] training NDE for {len(seeds)} seeds on GPUs {gpus}",
        flush=True,
    )
    phase_b: List[JobSpec] = []
    for seed in seeds:
        posterior_out = out_dir / "posteriors" / f"{stem}_s{seed}.npy"
        phase_b.append((
            f"nde_s{seed}",
            _build_nde_cmd(args, out_dir, seed, comp_paths, posterior_out),
            out_dir / "logs" / f"nde_s{seed}.log",
        ))
    res = _run_jobs_parallel(
        phase_b, gpus=gpus,
        xla_mem_fraction=args.xla_mem_fraction,
        dry_run=args.dry_run,
    )
    all_results.extend(res)
    _write_results(out_dir, all_results)

    failed = [r for r in res if r["returncode"] != 0]
    if failed:
        for r in failed:
            print(
                f"[FAIL] {r['name']} rc={r['returncode']} (log: {r['log']})",
                flush=True,
            )
        return 1

    if not args.dry_run:
        missing = [
            (seed, out_dir / "posteriors" / f"{stem}_s{seed}.npy")
            for seed in seeds
            if not (out_dir / "posteriors" / f"{stem}_s{seed}.npy").exists()
        ]
        if missing:
            for seed, p in missing:
                print(f"[FAIL] missing posterior s{seed}: {p}", flush=True)
            return 1

    print(
        f"[OK] {len(seeds)} seeds trained; posteriors in "
        f"{out_dir / 'posteriors'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
