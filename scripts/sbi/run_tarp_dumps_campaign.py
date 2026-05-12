#!/usr/bin/env python3
"""Dispatch all (arm, seed) SBC dump jobs that feed the TARP coverage plots.

Each job is an SBC re-run with ``--dump-posterior-samples`` and a per-arm
deterministic ``--rank-seed`` (so scatter across seeds reflects the
trained estimator, not the test-cosmology draw). Jobs are pooled across
GPUs; per-GPU XLA memory caps can be overridden with
``--xla-mem-fraction-by-gpu``.

Defaults match the plan: N=200 cosmologies, M=2000 posterior samples per
cosmology, all available seeds per arm — 17 jobs total.
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
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
SBI_DIR = REPO_ROOT / "scripts" / "sbi"
DEFAULT_OUTPUT_ROOT = SBI_DIR / "results" / "diagnostics" / "tarp_harm_cross"


@dataclass(frozen=True)
class Job:
    arm: str
    seed: int
    command: List[str]
    log_path: Path
    expected_dump: Path


L1_BASELINE = (
    SBI_DIR / "results" / "exploratory" / "cross_maps_campaign" / "jaxili_harm_cross_nobnt"
)
CNN_AUTO_BASELINE = (
    SBI_DIR
    / "results"
    / "exploratory"
    / "zero_mean_maps_parity_check"
    / "run_b_advanced_plain"
)
CNN_HARMC_BASELINE = (
    SBI_DIR / "results" / "exploratory" / "cnn_with_harm_cross_normalized"
)


def _l1_harm_cross_job(
    arm: str,
    seed: int,
    n_ranks: int,
    posterior_samples: int,
    rank_seed: int,
    output_root: Path,
    log_dir: Path,
    conda_env: str,
) -> Job:
    seed_root = L1_BASELINE / f"seed_{seed}" / "l1norm_cross_jaxili" / "nbody"
    cache_dir = L1_BASELINE / f"l1_cache_seed{seed}"
    dump_root = output_root / "dumps" / arm / f"seed_{seed}"
    run_tag = f"n{n_ranks}_m{posterior_samples}_seed{rank_seed}"
    cmd = [
        "conda", "run", "--no-capture-output", "-n", conda_env, "python",
        str(SBI_DIR / "run_sbc_harm_l1_nobnt.py"),
        "--baseline-root", str(L1_BASELINE),
        "--cache-dir", str(cache_dir),
        "--checkpoint-root", str(seed_root / "params_l1norm_cross_jaxili"),
        "--preprocessing-stats", str(seed_root / "l1_cross_jaxili_standardization.npz"),
        "--feature-mask", str(seed_root / "l1_cross_jaxili_feature_mask.npz"),
        "--output-root", str(dump_root),
        "--n-ranks", str(n_ranks),
        "--posterior-samples", str(posterior_samples),
        "--rank-seed", str(rank_seed),
        "--dump-posterior-samples",
    ]
    return Job(
        arm=arm,
        seed=seed,
        command=cmd,
        log_path=log_dir / f"{arm}_seed{seed}.log",
        expected_dump=dump_root / run_tag / "posterior_samples.npz",
    )


def _cnn_auto_only_job(
    arm: str,
    seed: int,
    n_ranks: int,
    posterior_samples: int,
    rank_seed: int,
    output_root: Path,
    log_dir: Path,
    conda_env: str,
) -> Job:
    dump_root = output_root / "dumps" / arm / f"seed_{seed}"
    run_tag = f"n{n_ranks}_m{posterior_samples}_seed{rank_seed}"
    cmd = [
        "conda", "run", "--no-capture-output", "-n", conda_env, "python",
        str(SBI_DIR / "run_sbc_cnn_nobnt.py"),
        "--baseline-root", str(CNN_AUTO_BASELINE),
        "--output-root", str(dump_root),
        "--condition", "nobnt",
        "--n-ranks", str(n_ranks),
        "--posterior-samples", str(posterior_samples),
        "--rank-seed", str(rank_seed),
        "--flow-seed", str(seed),
        "--dump-posterior-samples",
    ]
    return Job(
        arm=arm,
        seed=seed,
        command=cmd,
        log_path=log_dir / f"{arm}_seed{seed}.log",
        expected_dump=dump_root / run_tag / "posterior_samples.npz",
    )


def _cnn_harm_cross_job(
    arm: str,
    seed: int,
    arch: str,
    n_ranks: int,
    posterior_samples: int,
    rank_seed: int,
    output_root: Path,
    log_dir: Path,
    conda_env: str,
) -> Job:
    dump_root = output_root / "dumps" / arm
    run_tag = f"n{n_ranks}_m{posterior_samples}_seed{rank_seed}"
    cmd = [
        "conda", "run", "--no-capture-output", "-n", conda_env, "python",
        str(SBI_DIR / "run_sbc_cnn_harm_cross_nobnt.py"),
        "--compressor-arch", arch,
        "--seed", str(seed),
        "--output-root", str(dump_root),
        "--n-ranks", str(n_ranks),
        "--posterior-samples", str(posterior_samples),
        "--rank-seed", str(rank_seed),
    ]
    return Job(
        arm=arm,
        seed=seed,
        command=cmd,
        log_path=log_dir / f"{arm}_seed{seed}.log",
        expected_dump=dump_root / arch / f"seed_{seed}" / run_tag / "posterior_samples.npz",
    )


ARM_BUILDERS = {
    "l1_harm_cross": dict(seeds=(41, 42, 43, 44, 45, 46), builder=_l1_harm_cross_job),
    "cnn_auto_only": dict(seeds=(41, 42, 43, 44, 45), builder=_cnn_auto_only_job),
    "cnn_harm_cross_plain": dict(
        seeds=(41, 42, 43),
        builder=lambda **kw: _cnn_harm_cross_job(arch="plain", **kw),
    ),
    "cnn_harm_cross_gn": dict(
        seeds=(41, 42, 43),
        builder=lambda **kw: _cnn_harm_cross_job(arch="resnet50_gn", **kw),
    ),
}


def _derive_rank_seed(arm: str, seed: int, base: int) -> int:
    arm_hash = sum(ord(c) for c in arm) * 1009
    return int(base + arm_hash + seed * 17)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--conda-env", type=str, default="jaxili")
    p.add_argument("--gpus", type=str, default="0,1,2,3")
    p.add_argument("--xla-mem-fraction", type=float, default=0.6)
    p.add_argument(
        "--xla-mem-fraction-by-gpu",
        type=str,
        default="",
        help="Per-GPU overrides, e.g. '0:0.75,1:0.30'. Unlisted GPUs use --xla-mem-fraction.",
    )
    p.add_argument("--n-ranks", type=int, default=200)
    p.add_argument("--posterior-samples", type=int, default=2000)
    p.add_argument(
        "--rank-seed-base",
        type=int,
        default=20260511,
        help="Deterministic base for per-(arm,seed) rank seeds.",
    )
    p.add_argument(
        "--arms",
        type=str,
        default=",".join(ARM_BUILDERS.keys()),
        help="Comma-separated subset of arms to dispatch.",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Optional comma-separated subset of seeds (intersected per arm).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip (arm, seed) pairs whose posterior_samples.npz already exists.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _parse_gpu_mem_overrides(spec: str, gpus: List[str]) -> Dict[str, float]:
    if not spec.strip():
        return {}
    valid = set(gpus)
    out: Dict[str, float] = {}
    for tok in (t.strip() for t in spec.split(",") if t.strip()):
        gpu_id, frac_str = tok.split(":", 1)
        if gpu_id not in valid:
            raise ValueError(f"GPU '{gpu_id}' not in --gpus")
        frac = float(frac_str)
        if not (0.0 < frac <= 1.0):
            raise ValueError(f"Memory fraction {frac} out of (0,1]")
        out[gpu_id] = frac
    return out


def build_jobs(args: argparse.Namespace) -> List[Job]:
    output_root = args.output_root.resolve()
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    selected_arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seed_filter: Optional[set] = None
    if args.seeds.strip():
        seed_filter = {int(s) for s in args.seeds.split(",") if s.strip()}

    jobs: List[Job] = []
    for arm in selected_arms:
        if arm not in ARM_BUILDERS:
            raise ValueError(f"Unknown arm '{arm}'. Available: {list(ARM_BUILDERS)}")
        cfg = ARM_BUILDERS[arm]
        builder = cfg["builder"]
        for seed in cfg["seeds"]:
            if seed_filter is not None and seed not in seed_filter:
                continue
            rank_seed = _derive_rank_seed(arm, seed, args.rank_seed_base)
            job = builder(
                arm=arm,
                seed=seed,
                n_ranks=args.n_ranks,
                posterior_samples=args.posterior_samples,
                rank_seed=rank_seed,
                output_root=output_root,
                log_dir=log_dir,
                conda_env=args.conda_env,
            )
            if args.skip_existing and job.expected_dump.exists():
                print(f"[skip] {arm} seed={seed}: {job.expected_dump} exists")
                continue
            jobs.append(job)
    return jobs


def run_jobs_parallel(
    jobs: List[Job],
    gpus: List[str],
    mem_default: float,
    mem_overrides: Dict[str, float],
    dry_run: bool,
) -> List[Dict[str, object]]:
    q: queue.Queue[Job] = queue.Queue()
    for j in jobs:
        q.put(j)
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
            mem_frac = mem_overrides.get(gpu_id, mem_default)
            t0 = time.time()
            if dry_run:
                rc = 0
                job.log_path.write_text(
                    f"[dry-run] gpu={gpu_id} mem={mem_frac}\n{' '.join(cmd)}\n",
                    encoding="utf-8",
                )
            else:
                env = dict(os.environ)
                env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{mem_frac:.4f}"
                with open(job.log_path, "w", encoding="utf-8") as logf:
                    proc = subprocess.run(
                        cmd,
                        cwd=str(REPO_ROOT),
                        env=env,
                        stdout=logf,
                        stderr=subprocess.STDOUT,
                    )
                    rc = proc.returncode
            dt = time.time() - t0
            with lock:
                results.append(
                    {
                        "arm": job.arm,
                        "seed": int(job.seed),
                        "gpu": gpu_id,
                        "returncode": int(rc),
                        "seconds": float(dt),
                        "log": str(job.log_path),
                        "expected_dump": str(job.expected_dump),
                        "dump_exists": bool(job.expected_dump.exists()),
                        "xla_mem_fraction": float(mem_frac),
                        "cmd": cmd,
                    }
                )
                print(
                    f"[{job.arm} seed={job.seed}] gpu={gpu_id} rc={rc} "
                    f"in {dt:.1f}s -> dump_exists={job.expected_dump.exists()}"
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
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    mem_overrides = _parse_gpu_mem_overrides(args.xla_mem_fraction_by_gpu, gpus)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args)
    if not jobs:
        print("[campaign] No jobs to run.")
        return

    print(
        f"[campaign] {len(jobs)} jobs across {len(gpus)} GPU(s): "
        f"arms={sorted({j.arm for j in jobs})}, dry_run={args.dry_run}"
    )
    started = datetime.now(timezone.utc).isoformat()
    results = run_jobs_parallel(
        jobs=jobs,
        gpus=gpus,
        mem_default=args.xla_mem_fraction,
        mem_overrides=mem_overrides,
        dry_run=args.dry_run,
    )
    finished = datetime.now(timezone.utc).isoformat()

    failed = [r for r in results if int(r["returncode"]) != 0]
    summary = {
        "started_utc": started,
        "finished_utc": finished,
        "n_jobs": len(results),
        "n_failed": len(failed),
        "n_ranks": int(args.n_ranks),
        "posterior_samples": int(args.posterior_samples),
        "rank_seed_base": int(args.rank_seed_base),
        "gpus": gpus,
        "xla_mem_fraction_default": float(args.xla_mem_fraction),
        "xla_mem_fraction_by_gpu": mem_overrides,
        "results": results,
    }
    summary_path = output_root / "campaign_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[campaign] Done. {len(results) - len(failed)}/{len(results)} succeeded. "
        f"Summary: {summary_path}"
    )
    if failed:
        for r in failed:
            print(f"  FAIL {r['arm']} seed={r['seed']} rc={r['returncode']} log={r['log']}")


if __name__ == "__main__":
    main()
