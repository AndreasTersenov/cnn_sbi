#!/usr/bin/env python3
"""Run the cross-only L1 vs CNN campaign.

Trains and evaluates 5 arms on the 6 harmonic cross channels alone (slicing
the existing 10-channel cache at read time):

  - l1_cross_only                (3 seeds)
  - cnn_cross_only_plain         × dim ∈ {10, 20}   (3 seeds × 2 dims)
  - cnn_cross_only_resnet50_gn   × dim ∈ {10, 20}   (3 seeds × 2 dims)

15 runs total. NDE training budget is pinned identically across L1 and CNN
(50000 steps, batch 256, patience 30) so the comparison is fair on budget.
The L1/CNN compressor pipelines remain different by construction.

GPU pooling: jobs are dispatched onto a worker pool of GPUs (default 0,1,2).
At launch the orchestrator probes `nvidia-smi` for each target GPU and sets
XLA_PYTHON_CLIENT_MEM_FRACTION to (free/total) - 0.05, capped at 0.90,
which gives each job the maximum room available at that moment. The
fraction can also be overridden manually with --xla-mem-fraction-by-gpu.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
SBI_DIR = REPO_ROOT / "scripts" / "sbi"
DEFAULT_OUTPUT_ROOT = (
    SBI_DIR / "results" / "exploratory" / "cross_only_campaign"
)
HARM_CACHE = (
    SBI_DIR
    / "results"
    / "exploratory"
    / "cross_maps_campaign"
    / "full_sphere_cache_grid"
)


# ---------------------------------------------------------------------------
# Shared flags pinned across both L1 and CNN sides (fair-budget operationalization)
# ---------------------------------------------------------------------------
SHARED_FLAGS = [
    "--map-kind", "nbody",
    "--tfds-name", "NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48",
    "--field-size", "20",
    "--field-npix", "160",
    "--nbins", "4",
    "--tomo-bin-indices", "1,2,3,4",
    "--full-sphere-cross-cache", str(HARM_CACHE),
    "--channel-mode", "cross_only",
    "--no-wandb",
    "--plot",
]

NDE_BUDGET_FLAGS = [
    "--total-steps", "50000",
    "--batch-size", "256",
    "--patience", "30",
    "--save-every", "2000",
    "--npe-samples", "100000",
]

NDE_BUDGET_FLAGS_SMOKE = [
    "--total-steps", "2",
    "--batch-size", "16",
    "--patience", "0",
    "--save-every", "1",
    "--npe-samples", "64",
]

CNN_COMPRESSOR_FLAGS = [
    "--train-compressor",
    "--compressor-steps", "150000",
    "--compressor-save-every", "2000",
    # Match v1 batch size so v1 vs v2 isolates only the L1 noise-model change.
    "--compressor-batch-size", "128",
    "--compressor-dense-width", "256",
    "--compressor-train-split", "train",
    "--compressor-val-split", "test",
    "--nde-train-split", "train",
    "--nde-val-split", "test",
    "--cnn-map-route", "harmonic",
    "--harmonic-cache-regime", "nobnt",
    "--harmonic-normalize-input-channels",
    "--ds-batch-size", "500",
]

CNN_COMPRESSOR_FLAGS_SMOKE = [
    "--train-compressor",
    "--compressor-steps", "2",
    "--compressor-save-every", "1",
    "--compressor-batch-size", "16",
    "--compressor-dense-width", "256",
    "--compressor-train-split", "train",
    "--compressor-val-split", "test",
    "--nde-train-split", "train",
    "--nde-val-split", "test",
    "--cnn-map-route", "harmonic",
    "--harmonic-cache-regime", "nobnt",
    "--harmonic-normalize-input-channels",
    "--harmonic-train-realizations-limit", "1",
    "--harmonic-val-realizations-limit", "1",
    "--ds-batch-size", "16",
]

L1_FLAGS = [
    "--n-scales", "5",
    "--l1-nbins", "40",
    "--l1-min-snr", "-13",
    "--l1-max-snr", "13",
    "--pca-components", "0",
    "--cross-snr-percentile", "1.0",
    "--ds-batch-size", "96",  # internal cap; actual batch comes from realizations-per-batch below
    # Batch 10 cache realizations per GPU L1 call (10 × 48 = 480 patches/call,
    # ~5 GB) instead of 1 (~1.2 GB). ~5-10× faster L1 dataset compute.
    "--l1-realizations-per-batch", "10",
    "--learning-rate", "1e-4",
]

L1_FLAGS_SMOKE_EXTRA = [
    "--harmonic-calibration-realizations", "2",
    "--no-sample",
]


# ---------------------------------------------------------------------------
# Job model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Job:
    arm: str
    seed: int
    dim: Optional[int]
    runner: Path
    command: List[str]
    log_path: Path
    run_dir: Path
    posterior_out: Path
    figure_out: Path
    stage: str = "single"  # 'single' (L1), 'stage_a' (CNN compressor), 'stage_b' (CNN NDE)
    shared_save_dir: Optional[Path] = None
    shared_cache_dir: Optional[Path] = None


COMPRESSOR_SEED = 41  # Fixed seed for the shared CNN compressor in Stage A.


def _arm_run_dir(arm: str, seed: int, dim: Optional[int], root: Path) -> Path:
    if dim is None:
        return root / arm / f"seed_{seed}"
    return root / arm / f"dim_{dim}" / f"seed_{seed}"


def _arm_tag(arm: str, seed: int, dim: Optional[int]) -> str:
    if dim is None:
        return f"{arm}_s{seed}"
    return f"{arm}_d{dim}_s{seed}"


def _build_l1_job(
    seed: int,
    output_root: Path,
    conda_env: str,
    smoke: bool,
    cross_noise_model: str = "auto_scalar",
) -> Job:
    arm = "l1_cross_only"
    run_dir = _arm_run_dir(arm, seed, None, output_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_dir = run_dir / "save_params"
    cache_dir = run_dir / "cache"
    posteriors_dir = output_root / arm / "posteriors"
    figures_dir = output_root / arm / "figures"
    log_dir = output_root / arm / "logs"
    for d in (save_dir, cache_dir, posteriors_dir, figures_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)
    tag = _arm_tag(arm, seed, None)
    posterior_out = posteriors_dir / f"{tag}.npy"
    figure_out = figures_dir / f"{tag}.pdf"
    log_path = log_dir / f"{tag}.log"

    runner = SBI_DIR / "npe_l1norm_cross_jaxili_nbody_tomo.py"
    nde_flags = NDE_BUDGET_FLAGS_SMOKE if smoke else NDE_BUDGET_FLAGS
    extra = list(L1_FLAGS_SMOKE_EXTRA) if smoke else []
    if cross_noise_model != "auto_scalar":
        extra += ["--cross-noise-model", cross_noise_model]
    cmd = [
        "conda", "run", "--no-capture-output", "-n", conda_env, "python", "-u",
        str(runner),
        *SHARED_FLAGS,
        *L1_FLAGS,
        *nde_flags,
        *extra,
        "--seed", str(seed),
        "--save-dir", str(save_dir),
        "--cache-dir", str(cache_dir),
        "--posterior-out", str(posterior_out),
        "--figure-out", str(figure_out),
    ]
    return Job(
        arm=arm,
        seed=seed,
        dim=None,
        runner=runner,
        command=cmd,
        log_path=log_path,
        run_dir=run_dir,
        posterior_out=posterior_out,
        figure_out=figure_out,
    )


def _shared_compressor_dir(arm: str, dim: int, output_root: Path) -> Path:
    return output_root / "_shared_compressor" / arm / f"dim_{dim}"


def _build_cnn_stage_a_job(
    arm: str,
    dim: int,
    arch: str,
    output_root: Path,
    conda_env: str,
    smoke: bool,
) -> Job:
    """Stage A: train one shared compressor per (arm, dim); exit before NDE."""
    shared_root = _shared_compressor_dir(arm, dim, output_root)
    save_dir = shared_root / "save_params"
    cache_dir = shared_root / "cache"
    log_dir = shared_root / "logs"
    for d in (save_dir, cache_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)
    tag = f"{arm}_d{dim}_compressor"
    log_path = log_dir / f"{tag}.log"

    runner = SBI_DIR / "npe_cnn_nbody_tomo.py"
    comp_flags = CNN_COMPRESSOR_FLAGS_SMOKE if smoke else CNN_COMPRESSOR_FLAGS
    # Stage A trains a compressor but skips NDE. We still need to pass
    # NDE-budget flags (the runner parses them); they're harmless because
    # --exit-after-compress returns before NDE training starts. Pin to the
    # smoke variant to keep argparse happy with no side effects.
    nde_flags = NDE_BUDGET_FLAGS_SMOKE
    cmd = [
        "conda", "run", "--no-capture-output", "-n", conda_env, "python", "-u",
        str(runner),
        *SHARED_FLAGS,
        *comp_flags,
        "--compressor-arch", arch,
        "--compressor-dim", str(dim),
        *nde_flags,
        "--exit-after-compress",
        "--seed", str(COMPRESSOR_SEED),
        "--save-dir", str(save_dir),
        "--cache-dir", str(cache_dir),
        # Posterior / figure paths are unused by Stage A but the runner's
        # argparse defaults would write to repo-relative paths; pin them.
        "--posterior-out", str(shared_root / f"{tag}_unused.npy"),
        "--figure-out", str(shared_root / f"{tag}_unused.pdf"),
    ]
    return Job(
        arm=arm,
        seed=COMPRESSOR_SEED,
        dim=dim,
        runner=runner,
        command=cmd,
        log_path=log_path,
        run_dir=shared_root,
        posterior_out=shared_root / f"{tag}_unused.npy",
        figure_out=shared_root / f"{tag}_unused.pdf",
        stage="stage_a",
        shared_save_dir=save_dir,
        shared_cache_dir=cache_dir,
    )


def _find_latest_compressor_checkpoint(save_dir: Path) -> Tuple[Path, Path]:
    """Locate the highest-step compressor checkpoint under `save_dir`.

    train_compressor_vmim writes into a nested
    `vmim/<map_kind>/sigma_*/gal_density_*/bin_*/harmonic_*_ch*/` subtree, so
    we search recursively.
    """
    candidates = sorted(
        save_dir.rglob("params_nd_compressor_batch*.pkl"),
        key=lambda p: int(p.stem.split("batch")[-1]),
    )
    if not candidates:
        raise FileNotFoundError(
            f"No params_nd_compressor_batch*.pkl found under {save_dir}"
        )
    params_path = candidates[-1]
    last_step = int(params_path.stem.split("batch")[-1])
    state_path = params_path.parent / f"opt_state_resnet_batch{last_step}.pkl"
    if not state_path.exists():
        raise FileNotFoundError(
            f"Stage A produced params at {params_path} but state {state_path} "
            f"is missing — compressor checkpoint pair incomplete."
        )
    return params_path, state_path


def _build_cnn_stage_b_job(
    arm: str,
    seed: int,
    dim: int,
    arch: str,
    output_root: Path,
    conda_env: str,
    smoke: bool,
    compressor_params_path: Path,
    compressor_state_path: Path,
    shared_cache_dir: Path,
) -> Job:
    """Stage B: NDE training and sampling for one seed, reusing Stage A's
    compressor params + compressed dataset cache."""
    run_dir = _arm_run_dir(arm, seed, dim, output_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_dir = run_dir / "save_params"
    posteriors_dir = output_root / arm / f"dim_{dim}" / "posteriors"
    figures_dir = output_root / arm / f"dim_{dim}" / "figures"
    log_dir = output_root / arm / f"dim_{dim}" / "logs"
    for d in (save_dir, posteriors_dir, figures_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)
    tag = _arm_tag(arm, seed, dim)
    posterior_out = posteriors_dir / f"{tag}.npy"
    figure_out = figures_dir / f"{tag}.pdf"
    log_path = log_dir / f"{tag}.log"

    runner = SBI_DIR / "npe_cnn_nbody_tomo.py"
    nde_flags = NDE_BUDGET_FLAGS_SMOKE if smoke else NDE_BUDGET_FLAGS
    # Stage B reuses the Stage A compressor: drop --train-compressor; replace
    # the compressor-params / -state defaults with the produced paths; reuse
    # the shared cache_dir so compressed datasets are loaded (or recomputed
    # against the same compressor if metadata diverges).
    nontraining_comp_flags = [
        f for f in (CNN_COMPRESSOR_FLAGS_SMOKE if smoke else CNN_COMPRESSOR_FLAGS)
        if f != "--train-compressor"
    ]
    cmd = [
        "conda", "run", "--no-capture-output", "-n", conda_env, "python", "-u",
        str(runner),
        *SHARED_FLAGS,
        *nontraining_comp_flags,
        "--compressor-arch", arch,
        "--compressor-dim", str(dim),
        "--compressor-params", str(compressor_params_path),
        "--compressor-state", str(compressor_state_path),
        *nde_flags,
        "--seed", str(seed),
        "--save-dir", str(save_dir),
        "--cache-dir", str(shared_cache_dir),
        "--posterior-out", str(posterior_out),
        "--figure-out", str(figure_out),
    ]
    return Job(
        arm=arm,
        seed=seed,
        dim=dim,
        runner=runner,
        command=cmd,
        log_path=log_path,
        run_dir=run_dir,
        posterior_out=posterior_out,
        figure_out=figure_out,
        stage="stage_b",
        shared_save_dir=compressor_params_path.parent,
        shared_cache_dir=shared_cache_dir,
    )


ARMS: Dict[str, Dict[str, object]] = {
    "l1_cross_only": {
        "seeds": (41, 42, 43),
        "dims": (None,),
        "arch": None,
        "kind": "l1",
    },
    "cnn_cross_only_plain": {
        "seeds": (41, 42, 43),
        "dims": (10, 20),
        "arch": "plain",
        "kind": "cnn",
    },
    "cnn_cross_only_resnet50_gn": {
        "seeds": (41, 42, 43),
        "dims": (10, 20),
        "arch": "resnet50_gn",
        "kind": "cnn",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--conda-env", type=str, default="jaxili")
    p.add_argument("--gpus", type=str, default="0,1,2")
    p.add_argument(
        "--xla-mem-fraction",
        type=float,
        default=None,
        help=(
            "Fallback fraction used when nvidia-smi probe fails. If unset, "
            "default is 0.80."
        ),
    )
    p.add_argument(
        "--xla-mem-fraction-by-gpu",
        type=str,
        default="",
        help="Per-GPU overrides, e.g. '0:0.45,1:0.85'.",
    )
    p.add_argument(
        "--no-auto-mem-fraction",
        action="store_true",
        help="Disable the nvidia-smi free-memory probe (use fixed fraction).",
    )
    p.add_argument(
        "--mem-fraction-margin",
        type=float,
        default=0.05,
        help="Safety margin subtracted from the probed free fraction.",
    )
    p.add_argument(
        "--mem-fraction-cap",
        type=float,
        default=0.90,
        help="Maximum XLA_PYTHON_CLIENT_MEM_FRACTION.",
    )
    p.add_argument(
        "--arms",
        type=str,
        default=",".join(ARMS.keys()),
        help="Comma-separated subset of arms to run.",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Optional comma-separated seed subset.",
    )
    p.add_argument(
        "--dims",
        type=str,
        default="",
        help="Optional comma-separated compressor-dim subset (CNN arms only).",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny budget: 1 seed per arm, 2-step compressor/NDE.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip (arm, seed, dim) whose posterior.npy already exists.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the dispatch plan and per-job command without launching.",
    )
    p.add_argument(
        "--cross-noise-model",
        type=str,
        default="auto_scalar",
        choices=["auto_scalar", "channel_empirical_global"],
        help=(
            "L1 cross-channel noise model. 'auto_scalar' (default) keeps the "
            "v1 behavior. 'channel_empirical_global' applies the channel-aware "
            "σ scaling that spreads the cross-channel SNR distribution across "
            "all L1 bins. Appended to L1-arm runner args only; CNN unaffected."
        ),
    )
    return p.parse_args()


def _parse_gpu_mem_overrides(spec: str, gpus: List[str]) -> Dict[str, float]:
    if not spec.strip():
        return {}
    valid = set(gpus)
    out: Dict[str, float] = {}
    for tok in (t.strip() for t in spec.split(",") if t.strip()):
        gpu_id, frac_str = tok.split(":", 1)
        if gpu_id not in valid:
            raise ValueError(f"GPU '{gpu_id}' not in --gpus {gpus}")
        frac = float(frac_str)
        if not (0.0 < frac <= 1.0):
            raise ValueError(f"Memory fraction {frac} out of (0,1]")
        out[gpu_id] = frac
    return out


def _probe_gpu_mem_fraction(
    gpu_id: str,
    margin: float,
    cap: float,
    fallback: float,
) -> float:
    """Query nvidia-smi for free/total memory on `gpu_id` and return a fraction."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        line = out.stdout.strip().splitlines()[0]
        free_mb_s, total_mb_s = (s.strip() for s in line.split(","))
        free_mb = float(free_mb_s)
        total_mb = float(total_mb_s)
        if total_mb <= 0:
            return fallback
        frac = free_mb / total_mb - margin
        frac = max(0.10, min(cap, frac))
        return float(frac)
    except (subprocess.SubprocessError, FileNotFoundError, ValueError, IndexError):
        return fallback


def build_phase1_jobs(args: argparse.Namespace) -> List[Job]:
    """Build phase-1 jobs: L1 (single-stage) + CNN Stage A (one per arm,dim).

    Stage B jobs are built later, after phase 1 finishes, because they need
    the produced compressor-checkpoint paths.
    """
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    selected_arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seed_filter: Optional[set] = None
    if args.seeds.strip():
        seed_filter = {int(s) for s in args.seeds.split(",") if s.strip()}
    dim_filter: Optional[set] = None
    if args.dims.strip():
        dim_filter = {int(d) for d in args.dims.split(",") if d.strip()}

    jobs: List[Job] = []
    for arm in selected_arms:
        if arm not in ARMS:
            raise ValueError(f"Unknown arm '{arm}'. Available: {list(ARMS)}")
        cfg = ARMS[arm]
        seeds = cfg["seeds"]
        dims = cfg["dims"]
        if args.smoke:
            seeds = seeds[:1]
            if dims != (None,):
                dims = dims[:1]
        if cfg["kind"] == "l1":
            for seed in seeds:
                if seed_filter is not None and seed not in seed_filter:
                    continue
                job = _build_l1_job(
                    seed=seed,
                    output_root=output_root,
                    conda_env=args.conda_env,
                    smoke=args.smoke,
                    cross_noise_model=str(args.cross_noise_model),
                )
                if args.skip_existing and job.posterior_out.exists():
                    print(f"[skip] {job.arm} seed={seed}: posterior exists")
                    continue
                jobs.append(job)
        else:  # cnn
            for dim in dims:
                if dim_filter is not None and dim is not None and dim not in dim_filter:
                    continue
                stage_a = _build_cnn_stage_a_job(
                    arm=arm,
                    dim=int(dim),
                    arch=str(cfg["arch"]),
                    output_root=output_root,
                    conda_env=args.conda_env,
                    smoke=args.smoke,
                )
                if args.skip_existing and stage_a.shared_save_dir is not None:
                    try:
                        _find_latest_compressor_checkpoint(stage_a.shared_save_dir)
                        print(
                            f"[skip] {stage_a.arm} dim={dim}: compressor checkpoint "
                            f"already present under {stage_a.shared_save_dir}"
                        )
                        continue
                    except FileNotFoundError:
                        pass
                jobs.append(stage_a)
    return jobs


def build_stage_b_jobs(
    args: argparse.Namespace,
    phase1_results: List[Dict[str, object]],
) -> List[Job]:
    """Build Stage B (CNN NDE) jobs from Stage A's produced checkpoints."""
    output_root = args.output_root.resolve()
    selected_arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seed_filter: Optional[set] = None
    if args.seeds.strip():
        seed_filter = {int(s) for s in args.seeds.split(",") if s.strip()}
    dim_filter: Optional[set] = None
    if args.dims.strip():
        dim_filter = {int(d) for d in args.dims.split(",") if d.strip()}

    # Map from (arm, dim) -> (compressor_params_path, compressor_state_path, shared_cache_dir).
    stage_a_outputs: Dict[Tuple[str, int], Tuple[Path, Path, Path]] = {}
    for arm in selected_arms:
        if arm not in ARMS:
            continue
        cfg = ARMS[arm]
        if cfg["kind"] != "cnn":
            continue
        dims = cfg["dims"]
        if args.smoke:
            dims = dims[:1]
        for dim in dims:
            if dim_filter is not None and dim is not None and dim not in dim_filter:
                continue
            shared_root = _shared_compressor_dir(arm, int(dim), output_root)
            save_dir = shared_root / "save_params"
            cache_dir = shared_root / "cache"
            try:
                params_path, state_path = _find_latest_compressor_checkpoint(save_dir)
            except FileNotFoundError as e:
                print(f"[stage_b] {arm} dim={dim}: SKIP — {e}")
                continue
            stage_a_outputs[(arm, int(dim))] = (params_path, state_path, cache_dir)

    jobs: List[Job] = []
    for arm in selected_arms:
        if arm not in ARMS:
            continue
        cfg = ARMS[arm]
        if cfg["kind"] != "cnn":
            continue
        seeds = cfg["seeds"]
        dims = cfg["dims"]
        if args.smoke:
            seeds = seeds[:1]
            if dims != (None,):
                dims = dims[:1]
        for dim in dims:
            if dim_filter is not None and dim is not None and dim not in dim_filter:
                continue
            key = (arm, int(dim))
            if key not in stage_a_outputs:
                continue
            params_path, state_path, cache_dir = stage_a_outputs[key]
            for seed in seeds:
                if seed_filter is not None and seed not in seed_filter:
                    continue
                job = _build_cnn_stage_b_job(
                    arm=arm,
                    seed=int(seed),
                    dim=int(dim),
                    arch=str(cfg["arch"]),
                    output_root=output_root,
                    conda_env=args.conda_env,
                    smoke=args.smoke,
                    compressor_params_path=params_path,
                    compressor_state_path=state_path,
                    shared_cache_dir=cache_dir,
                )
                if args.skip_existing and job.posterior_out.exists():
                    print(f"[skip] {job.arm} seed={seed} dim={dim}: posterior exists")
                    continue
                jobs.append(job)
    return jobs


def run_jobs_parallel(
    jobs: List[Job],
    gpus: List[str],
    mem_overrides: Dict[str, float],
    fallback_mem: float,
    auto_mem: bool,
    margin: float,
    cap: float,
    dry_run: bool,
    progress_path: Path,
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
            if gpu_id in mem_overrides:
                mem_frac = mem_overrides[gpu_id]
            elif auto_mem:
                mem_frac = _probe_gpu_mem_fraction(
                    gpu_id=gpu_id,
                    margin=margin,
                    cap=cap,
                    fallback=fallback_mem,
                )
            else:
                mem_frac = fallback_mem

            cmd = list(job.command) + ["--cuda-visible-devices", gpu_id]
            t0 = time.time()
            if dry_run:
                rc = 0
                job.log_path.write_text(
                    f"[dry-run] gpu={gpu_id} mem={mem_frac}\n"
                    + " ".join(cmd) + "\n",
                    encoding="utf-8",
                )
            else:
                env = dict(os.environ)
                env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{mem_frac:.4f}"
                env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
                with open(job.log_path, "w", encoding="utf-8") as logf:
                    logf.write(
                        f"# gpu={gpu_id} mem_fraction={mem_frac:.4f} "
                        f"started_utc={datetime.now(timezone.utc).isoformat()}\n"
                    )
                    logf.write("# cmd:\n#   " + " ".join(cmd) + "\n\n")
                    logf.flush()
                    proc = subprocess.run(
                        cmd,
                        cwd=str(REPO_ROOT),
                        env=env,
                        stdout=logf,
                        stderr=subprocess.STDOUT,
                    )
                    rc = proc.returncode
            dt = time.time() - t0
            entry = {
                "arm": job.arm,
                "seed": int(job.seed),
                "dim": (None if job.dim is None else int(job.dim)),
                "gpu": gpu_id,
                "returncode": int(rc),
                "seconds": float(dt),
                "log": str(job.log_path),
                "run_dir": str(job.run_dir),
                "posterior_out": str(job.posterior_out),
                "posterior_exists": bool(job.posterior_out.exists()),
                "xla_mem_fraction": float(mem_frac),
                "cmd": cmd,
            }
            with lock:
                results.append(entry)
                progress_path.write_text(
                    json.dumps(results, indent=2),
                    encoding="utf-8",
                )
                print(
                    f"[{job.arm} seed={job.seed} dim={job.dim}] "
                    f"gpu={gpu_id} rc={rc} in {dt:.1f}s "
                    f"-> posterior_exists={job.posterior_out.exists()}"
                )
            q.task_done()

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip()
    except subprocess.SubprocessError:
        return "unknown"


def _cache_manifest_sha(cache_dir: Path) -> str:
    manifest = cache_dir / "manifest.json"
    if not manifest.exists():
        return "missing"
    try:
        import hashlib
        return hashlib.sha256(manifest.read_bytes()).hexdigest()
    except Exception:
        return "unknown"


def main() -> None:
    args = parse_args()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    mem_overrides = _parse_gpu_mem_overrides(args.xla_mem_fraction_by_gpu, gpus)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_dir = output_root / ".status"
    status_dir.mkdir(parents=True, exist_ok=True)

    fallback_mem = args.xla_mem_fraction if args.xla_mem_fraction is not None else 0.80
    auto_mem = not args.no_auto_mem_fraction

    # ----- Phase 1: L1 jobs + CNN Stage A (shared compressors) -----
    phase1_jobs = build_phase1_jobs(args)
    if not phase1_jobs:
        print("[campaign] No phase-1 jobs to run.")
        return

    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "cache_dir": str(HARM_CACHE),
        "cache_manifest_sha256": _cache_manifest_sha(HARM_CACHE),
        "shared_flags": SHARED_FLAGS,
        "nde_budget_flags": NDE_BUDGET_FLAGS,
        "cnn_compressor_flags": CNN_COMPRESSOR_FLAGS,
        "l1_flags": L1_FLAGS,
        "compressor_seed": int(COMPRESSOR_SEED),
        "arms": {
            arm: {
                "seeds": list(cfg["seeds"]),
                "dims": [None if d is None else int(d) for d in cfg["dims"]],
                "arch": cfg["arch"],
                "kind": cfg["kind"],
            }
            for arm, cfg in ARMS.items()
        },
        "gpus": gpus,
        "xla_mem_fraction_fallback": float(fallback_mem),
        "xla_mem_fraction_by_gpu": mem_overrides,
        "auto_mem_fraction": bool(auto_mem),
        "mem_fraction_margin": float(args.mem_fraction_margin),
        "mem_fraction_cap": float(args.mem_fraction_cap),
        "smoke": bool(args.smoke),
        "phase1_n_jobs": len(phase1_jobs),
        "phase1_jobs": [
            {
                "arm": j.arm,
                "seed": int(j.seed),
                "dim": (None if j.dim is None else int(j.dim)),
                "stage": j.stage,
                "run_dir": str(j.run_dir),
                "posterior_out": str(j.posterior_out),
                "figure_out": str(j.figure_out),
                "log_path": str(j.log_path),
                "cmd": j.command,
            }
            for j in phase1_jobs
        ],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(
        f"[campaign] Phase 1: {len(phase1_jobs)} jobs "
        f"(L1 single-stage + CNN Stage A shared compressors) "
        f"across {len(gpus)} GPU(s) (smoke={args.smoke}, dry_run={args.dry_run})"
    )
    phase1_progress = output_root / "campaign_progress_phase1.json"
    phase1_results = run_jobs_parallel(
        jobs=phase1_jobs,
        gpus=gpus,
        mem_overrides=mem_overrides,
        fallback_mem=fallback_mem,
        auto_mem=auto_mem,
        margin=args.mem_fraction_margin,
        cap=args.mem_fraction_cap,
        dry_run=args.dry_run,
        progress_path=phase1_progress,
    )
    phase1_failed = [r for r in phase1_results if int(r["returncode"]) != 0]
    print(
        f"[campaign] Phase 1 done. "
        f"{len(phase1_results) - len(phase1_failed)}/{len(phase1_results)} succeeded."
    )

    # ----- Phase 2: CNN Stage B (NDE per seed, reusing Stage A compressors) -----
    if args.dry_run:
        print("[campaign] Dry-run: skipping Phase 2 (Stage B requires Stage A artifacts).")
        phase2_results: List[Dict[str, object]] = []
    else:
        phase2_jobs = build_stage_b_jobs(args, phase1_results)
        if not phase2_jobs:
            print("[campaign] Phase 2: no Stage B jobs to run.")
            phase2_results = []
        else:
            manifest["phase2_n_jobs"] = len(phase2_jobs)
            manifest["phase2_jobs"] = [
                {
                    "arm": j.arm,
                    "seed": int(j.seed),
                    "dim": (None if j.dim is None else int(j.dim)),
                    "stage": j.stage,
                    "run_dir": str(j.run_dir),
                    "posterior_out": str(j.posterior_out),
                    "figure_out": str(j.figure_out),
                    "log_path": str(j.log_path),
                    "compressor_params": str(j.shared_save_dir),
                    "shared_cache_dir": str(j.shared_cache_dir),
                    "cmd": j.command,
                }
                for j in phase2_jobs
            ]
            (output_root / "manifest.json").write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )

            print(
                f"[campaign] Phase 2: {len(phase2_jobs)} jobs "
                f"(CNN Stage B NDE per seed) across {len(gpus)} GPU(s)"
            )
            phase2_progress = output_root / "campaign_progress_phase2.json"
            phase2_results = run_jobs_parallel(
                jobs=phase2_jobs,
                gpus=gpus,
                mem_overrides=mem_overrides,
                fallback_mem=fallback_mem,
                auto_mem=auto_mem,
                margin=args.mem_fraction_margin,
                cap=args.mem_fraction_cap,
                dry_run=args.dry_run,
                progress_path=phase2_progress,
            )

    results = list(phase1_results) + list(phase2_results)
    failed = [r for r in results if int(r["returncode"]) != 0]
    summary = {
        "started_utc": manifest["started_utc"],
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "n_jobs": len(results),
        "n_failed": len(failed),
        "gpus": gpus,
        "phase1_results": phase1_results,
        "phase2_results": phase2_results,
    }
    (output_root / "campaign_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(
        f"[campaign] All phases done. "
        f"{len(results) - len(failed)}/{len(results)} succeeded. "
        f"manifest = {output_root / 'manifest.json'}"
    )
    if failed:
        for r in failed:
            print(
                f"  FAIL {r['arm']} seed={r['seed']} dim={r['dim']} "
                f"rc={r['returncode']} log={r['log']}"
            )


if __name__ == "__main__":
    main()
