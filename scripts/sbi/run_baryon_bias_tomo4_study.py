#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import queue
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


DEFAULT_REPO_ROOT = str(Path(__file__).resolve().parents[2])
DEFAULT_OUTPUT_ROOT = str(Path(DEFAULT_REPO_ROOT) / "scripts" / "sbi" / "baryon_bias_tomo4_study")
DEFAULT_FIDUCIAL_ROOT = "/home/tersenov/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial"
DEFAULT_PERM_INDICES = ",".join(str(i) for i in range(20))
TRUTH = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493], dtype=np.float64)


@dataclass
class Job:
    name: str
    command: List[str]
    log_path: Path
    lock_key: Optional[str] = None


def _csv_tokens(value: str) -> List[str]:
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def _csv_ints(value: str) -> List[int]:
    return [int(tok) for tok in _csv_tokens(value)]


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"batch(\d+)\.pkl$", path.name)
    return int(match.group(1)) if match is not None else -1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run tomo4 baryon-bias inference: baryonified fiducial observations "
            "with no-bary-trained checkpoints (CNN, L1-jaxili, L1-VMIM)."
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
        "--perm-indices",
        type=str,
        default=DEFAULT_PERM_INDICES,
        help="Comma-separated fiducial permutation indices, e.g. '0,1,2,3'.",
    )
    p.add_argument("--fiducial-root", type=str, default=DEFAULT_FIDUCIAL_ROOT)
    p.add_argument(
        "--fiducial-map-name",
        type=str,
        default="projected_probes_maps_baryonified512.h5",
    )
    p.add_argument("--conda-env", type=str, default="jaxili")
    p.add_argument("--map-kind", type=str, default="nbody")
    p.add_argument("--tfds-name", type=str, default="NbodyCosmogridDatasetTomo/grid_20deg_160px")
    p.add_argument("--field-size", type=int, default=20)
    p.add_argument("--field-npix", type=int, default=160)
    p.add_argument("--nbins", type=int, default=4)
    p.add_argument("--tomo-bin-indices", type=str, default="1,2,3,4")
    p.add_argument("--npe-samples", type=int, default=100000)
    p.add_argument("--flow-save-every", type=int, default=500)
    p.add_argument("--flow-patience", type=int, default=30)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    # Reuse roots for no-bary trained artifacts.
    p.add_argument(
        "--cnn-baseline-root",
        type=str,
        default="scripts/sbi/nobnt_tomo_bins_crosscorr_study",
    )
    p.add_argument(
        "--l1-baseline-root",
        type=str,
        default="scripts/sbi/nobnt_tomo_bins_crosscorr_study_l1_jaxili_bestcfg",
    )
    p.add_argument(
        "--l1vmim-baseline-root",
        type=str,
        default="scripts/sbi/nobnt_tomo_bins_crosscorr_study",
    )

    # CNN (validated no-bary config).
    p.add_argument("--cnn-compressor-steps", type=int, default=60000)
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
    p.add_argument("--cnn-summary-clip-value", type=float, default=0.0)
    p.add_argument("--cnn-standardize-summary", action="store_true")

    # L1-jaxili (no PCA).
    p.add_argument("--l1-pca-components", type=int, default=0)
    p.add_argument("--n-scales", type=int, default=5)
    p.add_argument("--l1-nbins", type=int, default=40)
    p.add_argument("--l1-min-snr", type=float, default=-13.0)
    p.add_argument("--l1-max-snr", type=float, default=13.0)
    p.add_argument("--l1-summary-transform", type=str, default="log1p-zscore")
    p.add_argument("--l1-clip-value", type=float, default=5.0)
    p.add_argument("--l1-learning-rate", type=float, default=3e-4)
    p.add_argument("--l1-epochs", type=int, default=5000)
    p.add_argument("--l1-batch-size", type=int, default=128)
    p.add_argument("--l1-ds-batch-size", type=int, default=96)
    p.add_argument("--l1-checkpoint-name", type=str, default="params_l1norm_jaxili")

    # L1-VMIM (validated no-bary config).
    p.add_argument("--vmim-compressor-dim", type=int, default=64)
    p.add_argument("--vmim-compressor-hidden", type=str, default="768,768")
    p.add_argument("--vmim-compressor-nf-layers", type=int, default=10)
    p.add_argument("--vmim-compressor-nf-hidden", type=int, default=384)
    p.add_argument("--vmim-compressor-input-clip", type=float, default=6.0)
    p.add_argument("--vmim-flow-steps", type=int, default=12000)
    p.add_argument("--vmim-batch-size", type=int, default=256)
    p.add_argument("--vmim-nvp-layers", type=int, default=4)
    p.add_argument("--vmim-nvp-hidden", type=int, default=128)
    p.add_argument("--vmim-weight-decay", type=float, default=1e-4)
    p.add_argument("--vmim-grad-clip", type=float, default=1.0)
    p.add_argument("--vmim-summary-clip-value", type=float, default=0.0)
    p.add_argument("--vmim-standardize-summary", action="store_true", default=True)
    p.add_argument(
        "--stage-vmim-compressor-input-stats",
        action="store_true",
        default=True,
        help=(
            "Copy l1_vmim_compressor_input_standardization.npz from the no-bary "
            "compressor root into per-seed eval save dirs if missing."
        ),
    )
    return p.parse_args()


def run_jobs_parallel(
    jobs: List[Job],
    gpus: List[str],
    cwd: Path,
    dry_run: bool = False,
) -> List[Dict[str, object]]:
    q: queue.Queue[Job] = queue.Queue()
    for j in jobs:
        q.put(j)

    results: List[Dict[str, object]] = []
    result_lock = threading.Lock()
    group_lock_guard = threading.Lock()
    group_locks: Dict[str, threading.Lock] = {}

    def get_group_lock(key: str) -> threading.Lock:
        with group_lock_guard:
            if key not in group_locks:
                group_locks[key] = threading.Lock()
            return group_locks[key]

    def worker(gpu_id: str) -> None:
        while True:
            try:
                job = q.get_nowait()
            except queue.Empty:
                break

            job.log_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [str(x) for x in job.command] + ["--cuda-visible-devices", str(gpu_id)]

            lock_ctx = get_group_lock(job.lock_key) if job.lock_key else None
            t0 = time.time()
            if lock_ctx is not None:
                lock_ctx.acquire()
            try:
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
            finally:
                if lock_ctx is not None:
                    lock_ctx.release()

            dt = time.time() - t0
            with result_lock:
                results.append(
                    {
                        "name": job.name,
                        "gpu": gpu_id,
                        "returncode": rc,
                        "seconds": dt,
                        "log": str(job.log_path),
                        "cmd": cmd,
                        "lock_key": job.lock_key,
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
    failed = [r for r in results if int(r.get("returncode", 1)) != 0]
    if failed:
        first = failed[0]
        raise RuntimeError(
            f"{context} failed for {len(failed)}/{len(results)} jobs. "
            f"First failure: {first.get('name')} (log: {first.get('log')})"
        )


def _require_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _resolve_cnn_compressor_paths(baseline_root: Path, requested_step: int) -> Dict[str, Path]:
    base = (
        baseline_root
        / "cnn_compressor"
        / "tomo4_20deg160"
        / "vmim"
        / "nbody"
        / "sigma_0.26"
        / "gal_density_30"
        / "bin_4"
    )
    params = base / f"params_nd_compressor_batch{requested_step}.pkl"
    state = base / f"opt_state_resnet_batch{requested_step}.pkl"
    if params.exists() and state.exists():
        return {"params": params, "state": state}

    candidates = sorted(
        base.glob("params_nd_compressor_batch*.pkl"),
        key=_checkpoint_step,
    )
    for cand in reversed(candidates):
        step = _checkpoint_step(cand)
        st = base / f"opt_state_resnet_batch{step}.pkl"
        if st.exists():
            return {"params": cand, "state": st}
    raise FileNotFoundError(
        f"Could not resolve CNN compressor params/state under {base} "
        f"(requested step={requested_step})."
    )


def _resolve_l1vmim_compressor_paths(baseline_root: Path) -> Dict[str, Path]:
    base = (
        baseline_root
        / "l1vmim_compressor"
        / "tomo4_20deg160"
        / "vmim_l1"
        / "nbody"
        / "sigma_0.26"
        / "gal_density_30"
        / "bin_4"
    )
    params = base / "params_nd_compressor_best.pkl"
    state = base / "opt_state_resnet_best.pkl"
    _require_exists(params, "L1-VMIM compressor params")
    _require_exists(state, "L1-VMIM compressor state")
    return {"params": params, "state": state}


def _stage_vmim_input_stats(
    baseline_root: Path,
    seed_eval_dir: Path,
) -> Optional[Path]:
    src = (
        baseline_root
        / "l1vmim_compressor"
        / "tomo4_20deg160"
        / "l1_vmim_jaxili"
        / "nbody"
        / "l1_vmim_compressor_input_standardization.npz"
    )
    if not src.exists():
        return None

    dst_dir = seed_eval_dir / "l1_vmim_jaxili" / "nbody"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "l1_vmim_compressor_input_standardization.npz"
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def _summarize_posteriors(out_root: Path) -> None:
    rows: List[Dict[str, object]] = []
    for npy in sorted((out_root / "posteriors").glob("*.npy")):
        s = np.load(npy)
        rows.append(
            {
                "file": npy.name,
                "n_samples": int(s.shape[0]),
                "n_dim": int(s.shape[1]) if s.ndim == 2 else None,
                "bias_l2_all6": float(np.linalg.norm(np.mean(s, axis=0) - TRUTH)) if s.ndim == 2 else None,
            }
        )
    (out_root / "posterior_summary.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    script_dir = repo_root / "scripts" / "sbi"
    cnn_script = str(script_dir / "npe_cnn_jaxili_nbody_tomo.py")
    l1_script = str(script_dir / "npe_l1norm_jaxili_nbody_tomo.py")
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
    perm_indices = sorted(set(_csv_ints(args.perm_indices)))
    if not perm_indices:
        raise ValueError("--perm-indices cannot be empty.")

    fid_root = Path(args.fiducial_root).resolve()
    cnn_baseline_root = (repo_root / args.cnn_baseline_root).resolve()
    l1_baseline_root = (repo_root / args.l1_baseline_root).resolve()
    l1vmim_baseline_root = (repo_root / args.l1vmim_baseline_root).resolve()

    # Validate permutation maps exist.
    perm_map_paths: Dict[int, Path] = {}
    for perm in perm_indices:
        mp = fid_root / f"perm_{perm:04d}" / args.fiducial_map_name
        _require_exists(mp, f"baryonified fiducial map for perm_{perm:04d}")
        perm_map_paths[perm] = mp

    cnn_comp: Optional[Dict[str, Path]] = None
    vmim_comp: Optional[Dict[str, Path]] = None
    if "cnn" in methods:
        cnn_comp = _resolve_cnn_compressor_paths(cnn_baseline_root, args.cnn_compressor_steps)
    if "l1vmim" in methods:
        vmim_comp = _resolve_l1vmim_compressor_paths(l1vmim_baseline_root)

    # Validate no-train baseline artifacts per seed.
    for seed in seeds:
        if "cnn" in methods:
            cnn_seed_dir = (
                cnn_baseline_root
                / "cnn_eval"
                / "tomo4_20deg160"
                / f"seed_{seed}"
                / "cnn_jaxili"
                / "nbody"
            )
            _require_exists(
                cnn_seed_dir / "params_cnn_jaxili",
                f"CNN-jaxili checkpoint root for seed {seed}",
            )

        if "l1" in methods:
            l1_seed_dir = l1_baseline_root / "l1_eval" / "tomo4_20deg160" / f"seed_{seed}" / "l1norm_jaxili" / "nbody"
            _require_exists(l1_seed_dir / "l1_jaxili_standardization.npz", f"L1-jaxili preprocessing stats for seed {seed}")
            _require_exists(l1_seed_dir / "l1_jaxili_feature_mask.npz", f"L1-jaxili feature mask for seed {seed}")
            _require_exists(l1_seed_dir / "params_l1norm_jaxili", f"L1-jaxili checkpoint root for seed {seed}")

        if "l1vmim" in methods:
            vm_seed_eval = l1vmim_baseline_root / "l1vmim_eval" / "tomo4_20deg160" / f"seed_{seed}"
            vm_seed_flow = vm_seed_eval / "l1_vmim_jaxili" / "nbody"
            _require_exists(
                vm_seed_flow / "params_l1vmim_jaxili",
                f"L1-VMIM-jaxili checkpoint root for seed {seed}",
            )
            if args.vmim_standardize_summary:
                _require_exists(vm_seed_flow / "l1_vmim_summary_standardization.npz", f"L1-VMIM summary standardization stats for seed {seed}")
            if args.stage_vmim_compressor_input_stats:
                staged = _stage_vmim_input_stats(l1vmim_baseline_root, vm_seed_eval)
                if staged is None:
                    raise FileNotFoundError(
                        "Missing source L1-VMIM compressor input stats under "
                        f"{l1vmim_baseline_root / 'l1vmim_compressor' / 'tomo4_20deg160' / 'l1_vmim_jaxili' / 'nbody'}"
                    )
            if args.stage_vmim_compressor_input_stats:
                _require_exists(
                    vm_seed_flow / "l1_vmim_compressor_input_standardization.npz",
                    f"L1-VMIM compressor input stats for seed {seed}",
                )

    no_bary_baselines = {
        "cnn": {seed: str((cnn_baseline_root / "posteriors" / f"cnn_tomo4_20deg160_nobnt_s{seed}.npy").resolve()) for seed in seeds},
        "l1": {seed: str((l1_baseline_root / "posteriors" / f"l1_tomo4_20deg160_nobnt_s{seed}.npy").resolve()) for seed in seeds},
        "l1vmim": {seed: str((l1vmim_baseline_root / "posteriors" / f"l1vmim_tomo4_20deg160_nobnt_s{seed}.npy").resolve()) for seed in seeds},
    }

    manifest = {
        "repo_root": str(repo_root),
        "output_root": str(out_root),
        "study_type": "baryon_bias_tomo4_no_bary_training",
        "methods": methods,
        "seeds": seeds,
        "perm_indices": perm_indices,
        "gpus": gpus,
        "map_kind": args.map_kind,
        "tfds_name": args.tfds_name,
        "field_size": args.field_size,
        "field_npix": args.field_npix,
        "nbins": args.nbins,
        "tomo_bin_indices": args.tomo_bin_indices,
        "npe_samples": args.npe_samples,
        "fiducial_root": str(fid_root),
        "fiducial_map_name": args.fiducial_map_name,
        "perm_map_paths": {str(k): str(v) for k, v in perm_map_paths.items()},
        "baseline_roots": {
            "cnn": str(cnn_baseline_root),
            "l1": str(l1_baseline_root),
            "l1vmim": str(l1vmim_baseline_root),
        },
        "no_bary_posterior_baselines": no_bary_baselines,
        "cnn_compressor_paths": (
            {k: str(v) for k, v in cnn_comp.items()} if cnn_comp is not None else None
        ),
        "l1vmim_compressor_paths": (
            {k: str(v) for k, v in vmim_comp.items()} if vmim_comp is not None else None
        ),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    jobs: List[Job] = []
    for perm in perm_indices:
        fid_map = perm_map_paths[perm]
        for seed in seeds:
            tag = f"tomo4_20deg160_bary_perm{perm:04d}_s{seed}"

            if "cnn" in methods:
                if cnn_comp is None:
                    raise RuntimeError("Internal error: CNN compressor paths were not resolved.")
                posterior_out = out_root / "posteriors" / f"cnn_{tag}.npy"
                cnn_seed_save = cnn_baseline_root / "cnn_eval" / "tomo4_20deg160" / f"seed_{seed}"
                cmd = [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    args.conda_env,
                    "python",
                    cnn_script,
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
                    str(args.nbins),
                    "--tomo-bin-indices",
                    args.tomo_bin_indices,
                    "--fiducial-map",
                    str(fid_map),
                    "--cache-dir",
                    str(out_root / "cache" / f"cnn_seed{seed}"),
                    "--save-dir",
                    str(cnn_seed_save),
                    "--compressor-params",
                    str(cnn_comp["params"]),
                    "--compressor-state",
                    str(cnn_comp["state"]),
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
                    "--no-train",
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
                jobs.append(
                    Job(
                        name=f"cnn::perm{perm:04d}::s{seed}",
                        command=cmd,
                        log_path=out_root / "logs" / f"cnn_perm{perm:04d}_s{seed}.log",
                        lock_key=f"cnn_seed{seed}",
                    )
                )

            if "l1" in methods:
                posterior_out = out_root / "posteriors" / f"l1_{tag}.npy"
                l1_seed_save = l1_baseline_root / "l1_eval" / "tomo4_20deg160" / f"seed_{seed}"
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
                    args.tfds_name,
                    "--field-size",
                    str(args.field_size),
                    "--field-npix",
                    str(args.field_npix),
                    "--nbins",
                    str(args.nbins),
                    "--tomo-bin-indices",
                    args.tomo_bin_indices,
                    "--fiducial-map",
                    str(fid_map),
                    "--cache-dir",
                    str(out_root / "cache" / f"l1_seed{seed}"),
                    "--save-dir",
                    str(l1_seed_save),
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
                    "--summary-transform",
                    args.l1_summary_transform,
                    "--clip-value",
                    str(args.l1_clip_value),
                    "--learning-rate",
                    str(args.l1_learning_rate),
                    "--epochs",
                    str(args.l1_epochs),
                    "--batch-size",
                    str(args.l1_batch_size),
                    "--npe-samples",
                    str(args.npe_samples),
                    "--posterior-out",
                    str(posterior_out),
                    "--ds-batch-size",
                    str(args.l1_ds_batch_size),
                    "--checkpoint-name",
                    args.l1_checkpoint_name,
                    "--no-train",
                ]
                if args.plot:
                    cmd.extend(
                        [
                            "--plot",
                            "--figure-out",
                            str(out_root / "figures" / f"l1_{tag}.pdf"),
                        ]
                    )
                jobs.append(
                    Job(
                        name=f"l1::perm{perm:04d}::s{seed}",
                        command=cmd,
                        log_path=out_root / "logs" / f"l1_perm{perm:04d}_s{seed}.log",
                        lock_key=f"l1_seed{seed}",
                    )
                )

            if "l1vmim" in methods:
                if vmim_comp is None:
                    raise RuntimeError("Internal error: L1-VMIM compressor paths were not resolved.")
                posterior_out = out_root / "posteriors" / f"l1vmim_{tag}.npy"
                vm_seed_save = l1vmim_baseline_root / "l1vmim_eval" / "tomo4_20deg160" / f"seed_{seed}"
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
                    args.tfds_name,
                    "--field-size",
                    str(args.field_size),
                    "--field-npix",
                    str(args.field_npix),
                    "--nbins",
                    str(args.nbins),
                    "--tomo-bin-indices",
                    args.tomo_bin_indices,
                    "--fiducial-map",
                    str(fid_map),
                    "--n-scales",
                    str(args.n_scales),
                    "--l1-nbins",
                    str(args.l1_nbins),
                    "--l1-min-snr",
                    str(args.l1_min_snr),
                    "--l1-max-snr",
                    str(args.l1_max_snr),
                    "--cache-dir",
                    str(out_root / "cache" / f"l1vmim_seed{seed}"),
                    "--save-dir",
                    str(vm_seed_save),
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
                    str(vmim_comp["params"]),
                    "--compressor-state",
                    str(vmim_comp["state"]),
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
                    "--no-train",
                ]
                if args.vmim_standardize_summary:
                    cmd.append("--standardize-summary")
                else:
                    cmd.append("--no-standardize-summary")
                if args.plot:
                    cmd.extend(
                        [
                            "--plot",
                            "--figure-out",
                            str(out_root / "figures" / f"l1vmim_{tag}.pdf"),
                        ]
                    )
                jobs.append(
                    Job(
                        name=f"l1vmim::perm{perm:04d}::s{seed}",
                        command=cmd,
                        log_path=out_root / "logs" / f"l1vmim_perm{perm:04d}_s{seed}.log",
                        lock_key=f"l1vmim_seed{seed}",
                    )
                )

    if jobs:
        results = run_jobs_parallel(jobs, gpus, repo_root, args.dry_run)
        require_success(results, "Baryonified inference matrix")
    else:
        results = []

    (out_root / "job_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _summarize_posteriors(out_root)
    print(f"Completed baryon-bias tomo4 study. Artifacts in: {out_root}")


if __name__ == "__main__":
    main()
