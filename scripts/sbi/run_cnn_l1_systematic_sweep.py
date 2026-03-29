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
    p = argparse.ArgumentParser(description="Systematic CNN vs L1 benchmark runner")
    p.add_argument("--repo-root", type=str, default="/mnt/home/tersenov/software/cnn_sbi")
    p.add_argument("--gpus", type=str, default="0,1,2")
    p.add_argument("--map-kind", type=str, default="nbody")
    p.add_argument(
        "--methods",
        type=str,
        default="cnn,l1",
        help="Comma-separated methods to run from {cnn,cnn_jax,l1,l1_jax}.",
    )
    p.add_argument("--seeds", type=str, default="41,42,43")
    p.add_argument(
        "--compressor-steps",
        type=int,
        default=20000,
        help="Compressor training steps per variant (higher is safer for convergence).",
    )
    p.add_argument(
        "--compressor-save-every",
        type=int,
        default=2000,
        help="Validation/checkpoint cadence during compressor training.",
    )
    p.add_argument("--flow-steps", type=int, default=5000)
    p.add_argument("--npe-samples", type=int, default=20000)
    p.add_argument(
        "--eval-save-every",
        type=int,
        default=100,
        help="Validation/checkpoint cadence for eval flows.",
    )
    p.add_argument(
        "--eval-patience",
        type=int,
        default=10,
        help="Early-stopping patience in validation-check intervals.",
    )
    p.add_argument("--ds-batch-size-cnn", type=int, default=500)
    p.add_argument("--ds-batch-size-l1", type=int, default=256)
    p.add_argument(
        "--l1-min-snr",
        type=float,
        default=-10.0,
        help="Fixed minimum SNR for L1 histogram binning.",
    )
    p.add_argument(
        "--l1-max-snr",
        type=float,
        default=10.0,
        help="Fixed maximum SNR for L1 histogram binning.",
    )
    p.add_argument(
        "--plot-cnn-contours",
        action="store_true",
        help="Save CNN posterior contour figure for each eval run.",
    )
    p.add_argument(
        "--plot-l1-contours",
        action="store_true",
        help="Save L1 posterior contour figure for each eval run.",
    )
    p.add_argument("--skip-compressor-train", action="store_true")
    p.add_argument("--skip-evals", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--output-root",
        type=str,
        default="/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/systematic_runs",
    )
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
    script_dir = repo_root / "scripts" / "sbi"
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if "3" in gpus:
        raise ValueError("GPU 3 is forbidden for this sweep.")
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    methods = {m.strip() for m in args.methods.split(",") if m.strip()}
    if not methods.issubset({"cnn", "cnn_jax", "l1", "l1_jax"}):
        raise ValueError("--methods must be a subset of {cnn,cnn_jax,l1,l1_jax}.")
    if not methods:
        raise ValueError("--methods cannot be empty.")
    if args.compressor_steps < args.compressor_save_every:
        raise ValueError("--compressor-steps must be >= --compressor-save-every.")
    if args.compressor_steps < 10000 and not args.dry_run:
        print(
            "WARNING: --compressor-steps is < 10000. "
            "This often yields undertrained CNN compressors and inflated contours."
        )

    cnn_script = str(script_dir / "npe_cnn_nbody_tomo.py")
    cnn_jax_script = str(script_dir / "npe_cnn_jaxili_nbody_tomo.py")
    l1_script = str(script_dir / "npe_l1norm_nbody_tomo.py")
    l1_jax_script = str(script_dir / "npe_l1norm_jaxili_nbody_tomo.py")

    variants = [
        {
            "name": "tomo4_10deg80",
            "tfds": "NbodyCosmogridDatasetTomo/grid",
            "field_size": 10,
            "field_npix": 80,
            "tomo_bins": "1,2,3,4",
            "nbins": 4,
        },
        {
            "name": "tomo4_20deg160",
            "tfds": "NbodyCosmogridDatasetTomo/grid_20deg_160px",
            "field_size": 20,
            "field_npix": 160,
            "tomo_bins": "1,2,3,4",
            "nbins": 4,
        },
        {
            "name": "bin3_10deg80",
            "tfds": "NbodyCosmogridDatasetTomo/grid",
            "field_size": 10,
            "field_npix": 80,
            "tomo_bins": "3",
            "nbins": 1,
        },
        {
            "name": "bin3_20deg160",
            "tfds": "NbodyCosmogridDatasetTomo/grid_20deg_160px",
            "field_size": 20,
            "field_npix": 160,
            "tomo_bins": "3",
            "nbins": 1,
        },
    ]

    run_manifest: Dict[str, object] = {
        "gpus": gpus,
        "seeds": seeds,
        "methods": sorted(methods),
        "variants": variants,
        "compressor_steps": args.compressor_steps,
        "compressor_save_every": args.compressor_save_every,
        "flow_steps": args.flow_steps,
        "eval_save_every": args.eval_save_every,
        "eval_patience": args.eval_patience,
        "npe_samples": args.npe_samples,
        "l1_min_snr": args.l1_min_snr,
        "l1_max_snr": args.l1_max_snr,
    }
    (out_root / "manifest.json").write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8"
    )

    all_results: List[Dict[str, object]] = []
    compressor_paths: Dict[str, Dict[str, str]] = {}
    need_cnn = bool({"cnn", "cnn_jax"} & methods)

    if need_cnn and not args.skip_compressor_train:
        comp_jobs: List[Job] = []
        for v in variants:
            save_dir = out_root / "cnn_variant_training" / v["name"]
            cache_dir = out_root / "cache" / f"cnn_{v['name']}"
            comp_jobs.append(
                Job(
                    name=f"train_compressor::{v['name']}",
                    log_path=out_root / "logs" / f"train_compressor_{v['name']}.log",
                    command=[
                        "python",
                        cnn_script,
                        "--no-wandb",
                        "--map-kind",
                        args.map_kind,
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
                        "--cache-dir",
                        str(cache_dir),
                        "--save-dir",
                        str(save_dir),
                        "--train-compressor",
                        "--compressor-steps",
                        str(args.compressor_steps),
                        "--compressor-save-every",
                        str(args.compressor_save_every),
                        "--total-steps",
                        "1",
                        "--save-every",
                        "1",
                        "--no-sample",
                    ],
                )
            )
        # Run compressor jobs sequentially to avoid TFDS write races when variants
        # share the same dataset config.
        for job in comp_jobs:
            result = run_jobs_parallel([job], gpus, repo_root, args.dry_run)
            all_results.extend(result)
            if result and int(result[0]["returncode"]) != 0:
                (out_root / "job_results.json").write_text(
                    json.dumps(all_results, indent=2), encoding="utf-8"
                )
                raise RuntimeError(
                    f"Compressor job failed: {job.name}. "
                    f"See log: {job.log_path}"
                )

        for v in variants:
            save_dir = out_root / "cnn_variant_training" / v["name"]
            base = (
                save_dir
                / "vmim"
                / args.map_kind
                / "sigma_0.26"
                / "gal_density_30"
                / f"bin_{v['nbins']}"
            )
            compressor_paths[v["name"]] = {
                "params": str(base / f"params_nd_compressor_batch{args.compressor_steps}.pkl"),
                "state": str(base / f"opt_state_resnet_batch{args.compressor_steps}.pkl"),
            }
            if not args.dry_run:
                for label, pstr in compressor_paths[v["name"]].items():
                    p = Path(pstr)
                    if not p.exists():
                        (out_root / "job_results.json").write_text(
                            json.dumps(all_results, indent=2), encoding="utf-8"
                        )
                        raise FileNotFoundError(
                            f"Missing compressor {label} for {v['name']}: {p}"
                        )
    elif need_cnn:
        cp = json.loads((out_root / "compressor_paths.json").read_text(encoding="utf-8"))
        compressor_paths = cp

    if need_cnn:
        (out_root / "compressor_paths.json").write_text(
            json.dumps(compressor_paths, indent=2), encoding="utf-8"
        )

    if not args.skip_evals:
        (out_root / "posteriors").mkdir(parents=True, exist_ok=True)
        eval_jobs: List[Job] = []
        for v in variants:
            for seed in seeds:
                tag = f"{v['name']}_s{seed}"

                cnn_save = out_root / "cnn_eval" / v["name"] / f"seed_{seed}"
                cnn_cache = out_root / "cache" / f"cnn_{v['name']}"
                cnn_post = out_root / "posteriors" / f"cnn_{tag}.npy"
                if "cnn" in methods:
                    cnn_cmd = [
                        "python",
                        cnn_script,
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
                        "--cache-dir",
                        str(cnn_cache),
                        "--save-dir",
                        str(cnn_save),
                        "--compressor-params",
                        compressor_paths[v["name"]]["params"],
                        "--compressor-state",
                        compressor_paths[v["name"]]["state"],
                        "--total-steps",
                        str(args.flow_steps),
                        "--save-every",
                        str(args.eval_save_every),
                        "--patience",
                        str(args.eval_patience),
                        "--npe-samples",
                        str(args.npe_samples),
                        "--posterior-out",
                        str(cnn_post),
                        "--ds-batch-size",
                        str(args.ds_batch_size_cnn),
                    ]
                    if args.plot_cnn_contours:
                        cnn_cmd.extend(
                            [
                                "--plot",
                                "--figure-out",
                                str(out_root / "figures" / f"cnn_{tag}.png"),
                            ]
                        )
                    eval_jobs.append(
                        Job(
                            name=f"cnn_eval::{tag}",
                            log_path=out_root / "logs" / f"cnn_eval_{tag}.log",
                            command=cnn_cmd,
                        )
                    )

                cnn_jax_save = out_root / "cnn_jax_eval" / v["name"] / f"seed_{seed}"
                cnn_jax_cache = out_root / "cache" / f"cnn_jax_{v['name']}"
                cnn_jax_post = out_root / "posteriors" / f"cnn_jax_{tag}.npy"
                if "cnn_jax" in methods:
                    cnn_jax_cmd = [
                        "python",
                        cnn_jax_script,
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
                        "--cache-dir",
                        str(cnn_jax_cache),
                        "--save-dir",
                        str(cnn_jax_save),
                        "--compressor-params",
                        compressor_paths[v["name"]]["params"],
                        "--compressor-state",
                        compressor_paths[v["name"]]["state"],
                        "--total-steps",
                        str(args.flow_steps),
                        "--save-every",
                        str(args.eval_save_every),
                        "--patience",
                        str(args.eval_patience),
                        "--npe-samples",
                        str(args.npe_samples),
                        "--posterior-out",
                        str(cnn_jax_post),
                        "--ds-batch-size",
                        str(args.ds_batch_size_cnn),
                    ]
                    if args.plot_cnn_contours:
                        cnn_jax_cmd.extend(
                            [
                                "--plot",
                                "--figure-out",
                                str(out_root / "figures" / f"cnn_jax_{tag}.png"),
                            ]
                        )
                    eval_jobs.append(
                        Job(
                            name=f"cnn_jax_eval::{tag}",
                            log_path=out_root / "logs" / f"cnn_jax_eval_{tag}.log",
                            command=cnn_jax_cmd,
                        )
                    )

                l1_save = out_root / "l1_eval" / v["name"] / f"seed_{seed}"
                l1_cache = out_root / "cache" / f"l1_{v['name']}"
                l1_post = out_root / "posteriors" / f"l1_{tag}.npy"
                if "l1" in methods:
                    l1_cmd = [
                        "python",
                        l1_script,
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
                        "--cache-dir",
                        str(l1_cache),
                        "--save-dir",
                        str(l1_save),
                        "--total-steps",
                        str(args.flow_steps),
                        "--save-every",
                        str(args.eval_save_every),
                        "--patience",
                        str(args.eval_patience),
                        "--npe-samples",
                        str(args.npe_samples),
                        "--posterior-out",
                        str(l1_post),
                        "--ds-batch-size",
                        str(args.ds_batch_size_l1),
                        "--l1-min-snr",
                        str(args.l1_min_snr),
                        "--l1-max-snr",
                        str(args.l1_max_snr),
                    ]
                    if args.plot_l1_contours:
                        l1_cmd.extend(
                            [
                                "--plot",
                                "--figure-out",
                                str(out_root / "figures" / f"l1_{tag}.png"),
                            ]
                        )
                    eval_jobs.append(
                        Job(
                            name=f"l1_eval::{tag}",
                            log_path=out_root / "logs" / f"l1_eval_{tag}.log",
                            command=l1_cmd,
                        )
                    )

                l1_jax_save = out_root / "l1_jax_eval" / v["name"] / f"seed_{seed}"
                l1_jax_cache = out_root / "cache" / f"l1_jax_{v['name']}"
                l1_jax_post = out_root / "posteriors" / f"l1_jax_{tag}.npy"
                if "l1_jax" in methods:
                    l1_jax_cmd = [
                        "python",
                        l1_jax_script,
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
                        "--cache-dir",
                        str(l1_jax_cache),
                        "--save-dir",
                        str(l1_jax_save),
                        "--total-steps",
                        str(args.flow_steps),
                        "--save-every",
                        str(args.eval_save_every),
                        "--patience",
                        str(args.eval_patience),
                        "--npe-samples",
                        str(args.npe_samples),
                        "--posterior-out",
                        str(l1_jax_post),
                        "--ds-batch-size",
                        str(args.ds_batch_size_l1),
                        "--l1-min-snr",
                        str(args.l1_min_snr),
                        "--l1-max-snr",
                        str(args.l1_max_snr),
                    ]
                    if args.plot_l1_contours:
                        l1_jax_cmd.extend(
                            [
                                "--plot",
                                "--figure-out",
                                str(out_root / "figures" / f"l1_jax_{tag}.png"),
                            ]
                        )
                    eval_jobs.append(
                        Job(
                            name=f"l1_jax_eval::{tag}",
                            log_path=out_root / "logs" / f"l1_jax_eval_{tag}.log",
                            command=l1_jax_cmd,
                        )
                    )
        all_results.extend(run_jobs_parallel(eval_jobs, gpus, repo_root, args.dry_run))

    (out_root / "job_results.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )

    # Lightweight aggregation for all generated posteriors
    truth = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493], dtype=np.float64)
    rows = []
    post_dir = out_root / "posteriors"
    if post_dir.exists():
        for npy_path in sorted(post_dir.glob("*.npy")):
            s = np.load(npy_path)
            if npy_path.name.startswith("cnn_jax_"):
                method = "cnn_jax"
            elif npy_path.name.startswith("cnn_"):
                method = "cnn"
            elif npy_path.name.startswith("l1_jax_"):
                method = "l1_jax"
            else:
                method = "l1"
            rows.append(
                {
                    "file": npy_path.name,
                    "method": method,
                    "n_samples": int(s.shape[0]),
                    "std_sum": float(np.sum(np.std(s, axis=0))),
                    "bias_l2": float(np.linalg.norm(np.mean(s, axis=0) - truth)),
                }
            )
    (out_root / "posterior_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(f"Completed sweep orchestration. Artifacts in: {out_root}")


if __name__ == "__main__":
    main()
