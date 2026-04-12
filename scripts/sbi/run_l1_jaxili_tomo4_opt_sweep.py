#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import queue
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np


DEFAULT_REPO_ROOT = str(Path(__file__).resolve().parents[2])
DEFAULT_OUTPUT_ROOT = str(
    Path(DEFAULT_REPO_ROOT) / "scripts" / "sbi" / "l1_jaxili_tomo4_opt_sweep"
)
DEFAULT_BASELINE_FOM3 = 10219.58492462056


@dataclass(frozen=True)
class SweepConfig:
    summary_transform: str
    clip_value: float
    learning_rate: float
    batch_size: int
    epochs: int
    auto_calibrate_snr: bool

    @property
    def config_id(self) -> str:
        st = self.summary_transform.replace("-", "_")
        clip = str(self.clip_value).replace(".", "p").replace("-", "m")
        lr = f"{self.learning_rate:.0e}".replace("+", "").replace("-", "m")
        auto = "on" if self.auto_calibrate_snr else "off"
        return (
            f"st_{st}"
            f"__clip_{clip}"
            f"__lr_{lr}"
            f"__bs_{self.batch_size}"
            f"__ep_{self.epochs}"
            f"__ac_{auto}"
        )

    @property
    def cache_key(self) -> str:
        return "auto_snr_on" if self.auto_calibrate_snr else "auto_snr_off"


@dataclass
class Job:
    name: str
    config_id: str
    seed: int
    command: List[str]
    log_path: Path
    posterior_path: Path


def _csv_tokens(value: str) -> List[str]:
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def _csv_ints(value: str) -> List[int]:
    return [int(tok) for tok in _csv_tokens(value)]


def _csv_floats(value: str) -> List[float]:
    return [float(tok) for tok in _csv_tokens(value)]


def _csv_bools(value: str) -> List[bool]:
    bools: List[bool] = []
    for tok in _csv_tokens(value):
        low = tok.lower()
        if low in {"1", "true", "t", "yes", "y", "on"}:
            bools.append(True)
        elif low in {"0", "false", "f", "no", "n", "off"}:
            bools.append(False)
        else:
            raise ValueError(
                f"Invalid boolean token '{tok}' in --auto-calibrate-options. "
                "Use comma-separated values from {on,off,true,false,1,0}."
            )
    return bools


def _log_has_traceback(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Traceback" in line:
                return True
    return False


def _fom3_from_posterior(path: Path) -> tuple[float, float, float, bool, int]:
    samples = np.load(path)
    if samples.ndim != 2 or samples.shape[1] < 3 or samples.shape[0] < 2:
        return float("nan"), float("nan"), float("nan"), False, int(samples.shape[0])
    cov3 = np.cov(samples[:, :3], rowvar=False)
    sign, logdet = np.linalg.slogdet(cov3)
    if sign <= 0:
        return float("nan"), float("nan"), float(logdet), False, int(samples.shape[0])
    det = float(np.exp(logdet))
    fom3 = float(np.exp(-0.5 * logdet))
    return fom3, det, float(logdet), True, int(samples.shape[0])


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Focused no-BNT tomo4 L1+jaxili hyperparameter sweep with FoM3 ranking "
            "and robust top-2 reruns."
        )
    )
    p.add_argument("--repo-root", type=str, default=DEFAULT_REPO_ROOT)
    p.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--gpus", type=str, default="0,1,2")
    p.add_argument("--seed41", type=int, default=41)
    p.add_argument("--robust-seeds", type=str, default="42,43")
    p.add_argument("--summary-transforms", type=str, default="log1p-zscore,zscore,none")
    p.add_argument("--clip-values", type=str, default="5.0,8.0,0.0")
    p.add_argument("--learning-rates", type=str, default="1e-4,3e-4")
    p.add_argument("--batch-sizes", type=str, default="128,256")
    p.add_argument("--epochs-list", type=str, default="5000,10000")
    p.add_argument("--auto-calibrate-options", type=str, default="off,on")
    p.add_argument(
        "--grid-mode",
        type=str,
        choices=["focused", "cartesian"],
        default="focused",
        help=(
            "focused: compact two-block sweep; "
            "cartesian: full product over all provided values."
        ),
    )
    p.add_argument("--npe-samples", type=int, default=100000)
    p.add_argument("--ds-batch-size", type=int, default=96)
    p.add_argument("--conda-env", type=str, default="jaxili")
    p.add_argument("--baseline-fom3", type=float, default=DEFAULT_BASELINE_FOM3)
    p.add_argument(
        "--baseline-label",
        type=str,
        default="main no-BNT tomo4 L1+jaxili no-PCA",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_configs(args: argparse.Namespace) -> List[SweepConfig]:
    summary_transforms = _csv_tokens(args.summary_transforms)
    clip_values = _csv_floats(args.clip_values)
    learning_rates = _csv_floats(args.learning_rates)
    batch_sizes = _csv_ints(args.batch_sizes)
    epochs_list = _csv_ints(args.epochs_list)
    auto_flags = _csv_bools(args.auto_calibrate_options)

    if not summary_transforms:
        raise ValueError("--summary-transforms cannot be empty.")
    if not clip_values:
        raise ValueError("--clip-values cannot be empty.")
    if not learning_rates:
        raise ValueError("--learning-rates cannot be empty.")
    if not batch_sizes:
        raise ValueError("--batch-sizes cannot be empty.")
    if not epochs_list:
        raise ValueError("--epochs-list cannot be empty.")
    if not auto_flags:
        raise ValueError("--auto-calibrate-options cannot be empty.")

    baseline_transform = (
        "log1p-zscore" if "log1p-zscore" in summary_transforms else summary_transforms[0]
    )
    baseline_clip = 5.0 if 5.0 in clip_values else clip_values[0]
    baseline_lr = 1e-4 if 1e-4 in learning_rates else learning_rates[0]
    baseline_bs = 256 if 256 in batch_sizes else batch_sizes[0]
    baseline_epochs = 5000 if 5000 in epochs_list else epochs_list[0]
    baseline_auto = False if False in auto_flags else auto_flags[0]

    cfg_set: set[SweepConfig] = set()
    if args.grid_mode == "cartesian":
        cfg_set.update(
            SweepConfig(
                summary_transform=st,
                clip_value=float(cv),
                learning_rate=float(lr),
                batch_size=int(bs),
                epochs=int(ep),
                auto_calibrate_snr=bool(ac),
            )
            for st, cv, lr, bs, ep, ac in product(
                summary_transforms,
                clip_values,
                learning_rates,
                batch_sizes,
                epochs_list,
                auto_flags,
            )
        )
    else:
        # Block A: preprocess sweep with baseline NPE training hyperparameters.
        cfg_set.update(
            SweepConfig(
                summary_transform=st,
                clip_value=float(cv),
                learning_rate=float(baseline_lr),
                batch_size=int(baseline_bs),
                epochs=int(baseline_epochs),
                auto_calibrate_snr=bool(ac),
            )
            for st, cv, ac in product(summary_transforms, clip_values, auto_flags)
        )
        # Block B: NPE optimization sweep at baseline preprocessing setup.
        cfg_set.update(
            SweepConfig(
                summary_transform=baseline_transform,
                clip_value=float(baseline_clip),
                learning_rate=float(lr),
                batch_size=int(bs),
                epochs=int(ep),
                auto_calibrate_snr=bool(baseline_auto),
            )
            for lr, bs, ep in product(learning_rates, batch_sizes, epochs_list)
        )

    return sorted(
        cfg_set,
        key=lambda c: (
            c.auto_calibrate_snr,
            c.summary_transform,
            c.clip_value,
            c.learning_rate,
            c.batch_size,
            c.epochs,
        ),
    )


def build_job(
    cfg: SweepConfig,
    seed: int,
    repo_root: Path,
    out_root: Path,
    conda_env: str,
    npe_samples: int,
    ds_batch_size: int,
) -> Job:
    script = repo_root / "scripts" / "sbi" / "npe_l1norm_jaxili_nbody_tomo.py"
    run_dir = out_root / "l1_eval" / cfg.config_id / f"seed_{seed}"
    posterior_path = out_root / "posteriors" / f"{cfg.config_id}_s{seed}.npy"
    log_path = out_root / "logs" / f"{cfg.config_id}_s{seed}.log"
    cache_dir = out_root / "cache" / cfg.cache_key

    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        conda_env,
        "python",
        str(script),
        "--no-wandb",
        "--map-kind",
        "nbody",
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
        str(cache_dir),
        "--save-dir",
        str(run_dir),
        "--n-scales",
        "5",
        "--l1-nbins",
        "40",
        "--l1-min-snr",
        "-13",
        "--l1-max-snr",
        "13",
        "--pca-components",
        "0",
        "--summary-transform",
        cfg.summary_transform,
        "--clip-value",
        str(cfg.clip_value),
        "--epochs",
        str(cfg.epochs),
        "--batch-size",
        str(cfg.batch_size),
        "--learning-rate",
        str(cfg.learning_rate),
        "--ds-batch-size",
        str(ds_batch_size),
        "--npe-samples",
        str(npe_samples),
        "--posterior-out",
        str(posterior_path),
    ]
    if cfg.auto_calibrate_snr:
        cmd.append("--auto-calibrate-snr")

    return Job(
        name=f"{cfg.config_id}::s{seed}",
        config_id=cfg.config_id,
        seed=seed,
        command=cmd,
        log_path=log_path,
        posterior_path=posterior_path,
    )


def run_jobs_parallel(
    jobs: List[Job],
    gpus: List[str],
    cwd: Path,
    dry_run: bool = False,
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

            if (
                not dry_run
                and job.posterior_path.exists()
                and job.log_path.exists()
                and not _log_has_traceback(job.log_path)
            ):
                rc = 0
                skipped = True
            elif dry_run:
                rc = 0
                skipped = False
                job.log_path.write_text(f"[dry-run] {' '.join(cmd)}\n", encoding="utf-8")
            else:
                skipped = False
                with job.log_path.open("w", encoding="utf-8") as logf:
                    proc = subprocess.run(
                        cmd,
                        cwd=str(cwd),
                        stdout=logf,
                        stderr=subprocess.STDOUT,
                    )
                rc = int(proc.returncode)

            dt = float(time.time() - t0)
            print(
                f"[job] {job.name} gpu={gpu_id} rc={rc} "
                f"time={dt / 60.0:.2f}m skip={int(skipped)} log={job.log_path}"
            )

            with lock:
                results.append(
                    {
                        "name": job.name,
                        "config_id": job.config_id,
                        "seed": job.seed,
                        "gpu": str(gpu_id),
                        "returncode": rc,
                        "seconds": dt,
                        "skipped": bool(skipped),
                        "log": str(job.log_path),
                        "posterior": str(job.posterior_path),
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


def analyze_seed(
    configs: List[SweepConfig],
    seed: int,
    out_root: Path,
    result_lookup: Dict[tuple[str, int], Dict[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    cfg_by_id = {cfg.config_id: cfg for cfg in configs}
    for config_id, cfg in cfg_by_id.items():
        posterior_path = out_root / "posteriors" / f"{config_id}_s{seed}.npy"
        log_path = out_root / "logs" / f"{config_id}_s{seed}.log"
        result = result_lookup.get((config_id, seed), {})
        rc = int(result.get("returncode", 1 if not posterior_path.exists() else 0))
        traceback_found = _log_has_traceback(log_path)
        if posterior_path.exists() and rc == 0 and not traceback_found:
            fom3, det_cov3, logdet_cov3, valid_fom, n_samples = _fom3_from_posterior(
                posterior_path
            )
        else:
            fom3, det_cov3, logdet_cov3, valid_fom, n_samples = (
                float("nan"),
                float("nan"),
                float("nan"),
                False,
                0,
            )

        row = {
            "config_id": config_id,
            "seed": seed,
            "summary_transform": cfg.summary_transform,
            "clip_value": cfg.clip_value,
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "auto_calibrate_snr": cfg.auto_calibrate_snr,
            "returncode": rc,
            "traceback_found": traceback_found,
            "posterior_exists": posterior_path.exists(),
            "posterior": str(posterior_path),
            "log": str(log_path),
            "n_samples": n_samples,
            "fom3": fom3,
            "det_cov3": det_cov3,
            "logdet_cov3": logdet_cov3,
            "valid_fom": valid_fom,
        }
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    gpus = _csv_tokens(args.gpus)
    if not gpus:
        raise ValueError("--gpus cannot be empty.")

    robust_seeds = _csv_ints(args.robust_seeds)
    if not robust_seeds:
        raise ValueError("--robust-seeds cannot be empty.")

    configs = build_configs(args)
    if len(configs) < 2:
        raise ValueError("Need at least two configurations for ranking and top-2 reruns.")

    manifest = {
        "repo_root": str(repo_root),
        "output_root": str(out_root),
        "variant": "tomo4_20deg160_nobnt",
        "seed41": int(args.seed41),
        "robust_seeds": robust_seeds,
        "gpus": gpus,
        "npe_samples": int(args.npe_samples),
        "ds_batch_size": int(args.ds_batch_size),
        "l1_fixed": {
            "n_scales": 5,
            "l1_nbins": 40,
            "l1_min_snr": -13.0,
            "l1_max_snr": 13.0,
            "pca_components": 0,
        },
        "grid_mode": args.grid_mode,
        "grid_size": len(configs),
        "configs": [asdict(cfg) for cfg in configs],
        "baseline": {
            "label": args.baseline_label,
            "fom3_mean": float(args.baseline_fom3),
        },
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[manifest] wrote {out_root / 'manifest.json'}")

    all_results: List[Dict[str, object]] = []
    result_lookup: Dict[tuple[str, int], Dict[str, object]] = {}

    # Stage 1: seed 41 sweep, grouped by auto-calibration mode to avoid cache races.
    for auto_flag in (False, True):
        group = [cfg for cfg in configs if cfg.auto_calibrate_snr == auto_flag]
        if not group:
            continue
        warm_job = build_job(
            group[0],
            args.seed41,
            repo_root=repo_root,
            out_root=out_root,
            conda_env=args.conda_env,
            npe_samples=args.npe_samples,
            ds_batch_size=args.ds_batch_size,
        )
        print(
            f"[stage1] warm-up cache for auto_calibrate_snr={auto_flag} "
            f"with {warm_job.name}"
        )
        warm_results = run_jobs_parallel(
            [warm_job],
            gpus=[gpus[0]],
            cwd=repo_root,
            dry_run=args.dry_run,
        )
        all_results.extend(warm_results)
        for r in warm_results:
            result_lookup[(str(r["config_id"]), int(r["seed"]))] = r

        remaining_jobs = [
            build_job(
                cfg,
                args.seed41,
                repo_root=repo_root,
                out_root=out_root,
                conda_env=args.conda_env,
                npe_samples=args.npe_samples,
                ds_batch_size=args.ds_batch_size,
            )
            for cfg in group[1:]
        ]
        if remaining_jobs:
            print(
                f"[stage1] running {len(remaining_jobs)} jobs for "
                f"auto_calibrate_snr={auto_flag}"
            )
            stage_results = run_jobs_parallel(
                remaining_jobs,
                gpus=gpus,
                cwd=repo_root,
                dry_run=args.dry_run,
            )
            all_results.extend(stage_results)
            for r in stage_results:
                result_lookup[(str(r["config_id"]), int(r["seed"]))] = r

    # Analyze seed-41 ranking
    seed41_rows = analyze_seed(configs, args.seed41, out_root, result_lookup)
    _write_csv(
        out_root / "seed41_results.csv",
        seed41_rows,
        fieldnames=[
            "config_id",
            "seed",
            "summary_transform",
            "clip_value",
            "learning_rate",
            "batch_size",
            "epochs",
            "auto_calibrate_snr",
            "returncode",
            "traceback_found",
            "posterior_exists",
            "posterior",
            "log",
            "n_samples",
            "fom3",
            "det_cov3",
            "logdet_cov3",
            "valid_fom",
        ],
    )
    (out_root / "seed41_results.json").write_text(
        json.dumps(seed41_rows, indent=2),
        encoding="utf-8",
    )

    ranked = [
        row
        for row in seed41_rows
        if bool(row["valid_fom"]) and int(row["returncode"]) == 0 and not bool(row["traceback_found"])
    ]
    ranked.sort(key=lambda r: float(r["fom3"]), reverse=True)
    for i, row in enumerate(ranked, start=1):
        row["rank_seed41"] = i

    _write_csv(
        out_root / "seed41_ranked.csv",
        ranked,
        fieldnames=[
            "rank_seed41",
            "config_id",
            "seed",
            "summary_transform",
            "clip_value",
            "learning_rate",
            "batch_size",
            "epochs",
            "auto_calibrate_snr",
            "returncode",
            "traceback_found",
            "posterior_exists",
            "posterior",
            "log",
            "n_samples",
            "fom3",
            "det_cov3",
            "logdet_cov3",
            "valid_fom",
        ],
    )
    (out_root / "seed41_ranked.json").write_text(
        json.dumps(ranked, indent=2),
        encoding="utf-8",
    )
    if len(ranked) < 2:
        raise RuntimeError(
            f"Need at least 2 valid seed-41 runs, got {len(ranked)}. "
            f"See {out_root / 'seed41_results.json'}."
        )

    top2_ids = [str(ranked[0]["config_id"]), str(ranked[1]["config_id"])]
    cfg_by_id = {cfg.config_id: cfg for cfg in configs}
    top2_cfgs = [cfg_by_id[cfg_id] for cfg_id in top2_ids]
    print(f"[stage2] top-2 seed41 configs: {top2_ids}")

    # Stage 2: rerun top-2 on robust seeds.
    top2_jobs: List[Job] = []
    for cfg in top2_cfgs:
        for seed in robust_seeds:
            top2_jobs.append(
                build_job(
                    cfg=cfg,
                    seed=seed,
                    repo_root=repo_root,
                    out_root=out_root,
                    conda_env=args.conda_env,
                    npe_samples=args.npe_samples,
                    ds_batch_size=args.ds_batch_size,
                )
            )
    if top2_jobs:
        stage2_results = run_jobs_parallel(
            top2_jobs,
            gpus=gpus,
            cwd=repo_root,
            dry_run=args.dry_run,
        )
        all_results.extend(stage2_results)
        for r in stage2_results:
            result_lookup[(str(r["config_id"]), int(r["seed"]))] = r

    # Persist all job results.
    (out_root / "job_results.json").write_text(
        json.dumps(all_results, indent=2),
        encoding="utf-8",
    )

    # Robustness summary for top-2 across seeds 41/42/43.
    robust_rows: List[Dict[str, object]] = []
    robust_seeds_full = [int(args.seed41)] + robust_seeds
    for cfg in top2_cfgs:
        per_seed: List[Dict[str, object]] = []
        fom_vals: List[float] = []
        clean_logs = True
        all_returncode_zero = True
        all_posteriors_present = True

        for seed in robust_seeds_full:
            job_key = (cfg.config_id, seed)
            result = result_lookup.get(job_key, {})
            log_path = out_root / "logs" / f"{cfg.config_id}_s{seed}.log"
            posterior_path = out_root / "posteriors" / f"{cfg.config_id}_s{seed}.npy"
            traceback_found = _log_has_traceback(log_path)
            if traceback_found:
                clean_logs = False
            rc = int(result.get("returncode", 1 if not posterior_path.exists() else 0))
            if rc != 0:
                all_returncode_zero = False

            if posterior_path.exists() and rc == 0 and not traceback_found:
                fom3, det_cov3, logdet_cov3, valid_fom, n_samples = _fom3_from_posterior(
                    posterior_path
                )
                if valid_fom:
                    fom_vals.append(fom3)
            else:
                fom3, det_cov3, logdet_cov3, valid_fom, n_samples = (
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    False,
                    0,
                )
            if not posterior_path.exists():
                all_posteriors_present = False

            per_seed.append(
                {
                    "seed": seed,
                    "returncode": rc,
                    "traceback_found": traceback_found,
                    "posterior_exists": posterior_path.exists(),
                    "posterior": str(posterior_path),
                    "log": str(log_path),
                    "n_samples": n_samples,
                    "fom3": fom3,
                    "det_cov3": det_cov3,
                    "logdet_cov3": logdet_cov3,
                    "valid_fom": valid_fom,
                }
            )

        n_valid = len(fom_vals)
        mean_fom3 = float(np.mean(np.array(fom_vals, dtype=np.float64))) if n_valid else float("nan")
        std_fom3 = (
            float(np.std(np.array(fom_vals, dtype=np.float64), ddof=1))
            if n_valid > 1
            else 0.0 if n_valid == 1 else float("nan")
        )
        robust_rows.append(
            {
                "config_id": cfg.config_id,
                "summary_transform": cfg.summary_transform,
                "clip_value": cfg.clip_value,
                "learning_rate": cfg.learning_rate,
                "batch_size": cfg.batch_size,
                "epochs": cfg.epochs,
                "auto_calibrate_snr": cfg.auto_calibrate_snr,
                "seed41_fom3": next(
                    float(r["fom3"]) for r in per_seed if int(r["seed"]) == int(args.seed41)
                ),
                "fom3_mean": mean_fom3,
                "fom3_std": std_fom3,
                "n_valid": n_valid,
                "n_total": len(robust_seeds_full),
                "all_returncode_zero": all_returncode_zero,
                "all_posteriors_present": all_posteriors_present,
                "all_logs_clean_no_traceback": clean_logs,
                "per_seed": per_seed,
            }
        )

    robust_rows.sort(key=lambda r: float(r["fom3_mean"]), reverse=True)
    for i, row in enumerate(robust_rows, start=1):
        row["rank_robust"] = i

    _write_csv(
        out_root / "top2_robustness.csv",
        robust_rows,
        fieldnames=[
            "rank_robust",
            "config_id",
            "summary_transform",
            "clip_value",
            "learning_rate",
            "batch_size",
            "epochs",
            "auto_calibrate_snr",
            "seed41_fom3",
            "fom3_mean",
            "fom3_std",
            "n_valid",
            "n_total",
            "all_returncode_zero",
            "all_posteriors_present",
            "all_logs_clean_no_traceback",
        ],
    )
    (out_root / "top2_robustness.json").write_text(
        json.dumps(robust_rows, indent=2),
        encoding="utf-8",
    )

    best = robust_rows[0]
    if (
        not bool(best["all_returncode_zero"])
        or not bool(best["all_posteriors_present"])
        or not bool(best["all_logs_clean_no_traceback"])
    ):
        raise RuntimeError(
            "Best robust config has failed runs and/or traceback logs. "
            f"Inspect {out_root / 'top2_robustness.json'}."
        )

    baseline = float(args.baseline_fom3)
    best_mean = float(best["fom3_mean"])
    delta = best_mean - baseline
    ratio = best_mean / baseline if baseline != 0 else float("nan")
    percent = (delta / baseline) * 100.0 if baseline != 0 else float("nan")

    final_summary = {
        "best_config": best,
        "baseline": {
            "label": args.baseline_label,
            "fom3_mean": baseline,
        },
        "comparison_vs_baseline": {
            "delta_fom3": delta,
            "ratio": ratio,
            "percent_change": percent,
        },
        "artifacts": {
            "manifest": str((out_root / "manifest.json").resolve()),
            "job_results": str((out_root / "job_results.json").resolve()),
            "seed41_results_json": str((out_root / "seed41_results.json").resolve()),
            "seed41_results_csv": str((out_root / "seed41_results.csv").resolve()),
            "seed41_ranked_json": str((out_root / "seed41_ranked.json").resolve()),
            "seed41_ranked_csv": str((out_root / "seed41_ranked.csv").resolve()),
            "top2_robustness_json": str((out_root / "top2_robustness.json").resolve()),
            "top2_robustness_csv": str((out_root / "top2_robustness.csv").resolve()),
        },
    }
    (out_root / "final_selection.json").write_text(
        json.dumps(final_summary, indent=2),
        encoding="utf-8",
    )
    print(
        f"[done] best={best['config_id']} "
        f"mean_fom3={best_mean:.4f} vs baseline={baseline:.4f} "
        f"(delta={delta:.4f}, {percent:.2f}%)"
    )
    print(f"[done] artifacts in {out_root}")


if __name__ == "__main__":
    main()
