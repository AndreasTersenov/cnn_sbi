#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_REPO_ROOT
    / "scripts"
    / "sbi"
    / "results"
    / "final"
    / "paper_sbi_consolidation"
    / "cnn_noiseless_vs_noisy"
)
DEFAULT_BASELINE_ROOT = (
    DEFAULT_REPO_ROOT
    / "scripts"
    / "sbi"
    / "results"
    / "final"
    / "paper_sbi_consolidation"
    / "bnt_comparison_tomo4"
)
TRUTH = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493], dtype=np.float64)
PARAM_NAMES = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]


@dataclass(frozen=True)
class Job:
    name: str
    command: List[str]
    log_path: Path


def _csv_tokens(value: str) -> List[str]:
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def _csv_ints(value: str) -> Tuple[int, ...]:
    return tuple(int(tok) for tok in _csv_tokens(value))


def run_jobs_parallel(
    jobs: List[Job],
    gpus: List[str],
    cwd: Path,
    xla_mem_fraction: float,
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
            cmd = list(job.command) + ["--cuda-visible-devices", gpu_id]
            t0 = time.time()
            if dry_run:
                rc = 0
                job.log_path.write_text(
                    "[dry-run] " + " ".join(cmd) + "\n",
                    encoding="utf-8",
                )
            else:
                env = dict(os.environ)
                env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(xla_mem_fraction)
                with open(job.log_path, "w", encoding="utf-8") as logf:
                    proc = subprocess.run(
                        cmd,
                        cwd=str(cwd),
                        env=env,
                        stdout=logf,
                        stderr=subprocess.STDOUT,
                    )
                    rc = proc.returncode
            dt = time.time() - t0
            with lock:
                results.append(
                    {
                        "name": job.name,
                        "gpu": gpu_id,
                        "returncode": int(rc),
                        "seconds": float(dt),
                        "log": str(job.log_path),
                        "cmd": cmd,
                    }
                )
            q.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def ensure_tfds_prepared(
    tfds_name: str,
    conda_env: str,
    repo_root: Path,
    log_path: Path,
    dry_run: bool = False,
) -> None:
    if dry_run:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"[dry-run] prepare tfds: {tfds_name}\n",
            encoding="utf-8",
        )
        return
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        conda_env,
        "python",
        "-c",
        (
            "import tensorflow_datasets as tfds; "
            "import scripts.sbi.tf_dataset_nbody_tomo as _; "
            f"b=tfds.builder('{tfds_name}'); "
            "b.download_and_prepare(); "
            "print('TFDS prepared:', b.name, b.builder_config.name, b.version)"
        ),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"TFDS preparation failed for {tfds_name}. See log: {log_path}"
        )


def require_success(results: List[Dict[str, object]], context: str) -> None:
    failed = [r for r in results if int(r.get("returncode", 1)) != 0]
    if failed:
        first = failed[0]
        raise RuntimeError(
            f"{context} failed for {len(failed)}/{len(results)} jobs. "
            f"First failure: {first.get('name')} (log: {first.get('log')})"
        )


def _compressor_paths(output_root: Path, condition: str, compressor_steps: int) -> Dict[str, Path]:
    base = (
        output_root
        / "compressor"
        / condition
        / "vmim"
        / "nbody"
        / "sigma_0.0"
        / "gal_density_30"
        / "bin_4"
    )
    return {
        "params": base / f"params_nd_compressor_batch{compressor_steps}.pkl",
        "state": base / f"opt_state_resnet_batch{compressor_steps}.pkl",
    }


def _posterior_path(output_root: Path, condition: str, seed: int) -> Path:
    return output_root / "posteriors" / f"cnn_tomo4_20deg160_{condition}_noiseless_s{seed}.npy"


def _baseline_noisy_path(baseline_root: Path, condition: str, seed: int) -> Path:
    return baseline_root / "posteriors" / f"cnn_tomo4_20deg160_{condition}_s{seed}.npy"


def _fom3(samples: np.ndarray) -> float:
    cov3 = np.cov(samples[:, :3], rowvar=False)
    sign, logdet = np.linalg.slogdet(cov3)
    if sign <= 0:
        return float("nan")
    return float(np.exp(-0.5 * logdet))


def _om_s8_area_proxy(samples: np.ndarray) -> float:
    cov2 = np.cov(samples[:, :2], rowvar=False)
    det_cov2 = float(np.linalg.det(cov2))
    if det_cov2 <= 0:
        return float("nan")
    return float(np.pi * np.sqrt(det_cov2))


def _condition_metrics(paths: Iterable[Path]) -> Dict[str, float]:
    std_sum_vals = []
    fom_vals = []
    sigma8_vals = []
    area_vals = []
    for path in paths:
        samples = np.load(path)
        std = np.std(samples, axis=0)
        std_sum_vals.append(float(np.sum(std)))
        fom_vals.append(_fom3(samples))
        sigma8_vals.append(float(std[1]))
        area_vals.append(_om_s8_area_proxy(samples))
    std_sum = np.array(std_sum_vals, dtype=np.float64)
    fom = np.array(fom_vals, dtype=np.float64)
    sigma8 = np.array(sigma8_vals, dtype=np.float64)
    area = np.array(area_vals, dtype=np.float64)
    return {
        "n": int(len(std_sum)),
        "std_sum_mean": float(np.mean(std_sum)),
        "std_sum_std": float(np.std(std_sum, ddof=1)) if len(std_sum) > 1 else 0.0,
        "fom3_mean": float(np.nanmean(fom)),
        "fom3_std": float(np.nanstd(fom, ddof=1)) if len(fom) > 1 else 0.0,
        "sigma8_std_mean": float(np.mean(sigma8)),
        "sigma8_std_std": float(np.std(sigma8, ddof=1)) if len(sigma8) > 1 else 0.0,
        "om_s8_area_mean": float(np.nanmean(area)),
        "om_s8_area_std": float(np.nanstd(area, ddof=1)) if len(area) > 1 else 0.0,
    }


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(den) or den == 0.0:
        return float("nan")
    return float(num / den)


def _import_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from getdist import MCSamples, plots as gplot

    return plt, MCSamples, gplot


def _plot_overlay(
    out_path: Path,
    noisy_samples: np.ndarray,
    noiseless_samples: np.ndarray,
    title: str,
    dpi: int = 150,
) -> None:
    try:
        plt, MCSamples, gplot = _import_plotting()
    except Exception:
        return
    chain_noisy = MCSamples(
        samples=noisy_samples,
        names=PARAM_NAMES,
        labels=PARAM_NAMES,
        label="noisy",
        settings={"smooth_scale_2D": 0.7, "smooth_scale_1D": 0.7},
    )
    chain_noiseless = MCSamples(
        samples=noiseless_samples,
        names=PARAM_NAMES,
        labels=PARAM_NAMES,
        label="noiseless",
        settings={"smooth_scale_2D": 0.7, "smooth_scale_1D": 0.7},
    )
    g = gplot.get_subplot_plotter(subplot_size=1.4)
    g.settings.alpha_filled_add = 0.55
    g.triangle_plot(
        [chain_noisy, chain_noiseless],
        filled=True,
        contour_colors=["#1f77b4", "#d62728"],
        markers=TRUTH,
        marker_args={"color": "black", "lw": 1.0},
        legend_loc="upper right",
    )
    g.fig.suptitle(title, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(g.fig)


def _concat(paths: Iterable[Path]) -> np.ndarray:
    return np.concatenate([np.load(p) for p in paths], axis=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run CNN noiseless (sigma_e=0) no-BNT/BNT contours and compare them "
            "against existing noisy baselines."
        )
    )
    p.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    p.add_argument("--conda-env", type=str, default="jaxili")
    p.add_argument("--gpus", type=str, default="0,1,2,3")
    p.add_argument("--xla-mem-fraction", type=float, default=0.5)
    p.add_argument("--seeds", type=str, default="41,42,43")
    p.add_argument("--map-kind", type=str, default="nbody")
    p.add_argument("--tfds-name", type=str, default="NbodyCosmogridDatasetTomo/grid_20deg_160px")
    p.add_argument("--field-size", type=int, default=20)
    p.add_argument("--field-npix", type=int, default=160)
    p.add_argument("--nbins", type=int, default=4)
    p.add_argument("--tomo-bin-indices", type=str, default="1,2,3,4")
    p.add_argument("--sigma-e", type=float, default=0.0)
    p.add_argument("--galaxy-density", type=float, default=30 / 4)
    p.add_argument("--compressor-dim", type=int, default=6)
    p.add_argument("--compressor-conv-channels", type=str, default="64,128,256")
    p.add_argument("--compressor-dense-width", type=int, default=128)
    p.add_argument("--compressor-pool-window", type=int, default=16)
    p.add_argument("--compressor-pool-stride", type=int, default=8)
    p.add_argument("--compressor-steps", type=int, default=60000)
    p.add_argument("--compressor-save-every", type=int, default=2000)
    p.add_argument("--compressor-batch-size", type=int, default=128)
    p.add_argument("--compressor-lr", type=float, default=5e-4)
    p.add_argument("--flow-steps", type=int, default=5000)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--nvp-layers", type=int, default=4)
    p.add_argument("--nvp-hidden", type=int, default=128)
    p.add_argument("--ds-batch-size", type=int, default=500)
    p.add_argument("--npe-samples", type=int, default=100000)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    baseline_root = args.baseline_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    gpus = _csv_tokens(args.gpus)
    if not gpus:
        raise ValueError("--gpus cannot be empty.")
    if args.xla_mem_fraction <= 0.0 or args.xla_mem_fraction > 1.0:
        raise ValueError("--xla-mem-fraction must be in (0, 1].")
    seeds = _csv_ints(args.seeds)
    if not seeds:
        raise ValueError("--seeds cannot be empty.")

    manifest = {
        "repo_root": str(repo_root),
        "output_root": str(output_root),
        "baseline_root": str(baseline_root),
        "conda_env": args.conda_env,
        "gpus": gpus,
        "xla_mem_fraction": float(args.xla_mem_fraction),
        "seeds": list(seeds),
        "map_kind": args.map_kind,
        "tfds_name": args.tfds_name,
        "field_size": int(args.field_size),
        "field_npix": int(args.field_npix),
        "nbins": int(args.nbins),
        "tomo_bin_indices": args.tomo_bin_indices,
        "sigma_e": float(args.sigma_e),
        "galaxy_density": float(args.galaxy_density),
        "compressor_dim": int(args.compressor_dim),
        "compressor_conv_channels": args.compressor_conv_channels,
        "compressor_dense_width": int(args.compressor_dense_width),
        "compressor_pool_window": int(args.compressor_pool_window),
        "compressor_pool_stride": int(args.compressor_pool_stride),
        "compressor_steps": int(args.compressor_steps),
        "flow_steps": int(args.flow_steps),
        "npe_samples": int(args.npe_samples),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    ensure_tfds_prepared(
        tfds_name=args.tfds_name,
        conda_env=args.conda_env,
        repo_root=repo_root,
        log_path=output_root / "logs" / "tfds_prepare.log",
        dry_run=args.dry_run,
    )

    cnn_script = str(repo_root / "scripts" / "sbi" / "npe_cnn_nbody_tomo.py")
    all_results: List[Dict[str, object]] = []

    # Train noiseless compressors (one per condition).
    train_jobs: List[Job] = []
    for cond in ("nobnt", "bnt"):
        cond_flag = ["--apply-bnt"] if cond == "bnt" else []
        comp_paths = _compressor_paths(output_root, cond, args.compressor_steps)
        if comp_paths["params"].exists() and comp_paths["state"].exists():
            continue
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
            "--sigma-e",
            str(args.sigma_e),
            "--galaxy-density",
            str(args.galaxy_density),
            "--cache-dir",
            str(output_root / "cache" / f"compressor_{cond}"),
            "--save-dir",
            str(output_root / "compressor" / cond),
            "--train-compressor",
            "--compressor-dim",
            str(args.compressor_dim),
            "--compressor-conv-channels",
            args.compressor_conv_channels,
            "--compressor-dense-width",
            str(args.compressor_dense_width),
            "--compressor-pool-window",
            str(args.compressor_pool_window),
            "--compressor-pool-stride",
            str(args.compressor_pool_stride),
            "--compressor-steps",
            str(args.compressor_steps),
            "--compressor-save-every",
            str(args.compressor_save_every),
            "--compressor-batch-size",
            str(args.compressor_batch_size),
            "--compressor-lr",
            str(args.compressor_lr),
            "--total-steps",
            "1",
            "--save-every",
            "1",
            "--no-sample",
            "--no-standardize-summary",
        ] + cond_flag
        train_jobs.append(
            Job(
                name=f"train::{cond}",
                command=cmd,
                log_path=output_root / "logs" / f"train_{cond}.log",
            )
        )

    if train_jobs:
        train_results = run_jobs_parallel(
            train_jobs,
            gpus=gpus,
            cwd=repo_root,
            xla_mem_fraction=args.xla_mem_fraction,
            dry_run=args.dry_run,
        )
        all_results.extend(train_results)
        require_success(train_results, "Noiseless compressor training")

    # Evaluate no-BNT and BNT noiseless posteriors.
    eval_jobs: List[Job] = []
    for cond in ("nobnt", "bnt"):
        cond_flag = ["--apply-bnt"] if cond == "bnt" else []
        comp_paths = _compressor_paths(output_root, cond, args.compressor_steps)
        if (
            not args.dry_run
            and (
                not comp_paths["params"].exists()
                or not comp_paths["state"].exists()
            )
        ):
            raise FileNotFoundError(
                f"Missing noiseless compressor checkpoint for {cond}: {comp_paths}"
            )
        for seed in seeds:
            posterior_out = _posterior_path(output_root, cond, seed)
            if posterior_out.exists():
                continue
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
                "--sigma-e",
                str(args.sigma_e),
                "--galaxy-density",
                str(args.galaxy_density),
                "--cache-dir",
                str(output_root / "cache" / f"eval_{cond}"),
                "--save-dir",
                str(output_root / "eval" / cond / f"seed_{seed}"),
                "--compressor-dim",
                str(args.compressor_dim),
                "--compressor-conv-channels",
                args.compressor_conv_channels,
                "--compressor-dense-width",
                str(args.compressor_dense_width),
                "--compressor-pool-window",
                str(args.compressor_pool_window),
                "--compressor-pool-stride",
                str(args.compressor_pool_stride),
                "--compressor-params",
                str(comp_paths["params"]),
                "--compressor-state",
                str(comp_paths["state"]),
                "--total-steps",
                str(args.flow_steps),
                "--save-every",
                str(args.save_every),
                "--patience",
                str(args.patience),
                "--batch-size",
                str(args.batch_size),
                "--nvp-layers",
                str(args.nvp_layers),
                "--nvp-hidden",
                str(args.nvp_hidden),
                "--npe-samples",
                str(args.npe_samples),
                "--posterior-out",
                str(posterior_out),
                "--ds-batch-size",
                str(args.ds_batch_size),
                "--no-standardize-summary",
            ] + cond_flag
            if args.plot:
                cmd.extend(
                    [
                        "--plot",
                        "--figure-out",
                        str(output_root / "figures" / f"cnn_{cond}_noiseless_s{seed}.png"),
                    ]
                )
            eval_jobs.append(
                Job(
                    name=f"eval::{cond}::s{seed}",
                    command=cmd,
                    log_path=output_root / "logs" / f"eval_{cond}_s{seed}.log",
                )
            )

    if eval_jobs:
        eval_results = run_jobs_parallel(
            eval_jobs,
            gpus=gpus,
            cwd=repo_root,
            xla_mem_fraction=args.xla_mem_fraction,
            dry_run=args.dry_run,
        )
        all_results.extend(eval_results)
        require_success(eval_results, "Noiseless posterior evaluation")

    (output_root / "job_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    if args.dry_run:
        print(f"Dry-run complete. Outputs planned in: {output_root}")
        return

    # Analyze noisy vs noiseless shrink for both conditions.
    summary: Dict[str, object] = {
        "seeds": list(seeds),
        "conditions": {},
    }
    csv_rows: List[Dict[str, object]] = []

    for cond in ("nobnt", "bnt"):
        noisy_paths = [_baseline_noisy_path(baseline_root, cond, seed) for seed in seeds]
        noiseless_paths = [_posterior_path(output_root, cond, seed) for seed in seeds]
        for p in noisy_paths + noiseless_paths:
            if not p.exists():
                raise FileNotFoundError(f"Missing posterior for analysis: {p}")

        noisy_metrics = _condition_metrics(noisy_paths)
        noiseless_metrics = _condition_metrics(noiseless_paths)
        ratios = {
            "std_sum_ratio_noiseless_over_noisy": _safe_ratio(
                noiseless_metrics["std_sum_mean"],
                noisy_metrics["std_sum_mean"],
            ),
            "fom3_ratio_noiseless_over_noisy": _safe_ratio(
                noiseless_metrics["fom3_mean"],
                noisy_metrics["fom3_mean"],
            ),
            "sigma8_std_ratio_noiseless_over_noisy": _safe_ratio(
                noiseless_metrics["sigma8_std_mean"],
                noisy_metrics["sigma8_std_mean"],
            ),
            "om_s8_area_ratio_noiseless_over_noisy": _safe_ratio(
                noiseless_metrics["om_s8_area_mean"],
                noisy_metrics["om_s8_area_mean"],
            ),
        }

        summary["conditions"][cond] = {
            "noisy": noisy_metrics,
            "noiseless": noiseless_metrics,
            "ratios": ratios,
        }

        csv_rows.append(
            {
                "condition": cond,
                **{f"noisy_{k}": v for k, v in noisy_metrics.items()},
                **{f"noiseless_{k}": v for k, v in noiseless_metrics.items()},
                **ratios,
            }
        )

        _plot_overlay(
            out_path=output_root / "figures" / f"overlay_{cond}_noisy_vs_noiseless_combined.png",
            noisy_samples=_concat(noisy_paths),
            noiseless_samples=_concat(noiseless_paths),
            title=f"CNN tomo4 {cond}: noisy vs noiseless (combined seeds)",
        )

    (output_root / "noisy_vs_noiseless_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    with open(output_root / "noisy_vs_noiseless_summary.csv", "w", encoding="utf-8", newline="") as f:
        if csv_rows:
            fieldnames = list(csv_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    # Human-readable report.
    report_lines = [
        "# CNN noiseless vs noisy comparison",
        "",
        f"- Baseline noisy root: `{baseline_root}`",
        f"- Noiseless run root: `{output_root}`",
        f"- Seeds: `{','.join(str(s) for s in seeds)}`",
        "",
        "## Shrink/expansion summary",
        "",
        "| condition | std_ratio (noiseless/noisy) | fom_ratio | sigma8_std_ratio | Om-s8 area_ratio |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for cond in ("nobnt", "bnt"):
        ratios = summary["conditions"][cond]["ratios"]
        report_lines.append(
            f"| `{cond}` | "
            f"{ratios['std_sum_ratio_noiseless_over_noisy']:.4f} | "
            f"{ratios['fom3_ratio_noiseless_over_noisy']:.4f} | "
            f"{ratios['sigma8_std_ratio_noiseless_over_noisy']:.4f} | "
            f"{ratios['om_s8_area_ratio_noiseless_over_noisy']:.4f} |"
        )
    report_lines.extend(
        [
            "",
            "Overlay figures:",
            "- `figures/overlay_nobnt_noisy_vs_noiseless_combined.png`",
            "- `figures/overlay_bnt_noisy_vs_noiseless_combined.png`",
        ]
    )
    (output_root / "CNN_NOISELESS_VS_NOISY_REPORT.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print(f"Noiseless comparison complete. Outputs: {output_root}")


if __name__ == "__main__":
    main()
