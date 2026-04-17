#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


DEFAULT_REPO_ROOT = str(Path(__file__).resolve().parents[2])
DEFAULT_OUTPUT_ROOT = (
    Path(DEFAULT_REPO_ROOT)
    / "scripts"
    / "sbi"
    / "results"
    / "final"
    / "paper_sbi_consolidation"
    / "cnn_bnt_noise_curriculum_campaign"
)
TRUTH = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493], dtype=np.float64)
PARAM_NAMES = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]


@dataclass(frozen=True)
class Job:
    name: str
    command: List[str]
    log_path: Path


@dataclass(frozen=True)
class CampaignConfig:
    name: str
    seeds: Tuple[int, ...]
    compressor_arch: str
    compressor_dim: int
    compressor_steps: int
    compressor_conv_channels: str
    compressor_dense_width: int
    compressor_pool_window: int
    compressor_pool_stride: int
    resnet_small_channels: str
    resnet_small_blocks: str
    resnet_head_width: int
    resnet_v2: bool
    standardize_summary: bool
    summary_clip_value: float
    flow_steps: int
    nvp_layers: int
    nvp_hidden: int
    batch_size: int
    patience: int
    use_noise_curriculum: bool
    curriculum_sigma_factors: str
    curriculum_stage_fracs: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run matched no-BNT/BNT CNN campaigns with optional compressor "
            "noise curriculum for plain and ResNet baselines."
        )
    )
    p.add_argument("--repo-root", type=str, default=DEFAULT_REPO_ROOT)
    p.add_argument("--conda-env", type=str, default="jaxili")
    p.add_argument("--gpus", type=str, default="0,1,2,3")
    p.add_argument("--xla-mem-fraction", type=float, default=0.6)
    p.add_argument("--map-kind", type=str, default="nbody")
    p.add_argument(
        "--tfds-name",
        type=str,
        default="NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48",
    )
    p.add_argument(
        "--compressor-train-split",
        type=str,
        default="train[:70%]",
        help="TFDS split used to train compressor checkpoints.",
    )
    p.add_argument(
        "--compressor-val-split",
        type=str,
        default="test",
        help="TFDS split used for compressor validation/test loss.",
    )
    p.add_argument(
        "--nde-train-split",
        type=str,
        default="train[70%:]",
        help="TFDS split used to build NDE training summaries.",
    )
    p.add_argument(
        "--nde-val-split",
        type=str,
        default="test",
        help="TFDS split used to build NDE validation summaries.",
    )
    p.add_argument("--field-size", type=int, default=20)
    p.add_argument("--field-npix", type=int, default=160)
    p.add_argument("--nbins", type=int, default=4)
    p.add_argument("--tomo-bin-indices", type=str, default="1,2,3,4")
    p.add_argument("--npe-samples", type=int, default=100_000)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--compressor-save-every", type=int, default=3000)
    p.add_argument("--compressor-batch-size", type=int, default=128)
    p.add_argument("--compressor-lr", type=float, default=5e-4)
    p.add_argument("--ds-batch-size", type=int, default=500)
    p.add_argument(
        "--seeds",
        type=str,
        default="41,42,43,44,45",
        help="Comma-separated seeds used for all campaign configs.",
    )
    p.add_argument(
        "--curriculum-sigma-factors",
        type=str,
        default="0.0,0.25,0.5,0.75,1.0",
        help="Curriculum sigma multipliers passed to compressor training.",
    )
    p.add_argument(
        "--curriculum-stage-fracs",
        type=str,
        default="0.10,0.15,0.20,0.25,0.30",
        help="Curriculum stage fractions passed to compressor training.",
    )
    p.add_argument(
        "--resnet18-slowramp-stage-fracs",
        type=str,
        default="0.20,0.25,0.25,0.20,0.10",
        help=(
            "Stage fractions for follow-up variant "
            "`resnet18_curriculum_slowramp` (must sum to 1.0)."
        ),
    )
    p.add_argument(
        "--resnet18-long-compressor-steps",
        type=int,
        default=22500,
        help=(
            "Compressor steps for follow-up variant "
            "`resnet18_curriculum_long22k`."
        ),
    )
    p.add_argument(
        "--plain-fullnoise-match-compressor-steps",
        type=int,
        default=150000,
        help=(
            "Compressor steps for full-noise-matched variant "
            "`plain_curriculum_fullnoise_match`."
        ),
    )
    p.add_argument(
        "--plain-fullnoise-match-stage-fracs",
        type=str,
        default="0.05,0.05,0.05,0.05,0.80",
        help=(
            "Stage fractions for full-noise-matched variant "
            "`plain_curriculum_fullnoise_match` (must sum to 1.0)."
        ),
    )
    p.add_argument(
        "--resnet18-fullnoise-match-compressor-steps",
        type=int,
        default=20000,
        help=(
            "Compressor steps for full-noise-matched variant "
            "`resnet18_curriculum_fullnoise_match`."
        ),
    )
    p.add_argument(
        "--resnet18-fullnoise-match-stage-fracs",
        type=str,
        default="0.05,0.05,0.05,0.10,0.75",
        help=(
            "Stage fractions for full-noise-matched variant "
            "`resnet18_curriculum_fullnoise_match` (must sum to 1.0)."
        ),
    )
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--require-disjoint-train-examples",
        dest="require_disjoint_train_examples",
        action="store_true",
        help=(
            "Require disjoint exact training examples between compressor and NDE "
            "splits (no shared (cosmology, patch) combinations)."
        ),
    )
    p.add_argument(
        "--allow-overlap-train-examples",
        dest="require_disjoint_train_examples",
        action="store_false",
        help="Disable strict disjoint-example check in the CNN pipeline.",
    )
    p.set_defaults(require_disjoint_train_examples=True)
    p.add_argument(
        "--run-names",
        type=str,
        default="plain_ref,plain_curriculum,resnet18_ref,resnet18_curriculum",
        help=(
            "Comma-separated subset of configs to run. Available: "
            "plain_ref, plain_curriculum, plain_curriculum_fullnoise_match, "
            "resnet18_ref, resnet18_curriculum, "
            "resnet18_curriculum_fullnoise_match, "
            "resnet18_curriculum_slowramp, resnet18_curriculum_long22k."
        ),
    )
    return p.parse_args()


def _csv_tokens(value: str) -> List[str]:
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def _parse_seed_list(value: str) -> Tuple[int, ...]:
    seeds = []
    for token in _csv_tokens(value):
        seed = int(token)
        if seed < 0:
            raise ValueError("Seeds must be non-negative integers.")
        seeds.append(seed)
    if not seeds:
        raise ValueError("--seeds cannot be empty.")
    return tuple(seeds)


def _default_configs(
    seeds: Tuple[int, ...],
    curriculum_sigma_factors: str,
    curriculum_stage_fracs: str,
    resnet18_slowramp_stage_fracs: str,
    resnet18_long_compressor_steps: int,
    plain_fullnoise_match_compressor_steps: int,
    plain_fullnoise_match_stage_fracs: str,
    resnet18_fullnoise_match_compressor_steps: int,
    resnet18_fullnoise_match_stage_fracs: str,
) -> Dict[str, CampaignConfig]:
    plain_common = dict(
        seeds=seeds,
        compressor_arch="plain",
        compressor_dim=10,
        compressor_steps=120000,
        compressor_conv_channels="64,128,256",
        compressor_dense_width=256,
        compressor_pool_window=16,
        compressor_pool_stride=8,
        resnet_small_channels="64,128,256",
        resnet_small_blocks="2,2,2",
        resnet_head_width=256,
        resnet_v2=False,
        standardize_summary=False,
        summary_clip_value=5.0,
        flow_steps=10000,
        nvp_layers=8,
        nvp_hidden=256,
        batch_size=256,
        patience=50,
        curriculum_sigma_factors=curriculum_sigma_factors,
        curriculum_stage_fracs=curriculum_stage_fracs,
    )
    resnet18_common = dict(
        seeds=seeds,
        compressor_arch="resnet18",
        compressor_dim=6,
        compressor_steps=15000,
        compressor_conv_channels="64,128,256",
        compressor_dense_width=256,
        compressor_pool_window=16,
        compressor_pool_stride=8,
        resnet_small_channels="64,128,256",
        resnet_small_blocks="2,2,2",
        resnet_head_width=256,
        resnet_v2=False,
        standardize_summary=True,
        summary_clip_value=5.0,
        flow_steps=10000,
        nvp_layers=10,
        nvp_hidden=320,
        batch_size=256,
        patience=35,
        curriculum_sigma_factors=curriculum_sigma_factors,
        curriculum_stage_fracs=curriculum_stage_fracs,
    )
    return {
        "plain_ref": CampaignConfig(
            name="plain_ref",
            use_noise_curriculum=False,
            **plain_common,
        ),
        "plain_curriculum": CampaignConfig(
            name="plain_curriculum",
            use_noise_curriculum=True,
            **plain_common,
        ),
        "plain_curriculum_fullnoise_match": CampaignConfig(
            name="plain_curriculum_fullnoise_match",
            use_noise_curriculum=True,
            **dict(
                plain_common,
                compressor_steps=plain_fullnoise_match_compressor_steps,
                curriculum_stage_fracs=plain_fullnoise_match_stage_fracs,
            ),
        ),
        "resnet18_ref": CampaignConfig(
            name="resnet18_ref",
            use_noise_curriculum=False,
            **resnet18_common,
        ),
        "resnet18_curriculum": CampaignConfig(
            name="resnet18_curriculum",
            use_noise_curriculum=True,
            **resnet18_common,
        ),
        "resnet18_curriculum_fullnoise_match": CampaignConfig(
            name="resnet18_curriculum_fullnoise_match",
            use_noise_curriculum=True,
            **dict(
                resnet18_common,
                compressor_steps=resnet18_fullnoise_match_compressor_steps,
                curriculum_stage_fracs=resnet18_fullnoise_match_stage_fracs,
            ),
        ),
        "resnet18_curriculum_slowramp": CampaignConfig(
            name="resnet18_curriculum_slowramp",
            use_noise_curriculum=True,
            **dict(
                resnet18_common,
                curriculum_stage_fracs=resnet18_slowramp_stage_fracs,
            ),
        ),
        "resnet18_curriculum_long22k": CampaignConfig(
            name="resnet18_curriculum_long22k",
            use_noise_curriculum=True,
            **dict(
                resnet18_common,
                compressor_steps=resnet18_long_compressor_steps,
            ),
        ),
    }


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
                job.log_path.write_text(f"[dry-run] {' '.join(cmd)}\n", encoding="utf-8")
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
        log_path.write_text(f"[dry-run] prepare tfds: {tfds_name}\n", encoding="utf-8")
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


def _compressor_paths(config_root: Path, condition: str, compressor_steps: int) -> Dict[str, Path]:
    base = (
        config_root
        / "compressor"
        / condition
        / "vmim"
        / "nbody"
        / "sigma_0.26"
        / "gal_density_30"
        / "bin_4"
    )
    return {
        "params": base / f"params_nd_compressor_batch{compressor_steps}.pkl",
        "state": base / f"opt_state_resnet_batch{compressor_steps}.pkl",
    }


def _resolve_compressor_paths(
    config_root: Path,
    condition: str,
    compressor_steps: int,
) -> Dict[str, object]:
    requested = _compressor_paths(config_root, condition, compressor_steps)
    params = requested["params"]
    state = requested["state"]
    if params.exists() and state.exists():
        return {
            "params": params,
            "state": state,
            "resolved_step": int(compressor_steps),
            "requested_step": int(compressor_steps),
            "used_fallback": False,
        }

    base = params.parent
    pattern = re.compile(r"params_nd_compressor_batch(\d+)\.pkl$")
    candidates: List[Tuple[int, Path, Path]] = []
    for p in base.glob("params_nd_compressor_batch*.pkl"):
        m = pattern.match(p.name)
        if m is None:
            continue
        step = int(m.group(1))
        s = base / f"opt_state_resnet_batch{step}.pkl"
        if not s.exists():
            continue
        candidates.append((step, p, s))

    if not candidates:
        return {
            "params": params,
            "state": state,
            "resolved_step": int(compressor_steps),
            "requested_step": int(compressor_steps),
            "used_fallback": False,
        }

    within = [row for row in candidates if row[0] <= compressor_steps]
    chosen = max(within if within else candidates, key=lambda row: row[0])
    return {
        "params": chosen[1],
        "state": chosen[2],
        "resolved_step": int(chosen[0]),
        "requested_step": int(compressor_steps),
        "used_fallback": int(chosen[0]) != int(compressor_steps),
    }


def _posterior_path(config_root: Path, config_name: str, condition: str, seed: int) -> Path:
    return (
        config_root
        / "posteriors"
        / f"cnn_tomo4_20deg160_{condition}_{config_name}_s{seed}.npy"
    )


def _fom3(samples: np.ndarray) -> Tuple[float, float, float, bool]:
    cov3 = np.cov(samples[:, :3], rowvar=False)
    sign, logdet = np.linalg.slogdet(cov3)
    if sign <= 0:
        return float("nan"), float("nan"), float(logdet), False
    det_cov3 = float(np.exp(logdet))
    return float(np.exp(-0.5 * logdet)), det_cov3, float(logdet), True


def _metrics_for_paths(paths_by_cond: Dict[str, List[Path]]) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for cond, paths in paths_by_cond.items():
        for path in paths:
            samples = np.load(path)
            std_sum = float(np.sum(np.std(samples, axis=0)))
            sigma8_std = float(np.std(samples[:, 1]))
            bias_l2 = float(np.linalg.norm(np.mean(samples, axis=0) - TRUTH))
            fom3, det_cov3, logdet_cov3, valid_fom3 = _fom3(samples)
            seed = int(path.stem.split("_s")[-1])
            rows.append(
                {
                    "condition": cond,
                    "seed": seed,
                    "file": str(path),
                    "std_sum": std_sum,
                    "sigma8_std": sigma8_std,
                    "bias_l2": bias_l2,
                    "fom3": fom3,
                    "det_cov3": det_cov3,
                    "logdet_cov3": logdet_cov3,
                    "valid_fom3": bool(valid_fom3),
                }
            )

    def _agg(cond: str) -> Dict[str, object]:
        vals = [r for r in rows if r["condition"] == cond]
        stds = np.array([float(v["std_sum"]) for v in vals], dtype=np.float64)
        sig8 = np.array([float(v["sigma8_std"]) for v in vals], dtype=np.float64)
        biases = np.array([float(v["bias_l2"]) for v in vals], dtype=np.float64)
        foms = np.array([float(v["fom3"]) for v in vals], dtype=np.float64)
        valid_foms = foms[np.isfinite(foms)]
        return {
            "n": int(len(vals)),
            "std_sum_mean": float(np.mean(stds)) if len(stds) else float("nan"),
            "std_sum_std": float(np.std(stds, ddof=1)) if len(stds) > 1 else 0.0,
            "sigma8_std_mean": float(np.mean(sig8)) if len(sig8) else float("nan"),
            "sigma8_std_std": float(np.std(sig8, ddof=1)) if len(sig8) > 1 else 0.0,
            "bias_l2_mean": float(np.mean(biases)) if len(biases) else float("nan"),
            "bias_l2_std": float(np.std(biases, ddof=1)) if len(biases) > 1 else 0.0,
            "fom3_mean": float(np.mean(valid_foms)) if len(valid_foms) else float("nan"),
            "fom3_std": (
                float(np.std(valid_foms, ddof=1)) if len(valid_foms) > 1 else 0.0
            ),
            "valid_fom3_count": int(len(valid_foms)),
        }

    nobnt = _agg("nobnt")
    bnt = _agg("bnt")
    std_inflation = (
        float(bnt["std_sum_mean"] / nobnt["std_sum_mean"])
        if np.isfinite(float(nobnt["std_sum_mean"])) and float(nobnt["std_sum_mean"]) != 0.0
        else float("nan")
    )
    fom_ratio = (
        float(bnt["fom3_mean"] / nobnt["fom3_mean"])
        if np.isfinite(float(nobnt["fom3_mean"])) and float(nobnt["fom3_mean"]) != 0.0
        else float("nan")
    )
    sigma8_ratio = (
        float(bnt["sigma8_std_mean"] / nobnt["sigma8_std_mean"])
        if np.isfinite(float(nobnt["sigma8_std_mean"])) and float(nobnt["sigma8_std_mean"]) != 0.0
        else float("nan")
    )
    return {
        "rows": rows,
        "nobnt": nobnt,
        "bnt": bnt,
        "inflation_std_sum_bnt_over_nobnt": std_inflation,
        "fom3_ratio_bnt_over_nobnt": fom_ratio,
        "sigma8_std_ratio_bnt_over_nobnt": sigma8_ratio,
    }


def _rank_score(metric_block: Dict[str, object]) -> float:
    infl = float(metric_block.get("inflation_std_sum_bnt_over_nobnt", float("nan")))
    fomr = float(metric_block.get("fom3_ratio_bnt_over_nobnt", float("nan")))
    if not np.isfinite(infl) or not np.isfinite(fomr):
        return float("inf")
    return float(abs(infl - 1.0) + abs(fomr - 1.0))


def _import_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from getdist import MCSamples, plots as gplot

    return plt, MCSamples, gplot


def _plot_overlay(
    out_path: Path,
    samples_nobnt: np.ndarray,
    samples_bnt: np.ndarray,
    title: str,
    dpi: int,
) -> None:
    plt, MCSamples, gplot = _import_plotting()
    chain_nobnt = MCSamples(
        samples=samples_nobnt,
        names=PARAM_NAMES,
        labels=PARAM_NAMES,
        label="no-BNT",
        settings={"smooth_scale_2D": 0.7, "smooth_scale_1D": 0.7},
    )
    chain_bnt = MCSamples(
        samples=samples_bnt,
        names=PARAM_NAMES,
        labels=PARAM_NAMES,
        label="BNT",
        settings={"smooth_scale_2D": 0.7, "smooth_scale_1D": 0.7},
    )
    g = gplot.get_subplot_plotter(subplot_size=1.4)
    g.settings.alpha_filled_add = 0.55
    g.triangle_plot(
        [chain_nobnt, chain_bnt],
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


def _write_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "name",
        "compressor_arch",
        "compressor_dim",
        "compressor_steps",
        "use_noise_curriculum",
        "inflation_std_sum_bnt_over_nobnt",
        "fom3_ratio_bnt_over_nobnt",
        "sigma8_std_ratio_bnt_over_nobnt",
        "nobnt_fom3_mean",
        "bnt_fom3_mean",
        "rank_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "name": row["name"],
                    "compressor_arch": row["config"]["compressor_arch"],
                    "compressor_dim": row["config"]["compressor_dim"],
                    "compressor_steps": row["config"]["compressor_steps"],
                    "use_noise_curriculum": row["config"]["use_noise_curriculum"],
                    "inflation_std_sum_bnt_over_nobnt": row[
                        "inflation_std_sum_bnt_over_nobnt"
                    ],
                    "fom3_ratio_bnt_over_nobnt": row["fom3_ratio_bnt_over_nobnt"],
                    "sigma8_std_ratio_bnt_over_nobnt": row[
                        "sigma8_std_ratio_bnt_over_nobnt"
                    ],
                    "nobnt_fom3_mean": row["nobnt"]["fom3_mean"],
                    "bnt_fom3_mean": row["bnt"]["fom3_mean"],
                    "rank_score": row["rank_score"],
                }
            )


def _write_report(out_root: Path, ranked: List[Dict[str, object]]) -> None:
    if not ranked:
        return
    top = ranked[0]
    lines = [
        "# CNN noise-curriculum campaign report",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Best by rank score: `{top['name']}`",
        f"- Rank score formula: `abs(infl-1)+abs(fom_ratio-1)`",
        "",
        "## Ranked variants",
        "",
    ]
    for idx, row in enumerate(ranked, start=1):
        lines.extend(
            [
                f"{idx}. `{row['name']}`",
                (
                    "   - arch="
                    f"{row['config']['compressor_arch']}, "
                    f"curriculum={row['config']['use_noise_curriculum']}, "
                    f"infl={row['inflation_std_sum_bnt_over_nobnt']:.4f}, "
                    f"fom_ratio={row['fom3_ratio_bnt_over_nobnt']:.4f}, "
                    f"sigma8_ratio={row['sigma8_std_ratio_bnt_over_nobnt']:.4f}, "
                    f"rank={row['rank_score']:.4f}"
                ),
            ]
        )
    (out_root / "CNN_NOISE_CURRICULUM_REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    gpus = _csv_tokens(args.gpus)
    if not gpus:
        raise ValueError("--gpus cannot be empty.")
    if args.xla_mem_fraction <= 0.0 or args.xla_mem_fraction > 1.0:
        raise ValueError("--xla-mem-fraction must be in (0, 1].")
    if args.resnet18_long_compressor_steps < 1:
        raise ValueError("--resnet18-long-compressor-steps must be >= 1.")
    if args.plain_fullnoise_match_compressor_steps < 1:
        raise ValueError("--plain-fullnoise-match-compressor-steps must be >= 1.")
    if args.resnet18_fullnoise_match_compressor_steps < 1:
        raise ValueError(
            "--resnet18-fullnoise-match-compressor-steps must be >= 1."
        )

    seeds = _parse_seed_list(args.seeds)
    configs_by_name = _default_configs(
        seeds=seeds,
        curriculum_sigma_factors=args.curriculum_sigma_factors,
        curriculum_stage_fracs=args.curriculum_stage_fracs,
        resnet18_slowramp_stage_fracs=args.resnet18_slowramp_stage_fracs,
        resnet18_long_compressor_steps=args.resnet18_long_compressor_steps,
        plain_fullnoise_match_compressor_steps=args.plain_fullnoise_match_compressor_steps,
        plain_fullnoise_match_stage_fracs=args.plain_fullnoise_match_stage_fracs,
        resnet18_fullnoise_match_compressor_steps=args.resnet18_fullnoise_match_compressor_steps,
        resnet18_fullnoise_match_stage_fracs=args.resnet18_fullnoise_match_stage_fracs,
    )
    requested = _csv_tokens(args.run_names)
    if not requested:
        raise ValueError("--run-names cannot be empty.")
    unknown = [name for name in requested if name not in configs_by_name]
    if unknown:
        raise ValueError(f"Unknown run names: {unknown}.")
    configs = [configs_by_name[name] for name in requested]

    cnn_script = str(repo_root / "scripts" / "sbi" / "npe_cnn_nbody_tomo.py")

    manifest = {
        "repo_root": str(repo_root),
        "output_root": str(out_root),
        "conda_env": args.conda_env,
        "gpus": gpus,
        "xla_mem_fraction": float(args.xla_mem_fraction),
        "map_kind": args.map_kind,
        "tfds_name": args.tfds_name,
        "compressor_train_split": args.compressor_train_split,
        "compressor_val_split": args.compressor_val_split,
        "nde_train_split": args.nde_train_split,
        "nde_val_split": args.nde_val_split,
        "require_disjoint_train_examples": bool(args.require_disjoint_train_examples),
        "field_size": int(args.field_size),
        "field_npix": int(args.field_npix),
        "nbins": int(args.nbins),
        "tomo_bin_indices": str(args.tomo_bin_indices),
        "npe_samples": int(args.npe_samples),
        "compressor_batch_size": int(args.compressor_batch_size),
        "compressor_lr": float(args.compressor_lr),
        "compressor_save_every": int(args.compressor_save_every),
        "save_every": int(args.save_every),
        "ds_batch_size": int(args.ds_batch_size),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configs": [
            {
                "name": c.name,
                "seeds": list(c.seeds),
                "compressor_arch": c.compressor_arch,
                "compressor_dim": c.compressor_dim,
                "compressor_steps": c.compressor_steps,
                "compressor_conv_channels": c.compressor_conv_channels,
                "compressor_dense_width": c.compressor_dense_width,
                "compressor_pool_window": c.compressor_pool_window,
                "compressor_pool_stride": c.compressor_pool_stride,
                "resnet_small_channels": c.resnet_small_channels,
                "resnet_small_blocks": c.resnet_small_blocks,
                "resnet_head_width": c.resnet_head_width,
                "resnet_v2": bool(c.resnet_v2),
                "standardize_summary": bool(c.standardize_summary),
                "summary_clip_value": float(c.summary_clip_value),
                "flow_steps": c.flow_steps,
                "nvp_layers": c.nvp_layers,
                "nvp_hidden": c.nvp_hidden,
                "batch_size": c.batch_size,
                "patience": c.patience,
                "use_noise_curriculum": bool(c.use_noise_curriculum),
                "curriculum_sigma_factors": c.curriculum_sigma_factors,
                "curriculum_stage_fracs": c.curriculum_stage_fracs,
            }
            for c in configs
        ],
    }
    (out_root / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    ensure_tfds_prepared(
        args.tfds_name,
        args.conda_env,
        repo_root=repo_root,
        log_path=out_root / "logs" / "tfds_prepare.log",
        dry_run=args.dry_run,
    )

    all_job_results: List[Dict[str, object]] = []
    per_config_metrics: List[Dict[str, object]] = []

    for config in configs:
        cfg_root = out_root / config.name
        (cfg_root / "logs").mkdir(parents=True, exist_ok=True)
        (cfg_root / "posteriors").mkdir(parents=True, exist_ok=True)
        (cfg_root / "figures").mkdir(parents=True, exist_ok=True)

        train_jobs: List[Job] = []
        for cond in ("nobnt", "bnt"):
            comp_paths = _compressor_paths(cfg_root, cond, config.compressor_steps)
            if comp_paths["params"].exists() and comp_paths["state"].exists():
                continue
            cond_flag = ["--apply-bnt"] if cond == "bnt" else []
            curriculum_flags = []
            if config.use_noise_curriculum:
                curriculum_flags = [
                    "--compressor-noise-curriculum",
                    "--compressor-curriculum-sigma-factors",
                    config.curriculum_sigma_factors,
                    "--compressor-curriculum-stage-fracs",
                    config.curriculum_stage_fracs,
                ]
            train_jobs.append(
                Job(
                    name=f"train::{config.name}::{cond}",
                    log_path=cfg_root / "logs" / f"train_{cond}.log",
                    command=[
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
                        "--compressor-train-split",
                        args.compressor_train_split,
                        "--compressor-val-split",
                        args.compressor_val_split,
                        "--nde-train-split",
                        args.nde_train_split,
                        "--nde-val-split",
                        args.nde_val_split,
                        "--field-size",
                        str(args.field_size),
                        "--field-npix",
                        str(args.field_npix),
                        "--nbins",
                        str(args.nbins),
                        "--tomo-bin-indices",
                        args.tomo_bin_indices,
                        "--cache-dir",
                        str(cfg_root / "cache" / f"{cond}_train"),
                        "--save-dir",
                        str(cfg_root / "compressor" / cond),
                        "--train-compressor",
                        "--compressor-dim",
                        str(config.compressor_dim),
                        "--compressor-arch",
                        config.compressor_arch,
                        "--compressor-conv-channels",
                        config.compressor_conv_channels,
                        "--compressor-dense-width",
                        str(config.compressor_dense_width),
                        "--compressor-pool-window",
                        str(config.compressor_pool_window),
                        "--compressor-pool-stride",
                        str(config.compressor_pool_stride),
                        "--resnet-small-channels",
                        config.resnet_small_channels,
                        "--resnet-small-blocks",
                        config.resnet_small_blocks,
                        "--resnet-head-width",
                        str(config.resnet_head_width),
                        "--compressor-steps",
                        str(config.compressor_steps),
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
                    ]
                    + curriculum_flags
                    + (["--resnet-v2"] if config.resnet_v2 else [])
                    + (
                        ["--require-disjoint-train-examples"]
                        if args.require_disjoint_train_examples
                        else []
                    )
                    + cond_flag,
                )
            )

        if train_jobs:
            train_results = run_jobs_parallel(
                train_jobs,
                gpus,
                repo_root,
                xla_mem_fraction=args.xla_mem_fraction,
                dry_run=args.dry_run,
            )
            all_job_results.extend(train_results)
            require_success(train_results, f"Compressor training ({config.name})")

        eval_jobs: List[Job] = []
        resolved_compressor_steps: Dict[str, int] = {}
        for cond in ("nobnt", "bnt"):
            comp_paths = _resolve_compressor_paths(cfg_root, cond, config.compressor_steps)
            resolved_compressor_steps[cond] = int(comp_paths["resolved_step"])
            if (
                not args.dry_run
                and (
                    not Path(comp_paths["params"]).exists()
                    or not Path(comp_paths["state"]).exists()
                )
            ):
                raise FileNotFoundError(
                    f"Missing compressor checkpoint for {config.name}/{cond}: {comp_paths}"
                )
            if bool(comp_paths["used_fallback"]):
                print(
                    f"[{config.name}/{cond}] requested compressor step "
                    f"{comp_paths['requested_step']} not found; using step "
                    f"{comp_paths['resolved_step']} checkpoint."
                )
            for seed in config.seeds:
                posterior_out = _posterior_path(cfg_root, config.name, cond, seed)
                if posterior_out.exists():
                    continue
                cond_flag = ["--apply-bnt"] if cond == "bnt" else []
                std_flag = (
                    ["--standardize-summary"]
                    if config.standardize_summary
                    else ["--no-standardize-summary"]
                )
                curriculum_flags = []
                if config.use_noise_curriculum:
                    curriculum_flags = [
                        "--compressor-noise-curriculum",
                        "--compressor-curriculum-sigma-factors",
                        config.curriculum_sigma_factors,
                        "--compressor-curriculum-stage-fracs",
                        config.curriculum_stage_fracs,
                    ]
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
                    "--compressor-train-split",
                    args.compressor_train_split,
                    "--compressor-val-split",
                    args.compressor_val_split,
                    "--nde-train-split",
                    args.nde_train_split,
                    "--nde-val-split",
                    args.nde_val_split,
                    "--field-size",
                    str(args.field_size),
                    "--field-npix",
                    str(args.field_npix),
                    "--nbins",
                    str(args.nbins),
                    "--tomo-bin-indices",
                    args.tomo_bin_indices,
                    "--cache-dir",
                    str(cfg_root / "cache" / f"{cond}_eval"),
                    "--save-dir",
                    str(cfg_root / "eval" / cond / f"seed_{seed}"),
                    "--compressor-dim",
                    str(config.compressor_dim),
                    "--compressor-arch",
                    config.compressor_arch,
                    "--compressor-conv-channels",
                    config.compressor_conv_channels,
                    "--compressor-dense-width",
                    str(config.compressor_dense_width),
                    "--compressor-pool-window",
                    str(config.compressor_pool_window),
                    "--compressor-pool-stride",
                    str(config.compressor_pool_stride),
                    "--resnet-small-channels",
                    config.resnet_small_channels,
                    "--resnet-small-blocks",
                    config.resnet_small_blocks,
                    "--resnet-head-width",
                    str(config.resnet_head_width),
                    "--compressor-params",
                    str(comp_paths["params"]),
                    "--compressor-state",
                    str(comp_paths["state"]),
                    "--total-steps",
                    str(config.flow_steps),
                    "--save-every",
                    str(args.save_every),
                    "--patience",
                    str(config.patience),
                    "--batch-size",
                    str(config.batch_size),
                    "--nvp-layers",
                    str(config.nvp_layers),
                    "--nvp-hidden",
                    str(config.nvp_hidden),
                    "--summary-clip-value",
                    str(config.summary_clip_value),
                    "--npe-samples",
                    str(args.npe_samples),
                    "--posterior-out",
                    str(posterior_out),
                    "--ds-batch-size",
                    str(args.ds_batch_size),
                ] + std_flag + curriculum_flags + (
                    ["--resnet-v2"] if config.resnet_v2 else []
                ) + (
                    ["--require-disjoint-train-examples"]
                    if args.require_disjoint_train_examples
                    else []
                ) + cond_flag
                if args.plot:
                    cmd.extend(
                        [
                            "--plot",
                            "--figure-out",
                            str(
                                cfg_root
                                / "figures"
                                / f"cnn_tomo4_20deg160_{cond}_{config.name}_s{seed}.png"
                            ),
                        ]
                    )
                eval_jobs.append(
                    Job(
                        name=f"eval::{config.name}::{cond}::s{seed}",
                        log_path=cfg_root / "logs" / f"eval_{cond}_s{seed}.log",
                        command=cmd,
                    )
                )

        if eval_jobs:
            eval_results = run_jobs_parallel(
                eval_jobs,
                gpus,
                repo_root,
                xla_mem_fraction=args.xla_mem_fraction,
                dry_run=args.dry_run,
            )
            all_job_results.extend(eval_results)
            require_success(eval_results, f"Posterior evaluation ({config.name})")

        if args.dry_run:
            continue

        paths = {
            "nobnt": [
                _posterior_path(cfg_root, config.name, "nobnt", seed) for seed in config.seeds
            ],
            "bnt": [
                _posterior_path(cfg_root, config.name, "bnt", seed) for seed in config.seeds
            ],
        }
        for cond in ("nobnt", "bnt"):
            for p in paths[cond]:
                if not p.exists():
                    raise FileNotFoundError(f"Missing posterior for {config.name}/{cond}: {p}")

        metric_block = _metrics_for_paths(paths)
        metric_block["name"] = config.name
        metric_block["seeds"] = list(config.seeds)
        metric_block["config"] = {
            "compressor_arch": config.compressor_arch,
            "compressor_dim": config.compressor_dim,
            "compressor_steps": config.compressor_steps,
            "resolved_compressor_steps": resolved_compressor_steps,
            "compressor_conv_channels": config.compressor_conv_channels,
            "compressor_dense_width": config.compressor_dense_width,
            "compressor_pool_window": config.compressor_pool_window,
            "compressor_pool_stride": config.compressor_pool_stride,
            "resnet_small_channels": config.resnet_small_channels,
            "resnet_small_blocks": config.resnet_small_blocks,
            "resnet_head_width": config.resnet_head_width,
            "resnet_v2": bool(config.resnet_v2),
            "standardize_summary": config.standardize_summary,
            "summary_clip_value": config.summary_clip_value,
            "flow_steps": config.flow_steps,
            "nvp_layers": config.nvp_layers,
            "nvp_hidden": config.nvp_hidden,
            "batch_size": config.batch_size,
            "patience": config.patience,
            "use_noise_curriculum": bool(config.use_noise_curriculum),
            "curriculum_sigma_factors": config.curriculum_sigma_factors,
            "curriculum_stage_fracs": config.curriculum_stage_fracs,
        }
        metric_block["rank_score"] = _rank_score(metric_block)
        per_config_metrics.append(metric_block)
        (cfg_root / "metrics.json").write_text(
            json.dumps(metric_block, indent=2),
            encoding="utf-8",
        )

        if args.plot:
            _plot_overlay(
                cfg_root / "figures" / f"overlay_{config.name}_combined_bnt_vs_nobnt.png",
                _concat(paths["nobnt"]),
                _concat(paths["bnt"]),
                f"CNN {config.name}: BNT vs no-BNT (combined seeds)",
                args.dpi,
            )

    ranked = sorted(
        per_config_metrics,
        key=lambda row: (
            float(row.get("rank_score", float("inf"))),
            abs(float(row.get("fom3_ratio_bnt_over_nobnt", float("inf"))) - 1.0),
        ),
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configs": per_config_metrics,
        "ranked_configs": [
            {
                "name": row["name"],
                "rank_score": row["rank_score"],
                "inflation_std_sum_bnt_over_nobnt": row[
                    "inflation_std_sum_bnt_over_nobnt"
                ],
                "fom3_ratio_bnt_over_nobnt": row["fom3_ratio_bnt_over_nobnt"],
                "sigma8_std_ratio_bnt_over_nobnt": row[
                    "sigma8_std_ratio_bnt_over_nobnt"
                ],
                "nobnt_fom3_mean": row["nobnt"]["fom3_mean"],
                "bnt_fom3_mean": row["bnt"]["fom3_mean"],
                "use_noise_curriculum": row["config"]["use_noise_curriculum"],
                "compressor_arch": row["config"]["compressor_arch"],
            }
            for row in ranked
        ],
        "rank_score_formula": "abs(inflation_std_sum_bnt_over_nobnt-1)+abs(fom3_ratio_bnt_over_nobnt-1)",
        "fom_parity_formula": "abs(fom3_ratio_bnt_over_nobnt-1)",
        "job_results": all_job_results,
    }
    (out_root / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    if not args.dry_run:
        _write_summary_csv(out_root / "comparison_summary.csv", ranked)
        _write_report(out_root, ranked)
    (out_root / "job_results.json").write_text(
        json.dumps(all_job_results, indent=2),
        encoding="utf-8",
    )
    print(f"Done. Outputs in: {out_root}")


if __name__ == "__main__":
    main()
