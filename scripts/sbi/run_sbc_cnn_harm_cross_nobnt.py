#!/usr/bin/env python3
"""SBC rank statistics for the auto+harmonic-cross no-BNT CNN posteriors.

Mirrors `run_sbc_cnn_nobnt.py` but consumes the per-seed harm-cross CNN
artefacts under `cnn_with_harm_cross_normalized/[resnet50_gn/]seed_{S}/`,
applies the saved per-channel standardization (mean/std/clip) to the
cached summaries before flow sampling, and dumps full posterior samples
for downstream TARP coverage testing.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import tensorflow_probability as tfp_root
from tensorflow_probability.substrates import jax as tfp_jax

if not hasattr(np, "issctype"):
    def _np_issctype(rep) -> bool:
        try:
            return issubclass(np.dtype(rep).type, np.generic)
        except Exception:
            return False
    np.issctype = _np_issctype  # type: ignore[attr-defined]

if not hasattr(tfp_root, "substrates"):
    class _TFPSubstrates:
        jax = tfp_jax

    tfp_root.substrates = _TFPSubstrates()  # type: ignore[attr-defined]
from sbi_lens.normflow.models import AffineCoupling, ConditionalRealNVP

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PARAMETER_ORDER = ["Omega_m", "sigma_8", "w0", "h0", "n_s", "Omega_b"]
FIDUCIAL_THETA = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493], dtype=np.float32)

ARCH_CHOICES = ("plain", "resnet50_gn")


def _default_baseline_for_arch(repo_root: Path, arch: str) -> Path:
    base = (
        repo_root
        / "scripts"
        / "sbi"
        / "results"
        / "exploratory"
        / "cnn_with_harm_cross_normalized"
    )
    return base if arch == "plain" else base / arch


def build_flow(n_cosmo_params: int, n_layers: int, hidden: int):
    bijector_fn = partial(
        AffineCoupling,
        layers=[hidden] * 2,
        activation=jax.nn.silu,
    )
    nf_factory = partial(
        ConditionalRealNVP,
        n_layers=n_layers,
        bijector_fn=bijector_fn,
    )

    class NF(hk.Module):
        def __call__(self, y):
            return nf_factory(n_cosmo_params)(y)

    @hk.transform
    def nf_sample(y, n_samples):
        return NF()(y).sample(n_samples, seed=hk.next_rng_key())

    return nf_sample


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_output = (
        repo_root
        / "scripts"
        / "sbi"
        / "results"
        / "diagnostics"
        / "sbc_cnn_harm_cross_nobnt"
    )

    p = argparse.ArgumentParser(
        description=(
            "Run SBC rank-statistics for CNN auto+harm-cross no-BNT, using cached "
            "compressor summaries and the trained flow checkpoint from the "
            "cnn_with_harm_cross_normalized[/<arch>]/seed_{S}/ tree."
        )
    )
    p.add_argument(
        "--compressor-arch",
        type=str,
        default="plain",
        choices=ARCH_CHOICES,
        help="Which harm-cross baseline to evaluate (selects baseline-root default).",
    )
    p.add_argument(
        "--baseline-root",
        type=Path,
        default=None,
        help="Overrides --compressor-arch's default baseline root.",
    )
    p.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed of the trained estimator to evaluate (per-seed dir name suffix).",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=default_output,
        help="Root output dir; results land under <output-root>/<arch>/seed_<S>/<run-tag>/.",
    )
    p.add_argument("--cache-split", type=str, default="val", choices=("train", "val"))
    p.add_argument("--n-ranks", type=int, required=True)
    p.add_argument("--posterior-samples", type=int, default=2000)
    p.add_argument("--rank-seed", type=int, default=12345)
    p.add_argument(
        "--nvp-layers",
        type=int,
        default=4,
        help="Matches the npe_cnn_nbody_tomo.py training default used for harm-cross.",
    )
    p.add_argument(
        "--nvp-hidden",
        type=int,
        default=128,
        help="Matches the npe_cnn_nbody_tomo.py training default used for harm-cross.",
    )
    p.add_argument("--rank-bins", type=int, default=20)
    p.add_argument("--nonfid-eps", type=float, default=1e-8)
    p.add_argument("--cuda-visible-devices", type=str, default=None)
    p.add_argument("--xla-mem-fraction", type=float, default=None)
    p.add_argument(
        "--dump-posterior-samples",
        action="store_true",
        default=True,
        help=(
            "If set (default ON for this runner), write posterior_samples.npz "
            "alongside sbc_ranks.npz containing the full (N, M, 6) posterior "
            "samples and theta arrays — needed for downstream TARP joint-coverage "
            "testing. Pass --no-dump-posterior-samples to disable."
        ),
    )
    p.add_argument(
        "--no-dump-posterior-samples",
        dest="dump_posterior_samples",
        action="store_false",
    )
    args = p.parse_args()
    if args.baseline_root is None:
        repo_root = Path(__file__).resolve().parents[2]
        args.baseline_root = _default_baseline_for_arch(repo_root, args.compressor_arch)
    return args


def _resolve_seed_paths(
    baseline_root: Path, seed: int, cache_split: str
) -> Dict[str, Path]:
    seed_dir = baseline_root / f"seed_{seed}"
    if not seed_dir.exists():
        raise FileNotFoundError(f"Missing seed directory: {seed_dir}")
    cache_name = "cnn_val.npz" if cache_split == "val" else "cnn_train.npz"
    cache_path = seed_dir / "cache" / cache_name
    flow_dir = seed_dir / "save_params" / "cnn_vmim" / "nbody" / "harmonic_nobnt"
    params_path = flow_dir / "params_cnn_flow_best.pkl"
    summary_path = flow_dir / "flow_training_summary.json"
    std_path = flow_dir / "cnn_summary_standardization.npz"
    cache_meta_path = seed_dir / "cache" / "cnn_cache_meta.npz"
    for label, path in {
        "cache": cache_path,
        "flow params": params_path,
        "flow summary": summary_path,
        "standardization": std_path,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {label} file: {path}")
    return {
        "seed_dir": seed_dir,
        "cache_path": cache_path,
        "cache_meta_path": cache_meta_path,
        "flow_dir": flow_dir,
        "params_path": params_path,
        "summary_path": summary_path,
        "std_path": std_path,
    }


def apply_standardization(
    x: np.ndarray, mean: np.ndarray, std: np.ndarray, clip_value: float
) -> np.ndarray:
    x_std = (x - mean[None, :]) / std[None, :]
    if np.isfinite(clip_value) and clip_value > 0:
        x_std = np.clip(x_std, -clip_value, clip_value)
    return x_std.astype(np.float32, copy=False)


def draw_posterior_samples(
    nf_sample,
    flow_params: hk.Params,
    summary_obs: np.ndarray,
    n_samples: int,
    rng_key: jax.Array,
    max_attempts: int = 8,
) -> Tuple[np.ndarray, jax.Array, int]:
    summary_obs = np.asarray(summary_obs, dtype=np.float32).reshape(1, -1)
    gathered: List[np.ndarray] = []
    total = 0
    attempts = 0
    key = rng_key
    while total < n_samples and attempts < max_attempts:
        need = n_samples - total
        key, subkey = jax.random.split(key)
        y_cond = np.repeat(summary_obs, repeats=need, axis=0)
        draw = np.asarray(nf_sample.apply(flow_params, subkey, y_cond, need))
        draw = draw[np.all(np.isfinite(draw), axis=1)]
        if draw.size:
            gathered.append(draw)
            total += int(draw.shape[0])
        attempts += 1
    if total < n_samples:
        raise RuntimeError(
            f"Could not collect {n_samples} finite posterior samples "
            f"(got {total}) after {attempts} attempts."
        )
    return np.concatenate(gathered, axis=0)[:n_samples], key, attempts


def rank_metrics(ranks: np.ndarray, n_post: int, n_bins: int) -> Dict[str, object]:
    n_rank, n_dim = ranks.shape
    expected_mean = n_post / 2.0
    expected_var = n_post * (n_post + 2) / 12.0
    per_param = []
    for j in range(n_dim):
        r = ranks[:, j].astype(np.float64)
        hist, _ = np.histogram(r, bins=n_bins, range=(0, n_post + 1))
        expected = n_rank / float(n_bins)
        chi2 = float(np.sum((hist - expected) ** 2 / max(expected, 1e-12)))
        per_param.append(
            {
                "name": PARAMETER_ORDER[j],
                "mean_rank": float(r.mean()),
                "var_rank": float(r.var(ddof=0)),
                "expected_mean_rank": float(expected_mean),
                "expected_var_rank": float(expected_var),
                "mean_rank_z": float(
                    (r.mean() - expected_mean) / np.sqrt(expected_var / n_rank)
                ),
                "chi2_uniform_bins": chi2,
                "hist_counts": hist.astype(int).tolist(),
            }
        )
    return {
        "n_rank": int(n_rank),
        "posterior_samples": int(n_post),
        "n_bins": int(n_bins),
        "parameter_order": PARAMETER_ORDER,
        "per_parameter": per_param,
    }


def plot_rank_histograms(
    ranks: np.ndarray,
    n_post: int,
    n_bins: int,
    output_path: Path,
    title_prefix: str,
) -> None:
    n_rank = ranks.shape[0]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    expected = n_rank / float(n_bins)
    for j, ax in enumerate(axes.flat):
        ax.hist(
            ranks[:, j],
            bins=n_bins,
            range=(0, n_post + 1),
            color="#4C78A8",
            alpha=0.9,
        )
        ax.axhline(expected, color="#E45756", linestyle="--", linewidth=1.2)
        ax.set_title(PARAMETER_ORDER[j])
        ax.set_xlabel("Rank")
        ax.set_ylabel("Count")
    fig.suptitle(f"{title_prefix} SBC rank histograms (N={n_rank}, M={n_post})")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.n_ranks <= 0:
        raise ValueError("--n-ranks must be > 0")
    if args.posterior_samples <= 0:
        raise ValueError("--posterior-samples must be > 0")

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.xla_mem_fraction is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{float(args.xla_mem_fraction):.4f}"

    baseline_root = args.baseline_root.resolve()
    paths = _resolve_seed_paths(baseline_root, args.seed, args.cache_split)

    output_root = args.output_root.resolve() / args.compressor_arch / f"seed_{args.seed}"
    output_root.mkdir(parents=True, exist_ok=True)
    run_tag = f"n{args.n_ranks}_m{args.posterior_samples}_seed{args.rank_seed}"
    run_dir = output_root / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    cache = np.load(paths["cache_path"])
    theta = np.asarray(cache["theta"], dtype=np.float32)
    summaries_raw = np.asarray(cache["x"], dtype=np.float32)
    if summaries_raw.shape[1] != 10:
        raise ValueError(
            f"Expected 10-channel harm-cross summaries, got shape {summaries_raw.shape}"
        )

    std_npz = np.load(paths["std_path"])
    std_mean = np.asarray(std_npz["mean"], dtype=np.float32)
    std_std = np.asarray(std_npz["std"], dtype=np.float32)
    clip_value = float(std_npz["clip_value"])
    if std_mean.shape != (10,) or std_std.shape != (10,):
        raise ValueError(
            f"Bad standardization shapes: mean {std_mean.shape}, std {std_std.shape}"
        )
    summaries = apply_standardization(summaries_raw, std_mean, std_std, clip_value)

    with open(paths["summary_path"], "r", encoding="utf-8") as f:
        flow_summary = json.load(f)
    with open(paths["params_path"], "rb") as f:
        flow_params = pickle.load(f)

    nonfid_mask = np.max(np.abs(theta - FIDUCIAL_THETA[None, :]), axis=1) > args.nonfid_eps
    valid_indices = np.where(nonfid_mask)[0]
    if len(valid_indices) < args.n_ranks:
        raise ValueError(
            f"Only {len(valid_indices)} non-fiducial rows available, need {args.n_ranks}."
        )

    rng = np.random.default_rng(args.rank_seed)
    selected_indices = rng.choice(valid_indices, size=args.n_ranks, replace=False)
    selected_theta = theta[selected_indices]
    selected_summaries = summaries[selected_indices]

    print(
        f"[SBC] arch={args.compressor_arch} seed={args.seed} "
        f"flow_best_val_loss={flow_summary.get('best_val_loss')} "
        f"best_step={flow_summary.get('best_step')}"
    )
    print(
        f"[SBC] Split={args.cache_split}, non-fid rows selected={args.n_ranks}, "
        f"posterior samples per row={args.posterior_samples}, clip={clip_value}"
    )

    nf_sample = build_flow(
        n_cosmo_params=selected_theta.shape[1],
        n_layers=args.nvp_layers,
        hidden=args.nvp_hidden,
    )

    ranks = np.zeros((args.n_ranks, selected_theta.shape[1]), dtype=np.int32)
    attempts_used = np.zeros(args.n_ranks, dtype=np.int32)
    key = jax.random.PRNGKey(args.rank_seed + 2026)

    samples_dump = (
        np.empty(
            (args.n_ranks, args.posterior_samples, selected_theta.shape[1]),
            dtype=np.float32,
        )
        if args.dump_posterior_samples
        else None
    )

    for i in range(args.n_ranks):
        draws, key, attempts = draw_posterior_samples(
            nf_sample=nf_sample,
            flow_params=flow_params,
            summary_obs=selected_summaries[i],
            n_samples=args.posterior_samples,
            rng_key=key,
        )
        ranks[i] = np.sum(draws < selected_theta[i][None, :], axis=0).astype(np.int32)
        attempts_used[i] = int(attempts)
        if samples_dump is not None:
            samples_dump[i] = draws.astype(np.float32, copy=False)
        if (i + 1) % 20 == 0 or i + 1 == args.n_ranks:
            print(f"[SBC] Processed {i + 1}/{args.n_ranks}")

    rank_npz = run_dir / "sbc_ranks.npz"
    np.savez_compressed(
        rank_npz,
        ranks=ranks,
        true_theta=selected_theta,
        selected_indices=selected_indices.astype(np.int64),
        posterior_samples=np.int32(args.posterior_samples),
        attempts_used=attempts_used,
        parameter_order=np.array(PARAMETER_ORDER, dtype=object),
    )

    samples_npz: Optional[Path] = None
    if samples_dump is not None:
        samples_npz = run_dir / "posterior_samples.npz"
        np.savez_compressed(
            samples_npz,
            samples=samples_dump,
            theta=selected_theta,
            selected_indices=selected_indices.astype(np.int64),
            posterior_samples_per_cosmology=np.int32(args.posterior_samples),
            parameter_order=np.array(PARAMETER_ORDER, dtype=object),
        )
        print(f"[SBC] Dumped posterior samples to {samples_npz} (shape {samples_dump.shape})")

    metrics = rank_metrics(
        ranks=ranks, n_post=args.posterior_samples, n_bins=args.rank_bins
    )
    metrics["attempts_summary"] = {
        "min": int(attempts_used.min()),
        "max": int(attempts_used.max()),
        "mean": float(attempts_used.mean()),
    }
    metrics_path = run_dir / "sbc_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plot_path = run_dir / "sbc_rank_hist.png"
    plot_rank_histograms(
        ranks=ranks,
        n_post=args.posterior_samples,
        n_bins=args.rank_bins,
        output_path=plot_path,
        title_prefix=f"CNN harm-cross no-BNT ({args.compressor_arch}) seed {args.seed}",
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "cwd": str(Path.cwd().resolve()),
        "compressor_arch": args.compressor_arch,
        "baseline_root": str(baseline_root),
        "seed": int(args.seed),
        "seed_dir": str(paths["seed_dir"]),
        "cache_file": str(paths["cache_path"]),
        "cache_meta_file": str(paths["cache_meta_path"]),
        "cache_split": args.cache_split,
        "flow_params_path": str(paths["params_path"]),
        "flow_summary_path": str(paths["summary_path"]),
        "flow_summary": flow_summary,
        "standardization_path": str(paths["std_path"]),
        "standardization_mean": std_mean.tolist(),
        "standardization_std": std_std.tolist(),
        "standardization_clip_value": clip_value,
        "nvp_layers": int(args.nvp_layers),
        "nvp_hidden": int(args.nvp_hidden),
        "n_rank": int(args.n_ranks),
        "posterior_samples": int(args.posterior_samples),
        "rank_seed": int(args.rank_seed),
        "nonfid_eps": float(args.nonfid_eps),
        "parameter_order": PARAMETER_ORDER,
        "fiducial_theta": FIDUCIAL_THETA.tolist(),
        "gpu_policy": {
            "cuda_visible_devices_arg": args.cuda_visible_devices,
            "xla_mem_fraction_arg": args.xla_mem_fraction,
            "cuda_visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "xla_python_client_mem_fraction_env": os.environ.get(
                "XLA_PYTHON_CLIENT_MEM_FRACTION"
            ),
        },
        "outputs": {
            "rank_npz": str(rank_npz),
            "metrics_json": str(metrics_path),
            "rank_hist_png": str(plot_path),
            "posterior_samples_npz": str(samples_npz) if samples_npz is not None else None,
        },
    }
    (run_dir / "repro_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    latest_manifest = args.output_root.resolve() / args.compressor_arch / "latest_run.json"
    latest_manifest.parent.mkdir(parents=True, exist_ok=True)
    latest_manifest.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "seed": int(args.seed),
                "n_rank": int(args.n_ranks),
                "posterior_samples": int(args.posterior_samples),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[SBC] Done. Outputs in {run_dir}")


if __name__ == "__main__":
    main()
