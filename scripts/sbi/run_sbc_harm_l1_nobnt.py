#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PARAMETER_ORDER = ["Omega_m", "sigma_8", "w0", "h0", "n_s", "Omega_b"]
FIDUCIAL_THETA = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_baseline = (
        repo_root
        / "scripts"
        / "sbi"
        / "results"
        / "exploratory"
        / "cross_maps_campaign"
        / "jaxili_harm_cross_nobnt"
    )
    default_seed_root = default_baseline / "seed_41" / "l1norm_cross_jaxili" / "nbody"
    default_output = (
        repo_root
        / "scripts"
        / "sbi"
        / "results"
        / "diagnostics"
        / "sbc_harm_l1_nobnt"
    )

    p = argparse.ArgumentParser(
        description=(
            "Run SBC rank-statistics for harmonic-L1 no-BNT using cached L1 summaries "
            "and a trained jaxili checkpoint."
        )
    )
    p.add_argument("--baseline-root", type=Path, default=default_baseline)
    p.add_argument("--cache-dir", type=Path, default=default_baseline / "l1_cache_seed41")
    p.add_argument("--checkpoint-root", type=Path, default=default_seed_root / "params_l1norm_cross_jaxili")
    p.add_argument(
        "--preprocessing-stats",
        type=Path,
        default=default_seed_root / "l1_cross_jaxili_standardization.npz",
    )
    p.add_argument(
        "--feature-mask",
        type=Path,
        default=default_seed_root / "l1_cross_jaxili_feature_mask.npz",
    )
    p.add_argument("--output-root", type=Path, default=default_output)
    p.add_argument("--cache-split", type=str, default="val", choices=("train", "val"))
    p.add_argument("--n-ranks", type=int, required=True)
    p.add_argument("--posterior-samples", type=int, default=2000)
    p.add_argument("--rank-seed", type=int, default=12345)
    p.add_argument("--rank-bins", type=int, default=20)
    p.add_argument("--nonfid-eps", type=float, default=1e-8)
    p.add_argument("--max-attempts", type=int, default=8)
    p.add_argument("--cuda-visible-devices", type=str, default=None)
    p.add_argument("--xla-mem-fraction", type=float, default=None)
    p.add_argument(
        "--dump-posterior-samples",
        action="store_true",
        help=(
            "If set, write posterior_samples.npz alongside sbc_ranks.npz containing "
            "the full (N, M, 6) posterior samples and theta arrays — needed for "
            "downstream TARP joint-coverage testing. Adds ~50MB at N=1000, M=2000."
        ),
    )
    return p.parse_args()


def configure_runtime(cuda_visible_devices: Optional[str], xla_mem_fraction: Optional[float]) -> None:
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    if xla_mem_fraction is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(float(xla_mem_fraction))


def _summary_transform_flags(summary_transform: str) -> tuple[bool, bool, str]:
    if summary_transform == "log1p-zscore":
        return True, True, "log1p"
    if summary_transform == "log10p-zscore":
        return True, True, "log10p"
    if summary_transform == "zscore":
        return False, True, "none"
    if summary_transform == "log1p":
        return True, False, "log1p"
    if summary_transform == "log10p":
        return True, False, "log10p"
    if summary_transform == "none":
        return False, False, "none"
    raise ValueError(
        f"Unknown summary transform '{summary_transform}'. "
        "Expected one of: log1p-zscore, log10p-zscore, zscore, log1p, log10p, none."
    )


def preprocess_summaries(
    x: np.ndarray,
    summary_transform: str,
    clip_value: Optional[float],
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
) -> np.ndarray:
    apply_log, apply_standardize, log_kind = _summary_transform_flags(summary_transform)
    x = np.asarray(x, dtype=np.float32)

    if apply_log:
        if np.any(x < -1.0):
            raise ValueError(
                f"Summaries contain values < -1, cannot apply {log_kind} safely "
                f"(minimum={x.min():.6e})."
            )
        if log_kind == "log1p":
            x = np.log1p(x)
        elif log_kind == "log10p":
            x = np.log10(x + 1.0)
        else:
            raise ValueError(f"Unexpected log kind: {log_kind}")

    if apply_standardize:
        if mean is None or std is None:
            raise ValueError("Saved mean/std are required for standardized no-train checkpoints.")
        mean = np.asarray(mean)
        std = np.asarray(std)
        if mean.shape != (x.shape[1],) or std.shape != (x.shape[1],):
            raise ValueError(
                "Loaded standardization stats have incompatible shape: "
                f"mean={mean.shape}, std={std.shape}, expected={(x.shape[1],)}."
            )
        std = std.copy()
        std[std < 1e-12] = 1.0
        x = (x - mean) / std

    if clip_value is not None and clip_value > 0:
        x = np.clip(x, -clip_value, clip_value)
    return x.astype(np.float32)


def apply_saved_pca(
    x: np.ndarray,
    pca_components: np.ndarray,
    pca_mean: np.ndarray,
    pca_explained_variance: np.ndarray,
) -> np.ndarray:
    pca_components = np.asarray(pca_components)
    pca_mean = np.asarray(pca_mean)
    pca_explained_variance = np.asarray(pca_explained_variance)
    if pca_components.ndim != 2:
        raise ValueError(f"pca_components must be 2D, got shape {pca_components.shape}.")
    if pca_mean.shape != (pca_components.shape[1],):
        raise ValueError(
            f"pca_mean shape {pca_mean.shape} incompatible with pca_components {pca_components.shape}."
        )
    if pca_explained_variance.shape != (pca_components.shape[0],):
        raise ValueError(
            f"pca_explained_variance shape {pca_explained_variance.shape} incompatible with "
            f"number of PCA components {pca_components.shape[0]}."
        )
    whitening = np.sqrt(np.maximum(pca_explained_variance, 1e-12))
    centered = x - pca_mean
    projected = centered @ pca_components.T
    return (projected / whitening).astype(np.float32)


def _resolve_latest_jaxili_checkpoint_dir(checkpoint_root: Path) -> Path:
    nde_root = checkpoint_root / "NDE_w_Standardization"
    if not nde_root.exists():
        raise FileNotFoundError(f"Missing jaxili checkpoint directory '{nde_root}'.")

    version_dirs = sorted(
        [p for p in nde_root.glob("version_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1,
        reverse=True,
    )
    if not version_dirs:
        raise FileNotFoundError(f"No version_* checkpoint folders found under '{nde_root}'.")

    for version_dir in version_dirs:
        has_hparams = (version_dir / "hparams.json").exists()
        has_numeric_ckpt = any(p.is_dir() and p.name.isdigit() for p in version_dir.iterdir())
        if has_hparams and has_numeric_ckpt:
            return version_dir

    raise FileNotFoundError(
        f"No completed jaxili checkpoints found under '{nde_root}'. "
        "Found only temporary/incomplete checkpoint directories."
    )


def _normalize_jaxili_hparams_embedding_arrays(version_dir: Path) -> None:
    hparams_path = version_dir / "hparams.json"
    if not hparams_path.exists():
        return
    raw = hparams_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    embedding = data.get("model_hparams", {}).get("embedding_net")
    if not isinstance(embedding, str):
        return

    normalized_embedding = embedding.replace("\n", " ")
    normalized_embedding = re.sub(r"\s+", " ", normalized_embedding)
    normalized_embedding = re.sub(r"\s*,\s*", ", ", normalized_embedding)
    if normalized_embedding != embedding:
        data["model_hparams"]["embedding_net"] = normalized_embedding
        hparams_path.write_text(json.dumps(data, indent=4), encoding="utf-8")


def validate_theta_order(theta: np.ndarray) -> None:
    if theta.ndim != 2 or theta.shape[1] != len(PARAMETER_ORDER):
        raise ValueError(f"theta must have shape (N, {len(PARAMETER_ORDER)}), got {theta.shape}.")
    if not np.isfinite(theta).all():
        raise ValueError("theta contains non-finite values.")
    h0_median = float(np.median(theta[:, 3]))
    if h0_median > 2.0:
        raise ValueError(
            "theta[:,3] appears to be H0 (km/s/Mpc), expected h0=H0/100. "
            "Cannot enforce required parameter order [Omega_m, sigma_8, w0, h0, n_s, Omega_b]."
        )


def draw_posterior_samples(
    posterior,
    summary_obs: np.ndarray,
    n_samples: int,
    rng_key,
    jax,
    jnp,
    max_attempts: int = 8,
) -> Tuple[np.ndarray, object, int]:
    summary_obs = jnp.asarray(np.asarray(summary_obs, dtype=np.float32))
    gathered: List[np.ndarray] = []
    total = 0
    attempts = 0
    key = rng_key
    while total < n_samples and attempts < max_attempts:
        need = n_samples - total
        key, subkey = jax.random.split(key)
        draw = np.asarray(posterior.sample(x=summary_obs, num_samples=need, key=subkey))
        if draw.ndim == 1:
            draw = draw[None, :]
        elif draw.ndim > 2:
            draw = draw.reshape(draw.shape[0], -1)
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

    out = np.concatenate(gathered, axis=0)[:n_samples]
    if out.shape[1] != len(PARAMETER_ORDER):
        raise ValueError(
            f"Posterior samples have dim={out.shape[1]}, expected {len(PARAMETER_ORDER)}."
        )
    return out, key, attempts


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
                "mean_rank_z": float((r.mean() - expected_mean) / np.sqrt(expected_var / n_rank)),
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
) -> None:
    n_rank = ranks.shape[0]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    expected = n_rank / float(n_bins)
    for j, ax in enumerate(axes.flat):
        ax.hist(ranks[:, j], bins=n_bins, range=(0, n_post + 1), color="#4C78A8", alpha=0.9)
        ax.axhline(expected, color="#E45756", linestyle="--", linewidth=1.2)
        ax.set_title(PARAMETER_ORDER[j])
        ax.set_xlabel("Rank")
        ax.set_ylabel("Count")
    fig.suptitle(f"Harmonic-L1 no-BNT SBC rank histograms (N={n_rank}, M={n_post})")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.n_ranks <= 0:
        raise ValueError("--n-ranks must be > 0")
    if args.posterior_samples <= 0:
        raise ValueError("--posterior-samples must be > 0")
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be > 0")

    configure_runtime(args.cuda_visible_devices, args.xla_mem_fraction)

    import jax
    import jax.numpy as jnp

    try:
        from jaxili.inference import NPE
    except ImportError as exc:  # pragma: no cover - runtime env dependent
        raise ImportError(
            "Failed to import jaxili. Activate the conda environment 'jaxili' before running."
        ) from exc

    baseline_root = args.baseline_root.resolve()
    cache_dir = args.cache_dir.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    preprocessing_stats = args.preprocessing_stats.resolve()
    feature_mask_path = args.feature_mask.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_tag = f"n{args.n_ranks}_m{args.posterior_samples}_seed{args.rank_seed}"
    run_dir = output_root / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / ("l1_val.npz" if args.cache_split == "val" else "l1_train.npz")
    if not cache_file.exists():
        raise FileNotFoundError(f"Missing cache file: {cache_file}")
    if not preprocessing_stats.exists():
        raise FileNotFoundError(f"Missing preprocessing stats: {preprocessing_stats}")
    if not feature_mask_path.exists():
        raise FileNotFoundError(f"Missing feature mask: {feature_mask_path}")

    cache_npz = np.load(cache_file, mmap_mode="r")
    theta = np.asarray(cache_npz["theta"], dtype=np.float32)
    validate_theta_order(theta)

    nonfid_mask = np.max(np.abs(theta - FIDUCIAL_THETA[None, :]), axis=1) > args.nonfid_eps
    valid_indices = np.where(nonfid_mask)[0]
    if len(valid_indices) < args.n_ranks:
        raise ValueError(
            f"Only {len(valid_indices)} non-fiducial rows available, need {args.n_ranks}."
        )

    rng = np.random.default_rng(args.rank_seed)
    selected_indices = rng.choice(valid_indices, size=args.n_ranks, replace=False)
    selected_theta = theta[selected_indices]
    selected_raw_x = np.asarray(cache_npz["x"][selected_indices], dtype=np.float32)

    with np.load(preprocessing_stats, allow_pickle=False) as saved:
        stats = {k: np.array(saved[k]) for k in saved.files}

    summary_transform = str(stats["summary_transform"]) if "summary_transform" in stats else "log1p-zscore"
    clip_value = None
    if "clip_value" in stats:
        clip_raw = float(stats["clip_value"])
        clip_value = None if np.isnan(clip_raw) else clip_raw

    processed_x = preprocess_summaries(
        x=selected_raw_x,
        summary_transform=summary_transform,
        clip_value=clip_value,
        mean=stats.get("mean"),
        std=stats.get("std"),
    )

    pca_applied = False
    pca_source = "none"
    if {"pca_components", "pca_mean", "pca_explained_variance"}.issubset(stats.keys()):
        processed_x = apply_saved_pca(
            processed_x,
            pca_components=stats["pca_components"],
            pca_mean=stats["pca_mean"],
            pca_explained_variance=stats["pca_explained_variance"],
        )
        pca_applied = True
        pca_source = str(preprocessing_stats)

    with np.load(feature_mask_path, allow_pickle=False) as saved_mask:
        valid_mask = np.asarray(saved_mask["valid_mask"], dtype=bool)
    if valid_mask.ndim != 1:
        raise ValueError(f"Saved valid_mask must be 1D, got {valid_mask.shape}.")
    if valid_mask.shape[0] != processed_x.shape[1]:
        raise ValueError(
            "Saved valid_mask dimension does not match processed summary dim "
            f"({valid_mask.shape[0]} vs {processed_x.shape[1]})."
        )
    selected_summaries = processed_x[:, valid_mask].astype(np.float32)

    checkpoint_dir = _resolve_latest_jaxili_checkpoint_dir(checkpoint_root)
    _normalize_jaxili_hparams_embedding_arrays(checkpoint_dir)

    exmp_input = (
        jnp.zeros((1, len(PARAMETER_ORDER)), dtype=jnp.float32),
        jnp.zeros((1, selected_summaries.shape[1]), dtype=jnp.float32),
    )
    inference = NPE.load_from_checkpoints(
        checkpoint=str(checkpoint_dir),
        exmp_input=exmp_input,
    )
    posterior = inference.build_posterior()

    print(
        f"[SBC] Split={args.cache_split}, selected non-fid rows={args.n_ranks}, "
        f"posterior samples per row={args.posterior_samples}"
    )
    print(f"[SBC] Checkpoint={checkpoint_dir}")
    print(
        f"[SBC] Preprocess={summary_transform}, clip={clip_value}, "
        f"mask_kept={int(valid_mask.sum())}"
    )

    ranks = np.zeros((args.n_ranks, len(PARAMETER_ORDER)), dtype=np.int32)
    attempts_used = np.zeros(args.n_ranks, dtype=np.int32)
    key = jax.random.PRNGKey(args.rank_seed + 2026)

    samples_dump = (
        np.empty((args.n_ranks, args.posterior_samples, len(PARAMETER_ORDER)), dtype=np.float32)
        if args.dump_posterior_samples
        else None
    )

    for i in range(args.n_ranks):
        draws, key, attempts = draw_posterior_samples(
            posterior=posterior,
            summary_obs=selected_summaries[i],
            n_samples=args.posterior_samples,
            rng_key=key,
            jax=jax,
            jnp=jnp,
            max_attempts=args.max_attempts,
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

    metrics = rank_metrics(ranks=ranks, n_post=args.posterior_samples, n_bins=args.rank_bins)
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
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "cwd": str(Path.cwd().resolve()),
        "baseline_root": str(baseline_root),
        "cache_dir": str(cache_dir),
        "cache_file": str(cache_file),
        "cache_split": args.cache_split,
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_dir": str(checkpoint_dir),
        "preprocessing_stats": str(preprocessing_stats),
        "feature_mask": str(feature_mask_path),
        "summary_transform": summary_transform,
        "summary_clip_value": clip_value,
        "pca_applied": bool(pca_applied),
        "pca_source": pca_source,
        "summary_dim_after_mask": int(selected_summaries.shape[1]),
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

    latest_manifest = output_root / "latest_run.json"
    latest_manifest.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
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
