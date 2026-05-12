#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import tensorflow_probability as tfp_root
from tensorflow_probability.substrates import jax as tfp_jax

# NumPy 2.x compatibility for older TFP/JAX substrate code paths.
if not hasattr(np, "issctype"):
    def _np_issctype(rep) -> bool:
        try:
            return issubclass(np.dtype(rep).type, np.generic)
        except Exception:
            return False
    np.issctype = _np_issctype  # type: ignore[attr-defined]

# Some installed sbi_lens versions expect tensorflow_probability.substrates.jax
# via tensorflow_probability.substrates.
if not hasattr(tfp_root, "substrates"):
    class _TFPSubstrates:
        jax = tfp_jax

    tfp_root.substrates = _TFPSubstrates()  # type: ignore[attr-defined]
from sbi_lens.normflow.models import AffineCoupling, ConditionalRealNVP

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PARAMETER_ORDER = ["Omega_m", "sigma_8", "w0", "h0", "n_s", "Omega_b"]
FIDUCIAL_THETA = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493], dtype=np.float32)


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
    default_baseline = (
        repo_root
        / "scripts"
        / "sbi"
        / "results"
        / "exploratory"
        / "zero_mean_maps_parity_check"
        / "run_b_advanced_plain"
    )
    default_output = (
        repo_root
        / "scripts"
        / "sbi"
        / "results"
        / "diagnostics"
        / "sbc_cnn_nobnt"
    )

    p = argparse.ArgumentParser(
        description=(
            "Run SBC rank-statistics for CNN no-BNT using cached summaries and a "
            "trained no-BNT flow checkpoint from run_b_advanced_plain."
        )
    )
    p.add_argument("--baseline-root", type=Path, default=default_baseline)
    p.add_argument("--output-root", type=Path, default=default_output)
    p.add_argument("--condition", type=str, default="nobnt")
    p.add_argument("--cache-split", type=str, default="val", choices=("train", "val"))
    p.add_argument("--n-ranks", type=int, required=True)
    p.add_argument("--posterior-samples", type=int, default=2000)
    p.add_argument("--rank-seed", type=int, default=12345)
    p.add_argument("--flow-seed", type=int, default=None)
    p.add_argument("--flow-select", type=str, default="best_val_loss", choices=("best_val_loss", "first"))
    p.add_argument("--nvp-layers", type=int, default=8)
    p.add_argument("--nvp-hidden", type=int, default=256)
    p.add_argument("--rank-bins", type=int, default=20)
    p.add_argument("--nonfid-eps", type=float, default=1e-8)
    p.add_argument("--cuda-visible-devices", type=str, default=None)
    p.add_argument("--xla-mem-fraction", type=float, default=None)
    p.add_argument(
        "--dump-posterior-samples",
        action="store_true",
        help=(
            "If set, write posterior_samples.npz alongside sbc_ranks.npz containing "
            "the full (N, M, 6) posterior samples and theta arrays — needed for "
            "downstream TARP joint-coverage testing."
        ),
    )
    return p.parse_args()


def _flow_seed_from_path(seed_dir: Path) -> int:
    return int(seed_dir.name.split("_", 1)[1])


def select_flow_checkpoint(
    baseline_root: Path,
    condition: str,
    requested_seed: int | None,
    strategy: str,
) -> Dict[str, object]:
    eval_root = baseline_root / "eval" / condition
    if not eval_root.exists():
        raise FileNotFoundError(f"Missing eval root: {eval_root}")

    seed_dirs = sorted([p for p in eval_root.glob("seed_*") if p.is_dir()], key=_flow_seed_from_path)
    if not seed_dirs:
        raise FileNotFoundError(f"No seed directories under {eval_root}")

    candidates: List[Dict[str, object]] = []
    for seed_dir in seed_dirs:
        seed = _flow_seed_from_path(seed_dir)
        flow_dir = seed_dir / "cnn_vmim" / "nbody"
        params_path = flow_dir / "params_cnn_flow_best.pkl"
        summary_path = flow_dir / "flow_training_summary.json"
        if not params_path.exists() or not summary_path.exists():
            continue
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        candidates.append(
            {
                "seed": seed,
                "params_path": params_path,
                "summary_path": summary_path,
                "summary": summary,
                "best_val_loss": float(summary.get("best_val_loss", np.inf)),
                "best_step": int(summary.get("best_step", -1)),
            }
        )
    if not candidates:
        raise FileNotFoundError(f"No valid flow checkpoints under {eval_root}")

    if requested_seed is not None:
        matches = [c for c in candidates if int(c["seed"]) == int(requested_seed)]
        if not matches:
            raise FileNotFoundError(f"Requested --flow-seed={requested_seed} not found in {eval_root}")
        return matches[0]

    if strategy == "first":
        return sorted(candidates, key=lambda c: int(c["seed"]))[0]
    return min(candidates, key=lambda c: float(c["best_val_loss"]))


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
    fig.suptitle(f"CNN no-BNT SBC rank histograms (N={n_rank}, M={n_post})")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.n_ranks <= 0:
        raise ValueError("--n-ranks must be > 0")
    if args.posterior_samples <= 0:
        raise ValueError("--posterior-samples must be > 0")

    baseline_root = args.baseline_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_tag = f"n{args.n_ranks}_m{args.posterior_samples}_seed{args.rank_seed}"
    run_dir = output_root / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    cache_name = "cnn_val.npz" if args.cache_split == "val" else "cnn_train.npz"
    cache_path = baseline_root / "cache" / f"{args.condition}_zeromean_eval" / cache_name
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cache file: {cache_path}")
    cache = np.load(cache_path)
    theta = np.asarray(cache["theta"], dtype=np.float32)
    summaries = np.asarray(cache["x"], dtype=np.float32)

    flow_info = select_flow_checkpoint(
        baseline_root=baseline_root,
        condition=args.condition,
        requested_seed=args.flow_seed,
        strategy=args.flow_select,
    )
    params_path = Path(flow_info["params_path"])
    with open(params_path, "rb") as f:
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
        f"[SBC] Using flow seed {flow_info['seed']} "
        f"(best_val_loss={flow_info['best_val_loss']:.6f}, best_step={flow_info['best_step']})"
    )
    print(
        f"[SBC] Split={args.cache_split}, selected non-fid rows={args.n_ranks}, "
        f"posterior samples per row={args.posterior_samples}"
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
        np.empty((args.n_ranks, args.posterior_samples, selected_theta.shape[1]), dtype=np.float32)
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

    samples_npz: Path | None = None
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
        "condition": args.condition,
        "cache_file": str(cache_path),
        "cache_split": args.cache_split,
        "flow_seed": int(flow_info["seed"]),
        "flow_select_strategy": args.flow_select,
        "flow_params_path": str(params_path),
        "flow_training_summary_path": str(flow_info["summary_path"]),
        "flow_training_summary": flow_info["summary"],
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
