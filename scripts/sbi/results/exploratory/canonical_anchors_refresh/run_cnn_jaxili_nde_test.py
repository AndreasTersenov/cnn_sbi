#!/usr/bin/env python
"""
NDE-swap test: take the canonical CNN compressor's compressed summaries
and train a jaxili NPE (MAF) instead of the in-repo RealNVP.

Isolates the NDE architecture as the variable:
  - Same compressor (plain CNN, cdim=10, best_val checkpoint)
  - Same compressed data (from --exit-after-compress cache)
  - Same observed summary
  - Different density estimator: jaxili ConditionalMAF vs sbi_lens RealNVP

Usage:
  # Phase 1: compress (re-uses existing compressor, ~12 min on GPU 1)
  conda run -n jaxili python scripts/sbi/npe_cnn_nbody_tomo.py \
      --exit-after-compress --cache-dir <out>/cache_s41 \
      [all canonical CNN cross flags] --seed 41

  # Phase 2: jaxili NDE + diagnostics (~5-15 min on GPU 1)
  conda run -n jaxili python scripts/sbi/results/exploratory/canonical_anchors_refresh/run_cnn_jaxili_nde_test.py \
      --cache-dir <out>/cache_s41 --seed 41 --out-dir <out>/jaxili_nde_test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

try:
    from jaxili.inference import NPE
except ImportError as exc:
    raise ImportError("jaxili not available in this environment.") from exc


def parse_args():
    p = argparse.ArgumentParser(
        description="jaxili NPE on CNN-compressed summaries (NDE-swap test)"
    )
    p.add_argument("--cache-dir", required=True,
                    help="Dir with cnn_train.npz, cnn_val.npz from --exit-after-compress")
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=128)
    p.add_argument("--decay-steps", type=int, default=10000)
    p.add_argument("--npe-samples", type=int, default=100000)
    p.add_argument("--maf-hidden", type=str, default="50,50",
                    help="Comma-separated hidden layer widths for jaxili MAF MADE layers")
    p.add_argument("--maf-layers", type=int, default=5,
                    help="Number of MAF MADE layers")
    p.add_argument("--patience", type=int, default=20,
                    help="Early stopping patience (epochs without improvement)")
    p.add_argument("--min-delta", type=float, default=1e-3,
                    help="Minimum val loss improvement to reset patience")
    p.add_argument("--standardize-summary", action="store_true", default=True,
                    help="Apply z-score + clip to CNN summaries before jaxili (matches canonical)")
    p.add_argument("--clip-value", type=float, default=5.0)
    p.add_argument("--cuda-visible-devices", type=str, default="1")
    return p.parse_args()


def load_compressed_cache(cache_dir: Path):
    train_f = cache_dir / "cnn_train.npz"
    val_f = cache_dir / "cnn_val.npz"
    if not train_f.exists() or not val_f.exists():
        raise FileNotFoundError(
            f"Missing {train_f} or {val_f}. Run Phase 1 (--exit-after-compress) first."
        )
    d_tr = np.load(train_f)
    d_va = np.load(val_f)
    print(f"Loaded compressed cache from {cache_dir}")
    print(f"  Train: theta={d_tr['theta'].shape}, x={d_tr['x'].shape}")
    print(f"  Val:   theta={d_va['theta'].shape}, x={d_va['x'].shape}")
    return (
        {"theta": d_tr["theta"], "x": d_tr["x"]},
        {"theta": d_va["theta"], "x": d_va["x"]},
    )


def standardize_summaries(train_x, val_x, obs_x, clip_value=5.0):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-8] = 1.0
    train_s = np.clip((train_x - mean) / std, -clip_value, clip_value)
    val_s = np.clip((val_x - mean) / std, -clip_value, clip_value)
    obs_s = np.clip((obs_x - mean) / std, -clip_value, clip_value)
    print(f"  Summary standardization: mean range [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  Summary standardization: std  range [{std.min():.4f}, {std.max():.4f}]")
    return train_s, val_s, obs_s, mean, std


def compute_fom3(samples):
    cov3 = np.cov(samples[:, :3], rowvar=False)
    sign, logdet = np.linalg.slogdet(cov3)
    if sign <= 0:
        return float("nan")
    return float(np.exp(-0.5 * logdet))


def make_diagnostic_plots(out_dir, train_theta, train_x, val_theta, val_x,
                          obs_x, samples, truth, metrics, std_mean, std_std,
                          reference_fom3=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: Training curves ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if hasattr(metrics, "train_loss"):
        train_loss = np.asarray(metrics.train_loss)
        ax = axes[0]
        ax.plot(train_loss, lw=0.5, alpha=0.7, label="train loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Neg log-prob")
        ax.set_title("Train loss")
        ax.legend()

    if hasattr(metrics, "val_loss"):
        val_loss = np.asarray(metrics.val_loss)
        ax = axes[1]
        ax.plot(val_loss, lw=1.0, color="C1", label="val loss")
        best_epoch = int(np.nanargmin(val_loss))
        best_val = float(np.nanmin(val_loss))
        ax.axhline(best_val, ls="--", color="gray", lw=0.8,
                    label=f"best={best_val:.3f} @ep{best_epoch}")
        ax.axvline(best_epoch, ls=":", color="gray", lw=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Neg log-prob")
        ax.set_title("Validation loss")
        ax.legend()

        # Also plot train + val together
        ax = axes[2]
        if hasattr(metrics, "train_loss"):
            ax.plot(np.asarray(metrics.train_loss), lw=0.5, alpha=0.6, label="train")
        ax.plot(val_loss, lw=1.0, color="C1", label="val")
        ax.axhline(best_val, ls="--", color="gray", lw=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Neg log-prob")
        ax.set_title("Train vs Val (overfitting check)")
        ax.legend()

    fig.suptitle("jaxili MAF training curves (CNN compressed summaries)", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / "01_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: Compressed summary distributions ----
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    summary_dim = train_x.shape[1]
    for i in range(min(summary_dim, 10)):
        ax = axes[i // 5, i % 5]
        ax.hist(train_x[:, i], bins=80, alpha=0.5, density=True, label="train")
        ax.hist(val_x[:, i], bins=80, alpha=0.5, density=True, label="val")
        ax.axvline(obs_x[i], color="red", lw=2, label=f"obs={obs_x[i]:.3f}")
        ax.set_title(f"cdim {i}")
        if i == 0:
            ax.legend(fontsize=7)
    for i in range(summary_dim, 10):
        axes[i // 5, i % 5].set_visible(False)
    fig.suptitle("CNN compressed summary distributions (post-standardization)", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / "02_summary_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 3: Summary vs theta correlations ----
    param_names = [r"$\Omega_m$", r"$\sigma_8$", r"$w_0$", r"$h_0$", r"$n_s$", r"$\Omega_b$"]
    fig, axes = plt.subplots(3, min(summary_dim, 10), figsize=(3 * min(summary_dim, 10), 9))
    if axes.ndim == 1:
        axes = axes.reshape(-1, 1)
    for j in range(min(summary_dim, 10)):
        for p_idx in range(3):
            ax = axes[p_idx, j]
            subsample = np.random.choice(len(train_theta), min(5000, len(train_theta)), replace=False)
            ax.scatter(train_x[subsample, j], train_theta[subsample, p_idx],
                       s=0.5, alpha=0.3, rasterized=True)
            ax.set_xlabel(f"cdim {j}" if p_idx == 2 else "")
            ax.set_ylabel(param_names[p_idx] if j == 0 else "")
    fig.suptitle("Summary-parameter correlations (top 3 params × all cdims)", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / "03_summary_theta_correlations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 4: Corner plot ----
    try:
        from getdist import MCSamples, plots as gplot
        param_labels = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]
        mcsamples = MCSamples(samples=samples, names=param_labels, labels=param_labels)
        g = gplot.get_subplot_plotter(subplot_size=1.5)
        g.triangle_plot([mcsamples], filled=True,
                        markers=truth,
                        marker_args={"color": "red", "lw": 1.2})
        plt.savefig(out_dir / "04_corner_plot.png", dpi=150, bbox_inches="tight")
        plt.close()
    except ImportError:
        print("  getdist not available, skipping corner plot")

    # ---- Figure 5: Per-parameter bias & FoM3 summary ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    param_labels_short = [r"$\Omega_m$", r"$\sigma_8$", r"$w_0$", r"$h_0$", r"$n_s$", r"$\Omega_b$"]

    ax = axes[0]
    means = samples.mean(axis=0)
    stds = samples.std(axis=0)
    biases_sigma = (means - truth) / stds
    colors = ["C2" if abs(b) < 1 else ("C1" if abs(b) < 2 else "C3") for b in biases_sigma]
    ax.barh(range(6), biases_sigma, color=colors)
    ax.set_yticks(range(6))
    ax.set_yticklabels(param_labels_short)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(-1, color="gray", ls="--", lw=0.8)
    ax.axvline(1, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Bias / posterior σ")
    ax.set_title("Per-parameter bias")

    ax = axes[1]
    fom3_val = compute_fom3(samples)
    bar_labels = ["jaxili MAF\n(this run)"]
    bar_values = [fom3_val]
    bar_colors = ["C0"]
    if reference_fom3 is not None:
        for label, val in reference_fom3.items():
            bar_labels.append(label)
            bar_values.append(val)
            bar_colors.append("C1" if "RealNVP" in label else "C2")
    ax.bar(range(len(bar_values)), bar_values, color=bar_colors)
    ax.set_xticks(range(len(bar_values)))
    ax.set_xticklabels(bar_labels, fontsize=9)
    ax.set_ylabel("FoM3")
    ax.set_title("FoM3 comparison")
    for i, v in enumerate(bar_values):
        ax.text(i, v + max(bar_values) * 0.02, f"{v:.0f}", ha="center", fontsize=10)

    fig.suptitle(f"jaxili MAF NDE test — seed {samples.shape}", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / "05_bias_and_fom3.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 6: Posterior 1D marginals with truth ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, ax in enumerate(axes.flat):
        ax.hist(samples[:, i], bins=80, density=True, alpha=0.7, color="C0")
        ax.axvline(truth[i], color="red", lw=2, label=f"truth={truth[i]:.4f}")
        ax.axvline(means[i], color="blue", ls="--", lw=1.5, label=f"mean={means[i]:.4f}")
        ax.set_xlabel(param_labels_short[i])
        ax.set_title(f"{param_labels_short[i]}: bias={biases_sigma[i]:.2f}σ")
        ax.legend(fontsize=8)
    fig.suptitle("1D marginal posteriors", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / "06_1d_marginals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved 6 diagnostic figures to {out_dir}/")
    return [str(out_dir / f) for f in [
        "01_training_curves.png", "02_summary_distributions.png",
        "03_summary_theta_correlations.png", "04_corner_plot.png",
        "05_bias_and_fom3.png", "06_1d_marginals.png",
    ]]


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    print(f"CUDA_VISIBLE_DEVICES = {args.cuda_visible_devices}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    # Load compressed summaries
    dataset_train, dataset_val = load_compressed_cache(cache_dir)
    train_theta = dataset_train["theta"].astype(np.float32)
    train_x = dataset_train["x"].astype(np.float32)
    val_theta = dataset_val["theta"].astype(np.float32)
    val_x = dataset_val["x"].astype(np.float32)

    # Load observed summary from the cache meta or compute it
    meta_f = cache_dir / "cnn_cache_meta.npz"
    obs_f = cache_dir / "cnn_obs.npz"
    if obs_f.exists():
        obs_data = np.load(obs_f)
        obs_x = obs_data["x"].astype(np.float32)
        truth = obs_data["theta"].astype(np.float64)
    else:
        raise FileNotFoundError(
            f"Missing {obs_f}. The compression phase must save the observed summary."
        )

    print(f"  Observed summary shape: {obs_x.shape}")
    print(f"  Truth: {truth}")

    # Standardize (same as canonical CNN pipeline)
    std_mean, std_std = None, None
    if args.standardize_summary:
        print("######## SUMMARY STANDARDIZATION ########")
        train_x, val_x, obs_x, std_mean, std_std = standardize_summaries(
            train_x, val_x, obs_x, clip_value=args.clip_value,
        )

    # Combine train + val for jaxili (it does its own internal split)
    all_theta = np.concatenate([train_theta, val_theta], axis=0)
    all_x = np.concatenate([train_x, val_x], axis=0)
    print(f"  Combined data for jaxili: theta={all_theta.shape}, x={all_x.shape}")

    # Initialize jaxili NPE with custom MAF hyperparameters
    maf_hidden = [int(x) for x in args.maf_hidden.split(",")]
    maf_hparams = {
        "n_layers": args.maf_layers,
        "layers": maf_hidden,
        "activation": jax.nn.relu,
        "use_reverse": True,
        "seed": args.seed,
    }
    print(f"  MAF config: {args.maf_layers} layers, hidden={maf_hidden}")
    print(f"  Patience: {args.patience}, min_delta: {args.min_delta}")

    split_key = jax.random.PRNGKey(args.seed + 1)
    inference = NPE(model_hparams=maf_hparams)
    inference = inference.append_simulations(
        jnp.array(all_theta), jnp.array(all_x), key=split_key
    )

    # Train
    print("######## JAXILI NPE TRAINING ########")
    t0 = time.time()
    max_retries = 10
    for attempt in range(1, max_retries + 1):
        print(f"  Attempt {attempt}/{max_retries}")
        try:
            metrics, density_estimator = inference.train(
                checkpoint_path=str(out_dir / "jaxili_checkpoint"),
                num_epochs=args.epochs,
                learning_rate=args.learning_rate,
                training_batch_size=args.batch_size,
                patience=args.patience,
                warmup=args.warmup_steps,
                decay_steps=args.decay_steps,
                min_delta=args.min_delta,
            )
            break
        except Exception as exc:
            print(f"  Training failed: {exc}")
            if attempt == max_retries:
                raise
            inference = NPE(model_hparams=maf_hparams)
            inference = inference.append_simulations(
                jnp.array(all_theta), jnp.array(all_x), key=split_key
            )

    elapsed = time.time() - t0
    print(f"  Training completed in {elapsed:.1f}s")

    # Training summary — jaxili metrics object varies by version
    train_loss_raw = getattr(metrics, "train_loss", None)
    val_loss_raw = getattr(metrics, "val_loss", None)
    train_loss = np.asarray(train_loss_raw) if train_loss_raw is not None else np.array([])
    val_loss = np.asarray(val_loss_raw) if val_loss_raw is not None else np.array([])
    if train_loss.ndim == 0:
        train_loss = train_loss.reshape(1)
    if val_loss.ndim == 0:
        val_loss = val_loss.reshape(1)
    finite_val = val_loss[np.isfinite(val_loss)]
    best_epoch = int(np.nanargmin(val_loss)) if len(finite_val) > 0 else -1
    best_val = float(np.nanmin(val_loss)) if len(finite_val) > 0 else float("nan")
    final_train = float(train_loss[-1]) if len(train_loss) > 0 else float("nan")
    print(f"  Best val loss: {best_val:.4f} at epoch {best_epoch}")
    print(f"  Final train loss: {final_train:.4f}")
    print(f"  Total epochs: {len(val_loss)}")
    print(f"  Metrics attributes: {[a for a in dir(metrics) if not a.startswith('_')]}")

    # Sample posterior
    print("######## SAMPLING POSTERIOR ########")
    posterior = inference.build_posterior(density_estimator)
    obs_jnp = jnp.array(obs_x.reshape(1, -1))
    samples = np.array(posterior.sample(
        args.npe_samples,
        jax.random.PRNGKey(args.seed + 42),
        x=obs_jnp,
    ))
    nan_mask = np.any(np.isnan(samples), axis=1)
    n_nan = nan_mask.sum()
    if n_nan > 0:
        print(f"  Removed {n_nan} NaN samples")
        samples = samples[~nan_mask]
    print(f"  Generated {len(samples)} valid posterior samples")

    # Compute FoM3
    fom3 = compute_fom3(samples)
    print(f"  FoM3 = {fom3:.1f}")

    # Reference FoM3 values for comparison
    reference_fom3 = {
        "CNN RealNVP\n(canonical)": 12615,
        "CNN RealNVP\n(iter-108 train/train)": 23986,
        "L1 jaxili MAF\n(canonical)": 34004,
    }

    # Save
    np.save(out_dir / "posterior_cnn_jaxili_test.npy", samples)
    meta = {
        "method": "cnn_jaxili_nde_swap_test",
        "seed": args.seed,
        "nde_architecture": "jaxili ConditionalMAF",
        "nde_epochs_requested": args.epochs,
        "nde_epochs_used": len(val_loss),
        "nde_best_epoch": best_epoch,
        "nde_best_val_loss": best_val,
        "nde_final_train_loss": final_train,
        "nde_batch_size": args.batch_size,
        "nde_learning_rate": args.learning_rate,
        "nde_training_data": int(all_theta.shape[0]),
        "nde_jaxili_internal_train": "70% of combined",
        "compressor": "canonical plain CNN cdim=10 (frozen, from canonical_anchors_refresh)",
        "summary_standardized": args.standardize_summary,
        "summary_clip_value": args.clip_value,
        "fom3": fom3,
        "n_samples": len(samples),
        "truth": truth.tolist(),
        "training_time_seconds": elapsed,
        "reference_fom3": reference_fom3,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved meta to {out_dir / 'meta.json'}")

    # Diagnostic plots
    print("######## DIAGNOSTIC PLOTS ########")
    plot_files = make_diagnostic_plots(
        out_dir, train_theta, train_x, val_theta, val_x,
        obs_x, samples, truth, metrics, std_mean, std_std,
        reference_fom3=reference_fom3,
    )

    # Final summary
    print()
    print("=" * 60)
    print("NDE-SWAP TEST RESULT")
    print("=" * 60)
    print(f"  FoM3 (jaxili MAF on CNN summaries):  {fom3:.0f}")
    print(f"  FoM3 (canonical RealNVP):            {reference_fom3['CNN RealNVP' + chr(10) + '(canonical)']}")
    print(f"  FoM3 (iter-108 RealNVP train/train): {reference_fom3['CNN RealNVP' + chr(10) + '(iter-108 train/train)']}")
    print(f"  FoM3 (L1 jaxili MAF canonical):      {reference_fom3['L1 jaxili MAF' + chr(10) + '(canonical)']}")
    print(f"  NDE epochs: {len(val_loss)} (best @{best_epoch})")
    print(f"  Best val loss: {best_val:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
