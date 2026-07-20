#!/usr/bin/env python
"""
Fast jaxili NPE on pre-cached CNN compressed summaries.

Bypasses the full CNN pipeline. Loads:
  - cnn_train.npz / cnn_val.npz (pre-compressed summaries + theta)
  - Compressor checkpoint (only to compress the obs map)
  - Standardization stats (to match the standardization applied during caching)

Then runs jaxili NPE with configurable batch size and MAF hyperparameters.
"""
from __future__ import annotations
import argparse, json, os, pickle, sys, time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def parse_args():
    p = argparse.ArgumentParser(description="Fast jaxili NPE on cached CNN summaries")
    p.add_argument("--train-npz", required=True, help="Path to cnn_train.npz")
    p.add_argument("--val-npz", required=True, help="Path to cnn_val.npz")
    p.add_argument("--obs-posterior-npy", required=True,
                    help="Path to an EXISTING posterior .npy from the same compressor+obs. "
                         "Used only to extract the truth vector for plotting.")
    p.add_argument("--obs-meta-json", required=True,
                    help="Path to the .meta.json for that posterior (has truth_parameters "
                         "and summary_standardization_file)")
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--label", type=str, default="cnn_jaxili",
                    help="Label for this arm (used in plot titles)")
    # jaxili config
    p.add_argument("--epochs", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--min-delta", type=float, default=0.0005)
    p.add_argument("--maf-hidden", type=str, default="50,50")
    p.add_argument("--maf-layers", type=int, default=5)
    p.add_argument("--npe-samples", type=int, default=100000)
    p.add_argument("--warmup-steps", type=int, default=128)
    p.add_argument("--decay-steps", type=int, default=10000)
    # Preprocessing
    p.add_argument("--standardize", action="store_true", default=True)
    p.add_argument("--standardization-npz", type=str, default=None,
                    help="Path to cnn_summary_standardization.npz (mean/std). "
                         "If not set, computes from training data.")
    p.add_argument("--clip-value", type=float, default=5.0)
    p.add_argument("--cuda-visible-devices", type=str, default="1")
    return p.parse_args()


def fom3(s):
    c = np.cov(s[:, :3], rowvar=False)
    sign, ld = np.linalg.slogdet(c)
    return np.exp(-0.5 * ld) if sign > 0 else float("nan")


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    from jaxili.inference import NPE

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    maf_hidden = [int(x) for x in args.maf_hidden.split(",")]

    # 1) Load pre-cached compressed data
    print("######## LOADING CACHED DATA ########")
    d_tr = np.load(args.train_npz)
    d_va = np.load(args.val_npz)
    train_theta = d_tr["theta"].astype(np.float32)
    train_x = d_tr["x"].astype(np.float32)
    val_theta = d_va["theta"].astype(np.float32)
    val_x = d_va["x"].astype(np.float32)
    print(f"  Train: {train_x.shape}, Val: {val_x.shape}")

    # 2) Get truth and obs compressed summary from the existing posterior meta
    with open(args.obs_meta_json) as f:
        meta = json.load(f)

    truth_raw = meta.get("truth_parameters") or meta.get("truth")
    if truth_raw is None:
        truth = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493], dtype=np.float64)
        print(f"  No truth in meta — using CosmoGrid fiducial")
    else:
        truth = np.array(truth_raw, dtype=np.float64)
    print(f"  Truth: {truth}")

    # The obs compressed summary: extract from the standardization file or
    # from the summary_standardization_file + the posterior file
    std_file = args.standardization_npz or meta.get("summary_standardization_file")
    if std_file and Path(std_file).exists():
        std_data = np.load(std_file)
        std_mean = std_data["mean"].astype(np.float32)
        std_std = std_data["std"].astype(np.float32)
        print(f"  Loaded standardization from {std_file}")
        print(f"  mean range: [{std_mean.min():.4f}, {std_mean.max():.4f}]")
        print(f"  std range:  [{std_std.min():.4f}, {std_std.max():.4f}]")
    else:
        std_mean = train_x.mean(axis=0)
        std_std = train_x.std(axis=0)
        std_std[std_std < 1e-8] = 1.0
        print(f"  Computed standardization from training data")

    # We need the obs compressed summary. The original run compressed it in memory.
    # Reconstruct: the posterior was sampled conditioned on the obs summary.
    # The obs summary (pre-standardization) should be close to the posterior mean
    # mapped back through the flow. But we can't easily recover it.
    #
    # Alternative: compute it. We need the compressor params + the obs map.
    # The meta has the compressor path. Let me try loading and compressing.
    obs_x = None

    # Try to load from cnn_obs.npz if it exists alongside the cache
    cache_dir = Path(args.train_npz).parent
    obs_npz = cache_dir / "cnn_obs.npz"
    if obs_npz.exists():
        obs_data = np.load(obs_npz)
        obs_x = obs_data["x"].astype(np.float32)
        print(f"  Loaded obs summary from {obs_npz}")
    else:
        # Compute obs from compressor
        print("  No cnn_obs.npz found. Computing obs from compressor...")
        import haiku as hk

        # Build compressor (same architecture)
        comp_params_path = meta.get("compressor_params_path") or meta.get("compressor_params")
        comp_state_path = meta.get("compressor_state_path") or meta.get("compressor_state")
        if comp_params_path is None:
            raise ValueError("Cannot find compressor_params_path in meta")

        with open(comp_params_path, "rb") as f:
            comp_params = pickle.load(f)
        with open(comp_state_path, "rb") as f:
            comp_state = pickle.load(f)
        print(f"  Loaded compressor from {comp_params_path}")

        sys.path.insert(0, str(Path(args.train_npz).parents[4] / "scripts" / "sbi"))
        from npe_cnn_nbody_tomo import build_compressors, load_observed_from_harmonic_cache

        _, compressor_eval = build_compressors(
            output_dim=train_x.shape[1],
            arch="plain",
            conv_channels=(64, 128, 256),
            dense_width=256,
            pool_window=16,
            pool_stride=8,
        )

        # Load obs map
        cache_path = meta.get("full_sphere_cache_dir")
        regime = meta.get("harmonic_regime", "nobnt")
        if cache_path:
            from npe_cnn_nbody_tomo import load_observed_from_harmonic_cache
            m_data, _, _ = load_observed_from_harmonic_cache(
                cache_dir=Path(cache_path), regime=regime,
                meta_path=None,
                channel_scale=np.array(meta["harmonic_channel_scale"], dtype=np.float32) if "harmonic_channel_scale" in meta else None,
            )
        else:
            from npe_cnn_nbody_tomo import load_observed_map
            import healpy as hp
            m_data, _, _ = load_observed_map(
                meta.get("cosmogrid_meta", "/home/tersenov/CosmoGridV1/CosmoGridV1_metainfo.h5"),
                meta.get("fiducial_map", "/home/tersenov/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial/perm_0000/projected_probes_maps_nobaryons512.h5"),
                20, 160, 512, 4, (1, 2, 3, 4),
                0.26, 30/4,
                jax.random.PRNGKey(args.seed),
                zero_mean_maps=True,
            )

        n_ch = m_data.shape[-1]
        obs_raw, _ = compressor_eval.apply(
            comp_params, comp_state, None,
            m_data.reshape([1, 160, 160, n_ch]),
        )
        obs_x = np.array(obs_raw).squeeze().astype(np.float32)
        print(f"  Obs compressed: {obs_x}")

    # 3) Standardize
    if args.standardize:
        print("######## STANDARDIZATION ########")
        std_std_safe = std_std.copy()
        std_std_safe[std_std_safe < 1e-8] = 1.0
        train_x = np.clip((train_x - std_mean) / std_std_safe, -args.clip_value, args.clip_value)
        val_x = np.clip((val_x - std_mean) / std_std_safe, -args.clip_value, args.clip_value)
        obs_x = np.clip((obs_x - std_mean) / std_std_safe, -args.clip_value, args.clip_value)
        print(f"  Applied z-score + clip ±{args.clip_value}")

    # 4) Combine and feed to jaxili
    all_theta = np.concatenate([train_theta, val_theta], axis=0)
    all_x = np.concatenate([train_x, val_x], axis=0)
    print(f"  Combined: theta={all_theta.shape}, x={all_x.shape}")

    maf_hparams = {
        "n_layers": args.maf_layers,
        "layers": maf_hidden,
        "activation": jax.nn.relu,
        "use_reverse": True,
        "seed": args.seed,
    }
    print(f"  MAF: {args.maf_layers} layers, hidden={maf_hidden}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Patience: {args.patience}, min_delta: {args.min_delta}")

    split_key = jax.random.PRNGKey(args.seed + 1)
    inference = NPE(model_hparams=maf_hparams)
    inference = inference.append_simulations(
        jnp.array(all_theta), jnp.array(all_x), key=split_key
    )

    # 5) Train
    print("######## TRAINING ########")
    t0 = time.time()
    metrics, density_estimator = inference.train(
        checkpoint_path=str(out_dir / "jaxili_ckpt"),
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        training_batch_size=args.batch_size,
        patience=args.patience,
        warmup=args.warmup_steps,
        decay_steps=args.decay_steps,
        min_delta=args.min_delta,
    )
    elapsed = time.time() - t0
    print(f"  Training: {elapsed:.1f}s")

    # 6) Sample
    print("######## SAMPLING ########")
    posterior = inference.build_posterior(density_estimator)
    obs_jnp = jnp.array(obs_x.reshape(1, -1))
    samples = np.array(posterior.sample(
        args.npe_samples,
        jax.random.PRNGKey(args.seed + 42),
        x=obs_jnp,
    ))
    nan_mask = np.any(np.isnan(samples), axis=1)
    if nan_mask.sum() > 0:
        print(f"  Removed {nan_mask.sum()} NaN samples")
        samples = samples[~nan_mask]
    print(f"  {len(samples)} valid samples")

    f3 = fom3(samples)
    print(f"  FoM3 = {f3:.1f}")

    # 7) Save
    np.save(out_dir / "posterior.npy", samples)
    result = {
        "label": args.label,
        "fom3": f3,
        "n_samples": len(samples),
        "truth": truth.tolist(),
        "training_time_s": elapsed,
        "batch_size": args.batch_size,
        "maf_hidden": maf_hidden,
        "maf_layers": args.maf_layers,
        "patience": args.patience,
        "train_shape": list(all_theta.shape),
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    # 8) Quick corner plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from getdist import MCSamples, plots as gplot

        pn = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]
        mc = MCSamples(samples=samples, names=pn, labels=pn,
                       label=f"{args.label} (FoM3={f3:.0f})")
        g = gplot.get_subplot_plotter(subplot_size=1.5)
        g.triangle_plot([mc], filled=True, markers=truth,
                        marker_args={"color": "red", "lw": 1.2})
        plt.savefig(out_dir / "corner.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved corner plot")
    except Exception as e:
        print(f"  Corner plot failed: {e}")

    print(f"\n{'='*60}")
    print(f"  {args.label}: FoM3 = {f3:.0f}  ({elapsed:.0f}s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
