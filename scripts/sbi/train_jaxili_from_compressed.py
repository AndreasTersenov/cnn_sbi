#!/usr/bin/env python3
"""Phase 0a — standalone jaxili MAF NDE on a pre-compressed dataset.

⚠️ DRAFT (2026-05-28, Part-2 overnight session). NOT yet run end-to-end.
Mirrors the jaxili NDE setup of `npe_l1norm_cross_jaxili_nbody_tomo.py` so the
CNN NDE arms are a *controlled* match to the L1 arms (same NPE() defaults, same
NaN-retry, same FoM3). Review before production use; see VERIFY comments.

Consumes a compressed cache produced by `npe_cnn_nbody_tomo.py --exit-after-compress`:
    <compressed-dir>/cnn_train.npz   {theta (N,6), x (N,D)}
    <compressed-dir>/cnn_val.npz     {theta (M,6), x (M,D)}   (diagnostic only)
    <compressed-dir>/cnn_obs.npz     {x (D,), theta (6,)}     (single observed point)

For each seed (and optional extra obs/perm files), trains a jaxili MAF, samples
100k, and writes posterior .npy + .meta.json + .fom.json (FoM3 + 2D FoM + marg σ).

VERIFY before production:
  - cnn_obs.npz holds ONE obs point (perm baked into compression). Multi-perm CNN
    arms need one obs file per perm (--obs-files) or re-compression per perm.
  - theta is assumed already in model units (h0/100): obs truth showed h0=0.6736.
  - Feeds cnn_train (theta,x) to NPE; jaxili does its own internal train/val split
    (cnn_val.npz is loaded only for an external val-loss diagnostic).
  - NPE() uses jaxili defaults (matches the L1 arms). If the campaign pins a specific
    MAF arch (hidden=[50,50], 5 transforms), set it where NPE() is constructed in BOTH
    this script and the L1 path so they stay identical.

Usage:
  python train_jaxili_from_compressed.py \
    --compressed-dir .../compressed/auto_rnvp_split70 \
    --arm-label cnn_auto_rnvp_nostd --output-dir .../posteriors \
    --seeds 41,42,43 --epochs 50000 --batch-size 256 --learning-rate 1e-4 \
    --cuda-visible-devices 1
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

FIDUCIAL = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493])
PARAM_KEYS = ["Omega_m", "sigma_8", "w_0", "h_0", "n_s", "Omega_b"]
FOM3_IDX = [0, 1, 2]


# ---- metrics (copied verbatim from npe_l1norm_cross_jaxili_nbody_tomo.py) ----
def compute_fom3(samples: np.ndarray) -> dict:
    if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] < 3:
        return {"fom3": float("nan"), "det_cov3": float("nan"),
                "logdet_cov3": float("nan"), "valid_fom3": False}
    cov3 = np.cov(samples[:, :3], rowvar=False)
    sign, logdet = np.linalg.slogdet(cov3)
    if sign <= 0:
        return {"fom3": float("nan"), "det_cov3": float("nan"),
                "logdet_cov3": float(logdet), "valid_fom3": False}
    return {"fom3": float(np.exp(-0.5 * logdet)), "det_cov3": float(np.exp(logdet)),
            "logdet_cov3": float(logdet), "valid_fom3": True}


def fom2d(samples: np.ndarray) -> dict:
    """2D FoM = 1/sqrt(det C_2) per parameter pair (FoM3 is single-seed fragile;
    2D areas are the recommended secondary metric)."""
    out = {}
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        C = np.cov(samples[:, [i, j]], rowvar=False)
        det = np.linalg.det(C)
        key = f"fom2d_{PARAM_KEYS[i]}_{PARAM_KEYS[j]}"
        out[key] = float(1.0 / np.sqrt(det)) if det > 0 else float("nan")
    return out


def marginal_stats(samples: np.ndarray) -> dict:
    return {
        "sigma": {PARAM_KEYS[i]: float(np.std(samples[:, i])) for i in range(6)},
        "bias": {PARAM_KEYS[i]: float(np.mean(samples[:, i]) - FIDUCIAL[i]) for i in range(6)},
    }


def setup_env(cuda_devices: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    print(f"CUDA_VISIBLE_DEVICES = {cuda_devices}")


def load_compressed(d: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tr = np.load(d / "cnn_train.npz")
    obs = np.load(d / "cnn_obs.npz")
    return (tr["theta"].astype(np.float32), tr["x"].astype(np.float32),
            obs["x"].astype(np.float32), obs["theta"].astype(np.float64))


def standardize_and_mask(x_tr, x_obs, do_std: bool, min_var: float):
    """Feature-mask zero/low-variance cols; optional z-score from train stats."""
    var = x_tr.var(axis=0)
    mask = var > min_var
    x_tr, x_obs = x_tr[:, mask], x_obs[mask]
    stats = {"feature_mask": mask}
    if do_std:
        mu, sd = x_tr.mean(0), x_tr.std(0)
        sd = np.where(sd > 0, sd, 1.0)
        x_tr = (x_tr - mu) / sd
        x_obs = (x_obs - mu) / sd
        stats.update({"mu": mu, "sd": sd})
    return x_tr.astype(np.float32), x_obs.astype(np.float32), stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compressed-dir", type=str, required=True)
    p.add_argument("--arm-label", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--seeds", type=str, default="41,42,43")
    p.add_argument("--obs-files", type=str, default="",
                   help="Comma-separated extra cnn_obs npz (one per perm). "
                        "Empty => use <compressed-dir>/cnn_obs.npz as perm 0.")
    p.add_argument("--standardize-summary", action="store_true")
    p.add_argument("--min-feature-variance", type=float, default=1e-12)
    p.add_argument("--epochs", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--decay-steps", type=int, default=10000)
    p.add_argument("--npe-samples", type=int, default=100000)
    p.add_argument("--cuda-visible-devices", type=str, default="1")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_env(args.cuda_visible_devices)
    import jax
    import jax.numpy as jnp
    from jaxili.inference import NPE
    # train_with_nan_retry lives in the L1 script; import it to stay identical.
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from npe_l1norm_cross_jaxili_nbody_tomo import train_with_nan_retry

    cdir = Path(args.compressed_dir)
    theta_tr, x_tr_raw, x_obs_raw, obs_truth = load_compressed(cdir)
    print(f"  train: theta{theta_tr.shape} x{x_tr_raw.shape} ; obs truth={obs_truth}")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    obs_files = [s for s in args.obs_files.split(",") if s.strip()]
    perms = list(range(len(obs_files))) if obs_files else [0]

    out_dir = Path(args.output_dir) / args.arm_label
    out_dir.mkdir(parents=True, exist_ok=True)

    for perm in perms:
        if obs_files:
            o = np.load(obs_files[perm])
            x_obs_raw = o["x"].astype(np.float32)
        x_tr, x_obs, stats = standardize_and_mask(
            x_tr_raw, x_obs_raw, args.standardize_summary, args.min_feature_variance)
        for seed in seeds:
            tag = f"{args.arm_label}_s{seed}_p{perm}"
            print(f"######## {tag} (D={x_tr.shape[1]}) ########")
            split_key = jax.random.PRNGKey(int(seed) + 1)
            params = jnp.asarray(theta_tr)
            data = jnp.asarray(x_tr)
            inference = NPE()
            inference = inference.append_simulations(params, data, key=split_key)
            ckpt = str(out_dir / f"ckpt_{tag}")
            inference, _metrics, _de = train_with_nan_retry(
                inference, ckpt, args.epochs, args.learning_rate, args.batch_size,
                args.warmup_steps, args.decay_steps, params, data, split_key)
            posterior = inference.build_posterior()
            sample_key = jax.random.PRNGKey(int(seed) + 7)
            samples = posterior.sample(x=jnp.asarray(x_obs),
                                       num_samples=args.npe_samples, key=sample_key)
            s = np.asarray(samples)
            s = s[np.all(np.isfinite(s), axis=1)]
            if s.shape[0] == 0:
                print(f"  [warn] all non-finite for {tag}; skipping save")
                continue
            npy = out_dir / f"{tag}.npy"
            np.save(npy, s)
            metrics = {**compute_fom3(s), **fom2d(s), **marginal_stats(s)}
            npy.with_suffix(".fom.json").write_text(json.dumps(metrics, indent=2))
            npy.with_suffix(".meta.json").write_text(json.dumps({
                "arm_label": args.arm_label, "seed": seed, "perm": perm,
                "compressed_dir": str(cdir), "summary_dim": int(x_tr.shape[1]),
                "standardize_summary": bool(args.standardize_summary),
                "epochs": args.epochs, "batch_size": args.batch_size,
                "learning_rate": args.learning_rate, "npe_samples": args.npe_samples,
                "obs_truth": obs_truth.tolist(),
            }, indent=2))
            print(f"  saved {npy.name}  FoM3={metrics['fom3']:.0f}")
    print("[done] train_jaxili_from_compressed")


if __name__ == "__main__":
    main()
