#!/usr/bin/env python
"""Compressor diagnostics from on-disk artifacts (CPU only, no GPU, no JAX).

Reads the compressor loss arrays and the already-compressed summary cache
(produced by npe_cnn_nbody_tomo.py --exit-after-compress) and emits a
multi-page diagnostic PDF + individual PNGs. Reusable across compressor arms.

Inputs (all already on disk after a compressor run):
  <compressor-dir>/loss_compressor_train.npy   (n_saves,)
  <compressor-dir>/loss_compressor_test.npy    (n_saves,)
  <compressed-dir>/cnn_train.npz   keys: theta (N,6) float32, x (N,cdim) float32
  <compressed-dir>/cnn_val.npz     same
  <compressed-dir>/cnn_obs.npz     keys: x (cdim,), theta (6,)
  <compressed-dir>/cnn_cache_meta.npz  config metadata

Plots:
  1. VMIM loss curve (train + test vs save index, best-val marked)
  2. Per-feature distributions (train vs val overlay + obs marker)
  3. Feature variance bar chart
  4. Feature correlation matrix (cdim x cdim)
  5. Each feature vs Omega_m (which dims carry cosmological signal)
  6. 2D scatter of the two highest-variance features, colored by Omega_m
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PARAM_NAMES = ["Omega_m", "sigma_8", "w_0", "h_0", "n_s", "Omega_b"]


def _load(compressor_dir: Path, compressed_dir: Path):
    ltr = np.load(compressor_dir / "loss_compressor_train.npy")
    lte = np.load(compressor_dir / "loss_compressor_test.npy")
    tr = np.load(compressed_dir / "cnn_train.npz")
    va = np.load(compressed_dir / "cnn_val.npz")
    obs = np.load(compressed_dir / "cnn_obs.npz")
    meta = {}
    meta_path = compressed_dir / "cnn_cache_meta.npz"
    if meta_path.is_file():
        m = np.load(meta_path, allow_pickle=True)
        for k in m.keys():
            v = m[k]
            meta[k] = v.item() if v.ndim == 0 else v.tolist()
    return ltr, lte, tr["x"], tr["theta"], va["x"], va["theta"], obs["x"], meta


def plot_loss_curve(ax, ltr, lte, save_every):
    steps = (np.arange(len(ltr)) + 1) * save_every
    ax.plot(steps, ltr, "-o", ms=3, label="train", color="tab:blue")
    ax.plot(steps, lte, "-o", ms=3, label="val (test split)", color="tab:red")
    best = int(np.argmin(lte))
    ax.axvline(steps[best], ls="--", color="k", alpha=0.6,
               label=f"best-val @ step {steps[best]} ({lte[best]:.3f})")
    drift = lte[-1] - lte[best]
    ax.set_title(f"VMIM compressor loss  (val drift after best: {drift:+.3f} nats)")
    ax.set_xlabel("compressor step")
    ax.set_ylabel("loss (-log prob)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def plot_feature_dists(fig, x_tr, x_va, x_obs):
    cdim = x_tr.shape[1]
    ncol = 5
    nrow = int(np.ceil(cdim / ncol))
    axes = fig.subplots(nrow, ncol).ravel()
    for d in range(cdim):
        ax = axes[d]
        lo = min(x_tr[:, d].min(), x_va[:, d].min())
        hi = max(x_tr[:, d].max(), x_va[:, d].max())
        bins = np.linspace(lo, hi, 60)
        ax.hist(x_tr[:, d], bins=bins, density=True, alpha=0.5, label="train", color="tab:blue")
        ax.hist(x_va[:, d], bins=bins, density=True, alpha=0.5, label="val", color="tab:orange")
        ax.axvline(x_obs[d], color="k", ls="--", lw=1, label="obs")
        ax.set_title(f"feature {d}", fontsize=8)
        ax.tick_params(labelsize=6)
        if d == 0:
            ax.legend(fontsize=6)
    for d in range(cdim, len(axes)):
        axes[d].axis("off")
    fig.suptitle("Compressed-feature distributions (train vs val, obs marked)")


def plot_variance_bar(ax, x_tr):
    var = x_tr.var(axis=0)
    ax.bar(np.arange(len(var)), var, color="tab:purple")
    ax.set_title("Per-feature variance (train)")
    ax.set_xlabel("feature index")
    ax.set_ylabel("variance")
    ax.grid(alpha=0.3, axis="y")


def plot_corr_matrix(ax, x_tr):
    c = np.corrcoef(x_tr.T)
    im = ax.imshow(c, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_title("Feature correlation matrix (train)")
    ax.set_xlabel("feature")
    ax.set_ylabel("feature")
    plt.colorbar(im, ax=ax, fraction=0.046)
    # annotate strong off-diagonal correlations
    cdim = c.shape[0]
    for i in range(cdim):
        for j in range(cdim):
            if i != j and abs(c[i, j]) > 0.5:
                ax.text(j, i, f"{c[i,j]:.1f}", ha="center", va="center",
                        fontsize=5, color="k")


def plot_feature_vs_om(fig, x_tr, theta_tr):
    cdim = x_tr.shape[1]
    om = theta_tr[:, 0]
    ncol = 5
    nrow = int(np.ceil(cdim / ncol))
    axes = fig.subplots(nrow, ncol).ravel()
    # subsample for scatter legibility
    n = min(4000, x_tr.shape[0])
    idx = np.random.default_rng(0).choice(x_tr.shape[0], n, replace=False)
    for d in range(cdim):
        ax = axes[d]
        ax.scatter(om[idx], x_tr[idx, d], s=2, alpha=0.3, color="tab:green")
        # correlation with Omega_m
        r = np.corrcoef(om, x_tr[:, d])[0, 1]
        ax.set_title(f"feat {d}  (r={r:+.2f})", fontsize=8)
        ax.tick_params(labelsize=6)
        if d >= cdim - ncol:
            ax.set_xlabel("Omega_m", fontsize=7)
    for d in range(cdim, len(axes)):
        axes[d].axis("off")
    fig.suptitle("Each compressed feature vs Omega_m (r = Pearson)")


def plot_top2_scatter(ax, x_tr, theta_tr):
    var = x_tr.var(axis=0)
    top2 = np.argsort(var)[-2:]
    n = min(6000, x_tr.shape[0])
    idx = np.random.default_rng(1).choice(x_tr.shape[0], n, replace=False)
    sc = ax.scatter(x_tr[idx, top2[0]], x_tr[idx, top2[1]],
                    c=theta_tr[idx, 0], s=4, alpha=0.5, cmap="viridis")
    ax.set_title(f"Top-2-variance features (idx {top2[0]},{top2[1]}) colored by Omega_m")
    ax.set_xlabel(f"feature {top2[0]}")
    ax.set_ylabel(f"feature {top2[1]}")
    plt.colorbar(sc, ax=ax, fraction=0.046, label="Omega_m")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compressor-dir", required=True, type=Path,
                   help="dir with loss_compressor_{train,test}.npy")
    p.add_argument("--compressed-dir", required=True, type=Path,
                   help="dir with cnn_{train,val,obs}.npz")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--label", default="compressor")
    p.add_argument("--save-every", type=int, default=2000,
                   help="compressor save_every (for loss-curve x-axis)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ltr, lte, x_tr, th_tr, x_va, th_va, x_obs, meta = _load(
        args.compressor_dir, args.compressed_dir)

    cdim = x_tr.shape[1]
    best = int(np.argmin(lte))
    summary = {
        "label": args.label,
        "cdim": int(cdim),
        "n_train": int(x_tr.shape[0]),
        "n_val": int(x_va.shape[0]),
        "best_val_save_index": best,
        "best_val_step": int((best + 1) * args.save_every),
        "best_val_loss": float(lte[best]),
        "final_val_loss": float(lte[-1]),
        "val_drift_after_best": float(lte[-1] - lte[best]),
        "feature_variance": x_tr.var(axis=0).tolist(),
        "max_abs_feature_corr_offdiag": float(
            np.max(np.abs(np.corrcoef(x_tr.T) - np.eye(cdim)))),
        "feature_corr_with_Omega_m": [
            float(np.corrcoef(th_tr[:, 0], x_tr[:, d])[0, 1]) for d in range(cdim)
        ],
        "meta": {k: meta.get(k) for k in (
            "compressor_arch", "compressor_dim", "compressor_conv_channels",
            "compressor_dense_width", "zero_mean_maps",
            "require_disjoint_train_examples", "cnn_map_route",
            "compressor_train_split", "nde_train_split")},
    }
    (args.out_dir / f"{args.label}_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2))

    # Individual figures
    f1 = plt.figure(figsize=(8, 5)); ax = f1.add_subplot(111)
    plot_loss_curve(ax, ltr, lte, args.save_every)
    f1.tight_layout(); f1.savefig(args.out_dir / f"{args.label}_loss_curve.png", dpi=130)

    f2 = plt.figure(figsize=(14, 3 * int(np.ceil(cdim / 5))))
    plot_feature_dists(f2, x_tr, x_va, x_obs)
    f2.tight_layout(); f2.savefig(args.out_dir / f"{args.label}_feature_dists.png", dpi=130)

    f3 = plt.figure(figsize=(8, 5)); ax = f3.add_subplot(111)
    plot_variance_bar(ax, x_tr)
    f3.tight_layout(); f3.savefig(args.out_dir / f"{args.label}_feature_variance.png", dpi=130)

    f4 = plt.figure(figsize=(7, 6)); ax = f4.add_subplot(111)
    plot_corr_matrix(ax, x_tr)
    f4.tight_layout(); f4.savefig(args.out_dir / f"{args.label}_corr_matrix.png", dpi=130)

    f5 = plt.figure(figsize=(14, 3 * int(np.ceil(cdim / 5))))
    plot_feature_vs_om(f5, x_tr, th_tr)
    f5.tight_layout(); f5.savefig(args.out_dir / f"{args.label}_feature_vs_omega_m.png", dpi=130)

    f6 = plt.figure(figsize=(7, 6)); ax = f6.add_subplot(111)
    plot_top2_scatter(ax, x_tr, th_tr)
    f6.tight_layout(); f6.savefig(args.out_dir / f"{args.label}_top2_scatter.png", dpi=130)

    # Combined PDF
    pdf_path = args.out_dir / f"{args.label}_compressor_diagnostics.pdf"
    with PdfPages(pdf_path) as pdf:
        for f in (f1, f2, f3, f4, f5, f6):
            pdf.savefig(f)
    for f in (f1, f2, f3, f4, f5, f6):
        plt.close(f)

    print(f"[diagnostics] {args.label}: wrote {pdf_path}")
    print(f"  best-val: step {summary['best_val_step']} "
          f"loss {summary['best_val_loss']:.4f}, "
          f"drift {summary['val_drift_after_best']:+.4f} nats")
    print(f"  max |off-diag feature corr|: {summary['max_abs_feature_corr_offdiag']:.3f}")
    print(f"  feature-Omega_m corr range: "
          f"[{min(summary['feature_corr_with_Omega_m']):+.2f}, "
          f"{max(summary['feature_corr_with_Omega_m']):+.2f}]")


if __name__ == "__main__":
    main()
