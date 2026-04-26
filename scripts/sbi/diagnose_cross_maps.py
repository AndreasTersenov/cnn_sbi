#!/usr/bin/env python
"""
Cross-map diagnostic for the L1 pipeline.

Draws a small TFDS sample, applies the same augmentation pipeline as the L1
pipeline (shape noise -> optional BNT -> flat-sky cross maps), and emits three
PDFs:

  1. cross_maps_gallery.pdf       -- 4 auto + 6 cross maps for one example
  2. cross_maps_snr_histograms.pdf -- per-channel SNR distribution across
                                     examples, with 1st/99th percentile marks
  3. cross_maps_l1_datavectors.pdf -- per-channel mean +/- std L1 datavector
                                     using the recommended SNR range

Also prints a recommended `--cross-map-min-snr` / `--cross-map-max-snr`.

Run with:
  conda run -n jaxili python scripts/sbi/diagnose_cross_maps.py \
    --n-examples 20 --apply-bnt \
    --out-dir scripts/sbi/results/diagnostics/cross_maps/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch

# Reuse the pipeline module so helpers stay in one place.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import tf_dataset_nbody_tomo as _tomo_builder  # noqa: F401

from npe_l1norm_nbody_tomo import (  # noqa: E402
    _cross_pairs,
    _compute_cross_maps_tf,
    _make_apod_window,
    apply_bnt_tf,
    build_l1_computer,
    compute_l1_single_map,
    pixel_noise_sigma,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tfds-name", type=str,
                   default="NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48")
    p.add_argument("--split", type=str, default="train[:1%]")
    p.add_argument("--n-examples", type=int, default=20)
    p.add_argument("--apply-bnt", action="store_true")
    p.add_argument(
        "--zero-mean-maps",
        action="store_true",
        help=(
            "Subtract per-example, per-channel spatial means after shape-noise "
            "injection and before optional BNT / cross-map computation."
        ),
    )
    p.add_argument("--cross-map-apodize", type=str, default="cosine",
                   choices=["none", "cosine"])
    p.add_argument("--sigma-e", type=float, default=0.26)
    p.add_argument("--galaxy-density", type=float, default=10.0)
    p.add_argument("--field-size", type=int, default=20)
    p.add_argument("--field-npix", type=int, default=160)
    p.add_argument("--map-kind", type=str, default="nbody",
                   choices=["nbody", "nbody_with_baryon_ia", "gaussian"])
    p.add_argument("--tomo-bin-indices", type=str, default="1,2,3,4")
    p.add_argument("--n-scales", type=int, default=5)
    p.add_argument("--l1-nbins", type=int, default=40)
    p.add_argument("--l1-implementation", type=str, default="cnn_sbi",
                   choices=["cnn_sbi", "cosmoford"])
    p.add_argument("--pct-low", type=float, default=1.0)
    p.add_argument("--pct-high", type=float, default=99.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str,
                   default="scripts/sbi/results/diagnostics/cross_maps/")
    return p.parse_args()


def _build_augmentation_with_zero_mean(
    *,
    map_kind: str,
    sigma_e: float,
    galaxy_density: float,
    field_size: int,
    field_npix: int,
    nbins: int,
    tomo_bin_indices: tuple[int, ...],
    apply_bnt: bool,
    zero_mean_maps: bool,
    cross_map_apodize: str,
):
    """Match the L1 pipeline ordering with optional per-channel demeaning."""
    noise_std = sigma_e / tf.sqrt(
        galaxy_density * (field_size * 60.0 / field_npix) ** 2
    )
    map_key = {
        "nbody": "map_nbody",
        "nbody_with_baryon_ia": "map_nbody_w_baryon_ia",
        "gaussian": "map_gaussian",
    }[map_kind]
    gather_indices = tf.constant([b - 1 for b in tomo_bin_indices], dtype=tf.int32)
    apod = _make_apod_window(field_npix, kind=cross_map_apodize)

    def augmentation_noise(example):
        x = tf.gather(example[map_key], gather_indices, axis=-1)
        x += tf.random.normal(shape=(field_npix, field_npix, nbins), stddev=noise_std)
        if zero_mean_maps:
            x = x - tf.reduce_mean(x, axis=[0, 1], keepdims=True)
        if apply_bnt:
            x = apply_bnt_tf(x)
        x = _compute_cross_maps_tf(x, apod)
        return {"maps": x, "theta": example["theta"]}

    def augmentation_flip(example):
        x = example["maps"]
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
        return {"maps": x, "theta": example["theta"]}

    def rescale_h(example):
        x = example["theta"]
        x = tf.tensor_scatter_nd_update(x, [[3]], [x[3] / 100.0])
        return {"maps": example["maps"], "theta": x}

    def augmentation(example):
        return rescale_h(augmentation_flip(augmentation_noise(example)))

    return augmentation


def _collect_examples(args: argparse.Namespace, tomo_bins: tuple[int, ...]) -> np.ndarray:
    """Run the pipeline's own augmentation and return (N, H, W, C) numpy maps."""
    tf.random.set_seed(args.seed)
    aug = _build_augmentation_with_zero_mean(
        map_kind=args.map_kind,
        sigma_e=args.sigma_e,
        galaxy_density=args.galaxy_density,
        field_size=args.field_size,
        field_npix=args.field_npix,
        nbins=len(tomo_bins),
        tomo_bin_indices=tomo_bins,
        apply_bnt=args.apply_bnt,
        zero_mean_maps=args.zero_mean_maps,
        cross_map_apodize=args.cross_map_apodize,
    )
    ds = tfds.load(args.tfds_name, split=args.split)
    ds = ds.take(args.n_examples)
    ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(min(args.n_examples, 16))
    ds = ds.prefetch(tf.data.AUTOTUNE)

    maps: list[np.ndarray] = []
    for batch in ds.as_numpy_iterator():
        maps.append(batch["maps"])
    if not maps:
        raise RuntimeError(f"No examples loaded from {args.tfds_name} [{args.split}]")
    return np.concatenate(maps, axis=0)


def _save_map_gallery(
    maps: np.ndarray, nbins: int, out_path: Path, field_size: int
) -> None:
    pairs = _cross_pairs(nbins)
    n_auto = nbins
    n_cross = len(pairs)
    ncols = max(n_auto, n_cross)
    fig, axes = plt.subplots(2, ncols, figsize=(3.0 * ncols, 6.4))
    if ncols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    extent = [-field_size / 2, field_size / 2, -field_size / 2, field_size / 2]
    example = maps[0]

    for b in range(n_auto):
        ax = axes[0, b]
        v = np.percentile(np.abs(example[..., b]), 99)
        ax.imshow(example[..., b], vmin=-v, vmax=v, cmap="viridis",
                  extent=extent, origin="lower")
        # show colorbar
        cbar = ax.figure.colorbar(ax.imshow(example[..., b], cmap="viridis",
                  extent=extent, origin="lower"), ax=ax)
        cbar.ax.set_ylabel("Map value", rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=8)
        ax.set_title(f"Auto bin {b + 1}")
        ax.set_xlabel("deg"); ax.set_ylabel("deg")
    for b in range(n_auto, ncols):
        axes[0, b].axis("off")

    for k, (i, j) in enumerate(pairs):
        ax = axes[1, k]
        ch = example[..., n_auto + k]
        v = np.percentile(np.abs(ch), 99)
        ax.imshow(ch, vmin=-v, vmax=v*100, cmap="viridis",
                  extent=extent, origin="lower")
        cbar = ax.figure.colorbar(ax.imshow(ch, cmap="viridis",
                  extent=extent, origin="lower"), ax=ax)
        cbar.ax.set_ylabel("Map value", rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=8)
        ax.set_title(f"Cross {i + 1}x{j + 1}")
        ax.set_xlabel("deg"); ax.set_ylabel("deg")
    for k in range(n_cross, ncols):
        axes[1, k].axis("off")

    fig.suptitle("Example 0 -- auto (top) vs cross (bottom) maps")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def _collect_snr_stats(
    maps: np.ndarray, args: argparse.Namespace, nbins: int, n_l1_channels: int,
) -> tuple[list[np.ndarray], float, float, float, float, float, float, float, float]:
    """Return per-channel SNR flat arrays and auto/cross (min,max,low-pct,high-pct)."""
    pixel_arcmin = args.field_size * 60.0 / args.field_npix
    noise_sigma = pixel_noise_sigma(
        args.sigma_e, args.galaxy_density, args.field_size, args.field_npix
    )
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = build_l1_computer(
        args.n_scales, pixel_arcmin, torch_device, args.l1_implementation
    )

    snr_per_channel: list[list[np.ndarray]] = [[] for _ in range(n_l1_channels)]
    map_dtype = np.float32 if args.l1_implementation == "cosmoford" else np.float64

    for idx in range(maps.shape[0]):
        for b in range(n_l1_channels):
            img = torch.from_numpy(maps[idx, :, :, b].astype(map_dtype)).to(torch_device)
            if args.l1_implementation == "cosmoford":
                stats.compute_wavelet_transform(img.float(), float(noise_sigma))
            else:
                stats.compute_wavelet_transform(
                    img, noise_sigma, subtract_coarse_mean=True
                )
            snr_per_channel[b].append(stats.snr_coeffs.detach().cpu().numpy().ravel())

    flats = [np.concatenate(v) for v in snr_per_channel]

    auto_flat = np.concatenate(flats[:nbins])
    auto_min, auto_max = float(auto_flat.min()), float(auto_flat.max())
    auto_lo = float(np.percentile(auto_flat, args.pct_low))
    auto_hi = float(np.percentile(auto_flat, args.pct_high))

    if n_l1_channels > nbins:
        cross_flat = np.concatenate(flats[nbins:])
        cross_min, cross_max = float(cross_flat.min()), float(cross_flat.max())
        cross_lo = float(np.percentile(cross_flat, args.pct_low))
        cross_hi = float(np.percentile(cross_flat, args.pct_high))
    else:
        cross_min = cross_max = cross_lo = cross_hi = float("nan")

    return (flats, auto_min, auto_max, auto_lo, auto_hi,
            cross_min, cross_max, cross_lo, cross_hi)


def _save_snr_histograms(
    flats: list[np.ndarray], nbins: int, out_path: Path,
    auto_lo: float, auto_hi: float, cross_lo: float, cross_hi: float,
    pct_low: float, pct_high: float,
) -> None:
    n = len(flats)
    ncols = max(5, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.6 * nrows))
    axes = np.atleast_2d(axes)
    pairs = _cross_pairs(nbins)

    for b in range(n):
        ax = axes[b // ncols, b % ncols]
        data = flats[b]
        ax.hist(data, bins=120, color="steelblue", alpha=0.8)
        if b < nbins:
            ax.axvline(auto_lo, color="k", ls="--", lw=1)
            ax.axvline(auto_hi, color="k", ls="--", lw=1)
            ax.set_title(f"Auto bin {b + 1}")
        else:
            i, j = pairs[b - nbins]
            ax.axvline(cross_lo, color="k", ls="--", lw=1)
            ax.axvline(cross_hi, color="k", ls="--", lw=1)
            ax.set_title(f"Cross {i + 1}x{j + 1}")
        ax.set_xlabel("SNR")
        ax.set_yscale("log")
    for b in range(n, nrows * ncols):
        axes[b // ncols, b % ncols].axis("off")

    fig.suptitle(
        f"Per-channel SNR (dashed: {pct_low:.1f}/{pct_high:.1f} percentiles)"
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def _save_l1_datavectors(
    maps: np.ndarray, args: argparse.Namespace, nbins: int, n_l1_channels: int,
    auto_lo: float, auto_hi: float, cross_lo: float, cross_hi: float,
    out_path: Path,
) -> None:
    pixel_arcmin = args.field_size * 60.0 / args.field_npix
    noise_sigma = pixel_noise_sigma(
        args.sigma_e, args.galaxy_density, args.field_size, args.field_npix
    )
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = build_l1_computer(
        args.n_scales, pixel_arcmin, torch_device, args.l1_implementation
    )

    per_example = []
    for idx in range(maps.shape[0]):
        v = compute_l1_single_map(
            maps[idx], noise_sigma, stats, args.l1_nbins, nbins,
            l1_min_snr=auto_lo, l1_max_snr=auto_hi,
            l1_implementation=args.l1_implementation,
            n_l1_channels=n_l1_channels,
            l1_min_snr_cross=cross_lo, l1_max_snr_cross=cross_hi,
        )
        per_example.append(v)
    stacked = np.stack(per_example, axis=0)  # (N, n_scales*l1_nbins*n_l1_channels)
    per_chan_len = args.n_scales * args.l1_nbins
    reshaped = stacked.reshape(stacked.shape[0], n_l1_channels, per_chan_len)
    mean = reshaped.mean(axis=0)
    std = reshaped.std(axis=0)

    ncols = max(5, n_l1_channels)
    nrows = int(np.ceil(n_l1_channels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.4 * nrows),
                             sharex=False)
    axes = np.atleast_2d(axes)
    pairs = _cross_pairs(nbins)
    xs = np.arange(per_chan_len)

    for b in range(n_l1_channels):
        ax = axes[b // ncols, b % ncols]
        ax.plot(xs, mean[b], color="C0", lw=0.9)
        ax.fill_between(xs, mean[b] - std[b], mean[b] + std[b],
                        color="C0", alpha=0.25)
        label = (f"Auto bin {b + 1}" if b < nbins
                 else f"Cross {pairs[b - nbins][0] + 1}x{pairs[b - nbins][1] + 1}")
        ax.set_title(label)
        ax.set_xlabel("bin index (n_scales * l1_nbins)")
    for b in range(n_l1_channels, nrows * ncols):
        axes[b // ncols, b % ncols].axis("off")

    fig.suptitle("Mean +/- std L1 datavector per channel")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tomo_bins = tuple(int(t) for t in args.tomo_bin_indices.split(","))
    nbins = len(tomo_bins)
    if args.apply_bnt and nbins != 4:
        raise ValueError("--apply-bnt requires exactly 4 tomographic bins")
    n_cross = (nbins * (nbins - 1)) // 2
    n_l1_channels = nbins + n_cross

    print(f"######## DIAGNOSE_CROSS_MAPS ########")
    print(f"  TFDS     : {args.tfds_name} [{args.split}]")
    print(f"  nbins    : {nbins} (+{n_cross} cross)  -> {n_l1_channels} channels")
    print(f"  apply_bnt: {args.apply_bnt}")
    print(f"  apodize  : {args.cross_map_apodize}")

    maps = _collect_examples(args, tomo_bins)
    print(f"  Loaded maps: shape={maps.shape}")
    assert maps.shape[-1] == n_l1_channels, (
        f"Expected {n_l1_channels} channels, got {maps.shape[-1]}"
    )

    _save_map_gallery(maps, nbins, out_dir / "cross_maps_gallery.pdf",
                      args.field_size)

    (flats, auto_min, auto_max, auto_lo, auto_hi,
     cross_min, cross_max, cross_lo, cross_hi) = _collect_snr_stats(
        maps, args, nbins, n_l1_channels,
    )
    print(
        f"  Auto  SNR : total=[{auto_min:.3f}, {auto_max:.3f}]  "
        f"pct=[{auto_lo:.3f}, {auto_hi:.3f}]"
    )
    if n_cross:
        print(
            f"  Cross SNR : total=[{cross_min:.3f}, {cross_max:.3f}]  "
            f"pct=[{cross_lo:.3f}, {cross_hi:.3f}]"
        )
        print(
            f"  Recommended: --cross-map-min-snr {cross_lo:.3f} "
            f"--cross-map-max-snr {cross_hi:.3f}"
        )

    _save_snr_histograms(
        flats, nbins, out_dir / "cross_maps_snr_histograms.pdf",
        auto_lo, auto_hi, cross_lo, cross_hi,
        args.pct_low, args.pct_high,
    )
    _save_l1_datavectors(
        maps, args, nbins, n_l1_channels,
        auto_lo, auto_hi, cross_lo, cross_hi,
        out_dir / "cross_maps_l1_datavectors.pdf",
    )

    np.savez(
        out_dir / "cross_maps_snr_summary.npz",
        auto_min=auto_min, auto_max=auto_max, auto_lo=auto_lo, auto_hi=auto_hi,
        cross_min=cross_min, cross_max=cross_max,
        cross_lo=cross_lo, cross_hi=cross_hi,
        pct_low=args.pct_low, pct_high=args.pct_high,
        n_examples=args.n_examples, apply_bnt=args.apply_bnt,
        cross_map_apodize=args.cross_map_apodize,
    )
    print(f"  Wrote {out_dir / 'cross_maps_snr_summary.npz'}")


if __name__ == "__main__":
    main()
