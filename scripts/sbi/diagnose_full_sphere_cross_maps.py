#!/usr/bin/env python
"""Diagnostic figures for the harmonic full-sphere cross-maps cache.

Reads from the cache directory built by `build_full_sphere_cross_cache.py`
and emits up to three figures per regime:

  1. `fullsphere_maps_{regime}.png` — 2x5 mollweide grid of the 4 auto + 6
     cross full-sphere maps for the snapshot cosmology. Sourced from
     `<cache>/_snapshot/fullsphere_{regime}_{cosmo_id}_perm{perm}.npz`.
     Shared symmetric color scale per row, clipped to the 99th-percentile
     of |value| to suppress outliers.

  2. `patch_gallery_{regime}.png` — 2x5 grid of one of the 48 gnomonic
     patches (default index 0) for the snapshot cosmology. Confirms that
     gnomonic projection preserves features at patch scale.

  3. `l1_per_cosmology_{regime}.png` — 10-panel strip (one per channel) of
     the wavelet L1 datavector averaged over `n_realizations * n_centers`
     patches, overlaid across the cosmologies in `--cosmo-ids`. Requires
     PyTorch + wl_stats_torch on a GPU. Skipped if any of the requested
     cosmologies are absent from the cache.

Usage:
  conda run -n jaxili python scripts/sbi/diagnose_full_sphere_cross_maps.py \
    --cache-dir <CACHE> \
    --regime nobnt \
    --cosmo-ids cosmo_fiducial \
    --n-realizations 1 \
    --out-dir <OUT>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


CHANNEL_LAYOUT = (
    "auto_1", "auto_2", "auto_3", "auto_4",
    "cross_12", "cross_13", "cross_14", "cross_23", "cross_24", "cross_34",
)
N_CHANNELS = len(CHANNEL_LAYOUT)
N_AUTO = 4


# -----------------------------------------------------------------------------
# Figure 1: full-sphere mollweide
# -----------------------------------------------------------------------------

def _symmetric_clip(arr: np.ndarray, q: float = 99.0) -> float:
    return float(np.percentile(np.abs(arr), q))


def figure_fullsphere_maps(
    cache_dir: Path,
    regime: str,
    cosmo_id: str,
    perm: int,
    out_path: Path,
) -> bool:
    import healpy as hp
    snap_path = cache_dir / "_snapshot" / f"fullsphere_{regime}_{cosmo_id}_perm{perm}.npz"
    if not snap_path.exists():
        print(f"  [skip] missing snapshot {snap_path}")
        return False
    with np.load(snap_path, allow_pickle=False) as d:
        full_auto = np.asarray(d["full_auto"])  # (4, npix)
        full_cross = np.asarray(d["full_cross"])  # (6, npix)
    all_maps = np.concatenate([full_auto, full_cross], axis=0)
    if all_maps.shape[0] != N_CHANNELS:
        raise ValueError(f"Snapshot has {all_maps.shape[0]} maps, expected {N_CHANNELS}")

    auto_clip = _symmetric_clip(all_maps[:N_AUTO])
    cross_clip = _symmetric_clip(all_maps[N_AUTO:])
    fig = plt.figure(figsize=(20, 7))
    for c in range(N_CHANNELS):
        ax_idx = c + 1
        is_auto = c < N_AUTO
        clip = auto_clip if is_auto else cross_clip
        cmap = "RdBu_r"
        hp.mollview(
            all_maps[c],
            sub=(2, 5, ax_idx),
            title=CHANNEL_LAYOUT[c],
            min=-clip,
            max=clip,
            cmap=cmap,
            cbar=True,
            badcolor="gray",
            margins=(0.0, 0.0, 0.0, 0.05),
            fig=fig.number,
        )
    fig.suptitle(
        f"Full-sphere maps — regime={regime}, {cosmo_id}, perm={perm}",
        fontsize=14, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")
    return True


# -----------------------------------------------------------------------------
# Figure 2: patch gallery
# -----------------------------------------------------------------------------

def figure_patch_gallery(
    cache_dir: Path,
    regime: str,
    cosmo_id: str,
    perm: int,
    patch_idx: int,
    split: str,
    out_path: Path,
) -> bool:
    npz_path = cache_dir / regime / split / f"{cosmo_id}_perm{perm}.npz"
    if not npz_path.exists():
        print(f"  [skip] missing {npz_path}")
        return False
    with np.load(npz_path, allow_pickle=False) as d:
        patches = np.asarray(d["patches"])  # (n_centers, H, W, 10)
        centers = np.asarray(d["patch_centers"])
    if patch_idx < 0 or patch_idx >= patches.shape[0]:
        raise IndexError(
            f"patch_idx={patch_idx} out of range [0, {patches.shape[0]})"
        )
    img = patches[patch_idx]  # (H, W, 10)
    auto_clip = _symmetric_clip(img[..., :N_AUTO])
    cross_clip = _symmetric_clip(img[..., N_AUTO:])
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for c in range(N_CHANNELS):
        ax = axes[c // 5, c % 5]
        is_auto = c < N_AUTO
        clip = auto_clip if is_auto else cross_clip
        im = ax.imshow(
            img[..., c],
            origin="lower",
            cmap="RdBu_r",
            vmin=-clip,
            vmax=clip,
            interpolation="nearest",
        )
        ax.set_title(CHANNEL_LAYOUT[c], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    lon, lat = float(centers[patch_idx, 0]), float(centers[patch_idx, 1])
    fig.suptitle(
        f"Patch {patch_idx} — regime={regime}, {cosmo_id}, perm={perm}, "
        f"center=({lon:.1f}°, {lat:.1f}°)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")
    return True


# -----------------------------------------------------------------------------
# Figure 3: L1 datavector overlay across cosmologies
# -----------------------------------------------------------------------------

_WL_STATS_PATH = "/home/tersenov/software/wl_stats_torch"


def _load_wl_stats():
    if _WL_STATS_PATH not in sys.path:
        sys.path.insert(0, _WL_STATS_PATH)
    import torch
    from wl_stats_torch import WLStatistics  # noqa: F401
    return torch, WLStatistics


def _l1_for_realization(
    patches: np.ndarray,
    stats,
    torch,
    noise_sigma: float,
    n_scales: int,
    l1_nbins: int,
    l1_min_snr_auto: float,
    l1_max_snr_auto: float,
    l1_min_snr_cross: float,
    l1_max_snr_cross: float,
    subtract_coarse_mean: bool = True,
) -> np.ndarray:
    """Return mean L1 datavector (n_scales*l1_nbins*N_CHANNELS,) over patches."""
    out_per_chan = []
    device = stats.device
    for c in range(N_CHANNELS):
        is_auto = c < N_AUTO
        ch_min = l1_min_snr_auto if is_auto else l1_min_snr_cross
        ch_max = l1_max_snr_auto if is_auto else l1_max_snr_cross
        img = torch.from_numpy(patches[:, :, :, c].astype(np.float64)).to(device)
        stats.compute_wavelet_transform(
            img, noise_sigma, subtract_coarse_mean=subtract_coarse_mean,
        )
        _, l1_norms = stats.compute_wavelet_l1_norms(
            n_bins=l1_nbins, min_snr=ch_min, max_snr=ch_max,
        )
        bin_vec = torch.cat(l1_norms, dim=-1)  # (n_patches, n_scales*l1_nbins)
        out_per_chan.append(bin_vec.cpu().numpy().mean(axis=0))
    return np.concatenate(out_per_chan)  # (N_CHANNELS * n_scales * l1_nbins,)


def figure_l1_per_cosmology(
    cache_dir: Path,
    regime: str,
    split: str,
    cosmo_ids: list[str],
    n_realizations: int,
    out_path: Path,
    n_scales: int = 5,
    l1_nbins: int = 40,
    pixel_arcmin: float = 7.5,
    sigma_e: float = 0.26,
    galaxy_density: float = 7.5,
    cross_snr_percentile: float = 1.0,
) -> bool:
    """Estimate cosmology-conditional mean L1 per channel, overlay across cosmos."""
    torch, WLStatistics = _load_wl_stats()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = WLStatistics(
        n_scales=n_scales, device=device, pixel_arcmin=pixel_arcmin, dtype=torch.float64,
    )

    # First pass: gather per-channel SNR samples for percentile calibration.
    print("  Calibrating SNR ranges...")
    all_files: dict[str, list[Path]] = {}
    for cid in cosmo_ids:
        all_files[cid] = sorted(
            (cache_dir / regime / split).glob(f"{cid}_perm*.npz")
        )[:n_realizations]
        if not all_files[cid]:
            # try other splits too
            for alt in ("obs", "train", "val"):
                if alt == split:
                    continue
                cand = sorted((cache_dir / regime / alt).glob(f"{cid}_perm*.npz"))[:n_realizations]
                if cand:
                    all_files[cid] = cand
                    print(f"  [info] {cid} not in split={split}, falling back to {alt}")
                    break
        if not all_files[cid]:
            print(f"  [skip] no realizations for {cid} in cache")
            return False

    pixel_arcmin_for_noise = pixel_arcmin
    noise_sigma = sigma_e / np.sqrt(galaxy_density * pixel_arcmin_for_noise ** 2)

    # First pass: SNR percentile calibration on the first cosmology.
    # torch.quantile caps at ~16M elements, so we reservoir-sample.
    first_cid = cosmo_ids[0]
    auto_min = float("inf")
    auto_max = float("-inf")
    cross_samples: list = []
    reservoir_per_chan = 200_000
    for f in all_files[first_cid][:1]:
        with np.load(f, allow_pickle=False) as d:
            patches = np.asarray(d["patches"])
        for c in range(N_CHANNELS):
            img = torch.from_numpy(patches[:, :, :, c].astype(np.float64)).to(device)
            stats.compute_wavelet_transform(img, noise_sigma, subtract_coarse_mean=True)
            snr = stats.snr_coeffs.reshape(-1)
            if c < N_AUTO:
                auto_min = min(auto_min, snr.min().item())
                auto_max = max(auto_max, snr.max().item())
            else:
                n = snr.numel()
                if n > reservoir_per_chan:
                    idx = torch.randint(0, n, (reservoir_per_chan,), device=snr.device)
                    sample = snr[idx]
                else:
                    sample = snr
                cross_samples.append(sample.detach().cpu())
    auto_span = auto_max - auto_min
    auto_min -= 0.05 * auto_span
    auto_max += 0.05 * auto_span
    pooled = torch.cat(cross_samples)
    lo_q = cross_snr_percentile / 100.0
    cross_min = float(torch.quantile(pooled, lo_q).item())
    cross_max = float(torch.quantile(pooled, 1.0 - lo_q).item())
    cross_span = cross_max - cross_min
    cross_min -= 0.05 * cross_span
    cross_max += 0.05 * cross_span
    print(f"  Auto SNR range  : [{auto_min:.3f}, {auto_max:.3f}]")
    print(f"  Cross SNR range : [{cross_min:.3f}, {cross_max:.3f}] (pct={cross_snr_percentile})")

    # Second pass: compute mean L1 per cosmology.
    l1_per_cosmo: dict[str, np.ndarray] = {}
    for cid, files in all_files.items():
        chunks = []
        for f in files:
            with np.load(f, allow_pickle=False) as d:
                patches = np.asarray(d["patches"])
            chunks.append(patches)
        all_patches = np.concatenate(chunks, axis=0)
        print(f"  {cid}: {all_patches.shape[0]} patches")
        l1_per_cosmo[cid] = _l1_for_realization(
            all_patches, stats, torch, noise_sigma,
            n_scales=n_scales, l1_nbins=l1_nbins,
            l1_min_snr_auto=auto_min, l1_max_snr_auto=auto_max,
            l1_min_snr_cross=cross_min, l1_max_snr_cross=cross_max,
        )

    # Plot: 10 panels, one per channel.
    fig, axes = plt.subplots(N_CHANNELS, 1, figsize=(13, 1.6 * N_CHANNELS), sharex=True)
    xs = np.arange(n_scales * l1_nbins)
    for c in range(N_CHANNELS):
        ax = axes[c]
        for cid, vec in l1_per_cosmo.items():
            ch = vec[c * n_scales * l1_nbins:(c + 1) * n_scales * l1_nbins]
            ax.plot(xs, ch, lw=1.0, label=cid)
        for k in range(1, n_scales):
            ax.axvline(k * l1_nbins, color="grey", lw=0.5, ls="--", alpha=0.6)
        ax.set_ylabel(CHANNEL_LAYOUT[c], fontsize=9)
        ax.tick_params(labelsize=8)
        if c == 0:
            ax.legend(loc="upper right", fontsize=7, ncol=min(len(l1_per_cosmo), 4))
        ax.set_xlim(0, n_scales * l1_nbins)
    axes[-1].set_xlabel(
        f"datavector index ({n_scales} scales × {l1_nbins} SNR bins, scale boundaries dashed)"
    )
    fig.suptitle(
        f"L1 datavector mean across cosmologies — regime={regime}, "
        f"{n_realizations} realization(s) × n_centers patches",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")
    return True


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--regime", type=str, default="nobnt", choices=["bnt", "nobnt"])
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--snapshot-cosmo-id", type=str, default="cosmo_fiducial",
        help="Cosmology id for the full-sphere mollweide and patch-gallery figures.",
    )
    p.add_argument("--snapshot-perm", type=int, default=0)
    p.add_argument("--patch-idx", type=int, default=0)
    p.add_argument(
        "--cosmo-ids", type=str, default="cosmo_fiducial",
        help="Comma-separated cosmology ids for the L1 overlay figure.",
    )
    p.add_argument("--cosmo-ids-split", type=str, default="train",
                   choices=["train", "val", "obs"])
    p.add_argument("--n-realizations", type=int, default=1)
    p.add_argument("--n-scales", type=int, default=5)
    p.add_argument("--l1-nbins", type=int, default=40)
    p.add_argument("--field-size", type=float, default=20.0)
    p.add_argument("--field-npix", type=int, default=160)
    p.add_argument("--sigma-e", type=float, default=0.26)
    p.add_argument("--galaxy-density", type=float, default=7.5)
    p.add_argument("--cross-snr-percentile", type=float, default=1.0)
    p.add_argument("--skip-fullsphere", action="store_true")
    p.add_argument("--skip-patch-gallery", action="store_true")
    p.add_argument("--skip-l1-overlay", action="store_true")
    p.add_argument("--cuda-visible-devices", type=str, default=None,
                   help="Optional CUDA device for the L1 figure.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pixel_arcmin = args.field_size * 60.0 / args.field_npix
    cosmo_ids = [c.strip() for c in args.cosmo_ids.split(",") if c.strip()]

    print(f"Cache dir   : {args.cache_dir}")
    print(f"Regime      : {args.regime}")
    print(f"Output dir  : {args.out_dir}")
    print(f"Snapshot    : {args.snapshot_cosmo_id} perm{args.snapshot_perm}")
    print(f"Cosmologies : {cosmo_ids}")

    if not args.skip_fullsphere:
        print("[1/3] Full-sphere mollweide ...")
        figure_fullsphere_maps(
            cache_dir=args.cache_dir, regime=args.regime,
            cosmo_id=args.snapshot_cosmo_id, perm=args.snapshot_perm,
            out_path=args.out_dir / f"fullsphere_maps_{args.regime}.png",
        )

    if not args.skip_patch_gallery:
        print("[2/3] Patch gallery ...")
        # Look in train, then val, then obs to find the snapshot cosmology in cache.
        for split_try in ("obs", "train", "val"):
            cand = args.cache_dir / args.regime / split_try / \
                f"{args.snapshot_cosmo_id}_perm{args.snapshot_perm}.npz"
            if cand.exists():
                figure_patch_gallery(
                    cache_dir=args.cache_dir, regime=args.regime,
                    cosmo_id=args.snapshot_cosmo_id, perm=args.snapshot_perm,
                    patch_idx=args.patch_idx, split=split_try,
                    out_path=args.out_dir / f"patch_gallery_{args.regime}.png",
                )
                break
        else:
            print(f"  [skip] {args.snapshot_cosmo_id} perm{args.snapshot_perm} not found")

    if not args.skip_l1_overlay:
        print("[3/3] L1 datavector overlay ...")
        figure_l1_per_cosmology(
            cache_dir=args.cache_dir, regime=args.regime,
            split=args.cosmo_ids_split, cosmo_ids=cosmo_ids,
            n_realizations=args.n_realizations,
            out_path=args.out_dir / f"l1_per_cosmology_{args.regime}.png",
            n_scales=args.n_scales, l1_nbins=args.l1_nbins,
            pixel_arcmin=pixel_arcmin, sigma_e=args.sigma_e,
            galaxy_density=args.galaxy_density,
            cross_snr_percentile=args.cross_snr_percentile,
        )

    print("Done.")


if __name__ == "__main__":
    main()
