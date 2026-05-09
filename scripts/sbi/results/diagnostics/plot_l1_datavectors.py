#!/usr/bin/env python
"""Generate L1 datavector diagnostic plots from cached SBI artifacts."""
from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable

import h5py
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

WL_STATS_PATH = "/home/tersenov/software/wl_stats_torch"
if WL_STATS_PATH not in sys.path:
    sys.path.insert(0, WL_STATS_PATH)
from wl_stats_torch import WLStatistics  # noqa: E402


FIDUCIAL_META = Path("/home/tersenov/CosmoGridV1/CosmoGridV1_metainfo.h5")
FIDUCIAL_MAP = Path(
    "/home/tersenov/CosmoGridV1/stage3_forecast/fiducial/"
    "cosmo_fiducial/perm_0000/projected_probes_maps_nobaryons512.h5"
)
FIDUCIAL_THETA = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493], dtype=np.float64)
REP_COLORS = ["C0", "C1", "C2", "C4", "C6", "C8"]


@dataclass(frozen=True)
class VariantConfig:
    name: str
    cache_dir: Path
    field_size_deg: int
    field_npix: int
    tomo_bin_indices: tuple[int, ...]
    n_scales: int
    l1_nbins: int
    l1_min_snr: float
    l1_max_snr: float
    l1_clamp_overflow: bool
    subtract_coarse_mean: bool

    @property
    def nbins(self) -> int:
        return len(self.tomo_bin_indices)

    @property
    def features_per_bin(self) -> int:
        return self.n_scales * self.l1_nbins


def parse_tomo_bins(spec: str) -> tuple[int, ...]:
    return tuple(int(s.strip()) for s in str(spec).split(",") if s.strip())


def infer_field_geometry(variant: str) -> tuple[int, int]:
    if "20deg160" in variant:
        return 20, 160
    if "10deg80" in variant:
        return 10, 80
    raise ValueError(f"Cannot infer field geometry from variant '{variant}'")


def find_cache_dir(variant: str, roots: Iterable[Path]) -> Path:
    for root in roots:
        cand = root / variant
        if (cand / "l1_cache_meta.npz").exists() and (cand / "l1_val.npz").exists():
            return cand
    raise FileNotFoundError(
        f"Missing cache for '{variant}'. Checked: {', '.join(str(r / variant) for r in roots)}"
    )


def load_variant_config(variant: str, roots: Iterable[Path]) -> VariantConfig:
    cache_dir = find_cache_dir(variant, roots)
    meta = np.load(cache_dir / "l1_cache_meta.npz")
    field_size, field_npix = infer_field_geometry(variant)
    return VariantConfig(
        name=variant,
        cache_dir=cache_dir,
        field_size_deg=field_size,
        field_npix=field_npix,
        tomo_bin_indices=parse_tomo_bins(meta["tomo_bin_indices"].item()),
        n_scales=int(meta["n_scales"]),
        l1_nbins=int(meta["l1_nbins"]),
        l1_min_snr=float(meta["l1_min_snr"]),
        l1_max_snr=float(meta["l1_max_snr"]),
        l1_clamp_overflow=bool(meta["l1_clamp_overflow"]),
        subtract_coarse_mean=bool(meta["subtract_coarse_mean"]),
    )


def pixel_noise_sigma(field_size_deg: float, field_npix: int, sigma_e: float, galaxy_density: float) -> float:
    reso_arcmin = field_size_deg * 60.0 / field_npix
    return sigma_e / np.sqrt(galaxy_density * reso_arcmin**2)


def project_observed_map(cfg: VariantConfig, noise_seed: int, sigma_e: float, galaxy_density: float) -> np.ndarray:
    reso = cfg.field_size_deg * 60.0 / cfg.field_npix
    projector = hp.projector.GnomonicProj(rot=[0, 0, 0], xsize=cfg.field_npix, ysize=cfg.field_npix, reso=reso)

    with h5py.File(FIDUCIAL_MAP, "r") as f:
        kg = f["kg"]
        projected = []
        for tomo_bin in cfg.tomo_bin_indices:
            full_map = np.array(kg[f"stage3_lensing{tomo_bin}"])
            projected.append(projector.projmap(full_map, vec2pix_func=partial(hp.vec2pix, 512)))

    observed = np.stack(projected, axis=-1).astype(np.float32)
    noise_std = pixel_noise_sigma(cfg.field_size_deg, cfg.field_npix, sigma_e, galaxy_density)
    rng = np.random.default_rng(noise_seed)
    observed += rng.normal(0.0, noise_std, size=observed.shape).astype(np.float32)
    return observed


def compute_observed_l1(cfg: VariantConfig, noise_seed: int, sigma_e: float, galaxy_density: float) -> np.ndarray:
    observed_map = project_observed_map(cfg, noise_seed=noise_seed, sigma_e=sigma_e, galaxy_density=galaxy_density)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pixel_arcmin = cfg.field_size_deg * 60.0 / cfg.field_npix
    noise_std = pixel_noise_sigma(cfg.field_size_deg, cfg.field_npix, sigma_e, galaxy_density)
    stats = WLStatistics(
        n_scales=cfg.n_scales,
        device=device,
        pixel_arcmin=pixel_arcmin,
        dtype=torch.float64,
    )

    all_scales = []
    for tomo_idx in range(cfg.nbins):
        img = torch.from_numpy(observed_map[:, :, tomo_idx].astype(np.float64)).to(device)
        stats.compute_wavelet_transform(
            img,
            noise_std,
            subtract_coarse_mean=cfg.subtract_coarse_mean,
        )
        _, l1_norms = stats.compute_wavelet_l1_norms(
            n_bins=cfg.l1_nbins,
            min_snr=cfg.l1_min_snr,
            max_snr=cfg.l1_max_snr,
            clamp_overflow=cfg.l1_clamp_overflow,
        )
        all_scales.append(torch.cat(l1_norms, dim=-1).cpu().numpy())

    return np.concatenate(all_scales)


def choose_representative_indices(theta: np.ndarray) -> list[tuple[str, int]]:
    om = theta[:, 0]
    s8 = theta[:, 1]
    om_q10, om_q90 = np.quantile(om, [0.10, 0.90])
    s8_q10, s8_q90 = np.quantile(s8, [0.10, 0.90])

    targets = [
        ("fiducial-nearest", (FIDUCIAL_THETA[0], FIDUCIAL_THETA[1])),
        ("low-Om low-s8", (om_q10, s8_q10)),
        ("high-Om low-s8", (om_q90, s8_q10)),
        ("low-Om high-s8", (om_q10, s8_q90)),
        ("high-Om high-s8", (om_q90, s8_q90)),
    ]

    scale = np.array([om.std(), s8.std()]) + 1e-8
    selected: list[int] = []
    out: list[tuple[str, int]] = []
    for name, (target_om, target_s8) in targets:
        dist = ((theta[:, 0] - target_om) / scale[0]) ** 2 + ((theta[:, 1] - target_s8) / scale[1]) ** 2
        if selected:
            dist[np.array(selected, dtype=int)] = np.inf
        idx = int(np.argmin(dist))
        selected.append(idx)
        out.append((name, idx))
    return out


def feature_slice(cfg: VariantConfig, tomo_idx: int, scale_idx: int) -> slice:
    start = tomo_idx * cfg.features_per_bin + scale_idx * cfg.l1_nbins
    return slice(start, start + cfg.l1_nbins)


def make_profiles_figure(
    cfg: VariantConfig,
    x: np.ndarray,
    theta: np.ndarray,
    observed_l1: np.ndarray,
    rep_indices: list[tuple[str, int]],
    out_file: Path,
) -> None:
    snr_edges = np.linspace(cfg.l1_min_snr, cfg.l1_max_snr, cfg.l1_nbins + 1)
    snr_centers = 0.5 * (snr_edges[:-1] + snr_edges[1:])

    fig, axes = plt.subplots(
        cfg.nbins,
        cfg.n_scales,
        figsize=(3.4 * cfg.n_scales, 2.9 * cfg.nbins),
        squeeze=False,
        sharex=True,
    )

    for b in range(cfg.nbins):
        for s in range(cfg.n_scales):
            ax = axes[b, s]
            sl = feature_slice(cfg, b, s)
            block = x[:, sl]
            mean = block.mean(axis=0)
            p16, p84 = np.percentile(block, [16.0, 84.0], axis=0)

            ax.fill_between(snr_centers, p16, p84, color="0.85", alpha=0.8, label="cache p16-p84" if (b, s) == (0, 0) else None)
            ax.plot(snr_centers, mean, color="black", ls="--", lw=1.2, label="cache mean" if (b, s) == (0, 0) else None)

            for i, (name, idx) in enumerate(rep_indices):
                curve = x[idx, sl]
                label = f"{name} (Om={theta[idx,0]:.3f}, s8={theta[idx,1]:.3f})" if (b, s) == (0, 0) else None
                ax.plot(snr_centers, curve, lw=1.2, color=REP_COLORS[i % len(REP_COLORS)], label=label)

            ax.plot(
                snr_centers,
                observed_l1[sl],
                color="crimson",
                lw=1.4,
                label="observed" if (b, s) == (0, 0) else None,
            )

            if b == cfg.nbins - 1:
                ax.set_xlabel("SNR bin center")
            if s == 0:
                ax.set_ylabel(f"tomo bin {cfg.tomo_bin_indices[b]}\nL1 value")
            ax.set_title(f"scale {s + 1}")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle(f"{cfg.name}: L1 datavector profiles per scale", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def make_scale_totals_line_figure(
    cfg: VariantConfig,
    x: np.ndarray,
    theta: np.ndarray,
    observed_l1: np.ndarray,
    rep_indices: list[tuple[str, int]],
    out_file: Path,
) -> None:
    fig, axes = plt.subplots(1, cfg.nbins, figsize=(4.2 * cfg.nbins, 3.6), squeeze=False, sharey=True)

    for b in range(cfg.nbins):
        ax = axes[0, b]
        scales = np.arange(1, cfg.n_scales + 1)

        mean_totals = []
        obs_totals = []
        for s in range(cfg.n_scales):
            sl = feature_slice(cfg, b, s)
            mean_totals.append(float(x[:, sl].sum(axis=1).mean()))
            obs_totals.append(float(observed_l1[sl].sum()))

        ax.plot(scales, mean_totals, "k--o", lw=1.5, ms=4, label="cache mean")
        ax.plot(scales, obs_totals, color="crimson", marker="o", lw=1.6, ms=4, label="observed")

        for i, (name, idx) in enumerate(rep_indices):
            totals = [float(x[idx, feature_slice(cfg, b, s)].sum()) for s in range(cfg.n_scales)]
            ax.plot(
                scales,
                totals,
                marker="o",
                ms=3,
                lw=1.1,
                color=REP_COLORS[i % len(REP_COLORS)],
                label=name,
            )

        ax.set_xticks(scales)
        ax.set_xlabel("Wavelet scale")
        ax.set_title(f"tomo bin {cfg.tomo_bin_indices[b]}")
        if b == 0:
            ax.set_ylabel("Total L1 per scale")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(f"{cfg.name}: scale-integrated L1 totals", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def make_scale_scatter_figure(
    cfg: VariantConfig,
    x: np.ndarray,
    theta: np.ndarray,
    observed_l1: np.ndarray,
    rep_indices: list[tuple[str, int]],
    out_file: Path,
    scatter_n: int,
) -> None:
    rng = np.random.default_rng(123)
    n = len(theta)
    if n > scatter_n:
        idx_sub = rng.choice(n, size=scatter_n, replace=False)
        x_sub = x[idx_sub]
        theta_sub = theta[idx_sub]
    else:
        x_sub = x
        theta_sub = theta

    fig, axes = plt.subplots(
        cfg.nbins,
        cfg.n_scales,
        figsize=(3.6 * cfg.n_scales, 2.8 * cfg.nbins),
        squeeze=False,
        sharex=True,
    )

    colorbar_obj = None
    for b in range(cfg.nbins):
        for s in range(cfg.n_scales):
            ax = axes[b, s]
            sl = feature_slice(cfg, b, s)
            yvals = x_sub[:, sl].sum(axis=1)
            colorbar_obj = ax.scatter(
                theta_sub[:, 0],
                yvals,
                c=theta_sub[:, 1],
                cmap="viridis",
                s=5,
                alpha=0.18,
                rasterized=True,
            )

            for i, (name, idx) in enumerate(rep_indices):
                total = float(x[idx, sl].sum())
                ax.scatter(theta[idx, 0], total, color=REP_COLORS[i % len(REP_COLORS)], s=35, marker="D", edgecolors="black", linewidths=0.4)
                if b == 0 and s == 0:
                    ax.text(theta[idx, 0], total, f" {name}", fontsize=7, va="center")

            obs_total = float(observed_l1[sl].sum())
            ax.axhline(obs_total, color="crimson", lw=1.1, ls="--")

            if b == cfg.nbins - 1:
                ax.set_xlabel(r"$\Omega_m$")
            if s == 0:
                ax.set_ylabel(f"bin {cfg.tomo_bin_indices[b]}\nscale-total L1")
            ax.set_title(f"scale {s + 1}")

    if colorbar_obj is not None:
        cbar = fig.colorbar(colorbar_obj, ax=axes.ravel().tolist(), fraction=0.012, pad=0.01)
        cbar.set_label(r"$\sigma_8$")

    fig.suptitle(f"{cfg.name}: scale totals vs cosmology", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def generate_variant_plots(
    cfg: VariantConfig,
    split: str,
    out_dir: Path,
    noise_seed: int,
    sigma_e: float,
    galaxy_density: float,
    scatter_n: int,
) -> list[Path]:
    data = np.load(cfg.cache_dir / f"l1_{split}.npz")
    theta = data["theta"]
    x = data["x"]

    expected_dim = cfg.features_per_bin * cfg.nbins
    if x.shape[1] != expected_dim:
        raise ValueError(
            f"{cfg.name}: cache feature dim {x.shape[1]} != expected {expected_dim} "
            f"({cfg.nbins} bins × {cfg.n_scales} scales × {cfg.l1_nbins} bins)"
        )

    observed_l1 = compute_observed_l1(cfg, noise_seed=noise_seed, sigma_e=sigma_e, galaxy_density=galaxy_density)
    if observed_l1.shape[0] != expected_dim:
        raise ValueError(f"{cfg.name}: observed dim {observed_l1.shape[0]} != expected {expected_dim}")

    rep_indices = choose_representative_indices(theta)

    out_files = [
        out_dir / f"{cfg.name}_{split}_profiles_by_scale.png",
        out_dir / f"{cfg.name}_{split}_scale_totals_lines.png",
        out_dir / f"{cfg.name}_{split}_scale_totals_vs_cosmo.png",
    ]

    make_profiles_figure(cfg, x, theta, observed_l1, rep_indices, out_files[0])
    make_scale_totals_line_figure(cfg, x, theta, observed_l1, rep_indices, out_files[1])
    make_scale_scatter_figure(cfg, x, theta, observed_l1, rep_indices, out_files[2], scatter_n=scatter_n)

    del x, theta, observed_l1
    gc.collect()
    return out_files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot L1 datavectors by scale and cosmology from cached artifacts.")
    p.add_argument(
        "--variants",
        type=str,
        default="l1_tomo4_20deg160,l1_bin3_20deg160",
        help="Comma-separated cache variant names.",
    )
    p.add_argument(
        "--cache-roots",
        type=str,
        default="scripts/sbi/systematic_runs_l1_rerun_proper/cache,scripts/sbi/systematic_runs_24/cache",
        help="Comma-separated cache root directories (searched in order).",
    )
    p.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Cache split to visualize.")
    p.add_argument("--out-dir", type=Path, default=Path("scripts/sbi/diagnostics/l1_datavectors"))
    p.add_argument("--noise-seed", type=int, default=41, help="Seed for observed-map shape noise.")
    p.add_argument("--sigma-e", type=float, default=0.26)
    p.add_argument("--galaxy-density", type=float, default=30.0 / 4.0)
    p.add_argument("--scatter-n", type=int, default=12000, help="Subsample size for cosmology scatter plots.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    cache_roots = [Path(p.strip()) for p in args.cache_roots.split(",") if p.strip()]

    for root in cache_roots:
        if not root.exists():
            raise FileNotFoundError(f"Cache root does not exist: {root}")

    if not FIDUCIAL_META.exists() or not FIDUCIAL_MAP.exists():
        raise FileNotFoundError(
            "Fiducial files missing. Expected: "
            f"{FIDUCIAL_META} and {FIDUCIAL_MAP}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    produced: list[Path] = []
    for variant in variants:
        cfg = load_variant_config(variant, cache_roots)
        produced.extend(
            generate_variant_plots(
                cfg,
                split=args.split,
                out_dir=args.out_dir,
                noise_seed=args.noise_seed,
                sigma_e=args.sigma_e,
                galaxy_density=args.galaxy_density,
                scatter_n=args.scatter_n,
            )
        )

    print("Produced files:")
    for f in produced:
        print(f"- {f}")


if __name__ == "__main__":
    main()
