#!/usr/bin/env python
"""Precompute full-sphere harmonic-space cross-maps as patch tensors.

For each (cosmology, realization) of CosmoGridV1, this script:

  1. Reads the 4 noiseless tomographic kappa maps from
     `kg/stage3_lensing{1..4}` of the per-realization HDF5 file.
  2. Adds shape noise on the full HEALPix sphere (per-pixel iid Gaussian,
     std = sigma_e / sqrt(galaxy_density * pixel_area_arcmin^2)).
  3. Computes spherical harmonic transforms (one per bin).
  4. For the BNT regime, applies the 4x4 BNT matrix in `a_lm` space
     (linear combinations of bins commute with SHT).
  5. Computes the 6 cross-pair `a_lm` products element-wise (Zurcher et al.
     2022 ad-hoc element-wise convention; not a true spherical convolution).
  6. Inverse-SHTs both the 4 auto and 6 cross alm arrays back to HEALPix
     real-space maps.
  7. Extracts 48 gnomonic 20deg/160px patches at the same deterministic
     patch centers used by `tf_dataset_nbody_tomo.py` for the
     `grid_20deg_160px_nonoverlap48` config.
  8. Demeans each patch per channel (matches `--zero-mean-maps`).
  9. Saves the resulting `(48, 160, 160, 10)` float32 tensor as a
     compressed .npz, plus theta and provenance metadata.

For the snapshot cosmology (default `cosmo_fiducial`, perm 0), the full
HEALPix arrays for one regime are also dumped under `_snapshot/` so the
diagnose script can render mollweide views without re-doing SHTs.

Train/val split mirrors the flat-sky TFDS builder exactly (cosmo_idx 1-899
=> train, 900-1299 => val, all from `parameters/grid`).

Resource model: each (cosmo, perm) job costs ~30-50s of healpy SHT (4
forward + 6 inverse at lmax=1024). Both regimes share the same forward
SHT and noisy maps, so building both is ~1.6x the cost of one regime.
With Pool(50) the fiducial subset (17 cosmos x 7 perms x 2 regimes =
~120 jobs) finishes in 3-5 minutes; the full grid (2500 x 7 x 2 = 35k
jobs) finishes in 6-10h and lands ~600GB on disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool, get_context
from pathlib import Path

import h5py
import healpy as hp
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from bnt_utils import BNT_MATRIX, BNT_MATRIX_VERSION  # noqa: E402
from tf_dataset_nbody_tomo import _build_non_overlapping_centers  # noqa: E402


CROSS_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (0, 3),
    (1, 2), (1, 3),
    (2, 3),
)
N_AUTO = 4
N_CROSS = len(CROSS_PAIRS)
N_CHANNELS = N_AUTO + N_CROSS
PARAM_FIELDS = ("Om", "s8", "w0", "H0", "ns", "Ob")


# -----------------------------------------------------------------------------
# Cosmology enumeration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class CosmoEntry:
    cosmo_idx: int        # row index in parameters/{subset}
    cosmo_id: str         # e.g. "cosmo_000002" or "cosmo_fiducial"
    subset: str           # "grid" or "fiducial"
    realization_dir: Path # absolute path to perm_NNNN parent
    theta: np.ndarray     # (6,) float64 [Om, s8, w0, h0, ns, Ob]; H0 left as is
    split: str            # "train" or "val" (matches flat-sky TFDS split)

    def perm_dir(self, perm: int, root: Path) -> Path:
        return self.realization_dir / f"perm_{perm:04d}"


def _remap_path_par(path_par: str) -> Path:
    """Mirror the path remap used by tf_dataset_nbody_tomo._generate_examples."""
    s = path_par.replace("CosmoGrid", "CosmoGridV1") \
                .replace("raw", "stage3_forecast") \
                .replace("grid", "new_grid")
    return Path("/home/tersenov") / s.lstrip("/")


def _theta_from_row(row) -> np.ndarray:
    return np.array([float(row[k]) for k in PARAM_FIELDS], dtype=np.float64)


def enumerate_cosmologies(
    meta_path: Path,
    cosmogrid_root: Path,
    subset: str,
    cosmo_limit: int | None,
) -> list[CosmoEntry]:
    entries: list[CosmoEntry] = []
    with h5py.File(meta_path, "r") as f:
        if subset not in ("fiducial", "grid", "all"):
            raise ValueError(f"--cosmo-subset={subset} not in {{fiducial,grid,all}}")
        groups = ("fiducial", "grid") if subset == "all" else (subset,)
        for grp_name in groups:
            ds = f["parameters"][grp_name]
            n = len(ds["path_par"])
            # Mirror flat-sky split for the grid subset; fiducial is val-only by
            # convention (training never sees fiducial — used for the obs map).
            for i in range(n):
                path_par = ds["path_par"][i].decode("utf-8")
                cosmo_id = path_par.rstrip("/").split("/")[-1]
                realization_dir = _remap_path_par(path_par)
                theta = _theta_from_row(ds[i])
                if grp_name == "grid":
                    split = "train" if 1 <= i < 900 else "val" if 900 <= i < 1300 else "skip"
                else:
                    split = "obs"  # fiducial is reserved for observation
                if split == "skip":
                    continue
                entries.append(CosmoEntry(
                    cosmo_idx=i,
                    cosmo_id=cosmo_id,
                    subset=grp_name,
                    realization_dir=realization_dir,
                    theta=theta,
                    split=split,
                ))
    if cosmo_limit is not None:
        entries = entries[:cosmo_limit]
    return entries


# -----------------------------------------------------------------------------
# Worker: per (cosmo, perm) build
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class BuildConfig:
    out_dir: Path
    map_label: str
    nside: int
    lmax: int
    sigma_e: float
    galaxy_density: float
    field_size: float
    field_npix: int
    reso_arcmin: float
    n_centers: int
    min_separation_deg: float
    center_nside: int
    noise_seed_base: int
    regimes: tuple[str, ...]
    snapshot_cosmo_id: str
    snapshot_perm: int
    max_abs_lat: float | None = None


def _per_pixel_noise_std(sigma_e: float, galaxy_density: float, nside: int) -> float:
    pix_area_arcmin2 = hp.nside2pixarea(nside, degrees=True) * 3600.0
    return float(sigma_e / np.sqrt(galaxy_density * pix_area_arcmin2))


def _patch_one_realization(
    full_maps: list[np.ndarray],
    centers: np.ndarray,
    nside: int,
    field_npix: int,
    reso_arcmin: float,
) -> np.ndarray:
    """Project a list of n_chan HEALPix maps onto `n_centers` gnomonic patches.

    Returns float32 array of shape (n_centers, field_npix, field_npix, n_chan).
    """
    n_chan = len(full_maps)
    n_centers = centers.shape[0]
    patches = np.empty((n_centers, field_npix, field_npix, n_chan), dtype=np.float32)
    vec2pix = partial(hp.vec2pix, nside)
    for k in range(n_centers):
        lon, lat = float(centers[k, 0]), float(centers[k, 1])
        proj = hp.projector.GnomonicProj(
            rot=[lon, lat, 0],
            xsize=field_npix,
            ysize=field_npix,
            reso=reso_arcmin,
        )
        for c, m in enumerate(full_maps):
            patches[k, :, :, c] = proj.projmap(m, vec2pix_func=vec2pix).astype(np.float32)
    # Per-(patch, channel) demean to mirror --zero-mean-maps in the L1 pipeline.
    patches -= patches.mean(axis=(1, 2), keepdims=True)
    return patches


# -----------------------------------------------------------------------------
# Shared compute (single source of truth for the .npz cache AND the TFDS builder)
# -----------------------------------------------------------------------------

def _noise_seed(cosmo_idx: int, perm: int, noise_seed_base: int) -> int:
    """Deterministic per-(cosmo, perm) shape-noise seed."""
    return int(noise_seed_base) + 100 * int(cosmo_idx) + int(perm)


def compute_noisy_alms(
    noiseless: np.ndarray, cosmo_idx: int, perm: int, cfg: BuildConfig
) -> np.ndarray:
    """Steps 2-3: add per-pixel iid sphere noise to the 4 auto bins, then SHT each.

    `noiseless` is (N_AUTO, npix) float64. Returns (N_AUTO, n_lm) complex128 alms,
    deterministic in (cosmo_idx, perm).
    """
    seed = _noise_seed(cosmo_idx, perm, cfg.noise_seed_base)
    rng = np.random.default_rng(seed)
    noise_std = _per_pixel_noise_std(cfg.sigma_e, cfg.galaxy_density, cfg.nside)
    noisy = noiseless + rng.normal(0.0, noise_std, size=noiseless.shape)
    return np.stack(
        [hp.map2alm(noisy[b], lmax=cfg.lmax, iter=0) for b in range(N_AUTO)],
        axis=0,
    )


def cross_patches_from_alms(
    alms: np.ndarray, regime: str, centers: np.ndarray, cfg: BuildConfig
) -> tuple[np.ndarray, list, list]:
    """Steps 5-8 for one regime: optional BNT on the alm bin-axis -> element-wise
    alm cross products -> iSHT auto+cross -> patch extraction + per-patch demean.

    Returns (patches[n_centers, H, W, N_CHANNELS] f32, full_auto, full_cross).
    """
    regime_alms = alms.copy()
    if regime == "bnt":
        # BNT is a 4x4 linear combination on the bin axis; it commutes with SHT,
        # so applying BNT_MATRIX directly to alms is equivalent to applying it on
        # the maps and re-doing the SHT.
        regime_alms = (BNT_MATRIX.astype(np.float64) @ regime_alms).astype(
            regime_alms.dtype
        )
    elif regime != "nobnt":
        raise ValueError(f"Unknown regime '{regime}'")

    cross_alms = [regime_alms[i] * regime_alms[j] for (i, j) in CROSS_PAIRS]
    full_auto = [hp.alm2map(regime_alms[b], nside=cfg.nside, lmax=cfg.lmax)
                 for b in range(N_AUTO)]
    full_cross = [hp.alm2map(a, nside=cfg.nside, lmax=cfg.lmax) for a in cross_alms]
    patches = _patch_one_realization(
        full_auto + full_cross,
        centers=centers,
        nside=cfg.nside,
        field_npix=cfg.field_npix,
        reso_arcmin=cfg.reso_arcmin,
    )
    return patches, full_auto, full_cross


def compute_cross_patches(
    noiseless: np.ndarray, cosmo_idx: int, perm: int, regime: str,
    centers: np.ndarray, cfg: BuildConfig
) -> np.ndarray:
    """Full per-(cosmo, perm, regime) compute: noiseless 4-bin maps -> 10-channel
    patches. The one entry point the TFDS builder reuses to stay bit-identical to
    the `.npz` cache.
    """
    alms = compute_noisy_alms(noiseless, cosmo_idx, perm, cfg)
    patches, _, _ = cross_patches_from_alms(alms, regime, centers, cfg)
    return patches


def _worker(job: tuple[CosmoEntry, int], cfg: BuildConfig) -> tuple[str, dict]:
    """Build one (cosmo, perm) job. Returns (status, info dict for manifest)."""
    # Pin healpy / OpenMP to 1 thread per worker process to avoid oversubscription
    # under multiprocessing.Pool. healpy honours OMP_NUM_THREADS at runtime.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    entry, perm = job
    perm_dir = entry.realization_dir / f"perm_{perm:04d}"
    h5_path = perm_dir / f"projected_probes_maps_{cfg.map_label}512.h5"
    if not h5_path.exists():
        return ("missing", {"cosmo_id": entry.cosmo_id, "perm": perm,
                            "h5_path": str(h5_path)})

    centers = _build_non_overlapping_centers(
        n_centers=cfg.n_centers,
        min_separation_deg=cfg.min_separation_deg,
        center_nside=cfg.center_nside,
        max_abs_lat=cfg.max_abs_lat,
    )
    info: dict = {
        "cosmo_id": entry.cosmo_id,
        "cosmo_idx": entry.cosmo_idx,
        "subset": entry.subset,
        "split": entry.split,
        "perm": perm,
        "h5_path": str(h5_path),
        "regimes": list(cfg.regimes),
        "files": {},
    }

    # Skip if all required outputs already exist.
    pending_regimes = []
    for regime in cfg.regimes:
        out_path = cfg.out_dir / regime / entry.split / f"{entry.cosmo_id}_perm{perm}.npz"
        if out_path.exists():
            info["files"][regime] = {"path": str(out_path), "skipped": True}
            continue
        pending_regimes.append(regime)
    if not pending_regimes:
        return ("skip-existing", info)

    # 1. Load 4 noiseless tomographic kappa maps.
    with h5py.File(h5_path, "r") as f:
        kg = f["kg"]
        noiseless = np.stack(
            [np.asarray(kg[f"stage3_lensing{b}"], dtype=np.float64) for b in (1, 2, 3, 4)],
            axis=0,
        )  # (4, npix)

    # 2-3. Per-pixel sphere noise + SHT of the 4 auto bins (shared across regimes).
    seed = _noise_seed(entry.cosmo_idx, perm, cfg.noise_seed_base)
    alms = compute_noisy_alms(noiseless, entry.cosmo_idx, perm, cfg)

    snapshot_dir = cfg.out_dir / "_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    for regime in pending_regimes:
        # 5-8. cross products -> iSHT -> patch extraction + demean (shared compute).
        patches, full_auto, full_cross = cross_patches_from_alms(
            alms, regime, centers, cfg
        )

        out_path = cfg.out_dir / regime / entry.split / f"{entry.cosmo_id}_perm{perm}.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            patches=patches,
            patch_centers=centers.astype(np.float32),
            theta=entry.theta.astype(np.float64),
            cosmo_id=entry.cosmo_id,
            cosmo_idx=np.int64(entry.cosmo_idx),
            perm=np.int64(perm),
            regime=regime,
            split=entry.split,
            noise_seed=np.int64(seed),
            sigma_e=np.float64(cfg.sigma_e),
            galaxy_density=np.float64(cfg.galaxy_density),
            nside=np.int64(cfg.nside),
            lmax=np.int64(cfg.lmax),
            field_size=np.float64(cfg.field_size),
            field_npix=np.int64(cfg.field_npix),
            reso_arcmin=np.float64(cfg.reso_arcmin),
            map_label=cfg.map_label,
            bnt_matrix_version=BNT_MATRIX_VERSION if regime == "bnt" else "none",
        )
        info["files"][regime] = {"path": str(out_path), "skipped": False}

        if entry.cosmo_id == cfg.snapshot_cosmo_id and perm == cfg.snapshot_perm:
            snap_path = snapshot_dir / f"fullsphere_{regime}_{entry.cosmo_id}_perm{perm}.npz"
            np.savez_compressed(
                snap_path,
                full_auto=np.stack(full_auto, axis=0).astype(np.float32),
                full_cross=np.stack(full_cross, axis=0).astype(np.float32),
                cosmo_id=entry.cosmo_id,
                perm=np.int64(perm),
                regime=regime,
                nside=np.int64(cfg.nside),
                lmax=np.int64(cfg.lmax),
                noise_seed=np.int64(seed),
            )
            info.setdefault("snapshots", {})[regime] = str(snap_path)

    return ("ok", info)


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def _hash_args(args: argparse.Namespace) -> str:
    payload = json.dumps(vars(args), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cosmo-meta", type=Path,
                   default=Path("/home/tersenov/CosmoGridV1/CosmoGridV1_metainfo.h5"))
    p.add_argument("--cosmogrid-root", type=Path,
                   default=Path("/home/tersenov/CosmoGridV1"))
    p.add_argument("--cosmo-subset", type=str, default="fiducial",
                   choices=["fiducial", "grid", "all"])
    p.add_argument("--cosmo-limit", type=int, default=None)
    p.add_argument("--cosmo-id", type=str, default=None,
                   help="If set, build ONLY the cosmology with this id (e.g. "
                        "cosmo_fiducial). Filters the enumerated subset.")
    p.add_argument("--realizations", type=str, default="0,1,2,3,4,5,6")
    p.add_argument("--regime", type=str, default="both",
                   choices=["bnt", "nobnt", "both"])
    p.add_argument("--map-label", type=str, default="nobaryons",
                   choices=["nobaryons", "baryonified"])
    p.add_argument("--nside-source", type=int, default=512)
    p.add_argument("--lmax", type=int, default=1024)
    p.add_argument("--sigma-e", type=float, default=0.26)
    p.add_argument("--galaxy-density", type=float, default=10.0)
    p.add_argument("--noise-seed-base", type=int, default=12345)
    p.add_argument("--field-size", type=float, default=20.0)
    p.add_argument("--field-npix", type=int, default=160)
    p.add_argument("--reso-arcmin", type=float, default=None,
                   help="Defaults to field_size*60/field_npix (matches flat-sky baseline).")
    p.add_argument("--n-centers", type=int, default=48)
    p.add_argument("--min-separation-deg", type=float, default=28.5)
    p.add_argument("--max-abs-lat", type=float, default=None,
                   help="Exclude patch centers with |lat|>=this (deg) BEFORE selection "
                        "(10deg campaign: 75 -> no near-pole patches). None = 20deg behavior.")
    p.add_argument("--center-nside", type=int, default=32)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--num-workers", type=int, default=50)
    p.add_argument("--snapshot-cosmo-id", type=str, default="cosmo_fiducial")
    p.add_argument("--snapshot-perm", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    realizations = tuple(int(p) for p in args.realizations.split(",") if p.strip() != "")
    regimes: tuple[str, ...]
    if args.regime == "both":
        regimes = ("bnt", "nobnt")
    else:
        regimes = (args.regime,)

    reso = (args.reso_arcmin if args.reso_arcmin is not None
            else args.field_size * 60.0 / args.field_npix)

    cfg = BuildConfig(
        out_dir=args.out_dir,
        map_label=args.map_label,
        nside=args.nside_source,
        lmax=args.lmax,
        sigma_e=args.sigma_e,
        galaxy_density=args.galaxy_density,
        field_size=args.field_size,
        field_npix=args.field_npix,
        reso_arcmin=reso,
        n_centers=args.n_centers,
        min_separation_deg=args.min_separation_deg,
        center_nside=args.center_nside,
        noise_seed_base=args.noise_seed_base,
        regimes=regimes,
        snapshot_cosmo_id=args.snapshot_cosmo_id,
        snapshot_perm=args.snapshot_perm,
        max_abs_lat=args.max_abs_lat,
    )

    cosmologies = enumerate_cosmologies(
        meta_path=args.cosmo_meta,
        cosmogrid_root=args.cosmogrid_root,
        subset=args.cosmo_subset,
        cosmo_limit=args.cosmo_limit,
    )
    if args.cosmo_id is not None:
        cosmologies = [c for c in cosmologies if c.cosmo_id == args.cosmo_id]
        if not cosmologies:
            raise SystemExit(f"--cosmo-id={args.cosmo_id} matched no cosmology in "
                             f"subset={args.cosmo_subset}.")
    jobs: list[tuple[CosmoEntry, int]] = [
        (entry, perm) for entry in cosmologies for perm in realizations
    ]

    print(f"Cosmologies enumerated: {len(cosmologies)} "
          f"(subsets={set(c.subset for c in cosmologies)}, "
          f"splits={set(c.split for c in cosmologies)})")
    print(f"Realizations per cosmology: {realizations}")
    print(f"Regimes built: {regimes}")
    print(f"Total jobs: {len(jobs)}")
    print(f"Output directory: {args.out_dir}")
    print(f"Workers: {args.num_workers}")
    print(f"Per-pixel noise std @ nside={args.nside_source}: "
          f"{_per_pixel_noise_std(args.sigma_e, args.galaxy_density, args.nside_source):.6f}")

    t0 = time.time()
    results: list[dict] = []
    n_ok = n_skip = n_missing = 0

    if args.num_workers <= 1:
        # Serial path (smoke testing).
        for j, job in enumerate(jobs):
            status, info = _worker(job, cfg)
            results.append({"status": status, **info})
            n_ok += int(status == "ok")
            n_skip += int(status == "skip-existing")
            n_missing += int(status == "missing")
            if not args.quiet:
                elapsed = time.time() - t0
                print(f"  [{j+1}/{len(jobs)}] {status} {info.get('cosmo_id')} "
                      f"perm{info.get('perm')} (elapsed {elapsed:.1f}s)")
    else:
        ctx = get_context("spawn")
        # Pre-import healpy / numpy in workers via initializer to amortize startup.
        with ctx.Pool(processes=args.num_workers, initializer=_init_worker) as pool:
            worker_fn = partial(_worker, cfg=cfg)
            for j, (status, info) in enumerate(pool.imap_unordered(worker_fn, jobs, chunksize=1)):
                results.append({"status": status, **info})
                n_ok += int(status == "ok")
                n_skip += int(status == "skip-existing")
                n_missing += int(status == "missing")
                if not args.quiet and (j + 1) % 25 == 0:
                    elapsed = time.time() - t0
                    rate = (j + 1) / max(elapsed, 1.0)
                    eta = (len(jobs) - (j + 1)) / max(rate, 1e-6)
                    print(f"  [{j+1}/{len(jobs)}] elapsed {elapsed:.1f}s "
                          f"({rate:.2f} job/s, ETA {eta:.0f}s)  ok={n_ok} skip={n_skip} missing={n_missing}")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s. ok={n_ok} skip={n_skip} missing={n_missing}")

    # Write the manifest.
    manifest_path = args.out_dir / "manifest.json"
    manifest = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "args_sha256": _hash_args(args),
        "bnt_matrix_version": BNT_MATRIX_VERSION,
        "cross_pairs": [list(p) for p in CROSS_PAIRS],
        "n_channels": N_CHANNELS,
        "channel_layout": (
            [f"auto_{b+1}" for b in range(N_AUTO)]
            + [f"cross_{i+1}{j+1}" for (i, j) in CROSS_PAIRS]
        ),
        "n_jobs": len(jobs),
        "n_ok": n_ok,
        "n_skip_existing": n_skip,
        "n_missing": n_missing,
        "elapsed_sec": elapsed,
        "results": results,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {manifest_path}")


def _init_worker() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


if __name__ == "__main__":
    main()
