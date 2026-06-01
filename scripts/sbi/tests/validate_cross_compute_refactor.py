#!/usr/bin/env python
"""Bit-exact validation of the refactored cross-map compute.

Recomputes one (cosmo, perm) realization via the extracted shared function
`compute_cross_patches` and compares to the existing `.npz` cache file. Must be
max abs diff 0.0 — that proves (a) the refactor preserved behaviour and (b) the
function the TFDS builder will reuse reproduces the cache bit-for-bit.

CPU-only (~40s for the 4 SHTs + 10 iSHTs). Run:
  conda run -n jaxili python scripts/sbi/tests/validate_cross_compute_refactor.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

_HERE = Path(__file__).resolve().parent
_SBI = _HERE.parent
if str(_SBI) not in sys.path:
    sys.path.insert(0, str(_SBI))

import build_full_sphere_cross_cache as B  # noqa: E402
from tf_dataset_nbody_tomo import _build_non_overlapping_centers  # noqa: E402

CACHE = (_SBI / "results" / "exploratory" / "cross_maps_campaign"
         / "full_sphere_cache_grid")
META = Path("/home/tersenov/CosmoGridV1/CosmoGridV1_metainfo.h5")
ROOT = Path("/home/tersenov/CosmoGridV1")
REGIME, COSMO_ID, PERM = "nobnt", "cosmo_000002", 0


def main() -> int:
    ref_path = CACHE / REGIME / "train" / f"{COSMO_ID}_perm{PERM}.npz"
    if not ref_path.exists():
        print(f"reference .npz not found: {ref_path}")
        return 2
    ref = np.load(ref_path, allow_pickle=True)
    ref_patches = ref["patches"]
    ref_cosmo_idx = int(ref["cosmo_idx"])
    ref_seed = int(ref["noise_seed"])
    print(f"reference: {ref_path.name}  patches={ref_patches.shape} {ref_patches.dtype} "
          f"cosmo_idx={ref_cosmo_idx} seed={ref_seed}")

    # Match the cache's build config exactly (verified from its manifest.json).
    cfg = B.BuildConfig(
        out_dir=Path("/tmp/_unused"), map_label="nobaryons",
        nside=512, lmax=1024, sigma_e=0.26, galaxy_density=10.0,
        field_size=20.0, field_npix=160, reso_arcmin=20.0 * 60.0 / 160.0,
        n_centers=48, min_separation_deg=28.5, center_nside=32,
        noise_seed_base=12345, regimes=(REGIME,),
        snapshot_cosmo_id="cosmo_fiducial", snapshot_perm=0,
    )

    entries = B.enumerate_cosmologies(META, ROOT, subset="grid", cosmo_limit=5)
    match = [e for e in entries if e.cosmo_id == COSMO_ID]
    if not match:
        print(f"{COSMO_ID} not in first 5 grid cosmologies: {[e.cosmo_id for e in entries]}")
        return 2
    entry = match[0]
    assert entry.cosmo_idx == ref_cosmo_idx, (entry.cosmo_idx, ref_cosmo_idx)
    assert B._noise_seed(entry.cosmo_idx, PERM, cfg.noise_seed_base) == ref_seed

    h5_path = entry.realization_dir / f"perm_000{PERM}" / f"projected_probes_maps_{cfg.map_label}512.h5"
    with h5py.File(h5_path, "r") as f:
        kg = f["kg"]
        noiseless = np.stack(
            [np.asarray(kg[f"stage3_lensing{b}"], dtype=np.float64) for b in (1, 2, 3, 4)],
            axis=0,
        )

    centers = _build_non_overlapping_centers(
        n_centers=cfg.n_centers, min_separation_deg=cfg.min_separation_deg,
        center_nside=cfg.center_nside,
    )
    patches = B.compute_cross_patches(noiseless, entry.cosmo_idx, PERM, REGIME, centers, cfg)

    same_shape = patches.shape == ref_patches.shape
    max_abs = float(np.max(np.abs(patches - ref_patches))) if same_shape else float("inf")
    print(f"recomputed: patches={patches.shape} {patches.dtype}")
    print(f"shape match: {same_shape}")
    print(f"MAX ABS DIFF vs cache: {max_abs:.3e}")
    ok = same_shape and max_abs == 0.0
    print("RESULT:", "PASS (bit-exact)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
