#!/usr/bin/env python3
"""Compress the 9000 fiducial observations through a trained CNN (ResNet-18 VMIM) arm.

Reconstruction of build_fiducial_summaries_cnn.py (destroyed: 12695 bytes of zeros).
Produces <arm>/fiducial_summaries.npz with the SAME keys/dtypes/ordering as the L1 arms'
(S (9000,10) float32, perm int32, patch int32), so population_sweep.py consumes it unchanged.

WHY THIS IS RECONSTRUCTIBLE EXACTLY
-----------------------------------
The obs path in npe_cnn_nbody_tomo.py (flat_local route) is three steps:

    load_observed_from_harmonic_cache(cache, regime, cosmo_id, perm, patch_idx,
                                      channel_scale=None,            # RAW autos
                                      channel_slice=slice(0, nbins))
    -> flat_cross_transform(m)          # make_flat_cross_transform(op, RMS, roll, bnt)
    -> compressor_eval.apply(params, state, None, m.reshape(1,H,W,C))

Every input to those three steps is persisted by the training run:
  * the frozen per-channel RMS  -> cache/cnn_cache_meta.npz['info_channel_scale']
    (the driver added this key precisely because "previously it lived only in stdout logs")
  * op / roll_frac / bnt / arch / dim / head_width / v2 / checkpoint path + sha256
    -> the rest of cnn_cache_meta.npz
So nothing here is guessed. We import the driver's own functions rather than reimplementing.

THE G1 GATE (do not remove)
---------------------------
The driver also persisted the compressed OBSERVED vector for (cosmo_fiducial, perm 0,
patch 0) to cache/cnn_obs.npz. This script recomputes that one vector through its own
code path and requires max|delta| <= --g1-tol (9e-4, the tolerance the arch-sweep handoff
records). If the gate fails, every one of the 9000 summaries is suspect and we stop:
a silent mismatch here is exactly the failure mode that made the L1 driver's own
--fiducial-summaries-out unusable (it binned with scalar SNR ranges while obs used
per-channel ones, and only its gate caught it).

Also re-checks that the compressor checkpoint on disk still hashes to what the cache
metadata recorded, so an arm can never be evaluated with a different seed's weights.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SRC = os.environ.get("CNN_SBI_SRC",
                     "/lustre/fswork/projects/rech/prk/ulx34io/recovery/rescued_scripts")
sys.path.insert(0, SRC)


def _meta_get(meta, key, cast=None, default=None):
    if key not in meta.files:
        if default is None:
            raise SystemExit(f"cnn_cache_meta.npz is missing required key {key!r}")
        return default
    v = meta[key]
    v = v.item() if getattr(v, "ndim", 0) == 0 else v
    return cast(v) if cast is not None else v


def _parse_int_tuple(spec) -> tuple[int, ...]:
    s = str(spec).strip()
    if not s:
        return ()
    return tuple(int(t) for t in s.replace("(", "").replace(")", "").split(",") if t.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", required=True,
                    help="CNN arm dir (contains cache/cnn_cache_meta.npz and cache/cnn_obs.npz)")
    ap.add_argument("--obs-cache", required=True,
                    help="fiducial obs cache ROOT (the loader appends <regime>/obs itself)")
    ap.add_argument("--out", default=None, help="default: <arm-dir>/fiducial_summaries.npz")
    ap.add_argument("--max-perm", type=int, default=50, help="perms 0..max_perm-1 (9000 obs)")
    ap.add_argument("--cosmo-id", default="cosmo_fiducial")
    ap.add_argument("--g1-tol", type=float, default=9e-4)
    ap.add_argument("--batch-size", type=int, default=180, help="patches per compressor call")
    ap.add_argument("--skip-hash-check", action="store_true",
                    help="only for a deliberately re-pointed checkpoint; normally leave off")
    ap.add_argument("--cuda-visible-devices", default="0")
    a = ap.parse_args()

    arm = Path(a.arm_dir)
    cache = arm / "cache" if (arm / "cache" / "cnn_cache_meta.npz").exists() else arm
    meta_p, obs_p = cache / "cnn_cache_meta.npz", cache / "cnn_obs.npz"
    for p in (meta_p, obs_p):
        if not p.exists():
            raise SystemExit(f"missing {p} -- did the arm finish --exit-after-compress?")
    out_p = Path(a.out) if a.out else arm / "fiducial_summaries.npz"

    meta = np.load(meta_p, allow_pickle=True)
    route = _meta_get(meta, "cnn_map_route", str)
    if route != "flat_local":
        raise SystemExit(f"this script only reconstructs the flat_local route (got {route!r})")

    regime = _meta_get(meta, "harmonic_regime", str)
    op = _meta_get(meta, "cross_op", str)
    nbins = _meta_get(meta, "nbins", int)
    n_ch = _meta_get(meta, "cnn_input_channels", int)
    roll = _meta_get(meta, "flatsky_roll_frac", float)
    bnt = bool(_meta_get(meta, "flatsky_bnt", int, default=0))   # key present only when BNT on
    arch = _meta_get(meta, "compressor_arch", str)
    dim = _meta_get(meta, "compressor_dim", int)
    scale = np.asarray(_meta_get(meta, "info_channel_scale"), dtype=np.float32)
    params_path = _meta_get(meta, "compressor_params_path", str)
    params_sha = _meta_get(meta, "compressor_params_sha256", str)
    state_path = _meta_get(meta, "compressor_state_path", str)
    state_sha = _meta_get(meta, "compressor_state_sha256", str)

    if scale.size != n_ch:
        raise SystemExit(f"info_channel_scale has {scale.size} entries, expected {n_ch}")

    print(f"=== CNN fiducial summaries: {arm.name} ===")
    print(f"  route={route} regime={regime} op={op} bnt={bnt} nbins={nbins} -> {n_ch} ch")
    print(f"  arch={arch} dim={dim} roll_frac={roll}")
    print(f"  checkpoint policy (effective) = "
          f"{_meta_get(meta, 'info_checkpoint_policy_effective', str, default='?')}")
    print(f"  frozen per-channel RMS = {scale}")

    from npe_cnn_nbody_tomo import (setup_environment, build_compressors,
                                    load_compressor_params, load_observed_from_harmonic_cache,
                                    make_flat_cross_transform, file_sha256,
                                    _assert_zero_mean_patches)
    from flatsky_cross import n_output_channels

    if n_output_channels(nbins, op) != n_ch:
        raise SystemExit(f"n_output_channels({nbins},{op})={n_output_channels(nbins, op)} "
                         f"!= cnn_input_channels={n_ch}")

    # The checkpoint must be byte-identical to the one the cache was fingerprinted with,
    # otherwise this arm's train summaries and its obs summaries come from different weights.
    if not a.skip_hash_check:
        for label, p, want in (("params", params_path, params_sha),
                               ("state", state_path, state_sha)):
            if not want:
                raise SystemExit(f"cache metadata has no {label} sha256 to verify against")
            got = file_sha256(Path(p))
            if got != want:
                raise SystemExit(f"{label} checkpoint hash mismatch\n  {p}\n"
                                 f"  meta={want}\n  disk={got}")
            print(f"  {label} sha256 OK ({want[:16]}...)")

    setup_environment(a.cuda_visible_devices)
    import jax, jax.numpy as jnp

    _, compressor_eval = build_compressors(
        dim=dim, arch=arch,
        conv_channels=_parse_int_tuple(_meta_get(meta, "compressor_conv_channels", str)),
        dense_width=_meta_get(meta, "compressor_dense_width", int),
        pool_window=_meta_get(meta, "compressor_pool_window", int),
        pool_stride=_meta_get(meta, "compressor_pool_stride", int),
        resnet_small_channels=_parse_int_tuple(_meta_get(meta, "resnet_small_channels", str)),
        resnet_small_blocks=_parse_int_tuple(_meta_get(meta, "resnet_small_blocks", str)),
        resnet_head_width=_meta_get(meta, "resnet_head_width", int),
        resnet_v2=bool(_meta_get(meta, "resnet_v2", int)),
    )
    comp_params, comp_state = load_compressor_params(params_path, state_path)
    transform = make_flat_cross_transform(op, scale, roll, bnt=bnt)

    @jax.jit
    def _compress(built):                       # (B,H,W,C) -> (B,dim)
        y, _ = compressor_eval.apply(comp_params, comp_state, None, built)
        return y

    obs_root = Path(a.obs_cache).resolve()
    ch_slice = slice(0, nbins)

    # ---------------- G1 gate ----------------
    # Recompute the ONE observation the driver already compressed and persisted.
    print("\n######## G1 GATE ########")
    ref = np.load(obs_p)["x"].astype(np.float64).squeeze()
    m0, _cosmo, truth = load_observed_from_harmonic_cache(
        cache_dir=obs_root, regime=regime, cosmo_id=a.cosmo_id, perm=0, patch_idx=0,
        meta_path=None, channel_scale=None, channel_slice=ch_slice,
    )
    built0 = np.asarray(transform(m0), dtype=np.float32)          # unbatched, as the driver did
    got = np.asarray(_compress(built0.reshape(1, *built0.shape))).squeeze().astype(np.float64)
    delta = float(np.abs(got - ref).max())
    print(f"  reference (cnn_obs.npz) = {ref}")
    print(f"  recomputed              = {got}")
    print(f"  max|delta| = {delta:.3e}   tol = {a.g1_tol:.1e}")
    if not np.isfinite(delta) or delta > a.g1_tol:
        raise SystemExit(
            f"G1 GATE FAILED: max|delta|={delta:.3e} > {a.g1_tol:.1e}. The obs path here does "
            "not reproduce the driver's own compressed observation, so all 9000 summaries "
            "would be wrong. Do not proceed -- diagnose (channel scale? bnt? checkpoint?)."
        )
    print("  G1 PASS")

    # ---------------- bulk ----------------
    print(f"\n######## COMPRESS {a.max_perm} perms x 180 patches ########")
    t0 = time.time()
    S_parts, perm_parts, patch_parts = [], [], []
    n_patches_ref = None
    for perm in range(a.max_perm):
        npz = obs_root / regime / "obs" / f"{a.cosmo_id}_perm{perm}.npz"
        if not npz.exists():
            raise SystemExit(f"missing obs file {npz}")
        with np.load(npz, allow_pickle=False) as d:
            patches = np.asarray(d["patches"], dtype=np.float32)
        _assert_zero_mean_patches(patches, str(npz))              # same guard as the driver
        autos = patches[..., ch_slice]
        n_p = autos.shape[0]
        if n_patches_ref is None:
            n_patches_ref = n_p
            print(f"  {n_p} patches/perm, autos {autos.shape[1:]} -> built {n_ch} ch")
        elif n_p != n_patches_ref:
            raise SystemExit(f"perm {perm} has {n_p} patches, expected {n_patches_ref}")

        outs = []
        for lo in range(0, n_p, a.batch_size):
            blk = autos[lo:lo + a.batch_size]
            outs.append(np.asarray(_compress(transform(blk))))
        S_parts.append(np.concatenate(outs, 0).astype(np.float32))
        perm_parts.append(np.full(n_p, perm, dtype=np.int32))
        patch_parts.append(np.arange(n_p, dtype=np.int32))
        if (perm + 1) % 10 == 0:
            print(f"  perm {perm+1}/{a.max_perm} ({time.time()-t0:.0f}s)", flush=True)

    S = np.concatenate(S_parts, 0)
    perm_arr = np.concatenate(perm_parts, 0)
    patch_arr = np.concatenate(patch_parts, 0)

    if not np.isfinite(S).all():
        raise SystemExit(f"{int((~np.isfinite(S)).any(1).sum())} non-finite summaries produced")
    # The bulk path is batched; the gate ran unbatched. Confirm they agree on obs 0 so the
    # batching itself cannot have changed anything (BatchNorm in eval mode must not see batch stats).
    d_batched = float(np.abs(S[0].astype(np.float64) - ref).max())
    print(f"\n  batched-vs-reference on (perm 0, patch 0): max|delta| = {d_batched:.3e}")
    if d_batched > a.g1_tol:
        raise SystemExit(f"batched path disagrees with the persisted obs vector "
                         f"({d_batched:.3e} > {a.g1_tol:.1e}) -- BatchNorm leaking batch stats?")

    np.savez(out_p, S=S, perm=perm_arr, patch=patch_arr,
             truth=np.asarray(truth, dtype=np.float64))
    prov = dict(arm=str(arm), out=str(out_p), n_obs=int(S.shape[0]), dim=int(S.shape[1]),
                regime=regime, cross_op=op, bnt=bool(bnt), arch=arch,
                n_perms=int(a.max_perm), n_patches=int(n_patches_ref),
                channel_scale=[float(x) for x in scale],
                params_sha256=params_sha, g1_max_abs_delta=delta,
                g1_tol=a.g1_tol, batched_max_abs_delta=d_batched,
                elapsed_s=round(time.time() - t0, 1))
    json.dump(prov, open(out_p.with_suffix(".provenance.json"), "w"), indent=2)

    print(f"\n  wrote {out_p}  S={S.shape} perm={perm_arr.shape} patch={patch_arr.shape}")
    print(f"  S mean/std (first 4 dims): {S.mean(0)[:4]} / {S.std(0)[:4]}")
    print(f"  done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
