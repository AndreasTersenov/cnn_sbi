#!/usr/bin/env python3
"""Compute CNN-VMIM summaries for the full-200 fiducial set (9600 patches).

Reuses the EXACT training/obs preprocessing by importing the leaf functions from
npe_cnn_nbody_tomo.py:
  - compute_harmonic_channel_rms  (channel_scale, from the GRID train[:70%] split)
  - build_compressors / load_compressor_params (the trained plain-CNN compressor)
  - per-patch path identical to load_observed_from_harmonic_cache:
        m = patches[patch][..., channel_slice] / channel_scale ; summary = compressor(m)

G1 self-check: the summary at (--g1-perm, --g1-patch) must reproduce the saved
cnn_obs.npz x for the arm (the campaign's obs summary). Aborts on mismatch.

Output: <out> with S (Nperm*48, dim) float32, perm (N,), patch (N,), theta (6,).
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
GRID_CACHE = REPO / "results/exploratory/cross_maps_campaign/full_sphere_cache_grid"
FID_CACHE = REPO / "results/exploratory/cross_maps_campaign/full_sphere_cache_fiducial"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arm-label", required=True)
    p.add_argument("--params-pkl", required=True)
    p.add_argument("--state-pkl", required=True)
    p.add_argument("--n-channels", type=int, required=True, help="4 (auto) or 10 (auto+cross)")
    p.add_argument("--dim", type=int, default=10)
    p.add_argument("--conv-channels", default="64,128,256")
    p.add_argument("--dense-width", type=int, default=256)
    p.add_argument("--pool-window", type=int, default=16)
    p.add_argument("--pool-stride", type=int, default=8)
    p.add_argument("--perms", default="0-199")
    p.add_argument("--regime", default="nobnt")
    p.add_argument("--cosmo-id", default="cosmo_fiducial")
    p.add_argument("--out", required=True)
    # 10deg / tfds_cross repoint: the grid .npz cache is deleted, so channel_scale
    # comes from the SAME deterministic TFDS estimator used in training; the fiducial
    # obs cache is the kept 10deg one. Back-compat: omit these -> 20deg grid-cache path.
    p.add_argument("--cross-tfds-name", default="",
                   help="If set, compute channel_scale from this TFDS (tfds_cross arm) "
                        "instead of the (deleted) grid .npz cache.")
    p.add_argument("--cross-tfds-data-dir", default="/home/tersenov/tensorflow_datasets")
    p.add_argument("--cross-op", default="",
                   help="Flat-local route: build the patch-local cross on-device per this op "
                        "{none,conv,product,both} from autos ch 0..nbins-1 (NOT the leaky "
                        "cross channels). Empty = legacy tfds_cross/grid path.")
    p.add_argument("--nbins", type=int, default=4,
                   help="Number of auto/tomo bins to read for the flat-local route (--cross-op).")
    p.add_argument("--flatsky-roll-frac", type=float, default=0.10,
                   help="Apodization roll fraction for the flat-local conv op (LOCKED 0.10).")
    p.add_argument("--fid-cache-dir", default="",
                   help="Override the fiducial obs cache dir (10deg: full_sphere_cache_fiducial_10deg).")
    p.add_argument("--channel-rms-nsample", type=int, default=8000,
                   help="n examples for compute_cross_tfds_channel_rms (must match training=8000).")
    p.add_argument("--expect-params-sha", default="",
                   help="If set, assert the loaded compressor params pkl has this sha256 "
                        "(from the arm's cnn_cache_meta) -> guarantees the exact Phase-C compressor.")
    p.add_argument("--expect-state-sha", default="")
    p.add_argument("--g1-obs-npz", default="",
                   help="Optional cnn_obs.npz to reproduce at (g1-perm,g1-patch). If absent "
                        "(Phase C didn't --exit-after-compress), correctness rests on the "
                        "sha-pinned compressor + deterministic channel_scale instead.")
    p.add_argument("--g1-perm", type=int, default=0)
    p.add_argument("--g1-patch", type=int, default=0)
    # 1e-3 tolerates float32 forward-pass noise on O(1) summaries (e.g. MAF arm
    # lands at 2.8e-4) while still catching real bugs (wrong compressor/scale = O(0.1-1)).
    p.add_argument("--g1-rtol", type=float, default=1e-3)
    p.add_argument("--g1-atol", type=float, default=1e-3)
    p.add_argument("--cuda-visible-devices", default="1")
    return p.parse_args()


def _perms(spec):
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-"); out += list(range(int(a), int(b) + 1))
        elif tok:
            out.append(int(tok))
    return out


def main():
    a = parse_args()
    sys.path.insert(0, str(REPO))
    from train_jaxili_from_compressed import setup_env  # honors CUDA env safely
    setup_env(a.cuda_visible_devices)
    import jax
    from npe_cnn_nbody_tomo import (
        build_compressors, load_compressor_params, compute_harmonic_channel_rms, file_sha256,
    )

    fid_cache = Path(a.fid_cache_dir).resolve() if a.fid_cache_dir else FID_CACHE
    flat_local = bool(a.cross_op)
    flat_transform = None
    if flat_local:
        # Flat-local route: read RAW autos ch 0..nbins-1 and build the patch-local cross
        # on-device with the SAME estimator + transform as training (so G1 reproduces).
        from npe_cnn_nbody_tomo import (
            compute_flat_cross_channel_rms, make_flat_cross_transform,
        )
        ch_slice = slice(0, a.nbins)
    else:
        ch_slice = slice(0, a.n_channels) if a.n_channels != 10 else None
    print(f"[{a.arm_label}] flat_local={flat_local} op={a.cross_op or '-'} "
          f"channel_slice={ch_slice} dim={a.dim} fid_cache={fid_cache}", flush=True)

    # channel_scale (per-channel RMS; the CNN DIVIDE convention):
    if flat_local:
        channel_scale = compute_flat_cross_channel_rms(
            tfds_name=a.cross_tfds_name, data_dir=a.cross_tfds_data_dir,
            op=a.cross_op, nbins=a.nbins, split="train",
            n_sample=a.channel_rms_nsample, roll_frac=a.flatsky_roll_frac,
        )
        channel_scale = np.asarray(channel_scale, dtype=np.float32)
        flat_transform = make_flat_cross_transform(a.cross_op, channel_scale, a.flatsky_roll_frac)
    elif a.cross_tfds_name:
        # tfds_cross arm: SAME deterministic estimator as B-1 training (n_sample must
        # match training=8000) -> byte-identical scale; the grid .npz cache is deleted.
        from tfds_cross_tfdata_loader import compute_cross_tfds_channel_rms
        channel_scale = compute_cross_tfds_channel_rms(
            tfds_name=a.cross_tfds_name, data_dir=a.cross_tfds_data_dir,
            split="train", n_sample=a.channel_rms_nsample, channel_slice=ch_slice,
        )
    else:
        channel_scale = compute_harmonic_channel_rms(
            cache_dir=GRID_CACHE, regime=a.regime, split="train[:70%]",
            max_realizations=None, channel_slice=ch_slice,
        )
    channel_scale = np.asarray(channel_scale, dtype=np.float32)
    print(f"  channel_scale={channel_scale}", flush=True)
    if not (3e-3 <= float(channel_scale.max()) <= 2e-2):
        print(f"  [warn] max channel_scale {float(channel_scale.max()):.3e} outside the "
              "Phase-A auto bound [3e-3,2e-2] -- check dataset/channel_mode.", flush=True)
    # Pin the loaded compressor to the arm's stored sha256 (exact Phase-C compressor).
    if a.expect_params_sha:
        _g = file_sha256(Path(a.params_pkl))
        assert _g == a.expect_params_sha, f"params sha mismatch: {_g} != {a.expect_params_sha}"
        print(f"  [sha] compressor params match arm meta ({_g[:16]}...)", flush=True)
    if a.expect_state_sha:
        _gs = file_sha256(Path(a.state_pkl))
        assert _gs == a.expect_state_sha, f"state sha mismatch: {_gs} != {a.expect_state_sha}"
        print(f"  [sha] compressor state match arm meta ({_gs[:16]}...)", flush=True)

    conv = tuple(int(c) for c in a.conv_channels.split(","))
    _, comp_eval = build_compressors(
        dim=a.dim, arch="plain", conv_channels=conv, dense_width=a.dense_width,
        pool_window=a.pool_window, pool_stride=a.pool_stride,
        resnet_small_channels=(64, 128, 256), resnet_small_blocks=(2, 2, 2),
        resnet_head_width=256, resnet_v2=False,
    )
    params, state = load_compressor_params(a.params_pkl, a.state_pkl)

    def summarize(maps_np):  # (B,160,160,nch) already sliced+scaled -> (B,dim)
        y, _ = comp_eval.apply(params, state, None, maps_np)
        return np.asarray(y, dtype=np.float32).reshape(maps_np.shape[0], -1)

    perms = _perms(a.perms)
    S, P, PA, TH = [], [], [], []
    t0 = time.time()
    for i, p in enumerate(perms):
        npz = fid_cache / a.regime / "obs" / f"{a.cosmo_id}_perm{p}.npz"
        with np.load(npz, allow_pickle=False) as d:
            patches = np.asarray(d["patches"], dtype=np.float32)
            theta = np.asarray(d["theta"], dtype=np.float64)
        if flat_local:
            # RAW autos -> build patch-local cross + whiten on-device (same as training/obs).
            autos = patches[..., ch_slice]
            patches = np.asarray(flat_transform(autos), dtype=np.float32)
        else:
            if ch_slice is not None:
                patches = patches[..., ch_slice]
            patches = patches / channel_scale
        y = summarize(patches)  # (n_patches, dim)
        S.append(y)
        P.append(np.full(y.shape[0], p, np.int32))
        PA.append(np.arange(y.shape[0], dtype=np.int32))
        TH.append(np.broadcast_to(theta, (y.shape[0], theta.shape[0])).copy())
        if (i + 1) % 25 == 0:
            print(f"  perm {p} ({i+1}/{len(perms)}) {time.time()-t0:.0f}s", flush=True)
    S = np.concatenate(S, 0); P = np.concatenate(P, 0)
    PA = np.concatenate(PA, 0); TH = np.concatenate(TH, 0)
    print(f"  summaries {S.shape} in {time.time()-t0:.0f}s", flush=True)

    # ---- G1 self-check: reproduce the saved obs summary (if a reference exists) ----
    maxdev = -1.0
    if a.g1_obs_npz:
        ref = np.load(a.g1_obs_npz)["x"].astype(np.float32).reshape(-1)
        mine = S[(P == a.g1_perm) & (PA == a.g1_patch)][0]
        ok = np.allclose(mine, ref, rtol=a.g1_rtol, atol=a.g1_atol)
        maxdev = float(np.max(np.abs(mine - ref)))
        print(f"  [G1] arm={a.arm_label} reproduce cnn_obs@(p{a.g1_perm},patch{a.g1_patch}): "
              f"max|Δ|={maxdev:.3e} -> {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            print(f"  [G1] ref ={ref}\n  [G1] mine={mine}", flush=True)
            raise SystemExit(f"G1 FAILED for {a.arm_label} (max|Δ|={maxdev:.3e}); aborting (no garbage).")
    else:
        # No saved cnn_obs (Phase C didn't --exit-after-compress). Correctness rests on
        # the deterministic channel_scale (same estimator+args as training) + the
        # sha-pinned compressor checkpoint, both verified above.
        print(f"  [G1] no --g1-obs-npz given; correctness via sha-pinned compressor + "
              f"deterministic channel_scale (verify --expect-params-sha was passed).", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, S=S, perm=P, patch=PA, theta=TH,
             channel_scale=channel_scale, g1_maxdev=np.float64(maxdev))
    print(f"[{a.arm_label}] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
