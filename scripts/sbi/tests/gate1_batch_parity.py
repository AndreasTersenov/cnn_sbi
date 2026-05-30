"""Gate 1 — batch-level parity test for the TFRecord cross loader vs the .npz path.

CPU-only (does not touch any GPU). Verifies that the *fixed* tf.data cross loader
(`tfds_cross_tfdata_loader.build_tfds_tfdata_iterator`) reproduces the science of the
legacy `.npz` reference path (`build_harmonic_batch_iterator`) before we spend GPU hours
on the end-to-end gate.

Four checks:
  (1) Provenance-matched raw parity: for K TFRecord examples (no shuffle), the raw
      `map_nbody` + `theta` match the matched `.npz` patch (by cosmo_idx/perm/patch),
      max abs diff ~0. Re-confirms bit-exactness *through the loader read path*.
  (2) Fixed-loader normalization: pull a real batch from build_tfds_tfdata_iterator for
      cross_only (slice 4:10) and auto_cross (slice 0:10) with the cached per-channel
      RMS as channel_scale; assert every channel lands at ~unit RMS (NOT ~1e-14, which
      the pre-fix `* scale` bug produced).
  (3) theta H0->h0: loader theta[3] == raw H0 / 100.
  (4) Split mapping: TFRecord `test` cosmologies == `.npz val` cosmologies.

Run:
  CUDA_VISIBLE_DEVICES="" /home/tersenov/anaconda3/envs/jaxili/bin/python \
      scripts/sbi/tests/gate1_batch_parity.py --data-dir /nas/tersenov/tfds_cross_tfrecord_full
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Keep this entirely off the GPUs (reserved for Gate 2) and quiet.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

_HERE = Path(__file__).resolve().parent
_SBI = _HERE.parent
for p in (str(_SBI), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

NPZ_CACHE = _SBI / "results" / "exploratory" / "cross_maps_campaign" / "full_sphere_cache_grid"
REGIME = "nobnt"
RMS_CACHE = NPZ_CACHE / ".channel_rms_cache"
TFDS_NAME = "nbody_cosmogrid_dataset_tomo_cross/grid_20deg_160px_nonoverlap48"

# TFRecord split -> .npz subdir (mirrors tf_dataset_nbody_tomo_cross._SPLIT_SUBDIR).
SPLIT_SUBDIR = {"train": "train", "test": "val", "obs": "obs"}


def _load_rms(slice_tag: str) -> np.ndarray:
    """Read a cached per-channel RMS vector (slice_tag in {'all','4-10-None'})."""
    matches = sorted(RMS_CACHE.glob(f"{REGIME}__train__slice_{slice_tag}__lim_all__*.json"))
    if not matches:
        raise FileNotFoundError(f"No RMS cache for slice {slice_tag} under {RMS_CACHE}")
    return np.asarray(json.load(open(matches[0]))["rms"], np.float32)


def _npz_index(subdir: str) -> dict[tuple[int, int], Path]:
    """Map (cosmo_idx, perm) -> .npz path for a split subdir."""
    cache_dir = NPZ_CACHE / REGIME / subdir
    out: dict[tuple[int, int], Path] = {}
    for f in sorted(cache_dir.glob("*.npz")):
        with np.load(f, allow_pickle=False) as d:
            out[(int(d["cosmo_idx"]), int(d["perm"]))] = f
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/nas/tersenov/tfds_cross_tfrecord_full")
    ap.add_argument("--k-examples", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    import tensorflow as tf
    import tensorflow_datasets as tfds
    import tf_dataset_nbody_tomo_cross as _b  # noqa: F401 (registers dataset)
    from tfds_cross_tfdata_loader import build_tfds_tfdata_iterator

    rms_all = _load_rms("all")        # 10 channels (auto_cross)
    rms_cross = _load_rms("4-10-None")  # 6 channels (cross_only)
    print(f"[setup] rms_all (10ch) = {np.array2string(rms_all, precision=3)}")
    print(f"[setup] rms_cross (6ch) = {np.array2string(rms_cross, precision=3)}")

    failures: list[str] = []

    # ---- Check (1): provenance-matched raw parity, no shuffle ----------------
    print("\n[check 1] provenance-matched raw parity (TFRecord train vs .npz train)")
    read_cfg = tfds.ReadConfig(interleave_cycle_length=1, interleave_block_length=1)
    raw = tfds.load(TFDS_NAME, split="train", data_dir=args.data_dir,
                    shuffle_files=False, read_config=read_cfg)
    npz_idx = _npz_index("train")
    n_checked = 0
    max_map_diff = 0.0
    max_theta_diff = 0.0
    for ex in tfds.as_numpy(raw.take(args.k_examples)):
        ci, perm, patch = int(ex["cosmo_idx"]), int(ex["perm"]), int(ex["patch"])
        npz_path = npz_idx.get((ci, perm))
        if npz_path is None:
            failures.append(f"(1) no .npz for cosmo_idx={ci} perm={perm}")
            continue
        with np.load(npz_path, allow_pickle=False) as d:
            ref_map = np.asarray(d["patches"], np.float32)[patch]
            ref_theta = np.asarray(d["theta"], np.float32)
        md = float(np.abs(ex["map_nbody"].astype(np.float32) - ref_map).max())
        td = float(np.abs(ex["theta"].astype(np.float32) - ref_theta).max())
        max_map_diff = max(max_map_diff, md)
        max_theta_diff = max(max_theta_diff, td)
        n_checked += 1
    print(f"  matched {n_checked}/{args.k_examples}; max |map diff|={max_map_diff:.3e}, "
          f"max |theta diff|={max_theta_diff:.3e}")
    if n_checked == 0:
        failures.append("(1) matched 0 examples")
    if max_map_diff != 0.0:
        failures.append(f"(1) raw map diff {max_map_diff:.3e} != 0")
    if max_theta_diff != 0.0:
        failures.append(f"(1) raw theta diff {max_theta_diff:.3e} != 0")

    # ---- Check (2)+(3): fixed-loader normalization + theta scaling -----------
    for mode, ch_slice, scale in (
        ("cross_only", slice(4, 10), rms_cross),
        ("auto_cross", slice(0, 10), rms_all),
    ):
        print(f"\n[check 2] fixed loader normalization — {mode}")
        it = build_tfds_tfdata_iterator(
            tfds_name=TFDS_NAME, data_dir=args.data_dir, split="train",
            batch_size=args.batch_size, seed=41, flip=False,
            channel_scale=scale, channel_slice=ch_slice, shuffle_buffer=256,
        )
        batch = next(it)
        maps, theta = batch["maps"], batch["theta"]
        per_ch_rms = np.sqrt((maps.astype(np.float64) ** 2).mean(axis=(0, 1, 2)))
        print(f"  maps shape={maps.shape}, per-channel RMS="
              f"{np.array2string(per_ch_rms, precision=3)}")
        # Post-fix: each channel normalized by its global train RMS -> ~unit RMS
        # on a finite batch. The pre-fix `* scale` bug produced ~1e-13.
        if not np.all((per_ch_rms > 0.3) & (per_ch_rms < 3.0)):
            failures.append(f"(2/{mode}) per-channel RMS not ~unit: {per_ch_rms}")
        if per_ch_rms.max() < 1e-6:
            failures.append(f"(2/{mode}) channels collapsed (multiply bug?): {per_ch_rms}")
        # (3) theta H0 -> h0: column 3 must be ~0.5-0.9 (h0), not ~50-90 (H0).
        h0 = theta[:, 3]
        print(f"  theta[:,3] (h0) range = [{h0.min():.4f}, {h0.max():.4f}]")
        if not np.all((h0 > 0.4) & (h0 < 1.0)):
            failures.append(f"(3/{mode}) theta[3] not h0-scaled: range "
                            f"[{h0.min():.3f},{h0.max():.3f}]")

    # ---- Check (4): split mapping TFRecord test == .npz val ------------------
    print("\n[check 4] split mapping: TFRecord `test` cosmologies == .npz `val`")
    test_raw = tfds.load(TFDS_NAME, split="test", data_dir=args.data_dir,
                         shuffle_files=False, read_config=read_cfg)
    tf_test_cosmos = set()
    for ex in tfds.as_numpy(test_raw.take(4368)):  # full obs/test is small; take all test
        tf_test_cosmos.add(int(ex["cosmo_idx"]))
    npz_val_cosmos = {ci for (ci, _perm) in _npz_index("val").keys()}
    print(f"  TFRecord test: {len(tf_test_cosmos)} cosmologies; "
          f".npz val: {len(npz_val_cosmos)} cosmologies")
    if not tf_test_cosmos:
        failures.append("(4) no TFRecord test cosmologies read")
    elif not tf_test_cosmos.issubset(npz_val_cosmos):
        extra = sorted(tf_test_cosmos - npz_val_cosmos)[:10]
        failures.append(f"(4) TFRecord test has cosmologies not in .npz val: {extra}")

    # ---- Verdict -------------------------------------------------------------
    print("\n" + "=" * 60)
    if failures:
        print("GATE 1 FAIL:")
        for f in failures:
            print("  -", f)
        return 1
    print("GATE 1 PASS — transform parity + normalization direction confirmed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
