#!/usr/bin/env python
"""Convert the harmonic cross-map `.npz` cache to TFRecord shards on /nas.

One-time converter implementing §3.1 of
`scripts/sbi/HARMONIC_TFRECORD_IMPLEMENTATION_SPEC.md`. It reformats bytes only
(decompress `.npz` zlib -> reserialise -> write TFRecord); the numerical content
of every patch is preserved *bit-for-bit* (raw float32/float64 `tobytes()`).
There is NO GPU work here -- the GPU payoff is entirely on the training-read side
(`build_harmonic_tfrecord_iterator` in `npe_cnn_nbody_tomo.py`).

Sharding scheme (CRITICAL -- preserves split slicing, spec §1.5): one TFRecord
shard per source `.npz` file, named with the identical stem
(`{cosmo_id}_perm{perm}.tfrecord`), under `<out-dir>/<regime>/<split>/`. The
sorted shard list is therefore order-isomorphic to the sorted `.npz` list, so
`round(frac*n)` split slicing selects the same realisations on both paths.

Each shard holds 48 `tf.train.Example` records (one per patch, in patch index
order 0..47). The full 10 channels and the raw theta (H0=68.5, NOT /100) are
stored as-is; channel slice/scale and the H0/100 conversion all happen at read
time, identical to the `.npz` path.
"""

from __future__ import annotations

# The converter is CPU + disk-I/O only. Force CPU-only BEFORE any TF import so
# worker processes never grab GPU memory (project rule: GPU 1 only for real
# jobs; the converter must touch no GPU at all). This env var is inherited by
# forked Pool workers.
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

# Match the source cache layout (build_full_sphere_cross_cache.py).
PATCHES_PER_SHARD = 48
PATCH_SHAPE = (160, 160, 10)
N_CHANNELS = 10
ZERO_MEAN_ATOL = 1e-4

# TensorFlow is imported lazily inside worker processes (see `_get_tf`). The
# parent process must NOT import it: with the default fork start method on
# Linux, importing TF before forking can deadlock the workers, and we also want
# the parent to stay light. The parent needs no TF (manifest content-hash is
# computed from the source `.npz` bytes, which are written verbatim).
_TF = None


def _get_tf():
    global _TF
    if _TF is None:
        import tensorflow as tf  # noqa: E402  (lazy, per worker)

        _TF = tf
    return _TF


def _bytes_feature(value: bytes):
    tf = _get_tf()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value: int):
    tf = _get_tf()
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def _assert_zero_mean_patches(patches: np.ndarray, source: str) -> None:
    """Mirror of `_assert_zero_mean_patches` in npe_cnn_nbody_tomo.py (spec §1.6).

    Replicated inline (not imported) so the converter does not pull in the
    jax/haiku training module across 24 worker processes -- that import would
    risk grabbing GPU memory and violates the CPU-only/GPU-1-only contract. The
    logic is identical: max over channels of |mean over (H,W) of patch| <= atol.
    A failure means the SOURCE cache is corrupt; abort the whole build.
    """
    residual = float(np.abs(patches.mean(axis=(1, 2))).max())
    if residual > ZERO_MEAN_ATOL:
        raise ValueError(
            "Harmonic cache zero-mean compatibility check failed for "
            f"{source}: max per-channel patch mean residual "
            f"{residual:.3e} > {ZERO_MEAN_ATOL:.1e}."
        )


def _comp_type(compress: str) -> str:
    """Map the CLI compression name to tf's compression_type string."""
    return "" if compress.upper() == "NONE" else "GZIP"


def _convert_one_file(task: dict) -> dict:
    """Convert one source `.npz` -> one `.tfrecord` shard. Worker entry point."""
    tf = _get_tf()
    src_path = Path(task["src_path"])
    out_path = Path(task["out_path"])
    regime = task["regime"]
    split = task["split"]
    compress = task["compress"]
    overwrite = bool(task["overwrite"])
    comp_type = _comp_type(compress)

    # Idempotency: skip an existing shard with the expected record count.
    if out_path.exists() and not overwrite:
        try:
            n_existing = sum(
                1
                for _ in tf.data.TFRecordDataset(
                    str(out_path), compression_type=comp_type
                )
            )
        except Exception:
            n_existing = -1
        if n_existing == PATCHES_PER_SHARD:
            return {"status": "skipped", "src": src_path.name, "n": n_existing}
        # else: corrupt / wrong count -> rewrite below.

    with np.load(src_path, allow_pickle=False) as d:
        patches = np.asarray(d["patches"], dtype=np.float32)
        theta = np.asarray(d["theta"], dtype=np.float64)
        cosmo_id = str(d["cosmo_id"])
        perm = int(d["perm"])

    if patches.ndim != 4 or tuple(patches.shape[1:]) != PATCH_SHAPE:
        raise ValueError(
            f"Unexpected patch shape in {src_path}: {patches.shape} "
            f"(expected (N, {PATCH_SHAPE[0]}, {PATCH_SHAPE[1]}, {PATCH_SHAPE[2]}))."
        )
    if theta.shape != (6,):
        raise ValueError(f"Unexpected theta shape in {src_path}: {theta.shape}.")

    # Assert zero-mean on the full block (== asserting every patch; spec §1.6).
    _assert_zero_mean_patches(patches, str(src_path))

    # theta is identical for all patches in a shard; serialise once.
    theta_bytes = theta.astype(np.float64, copy=False).tobytes()
    cosmo_id_bytes = cosmo_id.encode("utf-8")
    regime_bytes = regime.encode("utf-8")
    split_bytes = split.encode("utf-8")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    options = tf.io.TFRecordOptions(compression_type=comp_type)
    n_written = 0
    with tf.io.TFRecordWriter(str(tmp_path), options) as writer:
        for patch_idx in range(patches.shape[0]):
            # (160,160,10) float32, C-order (row-major) raw bytes. The leading
            # slice of a C-contiguous array is itself C-contiguous; tobytes()
            # is C-order regardless, so this is the exact buffer the reader
            # will decode_raw + reshape back to (160,160,10).
            patch = patches[patch_idx]
            feature = {
                "patch": _bytes_feature(np.ascontiguousarray(patch).tobytes()),
                "theta": _bytes_feature(theta_bytes),
                "cosmo_id": _bytes_feature(cosmo_id_bytes),
                "perm": _int64_feature(perm),
                "patch_idx": _int64_feature(patch_idx),
                "regime": _bytes_feature(regime_bytes),
                "split": _bytes_feature(split_bytes),
            }
            example = tf.train.Example(
                features=tf.train.Features(feature=feature)
            )
            writer.write(example.SerializeToString())
            n_written += 1
    os.replace(tmp_path, out_path)
    return {"status": "written", "src": src_path.name, "n": n_written}


def _list_npz_files(split_dir: Path) -> list[Path]:
    files = sorted(p for p in split_dir.iterdir() if p.suffix == ".npz")
    return files


def _content_hash_first3_train(
    cache_dir: Path, regime: str, limit_files: int | None
) -> dict | None:
    """SHA256 over the raw float32 patch bytes of the first 3 sorted train shards.

    Computed from the SOURCE `.npz` patches (written verbatim into the
    TFRecord), in (file-sorted, patch 0..47) order. The equivalence test can
    recompute this from either path -- it is a cheap on-disk integrity anchor.
    """
    train_dir = cache_dir / regime / "train"
    if not train_dir.exists():
        return None
    files = _list_npz_files(train_dir)
    if limit_files is not None:
        files = files[:limit_files]
    files = files[:3]
    if not files:
        return None
    hasher = hashlib.sha256()
    total_patches = 0
    for path in files:
        with np.load(path, allow_pickle=False) as d:
            patches = np.asarray(d["patches"], dtype=np.float32)
        for patch_idx in range(patches.shape[0]):
            hasher.update(np.ascontiguousarray(patches[patch_idx]).tobytes())
            total_patches += 1
    return {
        "files": [p.name for p in files],
        "n_patches": total_patches,
        "sha256": hasher.hexdigest(),
    }


def _read_source_manifest_sha(cache_dir: Path) -> str:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        return ""
    try:
        with manifest_path.open("r") as f:
            payload = json.load(f)
        return str(payload.get("args_sha256", ""))
    except (json.JSONDecodeError, OSError):
        return ""


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert harmonic cross-map .npz cache to TFRecord shards."
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="The .npz cache root (e.g. .../full_sphere_cache_grid).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="TFRecord root on /nas, e.g. "
        "/nas/tersenov/harmonic_tfrecord/full_sphere_cache_grid",
    )
    p.add_argument(
        "--regime",
        type=str,
        default="both",
        choices=["nobnt", "bnt", "both"],
        help="Which regime(s) to convert (default: both present).",
    )
    p.add_argument(
        "--splits",
        type=str,
        default="train,val,obs",
        help="Comma-separated splits to convert (default: train,val,obs).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=24,
        help="Parallel conversion processes (I/O-bound sweet spot ~24; 16 if "
        "the overnight job is live; 50 cap).",
    )
    p.add_argument(
        "--compress",
        type=str,
        default="NONE",
        choices=["NONE", "GZIP"],
        help="TFRecord compression. NONE recommended; reader must match.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-convert even if a valid shard already exists.",
    )
    p.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Dev flag: convert only the first N sorted files per split.",
    )
    return p.parse_args()


def _build_task_list(args: argparse.Namespace) -> tuple[list[dict], list[str], dict]:
    cache_dir = Path(args.cache_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"--cache-dir not found: {cache_dir}")

    regimes = ["nobnt", "bnt"] if args.regime == "both" else [args.regime]
    requested_splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    tasks: list[dict] = []
    seen_regimes: list[str] = []
    per_split_files: dict = {}  # (regime, split) -> [src filenames sorted]
    for regime in regimes:
        regime_dir = cache_dir / regime
        if not regime_dir.is_dir():
            print(f"  [skip] regime dir absent: {regime_dir}")
            continue
        seen_regimes.append(regime)
        for split in requested_splits:
            split_dir = regime_dir / split
            if not split_dir.is_dir():
                print(f"  [skip] split dir absent: {split_dir}")
                continue
            files = _list_npz_files(split_dir)
            if args.limit_files is not None:
                files = files[: args.limit_files]
            if not files:
                print(f"  [skip] no .npz under {split_dir}")
                continue
            per_split_files[(regime, split)] = [f.name for f in files]
            for src in files:
                out_path = out_dir / regime / split / (src.stem + ".tfrecord")
                tasks.append(
                    {
                        "src_path": str(src),
                        "out_path": str(out_path),
                        "regime": regime,
                        "split": split,
                        "compress": args.compress,
                        "overwrite": bool(args.overwrite),
                    }
                )
    return tasks, seen_regimes, per_split_files


def _write_manifests(
    args: argparse.Namespace,
    seen_regimes: list[str],
    per_split_files: dict,
) -> None:
    cache_dir = Path(args.cache_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    source_sha = _read_source_manifest_sha(cache_dir)
    for regime in seen_regimes:
        regime_splits = {
            split: names
            for (rg, split), names in per_split_files.items()
            if rg == regime
        }
        if not regime_splits:
            continue
        per_split = {}
        for split, names in regime_splits.items():
            shard_names = sorted(n[:-4] + ".tfrecord" for n in names)
            per_split[split] = {
                "shards": shard_names,
                "shard_count": len(shard_names),
                "patch_count": len(shard_names) * PATCHES_PER_SHARD,
            }
        content_hash = _content_hash_first3_train(
            cache_dir, regime, args.limit_files
        )
        manifest = {
            "source_cache_dir": str(cache_dir),
            "source_manifest_args_sha256": source_sha,
            "regime": regime,
            "compression": args.compress,
            "n_channels": N_CHANNELS,
            "patch_shape": list(PATCH_SHAPE),
            "patch_dtype": "float32",
            "theta_dtype": "float64",
            "patches_per_shard": PATCHES_PER_SHARD,
            "limit_files": args.limit_files,
            "splits": per_split,
            "content_hash_first3_train": content_hash,
        }
        manifest_path = out_dir / regime / "tfrecord_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = manifest_path.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(manifest, f, indent=2)
        tmp.replace(manifest_path)
        print(f"  Wrote manifest {manifest_path}")


def main() -> int:
    args = _parse_args()
    print("######## build_harmonic_tfrecord ########")
    print(f"  cache-dir   = {args.cache_dir}")
    print(f"  out-dir     = {args.out_dir}")
    print(f"  regime      = {args.regime}")
    print(f"  splits      = {args.splits}")
    print(f"  workers     = {args.workers}")
    print(f"  compress    = {args.compress}")
    print(f"  overwrite   = {args.overwrite}")
    print(f"  limit-files = {args.limit_files}")

    tasks, seen_regimes, per_split_files = _build_task_list(args)
    if not tasks:
        print("  No files to convert. Nothing to do.")
        return 0
    print(f"  Total shards to (re)check: {len(tasks)}")
    for (regime, split), names in sorted(per_split_files.items()):
        print(f"    {regime}/{split}: {len(names)} files")

    n_workers = max(1, min(int(args.workers), 50))
    t0 = time.time()
    n_written = 0
    n_skipped = 0
    n_done = 0
    n_total = len(tasks)

    # Default fork start method on Linux is fine: the parent has NOT imported
    # TF (it is imported lazily inside each worker), so there is no
    # fork-after-tf-init deadlock.
    with mp.Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(_convert_one_file, tasks, chunksize=4):
            n_done += 1
            if result["status"] == "written":
                n_written += 1
            elif result["status"] == "skipped":
                n_skipped += 1
            if n_done % 200 == 0 or n_done == n_total:
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0.0
                print(
                    f"  [{n_done}/{n_total}] written={n_written} "
                    f"skipped={n_skipped} ({rate:.1f} shards/s, "
                    f"{elapsed:.0f}s)",
                    flush=True,
                )

    _write_manifests(args, seen_regimes, per_split_files)
    elapsed = time.time() - t0
    print(
        f"  DONE: {n_written} written, {n_skipped} skipped, "
        f"{n_total} total in {elapsed:.0f}s."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
