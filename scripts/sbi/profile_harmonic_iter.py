#!/usr/bin/env python
"""One-off profiler for the harmonic-cache iterator.

Times pure I/O cost per yielded batch and per step component so we can decide
whether async prefetch alone is enough or whether the working-set pool is also
needed. No GPU work — we already know real step time (~0.48 s at 2.1 it/s).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from npe_cnn_nbody_tomo import (
    _list_harmonic_cache_files,
    build_harmonic_batch_iterator,
)

HARM_CACHE = Path(
    "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/"
    "cross_maps_campaign/full_sphere_cache_grid"
)


def time_raw_load(files: list[Path], n: int) -> dict:
    """Pure `np.load + decompress` cost per file (no slicing, no shuffle)."""
    samples = []
    for path in files[:n]:
        t0 = time.perf_counter()
        with np.load(path, allow_pickle=False) as d:
            _ = np.asarray(d["patches"], dtype=np.float32)
            _ = np.asarray(d["theta"], dtype=np.float64)
        samples.append(time.perf_counter() - t0)
    arr = np.asarray(samples)
    return {
        "n": n,
        "mean_s": float(arr.mean()),
        "median_s": float(np.median(arr)),
        "p95_s": float(np.percentile(arr, 95)),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
    }


def time_iterator(
    cache_dir: Path,
    regime: str,
    split: str,
    batch_size: int,
    channel_slice: slice | None,
    n_batches: int,
    simulate_compute_ms: float = 0.0,
) -> dict:
    """End-to-end iterator yield cost (load + slice + scale + flip + perm)."""
    it = build_harmonic_batch_iterator(
        cache_dir=cache_dir,
        regime=regime,
        split=split,
        batch_size=batch_size,
        seed=42,
        flip=True,
        max_realizations=None,
        channel_scale=None,
        channel_slice=channel_slice,
    )
    samples = []
    actual_batch_sizes = []
    # warmup
    next(it)
    for _ in range(5):
        next(it)
    for _ in range(n_batches):
        t0 = time.perf_counter()
        ex = next(it)
        samples.append(time.perf_counter() - t0)
        actual_batch_sizes.append(int(ex["maps"].shape[0]))
        if simulate_compute_ms > 0:
            time.sleep(simulate_compute_ms / 1000.0)
    arr = np.asarray(samples)
    bs = np.asarray(actual_batch_sizes)
    return {
        "n_batches": n_batches,
        "mean_s": float(arr.mean()),
        "median_s": float(np.median(arr)),
        "p95_s": float(np.percentile(arr, 95)),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
        "fraction_zero_time": float((arr < 1e-3).mean()),
        "actual_batch_mean": float(bs.mean()),
        "actual_batch_min": int(bs.min()),
        "actual_batch_max": int(bs.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=HARM_CACHE)
    ap.add_argument("--regime", default="nobnt")
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-files", type=int, default=20)
    ap.add_argument("--n-batches", type=int, default=60)
    ap.add_argument("--channel-mode", choices=["auto_cross", "cross_only"], default="cross_only")
    ap.add_argument(
        "--simulate-compute-ms",
        type=float,
        default=0.0,
        help="Sleep this long after each yield to simulate compute; lets the prefetcher overlap.",
    )
    args = ap.parse_args()

    print(f"Cache dir: {args.cache_dir}")
    print(f"Regime: {args.regime}  Split: {args.split}")
    print(f"Channel mode: {args.channel_mode}")

    files = _list_harmonic_cache_files(args.cache_dir, args.regime, args.split)
    print(f"Total files in split: {len(files)}")

    # Pure raw-load timing
    print("\n[1/2] Timing raw np.load + decompress (no slice/shuffle) ...")
    raw = time_raw_load(files, args.n_files)
    print(
        f"  n={raw['n']}  mean={raw['mean_s']*1000:.1f}ms  "
        f"median={raw['median_s']*1000:.1f}ms  "
        f"p95={raw['p95_s']*1000:.1f}ms  "
        f"min={raw['min_s']*1000:.1f}ms  max={raw['max_s']*1000:.1f}ms"
    )

    # Iterator timing (includes slice/scale/flip/perm), with channel_mode honored
    channel_slice = slice(4, 10) if args.channel_mode == "cross_only" else None
    print(f"\n[2/2] Timing build_harmonic_batch_iterator (batch_size={args.batch_size}) ...")
    iter_stats = time_iterator(
        cache_dir=args.cache_dir,
        regime=args.regime,
        split=args.split,
        batch_size=args.batch_size,
        channel_slice=channel_slice,
        n_batches=args.n_batches,
        simulate_compute_ms=args.simulate_compute_ms,
    )
    print(
        f"  n_batches={iter_stats['n_batches']}  "
        f"mean={iter_stats['mean_s']*1000:.1f}ms  "
        f"median={iter_stats['median_s']*1000:.1f}ms  "
        f"p95={iter_stats['p95_s']*1000:.1f}ms"
    )
    print(
        f"  zero-time fraction (cheap intra-file yields): "
        f"{iter_stats['fraction_zero_time']*100:.1f}%"
    )
    print(
        f"  actual batch sizes: mean={iter_stats['actual_batch_mean']:.1f}  "
        f"min={iter_stats['actual_batch_min']}  max={iter_stats['actual_batch_max']}"
    )

    print("\n=== Interpretation ===")
    # Compare against known real step time
    real_step_s = 0.476  # 2.1 it/s on resnet50_gn (current production)
    io_share = iter_stats["mean_s"] / real_step_s
    print(f"Assumed real step time (2.1 it/s):       {real_step_s*1000:.0f} ms")
    print(f"Iterator yield mean (this run):          {iter_stats['mean_s']*1000:.1f} ms")
    print(f"=> I/O share of step:                    {io_share*100:.1f}%")
    if io_share > 0.4:
        print(
            "   I/O is a major share. Async prefetch (option A) alone "
            "should yield ~1.5-2x speedup. Working-set pool (B) recovers "
            "the configured batch size on top of that."
        )
    elif io_share > 0.2:
        print(
            "   I/O is a moderate share. Async prefetch will help (~1.2-1.5x); "
            "B is mostly valuable for the batch-size recovery."
        )
    else:
        print(
            "   I/O share is small. Most time is compute-bound; "
            "pipeline rewrites will not help much."
        )

    print(
        f"\nActual batch size in iterator: {iter_stats['actual_batch_mean']:.1f} "
        f"(configured: {args.batch_size}) — "
        f"{'OK' if iter_stats['actual_batch_mean'] >= args.batch_size * 0.9 else 'CAPPED BY FILE GRANULARITY (48 patches/file)'}"
    )


if __name__ == "__main__":
    main()
