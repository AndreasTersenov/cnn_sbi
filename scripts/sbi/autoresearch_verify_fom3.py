#!/usr/bin/env python3
"""Verify-wrapper for autoresearch: emit mean-of-seeds FoM3 and per-seed min
from a list of posterior .npy paths. FoM3 definition matches compare_probes_configs.py.

Usage:
    python autoresearch_verify_fom3.py /path/to/p_s41.npy /path/to/p_s42.npy ...
    python autoresearch_verify_fom3.py --posteriors-glob '/path/to/posteriors/*_s4?.npy'

Stdout:
    per_seed_fom3: <path> → <value>   (one line per seed; for provenance)
    fom3_per_seed_min: <value>        (Guard metric)
    fom3_std: <value>                 (advisory)
    fom3_mean: <value>                (Verify metric — autoresearch greps this)

Exits nonzero if no posteriors loaded.
"""
from __future__ import annotations
import argparse
import glob
import sys
import numpy as np


def fom3(samples: np.ndarray) -> float:
    """3-D FoM3 = 1 / sqrt(det(C_{Omega_m, sigma_8, w_0}))."""
    C = np.cov(samples[:, :3].T)
    return 1.0 / np.sqrt(np.linalg.det(C))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*",
                    help="Posterior .npy paths (or use --posteriors-glob)")
    ap.add_argument("--posteriors-glob", default=None,
                    help="Glob for posterior files (e.g. /path/to/*_s4?.npy)")
    ap.add_argument("--cap-samples", type=int, default=100_000,
                    help="Cap per-seed samples (matches compare_probes_configs.py)")
    args = ap.parse_args()

    paths = list(args.paths)
    if args.posteriors_glob:
        paths += sorted(glob.glob(args.posteriors_glob))
    if not paths:
        print("ERROR: no posterior paths provided", file=sys.stderr)
        return 1

    per_seed = []
    for p in paths:
        try:
            x = np.load(p, allow_pickle=False)
        except (FileNotFoundError, OSError) as e:
            print(f"  [skip] {p}: {e}", file=sys.stderr)
            continue
        if x.ndim != 2 or x.shape[1] < 3:
            print(f"  [skip] {p}: bad shape {x.shape}", file=sys.stderr)
            continue
        if x.shape[0] > args.cap_samples:
            x = x[:args.cap_samples]
        v = fom3(x)
        per_seed.append(v)
        print(f"per_seed_fom3: {p} → {v:.4f}")

    if not per_seed:
        print("ERROR: no posteriors loaded successfully", file=sys.stderr)
        return 1

    mean = float(np.mean(per_seed))
    std = float(np.std(per_seed)) if len(per_seed) > 1 else 0.0
    print(f"fom3_per_seed_min: {min(per_seed):.4f}")
    print(f"fom3_std: {std:.4f}")
    print(f"fom3_mean: {mean:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())