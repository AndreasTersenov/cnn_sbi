#!/usr/bin/env python3
"""Validate gpu_metrics.all_metrics_gpu against the CPU reference (all_metrics_cpu, = the
existing train_jaxili compute_fom3/fom2d/marginal_stats) and time both paths.

CORRECTNESS: synthetic 30000x6 posterior samples (realistic marginal scales + correlations);
GPU-float32 metrics vs CPU-float64 reference; report max relative diff per metric; PASS if
all within --rtol (FoM3 is the sensitive one — slogdet of a 3x3 cov).

SPEED: per-"patch" metrics+transfer cost, starting from an on-DEVICE array (as post.sample
returns). Path A (current) = np.asarray(device array) + CPU numpy metrics. Path B (new) =
all_metrics_gpu (jnp on device + 17-scalar transfer). Median over many iters; reports speedup.
Run on GPU (CUDA_VISIBLE_DEVICES=0, small mem) for a real measurement; on CPU it still checks
correctness. Sampling cost itself is unchanged and NOT measured here.
"""
from __future__ import annotations
import argparse, os, time
import numpy as np

FIDUCIAL = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493])


def make_samples(n, seed=0):
    """Realistic correlated 6D posterior draws (scales ~ the campaign marginals)."""
    rng = np.random.default_rng(seed)
    sig = np.array([0.025, 0.038, 0.13, 0.05, 0.04, 0.01])
    # correlation: strong Om-s8-w0 block (FoM3 subspace), mild elsewhere
    C = np.eye(6)
    C[0, 1] = C[1, 0] = 0.6; C[0, 2] = C[2, 0] = -0.5; C[1, 2] = C[2, 1] = -0.4
    cov = np.outer(sig, sig) * C
    x = rng.multivariate_normal(FIDUCIAL + np.array([-0.003, 0.007, -0.045, 0, 0, 0]), cov, size=n)
    return x.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--mem-fraction", default="0.10")
    args = ap.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", args.mem_fraction)
    import jax, jax.numpy as jnp
    from gpu_metrics import all_metrics_gpu, all_metrics_cpu
    dev = jax.devices()[0].platform
    print(f"jax device: {dev}  | n={args.n} iters={args.iters} rtol={args.rtol}")

    x_np = make_samples(args.n)
    x_dev = jnp.asarray(x_np)

    # ---------- correctness ----------
    mc = all_metrics_cpu(x_np, FIDUCIAL)
    mg = all_metrics_gpu(x_dev, FIDUCIAL)
    print("\n## correctness (GPU-f32 vs CPU-f64 reference)")
    worst = 0.0
    for k in mc:
        if k == "valid_fom3":
            ok = bool(mc[k]) == bool(mg[k]); print(f"  {k:26s} cpu={mc[k]} gpu={mg[k]} {'OK' if ok else 'MISMATCH'}")
            continue
        a, b = float(mc[k]), float(mg[k])
        rel = abs(a - b) / (abs(a) + 1e-30)
        worst = max(worst, rel)
        flag = "" if rel <= args.rtol else "  <<< EXCEEDS rtol"
        print(f"  {k:26s} cpu={a:+.6g} gpu={b:+.6g} rel={rel:.2e}{flag}")
    verdict = "PASS" if worst <= args.rtol else "FAIL"
    print(f"  -> worst relative diff = {worst:.2e}  [{verdict} @ rtol {args.rtol}]")

    # ---------- speed ----------
    print("\n## speed (metrics+transfer per patch; sampling NOT included)")
    # warmup (JIT/compile)
    _ = all_metrics_gpu(x_dev, FIDUCIAL); _ = all_metrics_cpu(np.asarray(x_dev), FIDUCIAL)

    tA = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        h = np.asarray(x_dev)                 # device->host transfer (as current sample_pooled does)
        _ = all_metrics_cpu(h, FIDUCIAL)      # CPU numpy metrics
        tA.append(time.perf_counter() - t0)
    tB = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        _ = all_metrics_gpu(x_dev, FIDUCIAL)  # on-device metrics + 17-scalar transfer
        tB.append(time.perf_counter() - t0)
    a_ms = np.median(tA) * 1e3; b_ms = np.median(tB) * 1e3
    print(f"  path A (current: transfer + CPU numpy): {a_ms:7.2f} ms/patch  (p16-84 {np.percentile(tA,16)*1e3:.2f}-{np.percentile(tA,84)*1e3:.2f})")
    print(f"  path B (GPU metrics + scalar transfer): {b_ms:7.2f} ms/patch  (p16-84 {np.percentile(tB,16)*1e3:.2f}-{np.percentile(tB,84)*1e3:.2f})")
    print(f"  -> metrics speedup x{a_ms/b_ms:.1f}; saves ~{a_ms-b_ms:.1f} ms/patch")
    print(f"     (context: per-patch total was ~460-680 ms incl. sampling; this is the removable CPU part)")


if __name__ == "__main__":
    main()
