#!/usr/bin/env python3
"""Benchmark + bit-identity gate for jitting the population-sweep posterior sampling.

The audit (PIPELINE_AUDIT_2026-06-10.md §d) found the per-obs sampling loop in
population_sweep_flatsky.py is host-dispatch-bound: jaxili's posterior .sample is un-jitted
eager MAF inversion (~600 tiny dispatches + Flax mask rebuilds per call). This measures, on a
reloaded production checkpoint (CNN 10-d arm; reloads bit-exact):

  1. baseline   — the current eager loop (keys PRNGKey(seed*100003 + i), exactly as production)
  2. jit        — per-posterior jitted closure, same per-obs keys
  3. vmap[B]    — jit(vmap(...)) over obs chunks of size B, same per-obs keys

plus the adoption gates: bit-identity (eager vs jit vs vmap on the first --bit-check-obs obs)
and metric-level FoM3 deviation. Records host load + GPU co-tenancy alongside the timings
(feedback_benchmark_dont_assume). CPU-light; ONE GPU process — run on GPU 1 when free.

Typical invocation:
  python bench_sample_jit.py --cuda-visible-devices 1 --n-obs 200 --vmap-batches 16,64,256
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

REPO = "/mnt/home/tersenov/software/cnn_sbi"
SBI = f"{REPO}/scripts/sbi"
FC = Path(f"{SBI}/results/exploratory/flatsky_cross_2026_06")
CNNP = FC / "cnn_phase"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", default=str(CNNP / "representative_corner/flat_none/ckpts/s41"),
                   help="jaxili checkpoint root of ONE trained NDE (CNN 10-d arm reloads bit-exact)")
    p.add_argument("--train-cache-dir", default=str(CNNP / "cnn_none_s41/cache"))
    p.add_argument("--cache-prefix", default="cnn")
    p.add_argument("--fiducial-summaries-npz",
                   default=str(CNNP / "fiducial_summaries/fiducial_summaries_none.npz"))
    p.add_argument("--preproc-transform", default="none")
    p.add_argument("--clip-value", type=float, default=0.0)
    p.add_argument("--min-feature-variance", type=float, default=1e-12)
    p.add_argument("--seed", type=int, default=41, help="NDE seed (sets the production key pattern)")
    p.add_argument("--n-obs", type=int, default=200)
    p.add_argument("--m-samples", type=int, default=2000)
    p.add_argument("--vmap-batches", default="16,64,256")
    p.add_argument("--bit-check-obs", type=int, default=10)
    p.add_argument("--cuda-visible-devices", default="1")
    p.add_argument("--out", default=str(FC / "bench_sample_jit.json"))
    return p.parse_args()


def host_context():
    ctx = {"loadavg": os.getloadavg(), "time_utc": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        ctx["nvidia_smi"] = smi.stdout.strip().splitlines()
    except Exception as e:
        ctx["nvidia_smi"] = f"unavailable: {e}"
    return ctx


def main():
    a = parse_args()
    sys.path.insert(0, REPO); sys.path.insert(0, SBI)
    from train_jaxili_from_compressed import setup_env, compute_fom3
    setup_env(a.cuda_visible_devices)
    import jax, jax.numpy as jnp
    from jaxili.inference import NPE
    from npe_l1norm_cross_jaxili_nbody_tomo import (
        preprocess_summaries, filter_zero_variance_bins,
        _resolve_latest_jaxili_checkpoint_dir, _normalize_jaxili_hparams_embedding_arrays)

    # --- data + obs, exactly the sweep's preprocessing path ---
    cdir = Path(a.train_cache_dir)
    tr = np.load(cdir / f"{a.cache_prefix}_train.npz"); va = np.load(cdir / f"{a.cache_prefix}_val.npz")
    x_tr_raw = tr["x"].astype(np.float64); x_va_raw = va["x"].astype(np.float64)
    clip = a.clip_value if a.clip_value > 0 else None
    tr_p, _, _, mean, std = preprocess_summaries(
        x_tr_raw, x_va_raw[:1], x_va_raw[:1], summary_transform=a.preproc_transform, clip_value=clip)
    mask, _ = filter_zero_variance_bins(tr_p, min_variance=a.min_feature_variance, verbose=False)
    fz = np.load(a.fiducial_summaries_npz)
    _, _, fid_p, _, _ = preprocess_summaries(
        x_tr_raw, x_va_raw[:1], fz["S"].astype(np.float64)[:a.n_obs],
        summary_transform=a.preproc_transform, clip_value=clip, mean=mean, std=std)
    x_obs = fid_p[:, mask].astype(np.float32)
    N, M, seed = x_obs.shape[0], a.m_samples, a.seed
    xdim = int(mask.sum())
    print(f"obs {x_obs.shape}, M={M}, seed={seed}", flush=True)

    # --- reload the trained posterior (CNN 10-d reloads bit-exact; abs path) ---
    vdir = _resolve_latest_jaxili_checkpoint_dir(Path(a.ckpt_root))
    _normalize_jaxili_hparams_embedding_arrays(vdir)
    exmp = (jnp.zeros((1, 6), jnp.float32), jnp.zeros((1, xdim), jnp.float32))
    inf = NPE.load_from_checkpoints(checkpoint=str(vdir), exmp_input=exmp)
    post = inf.build_posterior()
    params = post.state.params
    model = post.model
    keys = [jax.random.PRNGKey(seed * 100003 + i) for i in range(N)]
    x_dev = jnp.asarray(x_obs)

    results = {"context": host_context(), "n_obs": N, "m_samples": M, "xdim": xdim,
               "ckpt_root": a.ckpt_root}

    # --- 1. baseline: the production eager loop ---
    t0 = time.time()
    base = [np.asarray(post.sample(x=jnp.asarray(x_obs[i]), num_samples=M, key=keys[i]))
            for i in range(N)]
    t_base = time.time() - t0
    results["baseline_s"] = t_base
    results["baseline_ms_per_obs"] = 1e3 * t_base / N
    print(f"baseline (eager loop): {t_base:.1f}s  ({1e3*t_base/N:.0f} ms/obs)", flush=True)

    # --- 2. jit closure, same keys ---
    fn = jax.jit(lambda x, k: model.apply({"params": params}, x, M, k, method="sample"))
    t0 = time.time(); _ = np.asarray(fn(x_dev[0], keys[0])); t_compile = time.time() - t0
    t0 = time.time()
    jitted = [np.asarray(fn(x_dev[i], keys[i])) for i in range(N)]
    t_jit = time.time() - t0
    results["jit_compile_s"] = t_compile
    results["jit_s"] = t_jit
    results["jit_ms_per_obs"] = 1e3 * t_jit / N
    results["jit_speedup"] = t_base / t_jit if t_jit > 0 else None
    print(f"jit: compile {t_compile:.1f}s, steady {t_jit:.1f}s "
          f"({1e3*t_jit/N:.0f} ms/obs, {t_base/t_jit:.1f}x)", flush=True)

    # --- 3. vmapped chunks, same per-obs keys ---
    vfn = jax.jit(jax.vmap(lambda x, k: model.apply({"params": params}, x, M, k, method="sample")))
    results["vmap"] = {}
    vmapped_first = None
    for B in [int(b) for b in a.vmap_batches.split(",") if b.strip()]:
        nb = N // B
        if nb == 0:
            print(f"vmap B={B}: skipped (N={N} < B)", flush=True)
            continue
        kk = jnp.stack(keys[:nb * B]).reshape(nb, B, -1)
        xx = x_dev[:nb * B].reshape(nb, B, -1)
        t0 = time.time(); _ = np.asarray(vfn(xx[0], kk[0])); t_vc = time.time() - t0
        t0 = time.time()
        chunks = [np.asarray(vfn(xx[j], kk[j])) for j in range(nb)]
        t_v = time.time() - t0
        per_obs = 1e3 * t_v / (nb * B)
        results["vmap"][B] = {"compile_s": t_vc, "steady_s": t_v, "ms_per_obs": per_obs,
                              "speedup": (t_base / N) / (t_v / (nb * B))}
        print(f"vmap B={B}: compile {t_vc:.1f}s, steady {t_v:.1f}s over {nb*B} obs "
              f"({per_obs:.0f} ms/obs, {results['vmap'][B]['speedup']:.1f}x)", flush=True)
        if vmapped_first is None:
            vmapped_first = np.concatenate(chunks, 0)[:a.bit_check_obs]

    # --- adoption gates: bit-identity + metric deviation ---
    nb_chk = min(a.bit_check_obs, N)
    gate = {}
    eq_jit = [bool(np.array_equal(base[i], jitted[i])) for i in range(nb_chk)]
    dev_jit = float(max(np.max(np.abs(base[i] - jitted[i])) for i in range(nb_chk)))
    gate["jit_bit_identical"] = all(eq_jit)
    gate["jit_max_abs_dev"] = dev_jit
    if vmapped_first is not None:
        eq_v = [bool(np.array_equal(base[i], vmapped_first[i])) for i in range(nb_chk)]
        dev_v = float(max(np.max(np.abs(base[i] - vmapped_first[i])) for i in range(nb_chk)))
        gate["vmap_bit_identical"] = all(eq_v)
        gate["vmap_max_abs_dev"] = dev_v
    f3_base = np.array([compute_fom3(b)["fom3"] for b in base[:50]])
    f3_jit = np.array([compute_fom3(j)["fom3"] for j in jitted[:50]])
    ok = np.isfinite(f3_base) & np.isfinite(f3_jit)
    gate["fom3_max_rel_dev_jit"] = float(np.max(np.abs(f3_jit[ok] - f3_base[ok]) / f3_base[ok]))
    results["gate"] = gate
    print(f"GATE: jit bit-identical={gate['jit_bit_identical']} (max|Δ|={gate['jit_max_abs_dev']:.2e}); "
          f"vmap bit-identical={gate.get('vmap_bit_identical')} "
          f"(max|Δ|={gate.get('vmap_max_abs_dev', float('nan')):.2e}); "
          f"FoM3 max rel dev (jit, 50 obs) = {gate['fom3_max_rel_dev_jit']:.2e}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
