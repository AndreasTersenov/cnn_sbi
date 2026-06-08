#!/usr/bin/env python3
"""Thread 1 (geometry map) — systematic per-patch-INDEX re-sampling for ONE arm.

Reuses the campaign-EXACT NDE + preprocessing (same as fiducial_analyze.py, which is
G3-validated) but, instead of step1(mean-dv)/step2(300 random patches), sweeps a
BALANCED grid `patch_indices x perms` (default all 48 indices x all 200 perms) and
records, per (patch_idx, perm):

  fom3, valid_fom3, fom2d_* (3 pairs), sig_* (6), mean_* (6), bias_* (6), pull_* (6)

mean/bias/pull are the NEW columns (the existing per_patch_fom.csv lacks the posterior
mean -> bias was unavailable). bias = mean - FIDUCIAL; pull = bias/sigma (SBC z-score).
Geometry (lon,lat per patch index from the perm-invariant patch_centers) is attached.

This single grid serves Thread 1 (geometry map vs latitude), Thread 2 (geometry-vs-
realization variance decomposition; balanced layout), Thread 3 (per-index bias structure).

G3 end-to-end gate (perm0/patch0 3-seed-pooled FoM3 within --g3-tol of the campaign value)
runs before the sweep; FAIL => abort (no garbage), identical to fiducial_analyze.py.
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
DC = REPO / "results/exploratory/definitive_comparison"
FIDCACHE = REPO / "results/exploratory/cross_maps_campaign/full_sphere_cache_fiducial"


def parse_intset(spec: str) -> list[int]:
    """'0-47' or '0,1,5-7' -> sorted unique int list."""
    out: set[int] = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(tok))
    return sorted(out)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache-dir", required=True)
    p.add_argument("--cache-prefix", choices=["cnn", "l1"], required=True)
    p.add_argument("--summaries-npz", required=True)
    p.add_argument("--arm-label", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--preproc-transform", default="none",
                   choices=["none", "zscore", "log1p-zscore", "log10p-zscore"])
    p.add_argument("--clip-value", type=float, default=0.0, help="0 => no clip")
    p.add_argument("--min-feature-variance", type=float, default=1e-12)
    p.add_argument("--seeds", default="41,42,43")
    p.add_argument("--patch-indices", default="0-47")
    p.add_argument("--perms", default="0-199")
    p.add_argument("--samples-per-seed", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--decay-steps", type=int, default=10000)
    p.add_argument("--expected-fom3", type=float, default=0.0, help="G3 target (perm0/patch0); 0 => skip")
    p.add_argument("--g3-tol", type=float, default=0.20)
    p.add_argument("--flush-every", type=int, default=200, help="rewrite CSV every N patches")
    p.add_argument("--cuda-visible-devices", default="1")
    p.add_argument("--fid-cache-dir", default="",
                   help="Fiducial cache for patch_centers (geometry). Default = the 20deg "
                        "cache; for 10deg pass full_sphere_cache_fiducial_10deg.")
    return p.parse_args()


def main():
    a = parse_args()
    sys.path.insert(0, str(REPO))
    from train_jaxili_from_compressed import (
        setup_env, compute_fom3, fom2d, marginal_stats, PARAM_KEYS, FIDUCIAL,
    )
    setup_env(a.cuda_visible_devices)
    import jax, jax.numpy as jnp
    from jaxili.inference import NPE
    from npe_l1norm_cross_jaxili_nbody_tomo import (
        train_with_nan_retry, preprocess_summaries, filter_zero_variance_bins,
    )

    cdir = Path(a.train_cache_dir)
    tr = np.load(cdir / f"{a.cache_prefix}_train.npz")
    theta_tr = tr["theta"].astype(np.float32)
    x_tr_raw = tr["x"].astype(np.float64)

    z = np.load(a.summaries_npz)
    S_raw = z["S"].astype(np.float64)
    perm = z["perm"].astype(int); patch = z["patch"].astype(int)
    # Bias uses FIDUCIAL (the train-cache convention, h0=0.6736, verified on both train caches).
    # The summaries `theta` is metadata only and the CNN extractor stored RAW H0 (67.36) while
    # L1 stored H0/100 (0.6736) -> verify only the 3 FoM3 params; warn on the h0 convention.
    truth = np.asarray(FIDUCIAL, float)
    meta_truth = np.asarray(z["theta"][0], float)
    assert np.allclose(meta_truth[:3], FIDUCIAL[:3]), \
        f"FoM3-param truth mismatch: {meta_truth[:3]} vs {FIDUCIAL[:3]}"
    if not np.allclose(meta_truth, FIDUCIAL):
        print(f"  NOTE: summary-meta theta {meta_truth} differs from FIDUCIAL {FIDUCIAL} "
              f"(h0 unit convention); bias uses FIDUCIAL.", flush=True)
    assert S_raw.shape[1] == x_tr_raw.shape[1], \
        f"dim mismatch S={S_raw.shape[1]} vs train={x_tr_raw.shape[1]}"

    # (perm,patch) -> global row, robust to ordering
    pos = {(int(perm[k]), int(patch[k])): k for k in range(len(perm))}

    # geometry: perm-invariant patch_centers (lon,lat) from the fiducial cache
    fidcache = Path(a.fid_cache_dir) if a.fid_cache_dir else FIDCACHE
    centers = np.asarray(np.load(fidcache / "nobnt" / "obs" / "cosmo_fiducial_perm0.npz")["patch_centers"])

    indices = parse_intset(a.patch_indices)
    perms = parse_intset(a.perms)
    print(f"[{a.arm_label}] train x{x_tr_raw.shape}; S{S_raw.shape} "
          f"({len(np.unique(perm))} perms); grid {len(indices)} indices x {len(perms)} perms "
          f"= {len(indices)*len(perms)} patches; transform={a.preproc_transform} "
          f"clip={a.clip_value} min_var={a.min_feature_variance}", flush=True)

    # ---- preprocessing fit on TRAIN, applied to S (campaign functions; identical to fiducial_analyze) ----
    clip = a.clip_value if a.clip_value > 0 else None
    tr_proc, _, S_proc, mean, std = preprocess_summaries(
        x_tr_raw, x_tr_raw[:1], S_raw, summary_transform=a.preproc_transform, clip_value=clip)
    mask, n_removed = filter_zero_variance_bins(tr_proc, min_variance=a.min_feature_variance, verbose=False)
    x_tr = tr_proc[:, mask].astype(np.float32)
    S = S_proc[:, mask].astype(np.float32)
    print(f"  masked dim {x_tr.shape[1]}/{mask.size} (removed {n_removed})", flush=True)

    # ---- train one NDE per seed (campaign-exact) ----
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    out_dir = Path(a.output_dir) / a.arm_label
    out_dir.mkdir(parents=True, exist_ok=True)
    posteriors = []
    params = jnp.asarray(theta_tr); data = jnp.asarray(x_tr)
    for seed in seeds:
        t0 = time.time()
        sk = jax.random.PRNGKey(int(seed) + 1)
        inf = NPE().append_simulations(params, data, key=sk)
        ckpt = str((out_dir / f"ckpt_{a.arm_label}_s{seed}").resolve())
        inf, _m, _d = train_with_nan_retry(inf, ckpt, a.epochs, a.learning_rate,
                                           a.batch_size, a.warmup_steps, a.decay_steps,
                                           params, data, sk)
        posteriors.append((seed, inf.build_posterior()))
        print(f"  NDE seed {seed} trained in {time.time()-t0:.0f}s", flush=True)

    def sample_pooled(x_obs, n_per_seed):
        out = []
        for seed, post in posteriors:
            k = jax.random.PRNGKey(int(seed) + 7)
            out.append(np.asarray(post.sample(x=jnp.asarray(x_obs), num_samples=n_per_seed, key=k)))
        s = np.concatenate(out, 0)
        return s[np.all(np.isfinite(s), axis=1)]

    # ---- G3: reproduce campaign perm0/patch0 FoM3 before trusting the sweep ----
    if a.expected_fom3 > 0:
        g3 = sample_pooled(S[pos[(0, 0)]], a.samples_per_seed)
        f0 = compute_fom3(g3)["fom3"]
        rel = abs(f0 - a.expected_fom3) / a.expected_fom3
        ok = rel <= a.g3_tol
        print(f"  [G3] perm0/patch0 FoM3={f0:.0f} vs campaign {a.expected_fom3:.0f} "
              f"(rel {rel*100:.1f}%) -> {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            raise SystemExit(f"[G3] FAILED for {a.arm_label}: {f0:.0f} vs {a.expected_fom3:.0f} "
                             f"(rel {rel*100:.1f}% > {a.g3_tol*100:.0f}%); aborting (no garbage).")

    # ---- balanced grid sweep ----
    csv_path = out_dir / "per_patch_grid.csv"
    fieldnames = (["patch", "perm", "patch_global_idx", "lon", "lat",
                   "fom3", "valid_fom3",
                   "fom2d_Omega_m_sigma_8", "fom2d_Omega_m_w_0", "fom2d_sigma_8_w_0"]
                  + [f"sig_{p}" for p in PARAM_KEYS]
                  + [f"mean_{p}" for p in PARAM_KEYS]
                  + [f"bias_{p}" for p in PARAM_KEYS]
                  + [f"pull_{p}" for p in PARAM_KEYS])

    def write_csv(rows):
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(rows)

    rows = []
    missing = []
    total = len(indices) * len(perms)
    t0 = time.time()
    done = 0
    for pi in indices:
        for pm in perms:
            key = (pm, pi)
            if key not in pos:
                missing.append(key); continue
            sp = sample_pooled(S[pos[key]], a.samples_per_seed)
            f3 = compute_fom3(sp); f2 = fom2d(sp); mg = marginal_stats(sp)
            row = {"patch": pi, "perm": pm, "patch_global_idx": int(pos[key]),
                   "lon": float(centers[pi, 0]), "lat": float(centers[pi, 1]),
                   "fom3": f3["fom3"], "valid_fom3": f3["valid_fom3"], **f2}
            for p in PARAM_KEYS:
                sig = mg["sigma"][p]; bias = mg["bias"][p]
                row[f"sig_{p}"] = sig
                row[f"mean_{p}"] = bias + FIDUCIAL[PARAM_KEYS.index(p)]
                row[f"bias_{p}"] = bias
                row[f"pull_{p}"] = bias / sig if sig > 0 else float("nan")
            rows.append(row)
            done += 1
            if done % 50 == 0:
                rate = (time.time() - t0) / done
                eta = rate * (total - done) / 60.0
                print(f"  [sweep] {done}/{total} ({rate:.3f}s/patch, ETA {eta:.1f} min)", flush=True)
            if done % a.flush_every == 0:
                write_csv(rows)

    write_csv(rows)
    # npz mirror (arrays) for the analysis pass
    arr = {k: np.array([r[k] for r in rows]) for k in fieldnames}
    np.savez(out_dir / "per_patch_grid.npz", truth=truth, centers=centers, **arr)
    meta = {"arm": a.arm_label, "n_indices": len(indices), "n_perms": len(perms),
            "n_patches": len(rows), "n_missing": len(missing),
            "samples_per_seed": a.samples_per_seed, "seeds": seeds,
            "rate_s_per_patch": (time.time() - t0) / max(done, 1),
            "preproc": {"transform": a.preproc_transform, "clip": a.clip_value,
                        "min_var": a.min_feature_variance}}
    (out_dir / "sweep_meta.json").write_text(json.dumps(meta, indent=2))
    if missing:
        print(f"  WARNING: {len(missing)} (perm,patch) pairs missing from summaries (skipped)", flush=True)
    print(f"[{a.arm_label}] DONE: {len(rows)} patches, {meta['rate_s_per_patch']:.3f}s/patch. "
          f"-> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
