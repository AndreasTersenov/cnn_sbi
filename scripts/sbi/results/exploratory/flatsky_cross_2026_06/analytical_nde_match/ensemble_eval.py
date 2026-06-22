#!/usr/bin/env python3
"""Compressor-ENSEMBLE per-obs FoM3 over the 9000-fiducial population (the properly-calibrated
joint ℓ1 number). Pools the posteriors of the 3 compressor-seed arms PER physical obs (each
compressor sees the same obs through its own compression), exactly the ensemble that calibrated
the gate (SBC 0.31->0.298 noBNT). Reuses train_sbilens_realnvp + the FoM3 machinery from the
production pipeline. Writes <out>/per_patch_metrics.npz + median_summary.json.

Usage: ensemble_eval.py --mode {nobnt,bnt} --out <dir>
"""
import argparse
import json
import sys
import time
from types import SimpleNamespace
from pathlib import Path
import numpy as np

REPO = "/mnt/home/tersenov/software/cnn_sbi"; SBI = f"{REPO}/scripts/sbi"
A = f"{SBI}/results/exploratory/flatsky_cross_2026_06/analytical_nde_match"
ARMS = {
    "nobnt": [f"{A}/jointl1_nobnt", f"{A}/jointl1_nobnt_s42", f"{A}/jointl1_nobnt_s43"],
    "bnt":   [f"{A}/jointl1_bnt",   f"{A}/jointl1_bnt_s42",   f"{A}/jointl1_bnt_s43"],
}
FLOW = SimpleNamespace(flow_total_steps=50000, flow_batch_size=128, flow_lr_init=1e-3,
                       flow_lr_end=1e-5, flow_save_every=2000, flow_patience=20,
                       flow_grad_clip=1.0, flow_weight_decay=1e-4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["nobnt", "bnt"], default="nobnt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cuda-visible-devices", default="2")
    ap.add_argument("--m-samples", type=int, default=2000)
    ap.add_argument("--n-obs", type=int, default=9000)
    ap.add_argument("--max-perm", type=int, default=50)
    ap.add_argument("--seeds", default="41,42,43")
    a = ap.parse_args()
    sys.path.insert(0, REPO); sys.path.insert(0, SBI)
    from train_jaxili_from_compressed import setup_env, compute_fom3, marginal_stats, fom2d
    setup_env(a.cuda_visible_devices)
    import jax, jax.numpy as jnp
    from npe_l1norm_cross_jaxili_nbody_tomo import preprocess_summaries, filter_zero_variance_bins
    from train_nde_from_compressed import train_sbilens_realnvp

    seeds = [int(s) for s in a.seeds.split(",")]
    arms = []
    sel_ref = perm_ref = patch_ref = truth = None
    for adir in ARMS[a.mode]:
        cdir = Path(adir) / "cache"
        tr = np.load(cdir / "l1_train.npz"); va = np.load(cdir / "l1_val.npz")
        theta_tr = tr["theta"].astype(np.float32)
        x_tr_raw = tr["x"].astype(np.float64); x_va_raw = va["x"].astype(np.float64)
        tr_p, _, _, mean, std = preprocess_summaries(
            x_tr_raw, x_va_raw[:1], x_va_raw[:1], summary_transform="none", clip_value=None)
        mask, _ = filter_zero_variance_bins(tr_p, min_variance=1e-12, verbose=False)
        x_tr = tr_p[:, mask].astype(np.float32); dim = x_tr.shape[1]
        fz = np.load(str(Path(adir) / "fiducial_summaries.npz"))
        S = fz["S"].astype(np.float64); perm = fz["perm"]
        sel = np.where(perm < a.max_perm)[0][:a.n_obs]
        _, _, fid_p, _, _ = preprocess_summaries(
            x_tr_raw, x_va_raw[:1], S[sel], summary_transform="none", clip_value=None, mean=mean, std=std)
        x_obs = fid_p[:, mask].astype(np.float32)
        if sel_ref is None:
            sel_ref, perm_ref, patch_ref = sel, perm[sel], fz["patch"][sel]
            truth = next((fz[k] for k in ("truth", "theta") if k in fz.files), None)
        else:
            assert np.array_equal(sel, sel_ref), f"{adir}: sel misaligned vs first arm"
        out = Path(adir) / "_tmp_ens_flow"; out.mkdir(parents=True, exist_ok=True)
        print(f"[{Path(adir).name}] train dim {dim} ...", flush=True)
        samplers = train_sbilens_realnvp(4, 128, theta_tr, x_tr, seeds, a.m_samples, dim, out, FLOW)
        arms.append((samplers, jnp.asarray(x_obs)))

    N = arms[0][1].shape[0]
    print(f"[ensemble] pooling {len(arms)} compressors x {len(seeds)} seeds over {N} obs", flush=True)
    sig = np.full((N, 3), np.nan); pair = np.full((N, 3), np.nan); fom3 = np.full(N, np.nan)
    t0 = time.time()
    for i in range(N):
        pooled = []
        for samplers, x_dev in arms:
            for seed, fn in samplers:
                k = jax.random.PRNGKey(seed * 100003 + int(sel_ref[i]))
                pooled.append(np.asarray(fn(x_dev[i], k)))
        ps = np.concatenate(pooled, 0); ps = ps[np.all(np.isfinite(ps), 1)]
        if ps.shape[0] < 100:
            continue
        ms = marginal_stats(ps); f2 = fom2d(ps)
        sig[i] = [ms["sigma"][q] for q in ("Omega_m", "sigma_8", "w_0")]
        pair[i] = [f2["fom2d_Omega_m_sigma_8"], f2["fom2d_Omega_m_w_0"], f2["fom2d_sigma_8_w_0"]]
        fom3[i] = compute_fom3(ps)["fom3"]
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{N} ({time.time()-t0:.0f}s)", flush=True)
    outd = Path(a.out); outd.mkdir(parents=True, exist_ok=True)
    np.savez(outd / "per_patch_metrics.npz", sigma=sig, fom2d=pair, fom3=fom3,
             perm=perm_ref, patch=patch_ref, sel=sel_ref,
             truth=(truth if truth is not None else np.array([])))
    g = np.isfinite(fom3)
    med = dict(arm=f"jointl1_{a.mode}_ensemble", n=int(g.sum()),
               sigma_Om=float(np.median(sig[g, 0])), sigma_s8=float(np.median(sig[g, 1])),
               sigma_w0=float(np.median(sig[g, 2])), fom3=float(np.median(fom3[g])))
    json.dump(med, open(outd / "median_summary.json", "w"), indent=2)
    print(f"\n[ENSEMBLE {a.mode}] N={med['n']} median FoM3={med['fom3']:.0f} "
          f"sigma={med['sigma_Om']:.3f}/{med['sigma_s8']:.3f}/{med['sigma_w0']:.3f}", flush=True)


if __name__ == "__main__":
    main()
