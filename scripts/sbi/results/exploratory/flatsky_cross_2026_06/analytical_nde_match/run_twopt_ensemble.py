#!/usr/bin/env python3
"""3-compressor deep-ensemble FoM3 for the twopt-split delta arms (PLAN_2PT_SPLIT.md).

The single-compressor product_cov FAILed calibration (n=9000 gate) -> its +708 is inflated by
over-confidence. The joint-l1 fix: pool 3 VMIM compressors (seeds 41/42/43) per obs, which
de-inflates + calibrates. This reproduces that for auto_cov / conv_cov / product_cov to get a
calibration-honest ΔNG(product) (and firm up ΔNG(conv)).

Step 1: build 3 VMIM compressor dirs per arm (seeds 41/42/43) from the raw cache.
Step 2: pool the 3 compressors x 3 NDE seeds per obs, recompute FoM3 (verbatim ensemble_eval logic).
GPU 0. Resumable. Writes RESULT_TWOPT_SPLIT_ENSEMBLE.md.
"""
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace
from pathlib import Path
import numpy as np

REPO = "/mnt/home/tersenov/software/cnn_sbi"
SBI = f"{REPO}/scripts/sbi"
A = f"{SBI}/results/exploratory/flatsky_cross_2026_06/analytical_nde_match"
TS = f"{A}/twopt_split"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
GPU = "0"
MEM = "0.85"
SEEDS = [41, 42, 43]
ARMS = ["auto_cov", "conv_cov", "product_cov"]
FLOW = SimpleNamespace(flow_total_steps=50000, flow_batch_size=128, flow_lr_init=1e-3,
                       flow_lr_end=1e-5, flow_save_every=2000, flow_patience=20,
                       flow_grad_clip=1.0, flow_weight_decay=1e-4)


def sh(cmd, log):
    env = dict(os.environ, XLA_PYTHON_CLIENT_MEM_FRACTION=MEM, CUDA_VISIBLE_DEVICES=GPU)
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"  [{(time.time()-t0)/60:.1f} min rc={rc}] {os.path.basename(log)}", flush=True)
    return rc


def build_compressors():
    """Step 1 (subprocesses): 3 VMIM compressor dirs per arm."""
    for arm in ARMS:
        raw_cc, raw_cf = f"{TS}/{arm}/raw_cache", f"{TS}/{arm}/raw_fid.npz"
        for s in SEEDS:
            cdir = f"{TS}/{arm}/ens/s{s}"
            if Path(f"{cdir}/cache/l1_train.npz").exists():
                continue
            print(f"[build] {arm} s{s} VMIM ...", flush=True)
            sh([PY, f"{SBI}/vmim_from_cache.py", "--cache-dir", raw_cc, "--fid-npz", raw_cf,
                "--out-cache", f"{cdir}/cache", "--out-fid", f"{cdir}/fiducial_summaries.npz",
                "--summary-dim", "10", "--seed", str(s), "--cuda-visible-devices", GPU],
               f"{TS}/logs/{arm}_ens_s{s}_vmim.log")


def pool_arm(arm, n_obs=9000, max_perm=50, m_samples=2000):
    """Step 2 (in-process jax): pool the arm's 3 compressors x 3 NDE seeds per obs -> FoM3."""
    sys.path.insert(0, REPO); sys.path.insert(0, SBI)
    from train_jaxili_from_compressed import setup_env, compute_fom3, marginal_stats, fom2d
    setup_env(GPU)
    import jax, jax.numpy as jnp
    from npe_l1norm_cross_jaxili_nbody_tomo import preprocess_summaries, filter_zero_variance_bins
    from train_nde_from_compressed import train_sbilens_realnvp

    arms, sel_ref, truth = [], None, None
    for s in SEEDS:
        adir = Path(f"{TS}/{arm}/ens/s{s}")
        tr = np.load(adir / "cache/l1_train.npz"); va = np.load(adir / "cache/l1_val.npz")
        theta_tr = tr["theta"].astype(np.float32)
        x_tr_raw = tr["x"].astype(np.float64); x_va_raw = va["x"].astype(np.float64)
        tr_p, _, _, mean, std = preprocess_summaries(
            x_tr_raw, x_va_raw[:1], x_va_raw[:1], summary_transform="none", clip_value=None)
        mask, _ = filter_zero_variance_bins(tr_p, min_variance=1e-12, verbose=False)
        x_tr = tr_p[:, mask].astype(np.float32); dim = x_tr.shape[1]
        fz = np.load(str(adir / "fiducial_summaries.npz"))
        S = fz["S"].astype(np.float64); perm = fz["perm"]
        sel = np.where(perm < max_perm)[0][:n_obs]
        _, _, fid_p, _, _ = preprocess_summaries(
            x_tr_raw, x_va_raw[:1], S[sel], summary_transform="none", clip_value=None, mean=mean, std=std)
        x_obs = fid_p[:, mask].astype(np.float32)
        if sel_ref is None:
            sel_ref = sel; truth = next((fz[k] for k in ("truth", "theta") if k in fz.files), None)
        else:
            assert np.array_equal(sel, sel_ref), f"{adir}: sel misaligned"
        out = adir / "_tmp_ens_flow"; out.mkdir(parents=True, exist_ok=True)
        print(f"  [{arm}/s{s}] train dim {dim} ...", flush=True)
        samplers = train_sbilens_realnvp(4, 128, theta_tr, x_tr, SEEDS, m_samples, dim, out, FLOW)
        arms.append((samplers, jnp.asarray(x_obs)))

    N = arms[0][1].shape[0]
    print(f"  [ensemble {arm}] pooling {len(arms)}x{len(SEEDS)} over {N} obs", flush=True)
    fom3 = np.full(N, np.nan); sig = np.full((N, 3), np.nan); t0 = time.time()
    for i in range(N):
        pooled = []
        for samplers, x_dev in arms:
            for seed, fn in samplers:
                k = jax.random.PRNGKey(seed * 100003 + int(sel_ref[i]))
                pooled.append(np.asarray(fn(x_dev[i], k)))
        ps = np.concatenate(pooled, 0); ps = ps[np.all(np.isfinite(ps), 1)]
        if ps.shape[0] < 100:
            continue
        ms = marginal_stats(ps)
        sig[i] = [ms["sigma"][q] for q in ("Omega_m", "sigma_8", "w_0")]
        fom3[i] = compute_fom3(ps)["fom3"]
        if (i + 1) % 2000 == 0:
            print(f"    {i+1}/{N} ({time.time()-t0:.0f}s)", flush=True)
    outd = Path(f"{TS}/{arm}/ens"); outd.mkdir(parents=True, exist_ok=True)
    g = np.isfinite(fom3)
    med = dict(arm=f"{arm}_ensemble", n=int(g.sum()),
               sigma_Om=float(np.median(sig[g, 0])), sigma_s8=float(np.median(sig[g, 1])),
               sigma_w0=float(np.median(sig[g, 2])), fom3=float(np.median(fom3[g])))
    json.dump(med, open(outd / "median_summary.json", "w"), indent=2)
    print(f"  [ENSEMBLE {arm}] N={med['n']} FoM3={med['fom3']:.0f}", flush=True)
    return med["fom3"]


def main():
    Path(f"{TS}/logs").mkdir(parents=True, exist_ok=True)
    build_compressors()          # all VMIM subprocesses first (before importing jax)
    res = {}
    for arm in ARMS:
        mj = Path(f"{TS}/{arm}/ens/median_summary.json")
        res[arm] = json.load(open(mj))["fom3"] if mj.exists() else pool_arm(arm)
        print(f"  -> {arm} ensemble FoM3 = {res[arm]}", flush=True)

    def dd(a, b):
        return None if (res.get(a) is None or res.get(b) is None) else res[a] - res[b]
    dc, dp = dd("conv_cov", "auto_cov"), dd("product_cov", "auto_cov")
    rows = ["# Two-point split — 3-COMPRESSOR ENSEMBLE (de-inflated, n=9000)",
            "Pool VMIM seeds 41/42/43 x 3 NDE seeds per obs (same fix as joint-l1).",
            "Single-compressor gated refs: auto_cov 2916 (PASS-caveat), conv_cov 3221 (PASS-caveat),"
            " product_cov 3624 (FAIL, over-confident).", "",
            "| arm | ensemble FoM3 |", "|---|---|"]
    for a in ARMS:
        v = res.get(a); rows.append(f"| {a} | {v:.0f} |" if v else f"| {a} | FAILED |")
    rows += ["",
             f"- ΔNG(conv)    conv_cov - auto_cov    = {None if dc is None else round(dc)}",
             f"- ΔNG(product) product_cov - auto_cov = {None if dp is None else round(dp)}",
             "", "De-inflated (ensemble) deltas. TARP/SBC gate on the pooled posteriors is the"
             " follow-up confirmation (precedent: joint-l1 + BNT-autoprod ensembles calibrated)."]
    open(f"{TS}/RESULT_TWOPT_SPLIT_ENSEMBLE.md", "w").write("\n".join(rows) + "\n")
    print("\n" + "\n".join(rows), flush=True)


if __name__ == "__main__":
    main()
