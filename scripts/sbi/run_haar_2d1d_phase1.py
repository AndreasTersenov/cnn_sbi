#!/usr/bin/env python
"""Phase-1 orchestrator for the 2D-1D Haar wavelet-ℓ1 arms (PLAN_2D1D_PHASE_1_2.md).

Runs, SEQUENTIALLY on one GPU (co-tenant-safe), the trusted per-stage scripts:
  build (build_flatsky_haar_arm.py) -> sweep (population_sweep_flatsky.py)
  -> gate (tarp_stratified_val.py + run_tarp_coverage.py) -> verdicts (copied from
  run_laneB_gate_c.py verbatim).
Resumable: each stage is skipped if its output already exists. Produces ARTIFACTS only
(median_summary.json, dumps, curves, verdicts.json, a GATE_C table) — the scientific
reading / RESULT doc is written by hand afterwards.

Arms:
  haar_nobnt     : mix=haar      (faithful 2D-1D Haar ℓ1, no-BNT)            [goal 1]
  haar_bnt_uncut : mix=haar_bnt  (Haar across uncut BNT channels)            [goal 2]
  autohaar_nobnt : flat_none autos (+) haar_nobnt channels (concat)         [goal 1 augmented]
"""
import argparse, glob, json, os, subprocess, time, zipfile
import numpy as np
from numpy.lib import format as npformat
from pathlib import Path

REPO = "/mnt/home/tersenov/software/cnn_sbi"; SBI = f"{REPO}/scripts/sbi"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
FS = f"{SBI}/results/exploratory/flatsky_cross_2026_06"
OM2 = f"{FS}/overnight_menu_2"
LC = f"{FS}/gate_c/lc2st"                          # fiducial summaries dir
GC = f"{OM2}/gate_c_2d1d"; LOGS = f"{OM2}/logs"
SWEEP = f"{FS}/population_sweep"
FLATNONE = f"{FS}/l1_matrix/l1_none_cache/flat_local_none"
FID_NONE = f"{LC}/fiducial_summaries_none.npz"

ARMS = ["haar_nobnt", "haar_bnt_uncut", "autohaar_nobnt"]
MIX = {"haar_nobnt": "haar", "haar_bnt_uncut": "haar_bnt"}     # autohaar built by concat
TERCILES = ("LOW", "MID", "HIGH")
DEV_PASS, DEV_CAVEAT = 0.05, 0.10
STD_LO, STD_HI, STD_FAIL = 0.275, 0.305, 0.02


def sh(cmd, log, env_extra=None):
    env = dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="3",
               XLA_PYTHON_CLIENT_PREALLOCATE="false",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", CNN_CPU_THREADS="8")
    if env_extra:
        env.update(env_extra)
    print(f"  $ {' '.join(cmd)}  (log {log})", flush=True)
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=SBI, env=env, stdout=f, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL).returncode
    if rc != 0:
        print(f"  [FAIL rc={rc}] see {log}", flush=True)
    return rc


def cache_ok(arm):
    return Path(OM2, arm, "cache", "l1_train.npz").exists() and \
           Path(OM2, arm, "cache", "l1_val.npz").exists()


def build_haar(arm, gpu):
    if cache_ok(arm):
        print(f"[build] {arm} cache exists -> skip", flush=True); return 0
    return sh([PY, "build_flatsky_haar_arm.py", "--mix", MIX[arm],
               "--out-cache", f"{OM2}/{arm}/cache",
               "--out-fid", f"{LC}/fiducial_summaries_{arm}.npz"],
              f"{LOGS}/build_{arm}.log", {"CUDA_VISIBLE_DEVICES": str(gpu)})


def build_autohaar():
    arm = "autohaar_nobnt"
    if cache_ok(arm):
        print(f"[build] {arm} cache exists -> skip", flush=True); return 0
    print(f"[build] {arm} = concat(flat_none, haar_nobnt)", flush=True)
    os.makedirs(f"{OM2}/{arm}/cache", exist_ok=True)
    for split in ("train", "val"):
        a = np.load(f"{FLATNONE}/l1_{split}.npz"); b = np.load(f"{OM2}/haar_nobnt/cache/l1_{split}.npz")
        assert np.array_equal(a["theta"].astype(np.float64), b["theta"].astype(np.float64)), \
            f"{split} theta mismatch flat_none vs haar_nobnt"
        x = np.concatenate([a["x"], b["x"]], axis=1).astype(np.float32)
        np.savez(f"{OM2}/{arm}/cache/l1_{split}.npz", theta=a["theta"], x=x)
        print(f"  {split}: {a['x'].shape[1]} (+) {b['x'].shape[1]} -> {x.shape}", flush=True)
    # fiducial concat
    fa = np.load(FID_NONE); fb = np.load(f"{LC}/fiducial_summaries_haar_nobnt.npz")
    assert np.array_equal(fa["perm"], fb["perm"]) and np.array_equal(fa["patch"], fb["patch"])
    S = np.concatenate([fa["S"], fb["S"]], axis=1).astype(np.float32)
    out = {"S": S, "perm": fa["perm"], "patch": fa["patch"]}
    for k in ("truth", "theta"):
        if k in fa.files:
            out[k] = fa[k]
    np.savez(f"{LC}/fiducial_summaries_{arm}.npz", **out)
    print(f"  fiducial S -> {S.shape}", flush=True)
    return 0


def sweep(arm, gpu, mem):
    out = f"{SWEEP}/{arm}"
    if Path(out, "median_summary.json").exists():
        print(f"[sweep] {arm} done -> skip", flush=True); return 0
    return sh([PY, "population_sweep_flatsky.py", "--train-cache-dir", f"{OM2}/{arm}/cache",
               "--cache-prefix", "l1", "--arm-label", arm,
               "--fiducial-summaries-npz", f"{LC}/fiducial_summaries_{arm}.npz",
               "--output-dir", out, "--preproc-transform", "log1p-zscore",
               "--clip-value", "5.0", "--min-feature-variance", "1e-5",
               "--seeds", "41,42,43", "--cuda-visible-devices", str(gpu)],
              f"{LOGS}/sweep_{arm}.log", {"XLA_PYTHON_CLIENT_MEM_FRACTION": str(mem)})


def gate_tarp(arm, gpu, mem):
    if glob.glob(f"{GC}/tarp_drp/dumps/{arm}_*/seed_*/n*_m*/posterior_samples.npz"):
        print(f"[gate] {arm} dumps exist -> skip", flush=True); return 0
    return sh([PY, "tarp_stratified_val.py", "--train-cache-dir", f"{OM2}/{arm}/cache",
               "--cache-prefix", "l1", "--arm-label", arm,
               "--dumps-root", f"{GC}/tarp_drp/dumps", "--preproc-transform", "log1p-zscore",
               "--clip-value", "5", "--min-feature-variance", "1e-5", "--seeds", "41,42,43",
               "--cuda-visible-devices", str(gpu)],
              f"{LOGS}/tarp_{arm}.log", {"XLA_PYTHON_CLIENT_MEM_FRACTION": str(mem)})


# ---- verdict helpers (verbatim from run_laneB_gate_c.py) ----
def sbc_from_dumps(arm):
    rs = []
    for f in sorted(glob.glob(f"{GC}/tarp_drp/dumps/{arm}_*/seed_*/n*_m*/posterior_samples.npz")):
        z = np.load(f); rs.append((z["samples"] < z["theta"][:, None, :]).mean(1))
    if not rs:
        return None
    from scipy import stats as st
    r = np.concatenate(rs, 0)
    return {"n": int(r.shape[0]), "mean": [float(m) for m in r.mean(0)[:3]],
            "std": [float(s) for s in r.std(0)[:3]],
            "min_ks_p": float(min(st.kstest(r[:, i], "uniform").pvalue for i in range(3)))}


def tarp_devs(arm, dim=3):
    out = {}
    for terc in TERCILES:
        worst = None
        for f in sorted(glob.glob(f"{GC}/tarp_drp/curves/tarp_curve_{arm}_{terc}_seed*_dim{dim}.npz")):
            z = np.load(f); a = np.asarray(z["alpha"]); e = np.asarray(z["ecp_bootstrap"]).mean(0)
            i = int(np.argmax(np.abs(e - a))); d = float(e[i] - a[i])
            if worst is None or abs(d) > abs(worst):
                worst = d
        if worst is not None:
            out[terc] = worst
    return out


def net_bias(arm, dim=3):
    devs = []
    for f in glob.glob(f"{GC}/tarp_drp/curves/tarp_curve_{arm}_*_seed*_dim{dim}.npz"):
        z = np.load(f); a = z["alpha"]; e = z["ecp_bootstrap"].mean(0)
        devs.append(float(np.trapz(e - a, a) * 2))
    return (float(np.mean(devs)), float(np.std(devs))) if devs else (None, None)


def verdict(devs, sbc):
    if not devs or not sbc:
        return "INCOMPLETE"
    worst = max(abs(d) for d in devs.values())
    stds = sbc["std"]
    std_off = max((max(0.0, STD_LO - s, s - STD_HI) for s in stds), default=9)
    if worst > DEV_CAVEAT or std_off >= STD_FAIL:
        return "FAIL"
    if worst <= DEV_PASS and std_off == 0.0:
        return "PASS"
    return "PASS-with-caveat"


def write_artifacts():
    rows, vj = [], {}
    for arm in ARMS:
        ms = {}
        msf = Path(SWEEP, arm, "median_summary.json")
        if msf.exists():
            ms = json.loads(msf.read_text())
        d = tarp_devs(arm); s = sbc_from_dumps(arm); nb = net_bias(arm); v = verdict(d, s)
        vj[arm] = {"fom3": ms.get("fom3"), "sigma": [ms.get("sigma_Om"), ms.get("sigma_s8"),
                   ms.get("sigma_w0")], "tarp": d, "net_bias": nb, "sbc": s, "verdict": v}
        dcell = "/".join(f"{d[t]:+.3f}" if t in d else "—" for t in TERCILES)
        nbcell = f"{nb[0]:+.3f}±{nb[1]:.3f}" if nb[0] is not None else "—"
        scell = ",".join(f"{x:.3f}" for x in s["std"]) if s else "—"
        f3 = f"{ms.get('fom3'):.0f}" if ms.get("fom3") else "—"
        sig = (f"{ms.get('sigma_Om'):.3f},{ms.get('sigma_s8'):.3f},{ms.get('sigma_w0'):.3f}"
               if ms.get("sigma_Om") else "—")
        rows.append(f"| {arm} | {f3} | {sig} | {dcell} | {nbcell} | {scell} | **{v}** |")
    Path(GC).mkdir(parents=True, exist_ok=True)
    json.dump(vj, open(f"{GC}/verdicts.json", "w"), indent=2)
    L = ["# GATE C — 2D-1D Haar arms (artifacts; reading added by hand)\n",
         "Baselines: flat_none 2405, flat_product 2875 (same common-MAF path).\n",
         "| arm | FoM3 | σ(Om,s8,w0) | TARP LOW/MID/HIGH | net bias | SBC std | verdict |",
         "|---|---|---|---|---|---|---|", *rows,
         "", "(+ net bias = conservative/over-covers; − = over-confident. SBC uniform=0.289.)"]
    Path(GC, "GATE_C_2D1D.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    print(f"\nwrote {GC}/verdicts.json + GATE_C_2D1D.md", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--mem-fraction", type=float, default=0.30)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True); os.makedirs(f"{GC}/tarp_drp", exist_ok=True)
    t0 = time.time()
    if not a.report_only:
        for arm in ("haar_nobnt", "haar_bnt_uncut"):
            print(f"\n===== BUILD {arm} ({time.time()-t0:.0f}s) =====", flush=True)
            build_haar(arm, a.gpu)
        print(f"\n===== BUILD autohaar_nobnt ({time.time()-t0:.0f}s) =====", flush=True)
        build_autohaar()
        for arm in ARMS:
            print(f"\n===== SWEEP {arm} ({time.time()-t0:.0f}s) =====", flush=True)
            sweep(arm, a.gpu, a.mem_fraction)
        for arm in ARMS:
            print(f"\n===== GATE/TARP {arm} ({time.time()-t0:.0f}s) =====", flush=True)
            gate_tarp(arm, a.gpu, a.mem_fraction)
        print(f"\n===== COVERAGE ({time.time()-t0:.0f}s) =====", flush=True)
        sh([PY, "run_tarp_coverage.py", "--dumps-root", f"{GC}/tarp_drp/dumps",
            "--outdir", f"{GC}/tarp_drp", "--dims", "3"], f"{LOGS}/coverage.log",
           {"CUDA_VISIBLE_DEVICES": str(a.gpu)})
    write_artifacts()
    print(f"\nPHASE 1 ORCHESTRATION DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
