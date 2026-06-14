#!/usr/bin/env python
"""Phase-2 orchestrator: 2D-1D Haar SCATTERING ℓ1 arms (Approach B). Resumable; same pattern as
run_haar_2d1d_phase1.py (build → sweep → gate → verdicts), reusing the trusted per-stage scripts.

Arms (vs Phase-1 linear: haar_nobnt 2676/FAIL, haar_bnt_uncut 885; baselines auto 2405, product 2875):
  haarscat_nobnt : modulus-Haar ℓ1, no-BNT   [goal 1 — does the modulus beat the ~2900 linear ceiling?]
  haarscat_bnt   : modulus-Haar ℓ1, BNT space [goal 2 — does sum-of-moduli survive BNT (vs 885 collapse)?]
Needs the empirical noise freezes (flatsky_haar_scatter_sigma_{none,bnt}.npz); runs them if missing.
"""
import argparse, glob, json, os, subprocess, time
import numpy as np
from pathlib import Path

REPO = "/mnt/home/tersenov/software/cnn_sbi"; SBI = f"{REPO}/scripts/sbi"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
FS = f"{SBI}/results/exploratory/flatsky_cross_2026_06"
OM2 = f"{FS}/overnight_menu_2"; LC = f"{FS}/gate_c/lc2st"
GC = f"{OM2}/gate_c_haarscat"; LOGS = f"{OM2}/logs"; SWEEP = f"{FS}/population_sweep"

ARMS = ["haarscat_nobnt", "haarscat_bnt"]
PRE = {"haarscat_nobnt": "none", "haarscat_bnt": "bnt"}
TERCILES = ("LOW", "MID", "HIGH")
DEV_PASS, DEV_CAVEAT = 0.05, 0.10
STD_LO, STD_HI, STD_FAIL = 0.275, 0.305, 0.02


def sh(cmd, log, env_extra=None):
    env = dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="3",
               XLA_PYTHON_CLIENT_PREALLOCATE="false",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
    if env_extra:
        env.update(env_extra)
    print(f"  $ {' '.join(cmd)}  (log {log})", flush=True)
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=SBI, env=env, stdout=f, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL).returncode
    if rc != 0:
        print(f"  [FAIL rc={rc}] see {log}", flush=True)
    return rc


def freeze(pre, gpu):
    if Path(f"{FS}/flatsky_haar_scatter_sigma_{pre}.npz").exists():
        print(f"[freeze] {pre} exists -> skip", flush=True); return 0
    return sh([PY, "freeze_haar_scatter_noise.py", "--pre-basis", pre, "--workers", "16"],
              f"{LOGS}/freeze_scatter_{pre}.log",
              {"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})


def build(arm, gpu):
    if Path(OM2, arm, "cache", "l1_train.npz").exists():
        print(f"[build] {arm} exists -> skip", flush=True); return 0
    return sh([PY, "build_flatsky_haar_scatter_arm.py", "--pre-basis", PRE[arm],
               "--out-cache", f"{OM2}/{arm}/cache", "--out-fid", f"{LC}/fiducial_summaries_{arm}.npz"],
              f"{LOGS}/build_{arm}.log", {"CUDA_VISIBLE_DEVICES": str(gpu)})


def sweep(arm, gpu, mem):
    if Path(SWEEP, arm, "median_summary.json").exists():
        print(f"[sweep] {arm} done -> skip", flush=True); return 0
    return sh([PY, "population_sweep_flatsky.py", "--train-cache-dir", f"{OM2}/{arm}/cache",
               "--cache-prefix", "l1", "--arm-label", arm,
               "--fiducial-summaries-npz", f"{LC}/fiducial_summaries_{arm}.npz",
               "--output-dir", f"{SWEEP}/{arm}", "--preproc-transform", "log1p-zscore",
               "--clip-value", "5.0", "--min-feature-variance", "1e-5", "--seeds", "41,42,43",
               "--cuda-visible-devices", str(gpu)],
              f"{LOGS}/sweep_{arm}.log", {"XLA_PYTHON_CLIENT_MEM_FRACTION": str(mem)})


def gate(arm, gpu, mem):
    if glob.glob(f"{GC}/tarp_drp/dumps/{arm}_*/seed_*/n*_m*/posterior_samples.npz"):
        print(f"[gate] {arm} dumps exist -> skip", flush=True); return 0
    return sh([PY, "tarp_stratified_val.py", "--train-cache-dir", f"{OM2}/{arm}/cache",
               "--cache-prefix", "l1", "--arm-label", arm, "--dumps-root", f"{GC}/tarp_drp/dumps",
               "--preproc-transform", "log1p-zscore", "--clip-value", "5", "--min-feature-variance",
               "1e-5", "--seeds", "41,42,43", "--cuda-visible-devices", str(gpu)],
              f"{LOGS}/tarp_{arm}.log", {"XLA_PYTHON_CLIENT_MEM_FRACTION": str(mem)})


def sbc_from_dumps(arm):
    rs = []
    for f in sorted(glob.glob(f"{GC}/tarp_drp/dumps/{arm}_*/seed_*/n*_m*/posterior_samples.npz")):
        z = np.load(f); rs.append((z["samples"] < z["theta"][:, None, :]).mean(1))
    if not rs:
        return None
    from scipy import stats as st
    r = np.concatenate(rs, 0)
    return {"std": [float(s) for s in r.std(0)[:3]],
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
        z = np.load(f); devs.append(float(np.trapz(z["ecp_bootstrap"].mean(0) - z["alpha"], z["alpha"]) * 2))
    return (float(np.mean(devs)), float(np.std(devs))) if devs else (None, None)


def verdict(devs, sbc):
    if not devs or not sbc:
        return "INCOMPLETE"
    worst = max(abs(d) for d in devs.values())
    std_off = max((max(0.0, STD_LO - s, s - STD_HI) for s in sbc["std"]), default=9)
    if worst > DEV_CAVEAT or std_off >= STD_FAIL:
        return "FAIL"
    return "PASS" if (worst <= DEV_PASS and std_off == 0.0) else "PASS-with-caveat"


def write_artifacts():
    rows, vj = [], {}
    for arm in ARMS:
        ms = json.loads(Path(SWEEP, arm, "median_summary.json").read_text()) \
            if Path(SWEEP, arm, "median_summary.json").exists() else {}
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
    L = ["# GATE C — 2D-1D Haar SCATTERING arms (Phase 2; artifacts, reading by hand)\n",
         "vs Phase-1 LINEAR: haar_nobnt 2676/FAIL, haar_bnt_uncut 885. Baselines: auto 2405, product 2875.\n",
         "| arm | FoM3 | σ(Om,s8,w0) | TARP LOW/MID/HIGH | net bias | SBC std | verdict |",
         "|---|---|---|---|---|---|---|", *rows,
         "", "(+ net bias = conservative; − = over-confident. SBC uniform=0.289.)"]
    Path(GC, "GATE_C_HAARSCAT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--mem-fraction", type=float, default=0.30)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True); os.makedirs(f"{GC}/tarp_drp", exist_ok=True)
    t0 = time.time()
    if not a.report_only:
        for pre in ("none", "bnt"):
            print(f"\n===== FREEZE {pre} ({time.time()-t0:.0f}s) =====", flush=True); freeze(pre, a.gpu)
        for arm in ARMS:
            print(f"\n===== BUILD {arm} ({time.time()-t0:.0f}s) =====", flush=True); build(arm, a.gpu)
        for arm in ARMS:
            print(f"\n===== SWEEP {arm} ({time.time()-t0:.0f}s) =====", flush=True); sweep(arm, a.gpu, a.mem_fraction)
        for arm in ARMS:
            print(f"\n===== GATE {arm} ({time.time()-t0:.0f}s) =====", flush=True); gate(arm, a.gpu, a.mem_fraction)
        print(f"\n===== COVERAGE ({time.time()-t0:.0f}s) =====", flush=True)
        sh([PY, "run_tarp_coverage.py", "--dumps-root", f"{GC}/tarp_drp/dumps",
            "--outdir", f"{GC}/tarp_drp", "--dims", "3"], f"{LOGS}/coverage_haarscat.log",
           {"CUDA_VISIBLE_DEVICES": str(a.gpu)})
    write_artifacts()
    print(f"\nPHASE 2 ORCHESTRATION DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
