#!/usr/bin/env python
"""GATE C for the lane-B post-cut arms (PLAN_M4_GATE_C.md). Adapted from run_joint_gate_c.py.
Pinned to GPU 2 (--gpus default 2,2 = 2 packed jobs). TARP-stratified-val + coverage + SBC
+ derived-verdict report. preproc = log1p-zscore/clip5/min-var1e-5 (L1-type datavectors)."""
import argparse, glob, json, os, subprocess, time, zipfile
import numpy as np
from numpy.lib import format as npformat
from pathlib import Path

REPO = "/mnt/home/tersenov/software/cnn_sbi"; SBI = f"{REPO}/scripts/sbi"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
OM2 = f"{SBI}/results/exploratory/flatsky_cross_2026_06/overnight_menu_2"
GC = f"{OM2}/gate_c_laneB"; LOGS = f"{GC}/logs"
ARMS = ["B0_bntcut_l1", "B1_bntcut_sums", "B2_bntcut_deep2", "B3_nobnt_unicut"]
FOM = {"B0_bntcut_l1": 268, "B1_bntcut_sums": 596, "B2_bntcut_deep2": 613,
       "B3_nobnt_unicut": 337}
TERCILES = ("LOW", "MID", "HIGH")
DEV_PASS, DEV_CAVEAT = 0.05, 0.10
STD_LO, STD_HI, STD_FAIL = 0.275, 0.305, 0.02


def npz_shape(path, key):
    with zipfile.ZipFile(path) as z:
        with z.open(f"{key}.npy") as f:
            v = npformat.read_magic(f); shape, _, dt = npformat._read_array_header(f, v)
    return shape


def preflight():
    print("===== preflight =====", flush=True)
    for arm in ARMS:
        c = Path(OM2, arm, "cache")
        ts = npz_shape(c / "l1_train.npz", "theta"); xs = npz_shape(c / "l1_train.npz", "x")
        assert ts[1] == 6 and ts[0] == xs[0], (arm, ts, xs)
        print(f"  {arm}: train theta{ts} x{xs} OK", flush=True)


def tarp_cmd(arm, gpu):
    return [PY, "tarp_stratified_val.py", "--train-cache-dir", f"{OM2}/{arm}/cache",
            "--cache-prefix", "l1", "--arm-label", arm,
            "--dumps-root", f"{GC}/tarp_drp/dumps",
            "--preproc-transform", "log1p-zscore", "--clip-value", "5",
            "--min-feature-variance", "1e-5", "--seeds", "41,42,43",
            "--cuda-visible-devices", str(gpu)]


def run_phase(jobs, slots, mem):
    os.makedirs(LOGS, exist_ok=True)
    pending = list(jobs); slot = [None] * len(slots); t0 = time.time(); failed = {}

    def launch(arm, gpu):
        log = open(f"{LOGS}/tarp_{arm}.log", "w")
        env = dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="3",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION=str(mem),
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", CNN_CPU_THREADS="8",
                   OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
        p = subprocess.Popen(tarp_cmd(arm, gpu), cwd=SBI, env=env, stdout=log,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        print(f"[{time.time()-t0:6.0f}s] LAUNCH {arm} GPU{gpu} (pid {p.pid})", flush=True)
        return (arm, p, log)

    while pending or any(slot):
        for i, gpu in enumerate(slots):
            s = slot[i]
            if s and s[1].poll() is not None:
                arm, p, log = s; log.close(); slot[i] = None
                if p.returncode != 0:
                    failed[arm] = p.returncode
                print(f"[{time.time()-t0:6.0f}s] {'DONE' if p.returncode==0 else 'FAIL'} {arm}",
                      flush=True)
        for i, gpu in enumerate(slots):
            if slot[i] is None and pending:
                slot[i] = launch(pending.pop(0), gpu)
        time.sleep(10)
    return failed


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


def write_report(failed):
    L = ["# GATE C — lane-B post-cut arms (derived verdicts; PLAN_M4_GATE_C.md)\n",
         "Does the post-cut recombination gain (B1/B2 ~1.8x the uniform-cut noBNT B3) survive",
         "calibration? TARP-stratified-val (600 pts, FoM3 terciles, 3 seeds) + SBC. preproc",
         "log1p-zscore/clip5/min-var1e-5; GPU 2.\n",
         "| arm | FoM3 | TARP HIGH/MID/LOW (dim3) | net bias | SBC std (Om,s8,w0) | verdict |",
         "|---|---|---|---|---|---|"]
    V = {}
    for arm in ARMS:
        d = tarp_devs(arm); s = sbc_from_dumps(arm); nb = net_bias(arm)
        V[arm] = verdict(d, s)
        dcell = "/".join(f"{d[t]:+.3f}" if t in d else "—" for t in ("HIGH", "MID", "LOW"))
        nbcell = f"{nb[0]:+.3f}±{nb[1]:.3f}" if nb[0] is not None else "—"
        scell = ",".join(f"{x:.3f}" for x in s["std"]) if s else "—"
        L.append(f"| {arm} | {FOM[arm]} | {dcell} | {nbcell} | {scell} | **{V[arm]}** |")
    L += ["", "(+ net bias = conservative/over-covers; − = over-confident. Uniform=0.289.)",
          "", "## Reading (registered band P-B, PLAN_M4_GATE_C.md)"]
    b1b2_ok = all(V[a] in ("PASS", "PASS-with-caveat") for a in
                  ("B1_bntcut_sums", "B2_bntcut_deep2"))
    b1b2_clean = all(V[a] == "PASS" for a in ("B1_bntcut_sums", "B2_bntcut_deep2"))
    if b1b2_clean and V["B3_nobnt_unicut"] in ("PASS", "PASS-with-caveat"):
        L.append("- B1/B2 PASS clean and B3 calibrated -> the ~1.8x gain over the uniform-cut "
                 "noBNT analysis is CALIBRATION-CLEAN and real. M4 message holds (pending the "
                 "2nd-schedule robustness check).")
    elif b1b2_ok:
        L.append("- B1/B2 pass-with-caveat -> gain is broadly real but carry the calibration "
                 "caveat (named tercile/direction); quote calibrated marginals alongside FoM3.")
    else:
        L.append("- B1/B2 FAIL their band -> the ~1.8x is partly inflated by miscalibration; "
                 "downgrade to 'comparable' and quote calibrated marginals only.")
    if failed:
        L += ["", f"FAILURES: {failed}"]
    Path(GC, "GATE_C_LANEB.md").write_text("\n".join(L) + "\n")
    print(f"wrote {GC}/GATE_C_LANEB.md", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="2,2")
    ap.add_argument("--mem-fraction", type=float, default=0.42)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    os.chdir(SBI); os.makedirs(GC, exist_ok=True)
    preflight()
    slots = [int(g) for g in a.gpus.split(",")]
    failed = {}
    if not a.report_only:
        failed = run_phase(ARMS, slots, a.mem_fraction)
        rc = subprocess.run([PY, "run_tarp_coverage.py", "--dumps-root", f"{GC}/tarp_drp/dumps",
                             "--outdir", f"{GC}/tarp_drp", "--dims", "3"], cwd=SBI).returncode
        if rc != 0:
            failed["coverage"] = rc
    write_report(failed)
    print("LANE-B GATE C DONE", flush=True)


if __name__ == "__main__":
    main()
