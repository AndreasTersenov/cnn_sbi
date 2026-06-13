#!/usr/bin/env python
"""Overnight menu 2 driver (PLAN_OVERNIGHT_MENU_2.md) — lanes A/B/C/D + packing bench.

Phases:
  0. bench   — B1-spec sweep-packing benchmark on REAL workload (B3 screen x3 solo serial
               on one GPU, then x3 concurrent 3-pack on the same GPU) -> PACKING_RESULT.md;
               accept 3-pack iff aggregate throughput >= 0.9 * 3 * solo. Runs while the
               other GPUs do phase 1. (B2/B3 compressor benchmarks DEFERRED: no compressor
               workload in this campaign — noted in the result.)
  1. build   — per-arm cache builds (freeze_p3 -> D1 dependency handled; A1 VMIM
               self-time-boxed 110 min).
  2. screen  — 1 MAF seed / 3000 obs per arm (smoke + cost anchor).
  3. full    — 3 seeds / 9000 obs. ALWAYS-ESCALATE by design: every arm's pre-registered
               branch sentences need full rigor; screening only catches pathologies.
  4. gate    — tarp_stratified_val for A1 / A2 / C2 (the K-trend calibration points)
               + run_tarp_coverage + SBC -> appended to the result.
  5. result  — OVERNIGHT2_RESULT.md: derived numbers + the registered branch sentences
               (template fill ONLY — bands from the plan, no new readings).

Hard deadline 07:00 UTC: no new job starts after it; running jobs finish.
Detached:  (cd scripts/sbi && setsid nohup <py> run_overnight_menu_2.py > .../driver.out 2>&1 &)
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = "/mnt/home/tersenov/software/cnn_sbi"
SBI = f"{REPO}/scripts/sbi"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
BASE = f"{SBI}/results/exploratory/flatsky_cross_2026_06"
OM = f"{BASE}/overnight_menu"
OM2 = f"{BASE}/overnight_menu_2"
LOGS = f"{OM2}/logs"
GC2 = f"{OM2}/gate_c"

DEADLINE = time.mktime(time.strptime("2026-06-13 07:00:00", "%Y-%m-%d %H:%M:%S"))
FOREIGN_LIMIT_MB = 12000
JOB_TIMEOUT_S = {"build_A1_pair2d_vmim": 3.0 * 3600}
DEFAULT_TIMEOUT_S = 1.8 * 3600

# baselines (full rigor, pooled medians)
L1_AUTO, L1_PROD, L1_BNT = 2405.0, 2875.0, 364.0
PAIR2D_K10, JOINTL1_K10 = 2794.0, 2788.0
PAIR2D_K10_BNT_FIXED_RATIO = 0.522

SCHED_M = "3,4;2,3,4;1,2,3,4;0,1,2,3,4"
SCHED_U = "3,4;3,4;3,4;3,4"

STD_PREPROC = ("log1p-zscore", "5", "1e-5")
VMIM_PREPROC = ("none", "0", "1e-12")


def arm_dir(name):
    return f"{OM2}/{name}"


def joint_build(name, stat, basis, k, adaptive=False):
    cmd = [PY, "build_flatsky_joint_arm.py", "--stat", stat, "--basis", basis,
           "--k", str(k), "--dequantize",
           "--out-cache", f"{arm_dir(name)}/cache",
           "--out-fid", f"{arm_dir(name)}/fiducial_summaries.npz"]
    if adaptive:
        cmd.append("--adaptive-ranges")
    return cmd


# name -> dict(build cmd or None(pre-built), preproc, gate(bool), big(bool), after=[deps])
ARMS = {
    # lane A
    "A1_pair2d_vmim": dict(
        build=[PY, "vmim_from_cache.py",
               "--cache-dir", f"{OM}/pair2dq_nobnt/cache",
               "--fid-npz", f"{OM}/pair2dq_nobnt/fiducial_summaries.npz",
               "--out-cache", f"{arm_dir('A1_pair2d_vmim')}/cache",
               "--out-fid", f"{arm_dir('A1_pair2d_vmim')}/fiducial_summaries.npz",
               "--steps", "30000", "--max-minutes", "110"],
        preproc=VMIM_PREPROC, gate=True, big=False, after=[]),
    "A2_pair2d_k8": dict(build=joint_build("A2_pair2d_k8", "pair2d", "nobnt", 8),
                         preproc=STD_PREPROC, gate=True, big=False, after=[]),
    # lane B (B0/B3 caches pre-built by mask_flatsky_cache.py before launch)
    "B0_bntcut_l1": dict(build=None, preproc=STD_PREPROC, gate=False, big=False, after=[]),
    "B1_bntcut_sums": dict(
        build=[PY, "build_flatsky_postcut_arm.py", "--variant", "cutsum6",
               "--keep", SCHED_M,
               "--base-cache", f"{arm_dir('B0_bntcut_l1')}/cache",
               "--base-fid", f"{arm_dir('B0_bntcut_l1')}/fiducial_summaries.npz",
               "--out-cache", f"{arm_dir('B1_bntcut_sums')}/cache",
               "--out-fid", f"{arm_dir('B1_bntcut_sums')}/fiducial_summaries.npz"],
        preproc=STD_PREPROC, gate=False, big=False, after=[]),
    "B2_bntcut_deep2": dict(
        build=[PY, "build_flatsky_postcut_arm.py", "--variant", "cutdeep2",
               "--keep", SCHED_M,
               "--base-cache", f"{arm_dir('B0_bntcut_l1')}/cache",
               "--base-fid", f"{arm_dir('B0_bntcut_l1')}/fiducial_summaries.npz",
               "--out-cache", f"{arm_dir('B2_bntcut_deep2')}/cache",
               "--out-fid", f"{arm_dir('B2_bntcut_deep2')}/fiducial_summaries.npz"],
        preproc=STD_PREPROC, gate=False, big=False, after=[]),
    "B3_nobnt_unicut": dict(build=None, preproc=STD_PREPROC, gate=False, big=False, after=[]),
    # lane C
    "C1_pair2d_bnt_ar": dict(build=joint_build("C1_pair2d_bnt_ar", "pair2d", "bnt", 10,
                                               adaptive=True),
                             preproc=STD_PREPROC, gate=False, big=False, after=[]),
    "C2_pair2d_k15": dict(build=joint_build("C2_pair2d_k15", "pair2d", "nobnt", 15),
                          preproc=STD_PREPROC, gate=True, big=True, after=[]),
    "C3_pair2d_k15_bnt_ar": dict(build=joint_build("C3_pair2d_k15_bnt_ar", "pair2d",
                                                   "bnt", 15, adaptive=True),
                                 preproc=STD_PREPROC, gate=False, big=True, after=[]),
    # lane D (after the product3 sigma freeze)
    "D1_l1_product3": dict(
        build=[PY, "build_flatsky_product3_arm.py",
               "--out-cache", f"{arm_dir('D1_l1_product3')}/cache",
               "--out-fid", f"{arm_dir('D1_l1_product3')}/fiducial_summaries.npz"],
        preproc=STD_PREPROC, gate=False, big=False, after=["freeze_p3"]),
}

PRELUDES = {
    "freeze_p3": [PY, "freeze_flatsky_cross_noise.py", "--op", "product3"],
}


def sweep_cmd(name, full, gpu, mem):
    arm = ARMS[name]
    outdir = f"{arm_dir(name)}/population_sweep{'_full' if full else ''}"
    pre = arm["preproc"]
    return [PY, "population_sweep_flatsky.py",
            "--train-cache-dir", f"{arm_dir(name)}/cache", "--cache-prefix", "l1",
            "--arm-label", f"om2_{name}{'_full' if full else ''}",
            "--fiducial-summaries-npz", f"{arm_dir(name)}/fiducial_summaries.npz",
            "--output-dir", outdir,
            "--preproc-transform", pre[0], "--clip-value", pre[1],
            "--min-feature-variance", pre[2],
            "--seeds", "41,42,43" if full else "41",
            "--n-obs", "9000" if full else "3000", "--max-perm", "50",
            "--m-samples", "2000", "--cuda-visible-devices", str(gpu)], outdir


def gate_cmd(name, gpu):
    pre = ARMS[name]["preproc"]
    return [PY, "tarp_stratified_val.py",
            "--train-cache-dir", f"{arm_dir(name)}/cache", "--cache-prefix", "l1",
            "--arm-label", name, "--dumps-root", f"{GC2}/tarp_drp/dumps",
            "--preproc-transform", pre[0], "--clip-value", pre[1],
            "--min-feature-variance", pre[2], "--seeds", "41,42,43",
            "--cuda-visible-devices", str(gpu)]


def gpu_foreign_mb(gpu):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits", "-i", str(gpu)],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return 1 << 30


def status_append(line):
    with open(f"{OM2}/OVERNIGHT2_STATUS.md", "a") as fh:
        fh.write(line + "\n")
    print("STATUS: " + line, flush=True)


def job_env(gpu, mem, threads=8):
    t = str(threads)
    return dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="3",
                XLA_PYTHON_CLIENT_PREALLOCATE="false",
                XLA_PYTHON_CLIENT_MEM_FRACTION=str(mem),
                PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                CNN_CPU_THREADS=t, OMP_NUM_THREADS=t, MKL_NUM_THREADS=t,
                OPENBLAS_NUM_THREADS=t, NUMEXPR_NUM_THREADS=t,
                TF_NUM_INTRAOP_THREADS=t, TF_NUM_INTEROP_THREADS="2",
                CUDA_VISIBLE_DEVICES=str(gpu))


def run_job(tag, cmd, gpu, mem, t0, threads=8):
    os.makedirs(LOGS, exist_ok=True)
    # sweeps run at 5 threads (9 packed jobs x 5 = 45 <= the 50-CPU budget); builds at 8
    env = job_env(gpu, mem, threads)
    print(f"[{time.time()-t0:7.0f}s] ===== {tag} ===== (GPU{gpu} mem{mem})", flush=True)
    timeout = JOB_TIMEOUT_S.get(tag, DEFAULT_TIMEOUT_S)
    tj = time.time()
    with open(f"{LOGS}/{tag}.log", "w") as log:
        try:
            rc = subprocess.run(cmd, cwd=SBI, env=env, stdout=log,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -99
    dt = time.time() - tj
    print(f"[{time.time()-t0:7.0f}s] {'DONE' if rc == 0 else 'FAIL'} {tag} "
          f"(rc={rc}, {dt:.0f}s)", flush=True)
    return rc == 0, dt


def deadline_ok():
    return time.time() < DEADLINE


def _worker(gpu, mem, q, fn, t0):
    while deadline_ok():
        try:
            item = q.get_nowait()
        except queue.Empty:
            return
        waited = 0
        while gpu_foreign_mb(gpu) > FOREIGN_LIMIT_MB and waited < 1800:
            print(f"[{time.time()-t0:7.0f}s] GPU{gpu} foreign-busy — waiting 300 s", flush=True)
            time.sleep(300); waited += 300
        fn(item, gpu, mem)


def run_pool(items, slots, fn, t0):
    """slots: list of (gpu, mem) — duplicated gpu = packing."""
    q = queue.Queue()
    for it in items:
        q.put(it)
    threads = [threading.Thread(target=_worker, args=(g, m, q, fn, t0), daemon=True)
               for g, m in slots]
    [th.start() for th in threads]
    [th.join() for th in threads]


def med(outdir):
    f = Path(outdir) / "median_summary.json"
    return json.load(open(f)) if f.exists() else None


# ---------------------------------------------------------------------------
# Packing benchmark (phase 0; B1-spec, 3-pack only, real workload = B3 screen)
# ---------------------------------------------------------------------------
def packing_benchmark(gpu, t0):
    bdir = f"{OM2}/packing_benchmarks"
    os.makedirs(bdir, exist_ok=True)
    name = "B3_nobnt_unicut"
    solo_times, pack_times = [], []
    for rep in range(3):
        if not deadline_ok():
            return None
        cmd, _ = sweep_cmd(name, False, gpu, 0)
        cmd[cmd.index("--output-dir") + 1] = f"{bdir}/solo_rep{rep}"
        cmd[cmd.index("--arm-label") + 1] = f"bench_solo_{rep}"
        ok, dt = run_job(f"bench_solo_{rep}", cmd, gpu, 0.40, t0, threads=5)
        if ok:
            solo_times.append(dt)
    procs = []
    tj = time.time()
    env_base = job_env(gpu, 0.30, threads=5)   # identical conditions to deployment
    for rep in range(3):
        cmd, _ = sweep_cmd(name, False, gpu, 0)
        cmd[cmd.index("--output-dir") + 1] = f"{bdir}/pack_rep{rep}"
        cmd[cmd.index("--arm-label") + 1] = f"bench_pack_{rep}"
        log = open(f"{LOGS}/bench_pack_{rep}.log", "w")
        procs.append((subprocess.Popen(cmd, cwd=SBI, env=env_base, stdout=log,
                                       stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL), log))
    for p, log in procs:
        p.wait(); log.close()
    pack_wall = time.time() - tj
    ok_pack = all(p.returncode == 0 for p, _ in procs)
    solo = float(np.median(solo_times)) if solo_times else float("nan")
    # aggregate throughput: 3 jobs / pack_wall vs 3 * (1 / solo)
    accept = (ok_pack and solo_times
              and (3.0 / pack_wall) >= 0.9 * 3.0 * (1.0 / solo))
    txt = [f"# Packing benchmark (B1 spec, 3-pack only; workload = {name} screen)",
           f"- solo reps (s): {[f'{x:.0f}' for x in solo_times]} (median {solo:.0f})",
           f"- 3-pack wall (s): {pack_wall:.0f} (all rc=0: {ok_pack})",
           f"- aggregate throughput ratio = (3/{pack_wall:.0f}) / (3/{solo:.0f}) "
           f"= {(solo / pack_wall) if pack_wall else float('nan'):.2f} (accept >= 0.9 per job-rate... "
           f"derived as 3-pack aggregate vs 3x solo)",
           f"- **3-pack {'ACCEPTED' if accept else 'REJECTED'}** for sweep phases tonight",
           "- B2/B3 (compressor packing / cross-class) DEFERRED: no compressor workload "
           "in this campaign.",
           f"- load1 at bench: {os.getloadavg()[0]:.1f}; foreign MB on GPU{gpu}: "
           f"{gpu_foreign_mb(gpu)}"]
    Path(bdir, "PACKING_RESULT.md").write_text("\n".join(txt) + "\n")
    status_append(f"- packing bench: solo med {solo:.0f}s, 3-pack wall {pack_wall:.0f}s "
                  f"-> {'ACCEPT' if accept else 'REJECT'}")
    return bool(accept)


import numpy as np  # noqa: E402  (used by packing_benchmark + result derivation)


# ---------------------------------------------------------------------------
# Result derivation (template fill of the registered branch sentences ONLY)
# ---------------------------------------------------------------------------
def tarp_signed_devs(arm, dim=3):
    import glob as _g
    out = {}
    for terc in ("HIGH", "MID", "LOW"):
        worst = None
        for f in sorted(_g.glob(f"{GC2}/tarp_drp/curves/tarp_curve_{arm}_{terc}_"
                                f"seed*_dim{dim}.npz")):
            z = np.load(f)
            a_ = np.asarray(z["alpha"]); e = np.asarray(z["ecp_bootstrap"]).mean(axis=0)
            i = int(np.argmax(np.abs(e - a_)))
            d = float(e[i] - a_[i])
            if worst is None or abs(d) > abs(worst):
                worst = d
        if worst is not None:
            out[terc] = worst
    return out


def write_result(results):
    fom = {n: (results[n].get("full") or {}).get("fom3") for n in ARMS}
    scr = {n: (results[n].get("screen") or {}).get("fom3") for n in ARMS}
    sig = {n: (results[n].get("full") or {}) for n in ARMS}

    L = ["# Overnight menu 2 — derived results (PLAN_OVERNIGHT_MENU_2.md)", "",
         "Screening = 1 seed/3000 obs; full = 3 seeds/9000 obs (always-escalate by design).",
         f"Baselines: l1-auto {L1_AUTO:.0f} | l1+product {L1_PROD:.0f} | l1-BNT {L1_BNT:.0f}"
         f" | pair2dq_nobnt {PAIR2D_K10:.0f} (ratio 0.522).", "",
         "| arm | screening FoM3 | full FoM3 | σ(Om) | σ(s8) | σ(w0) |",
         "|---|---|---|---|---|---|"]
    for n in ARMS:
        m = sig[n]
        L.append(f"| {n} | {scr[n]:.0f}" if scr[n] else f"| {n} | FAIL")
        L[-1] += (f" | {fom[n]:.0f}" if fom[n] else " | —")
        for k in ("sigma_Om", "sigma_s8", "sigma_w0"):
            v = m.get(k)
            L[-1] += (f" | {v:.3f}" if v else " | —")
        L[-1] += " |"

    L += ["", "## Branch-sentence resolution (bands registered in the plan BEFORE data)", ""]

    # lane A
    devs_a1 = tarp_signed_devs("A1_pair2d_vmim")
    if fom["A1_pair2d_vmim"] and devs_a1:
        worst = max(abs(d) for d in devs_a1.values())
        f_ok = fom["A1_pair2d_vmim"] >= 0.9 * PAIR2D_K10
        if worst <= 0.05 and f_ok:
            s = ("REHABILITATED: the joint PDF's miscalibration is a high-dimensional "
                 "count-feature artifact; compressed, the statistic is calibration-clean "
                 "at full constraining power (quote from the VMIM arm).")
        elif worst <= 0.05:
            s = (f"EDGE-WAS-MISCALIBRATION: calibrates but FoM3 "
                 f"{fom['A1_pair2d_vmim']:.0f} < 0.9 x {PAIR2D_K10:.0f}; downgrade final.")
        else:
            s = ("PATHOLOGY-NOT-DIMENSIONALITY: still miscalibrated at 10-d; tracks the "
                 "statistic/posterior geometry itself.")
        L.append(f"- **A1** (VMIM): FoM3 {fom['A1_pair2d_vmim']:.0f}, TARP devs {devs_a1} "
                 f"-> {s}")
    else:
        L.append("- **A1**: INCOMPLETE (time-box/failure) — lane carried by A2/A3 "
                 "(A3 already resolved: pooled-estimator over-confidence CONFIRMED, "
                 "see GATE_C_JOINT.md Addendum 2).")
    devs_a2 = tarp_signed_devs("A2_pair2d_k8")
    if devs_a2:
        high = devs_a2.get("HIGH")
        shrunk = high is not None and abs(high) <= 0.7 * 0.134
        L.append(f"- **A2** (K=8): TARP devs {devs_a2}, FoM3 {fom['A2_pair2d_k8'] or 0:.0f} -> "
                 + ("SPARSITY-DRIVEN: worst dev shrinks >=30% vs K=10's -0.134; coarser "
                    "grids are the calibratable regime." if shrunk else
                    "SPARSITY NOT THE DRIVER: dev does not shrink materially; count "
                    "features carry an intrinsic tail-calibration cost."))
    # K-trend
    devs_c2 = tarp_signed_devs("C2_pair2d_k15")
    L.append(f"- **K-trend** [FoM3, worst HIGH dev]: K=8 [{fom['A2_pair2d_k8'] or 0:.0f}, "
             f"{devs_a2.get('HIGH', float('nan')):+.3f}] | K=10 [{PAIR2D_K10:.0f}, -0.134] "
             f"| K=15 [{fom['C2_pair2d_k15'] or 0:.0f}, "
             f"{devs_c2.get('HIGH', float('nan')):+.3f}]")

    # lane B
    if fom["B2_bntcut_deep2"] and fom["B3_nobnt_unicut"]:
        r = fom["B2_bntcut_deep2"] / fom["B3_nobnt_unicut"]
        if r >= 0.9:
            s = ("BNT + two cleaned recombinations costs <=10% of the information while "
                 "retaining per-slice systematics rejection — the constructive resolution "
                 "of the BNT trade-off.")
        elif r >= 0.75:
            s = f"the trade-off is real and now quantified: {100*(1-r):.0f}% info for clean cuts."
        else:
            s = ("post-cut recombination cannot rescue per-channel statistics; the BNT "
                 "information cost in survey practice is substantial (honest negative).")
        L.append(f"- **B2/B3 = {r:.2f}** (B2 {fom['B2_bntcut_deep2']:.0f} / B3 "
                 f"{fom['B3_nobnt_unicut']:.0f}) -> {s}")
        for n, label in (("B0_bntcut_l1", "B0/B3"), ("B1_bntcut_sums", "B1/B3")):
            if fom[n]:
                L.append(f"- {label} = {fom[n] / fom['B3_nobnt_unicut']:.2f} "
                         f"({fom[n]:.0f})")
        L.append(f"- B3 vs uncut l1-auto: {fom['B3_nobnt_unicut'] / L1_AUTO:.2f} "
                 "(what the uniform cut costs a noBNT analysis)")
        L.append(f"- (schedule-conditional numbers: M = '{SCHED_M}', U = '{SCHED_U}')")

    # lane C
    if fom["C1_pair2d_bnt_ar"]:
        r1 = fom["C1_pair2d_bnt_ar"] / PAIR2D_K10
        L.append(f"- **C1** BNT-adaptive pairwise ratio r = {r1:.3f} (fixed-grid 0.522; "
                 f"registered band 0.52 < r < ~0.75) -> "
                 + ("placement explains the same majority for pairwise as for full-4D."
                    if r1 >= 0.65 else
                    "pairwise statistics are structurally more basis-fragile: marginal "
                    "incompleteness, not placement, dominates."))
    if fom["C2_pair2d_k15"]:
        rel = fom["C2_pair2d_k15"] / PAIR2D_K10 - 1.0
        L.append(f"- **C2** K=15 noBNT: {fom['C2_pair2d_k15']:.0f} ({rel:+.1%} vs K=10) -> "
                 + ("K=10 is the saturated regime; the parity comparison is "
                    "resolution-robust." if abs(rel) <= 0.05 else
                    "the joint-stat ceiling is higher than quoted; revisit parity upward "
                    "(lane-A calibration applies to this arm next)."))
    if fom["C3_pair2d_k15_bnt_ar"] and fom["C2_pair2d_k15"] and fom["C1_pair2d_bnt_ar"]:
        r3 = fom["C3_pair2d_k15_bnt_ar"] / fom["C2_pair2d_k15"]
        r1 = fom["C1_pair2d_bnt_ar"] / PAIR2D_K10
        L.append(f"- **C3** K=15 BNT-adaptive ratio r = {r3:.3f} (vs K=10 adaptive "
                 f"{r1:.3f}; staircase test, threshold +0.05) -> "
                 + ("finer K visibly staircase-approximates the shear; finite-K is the "
                    "binding constraint." if (r3 - r1) >= 0.05 else
                    "the shear residual is K-stubborn; only a learned linear front-end "
                    "transports."))

    # lane D (decisive needs > +5% AND every science marginal <= l1+product's)
    if fom.get("D1_l1_product3"):
        rel = fom["D1_l1_product3"] / L1_PROD - 1.0
        m1 = sig["D1_l1_product3"]
        prod_marg = {"sigma_Om": 0.048, "sigma_s8": 0.075, "sigma_w0": 0.238}
        marg_ok = all(m1.get(k) is not None and m1[k] <= ref + 1e-9
                      for k, ref in prod_marg.items())
        L.append(f"- **D1** auto+product+product3: {fom['D1_l1_product3']:.0f} "
                 f"({rel:+.1%} vs l1+product {L1_PROD:.0f}; marginals-le check: {marg_ok}; "
                 "decisive only if > +5% AND every science marginal <=) -> "
                 + ("ORDER-3 INFORMATION ACCESSIBLE — new thread for the morning "
                    "session (NOT extended tonight, per fence)." if (rel > 0.05 and marg_ok)
                    else "the accessible cross-bin information is pairwise-saturated: "
                    "measured through third order."))

    L += ["", "Gate dumps/curves: overnight_menu_2/gate_c/. Logs: overnight_menu_2/logs/.",
          "Packing: packing_benchmarks/PACKING_RESULT.md.",
          "A3 (pooled TARP) was resolved pre-launch: see GATE_C_JOINT.md Addendum 2."]
    Path(OM2, "OVERNIGHT2_RESULT.md").write_text("\n".join(L) + "\n")
    print(f"wrote {OM2}/OVERNIGHT2_RESULT.md", flush=True)


def regen_from_disk():
    """Rebuild the results dict from on-disk median jsons + gate curves and rewrite
    OVERNIGHT2_RESULT.md (idempotent; used after the K=15 reruns land)."""
    results = {n: {} for n in ARMS}
    for n in ARMS:
        for tag, sub in (("screen", "population_sweep"), ("full", "population_sweep_full")):
            results[n][tag] = med(f"{arm_dir(n)}/{sub}")
    write_result(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="1,0,2")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--regen-only", action="store_true",
                    help="rebuild OVERNIGHT2_RESULT.md from on-disk artifacts and exit")
    a = ap.parse_args()
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    if a.regen_only:
        regen_from_disk()
        return 0
    if a.dry_run:
        for k, cmd in PRELUDES.items():
            print(f"{k}: {' '.join(cmd)}")
        for n, arm in ARMS.items():
            print(f"{n}: " + (" ".join(arm["build"]) if arm["build"] else "(pre-built)"))
            c, _ = sweep_cmd(n, False, "<G>", 0)
            print("   screen: " + " ".join(c[-14:]))
            if arm["gate"]:
                print("   gate: " + " ".join(gate_cmd(n, "<G>")[-10:]))
        return 0

    t0 = time.time()
    os.makedirs(OM2, exist_ok=True)
    status_append(f"## menu-2 run started {time.strftime('%F %T')} (GPUs {gpus})")
    results = {n: {} for n in ARMS}
    lock = threading.Lock()

    # ---- phase 0+1 overlapped: bench on gpus[0]; preludes+builds on the others ----
    bench_gpu = gpus[0]
    other = [g for g in gpus[1:]] or [bench_gpu]
    bench_out = {}

    def bench_thread():
        bench_out["accept"] = packing_benchmark(bench_gpu, t0)

    bt = threading.Thread(target=bench_thread, daemon=True)
    bt.start()

    ok, _ = run_job("freeze_p3", PRELUDES["freeze_p3"], other[0], 0.40, t0)
    if not ok:
        status_append("- freeze_p3 FAIL -> D1 auto-skipped")
        ARMS.pop("D1_l1_product3"); results.pop("D1_l1_product3")

    build_q = [n for n, arm in ARMS.items() if arm["build"]]

    def do_build(name, gpu, mem):
        ok, _ = run_job(f"build_{name}", ARMS[name]["build"], gpu, mem, t0)
        with lock:
            results[name]["built"] = ok
            if not ok:
                status_append(f"- {name}: BUILD FAIL (auto-skip)")

    run_pool(build_q, [(g, 0.40) for g in other], do_build, t0)
    bt.join()
    pack_ok = bench_out.get("accept")
    status_append(f"- builds done; 3-pack accepted: {pack_ok}")

    # ---- phase 2+3: screen then full, packed per the benchmark ----
    def slots_for(big):
        if pack_ok and not big:
            return [(g, 0.30) for g in gpus for _ in range(3)]
        return [(g, 0.45) for g in gpus]

    def do_sweep(full):
        def fn(name, gpu, mem):
            cmd, outdir = sweep_cmd(name, full, gpu, mem)
            ok, _ = run_job(f"{'full' if full else 'screen'}_{name}", cmd, gpu, mem, t0,
                            threads=5)
            with lock:
                m = med(outdir) if ok else None
                results[name]["full" if full else "screen"] = m
                tag = "FULL" if full else "screen"
                status_append(f"- {name}: {tag} "
                              + (f"FoM3 {m['fom3']:.0f}" if m else "FAIL"))
        return fn

    ready = [n for n in ARMS
             if ARMS[n]["build"] is None or results[n].get("built")]
    small = [n for n in ready if not ARMS[n]["big"]]
    big = [n for n in ready if ARMS[n]["big"]]
    run_pool(small, slots_for(False), do_sweep(False), t0)
    run_pool(big, slots_for(True), do_sweep(False), t0)
    # always-escalate: every screened-OK arm goes to full rigor
    esc_small = [n for n in small if results[n].get("screen")]
    esc_big = [n for n in big if results[n].get("screen")]
    status_append(f"- escalation (always-escalate): {sorted(esc_small + esc_big)}")
    run_pool(esc_small, slots_for(False), do_sweep(True), t0)
    run_pool(esc_big, slots_for(True), do_sweep(True), t0)

    # ---- phase 4: gates for the K-trend arms ----
    gates = [n for n in ARMS if ARMS[n].get("gate") and results[n].get("full")]

    def do_gate(name, gpu, mem):
        ok, _ = run_job(f"gate_{name}", gate_cmd(name, gpu), gpu, mem, t0)
        with lock:
            results[name]["gated"] = ok
            if not ok:
                status_append(f"- {name}: GATE FAIL")

    run_pool(gates, [(g, 0.45) for g in gpus], do_gate, t0)
    if gates:
        run_job("tarp_coverage", [PY, "run_tarp_coverage.py",
                                  "--dumps-root", f"{GC2}/tarp_drp/dumps",
                                  "--outdir", f"{GC2}/tarp_drp", "--dims", "3"],
                gpus[0], 0.45, t0)

    write_result(results)
    status_append(f"## menu-2 run complete {time.strftime('%F %T')} "
                  f"({(time.time()-t0)/3600:.1f} h)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
