#!/usr/bin/env python
"""full4d retry (overnight addendum): K=4 + dequantization (counts + seeded U(0,1)).

The K=5 undequantized arms NaN'd the MAF — diagnosed on the cache: median surviving
feature had ~4 distinct values (quasi-discrete sparse cells; classic flows-on-counts
pathology). Fix = the standard dequantization + coarser cells (256/scale, mean occupancy
~25). Both bases identical treatment => the P4b exact-invariance ratio is unaffected.
Build both arms -> screening -> full sweeps on BOTH if either screening succeeds (the
invariance ratio needs matched rigor). Appends to overnight_menu OVERNIGHT_STATUS.md and
writes a derived addendum into OVERNIGHT_RESULT.md.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = "/mnt/home/tersenov/software/cnn_sbi"
SBI = f"{REPO}/scripts/sbi"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
BASE = f"{SBI}/results/exploratory/flatsky_cross_2026_06"
OM = f"{BASE}/overnight_menu"
LOGS = f"{OM}/logs"
NOBNT_FOM, BNT_FOM = 2405.0, 364.0
ARMS = ["full4dq_nobnt", "full4dq_bnt"]


def build_cmd(name):
    basis = name.split("_")[1]
    return [PY, "build_flatsky_joint_arm.py", "--stat", "full4d", "--basis", basis,
            "--k", "4", "--dequantize",
            "--out-cache", f"{OM}/{name}/cache",
            "--out-fid", f"{OM}/{name}/fiducial_summaries.npz"]


def sweep_cmd(name, full):
    outdir = f"{OM}/{name}/population_sweep{'_full' if full else ''}"
    return [PY, "population_sweep_flatsky.py",
            "--train-cache-dir", f"{OM}/{name}/cache", "--cache-prefix", "l1",
            "--arm-label", f"overnight_{name}{'_full' if full else ''}",
            "--fiducial-summaries-npz", f"{OM}/{name}/fiducial_summaries.npz",
            "--output-dir", outdir,
            "--preproc-transform", "log1p-zscore", "--clip-value", "5",
            "--min-feature-variance", "1e-5",
            "--seeds", "41,42,43" if full else "41",
            "--n-obs", "9000" if full else "3000", "--max-perm", "50",
            "--m-samples", "2000"], outdir


def run_phase(name, cmd, gpu, t0):
    os.makedirs(LOGS, exist_ok=True)
    log_path = f"{LOGS}/{name}.log"
    print(f"[{time.time()-t0:7.0f}s] ===== {name} ===== (GPU{gpu})", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="3",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.40",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", CNN_CPU_THREADS="8",
               OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", CUDA_VISIBLE_DEVICES=str(gpu))
    with open(log_path, "w") as log:
        rc = subprocess.run(cmd, cwd=SBI, env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL).returncode
    print(f"[{time.time()-t0:7.0f}s] {'DONE' if rc == 0 else 'FAIL'} {name} (rc={rc})",
          flush=True)
    return rc == 0


def med(outdir):
    f = Path(outdir) / "median_summary.json"
    return json.load(open(f)) if f.exists() else None


def status(line):
    with open(f"{OM}/OVERNIGHT_STATUS.md", "a") as fh:
        fh.write(line + "\n")
    print("STATUS: " + line, flush=True)


def main():
    import argparse, threading, queue as q_
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="1,0")
    a = ap.parse_args()
    gpus = [g for g in a.gpus.split(",") if g]
    t0 = time.time()
    status(f"## full4d retry started {time.strftime('%F %T')} (K=4, dequantized)")
    res = {}
    lock = threading.Lock()

    def worker(gpu, jobs, full):
        while True:
            try:
                name = jobs.get_nowait()
            except q_.Empty:
                return
            if not full:
                if not run_phase(f"{name}_build", build_cmd(name), gpu, t0):
                    with lock:
                        status(f"- {name}: BUILD FAIL")
                    continue
            cmd, outdir = sweep_cmd(name, full)
            tag = "full" if full else "screen"
            if run_phase(f"{name}_{tag}", cmd + ["--cuda-visible-devices", str(gpu)],
                         gpu, t0):
                m = med(outdir)
                with lock:
                    res[(name, tag)] = m
                    status(f"- {name}: {tag} FoM3 {m['fom3']:.0f} "
                           f"(rec {(m['fom3']-BNT_FOM)/(NOBNT_FOM-BNT_FOM):.3f})")
            else:
                with lock:
                    status(f"- {name}: {tag.upper()} SWEEP FAIL")

    for full in (False, True):
        if full and not any((n, "screen") in res for n in ARMS):
            status("- retry: both screenings failed; no full sweeps")
            break
        jobs = q_.Queue()
        for n in ARMS:
            if full and (n, "screen") not in res:
                continue
            jobs.put(n)
        ths = [threading.Thread(target=worker, args=(g, jobs, full), daemon=True)
               for g in gpus]
        [t.start() for t in ths]
        [t.join() for t in ths]

    lines = ["", "## ADDENDUM — full4d retry (K=4, dequantized; the K=5 arms NaN'd the MAF "
             "on quasi-discrete sparse cells)", "",
             "| arm | screening FoM3 | full FoM3 |", "|---|---|---|"]
    for n in ARMS:
        s, f = res.get((n, "screen")), res.get((n, "full"))
        lines.append(f"| {n} | {s['fom3']:.0f} | {f['fom3']:.0f} |" if s and f else
                     f"| {n} | {s['fom3']:.0f} | —" + " |" if s else f"| {n} | FAIL | — |")
    fa, fb = res.get(("full4dq_nobnt", "full")), res.get(("full4dq_bnt", "full"))
    sa, sb = res.get(("full4dq_nobnt", "screen")), res.get(("full4dq_bnt", "screen"))
    for tag, x, y in (("full", fa, fb), ("screen", sa, sb)):
        if x and y:
            lines.append(f"\n**full4d basis-invariance ratio (BNT/noBNT, {tag}): "
                         f"{y['fom3']/x['fom3']:.3f}** (P4b predicts ≈1)")
            break
    with open(f"{OM}/OVERNIGHT_RESULT.md", "a") as fh:
        fh.write("\n".join(lines) + "\n")
    status(f"## full4d retry complete {time.strftime('%F %T')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
