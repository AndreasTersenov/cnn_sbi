#!/usr/bin/env python
"""Self-waiting rerun of the two K=15 arms (C2/C3) that OOM'd at mem-frac 0.45.

K=15 pair2d is 6750-dim -> the train matrix + sampling needs ~32 GB, so it must run
near-sole-tenant. This launcher POLLS for a genuinely free GPU in {0,1,2} (foreign mem
< 2000 MB AND util < 15% sustained), then runs C2 full + C3 full at mem 0.85 (sequential
on the one free GPU), the C2 gate + coverage, and finally regenerates OVERNIGHT2_RESULT.md.
Never tramples a foreign tenant (project rule). Detached:
  (cd scripts/sbi && setsid nohup <py> run_k15_rerun.py > .../k15_rerun.out 2>&1 &)
"""
import subprocess
import sys
import time

sys.path.insert(0, "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi")
from run_overnight_menu_2 import (PY, SBI, OM2, GC2, arm_dir, sweep_cmd, gate_cmd,
                                  run_job, job_env, status_append)

ARMS = ["C2_pair2d_k15", "C3_pair2d_k15_bnt_ar"]
MEM = 0.85
POLL_S = 300
FREE_MEM_MB = 2000
FREE_UTIL = 15


def gpu_state(g):
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits", "-i", str(g)],
            capture_output=True, text=True, timeout=20).stdout.strip().split(",")
        return int(out[0]), int(out[1])
    except Exception:
        return 1 << 30, 100


def wait_for_free_gpu(t0):
    while True:
        for g in (1, 0, 2):
            m, u = gpu_state(g)
            if m < FREE_MEM_MB and u < FREE_UTIL:
                # confirm it stays free for a second read (avoid a momentary lull)
                time.sleep(20)
                m2, u2 = gpu_state(g)
                if m2 < FREE_MEM_MB and u2 < FREE_UTIL:
                    status_append(f"- [k15] GPU{g} free ({m2}MB/{u2}%) -> launching K=15")
                    return g
        print(f"[{time.time()-t0:6.0f}s] no free GPU (need <{FREE_MEM_MB}MB,<{FREE_UTIL}%) "
              f"— waiting {POLL_S}s", flush=True)
        time.sleep(POLL_S)


def main():
    t0 = time.time()
    gpu = wait_for_free_gpu(t0)
    for name in ARMS:
        cmd, outdir = sweep_cmd(name, True, gpu, MEM)
        ok, dt = run_job(f"k15_full_{name}", cmd, gpu, MEM, t0, threads=8)
        status_append(f"- [k15] {name} full {'OK' if ok else 'FAIL'} ({dt:.0f}s)")
    # C2 gate (the K-trend calibration point) + coverage
    ok, _ = run_job("k15_gate_C2_pair2d_k15", gate_cmd("C2_pair2d_k15", gpu),
                    gpu, MEM, t0, threads=8)
    if ok:
        run_job("k15_coverage", [PY, "run_tarp_coverage.py",
                                 "--dumps-root", f"{GC2}/tarp_drp/dumps",
                                 "--outdir", f"{GC2}/tarp_drp", "--dims", "3"],
                gpu, MEM, t0, threads=8)
    # regenerate the result file with K=15 folded in
    subprocess.run([PY, "run_overnight_menu_2.py", "--regen-only"], cwd=SBI)
    status_append(f"- [k15] rerun complete ({(time.time()-t0)/3600:.1f} h) — "
                  "OVERNIGHT2_RESULT.md regenerated")
    print("K15 RERUN DONE", flush=True)


if __name__ == "__main__":
    main()
