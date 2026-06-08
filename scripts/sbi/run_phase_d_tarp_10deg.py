#!/usr/bin/env python
"""Proper varied-θ TARP-DRP for the 4 arms (the 20°-comparable test).

tarp_stratified_val.py draws θ from the prior (held-out VAL ensemble), so TARP-DRP is
VALID (unlike the fixed-θ fiducial Mahalanobis test in tarp_per_patch_fiducial.py).
Per arm: re-train 3 MAF seeds, sample posteriors at N val points, split by FoM3 tercile,
dump in run_tarp_coverage format. Then run_tarp_coverage.py -> DRP curves + per-arm figures.

4 arms, greedy 2-GPU scheduler, then the (CPU) coverage step. --dry-run prints commands.
"""
import argparse
import os
import subprocess
import time

REPO = "/mnt/home/tersenov/software/cnn_sbi"
SBI = f"{REPO}/scripts/sbi"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
PC = f"{SBI}/results/exploratory/definitive_comparison_10deg/phase_c"
TARP = f"{PC}/analysis/tarp_drp"
LOGS = f"{TARP}/logs"
GPUS = [1, 2]
SEEDS = "41,42,43"

ARMS = [
    ("l1_auto_cross", f"{PC}/l1_auto_cross_cache", "l1", "log1p-zscore", "5", "1e-5"),
    ("l1_auto_only", f"{PC}/l1_auto_only_cache", "l1", "log1p-zscore", "5", "1e-5"),
    ("cnn_auto_cross", f"{PC}/cnn_auto_cross_s41/cache", "cnn", "none", "0", "1e-12"),
    ("cnn_auto_only", f"{PC}/cnn_auto_only_s41/cache", "cnn", "none", "0", "1e-12"),
]


def cmd(arm, gpu):
    label, cache, prefix, t, c, mv = arm
    return [PY, "tarp_stratified_val.py", "--train-cache-dir", cache, "--cache-prefix", prefix,
            "--arm-label", label, "--dumps-root", f"{TARP}/dumps",
            "--preproc-transform", t, "--clip-value", c, "--min-feature-variance", mv,
            "--seeds", SEEDS, "--cuda-visible-devices", str(gpu)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        for a in ARMS:
            print(f"\n# {a[0]}\n" + " ".join(cmd(a, "<GPU>")))
        print(f"\n# then: {PY} run_tarp_coverage.py --dumps-root {TARP}/dumps --outdir {TARP} --dims 3 6")
        return
    os.makedirs(LOGS, exist_ok=True)
    os.chdir(SBI)
    pending = list(ARMS)
    slots = {g: None for g in GPUS}
    t0 = time.time()
    done, failed = [], {}

    def launch(arm, gpu):
        log = open(f"{LOGS}/{arm[0]}.log", "w")
        env = dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="3",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.4")
        p = subprocess.Popen(cmd(arm, gpu), cwd=SBI, env=env, stdout=log,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        print(f"[{time.time()-t0:7.0f}s] LAUNCH {arm[0]} on GPU{gpu} (pid {p.pid})", flush=True)
        return (arm[0], p, log)

    while pending or any(slots.values()):
        for g in GPUS:
            s = slots[g]
            if s and s[1].poll() is not None:
                nm, p, log = s
                log.close(); slots[g] = None
                (done.append(nm) if p.returncode == 0 else failed.__setitem__(nm, p.returncode))
                print(f"[{time.time()-t0:7.0f}s] {'DONE' if p.returncode==0 else 'FAIL'} {nm} rc={p.returncode}", flush=True)
        for g in GPUS:
            if slots[g] is None and pending:
                slots[g] = launch(pending.pop(0), g)
        time.sleep(10)

    print(f"\n=== TARP dumps done in {(time.time()-t0)/60:.1f} min: {done} {('FAILED '+str(failed)) if failed else 'all OK'} ===", flush=True)
    if not failed:
        print("=== running run_tarp_coverage (DRP curves + figures) ===", flush=True)
        r = subprocess.run([PY, "run_tarp_coverage.py", "--dumps-root", f"{TARP}/dumps",
                            "--outdir", TARP, "--dims", "3", "6"], cwd=SBI,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1"))
        print(f"run_tarp_coverage rc={r.returncode}", flush=True)


if __name__ == "__main__":
    main()
