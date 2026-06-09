#!/usr/bin/env python
"""GATE C — TARP-DRP (varied-theta, FoM3-tercile-stratified) for the flat-local L1 arms.

Reuses the established Phase-D infra (tarp_stratified_val.py): per arm, RE-TRAIN 3 MAF seeds
in-process (sidesteps the jaxili high-dim Standardizer reload-truncation), sample posteriors
at 600 held-out VAL points (theta ~ prior), split by FoM3 tercile, dump in run_tarp_coverage
format. Then run_tarp_coverage.py -> DRP curves per tercile/arm. The HIGH tercile sitting on
the diagonal = the tight posteriors are CALIBRATED (the de-leaked cross gain is real, not over-
tight). Greedy 2-GPU scheduler (GPU 1+2). --dry-run prints commands.
"""
import argparse, os, subprocess, time

REPO = "/mnt/home/tersenov/software/cnn_sbi"
SBI = f"{REPO}/scripts/sbi"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
ML = f"{SBI}/results/exploratory/flatsky_cross_2026_06/l1_matrix"
TARP = f"{SBI}/results/exploratory/flatsky_cross_2026_06/gate_c/tarp_drp"
LOGS = f"{TARP}/logs"
GPUS = [1, 2]
SEEDS = "41,42,43"

ARMS = [
    ("flat_none", f"{ML}/l1_none_cache/flat_local_none"),
    ("flat_conv", f"{ML}/l1_conv_cache/flat_local_conv"),
    ("flat_product", f"{ML}/l1_product_cache/flat_local_product"),
    ("flat_both", f"{ML}/l1_both_cache/flat_local_both"),
]


def cmd(arm, gpu):
    label, cache = arm
    return [PY, "tarp_stratified_val.py", "--train-cache-dir", cache, "--cache-prefix", "l1",
            "--arm-label", label, "--dumps-root", f"{TARP}/dumps",
            "--preproc-transform", "log1p-zscore", "--clip-value", "5",
            "--min-feature-variance", "1e-5", "--seeds", SEEDS,
            "--cuda-visible-devices", str(gpu)]


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
                   XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.5",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                   OMP_NUM_THREADS="6", MKL_NUM_THREADS="6", OPENBLAS_NUM_THREADS="6")
        p = subprocess.Popen(cmd(arm, gpu), cwd=SBI, env=env, stdout=log,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        print(f"[{time.time()-t0:7.0f}s] LAUNCH {arm[0]} GPU{gpu} (pid {p.pid})", flush=True)
        return (arm[0], p, log)

    while pending or any(slots.values()):
        for g in GPUS:
            s = slots[g]
            if s and s[1].poll() is not None:
                nm, p, log = s
                log.close(); slots[g] = None
                (done.append(nm) if p.returncode == 0 else failed.__setitem__(nm, p.returncode))
                print(f"[{time.time()-t0:7.0f}s] {'DONE' if p.returncode==0 else 'FAIL'} {nm} "
                      f"rc={p.returncode}", flush=True)
        for g in GPUS:
            if slots[g] is None and pending:
                slots[g] = launch(pending.pop(0), g)
        time.sleep(8)

    print(f"\n=== GATE C TARP dumps finished in {(time.time()-t0)/60:.1f} min ===")
    print(f"  done: {done}" + (f"  FAILED: {failed}" if failed else ""))
    if not failed:
        print("  running coverage...")
        subprocess.run([PY, "run_tarp_coverage.py", "--dumps-root", f"{TARP}/dumps",
                        "--outdir", TARP, "--dims", "3", "6"], cwd=SBI)


if __name__ == "__main__":
    main()
