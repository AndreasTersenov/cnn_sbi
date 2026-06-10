#!/usr/bin/env python
"""GATE C — TARP-DRP for the flat-local CNN arms (sibling of run_flatsky_gate_c_tarp.py).

Same Phase-D infra (tarp_stratified_val.py): per arm RE-TRAIN 3 common-MAF seeds in-process on
the CNN compressed cache, sample posteriors at 600 held-out VAL points (theta ~ prior), split by
FoM3 tercile, dump in run_tarp_coverage format -> DRP curves. The common MAF removes the NDE
confound (CNN compressor summary -> identical jaxili MAF the L1 arms use). CNN preproc = none /
clip 0 / min-var 1e-12 (run_phase_d_10deg convention), vs L1's log1p-zscore-clip5.
Greedy 2-GPU scheduler (GPU 1+2). --dry-run prints commands.
"""
import argparse, os, subprocess, time

REPO = "/mnt/home/tersenov/software/cnn_sbi"
SBI = f"{REPO}/scripts/sbi"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
CNN = f"{SBI}/results/exploratory/flatsky_cross_2026_06/cnn_phase"
TARP = f"{CNN}/gate_c/tarp_drp"
LOGS = f"{TARP}/logs"
GPUS = [1, 2]
SEEDS = "41,42,43"

ARMS = [f"{op}" for op in ("none", "conv", "product", "both")]


def cmd(op, gpu):
    return [PY, "tarp_stratified_val.py",
            "--train-cache-dir", f"{CNN}/cnn_{op}_s41/cache", "--cache-prefix", "cnn",
            "--arm-label", f"flat_{op}", "--dumps-root", f"{TARP}/dumps",
            "--preproc-transform", "none", "--clip-value", "0",
            "--min-feature-variance", "1e-12", "--seeds", SEEDS,
            "--cuda-visible-devices", str(gpu)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        for op in ARMS:
            print(f"\n# flat_{op}\n" + " ".join(cmd(op, "<GPU>")))
        print(f"\n# then: {PY} run_tarp_coverage.py --dumps-root {TARP}/dumps "
              f"--outdir {TARP} --dims 3 6")
        return
    os.makedirs(LOGS, exist_ok=True)
    os.chdir(SBI)
    pending = list(ARMS)
    slots = {g: None for g in GPUS}
    t0 = time.time()
    done, failed = [], {}

    def launch(op, gpu):
        log = open(f"{LOGS}/flat_{op}.log", "w")
        env = dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="3",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.5",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                   OMP_NUM_THREADS="6", MKL_NUM_THREADS="6", OPENBLAS_NUM_THREADS="6")
        p = subprocess.Popen(cmd(op, gpu), cwd=SBI, env=env, stdout=log,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        print(f"[{time.time()-t0:7.0f}s] LAUNCH flat_{op} GPU{gpu} (pid {p.pid})", flush=True)
        return (op, p, log)

    while pending or any(slots.values()):
        for g in GPUS:
            s = slots[g]
            if s and s[1].poll() is not None:
                op, p, log = s
                log.close(); slots[g] = None
                if p.returncode == 0:
                    done.append(op); print(f"[{time.time()-t0:7.0f}s] DONE   flat_{op} (GPU{g})", flush=True)
                else:
                    failed[op] = p.returncode
                    print(f"[{time.time()-t0:7.0f}s] FAIL   flat_{op} rc={p.returncode} "
                          f"(see {LOGS}/flat_{op}.log)", flush=True)
        for g in GPUS:
            if slots[g] is None and pending:
                slots[g] = launch(pending.pop(0), g)
        time.sleep(10)

    print(f"\n=== CNN GATE C TARP dumps done in {(time.time()-t0)/60:.1f} min ===")
    print(f"  done: {done}")
    if failed:
        print(f"  FAILED: {failed}")
    else:
        print(f"  Next: {PY} run_tarp_coverage.py --dumps-root {TARP}/dumps --outdir {TARP} --dims 3 6")


if __name__ == "__main__":
    main()
