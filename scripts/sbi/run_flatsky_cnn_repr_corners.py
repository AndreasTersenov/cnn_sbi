#!/usr/bin/env python
"""Orchestrate cnn_representative_corners.py for the 4 flat-local CNN arms (3-seed)."""
import argparse, os, subprocess, time

REPO = "/mnt/home/tersenov/software/cnn_sbi"; SBI = f"{REPO}/scripts/sbi"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
CNN = f"{SBI}/results/exploratory/flatsky_cross_2026_06/cnn_phase"
FS = f"{CNN}/fiducial_summaries"
OUT = f"{CNN}/representative_corner"
LOGS = f"{OUT}/logs"
ARMS = ["none", "conv", "product", "both"]


def cmd(op, gpu):
    return [PY, "cnn_representative_corners.py", "--arm-label", f"flat_{op}",
            "--train-cache-dir", f"{CNN}/cnn_{op}_s41/cache", "--cache-prefix", "cnn",
            "--fiducial-summaries-npz", f"{FS}/fiducial_summaries_{op}.npz",
            "--output-dir", f"{OUT}/flat_{op}", "--seeds", "41,42,43",
            "--preproc-transform", "none", "--clip-value", "0", "--min-feature-variance", "1e-12",
            "--cuda-visible-devices", str(gpu)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gpus", default="1,2,3")
    ap.add_argument("--mem-fraction", default="0.4")
    args = ap.parse_args()
    GPUS = [int(g) for g in args.gpus.split(",")]
    if args.dry_run:
        for op in ARMS:
            print(f"\n# flat_{op}\n" + " ".join(cmd(op, "<GPU>")))
        return
    os.makedirs(LOGS, exist_ok=True); os.chdir(SBI)
    pending = list(ARMS); slots = {g: None for g in GPUS}; t0 = time.time(); done, failed = [], {}

    def launch(op, gpu):
        log = open(f"{LOGS}/flat_{op}.log", "w")
        env = dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="3",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false",
                   XLA_PYTHON_CLIENT_MEM_FRACTION=str(args.mem_fraction),
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
                op, p, log = s; log.close(); slots[g] = None
                (done.append(op) if p.returncode == 0 else failed.__setitem__(op, p.returncode))
                print(f"[{time.time()-t0:7.0f}s] {'DONE' if p.returncode==0 else 'FAIL'} flat_{op}", flush=True)
        for g in GPUS:
            if slots[g] is None and pending:
                slots[g] = launch(pending.pop(0), g)
        time.sleep(10)
    print(f"\n=== CNN representative corners done in {(time.time()-t0)/60:.1f} min === done={done} failed={failed}")


if __name__ == "__main__":
    main()
