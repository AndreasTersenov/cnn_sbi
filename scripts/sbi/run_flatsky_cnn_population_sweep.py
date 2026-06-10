#!/usr/bin/env python
"""Population sweep: 4 flat-local CNN arms x 9000 fiducial obs -> median sigma/2D/FoM3.

Sibling of run_flatsky_population_sweep.py for the CNN arms. Per arm: retrain 3 common-MAF seeds
in-process on the CNN compressed cache, sample posteriors at 9000 fiducial obs (perm<50 x 180,
the CNN fiducial summaries), pool 3 seeds/obs, record per-patch sigma/2D/FoM3, report the MEDIAN.
Common MAF + same 9000-obs metric as L1 => apples-to-apples de-leaked L1-vs-CNN. CNN preproc =
none / clip 0 / min-var 1e-12. Greedy 2-GPU scheduler. --dry-run prints commands.
"""
import argparse, os, subprocess, time

REPO = "/mnt/home/tersenov/software/cnn_sbi"
SBI = f"{REPO}/scripts/sbi"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
CNN = f"{SBI}/results/exploratory/flatsky_cross_2026_06/cnn_phase"
FS = f"{CNN}/fiducial_summaries"
POP = f"{CNN}/population_sweep"
LOGS = f"{POP}/logs"

ARMS = [f"{op}" for op in ("none", "conv", "product", "both")]


def cmd(op, gpu):
    return [PY, "population_sweep_flatsky.py",
            "--train-cache-dir", f"{CNN}/cnn_{op}_s41/cache", "--cache-prefix", "cnn",
            "--arm-label", f"flat_{op}",
            "--fiducial-summaries-npz", f"{FS}/fiducial_summaries_{op}.npz",
            "--output-dir", f"{POP}/flat_{op}",
            "--preproc-transform", "none", "--clip-value", "0",
            "--min-feature-variance", "1e-12", "--seeds", "41,42,43",
            "--n-obs", "9000", "--max-perm", "50", "--m-samples", "2000",
            "--cuda-visible-devices", str(gpu)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gpus", default="1,2", help="Comma-separated GPU ids (default 1,2).")
    ap.add_argument("--mem-fraction", default="0.5",
                    help="XLA_PYTHON_CLIENT_MEM_FRACTION (lower on a GPU with pre-allocated mem).")
    args = ap.parse_args()
    GPUS = [int(g) for g in args.gpus.split(",")]
    if args.dry_run:
        for op in ARMS:
            print(f"\n# flat_{op}\n" + " ".join(cmd(op, "<GPU>")))
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
                   XLA_PYTHON_CLIENT_PREALLOCATE="false",
                   XLA_PYTHON_CLIENT_MEM_FRACTION=str(args.mem_fraction),
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                   OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
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

    print(f"\n=== CNN population sweep done in {(time.time()-t0)/60:.1f} min ===")
    print(f"  done: {done}")
    if failed:
        print(f"  FAILED: {failed}")


if __name__ == "__main__":
    main()
