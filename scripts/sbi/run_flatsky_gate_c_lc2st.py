#!/usr/bin/env python
"""GATE C — L-C2ST (local calibration at the fiducial) for the flat-local L1 arms.

lc2st_diagnostic.py: retrain MAF in-process (no reload truncation), then a from-scratch sklearn
L-C2ST at n_x0=30 typical fiducial obs, with a permutation null + a self-test power gate. Detects
LOCAL miscalibration at the fiducial that cancels globally (invisible to SBC) — e.g. an L1 w0
offset. frac_reject_p05 ~ 0.05 => locally calibrated. Greedy 2-GPU scheduler. --dry-run prints.
"""
import argparse, os, subprocess, time

REPO = "/mnt/home/tersenov/software/cnn_sbi"
SBI = f"{REPO}/scripts/sbi"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
ML = f"{SBI}/results/exploratory/flatsky_cross_2026_06/l1_matrix"
GC = f"{SBI}/results/exploratory/flatsky_cross_2026_06/gate_c"
LC = f"{GC}/lc2st"
LOGS = f"{LC}/logs"
GPUS = [1, 2]

ARMS = [
    ("flat_none", f"{ML}/l1_none_cache/flat_local_none", f"{LC}/fiducial_summaries_none.npz"),
    ("flat_conv", f"{ML}/l1_conv_cache/flat_local_conv", f"{LC}/fiducial_summaries_conv.npz"),
    ("flat_product", f"{ML}/l1_product_cache/flat_local_product", f"{LC}/fiducial_summaries_product.npz"),
    ("flat_both", f"{ML}/l1_both_cache/flat_local_both", f"{LC}/fiducial_summaries_both.npz"),
]


def cmd(arm, gpu):
    label, cache, fid = arm
    return [PY, "lc2st_diagnostic.py", "--train-cache-dir", cache, "--cache-prefix", "l1",
            "--arm-label", label, "--output-dir", f"{LC}/{label}",
            "--fiducial-summaries-npz", fid,
            "--preproc-transform", "log1p-zscore", "--clip-value", "5",
            "--min-feature-variance", "1e-5", "--seeds", "41,42,43",
            "--cuda-visible-devices", str(gpu)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        for a in ARMS:
            print(f"\n# {a[0]}\n" + " ".join(cmd(a, "<GPU>")))
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
                   OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
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

    print(f"\n=== GATE C L-C2ST finished in {(time.time()-t0)/60:.1f} min ===")
    print(f"  done: {done}" + (f"  FAILED: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
