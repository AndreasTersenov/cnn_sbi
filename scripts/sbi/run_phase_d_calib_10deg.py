#!/usr/bin/env python
"""Phase D calibration orchestrator: TARP + SBC + L-C2ST for the 4 arms on GPU 1+2.

- TARP (tarp_per_patch_fiducial): Mahalanobis-chi2_3 coverage of the fiducial truth over
  the patch population -> over-confident vs calibrated (is the tightness / FoM3 trustworthy?). 4 arms.
- SBC (sbc_diagnostic): global rank uniformity over the val grid cosmologies. 4 arms.
- L-C2ST (lc2st_diagnostic, clf logreg): local calibration at the fiducial. CNN arms only
  (L1's 2000-d x is underpowered/flaky for a plain L-C2ST classifier; decided 2026-06-06).

Each script re-trains 3 jaxili MAF seeds from the Phase-C cache (self-contained), matching
the arm's preproc (L1 log1p-zscore-clip5-mask1e-5; CNN none-mask1e-12). 10 jobs, greedy
2-GPU scheduler. --dry-run prints commands.
"""
import argparse
import os
import subprocess
import time

REPO = "/mnt/home/tersenov/software/cnn_sbi"
SBI = f"{REPO}/scripts/sbi"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
PC = f"{SBI}/results/exploratory/definitive_comparison_10deg/phase_c"
AN = f"{PC}/analysis"
LOGS = f"{AN}/calib_logs"
GPUS = [1, 2]
SEEDS = "41,42,43"

# (label, train_cache, prefix, summaries, transform, clip, minvar)
ARMS = [
    ("l1_auto_cross", f"{PC}/l1_auto_cross_cache", "l1",
     f"{PC}/fiducial_summaries/l1_auto_cross.npz", "log1p-zscore", "5", "1e-5"),
    ("l1_auto_only", f"{PC}/l1_auto_only_cache", "l1",
     f"{PC}/fiducial_summaries/l1_auto_only.npz", "log1p-zscore", "5", "1e-5"),
    ("cnn_auto_cross", f"{PC}/cnn_auto_cross_s41/cache", "cnn",
     f"{PC}/fiducial_summaries/cnn_auto_cross.npz", "none", "0", "1e-12"),
    ("cnn_auto_only", f"{PC}/cnn_auto_only_s41/cache", "cnn",
     f"{PC}/fiducial_summaries/cnn_auto_only.npz", "none", "0", "1e-12"),
]


def tarp(arm, gpu):
    label, cache, prefix, summ, t, c, mv = arm
    return [PY, "tarp_per_patch_fiducial.py", "--train-cache-dir", cache,
            "--cache-prefix", prefix, "--summaries-npz", summ, "--arm-label", label,
            "--dumps-root", f"{AN}/tarp/dumps", "--output-dir", f"{AN}/tarp",
            "--preproc-transform", t, "--clip-value", c, "--min-feature-variance", mv,
            "--seeds", SEEDS, "--cuda-visible-devices", str(gpu)]


def sbc(arm, gpu):
    label, cache, prefix, summ, t, c, mv = arm
    return [PY, "sbc_diagnostic.py", "--train-cache-dir", cache,
            "--cache-prefix", prefix, "--arm-label", label, "--output-dir", f"{AN}/sbc",
            "--preproc-transform", t, "--clip-value", c, "--min-feature-variance", mv,
            "--seeds", SEEDS, "--cuda-visible-devices", str(gpu)]


def lc2st(arm, gpu):
    label, cache, prefix, summ, t, c, mv = arm
    return [PY, "lc2st_diagnostic.py", "--train-cache-dir", cache,
            "--cache-prefix", prefix, "--arm-label", label, "--output-dir", f"{AN}/lc2st",
            "--fiducial-summaries-npz", summ, "--preproc-transform", t, "--clip-value", c,
            "--min-feature-variance", mv, "--seeds", SEEDS, "--clf-kind", "logreg",
            "--cuda-visible-devices", str(gpu)]


def build_jobs():
    jobs = []
    for a in ARMS:
        jobs.append((f"tarp_{a[0]}", tarp, a))
        jobs.append((f"sbc_{a[0]}", sbc, a))
    for a in ARMS:
        if a[2] == "cnn":  # L-C2ST CNN-only
            jobs.append((f"lc2st_{a[0]}", lc2st, a))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    jobs = build_jobs()
    if args.dry_run:
        for nm, fn, a in jobs:
            print(f"\n# {nm}\n" + " ".join(fn(a, "<GPU>")))
        print(f"\n({len(jobs)} jobs)")
        return
    os.makedirs(LOGS, exist_ok=True)
    os.chdir(SBI)
    pending = list(jobs)
    slots = {g: None for g in GPUS}
    t0 = time.time()
    done, failed = [], {}

    def launch(job, gpu):
        nm, fn, a = job
        log = open(f"{LOGS}/{nm}.log", "w")
        env = dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="3",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.4")
        p = subprocess.Popen(fn(a, gpu), cwd=SBI, env=env, stdout=log,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        print(f"[{time.time()-t0:7.0f}s] LAUNCH {nm} on GPU{gpu} (pid {p.pid})", flush=True)
        return (nm, p, log)

    while pending or any(slots.values()):
        for g in GPUS:
            s = slots[g]
            if s and s[1].poll() is not None:
                nm, p, log = s
                log.close()
                slots[g] = None
                (done.append(nm) if p.returncode == 0 else failed.__setitem__(nm, p.returncode))
                print(f"[{time.time()-t0:7.0f}s] {'DONE' if p.returncode==0 else 'FAIL'} {nm} "
                      f"rc={p.returncode}", flush=True)
        for g in GPUS:
            if slots[g] is None and pending:
                slots[g] = launch(pending.pop(0), g)
        time.sleep(10)

    print(f"\n=== Phase D calibration finished in {(time.time()-t0)/60:.1f} min ===")
    print(f"  done ({len(done)}): {done}" + (f" | FAILED: {failed}" if failed else " | all OK"))


if __name__ == "__main__":
    main()
