#!/usr/bin/env python
"""§5.4 one-extra-deep-channel test campaign (PLAN_BNTDEEP_TEST.md; GO 2026-06-11).

Pre-registered: recovered_deep = (deep5 - BNT)/(noBNT - BNT) >= 0.8 on the l1 auto arm
(pooled 3-MAF 9000-obs median FoM3). Phases: build (build_flatsky_bntdeep_arm.py: deep
sigma + calibration + train/val passes + alignment-asserted concat + fiducial concat) ->
jitted population sweep (3 MAF seeds) -> derived BNTDEEP_RESULT.md.

Detached:  setsid nohup python run_flatsky_bntdeep_campaign.py --gpus 1 &
"""
import argparse
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
BD = f"{BASE}/bntdeep_campaign"
LOGS = f"{BD}/logs"


def build_cmd(gpu):
    return [PY, "build_flatsky_bntdeep_arm.py"]


def sweep_cmd(gpu):
    return [PY, "population_sweep_flatsky.py",
            "--train-cache-dir", f"{BD}/l1_matrix/l1_none_cache/flat_local_none_bntdeep",
            "--cache-prefix", "l1", "--arm-label", "bntdeep_l1_none",
            "--fiducial-summaries-npz", f"{BD}/fiducial_summaries/fiducial_summaries_l1_none.npz",
            "--output-dir", f"{BD}/population_sweep/l1_none",
            "--preproc-transform", "log1p-zscore", "--clip-value", "5",
            "--min-feature-variance", "1e-5",
            "--seeds", "41,42,43", "--n-obs", "9000", "--max-perm", "50",
            "--m-samples", "2000", "--cuda-visible-devices", str(gpu)]


def run_phase(name, cmd, gpu, t0):
    os.makedirs(LOGS, exist_ok=True)
    log_path = f"{LOGS}/{name}.log"
    print(f"[{time.time()-t0:7.0f}s] ===== PHASE {name} ===== (GPU{gpu}, log {log_path})",
          flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="3",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.5",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", CNN_CPU_THREADS="8",
               OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
               CUDA_VISIBLE_DEVICES=str(gpu))
    with open(log_path, "w") as log:
        rc = subprocess.run(cmd, cwd=SBI, env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL).returncode
    print(f"[{time.time()-t0:7.0f}s] {'DONE' if rc == 0 else 'FAIL'} {name} (rc={rc})",
          flush=True)
    return rc == 0


def med(path):
    f = Path(path) / "median_summary.json"
    return json.load(open(f)) if f.exists() else None


def write_result():
    nob = med(f"{BASE}/population_sweep/flat_none")
    bnt = med(f"{BASE}/bnt_campaign/population_sweep/l1_none")
    d5 = med(f"{BD}/population_sweep/l1_none")
    L = ["# §5.4 one-extra-deep-channel test — derived result", "",
         "5-channel L1 = 4 untouched BNT maps + the plain bin average (deep channel).",
         "Pre-registered (BNT_THEORY_DEEP_DIVE.md §5.4, before any data): recovered >= 0.8.",
         "Pooled 3-MAF 9000-obs median FoM3; same MAF/sweep machinery as all arms.", ""]
    if not (nob and bnt and d5):
        L += ["**INCOMPLETE** — missing median_summary.json: "
              f"noBNT={bool(nob)} BNT={bool(bnt)} deep5={bool(d5)}"]
    else:
        rec = (d5["fom3"] - bnt["fom3"]) / (nob["fom3"] - bnt["fom3"])
        L += ["| arm | FoM3 | sigma(s8) | sigma(w0) |", "|---|---|---|---|",
              f"| L1 noBNT auto | {nob['fom3']:.0f} | {nob['sigma_s8']:.3f} | {nob['sigma_w0']:.3f} |",
              f"| L1 BNT auto | {bnt['fom3']:.0f} | {bnt['sigma_s8']:.3f} | {bnt['sigma_w0']:.3f} |",
              f"| L1 BNT + deep (5ch) | {d5['fom3']:.0f} | {d5['sigma_s8']:.3f} | {d5['sigma_w0']:.3f} |",
              "", f"**recovered = (deep5 − BNT)/(noBNT − BNT) = {rec:.3f}**", ""]
        if rec >= 0.8:
            L += ["**Verdict: prediction PASSES (>= 0.8)** — the no-deep-direction account is "
                  "supported: ONE fixed appended deep channel restores the bulk of the "
                  "per-channel information while leaving the nulled maps untouched."]
        elif rec >= 0.4:
            L += ["**Verdict: PARTIAL (0.4-0.8)** — the deep direction carries a substantial "
                  "share but the account is incomplete; the remainder lives in structure the "
                  "single average does not expose."]
        else:
            L += ["**Verdict: prediction REFUTED (< 0.4)** — the deep-direction account fails "
                  "its decisive test; revisit §5.3."]
        L += ["", "NB mechanism test in the UNCUT information-accounting setting — not a "
              "survey recipe (the deep channel would need conservative cuts; deep-dive "
              "§1.7 item 2 caveat)."]
    Path(BD, "BNTDEEP_RESULT.md").write_text("\n".join(L) + "\n")
    print(f"wrote {BD}/BNTDEEP_RESULT.md", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="1")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    gpu = a.gpus.split(",")[0]
    if a.dry_run:
        print("build:", " ".join(build_cmd(gpu)))
        print("sweep:", " ".join(sweep_cmd(gpu)))
        return 0
    t0 = time.time()
    if not run_phase("build", build_cmd(gpu), gpu, t0):
        print("build FAILED — aborting (no sweep)", flush=True)
        write_result()
        return 1
    if not run_phase("sweep", sweep_cmd(gpu), gpu, t0):
        print("sweep FAILED", flush=True)
        write_result()
        return 1
    write_result()
    print(f"campaign complete in {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
