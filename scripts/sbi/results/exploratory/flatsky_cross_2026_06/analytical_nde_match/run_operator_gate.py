#!/usr/bin/env python3
"""GATE C (TARP-DRP 600 + SBC) for the conv / both operator arms computed by run_operator_realnvp.py.

Andreas asked for these to be gated before they enter fig:ablation. Same gate as every other arm in
this campaign (run_joint_matched.py step 4): trains the matched RealNVP on the VMIM 10-D cache over
600 prior-drawn points x 3 seeds, dumps posteriors, computes TARP-DRP + SBC, writes verdict.json.
GPU 1 (free at launch). Resumable.
"""
import json
import os
import subprocess
import time
from pathlib import Path

PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
SBI = "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi"
A = f"{SBI}/results/exploratory/flatsky_cross_2026_06/analytical_nde_match"
OUT = f"{A}/operator_realnvp"
LOGD = f"{OUT}/logs"
GPU = "1"
MEM = "0.85"
SEEDS = "41,42,43"
STEPS = "50000"
NDE = ["--nde-family", "sbilens_realnvp", "--nde-layers", "4", "--nde-hidden", "128"]
ARMS = ["conv", "both"]


def sh(cmd, log):
    env = dict(os.environ, XLA_PYTHON_CLIENT_MEM_FRACTION=MEM, CUDA_VISIBLE_DEVICES=GPU)
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"  [{(time.time()-t0)/60:.1f} min rc={rc}] {os.path.basename(log)}", flush=True)
    return rc


def gate(arm):
    print(f"\n=== gate operator arm {arm} ===", flush=True)
    cc = f"{OUT}/{arm}/cache"
    G = f"{OUT}/{arm}/gate"
    if not Path(f"{G}/verdict.json").exists():
        sh([PY, f"{SBI}/tarp_stratified_val_nde.py", "--train-cache-dir", cc, "--cache-prefix", "l1",
            "--arm-label", f"op_{arm}", "--dumps-root", f"{G}/tarp_drp/dumps", *NDE,
            "--n-points", "600", "--seeds", SEEDS, "--flow-total-steps", STEPS,
            "--cuda-visible-devices", GPU], f"{LOGD}/{arm}_gate.log")
        sh([PY, f"{SBI}/run_tarp_coverage.py", "--dumps-root", f"{G}/tarp_drp/dumps",
            "--outdir", f"{G}/tarp_drp", "--dims", "3"], f"{LOGD}/{arm}_cov.log")
        sh([PY, f"{A}/gate_verdict.py", "--gate-dir", G, "--arms", f"op_{arm}",
            "--json-out", f"{G}/verdict.json"], f"{LOGD}/{arm}_verdict.log")
    try:
        return json.load(open(f"{G}/verdict.json"))
    except Exception as e:
        return {"error": str(e)}


def main():
    Path(LOGD).mkdir(parents=True, exist_ok=True)
    res = {a: gate(a) for a in ARMS}
    rows = ["# Operator conv/both GATE C (TARP-DRP 600 + SBC), matched RealNVP", ""]
    for a in ARMS:
        rows.append(f"- {a}: {json.dumps(res[a])}")
    open(f"{OUT}/RESULT_OPERATOR_GATE.md", "w").write("\n".join(rows) + "\n")
    print("\n" + "\n".join(rows), flush=True)


if __name__ == "__main__":
    main()
