#!/usr/bin/env python3
"""ESCALATION of the two-point-vs-non-Gaussian split (PLAN_2PT_SPLIT.md): n=9000, 3 NDE seeds,
+ GATE C (TARP-DRP 600 + SBC). Reuses the VMIM 10-D caches built by run_twopt_split.py (screen).

Arms: cov / auto_cov / conv_cov / product_cov. Gate the three delta arms (auto/conv/product _cov).
Same matched RealNVP + gate chain as the operator arms. GPU 0. Resumable.
"""
import json
import os
import subprocess
import time
from pathlib import Path

PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
SBI = "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi"
A = f"{SBI}/results/exploratory/flatsky_cross_2026_06/analytical_nde_match"
OUT = f"{A}/twopt_split"
LOGD = f"{OUT}/logs"
GPU = "0"
MEM = "0.85"
SEEDS = "41,42,43"
STEPS = "50000"
NDE = ["--nde-family", "sbilens_realnvp", "--nde-layers", "4", "--nde-hidden", "128"]
ARMS = ["cov", "auto_cov", "conv_cov", "product_cov"]
GATE_ARMS = ["auto_cov", "conv_cov", "product_cov"]


def sh(cmd, log):
    env = dict(os.environ, XLA_PYTHON_CLIENT_MEM_FRACTION=MEM, CUDA_VISIBLE_DEVICES=GPU)
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"  [{(time.time()-t0)/60:.1f} min rc={rc}] {os.path.basename(log)}", flush=True)
    return rc


def fom3(odir):
    try:
        return json.load(open(f"{odir}/median_summary.json"))["fom3"]
    except Exception:
        return None


def run_arm(arm):
    print(f"\n=== escalate {arm} ===", flush=True)
    cc, cf = f"{OUT}/{arm}/cache", f"{OUT}/{arm}/fid_summ.npz"
    if not Path(f"{cc}/l1_train.npz").exists():
        print(f"  MISSING VMIM cache for {arm} (run the screen first)"); return None, None

    out = f"{OUT}/{arm}/n9000"
    if not Path(f"{out}/median_summary.json").exists():
        print("  n=9000 x3 seeds FoM3 ...", flush=True)
        sh([PY, f"{SBI}/train_nde_from_compressed.py", "--train-cache-dir", cc,
            "--cache-prefix", "l1", "--arm-label", f"tp_{arm}", "--fiducial-summaries-npz", cf,
            "--output-dir", out, *NDE, "--n-obs", "9000", "--seeds", SEEDS,
            "--flow-total-steps", STEPS, "--cuda-visible-devices", GPU], f"{LOGD}/{arm}_n9000.log")

    verdict = None
    if arm in GATE_ARMS:
        G = f"{OUT}/{arm}/gate"
        if not Path(f"{G}/verdict.json").exists():
            print("  GATE: tarp_stratified -> coverage -> verdict ...", flush=True)
            sh([PY, f"{SBI}/tarp_stratified_val_nde.py", "--train-cache-dir", cc, "--cache-prefix", "l1",
                "--arm-label", f"tp_{arm}", "--dumps-root", f"{G}/tarp_drp/dumps", *NDE,
                "--n-points", "600", "--seeds", SEEDS, "--flow-total-steps", STEPS,
                "--cuda-visible-devices", GPU], f"{LOGD}/{arm}_gate.log")
            sh([PY, f"{SBI}/run_tarp_coverage.py", "--dumps-root", f"{G}/tarp_drp/dumps",
                "--outdir", f"{G}/tarp_drp", "--dims", "3"], f"{LOGD}/{arm}_cov.log")
            sh([PY, f"{A}/gate_verdict.py", "--gate-dir", G, "--arms", f"tp_{arm}",
                "--json-out", f"{G}/verdict.json"], f"{LOGD}/{arm}_verdict.log")
        try:
            verdict = json.load(open(f"{G}/verdict.json"))
        except Exception as e:
            verdict = {"error": str(e)}
    return fom3(out), verdict


def main():
    Path(LOGD).mkdir(parents=True, exist_ok=True)
    fom, verd = {}, {}
    for arm in ARMS:
        fom[arm], verd[arm] = run_arm(arm)
        print(f"  -> {arm} FoM3(n9000)={fom[arm]} verdict={verd[arm]}", flush=True)

    def d(a, b):
        return None if (fom.get(a) is None or fom.get(b) is None) else fom[a] - fom[b]

    rows = ["# Two-point vs non-Gaussian split — ESCALATION (n=9000, 3 seeds, gated)",
            "Matched VMIM 10-D -> sbilens_realnvp 4x128, no-BNT. cov = complete 2pt sector (P7).", "",
            "| arm | FoM3 n=9000 | gate verdict |", "|---|---|---|"]
    for a in ARMS:
        v = fom.get(a)
        gv = (verd.get(a) or {}).get("verdict") if isinstance(verd.get(a), dict) else verd.get(a)
        rows.append(f"| {a} | {v:.0f} | {gv or '-'} |" if v else f"| {a} | FAILED | {gv or '-'} |")
    pc, dc, dp = d("auto_cov", "cov"), d("conv_cov", "auto_cov"), d("product_cov", "auto_cov")
    rows += ["",
             f"- positive control  auto_cov - cov        = {None if pc is None else round(pc)}",
             f"- ΔNG(conv)         conv_cov - auto_cov    = {None if dc is None else round(dc)}",
             f"- ΔNG(product)      product_cov - auto_cov = {None if dp is None else round(dp)}",
             "", "Full per-arm gate JSON in twopt_split/<arm>/gate/verdict.json."]
    txt = "\n".join(rows) + "\n"
    open(f"{OUT}/RESULT_TWOPT_SPLIT_FULL.md", "w").write(txt)
    print("\n" + txt, flush=True)


if __name__ == "__main__":
    main()
