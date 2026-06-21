#!/usr/bin/env python3
"""Goal-1 D1 follow-up: gate the deep-channel ladder under the CONSERVATIVE estimator (jaxili MAF).

RealNVP over-recovered deep2 (FoM3 3498, recovered 1.53) but FAILED the gate (SBC std 0.318/0.325/
0.305 -> s8 over-confident). M1 established MAF is ~30% lower but more conservative on the SAME
10-D summary. Question: is the deep-channel rescue CALIBRATED under MAF, and at what recovery level?

Reuses the EXISTING 10-D compressed caches (same VMIM summaries, only the downstream NDE changes):
  nobnt_auto     : l1none_vmim_s41/cache
  bnt_auto       : bnt_auto/cache
  bnt_auto_deep2 : bnt_auto_deep2/cache
NDE = jaxili_maf (family default 5x50), seeds 41,42,43, n=9000 median, gated (TARP-DRP 600 + SBC).
GPU 2. Resumable. Writes RESULT_MAF_LADDER_GATE.md.
"""
import json
import os
import subprocess
import time
from pathlib import Path

PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
SBI = "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi"
A = f"{SBI}/results/exploratory/flatsky_cross_2026_06/analytical_nde_match"
LOGD = f"{A}/logs"
GPU = "2"
MEM = "0.8"
SEEDS = "41,42,43"
STEPS = "50000"
NDE = ["--nde-family", "jaxili_maf"]   # family default capacity (5x50)

ARMS = {
    "nobnt_auto": dict(cache=f"{A}/l1none_vmim_s41/cache", fid=f"{A}/l1none_vmim_s41/fiducial_summaries.npz", gate=False),
    "bnt_auto": dict(cache=f"{A}/bnt_auto/cache", fid=f"{A}/bnt_auto/fiducial_summaries.npz", gate=True),
    "bnt_auto_deep2": dict(cache=f"{A}/bnt_auto_deep2/cache", fid=f"{A}/bnt_auto_deep2/fiducial_summaries.npz", gate=True),
}


def sh(cmd, log):
    env = dict(os.environ, XLA_PYTHON_CLIENT_MEM_FRACTION=MEM)
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"  [{(time.time()-t0)/60:.1f} min, rc={rc}] {os.path.basename(log)}", flush=True)
    return rc


def fom3(odir):
    try:
        return json.load(open(f"{odir}/median_summary.json"))["fom3"]
    except Exception:
        return None


def main():
    Path(LOGD).mkdir(parents=True, exist_ok=True)
    res = {}
    for name, cfg in ARMS.items():
        print(f"\n=== MAF arm {name} ===", flush=True)
        out = f"{A}/{name}/maf_n9000"
        if not Path(f"{out}/median_summary.json").exists():
            sh([PY, f"{SBI}/train_nde_from_compressed.py", "--train-cache-dir", cfg["cache"],
                "--cache-prefix", "l1", "--arm-label", f"{name}_maf",
                "--fiducial-summaries-npz", cfg["fid"], "--output-dir", out, *NDE,
                "--n-obs", "9000", "--seeds", SEEDS, "--flow-total-steps", STEPS,
                "--cuda-visible-devices", GPU], f"{LOGD}/maf_{name}_n9000.log")
        verdict = None
        if cfg["gate"]:
            G = f"{A}/{name}/maf_gate"
            if not Path(f"{G}/verdict.json").exists():
                sh([PY, f"{SBI}/tarp_stratified_val_nde.py", "--train-cache-dir", cfg["cache"],
                    "--cache-prefix", "l1", "--arm-label", f"{name}_maf",
                    "--dumps-root", f"{G}/tarp_drp/dumps", *NDE, "--n-points", "600",
                    "--seeds", SEEDS, "--flow-total-steps", STEPS,
                    "--cuda-visible-devices", GPU], f"{LOGD}/maf_{name}_gate.log")
                sh([PY, f"{SBI}/run_tarp_coverage.py", "--dumps-root", f"{G}/tarp_drp/dumps",
                    "--outdir", f"{G}/tarp_drp", "--dims", "3"], f"{LOGD}/maf_{name}_cov.log")
                sh([PY, f"{A}/gate_verdict.py", "--gate-dir", G, "--arms", f"{name}_maf",
                    "--json-out", f"{G}/verdict.json"], f"{LOGD}/maf_{name}_verdict.log")
            try:
                verdict = json.load(open(f"{G}/verdict.json"))[0]
            except Exception:
                pass
        res[name] = dict(fom3=fom3(out), verdict=verdict)

    nob, bnt, d2 = (res[k]["fom3"] for k in ("nobnt_auto", "bnt_auto", "bnt_auto_deep2"))
    rec = (d2 - bnt) / (nob - bnt) if None not in (nob, bnt, d2) and (nob - bnt) else None
    rows = ["# Goal-1 D1 follow-up — deep-channel ladder under jaxili MAF (conservative NDE), gated\n",
            "Same 10-D VMIM summaries as the RealNVP ladder; only the downstream NDE changes (MAF 5x50).",
            "RealNVP ladder: noBNT 2437 / BNT 425 / deep2 3498 (rec 1.53); deep2 gate FAIL (s8 SBC 0.325).",
            "Old-path MAF reference (ungated): noBNT 2405 / BNT 364 / deep2 2573 (rec 1.082).\n",
            "| arm | MAF FoM3 n=9000 | gate verdict |", "|---|---|---|"]
    for name in ("nobnt_auto", "bnt_auto", "bnt_auto_deep2"):
        r = res[name]
        v = r["verdict"]
        vt = v["verdict"] if v else ("-" if name == "nobnt_auto" else "?")
        rows.append(f"| {name} | {r['fom3']:.0f} | {vt} |" if r["fom3"] else f"| {name} | ? | {vt} |")
    rows.append("")
    if rec is not None:
        rows.append(f"**recovered (MAF, n=9000) = ({d2:.0f} - {bnt:.0f})/({nob:.0f} - {bnt:.0f}) = {rec:.3f}**")
    out = "\n".join(rows) + "\n"
    open(f"{A}/RESULT_MAF_LADDER_GATE.md", "w").write(out)
    print("\n" + out, flush=True)


if __name__ == "__main__":
    main()
