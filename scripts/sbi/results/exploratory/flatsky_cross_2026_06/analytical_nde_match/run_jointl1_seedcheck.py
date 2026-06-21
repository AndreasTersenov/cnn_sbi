#!/usr/bin/env python3
"""3-seed COMPRESSOR robustness check on the Q1 winner jointl1_nobnt (FoM3 3754, PASS-caveat).

Single compressor seed (41) is the one caveat before "analytical jointl1 ≈ CNN, calibrated"
can touch M1 — pair2d/A1's apparent gains were COMPRESSOR-seed-sensitive. Re-VMIM the SAME
raw jointl1 datavector with seeds 42 & 43 (41 reused), each → sbi_lens RealNVP 4x128 (NDE
seeds 41,42,43 pooled, n=9000) → GATE C. Report the 3-seed FoM3 band + calibration.

Decision: band tight (~±5%) AND all calibrated (≤ PASS-with-caveat) → headline stands.
GPU 2. Resumable. Writes RESULT_JOINTL1_SEEDCHECK.md.
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
RAW = f"{A}/jointl1_nobnt_raw"
NDE = ["--nde-family", "sbilens_realnvp", "--nde-layers", "4", "--nde-hidden", "128"]
SEEDS_NEW = ["42", "43"]   # 41 already done as jointl1_nobnt


def sh(cmd, log):
    env = dict(os.environ, XLA_PYTHON_CLIENT_MEM_FRACTION=MEM, CUDA_VISIBLE_DEVICES=GPU)
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


def run_seed(s):
    print(f"\n=== compressor seed {s} ===", flush=True)
    odir = f"{A}/jointl1_nobnt_s{s}"
    cc, cf = f"{odir}/cache", f"{odir}/fiducial_summaries.npz"
    if not Path(f"{cc}/l1_train.npz").exists():
        print("  vmim ...", flush=True)
        sh([PY, f"{SBI}/vmim_from_cache.py", "--cache-dir", f"{RAW}/cache",
            "--fid-npz", f"{RAW}/fid.npz", "--out-cache", cc, "--out-fid", cf,
            "--summary-dim", "10", "--seed", s, "--cuda-visible-devices", GPU],
           f"{LOGD}/seedchk_s{s}_vmim.log")
    out = f"{odir}/n9000"
    if not Path(f"{out}/median_summary.json").exists():
        print("  train n9000 ...", flush=True)
        sh([PY, f"{SBI}/train_nde_from_compressed.py", "--train-cache-dir", cc,
            "--cache-prefix", "l1", "--arm-label", f"jointl1_s{s}",
            "--fiducial-summaries-npz", cf, "--output-dir", out, *NDE,
            "--n-obs", "9000", "--seeds", "41,42,43", "--flow-total-steps", "50000",
            "--cuda-visible-devices", GPU], f"{LOGD}/seedchk_s{s}_n9000.log")
    G = f"{odir}/gate"
    if not Path(f"{G}/verdict.json").exists():
        print("  gate ...", flush=True)
        sh([PY, f"{SBI}/tarp_stratified_val_nde.py", "--train-cache-dir", cc,
            "--cache-prefix", "l1", "--arm-label", f"jointl1_s{s}",
            "--dumps-root", f"{G}/tarp_drp/dumps", *NDE, "--n-points", "600",
            "--seeds", "41,42,43", "--flow-total-steps", "50000",
            "--cuda-visible-devices", GPU], f"{LOGD}/seedchk_s{s}_gate.log")
        sh([PY, f"{SBI}/run_tarp_coverage.py", "--dumps-root", f"{G}/tarp_drp/dumps",
            "--outdir", f"{G}/tarp_drp", "--dims", "3"], f"{LOGD}/seedchk_s{s}_cov.log")
        sh([PY, f"{A}/gate_verdict.py", "--gate-dir", G, "--arms", f"jointl1_s{s}",
            "--json-out", f"{G}/verdict.json"], f"{LOGD}/seedchk_s{s}_verdict.log")
    v = None
    try:
        v = json.load(open(f"{G}/verdict.json"))[0]["verdict"]
    except Exception:
        pass
    return dict(seed=s, fom3=fom3(out), verdict=v)


def main():
    Path(LOGD).mkdir(parents=True, exist_ok=True)
    rows = [dict(seed="41", fom3=fom3(f"{A}/jointl1_nobnt/n9000"), verdict="PASS-with-caveat")]
    for s in SEEDS_NEW:
        rows.append(run_seed(s))
    foms = [r["fom3"] for r in rows if r["fom3"]]
    mean = sum(foms) / len(foms) if foms else None
    spread = (max(foms) - min(foms)) / mean * 100 if foms and mean else None
    out = ["# jointl1_nobnt — 3-seed COMPRESSOR robustness (Q1 winner)\n",
           "VMIM(seed s) → sbi_lens RealNVP 4x128 (NDE 41,42,43 pooled, n=9000) → GATE C.",
           "Reference: ℓ1+product 3045, CNN 3326.\n",
           "| compressor seed | FoM3 n=9000 | gate |", "|---|---|---|"]
    for r in rows:
        f = f"{r['fom3']:.0f}" if r["fom3"] else "?"
        out.append(f"| {r['seed']} | {f} | {r['verdict'] or '?'} |")
    out.append("")
    if mean:
        out.append(f"**3-seed band: {min(foms):.0f}–{max(foms):.0f} (mean {mean:.0f}, spread {spread:.0f}%)**")
    txt = "\n".join(out) + "\n"
    open(f"{A}/RESULT_JOINTL1_SEEDCHECK.md", "w").write(txt)
    print("\n" + txt, flush=True)


if __name__ == "__main__":
    main()
