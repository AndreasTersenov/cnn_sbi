#!/usr/bin/env python3
"""Operator arms through the MATCHED best-NDE pipeline (VMIM 10-D -> sbi_lens RealNVP 4x128).

Purpose: the per-channel l1 cross-map operator ablation (auto / +conv / +product / +both) was
only ever evaluated in the fixed-shared-flow (MAF, Table 2) frame. To put ALL points on ONE
matched-RealNVP axis with the joint l1 (the single-plot ablation Andreas wants), recompute conv
and both through the SAME pipeline as the committed l1+product matched arm (3045).

PRODUCT is run first as a VALIDATION: this driver reuses the l1_matrix RAW datavector caches
(train/val) + the gate_c/lc2st RAW fiducial datavectors, which differ in provenance from the
committed l1product_vmim_s41 arm. If product reproduces ~3045 here, the caches are compatible and
conv/both are trustworthy on the same footing. (none=2448, product=3045, joint l1=3371 are committed.)

NB: these are NEW, currently-UNGATED numbers (no TARP/SBC here) — illustrative of the FoM3 trend
for the ablation figure. GPU 1 (free at launch). Resumable (skips completed steps).
"""
import json
import os
import subprocess
import time
from pathlib import Path

PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
SBI = "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi"
FC = f"{SBI}/results/exploratory/flatsky_cross_2026_06"
A = f"{FC}/analytical_nde_match"
OUT = f"{A}/operator_realnvp"
LOGD = f"{OUT}/logs"
GPU = "1"
MEM = "0.85"
SEEDS = "41,42,43"
STEPS = "50000"
NDE = ["--nde-family", "sbilens_realnvp", "--nde-layers", "4", "--nde-hidden", "128"]

# arm -> (raw train/val cache dir, raw fiducial datavector npz)
ARMS = {
    "product": (f"{FC}/l1_matrix/l1_product_cache/flat_local_product", f"{FC}/gate_c/lc2st/fiducial_summaries_product.npz"),
    "conv":    (f"{FC}/l1_matrix/l1_conv_cache/flat_local_conv",       f"{FC}/gate_c/lc2st/fiducial_summaries_conv.npz"),
    "both":    (f"{FC}/l1_matrix/l1_both_cache/flat_local_both",       f"{FC}/gate_c/lc2st/fiducial_summaries_both.npz"),
}


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


def run_arm(name, raw_cache, raw_fid):
    print(f"\n=== operator arm {name} ===", flush=True)
    odir = f"{OUT}/{name}"
    cc, cf = f"{odir}/cache", f"{odir}/fid_summ.npz"
    Path(odir).mkdir(parents=True, exist_ok=True)

    if not Path(f"{cc}/l1_train.npz").exists():
        print("  vmim compress -> 10-D ...", flush=True)
        if sh([PY, f"{SBI}/vmim_from_cache.py", "--cache-dir", raw_cache, "--fid-npz", raw_fid,
               "--out-cache", cc, "--out-fid", cf, "--summary-dim", "10", "--seed", "41",
               "--cuda-visible-devices", GPU], f"{LOGD}/{name}_vmim.log") != 0:
            print(f"  VMIM FAILED {name}"); return None

    out = f"{odir}/n9000"
    if not Path(f"{out}/median_summary.json").exists():
        print("  train RealNVP + eval n=9000 ...", flush=True)
        sh([PY, f"{SBI}/train_nde_from_compressed.py", "--train-cache-dir", cc,
            "--cache-prefix", "l1", "--arm-label", f"op_{name}", "--fiducial-summaries-npz", cf,
            "--output-dir", out, *NDE, "--n-obs", "9000", "--seeds", SEEDS,
            "--flow-total-steps", STEPS, "--cuda-visible-devices", GPU], f"{LOGD}/{name}_n9000.log")
    return fom3(out)


def main():
    Path(LOGD).mkdir(parents=True, exist_ok=True)
    res = {}
    for name, (rc, rf) in ARMS.items():
        res[name] = run_arm(name, rc, rf)
        print(f"  -> {name} FoM3(n9000) = {res[name]}", flush=True)
    rows = ["# Operator arms through matched RealNVP (VMIM 10-D -> sbi_lens RealNVP 4x128, n=9000)",
            "Reference frame: same pipeline as committed l1+product (3045). none=2448, joint l1=3371 (committed).",
            "Validation: 'product' here should reproduce ~3045. NEW/UNGATED numbers (no TARP/SBC).", "",
            "| operator | FoM3 n=9000 (RealNVP) | MAF frame (Table 2) |", "|---|---|---|"]
    maf = {"product": 2875, "conv": 2499, "both": 2910}
    for n in ("conv", "product", "both"):
        v = res.get(n)
        rows.append(f"| {n} | {v:.0f} | {maf[n]} |" if v else f"| {n} | FAILED | {maf[n]} |")
    txt = "\n".join(rows) + "\n"
    open(f"{OUT}/RESULT_OPERATOR_REALNVP.md", "w").write(txt)
    print("\n" + txt, flush=True)


if __name__ == "__main__":
    main()
