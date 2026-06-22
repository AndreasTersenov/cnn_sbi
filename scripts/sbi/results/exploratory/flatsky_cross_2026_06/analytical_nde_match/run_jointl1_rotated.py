#!/usr/bin/env python3
"""Shear-aware (rotated-grid) joint ℓ1 in BNT space — can it calibrate the 0.86 retention?

Q2 baseline (axis-aligned adaptive-ranges): jointl1_bnt raw retention 0.861 but GATE FAIL
(over-confident SBC 0.33, TARP worst dev 0.110) — the un-transported binning SHEAR (P4c).
Fix: per-(pair,scale) 2-D PCA-whitened grid (--rotated-binning) follows the cloud's tilt.
Since BNT mixes all 4 channels, the BNT pair is NOT a 2-D rotation of the original pair, so this
is a genuine pairwise transport (not the trivial full rotate-back).

Two arms (jointl1, k=10, --rotated-binning), matched pipeline VMIM→10-D→RealNVP, gated:
  jointl1_nobnt_rot, jointl1_bnt_rot.  Retention = bnt_rot / nobnt_rot.
Registered: if the shear was the gap, jointl1_bnt_rot calibrates to the noBNT bar (≤PASS-caveat:
SBC ≤~0.325, TARP worst ≤0.10) — that calibrates the BNT joint ℓ1 (retention may also rise).
If still FAIL, a fixed pairwise grid can't do it ⇒ last piece needs the learned front-end (M3).
GPU 2. Resumable. Writes RESULT_JOINTL1_ROTATED.md.
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
NDE = ["--nde-family", "sbilens_realnvp", "--nde-layers", "4", "--nde-hidden", "128"]
ARMS = {"jointl1_nobnt_rot": "nobnt", "jointl1_bnt_rot": "bnt"}


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


def run_arm(name, basis):
    print(f"\n=== arm {name} ===", flush=True)
    raw, odir = f"{A}/{name}_raw", f"{A}/{name}"
    Path(odir).mkdir(parents=True, exist_ok=True)
    if not Path(f"{raw}/cache/l1_train.npz").exists():
        print("  build (rotated) ...", flush=True)
        if sh([PY, f"{SBI}/build_flatsky_joint_arm.py", "--stat", "jointl1", "--basis", basis,
               "--k", "10", "--rotated-binning", "--out-cache", f"{raw}/cache",
               "--out-fid", f"{raw}/fid.npz"], f"{LOGD}/jl1rot_{name}_build.log") != 0:
            print(f"  BUILD FAILED {name}"); return None
    cc, cf = f"{odir}/cache", f"{odir}/fiducial_summaries.npz"
    if not Path(f"{cc}/l1_train.npz").exists():
        print("  vmim ...", flush=True)
        sh([PY, f"{SBI}/vmim_from_cache.py", "--cache-dir", f"{raw}/cache", "--fid-npz", f"{raw}/fid.npz",
            "--out-cache", cc, "--out-fid", cf, "--summary-dim", "10", "--seed", "41",
            "--cuda-visible-devices", GPU], f"{LOGD}/jl1rot_{name}_vmim.log")
    for tag, n in (("screen", "1000"), ("n9000", "9000")):
        out = f"{odir}/{tag}"
        if not Path(f"{out}/median_summary.json").exists():
            print(f"  train {tag} ...", flush=True)
            sh([PY, f"{SBI}/train_nde_from_compressed.py", "--train-cache-dir", cc, "--cache-prefix", "l1",
                "--arm-label", f"{name}_{tag}", "--fiducial-summaries-npz", cf, "--output-dir", out, *NDE,
                "--n-obs", n, "--seeds", SEEDS, "--flow-total-steps", STEPS, "--cuda-visible-devices", GPU],
               f"{LOGD}/jl1rot_{name}_{tag}.log")
    G = f"{odir}/gate"
    if not Path(f"{G}/verdict.json").exists():
        print("  gate ...", flush=True)
        sh([PY, f"{SBI}/tarp_stratified_val_nde.py", "--train-cache-dir", cc, "--cache-prefix", "l1",
            "--arm-label", name, "--dumps-root", f"{G}/tarp_drp/dumps", *NDE, "--n-points", "600",
            "--seeds", SEEDS, "--flow-total-steps", STEPS, "--cuda-visible-devices", GPU],
           f"{LOGD}/jl1rot_{name}_gate.log")
        sh([PY, f"{SBI}/run_tarp_coverage.py", "--dumps-root", f"{G}/tarp_drp/dumps",
            "--outdir", f"{G}/tarp_drp", "--dims", "3"], f"{LOGD}/jl1rot_{name}_cov.log")
        sh([PY, f"{A}/gate_verdict.py", "--gate-dir", G, "--arms", name,
            "--json-out", f"{G}/verdict.json"], f"{LOGD}/jl1rot_{name}_verdict.log")
    v = None
    try:
        v = json.load(open(f"{G}/verdict.json"))[0]
    except Exception:
        pass
    return dict(name=name, fom3_n9000=fom3(f"{odir}/n9000"),
                fom3_n1000=fom3(f"{odir}/screen"), verdict=v)


def main():
    Path(LOGD).mkdir(parents=True, exist_ok=True)
    res = {n: run_arm(n, b) for n, b in ARMS.items()}
    nob, bn = res["jointl1_nobnt_rot"], res["jointl1_bnt_rot"]
    ratio = (bn["fom3_n9000"] / nob["fom3_n9000"]
             if nob and bn and nob["fom3_n9000"] and bn["fom3_n9000"] else None)

    def vinfo(r):
        v = r.get("verdict") if r else None
        if not v:
            return "?", "?", "?"
        return v["verdict"], f"{max(abs(x) for x in v['devs'].values()):.3f}", \
            "/".join("%.3f" % s for s in v["sbc_std"])

    rows = ["# Shear-aware (rotated-grid) joint ℓ1 in BNT space — gated\n",
            "Per-(pair,scale) 2-D PCA-whitened binning (shear-aware transport). Matched pipeline.",
            "Baseline (axis-aligned adaptive-ranges): jointl1_nobnt 3754 PASS-caveat (SBC 0.31); "
            "jointl1_bnt 3232, raw ret 0.861, FAIL (SBC 0.33, dev 0.110).\n",
            "| arm | FoM3 n=9000 | gate | worst dev | SBC std |", "|---|---|---|---|---|"]
    for n in ("jointl1_nobnt_rot", "jointl1_bnt_rot"):
        r = res[n]
        vt, wd, sb = vinfo(r)
        f9 = f"{r['fom3_n9000']:.0f}" if r and r["fom3_n9000"] else "?"
        rows.append(f"| {n} | {f9} | {vt} | {wd} | {sb} |")
    rows.append("")
    if ratio is not None:
        rows.append(f"**rotated retention (n=9000) = {bn['fom3_n9000']:.0f}/{nob['fom3_n9000']:.0f} "
                    f"= {ratio:.3f}**  (axis-aligned was 0.861, FAIL)")
    open(f"{A}/RESULT_JOINTL1_ROTATED.md", "w").write("\n".join(rows) + "\n")
    print("\n" + "\n".join(rows), flush=True)


if __name__ == "__main__":
    main()
