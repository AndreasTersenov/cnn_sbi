#!/usr/bin/env python3
"""Joint ℓ1 / full4d through the MATCHED best-NDE pipeline — two questions, gated.

Thesis (Andreas): the wavelet JOINT ℓ1 (histogram of the across-channel coefficient vector)
is the COMPLETE cross-correlation statistic; products κ_iκ_j are only its 2nd-moment slice.
So it should (Q1) beat ℓ1+product on FoM3 IF calibrated, and (Q2) be BNT-lossless (full4d is
exactly basis-covariant, P4b) — the BNT collapse of ℓ1+product (0.26×) being the direct
evidence products miss most of the cross-correlation.

Standing prior (LANE_A_CONCLUSION): every joint-PDF FoM3 "gain" so far was estimation-path
OVER-CONFIDENCE (pair2d→RealNVP 4864 GATE FAIL; A1→MAF calibration FAIL; K-trend FoM3 DROPS
as grid refines). The matched 10-D pipeline + jointl1 (ℓ1-weighted, not counts) + full4d are
the UNTESTED variables. GATE is the arbiter; calibrated-gain-over-3045 is the bar; nothing
ungated is quoted.

Pipeline (every arm): build datavector → VMIM-MLP 10-D → sbi_lens RealNVP 4x128 (seeds
41,42,43) → n=9000 median FoM3 → GATE C (TARP-DRP 600 + SBC). All arms ADAPTIVE-RANGES
(transported binning = the P4c BNT-covariance mechanism). full4d K=5 + dequantize (counts,
sparse 4-D ⇒ coarse K well-populated); jointl1 K=10 (continuous ℓ1 cells).

Reference points (reused, not recomputed): ℓ1+product noBNT 3045 / BNT 779 (0.26×);
pair2d→RealNVP 4864 GATE FAIL (the cautionary baseline).
GPU 2 only. Resumable. Writes RESULT_JOINT_MATCHED.md.
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

# stat, basis, k, dequantize ; all adaptive-ranges
ARMS = {
    "jointl1_nobnt": dict(stat="jointl1", basis="nobnt", k="10", dequant=False),
    "jointl1_bnt":   dict(stat="jointl1", basis="bnt",   k="10", dequant=False),
    "full4d_nobnt":  dict(stat="full4d",  basis="nobnt", k="5",  dequant=True),
    "full4d_bnt":    dict(stat="full4d",  basis="bnt",   k="5",  dequant=True),
}


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


def run_arm(name, cfg):
    print(f"\n=== arm {name} ===", flush=True)
    raw = f"{A}/{name}_raw"
    odir = f"{A}/{name}"
    Path(odir).mkdir(parents=True, exist_ok=True)

    # 0. build datavector
    if not Path(f"{raw}/cache/l1_train.npz").exists():
        print("  build datavector ...", flush=True)
        cmd = [PY, f"{SBI}/build_flatsky_joint_arm.py", "--stat", cfg["stat"],
               "--basis", cfg["basis"], "--k", cfg["k"], "--adaptive-ranges",
               "--out-cache", f"{raw}/cache", "--out-fid", f"{raw}/fid.npz"]
        if cfg["dequant"]:
            cmd.append("--dequantize")
        if sh(cmd, f"{LOGD}/joint_{name}_build.log") != 0:
            print(f"  BUILD FAILED {name}"); return None

    # 1. VMIM compress -> 10-D
    cc, cf = f"{odir}/cache", f"{odir}/fiducial_summaries.npz"
    if not Path(f"{cc}/l1_train.npz").exists():
        print("  vmim compress ...", flush=True)
        sh([PY, f"{SBI}/vmim_from_cache.py", "--cache-dir", f"{raw}/cache",
            "--fid-npz", f"{raw}/fid.npz", "--out-cache", cc, "--out-fid", cf,
            "--summary-dim", "10", "--seed", "41", "--cuda-visible-devices", GPU],
           f"{LOGD}/joint_{name}_vmim.log")

    # 2. screen n=1000, 3. final n=9000
    for tag, n in (("screen", "1000"), ("n9000", "9000")):
        out = f"{odir}/{tag}"
        if not Path(f"{out}/median_summary.json").exists():
            print(f"  train {tag} ...", flush=True)
            sh([PY, f"{SBI}/train_nde_from_compressed.py", "--train-cache-dir", cc,
                "--cache-prefix", "l1", "--arm-label", f"{name}_{tag}",
                "--fiducial-summaries-npz", cf, "--output-dir", out, *NDE,
                "--n-obs", n, "--seeds", SEEDS, "--flow-total-steps", STEPS,
                "--cuda-visible-devices", GPU], f"{LOGD}/joint_{name}_{tag}.log")

    # 4. GATE C
    G = f"{odir}/gate"
    if not Path(f"{G}/verdict.json").exists():
        print("  gate ...", flush=True)
        sh([PY, f"{SBI}/tarp_stratified_val_nde.py", "--train-cache-dir", cc,
            "--cache-prefix", "l1", "--arm-label", name, "--dumps-root", f"{G}/tarp_drp/dumps",
            *NDE, "--n-points", "600", "--seeds", SEEDS, "--flow-total-steps", STEPS,
            "--cuda-visible-devices", GPU], f"{LOGD}/joint_{name}_gate.log")
        sh([PY, f"{SBI}/run_tarp_coverage.py", "--dumps-root", f"{G}/tarp_drp/dumps",
            "--outdir", f"{G}/tarp_drp", "--dims", "3"], f"{LOGD}/joint_{name}_cov.log")
        sh([PY, f"{A}/gate_verdict.py", "--gate-dir", G, "--arms", name,
            "--json-out", f"{G}/verdict.json"], f"{LOGD}/joint_{name}_verdict.log")
    v = None
    try:
        v = json.load(open(f"{G}/verdict.json"))[0]
    except Exception:
        pass
    return dict(name=name, fom3_n1000=fom3(f"{odir}/screen"),
                fom3_n9000=fom3(f"{odir}/n9000"), verdict=v)


def main():
    Path(LOGD).mkdir(parents=True, exist_ok=True)
    res = {}
    for name, cfg in ARMS.items():
        res[name] = run_arm(name, cfg)

    def f9(n):
        r = res.get(n) or {}
        return r.get("fom3_n9000")

    def vt(n):
        r = res.get(n) or {}
        v = r.get("verdict")
        return v["verdict"] if v else "?"

    rows = ["# Joint ℓ1 / full4d through the matched best-NDE pipeline — gated\n",
            "Pipeline: build → VMIM 10-D → sbi_lens RealNVP 4x128 (seeds 41,42,43) → n=9000 median; "
            "GATE C = TARP-DRP 600 + SBC. All arms adaptive-ranges (transported binning).",
            "Reference: ℓ1+product noBNT 3045 / BNT 779 (0.26×); pair2d→RealNVP 4864 GATE FAIL (cautionary).\n",
            "## Q1 — better statistic? (noBNT FoM3 vs ℓ1+product 3045; PASS-gate required)",
            "| arm | FoM3 n=1000 | FoM3 n=9000 | gate |", "|---|---|---|---|"]
    for n in ("jointl1_nobnt", "full4d_nobnt"):
        r = res.get(n) or {}
        f1 = f"{r.get('fom3_n1000'):.0f}" if r.get("fom3_n1000") else "?"
        f9v = f"{r.get('fom3_n9000'):.0f}" if r.get("fom3_n9000") else "?"
        rows.append(f"| {n} | {f1} | {f9v} | {vt(n)} |")
    rows += ["", "## Q2 — BNT-lossless? (BNT/noBNT ratio vs ℓ1+product's 0.26×)",
             "| statistic | noBNT | BNT | BNT/noBNT | BNT gate |", "|---|---|---|---|---|"]
    for stat, nob, bn in (("jointl1", "jointl1_nobnt", "jointl1_bnt"),
                          ("full4d", "full4d_nobnt", "full4d_bnt")):
        a9, b9 = f9(nob), f9(bn)
        ratio = f"{b9/a9:.3f}" if (a9 and b9) else "?"
        a9s = f"{a9:.0f}" if a9 else "?"
        b9s = f"{b9:.0f}" if b9 else "?"
        rows.append(f"| {stat} | {a9s} | {b9s} | {ratio} | {vt(bn)} |")
    out = "\n".join(rows) + "\n"
    open(f"{A}/RESULT_JOINT_MATCHED.md", "w").write(out)
    print("\n" + out, flush=True)


if __name__ == "__main__":
    main()
