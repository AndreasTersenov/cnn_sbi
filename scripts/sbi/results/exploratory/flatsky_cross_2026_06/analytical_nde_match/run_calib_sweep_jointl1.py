#!/usr/bin/env python3
"""Calibration sweep on the Q1 winner jointl1_nobnt: can we shave the mild marginal over-confidence
(pooled SBC std ~0.313/0.316/0.304, vs uniform 0.289; l1+product ~0.303, CNN ~0.289) toward
ideal WITHOUT losing the FoM3?

Lever (the one that centered l1+product): LOWER RealNVP capacity + LARGER NDE seed-ensemble
(more pooled flows → less per-flow over-confidence → wider marginals). Same 10-D jointl1 summary;
only the downstream flow capacity + ensemble size change.

For each (layers,hidden) with a 5-seed ensemble (41-45):
  screen  -> population-median FoM3 (n=1000)              [train_nde_from_compressed.py]
  GATE C  -> TARP-DRP (600 val pts) + SBC, pooled 5 seeds [tarp_stratified_val_nde -> coverage -> verdict]
GPU 2 only (0/1 have active foreign tenants today). Resumable. Writes calib_sweep_jointl1/SWEEP_RESULT.md.
"""
import json
import os
import subprocess
from pathlib import Path

SBI = "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi"
A = f"{SBI}/results/exploratory/flatsky_cross_2026_06/analytical_nde_match"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
CACHE = f"{A}/jointl1_nobnt/cache"
FID = f"{A}/jointl1_nobnt/fiducial_summaries.npz"
SEEDS = "41,42,43,44,45"                              # 5-seed ensemble (was 3)
CONFIGS = [(4, 128), (3, 128), (4, 64), (3, 64), (2, 64)]   # capacity grid (low = less over-fit)
GPU = "2"
MEM = "0.8"
LOGD = f"{A}/logs"


def sh(cmd, log):
    env = dict(os.environ, XLA_PYTHON_CLIENT_MEM_FRACTION=MEM, CUDA_VISIBLE_DEVICES=GPU)
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        return subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def run_config(layers, hidden):
    tag = f"jointl1_rnvp{layers}x{hidden}_ens5"
    odir = f"{A}/calib_sweep_jointl1/{tag}"
    G = f"{A}/calib_sweep_jointl1/gate_{tag}"
    Path(odir).mkdir(parents=True, exist_ok=True)
    if not (Path(f"{G}/verdict.json").exists() and Path(f"{odir}/median_summary.json").exists()):
        print(f"[run] {layers}x{hidden}", flush=True)
        sh([PY, f"{SBI}/train_nde_from_compressed.py", "--train-cache-dir", CACHE,
            "--cache-prefix", "l1", "--arm-label", tag, "--fiducial-summaries-npz", FID,
            "--output-dir", odir, "--nde-family", "sbilens_realnvp",
            "--nde-layers", str(layers), "--nde-hidden", str(hidden), "--n-obs", "1000",
            "--seeds", SEEDS, "--flow-total-steps", "50000", "--cuda-visible-devices", GPU],
           f"{LOGD}/jl1calib_{tag}_screen.log")
        sh([PY, f"{SBI}/tarp_stratified_val_nde.py", "--train-cache-dir", CACHE,
            "--cache-prefix", "l1", "--arm-label", tag, "--dumps-root", f"{G}/tarp_drp/dumps",
            "--nde-family", "sbilens_realnvp", "--nde-layers", str(layers), "--nde-hidden", str(hidden),
            "--n-points", "600", "--seeds", SEEDS, "--flow-total-steps", "50000",
            "--cuda-visible-devices", GPU], f"{LOGD}/jl1calib_{tag}_gate.log")
        sh([PY, f"{SBI}/run_tarp_coverage.py", "--dumps-root", f"{G}/tarp_drp/dumps",
            "--outdir", f"{G}/tarp_drp", "--dims", "3"], f"{LOGD}/jl1calib_{tag}_cov.log")
        sh([PY, f"{A}/gate_verdict.py", "--gate-dir", G, "--arms", tag,
            "--json-out", f"{G}/verdict.json"], f"{LOGD}/jl1calib_{tag}_verdict.log")
    fom3 = None
    try:
        fom3 = json.load(open(f"{odir}/median_summary.json"))["fom3"]
    except Exception:
        pass
    v = None
    try:
        v = json.load(open(f"{G}/verdict.json"))[0]
    except Exception:
        pass
    return dict(layers=layers, hidden=hidden, fom3=fom3, verdict=v)


def main():
    Path(LOGD).mkdir(parents=True, exist_ok=True)
    Path(f"{A}/calib_sweep_jointl1").mkdir(parents=True, exist_ok=True)
    rows = ["# jointl1 calibration sweep — RealNVP capacity × 5-seed ensemble\n",
            "Goal: shave marginal over-confidence (SBC std ~0.31) toward ideal (0.289) without losing FoM3.",
            "Baseline jointl1 4×128 / 3-seed = FoM3 3754, pooled SBC 0.313/0.316/0.304, TARP net -0.003.",
            "Reference: l1+product 3045 (SBC ~0.30), CNN 3326 (SBC ~0.29).\n",
            "| config | seeds | FoM3 | worst dev | net bias | SBC std (Om/s8/w0) | verdict |",
            "|---|---|---|---|---|---|---|"]
    for (L, H) in CONFIGS:
        r = run_config(L, H)
        v = r["verdict"]
        fom = f"{r['fom3']:.0f}" if r["fom3"] else "?"
        if v:
            worst = max(abs(x) for x in v["devs"].values())
            rows.append(f"| {L}×{H} | 5 | {fom} | {worst:.3f} | {v['net_bias'][0]:+.3f} | "
                        f"{'/'.join('%.3f' % s for s in v['sbc_std'])} | **{v['verdict']}** |")
            print(f"[done] {L}x{H} FoM3 {fom} SBC {['%.3f'%s for s in v['sbc_std']]} -> {v['verdict']}", flush=True)
        else:
            rows.append(f"| {L}×{H} | 5 | {fom} | ? | ? | ? | INCOMPLETE |")
    out = "\n".join(rows) + "\n"
    open(f"{A}/calib_sweep_jointl1/SWEEP_RESULT.md", "w").write(out)
    print("\n" + out, flush=True)


if __name__ == "__main__":
    main()
