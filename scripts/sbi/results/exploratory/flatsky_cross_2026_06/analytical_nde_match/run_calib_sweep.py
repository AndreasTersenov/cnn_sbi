#!/usr/bin/env python3
"""Calibration sweep (levers #1 capacity + #3 seed-ensemble): turn the l1+product PASS-with-caveat
into a clean PASS WITHOUT losing the CNN-level FoM3.

Baseline: l1+product VMIM(s41) -> sbi_lens RealNVP 4x128, 2-3 seeds = FoM3 3270, PASS-with-caveat
(net ~-0.02, SBC std ~0.30 = mildly over-confident). The MAF/RealNVP bracket says perfect calibration
sits just on the conservative side, so LOWER flow capacity and MORE pooled seeds should both push there.

For each RealNVP (layers,hidden) config, with a 5-seed ensemble (41-45):
  screen  -> population-median FoM3 (n=1000)              [train_nde_from_compressed.py]
  GATE C  -> TARP-DRP (600 val pts) + SBC, pooled 5 seeds [tarp_stratified_val_nde.py -> coverage -> verdict]
Distributes configs across the least-loaded of GPUs 0/1/2 (GPU 3 NEVER). Resumable (skips configs whose
verdict.json already exists). Writes calib_sweep/SWEEP_RESULT.md.
"""
import json
import os
import subprocess
import threading
from pathlib import Path

SBI = "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi"
A = f"{SBI}/results/exploratory/flatsky_cross_2026_06/analytical_nde_match"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
CACHE = f"{A}/l1product_vmim_s41/cache"
FID = f"{A}/l1product_vmim_s41/fiducial_summaries.npz"
SEEDS = "41,42,43,44,45"                       # lever #3: 5-seed ensemble (was 2-3)
CONFIGS = [(4, 128), (3, 128), (4, 64), (3, 64)]   # lever #1: RealNVP capacity grid
MEM = "0.45"
LOGD = f"{A}/logs"


def sh(cmd, log, gpu):
    env = dict(os.environ, XLA_PYTHON_CLIENT_MEM_FRACTION=MEM)
    with open(log, "w") as f:
        return subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def pick_gpus():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"]).decode()
        load = {}
        for line in out.strip().splitlines():
            i, m, u = [x.strip() for x in line.split(",")]
            if int(i) in (0, 1, 2):
                load[int(i)] = (int(m), int(u))
        order = sorted(load, key=lambda g: load[g][0] + load[g][1] * 100)
        print(f"[gpu] load(mem,util)={load} -> using {order}", flush=True)
        return order or [1, 2]
    except Exception as e:
        print(f"[gpu] nvidia-smi failed ({e}); default [0,1,2]", flush=True)
        return [0, 1, 2]


def run_config(layers, hidden, gpu):
    tag = f"rnvp{layers}x{hidden}_ens5"
    odir = f"{A}/calib_sweep/{tag}"
    G = f"{A}/calib_sweep/gate_{tag}"
    Path(odir).mkdir(parents=True, exist_ok=True)
    if Path(f"{G}/verdict.json").exists() and Path(f"{odir}/median_summary.json").exists():
        print(f"[skip] {tag} already done", flush=True)
    else:
        sh([PY, f"{SBI}/train_nde_from_compressed.py", "--train-cache-dir", CACHE,
            "--cache-prefix", "l1", "--arm-label", tag, "--fiducial-summaries-npz", FID,
            "--output-dir", odir, "--nde-family", "sbilens_realnvp",
            "--nde-layers", str(layers), "--nde-hidden", str(hidden), "--n-obs", "1000",
            "--seeds", SEEDS, "--flow-total-steps", "50000", "--cuda-visible-devices", str(gpu)],
           f"{LOGD}/calibsweep_{tag}_screen.log", gpu)
        sh([PY, f"{SBI}/tarp_stratified_val_nde.py", "--train-cache-dir", CACHE,
            "--cache-prefix", "l1", "--arm-label", tag, "--dumps-root", f"{G}/tarp_drp/dumps",
            "--nde-family", "sbilens_realnvp", "--nde-layers", str(layers), "--nde-hidden", str(hidden),
            "--n-points", "600", "--seeds", SEEDS, "--flow-total-steps", "50000",
            "--cuda-visible-devices", str(gpu)], f"{LOGD}/calibsweep_{tag}_gate.log", gpu)
        sh([PY, f"{SBI}/run_tarp_coverage.py", "--dumps-root", f"{G}/tarp_drp/dumps",
            "--outdir", f"{G}/tarp_drp", "--dims", "3"], f"{LOGD}/calibsweep_{tag}_cov.log", gpu)
        sh([PY, f"{A}/gate_verdict.py", "--gate-dir", G, "--arms", tag,
            "--json-out", f"{G}/verdict.json"], f"{LOGD}/calibsweep_{tag}_verdict.log", gpu)
    fom3 = None
    try:
        fom3 = json.load(open(f"{odir}/median_summary.json"))["fom3"]
    except Exception:
        pass
    verdict = None
    try:
        verdict = json.load(open(f"{G}/verdict.json"))[0]
    except Exception:
        pass
    return dict(tag=tag, layers=layers, hidden=hidden, fom3=fom3, verdict=verdict)


def main():
    Path(LOGD).mkdir(parents=True, exist_ok=True)
    Path(f"{A}/calib_sweep").mkdir(parents=True, exist_ok=True)
    gpus = pick_gpus()
    buckets = {g: [] for g in gpus}
    for i, cfg in enumerate(CONFIGS):
        buckets[gpus[i % len(gpus)]].append(cfg)
    results = {}
    lock = threading.Lock()

    def worker(gpu, cfgs):
        for (L, H) in cfgs:
            print(f"[run] {L}x{H} on GPU{gpu}", flush=True)
            r = run_config(L, H, gpu)
            with lock:
                results[r["tag"]] = r
            v = r["verdict"]["verdict"] if r["verdict"] else "?"
            print(f"[done] {r['tag']} FoM3 {r['fom3']} -> {v}", flush=True)

    threads = [threading.Thread(target=worker, args=(g, buckets[g])) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rows = ["# Calibration sweep — RealNVP capacity × 5-seed ensemble (levers #1+#3)\n",
            "Goal: clean PASS (worst dev ≤0.05 AND SBC std ∈[0.275,0.305]) at ~CNN FoM3.",
            "Baseline l1+product 4×128 / 2-3 seeds = 3270, PASS-with-caveat (net ~−0.02, SBC std ~0.30).",
            "CNN ref 3293 (PASS). Raw l1+product-MAF 2875 (PASS-clean).\n",
            "| config | seeds | FoM3 | worst dev | net bias | SBC std (Om/s8/w0) | verdict |",
            "|---|---|---|---|---|---|---|"]
    for tag in sorted(results):
        r = results[tag]
        v = r["verdict"]
        fom = f"{r['fom3']:.0f}" if r["fom3"] else "?"
        if v:
            worst = max(abs(x) for x in v["devs"].values())
            rows.append(f"| {r['layers']}×{r['hidden']} | 5 | {fom} | {worst:.3f} | "
                        f"{v['net_bias'][0]:+.3f} | {'/'.join('%.3f' % s for s in v['sbc_std'])} | "
                        f"**{v['verdict']}** |")
        else:
            rows.append(f"| {r['layers']}×{r['hidden']} | 5 | {fom} | ? | ? | ? | INCOMPLETE |")
    out = "\n".join(rows) + "\n"
    open(f"{A}/calib_sweep/SWEEP_RESULT.md", "w").write(out)
    print("\n" + out, flush=True)


if __name__ == "__main__":
    main()
