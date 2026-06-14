#!/usr/bin/env python3
"""Phase-1 NDE-family fan-out orchestrator (CNN-optimization 2026-06).

Runs train_nde_from_compressed.py over the NDE matrix on the FROZEN seed-41 auto-only summaries,
GPU 2 only, with a concurrency cap (polite co-residency on a shared card). Each arm writes
median_summary.json; at the end this writes SUMMARY.md ranked by FoM3.

  jaxili_maf_baseline : FULL config (3 seeds, 9000 obs) — harness self-test + baseline (~2312).
  A0..A3              : sbi_lens RealNVP capacity ladder (2-seed screen, 1000 obs).
  B2                  : jaxili RealNVP (framework control). B3: jaxili MDN (different family).

Screen first; promote winners to FULL (3 seeds / 9000 obs) afterwards. See
PLAN_CNN_NDE_SWEEP_2026-06-13.md.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import time
from pathlib import Path

REPO = "/mnt/home/tersenov/software/cnn_sbi"
SBI = f"{REPO}/scripts/sbi"
CNN = f"{SBI}/results/exploratory/flatsky_cross_2026_06/cnn_phase"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
OUT = f"{CNN}/nde_sweep_2026_06_13"
CACHE = f"{CNN}/cnn_none_s41/cache"
FIDSUMM = f"{CNN}/fiducial_summaries/fiducial_summaries_none.npz"

# (label, family, layers/transforms, hidden, seeds, n_obs)
ARMS = [
    ("jaxili_maf_baseline", "jaxili_maf",      5,  50, "41,42,43", 9000),  # self-test -> ~2312
    ("A0_sbilens_rnvp_4x128", "sbilens_realnvp", 4, 128, "41,42", 1000),    # production config
    ("A1_sbilens_rnvp_6x256", "sbilens_realnvp", 6, 256, "41,42", 1000),
    ("A2_sbilens_rnvp_8x256", "sbilens_realnvp", 8, 256, "41,42", 1000),
    ("A3_sbilens_rnvp_8x512", "sbilens_realnvp", 8, 512, "41,42", 1000),
    ("B2_jaxili_rnvp_5x50", "jaxili_realnvp",  5,  50, "41,42", 1000),
    ("B3_jaxili_mdn_10x50", "jaxili_mdn",     10,  50, "41,42", 1000),
]


def cmd(arm, gpu, mem_frac):
    label, family, layers, hidden, seeds, n_obs = arm
    return ([PY, "-u", f"{SBI}/train_nde_from_compressed.py",
             "--train-cache-dir", CACHE, "--cache-prefix", "cnn",
             "--fiducial-summaries-npz", FIDSUMM,
             "--arm-label", label, "--output-dir", f"{OUT}/{label}",
             "--nde-family", family, "--nde-layers", str(layers), "--nde-hidden", str(hidden),
             "--preproc-transform", "none", "--clip-value", "0", "--min-feature-variance", "1e-12",
             "--seeds", seeds, "--n-obs", str(n_obs), "--m-samples", "2000",
             "--cuda-visible-devices", str(gpu)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--mem-fraction", type=float, default=0.18)
    ap.add_argument("--only", default="", help="comma-separated arm labels to run (default all)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    Path(OUT).mkdir(parents=True, exist_ok=True)
    arms = [a for a in ARMS if not args.only or a[0] in args.only.split(",")]
    if args.dry_run:
        for a in arms:
            print(" ".join(cmd(a, args.gpu, args.mem_fraction)))
        return

    import os
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_PYTHON_CLIENT_MEM_FRACTION=str(args.mem_fraction), PYTHONUNBUFFERED="1")
    pending = list(arms)
    running = []   # (label, Popen, logfile)
    t0 = time.time()

    def launch(arm):
        label = arm[0]
        Path(f"{OUT}/{label}").mkdir(parents=True, exist_ok=True)
        lg = open(f"{OUT}/{label}/run.log", "w")
        p = subprocess.Popen(cmd(arm, args.gpu, args.mem_fraction), cwd=SBI, env=env,
                             stdout=lg, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        print(f"[{time.time()-t0:6.0f}s] LAUNCH {label} (pid {p.pid})", flush=True)
        return (label, p, lg)

    while pending or running:
        while pending and len(running) < args.concurrency:
            running.append(launch(pending.pop(0)))
        time.sleep(15)
        still = []
        for label, p, lg in running:
            if p.poll() is None:
                still.append((label, p, lg))
            else:
                lg.close()
                fom = "?"
                try:
                    fom = f"{json.load(open(f'{OUT}/{label}/median_summary.json'))['fom3']:.0f}"
                except Exception:
                    pass
                print(f"[{time.time()-t0:6.0f}s] DONE   {label} rc={p.returncode} FoM3={fom}", flush=True)
        running = still

    # SUMMARY.md ranked by FoM3
    rows = []
    for label, *_ in [(a[0],) for a in arms]:
        f = Path(f"{OUT}/{label}/median_summary.json")
        if f.exists():
            d = json.load(open(f))
            rows.append((d.get("fom3", float("nan")), label, d))
    rows.sort(key=lambda r: (-(r[0] if r[0] == r[0] else -1)))
    lines = ["# NDE-sweep SUMMARY (Phase 1)\n",
             "Frozen seed-41 auto-only summaries. Bar: L1+product 2875; jaxili-MAF baseline ~2312.\n",
             "| rank | arm | family | layers | hidden | n_obs | seeds | FoM3 | σ(Ωm,σ8,w0) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, (fom, label, d) in enumerate(rows, 1):
        lines.append(f"| {i} | {label} | {d.get('nde_family')} | {d.get('layers')} | "
                     f"{d.get('hidden')} | {d.get('n')} | — | {fom:.0f} | "
                     f"{d.get('sigma_Om'):.3f},{d.get('sigma_s8'):.3f},{d.get('sigma_w0'):.3f} |")
    Path(f"{OUT}/SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"\n[{time.time()-t0:.0f}s] ALL DONE -> {OUT}/SUMMARY.md", flush=True)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
