#!/usr/bin/env python3
"""Compressor-ENSEMBLE calibration of the BNT joint ℓ1 — the non-conformal fix.

Probe (noBNT) showed pooling 3 compressor-seed posteriors brings SBC 0.31 -> 0.298 (clean PASS):
the over-confidence is the single-compressor amortization artifact, washed out by a deep ensemble.
Test whether the SAME cleans BNT (where the single arm FAILed: SBC 0.34, dev 0.110).

Build BNT compressor seeds 42,43 (seed 41 = existing jointl1_bnt) on jointl1_bnt_raw/cache:
vmim(seed) -> RealNVP n9000 (FoM3) + GATE dumps. Then pool the 3 arms' posteriors per obs
(true compressor-ensemble) and recompute pooled TARP + SBC. Registered: if BNT over-confidence is
the same artifact, the 3-compressor ensemble -> SBC ~0.30, clean/caveat PASS = a CALIBRATED BNT
joint ℓ1 at ~0.86 retention (non-conformal). If it stays >0.325, the residual is the irreducible
4-D shear (the CNN's learned front-end).
GPU 2. Resumable. Writes RESULT_JOINTL1_BNT_ENSEMBLE.md.
"""
import glob
import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
SBI = "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi"
A = f"{SBI}/results/exploratory/flatsky_cross_2026_06/analytical_nde_match"
LOGD = f"{A}/logs"
GPU = "2"
MEM = "0.8"
RAW = f"{A}/jointl1_bnt_raw"
NDE = ["--nde-family", "sbilens_realnvp", "--nde-layers", "4", "--nde-hidden", "128"]
# arm dir, gate-dump arm-label  (seed 41 = the existing jointl1_bnt)
ARMS = {"41": ("jointl1_bnt", "jointl1_bnt"),
        "42": ("jointl1_bnt_s42", "jointl1_bnt_s42"),
        "43": ("jointl1_bnt_s43", "jointl1_bnt_s43")}


def sh(cmd, log):
    env = dict(os.environ, XLA_PYTHON_CLIENT_MEM_FRACTION=MEM, CUDA_VISIBLE_DEVICES=GPU)
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"  [{(time.time()-t0)/60:.1f} min, rc={rc}] {os.path.basename(log)}", flush=True)
    return rc


def build_seed(s):
    print(f"\n=== BNT compressor seed {s} ===", flush=True)
    odir = f"{A}/jointl1_bnt_s{s}"
    cc, cf = f"{odir}/cache", f"{odir}/fiducial_summaries.npz"
    if not Path(f"{cc}/l1_train.npz").exists():
        sh([PY, f"{SBI}/vmim_from_cache.py", "--cache-dir", f"{RAW}/cache", "--fid-npz", f"{RAW}/fid.npz",
            "--out-cache", cc, "--out-fid", cf, "--summary-dim", "10", "--seed", s,
            "--cuda-visible-devices", GPU], f"{LOGD}/jl1bntens_s{s}_vmim.log")
    if not Path(f"{odir}/n9000/median_summary.json").exists():
        sh([PY, f"{SBI}/train_nde_from_compressed.py", "--train-cache-dir", cc, "--cache-prefix", "l1",
            "--arm-label", f"jointl1_bnt_s{s}", "--fiducial-summaries-npz", cf,
            "--output-dir", f"{odir}/n9000", *NDE, "--n-obs", "9000", "--seeds", "41,42,43",
            "--flow-total-steps", "50000", "--cuda-visible-devices", GPU], f"{LOGD}/jl1bntens_s{s}_n9000.log")
    G = f"{odir}/gate"
    if not Path(f"{G}/tarp_drp/dumps").exists() or not list(Path(f"{G}/tarp_drp/dumps").glob("*/seed_*/*/posterior_samples.npz")):
        sh([PY, f"{SBI}/tarp_stratified_val_nde.py", "--train-cache-dir", cc, "--cache-prefix", "l1",
            "--arm-label", f"jointl1_bnt_s{s}", "--dumps-root", f"{G}/tarp_drp/dumps", *NDE,
            "--n-points", "600", "--seeds", "41,42,43", "--flow-total-steps", "50000",
            "--cuda-visible-devices", GPU], f"{LOGD}/jl1bntens_s{s}_gate.log")


def fom3(d):
    try:
        return json.load(open(f"{d}/n9000/median_summary.json"))["fom3"]
    except Exception:
        return None


def _key(f):
    m = re.search(r"dumps/(.+?)_(LOW|MID|HIGH)/seed_(\d+)/", f)
    return (m.group(2), m.group(3)) if m else None


def load_dumps(armdir):
    d = {}
    for f in glob.glob(f"{A}/{armdir}/gate/tarp_drp/dumps/*/seed_*/n*_m*/posterior_samples.npz"):
        k = _key(f)
        if k:
            z = np.load(f)
            d[k] = (np.asarray(z["samples"], np.float32)[:, :, :3], np.asarray(z["theta"], np.float32)[:, :3])
    return d


def calib(sampler, keys):
    import tarp
    S, T = [], []
    for k in keys:
        s, t = sampler(k)
        S.append(s); T.append(t)
    s = np.concatenate(S, 0); t = np.concatenate(T, 0)
    ecp, alpha = tarp.get_tarp_coverage(np.transpose(s, (1, 0, 2)), t, references="random",
                                        num_bootstrap=100, norm=True, bootstrap=True)
    m = ecp.mean(0)
    worst = float(m[np.argmax(np.abs(m - alpha))] - alpha[np.argmax(np.abs(m - alpha))])
    net = float(np.trapz(m - alpha, alpha) * 2)
    std = (s < t[:, None, :]).mean(1).std(0)
    return worst, net, std


def main():
    Path(LOGD).mkdir(parents=True, exist_ok=True)
    for s in ("42", "43"):
        build_seed(s)

    D = {s: load_dumps(ARMS[s][0]) for s in ARMS}
    keys = sorted(set(D["41"]) & set(D["42"]) & set(D["43"]))
    w1, n1, std1 = calib(lambda k: D["41"][k], keys)                       # single seed 41

    def ens(k):
        return np.concatenate([D[s][k][0] for s in ("41", "42", "43")], axis=1), D["41"][k][1]
    we, ne, stde = calib(ens, keys)                                        # 3-compressor ensemble

    foms = [fom3(f"{A}/{ARMS[s][0]}") for s in ARMS]
    foms = [f for f in foms if f]
    rows = ["# BNT joint ℓ1 — compressor-ENSEMBLE calibration (non-conformal)\n",
            "Single BNT arm FAILed (SBC 0.333/0.335/0.313, dev 0.110). noBNT probe: ensemble 0.31->0.298 (clean).",
            f"BNT per-seed FoM3 (n=9000): {'/'.join('%.0f' % f for f in foms)} (median {int(np.median(foms))}).",
            f"noBNT single 3754 / ensemble retention ref.\n",
            "| arm | TARP worst | net | SBC std (Om/s8/w0) |", "|---|---|---|---|",
            f"| single (s41) | {w1:+.3f} | {n1:+.3f} | {std1[0]:.3f}/{std1[1]:.3f}/{std1[2]:.3f} |",
            f"| **compressor-ensemble (x3)** | {we:+.3f} | {ne:+.3f} | **{stde[0]:.3f}/{stde[1]:.3f}/{stde[2]:.3f}** |",
            "",
            f"Verdict: ensemble SBC max {max(stde):.3f} (band <=0.305; FAIL >=0.325) -> "
            f"{'CALIBRATED (clean/caveat PASS) — BNT joint l1 calibrated, non-conformal' if max(stde) < 0.325 else 'still over-confident -> residual 4-D shear (learned front-end)'}"]
    open(f"{A}/RESULT_JOINTL1_BNT_ENSEMBLE.md", "w").write("\n".join(rows) + "\n")
    print("\n" + "\n".join(rows), flush=True)


if __name__ == "__main__":
    main()
