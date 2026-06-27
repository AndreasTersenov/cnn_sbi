#!/usr/bin/env python3
"""Compressor-ENSEMBLE calibration of the BNT l1 AUTO and +PRODUCT arms.

The single-compressor BNT auto/product arms are over-confident under SBC (auto crosses the
simultaneous band in all 3 params; +product in s8). The deep-ensemble fix calibrated joint l1
(single 0.31->ensemble 0.298 noBNT; 0.333->0.304 BNT) and a free no-BNT-product probe showed the
same (sup 0.054->0.026). Build BNT compressor seeds 42,43 on the SAME raw datavectors the s41 arms
used (bnt_campaign/l1_matrix), then pool the 3 compressor-seed posteriors and recompute SBC.

Recipe per (arm, seed): vmim_from_cache (raw -> 10-D) -> tarp_stratified_val_nde (gate dumps).
seed 41 = the existing single arm. Pooling = tercile-key (validated ~ per-obs to <=0.01 std, since
FoM3-stratification tracks cosmology). Resumable. GPU-arg. Writes RESULT_BNT_AUTOPROD_ENSEMBLE.md.

Usage:
  run_bnt_autoprod_ensemble.py --arms product --seeds 42 --gpu 0      # pilot one
  run_bnt_autoprod_ensemble.py --arms product,auto --seeds 42,43 --gpu 0
  run_bnt_autoprod_ensemble.py --pool-only                            # just recompute SBC
"""
import argparse
import glob
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
SBI = "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi"
BASE = f"{SBI}/results/exploratory/flatsky_cross_2026_06"
A = f"{BASE}/analytical_nde_match"
BC = f"{BASE}/bnt_campaign"
LOGD = f"{A}/logs"
MEM = "0.8"
NDE = ["--nde-family", "sbilens_realnvp", "--nde-layers", "4", "--nde-hidden", "128"]

# arm -> raw datavector cache, raw fiducial, the existing s41 gate-dump glob
ARMS = {
    "product": dict(
        raw_cache=f"{BC}/l1_matrix/l1_product_cache/flat_local_product_bnt",
        raw_fid=f"{BC}/fiducial_summaries/fiducial_summaries_l1_product.npz",
        s41_glob=f"{A}/gate_l1product_bnt_rnvp/tarp_drp/dumps/*/seed_*/n*_m*/posterior_samples.npz"),
    "auto": dict(
        raw_cache=f"{BC}/l1_matrix/l1_none_cache/flat_local_none_bnt",
        raw_fid=f"{BC}/fiducial_summaries/fiducial_summaries_l1_none.npz",
        s41_glob=f"{A}/bnt_auto/gate/tarp_drp/dumps/*/seed_*/n*_m*/posterior_samples.npz"),
}
G = np.linspace(0, 1, 400)


def sh(cmd, log, gpu):
    env = dict(os.environ, XLA_PYTHON_CLIENT_MEM_FRACTION=MEM, CUDA_VISIBLE_DEVICES=gpu)
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"  [{(time.time()-t0)/60:.1f} min, rc={rc}] {os.path.basename(log)}", flush=True)
    return rc


def build(arm, s, gpu):
    cfg = ARMS[arm]
    odir = f"{A}/ens_bnt_{arm}_s{s}"
    cc, cf = f"{odir}/cache", f"{odir}/fiducial_summaries.npz"
    if not Path(f"{cc}/l1_train.npz").exists():
        sh([PY, f"{SBI}/vmim_from_cache.py", "--cache-dir", cfg["raw_cache"], "--fid-npz", cfg["raw_fid"],
            "--out-cache", cc, "--out-fid", cf, "--summary-dim", "10", "--seed", s,
            "--cuda-visible-devices", gpu], f"{LOGD}/ensbnt_{arm}_s{s}_vmim.log", gpu)
    dumps = f"{odir}/gate/tarp_drp/dumps"
    have_dumps = Path(dumps).exists() and list(Path(dumps).glob("*/seed_*/*/posterior_samples.npz"))
    if not have_dumps:
        sh([PY, f"{SBI}/tarp_stratified_val_nde.py", "--train-cache-dir", cc, "--cache-prefix", "l1",
            "--arm-label", f"{arm}_bnt_s{s}", "--dumps-root", dumps, *NDE, "--n-points", "600",
            "--seeds", "41,42,43", "--flow-total-steps", "50000",
            "--cuda-visible-devices", gpu], f"{LOGD}/ensbnt_{arm}_s{s}_gate.log", gpu)


def _key(f):
    m = re.search(r"dumps/(.+?)_(LOW|MID|HIGH)/seed_(\d+)/", f)
    return (m.group(2), m.group(3)) if m else None


def _load(globpat):
    d = {}
    for f in glob.glob(globpat):
        k = _key(f)
        if k:
            z = np.load(f)
            d[k] = (z["samples"][:, :, :3].astype(np.float32), z["theta"][:, :3].astype(np.float32))
    return d


def _supstd(R):
    std = R.std(0)
    sup = np.array([np.abs(np.searchsorted(np.sort(R[:, j]), G, side="right") / len(R) - G).max()
                    for j in range(3)])
    return std, sup


def pool_sbc(arm):
    s41 = _load(ARMS[arm]["s41_glob"])
    s42 = _load(f"{A}/ens_bnt_{arm}_s42/gate/tarp_drp/dumps/*/seed_*/n*_m*/posterior_samples.npz")
    s43 = _load(f"{A}/ens_bnt_{arm}_s43/gate/tarp_drp/dumps/*/seed_*/n*_m*/posterior_samples.npz")
    if not (s42 and s43):
        return None
    # single (s41) and 3-compressor ensemble, tercile-key pooling (validated ~ per-obs)
    keys = sorted(set(s41) & set(s42) & set(s43))
    Rs = np.concatenate([(s41[k][0] < s41[k][1][:, None, :]).mean(1) for k in keys], 0)
    Re = []
    for k in keys:
        S = np.concatenate([s41[k][0], s42[k][0], s43[k][0]], axis=1)
        Re.append((S < s41[k][1][:, None, :]).mean(1))
    Re = np.concatenate(Re, 0)
    return dict(single=_supstd(Rs), ens=_supstd(Re), n=len(Re))


def report():
    P = ["Om", "s8", "w0"]
    rows = ["# BNT l1 auto / +product — compressor-ENSEMBLE calibration\n",
            "Single BNT arms over-confident (auto sup 0.077/0.077/0.068, +product 0.054/0.072/0.062;",
            "simultaneous band d99=0.066). Fix = 3-compressor deep ensemble (seeds 41/42/43), the joint-l1 lever.\n",
            "| arm | variant | SBC std (Om/s8/w0) | sup\\|F-r\\| (Om/s8/w0) | inside band? |",
            "|---|---|---|---|---|"]
    for arm in ("product", "auto"):
        r = pool_sbc(arm)
        if not r:
            rows.append(f"| {arm} | (pending build) | - | - | - |"); continue
        for var, (std, sup) in (("single (s41)", r["single"]), ("**ensemble x3**", r["ens"])):
            ok = "yes" if max(sup) <= 0.066 else f"NO (max {max(sup):.3f})"
            rows.append(f"| {arm} | {var} | {std[0]:.3f}/{std[1]:.3f}/{std[2]:.3f} | "
                        f"{sup[0]:.3f}/{sup[1]:.3f}/{sup[2]:.3f} | {ok} |")
    out = f"{A}/RESULT_BNT_AUTOPROD_ENSEMBLE.md"
    open(out, "w").write("\n".join(rows) + "\n")
    print("\n" + "\n".join(rows), flush=True)
    print(f"\nwrote {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="product,auto")
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--pool-only", action="store_true")
    a = ap.parse_args()
    Path(LOGD).mkdir(parents=True, exist_ok=True)
    if not a.pool_only:
        for arm in a.arms.split(","):
            for s in a.seeds.split(","):
                print(f"\n=== build {arm} seed {s} (GPU {a.gpu}) ===", flush=True)
                build(arm, s, a.gpu)
    report()


if __name__ == "__main__":
    main()
