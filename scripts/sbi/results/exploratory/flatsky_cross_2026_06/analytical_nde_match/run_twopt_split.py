#!/usr/bin/env python3
"""Two-point-vs-non-Gaussian split for the cross-map operators (PLAN_2PT_SPLIT.md).

By P7 the auto+cross wavelet (co)variance vector `cov` IS the complete two-point sector. Append
it to each L1 arm; anything a cross-map adds on top of `cov` is non-Gaussian by construction.

Arms (no-BNT, matched VMIM 10-D -> sbilens_realnvp 4x128, SCREEN = seed 41, n=1000):
  cov          : cov alone                        (2pt sector)
  auto_cov     : auto-l1     (+) cov              (baseline; auto_cov - cov = positive control)
  conv_cov     : (auto+conv)-l1     (+) cov       (conv_cov - auto_cov = conv non-Gaussian)
  product_cov  : (auto+product)-l1  (+) cov       (product_cov - auto_cov = product non-Gaussian)

GPU 0 (verified free at launch). Resumable (skips completed steps). Writes RESULT_TWOPT_SPLIT.md.
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
OUT = f"{A}/twopt_split"
LOGD = f"{OUT}/logs"
GPU = "0"
MEM = "0.85"
SEEDS = "41"          # SCREEN: single compressor seed
N_OBS = "1000"        # SCREEN
STEPS = "50000"
NDE = ["--nde-family", "sbilens_realnvp", "--nde-layers", "4", "--nde-hidden", "128"]

# arm -> (parent L1 cache dir | None, parent fiducial npz | None)
# NB: the lc2st fiducials lack a `truth` key (build_flatsky_joint_arm --append-fid needs it),
# so we point at truth-augmented copies built in twopt_split/parent_fids/ (see run notes).
PF = f"{OUT}/parent_fids"
ARMS = {
    "cov":         (None, None),
    "auto_cov":    (f"{FC}/l1_matrix/l1_none_cache/flat_local_none",       f"{PF}/fiducial_summaries_none.npz"),
    "conv_cov":    (f"{FC}/l1_matrix/l1_conv_cache/flat_local_conv",       f"{PF}/fiducial_summaries_conv.npz"),
    "product_cov": (f"{FC}/l1_matrix/l1_product_cache/flat_local_product", f"{PF}/fiducial_summaries_product.npz"),
}


def sh(cmd, log, extra_env=None):
    env = dict(os.environ, XLA_PYTHON_CLIENT_MEM_FRACTION=MEM, CUDA_VISIBLE_DEVICES=GPU)
    if extra_env:
        env.update(extra_env)
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


def run_arm(name, parent_cache, parent_fid):
    print(f"\n=== arm {name} ===", flush=True)
    odir = f"{OUT}/{name}"
    raw_cc, raw_cf = f"{odir}/raw_cache", f"{odir}/raw_fid.npz"
    vc, vf = f"{odir}/cache", f"{odir}/fid_summ.npz"
    Path(odir).mkdir(parents=True, exist_ok=True)

    # 1. build the cov (2pt) datavector, optionally appended to the parent L1 arm
    if not Path(f"{raw_cc}/l1_train.npz").exists():
        print("  build cov datavector ...", flush=True)
        cmd = [PY, f"{SBI}/build_flatsky_joint_arm.py", "--stat", "cov", "--basis", "nobnt",
               "--k", "10", "--out-cache", raw_cc, "--out-fid", raw_cf]
        if parent_cache:
            cmd += ["--append-to", parent_cache, "--append-fid", parent_fid]
        if sh(cmd, f"{LOGD}/{name}_build.log") != 0:
            print(f"  BUILD FAILED {name}"); return None

    # 2. VMIM compress -> 10-D
    if not Path(f"{vc}/l1_train.npz").exists():
        print("  vmim compress -> 10-D ...", flush=True)
        if sh([PY, f"{SBI}/vmim_from_cache.py", "--cache-dir", raw_cc, "--fid-npz", raw_cf,
               "--out-cache", vc, "--out-fid", vf, "--summary-dim", "10", "--seed", "41",
               "--cuda-visible-devices", GPU], f"{LOGD}/{name}_vmim.log") != 0:
            print(f"  VMIM FAILED {name}"); return None

    # 3. train sbi_lens RealNVP + eval (SCREEN: seed 41, n=1000)
    out = f"{odir}/screen"
    if not Path(f"{out}/median_summary.json").exists():
        print("  train RealNVP + eval (screen) ...", flush=True)
        sh([PY, f"{SBI}/train_nde_from_compressed.py", "--train-cache-dir", vc,
            "--cache-prefix", "l1", "--arm-label", f"tp_{name}", "--fiducial-summaries-npz", vf,
            "--output-dir", out, *NDE, "--n-obs", N_OBS, "--seeds", SEEDS,
            "--flow-total-steps", STEPS, "--cuda-visible-devices", GPU], f"{LOGD}/{name}_screen.log")
    return fom3(out)


def main():
    Path(LOGD).mkdir(parents=True, exist_ok=True)
    res = {}
    for name, (pc, pf) in ARMS.items():
        res[name] = run_arm(name, pc, pf)
        print(f"  -> {name} FoM3(screen) = {res[name]}", flush=True)

    def d(a, b):
        return None if (res.get(a) is None or res.get(b) is None) else res[a] - res[b]

    rows = ["# Two-point vs non-Gaussian split of the cross-map operators (SCREEN: s41, n=1000)",
            "Matched VMIM 10-D -> sbilens_realnvp 4x128, no-BNT. cov = complete 2pt sector (P7).",
            "Committed refs (RealNVP): auto 2448, +conv 2671, +product 3045, joint l1 3371, CNN 3326.", "",
            "| arm | FoM3 (screen) |", "|---|---|"]
    for n in ("cov", "auto_cov", "conv_cov", "product_cov"):
        v = res.get(n)
        rows.append(f"| {n} | {v:.0f} |" if v else f"| {n} | FAILED |")
    pc = d("auto_cov", "cov")
    dconv = d("conv_cov", "auto_cov")
    dprod = d("product_cov", "auto_cov")
    rows += ["",
             f"- positive control  auto_cov - cov       = {None if pc is None else round(pc)}  (>0 ⇒ test sees non-Gaussian)",
             f"- ΔNG(conv)         conv_cov - auto_cov   = {None if dconv is None else round(dconv)}  (≈0 ⇒ conv is 2pt-only)",
             f"- ΔNG(product)      product_cov - auto_cov = {None if dprod is None else round(dprod)}  (>0 ⇒ product has non-Gaussian)",
             "", "SCREEN only (1 seed, n=1000, ungated). Escalate to 3-seed/n=9000/gated if signs are clear."]
    txt = "\n".join(rows) + "\n"
    open(f"{OUT}/RESULT_TWOPT_SPLIT.md", "w").write(txt)
    print("\n" + txt, flush=True)


if __name__ == "__main__":
    main()
