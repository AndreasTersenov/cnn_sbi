#!/usr/bin/env python3
"""Goal-1 D1: port the BNT deep-channel rescue to the MATCHED best-NDE pipeline and gate it.

The §5.4 ladder (+2 deep -> 1.082x, "spanning") was measured with the OLD MAF/sweep machinery
on auto-only L1 and scored on FoM3 ONLY (not calibration-gated). The paper headline pipeline is
VMIM-MLP -> 10-D -> sbi_lens RealNVP 4x128, gated (TARP+SBC). This re-runs the auto-only ladder
THROUGH that exact pipeline and gates the rescue, so the rescue becomes a paper-grade result.

Ladder (all VMIM->RealNVP 4x128, seeds 41,42,43, n=9000 median; gate = TARP-DRP 600 + SBC):
  nobnt_auto       : noBNT l1-auto      (baseline; reuse existing l1none_vmim_s41 compressed cache)
  bnt_auto         : BNT l1-auto        (the collapse; NEW)
  bnt_auto_deep2   : BNT l1-auto + deep2 (4 BNT autos + avg + bin4; the RESCUE; NEW)

Registered (BEFORE looking): recovered = (deep2 - BNT)/(noBNT - BNT).
  >=0.8 AND deep2 PASSES gate -> "rescue confirmed in matched pipeline".
  0.4-0.8 -> partial-in-RealNVP. <0.4 -> rescue is MAF-specific (surprising).
  FoM3 recovers but gate FAILS -> over-confidence, NOT rescue (report as such).

GPU 2 only (sole effective tenant; GPU 0/1 foreign, GPU 3 never). Resumable: skips any step
whose output already exists. Writes RESULT_BNT_DEEP_RESCUE.md.
"""
import json
import os
import subprocess
import time
from pathlib import Path

PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
SBI = "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi"
BASE = f"{SBI}/results/exploratory/flatsky_cross_2026_06"
A = f"{BASE}/analytical_nde_match"
LOGD = f"{A}/logs"
GPU = "2"
MEM = "0.8"
SEEDS = "41,42,43"
STEPS = "50000"
NDE = ["--nde-family", "sbilens_realnvp", "--nde-layers", "4", "--nde-hidden", "128"]

ARMS = {
    "nobnt_auto": dict(
        compressed_cache=f"{A}/l1none_vmim_s41/cache",
        compressed_fid=f"{A}/l1none_vmim_s41/fiducial_summaries.npz",
        gate=False,
    ),
    "bnt_auto": dict(
        raw_cache=f"{BASE}/bnt_campaign/l1_matrix/l1_none_cache/flat_local_none_bnt",
        raw_fid=f"{BASE}/bnt_campaign/fiducial_summaries/fiducial_summaries_l1_none.npz",
        gate=True,
    ),
    "bnt_auto_deep2": dict(
        build=[PY, f"{SBI}/build_flatsky_bntdeep_arm.py", "--deep-mode", "deep2",
               "--out-cache", f"{A}/bnt_auto_deep2_raw/cache",
               "--out-fid", f"{A}/bnt_auto_deep2_raw/fid.npz"],
        raw_cache=f"{A}/bnt_auto_deep2_raw/cache",
        raw_fid=f"{A}/bnt_auto_deep2_raw/fid.npz",
        gate=True,
    ),
}


def sh(cmd, log):
    env = dict(os.environ, XLA_PYTHON_CLIENT_MEM_FRACTION=MEM)
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    dt = time.time() - t0
    print(f"  [{dt/60:.1f} min, rc={rc}] {os.path.basename(log)}", flush=True)
    return rc


def fom3(odir):
    try:
        return json.load(open(f"{odir}/median_summary.json"))["fom3"]
    except Exception:
        return None


def run_arm(name, cfg):
    print(f"\n=== arm {name} ===", flush=True)
    odir = f"{A}/{name}"
    Path(odir).mkdir(parents=True, exist_ok=True)

    # 0. build raw datavector cache if needed
    if "build" in cfg and not Path(f"{cfg['raw_cache']}/l1_train.npz").exists():
        print("  build raw cache ...", flush=True)
        if sh(cfg["build"], f"{LOGD}/rescue_{name}_build.log") != 0:
            print(f"  BUILD FAILED for {name}", flush=True)
            return None

    # 1. VMIM compress raw -> 10-D (skip if a compressed cache is already supplied/exists)
    if "compressed_cache" not in cfg:
        cc, cf = f"{odir}/cache", f"{odir}/fiducial_summaries.npz"
        cfg["compressed_cache"], cfg["compressed_fid"] = cc, cf
        if not Path(f"{cc}/l1_train.npz").exists():
            print("  vmim compress ...", flush=True)
            sh([PY, f"{SBI}/vmim_from_cache.py", "--cache-dir", cfg["raw_cache"],
                "--fid-npz", cfg["raw_fid"], "--out-cache", cc, "--out-fid", cf,
                "--summary-dim", "10", "--seed", "41", "--cuda-visible-devices", GPU],
               f"{LOGD}/rescue_{name}_vmim.log")
    cc, cf = cfg["compressed_cache"], cfg["compressed_fid"]

    # 2. screen (n=1000) then 3. final (n=9000)
    for tag, n in (("screen", "1000"), ("n9000", "9000")):
        out = f"{odir}/{tag}"
        if not Path(f"{out}/median_summary.json").exists():
            print(f"  train {tag} (n={n}) ...", flush=True)
            sh([PY, f"{SBI}/train_nde_from_compressed.py", "--train-cache-dir", cc,
                "--cache-prefix", "l1", "--arm-label", f"{name}_{tag}",
                "--fiducial-summaries-npz", cf, "--output-dir", out, *NDE,
                "--n-obs", n, "--seeds", SEEDS, "--flow-total-steps", STEPS,
                "--cuda-visible-devices", GPU], f"{LOGD}/rescue_{name}_{tag}.log")

    # 4. GATE C (TARP-DRP + SBC), only for the BNT arms
    verdict = None
    if cfg.get("gate"):
        G = f"{odir}/gate"
        if not Path(f"{G}/verdict.json").exists():
            print("  gate: tarp_stratified -> coverage -> verdict ...", flush=True)
            sh([PY, f"{SBI}/tarp_stratified_val_nde.py", "--train-cache-dir", cc,
                "--cache-prefix", "l1", "--arm-label", name,
                "--dumps-root", f"{G}/tarp_drp/dumps", *NDE, "--n-points", "600",
                "--seeds", SEEDS, "--flow-total-steps", STEPS,
                "--cuda-visible-devices", GPU], f"{LOGD}/rescue_{name}_gate.log")
            sh([PY, f"{SBI}/run_tarp_coverage.py", "--dumps-root", f"{G}/tarp_drp/dumps",
                "--outdir", f"{G}/tarp_drp", "--dims", "3"], f"{LOGD}/rescue_{name}_cov.log")
            sh([PY, f"{A}/gate_verdict.py", "--gate-dir", G, "--arms", name,
                "--json-out", f"{G}/verdict.json"], f"{LOGD}/rescue_{name}_verdict.log")
        try:
            verdict = json.load(open(f"{G}/verdict.json"))[0]
        except Exception:
            pass

    return dict(name=name, fom3_n1000=fom3(f"{odir}/screen"),
                fom3_n9000=fom3(f"{odir}/n9000"), verdict=verdict)


def main():
    Path(LOGD).mkdir(parents=True, exist_ok=True)
    res = {}
    for name, cfg in ARMS.items():
        res[name] = run_arm(name, cfg)

    nob = res["nobnt_auto"]["fom3_n9000"]
    bnt = res["bnt_auto"]["fom3_n9000"]
    dp2 = res["bnt_auto_deep2"]["fom3_n9000"]
    rec = None
    if None not in (nob, bnt, dp2) and (nob - bnt) != 0:
        rec = (dp2 - bnt) / (nob - bnt)

    rows = ["# Goal-1 D1 — BNT deep-channel rescue in the MATCHED best-NDE pipeline (gated)\n",
            "Pipeline: VMIM-MLP -> 10-D -> sbi_lens RealNVP 4x128, seeds 41,42,43, n=9000 median.",
            "Gate C: TARP-DRP (600 val pts, dims=3) + SBC, pooled 3 seeds.",
            "Registered: recovered=(deep2-BNT)/(noBNT-BNT); >=0.8 AND deep2 PASS gate => rescue confirmed.",
            "MAF-ladder reference (old path, FoM3-only): noBNT 2405 / BNT 364 / deep2 2573 (rec 1.082).\n",
            "| arm | FoM3 n=1000 | FoM3 n=9000 | gate verdict |",
            "|---|---|---|---|"]
    for name in ("nobnt_auto", "bnt_auto", "bnt_auto_deep2"):
        r = res[name] or {}
        v = r.get("verdict")
        vtxt = v["verdict"] if v else ("-" if name == "nobnt_auto" else "?")
        f1 = f"{r.get('fom3_n1000'):.0f}" if r.get("fom3_n1000") else "?"
        f9 = f"{r.get('fom3_n9000'):.0f}" if r.get("fom3_n9000") else "?"
        rows.append(f"| {name} | {f1} | {f9} | {vtxt} |")
    rows.append("")
    if rec is not None:
        rows.append(f"**recovered (n=9000) = ({dp2:.0f} - {bnt:.0f})/({nob:.0f} - {bnt:.0f}) = {rec:.3f}**")
    out = "\n".join(rows) + "\n"
    open(f"{A}/RESULT_BNT_DEEP_RESCUE.md", "w").write(out)
    print("\n" + out, flush=True)


if __name__ == "__main__":
    main()
