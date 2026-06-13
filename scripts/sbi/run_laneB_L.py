#!/usr/bin/env python
"""Schedule-L (light, realistic) lane-B: build the recombination arms, sweep FoM3, GATE C,
and write the M-vs-L robustness comparison (PLAN_M4_GATE_C.md, 2nd schedule = Andreas's L).

Serializes after run_laneB_gate_c (the schedule-M gate) so it doesn't contend on GPU 2.
Pinned to GPU 2. Sequential single-worker (safe unattended). Steps:
  build B1L (cutsum6) + B2L (cutdeep2) on the L-masked B0L base
  -> sweep FoM3 for B0L/B1L/B2L/B3L  -> GATE C (tarp + coverage) for all 4
  -> GATE_C_LANEB_L.md + M_VS_L_ROBUSTNESS.md (does the ~1.8x rescue hold across schedules?)
Detached: (cd scripts/sbi && setsid nohup <py> run_laneB_L.py > .../laneB_L.out 2>&1 &)
"""
import glob, json, os, subprocess, time
from pathlib import Path
import numpy as np

REPO = "/mnt/home/tersenov/software/cnn_sbi"; SBI = f"{REPO}/scripts/sbi"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
B = f"{SBI}/results/exploratory/flatsky_cross_2026_06"; OM2 = f"{B}/overnight_menu_2"
GC = f"{OM2}/gate_c_laneB_L"; LOGS = f"{GC}/logs"
GPU = "2"; MEM = "0.45"
KEEP_L = "1,2,3,4;1,2,3,4;0,1,2,3,4;0,1,2,3,4"
PRE = ("log1p-zscore", "5", "1e-5")
# L arms; M-arm counterparts (for the M-vs-L table) live in overnight_menu_2/<Marm>/...
L_ARMS = ["B0L_bntcut_l1", "B1L_bntcut_sums", "B2L_bntcut_deep2", "B3L_nobnt_unicut"]
M_OF = {"B0L_bntcut_l1": "B0_bntcut_l1", "B1L_bntcut_sums": "B1_bntcut_sums",
        "B2L_bntcut_deep2": "B2_bntcut_deep2", "B3L_nobnt_unicut": "B3_nobnt_unicut"}
M_FOM = {"B0_bntcut_l1": 268, "B1_bntcut_sums": 596, "B2_bntcut_deep2": 613,
         "B3_nobnt_unicut": 337}


def status(line):
    with open(f"{OM2}/OVERNIGHT2_STATUS.md", "a") as fh:
        fh.write(line + "\n")
    print("STATUS: " + line, flush=True)


def run(tag, cmd, t0, timeout=2.5 * 3600):
    os.makedirs(LOGS, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="3",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION=MEM,
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", CNN_CPU_THREADS="8",
               OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", CUDA_VISIBLE_DEVICES=GPU)
    print(f"[{time.time()-t0:6.0f}s] === {tag} ===", flush=True)
    with open(f"{LOGS}/{tag}.log", "w") as log:
        try:
            rc = subprocess.run(cmd, cwd=SBI, env=env, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -99
    print(f"[{time.time()-t0:6.0f}s] {'OK' if rc==0 else 'FAIL'} {tag}", flush=True)
    return rc == 0


def med(d):
    f = Path(d) / "median_summary.json"
    return json.load(open(f)) if f.exists() else None


def sweep_cmd(arm):
    return [PY, "population_sweep_flatsky.py", "--train-cache-dir", f"{OM2}/{arm}/cache",
            "--cache-prefix", "l1", "--arm-label", f"laneBL_{arm}",
            "--fiducial-summaries-npz", f"{OM2}/{arm}/fiducial_summaries.npz",
            "--output-dir", f"{OM2}/{arm}/population_sweep_full",
            "--preproc-transform", PRE[0], "--clip-value", PRE[1],
            "--min-feature-variance", PRE[2], "--seeds", "41,42,43",
            "--n-obs", "9000", "--max-perm", "50", "--m-samples", "2000",
            "--cuda-visible-devices", GPU]


def gate_cmd(arm):
    return [PY, "tarp_stratified_val.py", "--train-cache-dir", f"{OM2}/{arm}/cache",
            "--cache-prefix", "l1", "--arm-label", arm, "--dumps-root", f"{GC}/tarp_drp/dumps",
            "--preproc-transform", PRE[0], "--clip-value", PRE[1],
            "--min-feature-variance", PRE[2], "--seeds", "41,42,43",
            "--cuda-visible-devices", GPU]


def postcut_cmd(variant, arm):
    return [PY, "build_flatsky_postcut_arm.py", "--variant", variant, "--keep", KEEP_L,
            "--base-cache", f"{OM2}/B0L_bntcut_l1/cache",
            "--base-fid", f"{OM2}/B0L_bntcut_l1/fiducial_summaries.npz",
            "--out-cache", f"{OM2}/{arm}/cache", "--out-fid", f"{OM2}/{arm}/fiducial_summaries.npz"]


def net_bias(arm):
    devs = []
    for f in glob.glob(f"{GC}/tarp_drp/curves/tarp_curve_{arm}_*_seed*_dim3.npz"):
        z = np.load(f); a = z["alpha"]; e = z["ecp_bootstrap"].mean(0)
        devs.append(float(np.trapz(e - a, a) * 2))
    return (float(np.mean(devs)), float(np.std(devs))) if devs else (None, None)


def sbc_std(arm):
    rs = []
    for f in glob.glob(f"{GC}/tarp_drp/dumps/{arm}_*/seed_*/n*_m*/posterior_samples.npz"):
        z = np.load(f); rs.append((z["samples"] < z["theta"][:, None, :]).mean(1))
    return list(np.concatenate(rs, 0).std(0)[:3]) if rs else None


def wait_for(proc_pat, t0):
    while subprocess.run(["pgrep", "-f", proc_pat], capture_output=True).returncode == 0:
        print(f"[{time.time()-t0:6.0f}s] waiting for {proc_pat} ...", flush=True)
        time.sleep(120)


def main():
    t0 = time.time()
    os.makedirs(GC, exist_ok=True)
    status(f"## lane-B schedule-L started {time.strftime('%F %T')} (GPU 2; after M gate)")
    wait_for("[r]un_laneB_gate_c", t0)
    # build recombination arms
    if not run("build_B1L", postcut_cmd("cutsum6", "B1L_bntcut_sums"), t0):
        status("- [L] B1L build FAIL")
    if not run("build_B2L", postcut_cmd("cutdeep2", "B2L_bntcut_deep2"), t0):
        status("- [L] B2L build FAIL")
    # sweeps
    fomL = {}
    for arm in L_ARMS:
        if Path(f"{OM2}/{arm}/cache/l1_train.npz").exists():
            run(f"sweep_{arm}", sweep_cmd(arm), t0)
            m = med(f"{OM2}/{arm}/population_sweep_full")
            fomL[arm] = m["fom3"] if m else None
            status(f"- [L] {arm} FoM3 {fomL[arm]:.0f}" if fomL.get(arm) else f"- [L] {arm} sweep FAIL")
    # gates
    for arm in L_ARMS:
        if Path(f"{OM2}/{arm}/cache/l1_train.npz").exists():
            run(f"gate_{arm}", gate_cmd(arm), t0)
    run("coverage", [PY, "run_tarp_coverage.py", "--dumps-root", f"{GC}/tarp_drp/dumps",
        "--outdir", f"{GC}/tarp_drp", "--dims", "3"], t0)

    # report: M vs L robustness
    L = ["# Lane-B M-vs-L robustness (PLAN_M4_GATE_C.md 2nd schedule = L, light/realistic)", "",
         "Does the post-cut rescue (cut-BNT + recombinations beats the uniform-cut noBNT "
         "analysis) hold across cut schedules? M = moderate (shallow channel keeps 2 coarsest "
         "scales), L = light (shallow channels drop only their finest scale).", "",
         "| arm | M FoM3 | L FoM3 | L net-bias | L SBC std | L verdict |",
         "|---|---|---|---|---|---|"]
    for arm in L_ARMS:
        m = M_FOM[M_OF[arm]]; lf = fomL.get(arm); nb = net_bias(arm); ss = sbc_std(arm)
        v = "INCOMPLETE"
        if nb[0] is not None and ss is not None:
            std_off = max(max(0.0, 0.275 - s, s - 0.305) for s in ss)
            v = ("PASS" if abs(nb[0]) <= 0.05 and std_off == 0 else
                 "FAIL" if abs(nb[0]) > 0.10 or std_off >= 0.02 else "PASS-with-caveat")
        lf_cell = f"{lf:.0f}" if lf else "—"
        nb_cell = f"{nb[0]:+.3f}±{nb[1]:.3f}" if nb[0] is not None else "—"
        ss_cell = ",".join(f"{x:.3f}" for x in ss) if ss else "—"
        L.append(f"| {arm} | {m} | {lf_cell} | {nb_cell} | {ss_cell} | **{v}** |")
    if fomL.get("B1L_bntcut_sums") and fomL.get("B3L_nobnt_unicut"):
        rB1 = fomL["B1L_bntcut_sums"] / fomL["B3L_nobnt_unicut"]
        rB2 = (fomL["B2L_bntcut_deep2"] / fomL["B3L_nobnt_unicut"]
               if fomL.get("B2L_bntcut_deep2") else float("nan"))
        L += ["", f"**L ratios: B1L/B3L = {rB1:.2f}, B2L/B3L = {rB2:.2f}** (M was 1.77 / 1.82). "
              "Robust-across-schedules if the L ratios are also clearly > 1 with B1L/B2L "
              "calibrated; schedule-dependent otherwise."]
    Path(GC, "M_VS_L_ROBUSTNESS.md").write_text("\n".join(L) + "\n")
    status(f"## lane-B schedule-L complete {time.strftime('%F %T')} ({(time.time()-t0)/3600:.1f} h)")
    print("LANE-B L DONE", flush=True)


if __name__ == "__main__":
    main()
