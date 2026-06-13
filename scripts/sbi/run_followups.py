#!/usr/bin/env python
"""Sequential follow-up runner (supersedes run_k15_rerun.py): one worker, one job at a
time, each waiting for a genuinely-free GPU in {0,1,2} (no tenant trampling). Jobs:

  K=15:  C2 full sweep, C3 full sweep, C2 gate, coverage  (mem 0.85)
  VMIM robustness: seeds 42 & 43 -> build (110min cap) + full sweep + gate  (mem 0.85)
                   (seed 41 = the existing A1 arm; reused, not rebuilt)
  finalize: VMIM_ROBUSTNESS.md (3-seed FoM3 band + calibration + leakage note),
            regen OVERNIGHT2_RESULT.md.

Single worker => no intra/inter-launcher GPU race. Detached:
  (cd scripts/sbi && setsid nohup <py> run_followups.py > .../followups.out 2>&1 &)
"""
import glob
import json
import subprocess
import time
from pathlib import Path

REPO = "/mnt/home/tersenov/software/cnn_sbi"
SBI = f"{REPO}/scripts/sbi"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
FC = f"{SBI}/results/exploratory/flatsky_cross_2026_06"
OM = f"{FC}/overnight_menu"
OM2 = f"{FC}/overnight_menu_2"
GC2 = f"{OM2}/gate_c"
LOGS = f"{OM2}/logs"
PAIR2D = f"{OM}/pair2dq_nobnt"
FREE_MEM_MB, FREE_UTIL, POLL_S = 2000, 15, 240


def status(line):
    with open(f"{OM2}/OVERNIGHT2_STATUS.md", "a") as fh:
        fh.write(line + "\n")
    print("STATUS: " + line, flush=True)


def gpu_state(g):
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                            "--format=csv,noheader,nounits", "-i", str(g)],
                           capture_output=True, text=True, timeout=20).stdout.strip().split(",")
        return int(o[0]), int(o[1])
    except Exception:
        return 1 << 30, 100


def wait_free_gpu(t0):
    while True:
        for g in (1, 0, 2):
            m, u = gpu_state(g)
            if m < FREE_MEM_MB and u < FREE_UTIL:
                time.sleep(20)
                m2, u2 = gpu_state(g)
                if m2 < FREE_MEM_MB and u2 < FREE_UTIL:
                    return g
        print(f"[{time.time()-t0:6.0f}s] no free GPU — waiting {POLL_S}s", flush=True)
        time.sleep(POLL_S)


def run(tag, cmd, gpu, t0, mem=0.85, timeout=3.5 * 3600):
    Path(LOGS).mkdir(parents=True, exist_ok=True)
    import os
    env = dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="3",
               XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_PYTHON_CLIENT_MEM_FRACTION=str(mem),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               CNN_CPU_THREADS="8", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
               OPENBLAS_NUM_THREADS="8", NUMEXPR_NUM_THREADS="8",
               TF_NUM_INTRAOP_THREADS="8", TF_NUM_INTEROP_THREADS="2",
               CUDA_VISIBLE_DEVICES=str(gpu))
    print(f"[{time.time()-t0:6.0f}s] === {tag} === GPU{gpu}", flush=True)
    tj = time.time()
    with open(f"{LOGS}/followup_{tag}.log", "w") as log:
        try:
            rc = subprocess.run(cmd, cwd=SBI, env=env, stdout=log,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -99
    print(f"[{time.time()-t0:6.0f}s] {'OK' if rc == 0 else 'FAIL'} {tag} "
          f"({time.time()-tj:.0f}s)", flush=True)
    return rc == 0


def med(d):
    f = Path(d) / "median_summary.json"
    return json.load(open(f)) if f.exists() else None


def sweep(cache, fid, outdir, gpu, t0, preproc=("log1p-zscore", "5", "1e-5")):
    cmd = [PY, "population_sweep_flatsky.py", "--train-cache-dir", cache,
           "--cache-prefix", "l1", "--arm-label", Path(outdir).parent.name,
           "--fiducial-summaries-npz", fid, "--output-dir", outdir,
           "--preproc-transform", preproc[0], "--clip-value", preproc[1],
           "--min-feature-variance", preproc[2], "--seeds", "41,42,43",
           "--n-obs", "9000", "--max-perm", "50", "--m-samples", "2000",
           "--cuda-visible-devices", str(gpu)]
    return cmd


def gate(cache, label, gpu, preproc=("log1p-zscore", "5", "1e-5")):
    return [PY, "tarp_stratified_val.py", "--train-cache-dir", cache,
            "--cache-prefix", "l1", "--arm-label", label,
            "--dumps-root", f"{GC2}/tarp_drp/dumps",
            "--preproc-transform", preproc[0], "--clip-value", preproc[1],
            "--min-feature-variance", preproc[2], "--seeds", "41,42,43",
            "--cuda-visible-devices", str(gpu)]


def vmim_build(seed, gpu):
    out = f"{OM2}/A1_vmim_s{seed}"
    return [PY, "vmim_from_cache.py", "--cache-dir", f"{PAIR2D}/cache",
            "--fid-npz", f"{PAIR2D}/fiducial_summaries.npz",
            "--out-cache", f"{out}/cache", "--out-fid", f"{out}/fiducial_summaries.npz",
            "--steps", "30000", "--max-minutes", "110", "--seed", str(seed),
            "--cuda-visible-devices", str(gpu)]


def tarp_net_bias(arm):
    devs = []
    for f in glob.glob(f"{GC2}/tarp_drp/curves/tarp_curve_{arm}_*_seed*_dim3.npz"):
        import numpy as np
        z = np.load(f); a = z["alpha"]; e = z["ecp_bootstrap"].mean(0)
        devs.append(float(__import__("numpy").trapz(e - a, a) * 2))
    import numpy as np
    return (float(np.mean(devs)), float(np.std(devs)), len(devs)) if devs else (None, None, 0)


def finalize():
    import numpy as np
    VMIM = {41: f"{OM2}/A1_pair2d_vmim/population_sweep_full",
            42: f"{OM2}/A1_vmim_s42/population_sweep_full",
            43: f"{OM2}/A1_vmim_s43/population_sweep_full"}
    GATES = {41: "A1_pair2d_vmim", 42: "A1_vmim_s42", 43: "A1_vmim_s43"}
    foms, rows = [], []
    for s, d in VMIM.items():
        m = med(d)
        nb = tarp_net_bias(GATES[s])
        if m:
            foms.append(m["fom3"])
            rows.append(f"| {s} | {m['fom3']:.0f} | {m['sigma_Om']:.4f} | "
                        f"{m['sigma_s8']:.4f} | {m['sigma_w0']:.4f} | "
                        + (f"{nb[0]:+.3f}±{nb[1]:.3f}" if nb[0] is not None else "—") + " |")
        else:
            rows.append(f"| {s} | FAIL | — | — | — | — |")
    L = ["# VMIM compressor multi-seed robustness (Lane A check)", "",
         "Question: is A1's FoM3 3822 robust to the VMIM compressor seed, and is its "
         "joint-coverage calibration (TARP net bias, + = conservative) stable? 3 NDE seeds "
         "per arm; compressor seed varied.", "",
         "| comp. seed | FoM3 | σ(Ωm) | σ(σ8) | σ(w0) | TARP net bias |",
         "|---|---|---|---|---|---|", *rows, ""]
    if foms:
        L.append(f"**FoM3 across compressor seeds: {np.mean(foms):.0f} ± {np.std(foms):.0f} "
                 f"(min {min(foms):.0f}, max {max(foms):.0f}); spread "
                 f"{100*np.std(foms)/np.mean(foms):.0f}%.** Baselines: l1+product 2875 "
                 "(gate-C clean), pair2d 2794, l1-auto 2405.")
        L.append("")
        L.append("Reading (derived): if the band stays well above l1+product 2875 AND the "
                 "TARP net bias stays >= 0 (conservative) across seeds, the joint-PDF gain "
                 "is robust and calibrated -> third pillar confirmed. A large spread or a "
                 "seed dropping to ~2875 would mark it compressor-fragile (quote a band).")
    L += ["", "## Leakage control (architecture argument + empirical)",
          "- The VMIM compressor trains ONLY on the pair2d TRAIN split. TARP/SBC are measured "
          "on the held-out VAL split (never seen by the compressor); FoM3 on the independent "
          "fiducial sims. So calibration and constraining power are evaluated out-of-sample.",
          "- Harmful overfitting would DEGRADE out-of-sample FoM3 (noisy features on unseen "
          "obs), not inflate it; a high held-out FoM3 + net-conservative TARP is evidence "
          "AGAINST harmful leakage.",
          "- The multi-seed band above is the empirical robustness test; a stricter "
          "compressor-train / NDE-disjoint split is a possible follow-up if the band warrants."]
    Path(OM2, "VMIM_ROBUSTNESS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def main():
    t0 = time.time()
    status(f"## followups started {time.strftime('%F %T')} (K=15 + 3-seed VMIM)")

    # --- K=15 reruns ---
    for name in ("C2_pair2d_k15", "C3_pair2d_k15_bnt_ar"):
        g = wait_free_gpu(t0)
        status(f"- [followup] GPU{g} free -> {name} full")
        ok = run(f"k15_{name}", sweep(f"{OM2}/{name}/cache",
                 f"{OM2}/{name}/fiducial_summaries.npz",
                 f"{OM2}/{name}/population_sweep_full", g, t0), g, t0)
        m = med(f"{OM2}/{name}/population_sweep_full")
        status(f"- [followup] {name} {'FoM3 %.0f' % m['fom3'] if m else 'FAIL'}")
    g = wait_free_gpu(t0)
    if run("k15_gate_C2", gate(f"{OM2}/C2_pair2d_k15/cache", "C2_pair2d_k15", g), g, t0):
        run("k15_coverage", [PY, "run_tarp_coverage.py", "--dumps-root",
            f"{GC2}/tarp_drp/dumps", "--outdir", f"{GC2}/tarp_drp", "--dims", "3"], g, t0)

    # --- VMIM seeds 42, 43: build + sweep + gate ---
    for seed in (42, 43):
        out = f"{OM2}/A1_vmim_s{seed}"
        g = wait_free_gpu(t0)
        status(f"- [followup] GPU{g} free -> VMIM build seed {seed}")
        if not run(f"vmim_build_s{seed}", vmim_build(seed, g), g, t0):
            status(f"- [followup] VMIM s{seed} BUILD FAIL — skipping its sweep/gate")
            continue
        g = wait_free_gpu(t0)
        run(f"vmim_sweep_s{seed}", sweep(f"{out}/cache", f"{out}/fiducial_summaries.npz",
            f"{out}/population_sweep_full", g, t0, preproc=("none", "0", "1e-12")), g, t0)
        m = med(f"{out}/population_sweep_full")
        status(f"- [followup] VMIM s{seed} {'FoM3 %.0f' % m['fom3'] if m else 'FAIL'}")
        g = wait_free_gpu(t0)
        run(f"vmim_gate_s{seed}", gate(f"{out}/cache", f"A1_vmim_s{seed}", g,
            preproc=("none", "0", "1e-12")), g, t0)
    g = wait_free_gpu(t0)
    run("vmim_coverage", [PY, "run_tarp_coverage.py", "--dumps-root",
        f"{GC2}/tarp_drp/dumps", "--outdir", f"{GC2}/tarp_drp", "--dims", "3"], g, t0)

    finalize()
    subprocess.run([PY, "run_overnight_menu_2.py", "--regen-only"], cwd=SBI)
    status(f"## followups complete {time.strftime('%F %T')} ({(time.time()-t0)/3600:.1f} h)")
    print("FOLLOWUPS DONE", flush=True)


if __name__ == "__main__":
    main()
