#!/usr/bin/env python
"""GATE C + representative corners for the BNT arms (paper pillar 2 validation).

Phases (corners FIRST so the BNT-vs-noBNT overlay figures land early):
  1. corners  — 3-MAF pooled posteriors at the typical/favorable obs per BNT arm
                (cnn_representative_corners.py / representative_corner_flatsky.py)
  2. overlays — bnt_corner_overlays.py (CPU; the 4 BNT-vs-noBNT corner figures)
  3. tarp     — tarp_stratified_val.py per BNT arm (600 held-out val points, FoM3 terciles)
  4. coverage — run_tarp_coverage.py (dims 3 6) + inline SBC ranks from the dumps
  5. lc2st    — lc2st_diagnostic.py for the CNN arms (10-d summary; works, unlike high-dim L1)
  6. report   — GATE_C_BNT.md with DERIVED verdicts (TARP max|ECP-alpha|, SBC mean/std/KS,
                L-C2ST reject fractions)

Greedy scheduler over --gpus (default 1,2). --dry-run prints commands.
Launch detached:  setsid nohup python run_bnt_gate_c.py > .../gate_c_bnt/driver.out 2>&1 &
"""
import argparse, glob, json, os, subprocess, time
import numpy as np
from pathlib import Path

REPO = "/mnt/home/tersenov/software/cnn_sbi"; SBI = f"{REPO}/scripts/sbi"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
FC = f"{SBI}/results/exploratory/flatsky_cross_2026_06"
BNT = f"{FC}/bnt_campaign"
GC = f"{BNT}/gate_c"
LOGS = f"{GC}/logs"

# (probe, op, train-cache-dir, fidsumm-npz, preproc(tuple))
ARMS = [
    ("cnn", "none", f"{BNT}/cnn_none_s41/cache",
     f"{BNT}/fiducial_summaries/fiducial_summaries_cnn_none_s41.npz",
     ("none", "0", "1e-12")),
    ("cnn", "product", f"{BNT}/cnn_product_s41/cache",
     f"{BNT}/fiducial_summaries/fiducial_summaries_cnn_product_s41.npz",
     ("none", "0", "1e-12")),
    ("l1", "none", f"{BNT}/l1_matrix/l1_none_cache/flat_local_none_bnt",
     f"{BNT}/fiducial_summaries/fiducial_summaries_l1_none.npz",
     ("log1p-zscore", "5", "1e-5")),
    ("l1", "product", f"{BNT}/l1_matrix/l1_product_cache/flat_local_product_bnt",
     f"{BNT}/fiducial_summaries/fiducial_summaries_l1_product.npz",
     ("log1p-zscore", "5", "1e-5")),
]
PNAMES = ["Om", "s8", "w0", "h0", "ns", "Ob"]


def corners_cmd(probe, op, cache, fid, preproc, gpu):
    out = f"{BNT}/representative_corner/{probe}_{op}"
    if probe == "cnn":
        return [PY, "cnn_representative_corners.py", "--arm-label", f"bnt_{probe}_{op}",
                "--train-cache-dir", cache, "--cache-prefix", "cnn",
                "--fiducial-summaries-npz", fid, "--output-dir", out,
                "--preproc-transform", preproc[0], "--clip-value", preproc[1],
                "--min-feature-variance", preproc[2],
                "--cuda-visible-devices", str(gpu)]
    return [PY, "representative_corner_flatsky.py",
            "--train-cache-dir", cache, "--arm-label", f"bnt_{probe}_{op}",
            "--fiducial-summaries-npz", fid, "--output-dir", out,
            "--cuda-visible-devices", str(gpu)]


def tarp_cmd(probe, op, cache, fid, preproc, gpu):
    return [PY, "tarp_stratified_val.py",
            "--train-cache-dir", cache, "--cache-prefix", probe,
            "--arm-label", f"bnt_{probe}_{op}", "--dumps-root", f"{GC}/tarp_drp/dumps",
            "--preproc-transform", preproc[0], "--clip-value", preproc[1],
            "--min-feature-variance", preproc[2], "--seeds", "41,42,43",
            "--cuda-visible-devices", str(gpu)]


def lc2st_cmd(probe, op, cache, fid, preproc, gpu):
    return [PY, "lc2st_diagnostic.py",
            "--train-cache-dir", cache, "--cache-prefix", "cnn",
            "--arm-label", f"bnt_{probe}_{op}", "--output-dir", f"{GC}/lc2st/bnt_{probe}_{op}",
            "--fiducial-summaries-npz", fid,
            "--preproc-transform", preproc[0], "--clip-value", preproc[1],
            "--min-feature-variance", preproc[2], "--seeds", "41,42,43",
            "--cuda-visible-devices", str(gpu)]


def run_phase(name, jobs, cmd_fn, gpus):
    print(f"\n===== PHASE {name} =====", flush=True)
    os.makedirs(LOGS, exist_ok=True)
    pending = list(jobs); slots = {g: None for g in gpus}; t0 = time.time(); failed = {}

    def launch(job, gpu):
        tag = f"{job[0]}_{job[1]}"
        try:
            c = cmd_fn(*job, gpu)
        except Exception as exc:
            failed[tag] = f"cmd-build: {exc}"
            print(f"[{time.time()-t0:6.0f}s] SKIP {name} {tag} ({exc})", flush=True)
            return None
        log = open(f"{LOGS}/{name}_{tag}.log", "w")
        env = dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="3",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.5",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", CNN_CPU_THREADS="8")
        p = subprocess.Popen(c, cwd=SBI, env=env, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL)
        print(f"[{time.time()-t0:6.0f}s] LAUNCH {name} {tag} GPU{gpu} (pid {p.pid})", flush=True)
        return (job, p, log)

    while pending or any(slots.values()):
        for g in gpus:
            s = slots[g]
            if s and s[1].poll() is not None:
                job, p, log = s; log.close(); slots[g] = None
                tag = f"{job[0]}_{job[1]}"
                if p.returncode != 0:
                    failed[tag] = p.returncode
                print(f"[{time.time()-t0:6.0f}s] {'DONE' if p.returncode==0 else 'FAIL'} "
                      f"{name} {tag}", flush=True)
        for g in gpus:
            if slots[g] is None and pending:
                slots[g] = launch(pending.pop(0), g)
        time.sleep(10)
    if failed:
        print(f"  [{name}] FAILED: {failed}", flush=True)
    return failed


def sbc_from_dumps(arm_label):
    """Per-seed SBC ranks from the TARP dumps; returns per-param mean/std + min KS p."""
    out = {}
    for seed in (41, 42, 43):
        rs = []
        for f in sorted(glob.glob(f"{GC}/tarp_drp/dumps/{arm_label}_*/seed_{seed}/n*_m*/"
                                  "posterior_samples.npz")):
            z = np.load(f)
            rs.append((z["samples"] < z["theta"][:, None, :]).mean(axis=1))
        if rs:
            out[seed] = np.concatenate(rs, axis=0)
    if not out:
        return None
    from scipy import stats as st
    ranks = np.concatenate(list(out.values()), axis=0)   # (N, 6)
    return {"n": int(ranks.shape[0]),
            "mean": [float(m) for m in ranks.mean(axis=0)[:3]],
            "std": [float(s) for s in ranks.std(axis=0)[:3]],
            "min_ks_p": float(min(st.kstest(ranks[:, i], "uniform").pvalue for i in range(3)))}


def tarp_devs():
    """Max |ECP - alpha| per (arm, dim) from the coverage curve npz files."""
    devs = {}
    for f in sorted(glob.glob(f"{GC}/tarp_drp/curves/**/*.npz", recursive=True)):
        z = np.load(f)
        keys = set(z.files)
        a = z["alpha"] if "alpha" in keys else (z["alphas"] if "alphas" in keys else None)
        if "ecp_bootstrap" in keys:
            e = z["ecp_bootstrap"].mean(axis=0)
        else:
            e = z["ecp"] if "ecp" in keys else (z["coverage"] if "coverage" in keys else None)
        if a is None or e is None:
            continue
        name = Path(f).stem
        devs[name] = float(np.max(np.abs(np.asarray(e) - np.asarray(a))))
    return devs


def write_report(failures):
    L = ["# GATE C — BNT arms (derived verdicts)\n",
         "Validates the BNT campaign posteriors (FLATSKY_BNT_RESULT.md). Same machinery as "
         "the no-BNT gates (all of which passed).\n"]
    # SBC
    L += ["## SBC (ranks from the TARP dumps; science params)",
          "| arm | n | mean(Om,s8,w0) | std (uniform=0.289) | min KS p |", "|---|---|---|---|---|"]
    for probe, op, *_ in ARMS:
        s = sbc_from_dumps(f"bnt_{probe}_{op}")
        if s:
            L.append(f"| {probe} {op} | {s['n']} | "
                     + ",".join(f"{m:.3f}" for m in s["mean"]) + " | "
                     + ",".join(f"{v:.3f}" for v in s["std"]) + f" | {s['min_ks_p']:.3f} |")
        else:
            L.append(f"| {probe} {op} | — | — | — | — |")
    # TARP
    devs = tarp_devs()
    if devs:
        L += ["", "## TARP max |ECP − α| per curve (dim-3 = science subspace)",
              "| curve | max dev |", "|---|---|"]
        for k in sorted(devs):
            L.append(f"| {k} | {devs[k]:.3f} |")
    else:
        L += ["", f"TARP curves not parsed — see {GC}/tarp_drp/ (run_tarp_coverage output)."]
    # L-C2ST
    L += ["", "## L-C2ST (CNN arms; local at fiducial)",
          "| arm | reject@p<0.05 | median p | self-test (H0 p / H1 p) |", "|---|---|---|---|"]
    for probe, op, *_ in ARMS:
        if probe != "cnn":
            continue
        hits = glob.glob(f"{GC}/lc2st/bnt_{probe}_{op}/**/lc2st_summary.json", recursive=True)
        if hits:
            d = json.load(open(hits[0]))
            g = d.get("gate", {})
            L.append(f"| cnn {op} | {d.get('frac_reject_p05', float('nan'))*100:.0f}% | "
                     f"{d.get('median_p', float('nan')):.2f} | "
                     f"{g.get('st_h0_median_p', float('nan')):.2f} / "
                     f"{g.get('st_h1_median_p', float('nan')):.2f} |")
        else:
            L.append(f"| cnn {op} | — | — | — |")
    if failures:
        L += ["", f"⚠ FAILURES: {failures}"]
    L += ["", "Corner overlays (BNT vs no-BNT): `bnt_campaign/figures/corner_bnt_vs_nobnt_*.png`."]
    Path(GC, "GATE_C_BNT.md").write_text("\n".join(L))
    print(f"wrote {GC}/GATE_C_BNT.md", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gpus", default="1,2")
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    os.chdir(SBI)
    if args.dry_run:
        for fn in (corners_cmd, tarp_cmd):
            for arm in ARMS:
                print(f"\n# {fn.__name__} {arm[0]}_{arm[1]}:\n" + " ".join(fn(*arm, "<GPU>")))
        for arm in ARMS:
            if arm[0] == "cnn":
                print(f"\n# lc2st {arm[1]}:\n" + " ".join(lc2st_cmd(*arm, "<GPU>")))
        return
    os.makedirs(GC, exist_ok=True)
    t0 = time.time(); failures = {}
    failures.update(run_phase("corners", ARMS, corners_cmd, gpus))
    rc = subprocess.run([PY, "bnt_corner_overlays.py"], cwd=SBI).returncode
    if rc != 0:
        failures["overlays"] = rc
    failures.update(run_phase("tarp", ARMS, tarp_cmd, gpus))
    rc = subprocess.run([PY, "run_tarp_coverage.py", "--dumps-root", f"{GC}/tarp_drp/dumps",
                         "--outdir", f"{GC}/tarp_drp", "--dims", "3", "6"], cwd=SBI).returncode
    if rc != 0:
        failures["coverage"] = rc
    failures.update(run_phase("lc2st", [a for a in ARMS if a[0] == "cnn"], lc2st_cmd, gpus))
    write_report(failures)
    print(f"\n=== BNT GATE C done in {(time.time()-t0)/3600:.1f} h ===")


if __name__ == "__main__":
    main()
