#!/usr/bin/env python
"""Overnight rescue-menu + joint-statistics screening campaign (PLAN_OVERNIGHT_MENU.md).

Arms (screening: 1 MAF seed, 3000 obs; escalation: 3 seeds, 9000 obs, re-sweep only):
  A1  cov50 appended to BNT-L1 (P7 Gaussian-sector share)        [always escalated]
  A2  unions6 L1 blocks appended to BNT-L1 (survey practice)
  B1/B2  pair2d K=10 joint PDF, noBNT/BNT
  B3/B4  full4d K=5 joint PDF, noBNT/BNT (exact basis-covariance test)
  C1/C2  jointl1 K=10, noBNT/BNT (NEW: joint wavelet l1)
Escalation rule: screening FoM3 >= 1680 (0.7 x 2405); pairs escalate together.
Incremental OVERNIGHT_STATUS.md after every phase; derived OVERNIGHT_RESULT.md +
HANDOFF_OVERNIGHT at the end.

Detached:  setsid nohup python run_flatsky_overnight_menu.py --gpus 1 &
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = "/mnt/home/tersenov/software/cnn_sbi"
SBI = f"{REPO}/scripts/sbi"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
BASE = f"{SBI}/results/exploratory/flatsky_cross_2026_06"
OM = f"{BASE}/overnight_menu"
LOGS = f"{OM}/logs"
BNT_CACHE = f"{BASE}/bnt_campaign/l1_matrix/l1_none_cache/flat_local_none_bnt"
BNT_FID = f"{BASE}/bnt_campaign/fiducial_summaries/fiducial_summaries_l1_none.npz"
NOBNT_FOM, BNT_FOM = 2405.0, 364.0
ESCALATE_FOM = 0.7 * NOBNT_FOM

# arm name -> (build cmd builder, pair partner or None)
def _joint(name, stat, basis, k, append=False):
    cache = f"{OM}/{name}/cache"
    fid = f"{OM}/{name}/fiducial_summaries.npz"
    cmd = [PY, "build_flatsky_joint_arm.py", "--stat", stat, "--basis", basis,
           "--k", str(k), "--out-cache", cache, "--out-fid", fid]
    if append:
        cmd += ["--append-to", BNT_CACHE, "--append-fid", BNT_FID]
    return cmd, cache, fid


ARMS = {}  # name -> dict(build, cache, fid, pair, kind)
_c, _ca, _f = _joint("A1_cov_bnt", "cov", "bnt", 10, append=True)
ARMS["A1_cov_bnt"] = dict(build=_c, cache=_ca, fid=_f, pair=None, kind="rescue")
ARMS["A2_unions6_bnt"] = dict(
    build=[PY, "build_flatsky_bntdeep_arm.py", "--deep-mode", "unions6",
           "--out-cache", f"{OM}/A2_unions6_bnt/cache",
           "--out-fid", f"{OM}/A2_unions6_bnt/fiducial_summaries.npz"],
    cache=f"{OM}/A2_unions6_bnt/cache",
    fid=f"{OM}/A2_unions6_bnt/fiducial_summaries.npz",
    pair=None, kind="rescue")
for stat, k in (("pair2d", 10), ("full4d", 5), ("jointl1", 10)):
    for basis in ("nobnt", "bnt"):
        nm = f"{stat}_{basis}"
        c, ca, f = _joint(nm, stat, basis, k)
        ARMS[nm] = dict(build=c, cache=ca, fid=f,
                        pair=f"{stat}_{'bnt' if basis == 'nobnt' else 'nobnt'}",
                        kind="joint")


def sweep_cmd(name, arm, gpu, full):
    outdir = f"{OM}/{name}/population_sweep_full" if full else f"{OM}/{name}/population_sweep"
    seeds = "41,42,43" if full else "41"
    nobs = "9000" if full else "3000"
    return [PY, "population_sweep_flatsky.py",
            "--train-cache-dir", arm["cache"], "--cache-prefix", "l1",
            "--arm-label", f"overnight_{name}{'_full' if full else ''}",
            "--fiducial-summaries-npz", arm["fid"], "--output-dir", outdir,
            "--preproc-transform", "log1p-zscore", "--clip-value", "5",
            "--min-feature-variance", "1e-5",
            "--seeds", seeds, "--n-obs", nobs, "--max-perm", "50",
            "--m-samples", "2000", "--cuda-visible-devices", str(gpu)], outdir


def run_phase(name, cmd, gpu, t0):
    os.makedirs(LOGS, exist_ok=True)
    log_path = f"{LOGS}/{name}.log"
    print(f"[{time.time()-t0:7.0f}s] ===== {name} ===== (GPU{gpu})", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="3",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.40",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", CNN_CPU_THREADS="8",
               OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", CUDA_VISIBLE_DEVICES=str(gpu))
    with open(log_path, "w") as log:
        rc = subprocess.run(cmd, cwd=SBI, env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL).returncode
    print(f"[{time.time()-t0:7.0f}s] {'DONE' if rc == 0 else 'FAIL'} {name} (rc={rc})",
          flush=True)
    return rc == 0


def gpu_foreign_mb(gpu):
    """Total memory in use on `gpu` (anything present before OUR launch = foreign)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits", "-i", str(gpu)],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return 1 << 30


def med(outdir):
    f = Path(outdir) / "median_summary.json"
    return json.load(open(f)) if f.exists() else None


def status_append(line):
    with open(f"{OM}/OVERNIGHT_STATUS.md", "a") as fh:
        fh.write(line + "\n")
    print("STATUS: " + line, flush=True)


def write_result(results, escalated):
    L = ["# Overnight menu — derived results (PLAN_OVERNIGHT_MENU.md)", "",
         "Screening = 1 MAF seed, 3000 obs; full = 3 seeds, 9000 obs. Baselines: "
         f"L1 noBNT auto {NOBNT_FOM:.0f}, L1 BNT auto {BNT_FOM:.0f} (pooled medians).", "",
         "| arm | kind | screening FoM3 | recovered* | full FoM3 | full recovered |",
         "|---|---|---|---|---|---|"]
    for name, r in results.items():
        sc = r.get("screen"); fu = r.get("full")
        def rec(m):
            return (m["fom3"] - BNT_FOM) / (NOBNT_FOM - BNT_FOM) if m else None
        sc_s = f"{sc['fom3']:.0f}" if sc else "FAIL"
        rc_s = f"{rec(sc):.3f}" if sc else "—"
        fu_s = f"{fu['fom3']:.0f}" if fu else ("—" if name not in escalated else "FAIL")
        fr_s = f"{rec(fu):.3f}" if fu else "—"
        L.append(f"| {name} | {ARMS[name]['kind']} | {sc_s} | {rc_s} | {fu_s} | {fr_s} |")
    L += ["", "*recovered = (arm − BNT)/(noBNT − BNT); for joint/nobnt arms read it as "
          "FoM3 relative positioning, the meaningful number is FoM3 itself vs 2405.", ""]
    for stat in ("pair2d", "full4d", "jointl1"):
        a, b = results.get(f"{stat}_nobnt", {}), results.get(f"{stat}_bnt", {})
        for tag in ("screen", "full"):
            if a.get(tag) and b.get(tag):
                ratio = b[tag]["fom3"] / a[tag]["fom3"]
                L.append(f"- **{stat} basis-invariance ratio (BNT/noBNT, {tag}): "
                         f"{ratio:.3f}** (P4b predicts ≈1 for full4d only)")
    L += ["", "Registered readings: A1 recovered = Gaussian-sector share of the l1's loss; "
          "A2 ≥0.95 expected (span); full4d ratio ≈1 is the exact-covariance test; pair2d/"
          "jointl1 ratios measure the pairwise approximation's basis fragility.",
          "", "NaN/failure notes and logs: overnight_menu/logs/."]
    Path(OM, "OVERNIGHT_RESULT.md").write_text("\n".join(L) + "\n")
    print(f"wrote {OM}/OVERNIGHT_RESULT.md", flush=True)


def write_handoff(results, escalated, t0):
    txt = f"""# HANDOFF — overnight menu run (2026-06-12 morning)

Run time {(time.time()-t0)/3600:.1f} h. Read `overnight_menu/OVERNIGHT_RESULT.md` (derived
tables incl. basis-invariance ratios) and `OVERNIGHT_STATUS.md` (chronology). Plan +
registered predictions: PLAN_OVERNIGHT_MENU.md. Arms that failed have logs in
overnight_menu/logs/<arm>_*.log.

Open follow-ups queued for Andreas:
- If joint arms underperform: the dimensionality question — VMIM-MLP compression of the
  joint datavectors (the l1vmim pattern in this repo) is the agreed next tool.
- Escalated arms: {sorted(escalated) or 'none'}.
- Writeups (deep-dive §4.3/§1.7/§1.8 updates + FLATSKY_BNT_RESULT) intentionally left for
  the interactive session — numbers should be discussed before the docs move.
"""
    Path(OM, "HANDOFF_OVERNIGHT_2026-06-12.md").write_text(txt)
    print("wrote handoff", flush=True)


FOREIGN_LIMIT_MB = 12000   # back off a GPU if a tenant holds more than this


def _polite_wait(gpu, t0):
    waited = 0
    while gpu_foreign_mb(gpu) > FOREIGN_LIMIT_MB and waited < 3600:
        print(f"[{time.time()-t0:7.0f}s] GPU{gpu} busy (foreign > {FOREIGN_LIMIT_MB} MB) — "
              "waiting 300 s", flush=True)
        time.sleep(300); waited += 300


def _screen_worker(gpu, q, results, lock, t0):
    while True:
        try:
            name = q.get_nowait()
        except queue.Empty:
            return
        _polite_wait(gpu, t0)
        arm = ARMS[name]
        if not run_phase(f"{name}_build", arm["build"], gpu, t0):
            with lock:
                status_append(f"- {name}: BUILD FAIL (GPU{gpu})")
            continue
        cmd, outdir = sweep_cmd(name, arm, gpu, full=False)
        if not run_phase(f"{name}_screen", cmd, gpu, t0):
            with lock:
                status_append(f"- {name}: SCREEN SWEEP FAIL (GPU{gpu})")
            continue
        m = med(outdir)
        with lock:
            results[name]["screen"] = m
            rec = (m["fom3"] - BNT_FOM) / (NOBNT_FOM - BNT_FOM) if m else float("nan")
            status_append(f"- {name}: screening FoM3 {m['fom3']:.0f} "
                          f"(recovered-equiv {rec:.3f}) [GPU{gpu}]")


def _full_worker(gpu, q, results, lock, t0):
    while True:
        try:
            name = q.get_nowait()
        except queue.Empty:
            return
        _polite_wait(gpu, t0)
        cmd, outdir = sweep_cmd(name, ARMS[name], gpu, full=True)
        if run_phase(f"{name}_full", cmd, gpu, t0):
            with lock:
                results[name]["full"] = med(outdir)
                status_append(f"- {name}: FULL FoM3 {results[name]['full']['fom3']:.0f} "
                              f"[GPU{gpu}]")
        else:
            with lock:
                status_append(f"- {name}: FULL SWEEP FAIL (GPU{gpu})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="1,0,2")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    if a.dry_run:
        for n, arm in ARMS.items():
            print(n, ":", " ".join(arm["build"]))
        return 0
    t0 = time.time()
    os.makedirs(OM, exist_ok=True)
    status_append(f"## run started {time.strftime('%F %T')} (GPUs {gpus})")
    results = {n: {} for n in ARMS}
    lock = threading.Lock()

    q1 = queue.Queue()
    for name in ARMS:
        q1.put(name)
    threads = [threading.Thread(target=_screen_worker, args=(g, q1, results, lock, t0),
                                daemon=True) for g in gpus]
    [th.start() for th in threads]
    [th.join() for th in threads]

    escalated = set()
    for name, arm in ARMS.items():
        m = results[name].get("screen")
        qualify = (name == "A1_cov_bnt") or (m and m["fom3"] >= ESCALATE_FOM)
        if qualify:
            escalated.add(name)
            if arm["pair"]:
                escalated.add(arm["pair"])
    status_append(f"- escalation set: {sorted(escalated)}")
    q2 = queue.Queue()
    for name in sorted(escalated):
        if results[name].get("screen"):
            q2.put(name)
    threads = [threading.Thread(target=_full_worker, args=(g, q2, results, lock, t0),
                                daemon=True) for g in gpus]
    [th.start() for th in threads]
    [th.join() for th in threads]

    write_result(results, escalated)
    write_handoff(results, escalated, t0)
    status_append(f"## run complete {time.strftime('%F %T')} "
                  f"({(time.time()-t0)/3600:.1f} h)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
