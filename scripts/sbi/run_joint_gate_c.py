#!/usr/bin/env python
"""GATE C for the overnight joint-statistic arms (PLAN_GATE_C_JOINT.md).

Adapted from run_bnt_gate_c.py, minus the corners phase (joint-arm corners already
exist in overnight_menu/corners/) and minus L-C2ST (underpowered at 3000-dim —
reference_lc2st_underpowered_highdim_l1; TARP+SBC accepted as GATE C for high-dim arms).

Phases:
  0. preflight — assert each arm's cache meta (dequantize=True, stat/basis/k/snr_range)
                 and the train/val shapes WITHOUT decompressing the big arrays
  1. tarp      — tarp_stratified_val.py per arm (600 val points, FoM3 terciles, 3 seeds;
                 flags mirror the full-rigor sweeps: log1p-zscore / clip 5 / min-var 1e-5 /
                 seeds 41,42,43 / epochs 50000 / batch 256 / lr 1e-4)
  2. coverage  — run_tarp_coverage.py (dims 3 6) on the dumps
  3. report    — GATE_C_JOINT.md with verdicts DERIVED from the registered bands
                 (PLAN_GATE_C_JOINT.md §5): PASS / PASS-with-caveat / FAIL + the
                 comparative-downgrade flag for the noBNT "equal-or-better" claim.

--gpus takes a SLOT list, e.g. "1,1,2,2" = two concurrent jobs on each of GPUs 1 and 2.
Launch detached:
  (cd .../scripts/sbi && setsid nohup <jaxili-python> run_joint_gate_c.py --gpus 1,1,2,2 \
     > .../overnight_menu/gate_c/driver.out 2>&1 &)
"""
import argparse, glob, json, os, subprocess, time, zipfile
import numpy as np
from numpy.lib import format as npformat
from pathlib import Path

REPO = "/mnt/home/tersenov/software/cnn_sbi"; SBI = f"{REPO}/scripts/sbi"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
FC = f"{SBI}/results/exploratory/flatsky_cross_2026_06"
OM = f"{FC}/overnight_menu"
GC = f"{OM}/gate_c"
LOGS = f"{GC}/logs"

# arm -> expected (stat, basis); cache dir is OM/<arm>/cache
ARMS = [
    ("pair2dq_nobnt", "pair2d", "nobnt"),
    ("jointl1q_nobnt", "jointl1", "nobnt"),
    ("pair2dq_bnt", "pair2d", "bnt"),
    ("jointl1q_bnt", "jointl1", "bnt"),
]
TERCILES = ("LOW", "MID", "HIGH")
# Registered verdict bands (PLAN_GATE_C_JOINT.md §5)
DEV_PASS, DEV_CAVEAT = 0.05, 0.10
STD_LO, STD_HI, STD_FAIL_MARGIN = 0.275, 0.305, 0.02


def npz_member_shape(path, key):
    """Shape+dtype of one npz member from its header (no decompression of the data)."""
    with zipfile.ZipFile(path) as z:
        with z.open(f"{key}.npy") as f:
            version = npformat.read_magic(f)
            shape, _fortran, dtype = npformat._read_array_header(f, version)
    return shape, dtype


def preflight():
    print("===== PHASE preflight =====", flush=True)
    for arm, stat, basis in ARMS:
        cdir = Path(OM, arm, "cache")
        meta = np.load(cdir / "l1_cache_meta.npz", allow_pickle=True)
        assert str(meta["stat"]) == stat, (arm, "stat", str(meta["stat"]))
        assert str(meta["basis"]) == basis, (arm, "basis", str(meta["basis"]))
        assert bool(meta["dequantize"]), (arm, "dequantize must be True")
        assert int(meta["k"]) == 10 and float(meta["snr_range"]) == 5.0, (arm, "k/snr_range")
        for split in ("l1_train", "l1_val"):
            ts, _ = npz_member_shape(cdir / f"{split}.npz", "theta")
            xs, _ = npz_member_shape(cdir / f"{split}.npz", "x")
            assert ts[1] == 6 and xs[1] == 3000 and ts[0] == xs[0], (arm, split, ts, xs)
        print(f"  {arm}: meta OK (stat={stat} basis={basis} dq=True), shapes OK", flush=True)


def tarp_cmd(arm, stat, basis, gpu):
    return [PY, "tarp_stratified_val.py",
            "--train-cache-dir", f"{OM}/{arm}/cache", "--cache-prefix", "l1",
            "--arm-label", arm, "--dumps-root", f"{GC}/tarp_drp/dumps",
            "--preproc-transform", "log1p-zscore", "--clip-value", "5",
            "--min-feature-variance", "1e-5", "--seeds", "41,42,43",
            "--cuda-visible-devices", str(gpu)]


def run_phase(name, jobs, cmd_fn, slots_gpus, mem_fraction):
    print(f"\n===== PHASE {name} =====", flush=True)
    os.makedirs(LOGS, exist_ok=True)
    pending = list(jobs); slots = [None] * len(slots_gpus); t0 = time.time(); failed = {}

    def launch(job, gpu):
        tag = job[0]
        log = open(f"{LOGS}/{name}_{tag}.log", "w")
        env = dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="3",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false",
                   XLA_PYTHON_CLIENT_MEM_FRACTION=str(mem_fraction),
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", CNN_CPU_THREADS="8")
        p = subprocess.Popen(cmd_fn(*job, gpu), cwd=SBI, env=env, stdout=log,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        print(f"[{time.time()-t0:6.0f}s] LAUNCH {name} {tag} GPU{gpu} (pid {p.pid})", flush=True)
        return (job, p, log)

    while pending or any(slots):
        for i, gpu in enumerate(slots_gpus):
            s = slots[i]
            if s and s[1].poll() is not None:
                job, p, log = s; log.close(); slots[i] = None
                if p.returncode != 0:
                    failed[job[0]] = p.returncode
                print(f"[{time.time()-t0:6.0f}s] {'DONE' if p.returncode==0 else 'FAIL'} "
                      f"{name} {job[0]}", flush=True)
        for i, gpu in enumerate(slots_gpus):
            if slots[i] is None and pending:
                slots[i] = launch(pending.pop(0), gpu)
        time.sleep(10)
    if failed:
        print(f"  [{name}] FAILED: {failed}", flush=True)
    return failed


def sbc_from_dumps(arm):
    """Pooled SBC ranks (all terciles, all seeds) from the TARP dumps; science params."""
    rs = []
    for f in sorted(glob.glob(f"{GC}/tarp_drp/dumps/{arm}_*/seed_*/n*_m*/posterior_samples.npz")):
        z = np.load(f)
        rs.append((z["samples"] < z["theta"][:, None, :]).mean(axis=1))
    if not rs:
        return None
    from scipy import stats as st
    ranks = np.concatenate(rs, axis=0)   # (N, 6)
    return {"n": int(ranks.shape[0]),
            "mean": [float(m) for m in ranks.mean(axis=0)[:3]],
            "std": [float(s) for s in ranks.std(axis=0)[:3]],
            "min_ks_p": float(min(st.kstest(ranks[:, i], "uniform").pvalue for i in range(3)))}


def tarp_signed_devs(arm, dim=3):
    """Per tercile: signed ECP−α at the worst-|dev| point, worst seed (bootstrap-mean ECP)."""
    out = {}
    for terc in TERCILES:
        worst = None
        for f in sorted(glob.glob(f"{GC}/tarp_drp/curves/tarp_curve_{arm}_{terc}_seed*_dim{dim}.npz")):
            z = np.load(f)
            a = np.asarray(z["alpha"]); e = np.asarray(z["ecp_bootstrap"]).mean(axis=0)
            i = int(np.argmax(np.abs(e - a)))
            d = float(e[i] - a[i])
            if worst is None or abs(d) > abs(worst):
                worst = d
        if worst is not None:
            out[terc] = worst
    return out


def derive_verdict(devs, sbc):
    worst_abs = max(abs(d) for d in devs.values()) if devs else float("nan")
    stds = sbc["std"] if sbc else []
    std_off = max((max(0.0, STD_LO - s, s - STD_HI) for s in stds), default=float("nan"))
    if not devs or not sbc:
        return "INCOMPLETE", worst_abs, std_off
    if worst_abs > DEV_CAVEAT or std_off >= STD_FAIL_MARGIN:
        return "FAIL", worst_abs, std_off
    if worst_abs <= DEV_PASS and std_off == 0.0:
        return "PASS", worst_abs, std_off
    return "PASS-with-caveat", worst_abs, std_off


def comparative_flag(devs, sbc):
    """noBNT 'equal-or-better than l1+product' downgrade test (PLAN §5 sensitivity note)."""
    min_signed = min(devs.values()) if devs else 0.0
    max_std = max(sbc["std"]) if sbc else 0.0
    return (min_signed <= -0.05) or (max_std >= 0.30)


def write_report(failures):
    L = ["# GATE C — joint-statistic arms (derived verdicts)\n",
         "Validates the overnight joint-stat posteriors (OVERNIGHT_RESULT.md addenda) per",
         "PLAN_GATE_C_JOINT.md. Machinery = the BNT gate's TARP+SBC; L-C2ST skipped",
         "(3000-dim, underpowered). NDE retrains mirror the full-rigor sweeps exactly",
         "(log1p-zscore / clip 5 / min-var 1e-5 / MAF seeds 41,42,43 / epochs 50000 /",
         "batch 256 / lr 1e-4). 600 val points, m=2000, FoM3-tercile stratified.\n"]
    sbc = {arm: sbc_from_dumps(arm) for arm, *_ in ARMS}
    devs = {arm: tarp_signed_devs(arm, dim=3) for arm, *_ in ARMS}

    L += ["## SBC (ranks pooled from the TARP dumps; science params)",
          "| arm | n | mean(Om,s8,w0) | std (uniform=0.289) | min KS p |", "|---|---|---|---|---|"]
    for arm, *_ in ARMS:
        s = sbc[arm]
        L.append(f"| {arm} | {s['n']} | " + ",".join(f"{m:.3f}" for m in s["mean"]) + " | "
                 + ",".join(f"{v:.3f}" for v in s["std"]) + f" | {s['min_ks_p']:.3f} |"
                 if s else f"| {arm} | — | — | — | — |")

    L += ["", "## TARP (dim-3 science subspace; signed max ECP − α, bootstrap-mean curve,",
          "worst seed per FoM3 tercile; positive = conservative, negative = over-confident)",
          "| arm | HIGH (tightest) | MID | LOW |", "|---|---|---|---|"]
    for arm, *_ in ARMS:
        d = devs[arm]
        L.append("| " + arm + " | " + " | ".join(
            (f"{d[t]:+.3f}" if t in d else "—") for t in ("HIGH", "MID", "LOW")) + " |")
    L += ["", "(no-BNT l1 reference from the flat-local gate: load-bearing arms |dev| <= 0.037;",
          "dim-6 curves in gate_c/tarp_drp/curves/.)"]

    L += ["", "## Verdicts (registered bands, PLAN_GATE_C_JOINT.md §5 — derived, not asserted)",
          f"Bands: PASS = all terciles |dev| <= {DEV_PASS} AND SBC std in [{STD_LO}, {STD_HI}];",
          f"PASS-with-caveat = worst |dev| in ({DEV_PASS}, {DEV_CAVEAT}] or std outside by < {STD_FAIL_MARGIN};",
          f"FAIL = |dev| > {DEV_CAVEAT} or std off by >= {STD_FAIL_MARGIN}.",
          "", "| arm | worst |dev| | SBC std excess | verdict |", "|---|---|---|---|"]
    verdicts = {}
    for arm, *_ in ARMS:
        v, worst_abs, std_off = derive_verdict(devs[arm], sbc[arm])
        verdicts[arm] = v
        L.append(f"| {arm} | {worst_abs:.3f} | {std_off:.3f} | **{v}** |")

    L += ["", "## Comparative check for the noBNT headline (registered sensitivity note)",
          "The 'marginals equal-or-better than l1+product' claim rests on a ~4% sigma_s8 edge",
          "over a comparator gated at |dev| <= 0.037. Downgrade trigger (derived): min signed",
          "dev <= -0.05 (systematic over-confidence) OR any science-param SBC std >= 0.30."]
    for arm, *_ in ARMS:
        if not arm.endswith("_nobnt"):
            continue
        flag = comparative_flag(devs[arm], sbc[arm])
        L.append(f"- {arm}: {'DOWNGRADE to comparable' if flag else 'no downgrade'} "
                 f"(min signed dev {min(devs[arm].values()):+.3f}, "
                 f"max std {max(sbc[arm]['std']):.3f})" if devs[arm] and sbc[arm]
                 else f"- {arm}: INCOMPLETE")

    L += ["", "## Registered-prediction adjudication",
          "- P-G1 (noBNT arms land like the gated l1 noBNT arms, PASS clean): "
          + ("HOLDS" if all(verdicts[a] == "PASS" for a, *_ in ARMS if a.endswith("_nobnt"))
             else "DOES NOT HOLD — see verdict table"),
          "- P-G2 (BNT-side arms PASS-with-caveat, worst |dev| in (0.05, 0.10]): "
          + ("HOLDS" if all(verdicts[a] == "PASS-with-caveat" for a, *_ in ARMS if a.endswith("_bnt"))
             else "DOES NOT HOLD — see verdict table")]
    if failures:
        L += ["", f"FAILURES: {failures}"]
    L += ["", "Corners (pre-existing, morning session): overnight_menu/corners/ + figures/."]
    Path(GC, "GATE_C_JOINT.md").write_text("\n".join(L) + "\n")
    print(f"wrote {GC}/GATE_C_JOINT.md", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gpus", default="1,1,2,2",
                    help="slot list; repeated GPU id = concurrent jobs on that GPU")
    ap.add_argument("--mem-fraction", type=float, default=0.40)
    ap.add_argument("--report-only", action="store_true",
                    help="skip tarp/coverage, just rebuild the report from existing artifacts")
    args = ap.parse_args()
    slots_gpus = [int(g) for g in args.gpus.split(",")]
    os.chdir(SBI)
    preflight()
    if args.dry_run:
        for arm in ARMS:
            print(f"\n# tarp {arm[0]}:\n" + " ".join(tarp_cmd(*arm, "<GPU>")))
        return
    os.makedirs(GC, exist_ok=True)
    t0 = time.time(); failures = {}
    if not args.report_only:
        failures.update(run_phase("tarp", ARMS, tarp_cmd, slots_gpus, args.mem_fraction))
        rc = subprocess.run([PY, "run_tarp_coverage.py", "--dumps-root", f"{GC}/tarp_drp/dumps",
                             "--outdir", f"{GC}/tarp_drp", "--dims", "3", "6"],
                            cwd=SBI).returncode
        if rc != 0:
            failures["coverage"] = rc
    write_report(failures)
    print(f"\n=== JOINT GATE C done in {(time.time()-t0)/60:.0f} min ===")


if __name__ == "__main__":
    main()
