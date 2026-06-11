#!/usr/bin/env python
"""WHITENING test (the BNT-inflation decomposition diagnostic; PAPER_BNT_* Part II).

Per-channel L1 in the noise-whitened BNT basis Q = (BB^T)^(-1/2) B (orthogonal => an
orthogonal rotation of the ORIGINAL basis with independent, equal-variance noise restored).
Decomposes the L1 BNT inflation: noBNT (2405/2875) vs WHITENED (?) vs BNT (364/637) —
whatever whitening recovers was the noise-correlation/per-map-S/N component; whatever gap
remains to noBNT is the irreducibly-joint (rotation-mixed) component. L1 arms only (the CNN
is basis-invariant by class closure). DIAGNOSTIC: this basis remixes the nulled kernels, so
it is NOT a practical post-BNT analysis recipe — pure information accounting.

Phases: sigma freeze (--mode whiten) -> L1 both-whiten build (solo) -> {none, product}
slices -> fiducial precompute + per-arm slice -> 2 jit sweeps -> derived WHITEN_RESULT.md.
Detached:  setsid nohup python run_flatsky_whiten_campaign.py &     --dry-run prints.
"""
import argparse, json, os, subprocess, sys, time
import numpy as np
from pathlib import Path

REPO = "/mnt/home/tersenov/software/cnn_sbi"; SBI = f"{REPO}/scripts/sbi"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
TFDS = "nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180"
DDIR = "/home/tersenov/tensorflow_datasets"
FID = f"{SBI}/results/exploratory/cross_maps_campaign/full_sphere_cache_fiducial_10deg"
BASE = f"{SBI}/results/exploratory/flatsky_cross_2026_06"
WH = f"{BASE}/whiten_campaign"
LOGS = f"{WH}/logs"
SIGMA_WH = f"{BASE}/flatsky_cross_noise_sigma_whiten.npz"
L1M = f"{WH}/l1_matrix"
BOTH_CACHE_WH = f"{L1M}/l1_both_cache/flat_local_both_whiten"
FID_BOTH_WH = f"{WH}/fiducial_both_datavectors_whiten.npz"
OPS = ["none", "product"]
OBS_PERM, OBS_PATCH = 0, 90
FEAT_PER_CH = 5 * 40


def freeze_cmd(gpu):
    return [PY, "freeze_flatsky_cross_noise.py", "--mode", "whiten", "--out-dir", BASE]


def l1_cmd(op, gpu, use_both_cache):
    d = f"{L1M}/l1_{op}_s41"
    cmd = [PY, "npe_l1norm_cross_jaxili_nbody_tomo.py",
           "--cross-maps-route", "flat_local", "--cross-op", op,
           "--flatsky-channel-mix", "whiten",
           "--cross-tfds-name", TFDS, "--cross-tfds-data-dir", DDIR,
           "--fiducial-obs-cache-dir", FID, "--flatsky-cross-sigma", SIGMA_WH,
           "--pca-components", "0",
           "--nde-perm-split", "5-6", "--nde-val-perm-split", "0-1",
           "--nbins", "4", "--tomo-bin-indices", "1,2,3,4",
           "--field-size", "10", "--field-npix", "80",
           "--n-scales", "5", "--l1-nbins", "40",
           "--harmonic-calibration-realizations", "20", "--ds-batch-size", "512",
           "--total-steps", "5000",
           "--summary-transform", "log1p-zscore", "--clip-value", "5", "--seed", "41",
           "--harmonic-obs-perm", str(OBS_PERM), "--harmonic-obs-patch-idx", str(OBS_PATCH),
           "--no-wandb", "--cuda-visible-devices", str(gpu),
           "--save-dir", d, "--cache-dir", f"{L1M}/l1_{op}_cache",
           "--posterior-out", f"{d}/posterior.npy", "--figure-out", f"{d}/corner.pdf"]
    if use_both_cache:
        cmd += ["--flatsky-both-cache", BOTH_CACHE_WH]
    return cmd


def l1_fid_cmd(gpu):
    return [PY, "precompute_fiducial_both_datavectors.py", "--mode", "whiten",
            "--both-cache", BOTH_CACHE_WH, "--sigma", SIGMA_WH, "--out", FID_BOTH_WH]


def sweep_cmd(op, gpu):
    return [PY, "population_sweep_flatsky.py",
            "--train-cache-dir", f"{L1M}/l1_{op}_cache/flat_local_{op}_whiten",
            "--cache-prefix", "l1", "--arm-label", f"whiten_l1_{op}",
            "--fiducial-summaries-npz", f"{WH}/fiducial_summaries/fiducial_summaries_l1_{op}.npz",
            "--output-dir", f"{WH}/population_sweep/l1_{op}",
            "--preproc-transform", "log1p-zscore", "--clip-value", "5",
            "--min-feature-variance", "1e-5",
            "--seeds", "41,42,43", "--n-obs", "9000", "--max-perm", "50",
            "--m-samples", "2000", "--cuda-visible-devices", str(gpu)]


def slice_l1_fiducials():
    sys.path.insert(0, SBI)
    import flatsky_cross_l1 as fxl
    z = np.load(FID_BOTH_WH)
    outdir = Path(WH) / "fiducial_summaries"; outdir.mkdir(parents=True, exist_ok=True)
    for op in OPS:
        cols = fxl.op_feature_columns(op, 4, FEAT_PER_CH)
        out = outdir / f"fiducial_summaries_l1_{op}.npz"
        np.savez(out, S=z["x"][:, cols], perm=z["perm"], patch=z["patch"],
                 theta=z["truth"], truth=z["truth"], mode="whiten")
        print(f"  sliced {op} -> {out}", flush=True)


def run_phase(name, jobs, cmd_fn, gpus):
    print(f"\n===== PHASE {name} =====", flush=True)
    os.makedirs(LOGS, exist_ok=True)
    pending = list(jobs); slots = {g: None for g in gpus}; t0 = time.time(); failed = {}

    def launch(job, gpu):
        tag = "_".join(str(x) for x in job) if job else name
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
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        p = subprocess.Popen(c, cwd=SBI, env=env, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL)
        print(f"[{time.time()-t0:6.0f}s] LAUNCH {name} {tag} GPU{gpu} (pid {p.pid})", flush=True)
        return (job, p, log)

    while pending or any(slots.values()):
        for g in gpus:
            s = slots[g]
            if s and s[1].poll() is not None:
                job, p, log = s; log.close(); slots[g] = None
                tag = "_".join(str(x) for x in job) if job else name
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


def write_report(failures):
    def med(path):
        f = Path(path) / "median_summary.json"
        return json.load(open(f))["fom3"] if (Path(path) / "median_summary.json").exists() else None

    refs = {
        ("none", "nobnt"): med(f"{BASE}/population_sweep/flat_none"),
        ("none", "bnt"): med(f"{BASE}/bnt_campaign/population_sweep/l1_none"),
        ("none", "whiten"): med(f"{WH}/population_sweep/l1_none"),
        ("product", "nobnt"): med(f"{BASE}/population_sweep/flat_product"),
        ("product", "bnt"): med(f"{BASE}/bnt_campaign/population_sweep/l1_product"),
        ("product", "whiten"): med(f"{WH}/population_sweep/l1_product"),
    }
    L = ["# Whitening test — decomposing the L1 BNT inflation\n",
         "Per-channel L1 in Q = (BB^T)^(-1/2) B (noise-whitened BNT = orthogonal rotation of "
         "the original basis). Pooled 3-MAF 9000-obs median FoM3. DIAGNOSTIC basis (remixes "
         "the nulled kernels — not a practical recipe).\n",
         "| arm | no-BNT | whitened | BNT | whiten/noBNT | recovered fraction* |",
         "|---|---|---|---|---|---|"]
    for op in OPS:
        n, w, b = refs[(op, "nobnt")], refs[(op, "whiten")], refs[(op, "bnt")]
        if all(v for v in (n, w, b)):
            rec = (w - b) / (n - b)
            L.append(f"| L1 {op} | {n:.0f} | {w:.0f} | {b:.0f} | {w/n:.2f}× | {rec:.0%} |")
        else:
            L.append(f"| L1 {op} | {n or '—'} | {w or '—'} | {b or '—'} | — | — |")
    L += ["", "*recovered fraction = (whiten − BNT) / (noBNT − BNT) in FoM3.",
          ""]
    n, w, b = refs[("none", "nobnt")], refs[("none", "whiten")], refs[("none", "bnt")]
    if all(v for v in (n, w, b)):
        rec = (w - b) / (n - b)
        if rec > 0.8:
            verdict = ("**Verdict:** whitening recovers most of the collapse ⇒ the BNT "
                       "inflation is DOMINANTLY the noise-correlation / per-map-S/N (basis) "
                       "component; the irreducibly-joint share is small.")
        elif rec > 0.4:
            verdict = ("**Verdict:** whitening recovers a substantial part but a large "
                       "irreducibly-joint component remains (information genuinely in the "
                       "cross-channel dependence that no per-channel basis choice restores).")
        else:
            verdict = ("**Verdict:** whitening recovers little ⇒ the inflation is dominated "
                       "by the irreducibly-joint component, not the noise basis.")
        L += [verdict]
    else:
        L += ["**Verdict:** INCOMPLETE — missing sweeps."]
    if failures:
        L += ["", f"⚠ FAILURES: {failures}"]
    Path(WH, "WHITEN_RESULT.md").write_text("\n".join(L))
    print(f"wrote {WH}/WHITEN_RESULT.md", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gpus", default="1,2")
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    os.chdir(SBI)
    if args.dry_run:
        print("# P0:\n" + " ".join(freeze_cmd("<G>")))
        print("\n# P1 both:\n" + " ".join(l1_cmd("both", "<G>", False)))
        for op in OPS:
            print(f"\n# P2 {op}:\n" + " ".join(l1_cmd(op, "<G>", True)))
        print("\n# P3 fid:\n" + " ".join(l1_fid_cmd("<G>")))
        for op in OPS:
            print(f"\n# P4 sweep {op}:\n" + " ".join(sweep_cmd(op, "<G>")))
        return
    os.makedirs(WH, exist_ok=True)
    t0 = time.time(); failures = {}
    if not Path(SIGMA_WH).exists():
        failures.update(run_phase("sigma", [()], lambda gpu: freeze_cmd(gpu), gpus[:1]))
    else:
        print(f"P0 sigma: {SIGMA_WH} exists, skipping.")
    failures.update(run_phase("both_build", [("both",)],
                              lambda op, gpu: l1_cmd(op, gpu, False), gpus[:1]))
    failures.update(run_phase("l1_arms", [(op,) for op in OPS],
                              lambda op, gpu: l1_cmd(op, gpu, True), gpus))
    failures.update(run_phase("l1_fid", [()], lambda gpu: l1_fid_cmd(gpu), gpus[:1]))
    try:
        slice_l1_fiducials()
    except Exception as exc:
        failures["slice"] = str(exc)
        print(f"  SLICE FAILED: {exc}", flush=True)
    failures.update(run_phase("sweep", [(op,) for op in OPS], sweep_cmd, gpus))
    write_report(failures)
    print(f"\n=== whiten campaign done in {(time.time()-t0)/3600:.1f} h ===")


if __name__ == "__main__":
    main()
