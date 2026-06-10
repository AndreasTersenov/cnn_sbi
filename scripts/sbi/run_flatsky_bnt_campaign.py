#!/usr/bin/env python
"""BNT flat-local campaign (paper pillar 2): does BNT inflate L1 but not the CNN?

Prediction ladder (NEXT_THREADS_PLAN_2026-06-10.md §B; inflation = FoM3_BNT / FoM3_noBNT,
pooled 3-MAF 9000-obs median, against the existing no-BNT arms):
  1. L1 auto-only inflates significantly (per-channel stat, blind to BNT's cross-bin noise),
  2. L1 auto+product inflates less (explicit cross channel restores some of it),
  3. CNN ~no inflation (channels in, VMIM extracts cross-bin info implicitly => BNT lossless).

Arms: L1 {none, product} BNT (both-build-once-slice, matching the no-BNT methodology) +
CNN {none, product} x compressor seeds {41,42,43} BNT (recipe-matched to the no-BNT baseline:
plain CNN, 80k steps, val-batches 1 — the multiseed lesson says never headline a
single-compressor-seed cross claim). Sweeps use the jitted sampler.

Phases: sigma-freeze (--bnt, once) -> L1 both-BNT build (solo) -> L1 arm slices + CNN
compressors -> fiducial summaries (CNN fidsumm x6, L1 precompute + per-arm slice) ->
8 population sweeps -> derived inflation report (BNT_CAMPAIGN_RESULT.md; verdicts computed
from the medians on disk, never asserted).

Detached launch (after Andreas's go):  setsid nohup python run_flatsky_bnt_campaign.py &
--dry-run prints every command without running anything.
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
BNT = f"{BASE}/bnt_campaign"
LOGS = f"{BNT}/logs"
SIGMA_BNT = f"{BASE}/flatsky_cross_noise_sigma_bnt.npz"
L1M = f"{BNT}/l1_matrix"
BOTH_CACHE_BNT = f"{L1M}/l1_both_cache/flat_local_both_bnt"   # cache-dir + script's op+bnt suffix
FID_BOTH_BNT = f"{BNT}/fiducial_both_datavectors_bnt.npz"
OPS = ["none", "product"]
CNN_SEEDS = [41, 42, 43]
OBS_PERM, OBS_PATCH = 0, 90
FEAT_PER_CH = 5 * 40    # n_scales * l1_nbins


def freeze_cmd(gpu):
    return [PY, "freeze_flatsky_cross_noise.py", "--bnt", "--out-dir", BASE]


def l1_cmd(op, gpu, use_both_cache):
    d = f"{L1M}/l1_{op}_s41"
    cmd = [PY, "npe_l1norm_cross_jaxili_nbody_tomo.py",
           "--cross-maps-route", "flat_local", "--cross-op", op, "--apply-bnt",
           "--cross-tfds-name", TFDS, "--cross-tfds-data-dir", DDIR,
           "--fiducial-obs-cache-dir", FID, "--flatsky-cross-sigma", SIGMA_BNT,
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
        cmd += ["--flatsky-both-cache", BOTH_CACHE_BNT]
    return cmd


def cnn_compressor_cmd(op, seed, gpu):
    # Recipe-matched to the no-BNT baseline (plain CNN, 80k, val-batches 1): the inflation
    # ratio must compare like-trained arms. The 160k recipe is a separate ablation.
    d = f"{BNT}/cnn_{op}_s{seed}"
    return [PY, "npe_cnn_nbody_tomo.py", "--train-compressor", "--exit-after-compress",
            "--cnn-map-route", "flat_local", "--cross-op", op, "--flatsky-bnt",
            "--cross-tfds-name", TFDS, "--cross-tfds-data-dir", DDIR, "--fiducial-obs-cache", FID,
            "--harmonic-cache-regime", "nobnt", "--harmonic-normalize-input-channels",
            "--cnn-perm-split", "0-4:5-6", "--zero-mean-maps", "--map-kind", "nbody",
            "--seed", str(seed), "--field-size", "10", "--field-npix", "80",
            "--nbins", "4", "--tomo-bin-indices", "1,2,3,4",
            "--compressor-arch", "plain", "--compressor-dim", "10", "--compressor-dense-width", "256",
            "--compressor-conv-channels", "64,128,256", "--compressor-steps", "80000",
            "--compressor-batch-size", "128", "--compressor-lr", "0.0005",
            "--compressor-checkpoint-policy", "best_val", "--compressor-val-batches", "1",
            "--no-wandb",
            "--harmonic-obs-perm", str(OBS_PERM), "--harmonic-obs-patch-idx", str(OBS_PATCH),
            "--cuda-visible-devices", str(gpu), "--save-dir", d, "--cache-dir", f"{d}/cache",
            "--posterior-out", f"{d}/posterior.npy", "--figure-out", f"{d}/corner.pdf"]


def cnn_fidsumm_cmd(op, seed, gpu):
    d = f"{BNT}/cnn_{op}_s{seed}"
    meta = dict(np.load(f"{d}/cache/cnn_cache_meta.npz", allow_pickle=True))
    return [PY, "build_fiducial_summaries_cnn.py", "--arm-label", f"bnt_{op}_s{seed}",
            "--params-pkl", str(meta["compressor_params_path"]),
            "--state-pkl", str(meta["compressor_state_path"]),
            "--expect-params-sha", str(meta["compressor_params_sha256"]),
            "--expect-state-sha", str(meta["compressor_state_sha256"]),
            "--n-channels", str(int(meta["cnn_input_channels"])), "--dim", "10",
            "--conv-channels", "64,128,256", "--dense-width", "256", "--pool-window", "16",
            "--pool-stride", "8", "--cross-op", op, "--nbins", "4", "--flatsky-roll-frac", "0.10",
            "--flatsky-bnt",
            "--cross-tfds-name", TFDS, "--cross-tfds-data-dir", DDIR, "--channel-rms-nsample", "8000",
            "--fid-cache-dir", FID, "--regime", "nobnt", "--cosmo-id", "cosmo_fiducial",
            "--perms", "0-49", "--g1-obs-npz", f"{d}/cache/cnn_obs.npz",
            "--g1-perm", str(OBS_PERM), "--g1-patch", str(OBS_PATCH),
            "--out", f"{BNT}/fiducial_summaries/fiducial_summaries_cnn_{op}_s{seed}.npz",
            "--cuda-visible-devices", str(gpu)]


def l1_fid_precompute_cmd(gpu):
    return [PY, "precompute_fiducial_both_datavectors.py", "--bnt",
            "--both-cache", BOTH_CACHE_BNT, "--sigma", SIGMA_BNT, "--out", FID_BOTH_BNT]


def sweep_cmd(probe, op, seed, gpu):
    if probe == "cnn":
        return [PY, "population_sweep_flatsky.py",
                "--train-cache-dir", f"{BNT}/cnn_{op}_s{seed}/cache",
                "--cache-prefix", "cnn", "--arm-label", f"bnt_cnn_{op}_s{seed}",
                "--fiducial-summaries-npz",
                f"{BNT}/fiducial_summaries/fiducial_summaries_cnn_{op}_s{seed}.npz",
                "--output-dir", f"{BNT}/population_sweep/cnn_{op}_s{seed}",
                "--preproc-transform", "none", "--clip-value", "0",
                "--min-feature-variance", "1e-12",
                "--seeds", "41,42,43", "--n-obs", "9000", "--max-perm", "50",
                "--m-samples", "2000", "--cuda-visible-devices", str(gpu)]
    return [PY, "population_sweep_flatsky.py",
            "--train-cache-dir", f"{L1M}/l1_{op}_cache/flat_local_{op}_bnt",
            "--cache-prefix", "l1", "--arm-label", f"bnt_l1_{op}",
            "--fiducial-summaries-npz", f"{BNT}/fiducial_summaries/fiducial_summaries_l1_{op}.npz",
            "--output-dir", f"{BNT}/population_sweep/l1_{op}",
            "--preproc-transform", "log1p-zscore", "--clip-value", "5",
            "--min-feature-variance", "1e-5",
            "--seeds", "41,42,43", "--n-obs", "9000", "--max-perm", "50",
            "--m-samples", "2000", "--cuda-visible-devices", str(gpu)]


def slice_l1_fiducials():
    """CPU: slice the BNT 'both' fiducial datavectors per arm (key S, as the sweep expects)."""
    sys.path.insert(0, SBI)
    import flatsky_cross_l1 as fxl
    z = np.load(FID_BOTH_BNT)
    outdir = Path(BNT) / "fiducial_summaries"; outdir.mkdir(parents=True, exist_ok=True)
    for op in OPS:
        cols = fxl.op_feature_columns(op, 4, FEAT_PER_CH)
        out = outdir / f"fiducial_summaries_l1_{op}.npz"
        np.savez(out, S=z["x"][:, cols], perm=z["perm"], patch=z["patch"],
                 theta=z["truth"], truth=z["truth"], bnt=True)
        print(f"  sliced {op}: {z['x'].shape} -> {(z['x'].shape[0], cols.size)} -> {out}",
              flush=True)


def run_phase(name, jobs, cmd_fn, gpus):
    """Greedy scheduler; jobs = list of tuples passed to cmd_fn(*job, gpu)."""
    print(f"\n===== PHASE {name} =====", flush=True)
    os.makedirs(LOGS, exist_ok=True)
    pending = list(jobs); slots = {g: None for g in gpus}; t0 = time.time(); failed = {}

    def tag(job):
        return "_".join(str(x) for x in job) if job else name

    def launch(job, gpu):
        try:
            c = cmd_fn(*job, gpu)
        except Exception as exc:
            failed[tag(job)] = f"cmd-build: {exc}"
            print(f"[{time.time()-t0:6.0f}s] SKIP {name} {tag(job)} "
                  f"(command build failed: {exc})", flush=True)
            return None
        log = open(f"{LOGS}/{name}_{tag(job)}.log", "w")
        env = dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="3",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.5",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", CNN_CPU_THREADS="8")
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        p = subprocess.Popen(c, cwd=SBI, env=env, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL)
        print(f"[{time.time()-t0:6.0f}s] LAUNCH {name} {tag(job)} GPU{gpu} (pid {p.pid})",
              flush=True)
        return (job, p, log)

    while pending or any(slots.values()):
        for g in gpus:
            s = slots[g]
            if s and s[1].poll() is not None:
                job, p, log = s; log.close(); slots[g] = None
                if p.returncode != 0:
                    failed[tag(job)] = p.returncode
                print(f"[{time.time()-t0:6.0f}s] {'DONE' if p.returncode==0 else 'FAIL'} "
                      f"{name} {tag(job)}", flush=True)
        for g in gpus:
            if slots[g] is None and pending:
                slots[g] = launch(pending.pop(0), g)
        time.sleep(10)
    if failed:
        print(f"  [{name}] FAILED: {failed}", flush=True)
    return failed


def write_report(failures):
    """Derived inflation report — every verdict computed from medians on disk."""
    def med(path):
        f = Path(path) / "median_summary.json"
        return json.load(open(f)) if f.exists() else None

    nobnt = {
        ("l1", "none"): med(f"{BASE}/population_sweep/flat_none"),
        ("l1", "product"): med(f"{BASE}/population_sweep/flat_product"),
        ("cnn", "none", 41): med(f"{BASE}/cnn_phase/population_sweep/flat_none"),
        ("cnn", "product", 41): med(f"{BASE}/cnn_phase/population_sweep/flat_product"),
        ("cnn", "none", 42): med(f"{BASE}/cnn_phase/multiseed/population_sweep/none_s42"),
        ("cnn", "none", 43): med(f"{BASE}/cnn_phase/multiseed/population_sweep/none_s43"),
        ("cnn", "product", 42): med(f"{BASE}/cnn_phase/multiseed/population_sweep/product_s42"),
        ("cnn", "product", 43): med(f"{BASE}/cnn_phase/multiseed/population_sweep/product_s43"),
    }
    L = ["# BNT flat-local campaign — inflation ratios (FoM3_BNT / FoM3_noBNT)\n",
         "Pooled 3-MAF 9000-obs medians; no-BNT references read from the existing campaign "
         "dirs. Predictions (plan §B): L1-auto inflates ≪1, L1+product less so, CNN ≈ 1.\n",
         "| arm | no-BNT FoM3 | BNT FoM3 | inflation (BNT/noBNT) |", "|---|---|---|---|"]
    ratios = {}
    for op in OPS:
        b = med(f"{BNT}/population_sweep/l1_{op}"); n = nobnt[("l1", op)]
        if b and n:
            ratios[("l1", op)] = b["fom3"] / n["fom3"]
            L.append(f"| L1 {op} | {n['fom3']:.0f} | {b['fom3']:.0f} | "
                     f"{ratios[('l1', op)]:.2f}× |")
        else:
            L.append(f"| L1 {op} | {'—' if not n else f'{n['fom3']:.0f}'} | — | — |")
    for op in OPS:
        per = []
        for s in CNN_SEEDS:
            b = med(f"{BNT}/population_sweep/cnn_{op}_s{s}"); n = nobnt[("cnn", op, s)]
            if b and n:
                r = b["fom3"] / n["fom3"]; per.append(r)
                L.append(f"| CNN {op} s{s} | {n['fom3']:.0f} | {b['fom3']:.0f} | {r:.2f}× |")
            else:
                L.append(f"| CNN {op} s{s} | {'—' if not n else f'{n['fom3']:.0f}'} | — | — |")
        if per:
            ratios[("cnn", op)] = float(np.mean(per))
            L.append(f"| **CNN {op} mean-of-seeds** |  |  | **{np.mean(per):.2f}×** |")
    if all(k in ratios for k in (("l1", "none"), ("l1", "product"), ("cnn", "none"))):
        l1a, l1p, ca = ratios[("l1", "none")], ratios[("l1", "product")], ratios[("cnn", "none")]
        p1 = l1a < 0.9
        p2 = l1p > l1a
        p3 = ca > 0.9
        L += ["", "**Prediction ladder (derived):**",
              f"1. L1-auto inflates (ratio {l1a:.2f} < 0.9): {'HOLDS' if p1 else 'does NOT hold'}",
              f"2. L1+product inflates less than L1-auto ({l1p:.2f} vs {l1a:.2f}): "
              f"{'HOLDS' if p2 else 'does NOT hold'}",
              f"3. CNN ≈ lossless (auto ratio {ca:.2f} > 0.9): {'HOLDS' if p3 else 'does NOT hold'}"
              + ("" if p3 else " — contingency ladder applies (160k recipe / advanced arch)"),
              "", "Caveat: FoM3 is correlation-sensitive; check σ/2D in the median_summary.json "
              "files before headlining, and GATE C the load-bearing BNT arms."]
    else:
        L += ["", "**INCOMPLETE — missing sweeps; no ladder verdict.**"]
    if failures:
        L += ["", f"⚠ FAILURES: {failures}"]
    Path(BNT, "BNT_CAMPAIGN_RESULT.md").write_text("\n".join(L))
    print(f"wrote {BNT}/BNT_CAMPAIGN_RESULT.md", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gpus", default="1,2")
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    os.chdir(SBI)
    if args.dry_run:
        print("# P0 sigma:\n" + " ".join(freeze_cmd("<GPU>")))
        print("\n# P1 both-build:\n" + " ".join(l1_cmd("both", "<GPU>", False)))
        for op in OPS:
            print(f"\n# P2 l1 {op}:\n" + " ".join(l1_cmd(op, "<GPU>", True)))
        for op in OPS:
            for s in CNN_SEEDS:
                print(f"\n# P3 cnn {op}_s{s}:\n" + " ".join(cnn_compressor_cmd(op, s, "<GPU>")))
        print("\n# P4 l1-fid:\n" + " ".join(l1_fid_precompute_cmd("<GPU>")))
        print("\n(+ cnn fidsumms, l1 slices, 8 sweeps, derived report)")
        return
    os.makedirs(BNT, exist_ok=True)
    t0 = time.time(); failures = {}
    if not Path(SIGMA_BNT).exists():
        failures.update(run_phase("sigma", [()], lambda gpu: freeze_cmd(gpu), gpus[:1]))
    else:
        print(f"P0 sigma: {SIGMA_BNT} exists, skipping.")
    failures.update(run_phase("both_build", [("both",)],
                              lambda op, gpu: l1_cmd(op, gpu, False), gpus[:1]))
    failures.update(run_phase("l1_arms", [(op,) for op in OPS],
                              lambda op, gpu: l1_cmd(op, gpu, True), gpus))
    failures.update(run_phase("cnn_compressor", [(op, s) for op in OPS for s in CNN_SEEDS],
                              cnn_compressor_cmd, gpus))
    failures.update(run_phase("fidsumm", [(op, s) for op in OPS for s in CNN_SEEDS],
                              cnn_fidsumm_cmd, gpus))
    failures.update(run_phase("l1_fid", [()], lambda gpu: l1_fid_precompute_cmd(gpu), gpus[:1]))
    try:
        slice_l1_fiducials()
    except Exception as exc:
        failures["slice_l1_fiducials"] = str(exc)
        print(f"  SLICE FAILED: {exc}", flush=True)
    sweep_jobs = ([("cnn", op, s) for op in OPS for s in CNN_SEEDS]
                  + [("l1", op, 41) for op in OPS])   # L1 seed is a label only
    failures.update(run_phase("sweep", sweep_jobs, sweep_cmd, gpus))
    write_report(failures)
    print(f"\n=== BNT campaign done in {(time.time()-t0)/3600:.1f} h ===")


if __name__ == "__main__":
    main()
