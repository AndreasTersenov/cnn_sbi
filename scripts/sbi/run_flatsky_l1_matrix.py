#!/usr/bin/env python
"""Flat-local L1 matrix (build-both-once-slice): 4 arms x 3 seeds, GPUs 1+2.

The four arms read the SAME autos and 'both' (16ch) is the superset, so none/conv/product
are exact column-slices of 'both'. Phase 1 builds the 'both' L1 datavector EXACTLY ONCE
(solo, no I/O contention; the expensive loader pass). Phase 2 runs all remaining 11 arm x
seed jobs by slicing that datavector (--flatsky-both-cache: no loader pass, obs computed
per-op as a single map) + training the jaxili MAF. Eliminates the redundant 4x loader passes
AND the disk-I/O thrashing that throttled the naive per-arm version (40/s -> ).

Greedy scheduler, GPU 1+2. Phase-2 jobs are NDE-light so they pack several per GPU; the
'both' build runs alone first (all others depend on it). --dry-run prints commands.
"""
import argparse, os, subprocess, time
from pathlib import Path

REPO = "/mnt/home/tersenov/software/cnn_sbi"
SBI = f"{REPO}/scripts/sbi"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
TFDS = "nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180"
DDIR = "/home/tersenov/tensorflow_datasets"
FID = f"{SBI}/results/exploratory/cross_maps_campaign/full_sphere_cache_fiducial_10deg"
SIGMA = f"{SBI}/results/exploratory/flatsky_cross_2026_06/flatsky_cross_noise_sigma.npz"
OUT = f"{SBI}/results/exploratory/flatsky_cross_2026_06/l1_matrix"
LOGS = f"{OUT}/logs"
BOTH_CACHE = f"{OUT}/l1_both_cache/flat_local_both"   # cache-dir + the script's op suffix
GPUS = [1, 2]
MAX_PER_GPU = 2          # datavector is big (323k x up-to-3200 + jaxili train-split gather copy ~
                         # 7-12GB in GPU); 2/GPU at 0.4 frac (16GB) fits, 4/GPU OOMed
SEEDS = [41, 42, 43]
OBS_PERM, OBS_PATCH = 0, 90


def l1_cmd(op, seed, gpu, steps, use_both_cache):
    d = f"{OUT}/l1_{op}_s{seed}"
    cmd = [
        PY, "npe_l1norm_cross_jaxili_nbody_tomo.py",
        "--cross-maps-route", "flat_local", "--cross-op", op,
        "--cross-tfds-name", TFDS, "--cross-tfds-data-dir", DDIR,
        "--fiducial-obs-cache-dir", FID, "--flatsky-cross-sigma", SIGMA,
        "--pca-components", "0",
        "--nde-perm-split", "5-6", "--nde-val-perm-split", "0-1",
        "--nbins", "4", "--tomo-bin-indices", "1,2,3,4",
        "--field-size", "10", "--field-npix", "80",
        "--n-scales", "5", "--l1-nbins", "40",
        "--harmonic-calibration-realizations", "20", "--ds-batch-size", "512",
        "--total-steps", str(steps),
        "--summary-transform", "log1p-zscore", "--clip-value", "5", "--seed", str(seed),
        "--harmonic-obs-perm", str(OBS_PERM), "--harmonic-obs-patch-idx", str(OBS_PATCH),
        "--no-wandb", "--cuda-visible-devices", str(gpu),
        "--save-dir", d, "--cache-dir", f"{OUT}/l1_{op}_cache",
        "--posterior-out", f"{d}/posterior.npy", "--figure-out", f"{d}/corner.pdf",
    ]
    if use_both_cache:
        cmd += ["--flatsky-both-cache", BOTH_CACHE]
    return cmd


def build_jobs(steps):
    jobs = [{"name": "l1_both_s41", "op": "both", "seed": 41, "steps": steps,
             "deps": [], "both_cache": False, "priority": 0}]   # PHASE 1: build once, solo
    for op in ("none", "conv", "product"):
        for seed in SEEDS:
            jobs.append({"name": f"l1_{op}_s{seed}", "op": op, "seed": seed, "steps": steps,
                         "deps": ["l1_both_s41"], "both_cache": True, "priority": 1})
    for seed in (42, 43):
        jobs.append({"name": f"l1_both_s{seed}", "op": "both", "seed": seed, "steps": steps,
                     "deps": ["l1_both_s41"], "both_cache": True, "priority": 1})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--total-steps", type=int, default=5000)
    args = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    jobs = sorted(build_jobs(args.total_steps), key=lambda j: j["priority"])

    if args.dry_run:
        print(f"=== build-both-slice dry-run: {len(jobs)} jobs (1 build + {len(jobs)-1} slice) ===")
        for j in jobs:
            dep = f" (after {j['deps']})" if j["deps"] else " [PHASE 1 BUILD]"
            print(f"\n# {j['name']}{dep}\n" + " ".join(l1_cmd(j["op"], j["seed"], "<GPU>", j["steps"], j["both_cache"])))
        return

    os.chdir(SBI)
    pending, done, failed = list(jobs), set(), {}
    running = []
    t0 = time.time()

    def launch(job, gpu):
        log = open(f"{LOGS}/{job['name']}.log", "w")
        env = dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="3",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.4",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                   OMP_NUM_THREADS="6", MKL_NUM_THREADS="6", OPENBLAS_NUM_THREADS="6")
        p = subprocess.Popen(l1_cmd(job["op"], job["seed"], gpu, job["steps"], job["both_cache"]),
                             cwd=SBI, env=env, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL)
        print(f"[{time.time()-t0:7.0f}s] LAUNCH {job['name']} GPU{gpu} (pid {p.pid})", flush=True)
        running.append({"name": job["name"], "p": p, "log": log, "gpu": gpu})

    while pending or running:
        for r in list(running):
            if r["p"].poll() is not None:
                r["log"].close(); running.remove(r); done.add(r["name"])
                tag = "DONE  " if r["p"].returncode == 0 else f"FAIL rc={r['p'].returncode}"
                if r["p"].returncode != 0:
                    failed[r["name"]] = r["p"].returncode
                print(f"[{time.time()-t0:7.0f}s] {tag} {r['name']} (GPU{r['gpu']})", flush=True)
        changed = True
        while changed:
            changed = False
            load = {g: sum(1 for r in running if r["gpu"] == g) for g in GPUS}
            free = [g for g in GPUS if load[g] < MAX_PER_GPU]
            if not free:
                break
            gpu = min(free, key=lambda g: load[g])
            for j in list(pending):
                if all(d in done for d in j["deps"]):
                    if any(d in failed for d in j["deps"]):
                        print(f"[{time.time()-t0:7.0f}s] SKIP   {j['name']} (dep failed)", flush=True)
                        pending.remove(j); changed = True; break
                    pending.remove(j); launch(j, gpu); changed = True; break
        time.sleep(8)

    print(f"\n=== build-both-slice finished in {(time.time()-t0)/60:.1f} min ===")
    print(f"  done: {sorted(done - set(failed))}")
    print(f"  FAILED: {failed}" if failed else f"  all {len(jobs)} jobs OK")


if __name__ == "__main__":
    main()
