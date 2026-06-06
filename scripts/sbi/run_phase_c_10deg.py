#!/usr/bin/env python
"""Phase C orchestrator: 4 arms × 3 seeds on the 10deg unified TFDS, GPU 1+2.

Arms: CNN/L1 × auto_cross/auto_only. Seeds 41/42/43. Split: compressor perms 0-4 /
NDE perms 5-6 (example-disjoint). L1 datavector is cached per ARM (seed not in the
cache key -> cross-seed dedup), so each arm's seed-41 builds the cache and seeds 42/43
reuse it; the scheduler enforces that dependency. CNN trains a full compressor+NDE per seed.

Greedy 2-GPU scheduler (GPU 1 + 2 only). --dry-run prints the 12 commands + schedule
without executing. See PLAN_PHASE_C.md.
"""
import argparse
import os
import subprocess
import time
from pathlib import Path

REPO = "/mnt/home/tersenov/software/cnn_sbi"
SBI = f"{REPO}/scripts/sbi"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
TFDS = "nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180"
DDIR = "/home/tersenov/tensorflow_datasets"
FID = f"{SBI}/results/exploratory/cross_maps_campaign/full_sphere_cache_fiducial_10deg"
OUT = f"{SBI}/results/exploratory/definitive_comparison_10deg/phase_c"
LOGS = f"{OUT}/logs"
GPUS = [1, 2]
SEEDS = [41, 42, 43]
OBS_PERM, OBS_PATCH = 0, 90


def cnn_cmd(mode, seed, gpu, steps):
    d = f"{OUT}/cnn_{mode}_s{seed}"
    return [
        PY, "npe_cnn_nbody_tomo.py", "--train-compressor",
        "--cnn-map-route", "tfds_cross", "--cross-tfds-name", TFDS,
        "--cross-tfds-data-dir", DDIR, "--fiducial-obs-cache", FID,
        "--harmonic-cache-regime", "nobnt", "--harmonic-normalize-input-channels",
        "--channel-mode", mode, "--cnn-perm-split", "0-4:5-6",
        "--zero-mean-maps", "--map-kind", "nbody", "--seed", str(seed),
        "--field-size", "10", "--field-npix", "80",
        "--nbins", "4", "--tomo-bin-indices", "1,2,3,4",
        "--compressor-arch", "plain", "--compressor-dim", "10",
        "--compressor-dense-width", "256", "--compressor-conv-channels", "64,128,256",
        "--compressor-steps", str(steps), "--compressor-batch-size", "128",
        "--compressor-lr", "0.0005", "--compressor-checkpoint-policy", "best_val",
        "--npe-samples", "100000", "--no-wandb",
        "--harmonic-obs-perm", str(OBS_PERM), "--harmonic-obs-patch-idx", str(OBS_PATCH),
        "--cuda-visible-devices", str(gpu),
        "--save-dir", d, "--cache-dir", f"{d}/cache",
        "--posterior-out", f"{d}/posterior.npy", "--figure-out", f"{d}/corner.pdf",
    ]


def l1_cmd(mode, seed, gpu):
    d = f"{OUT}/l1_{mode}_s{seed}"
    return [
        PY, "npe_l1norm_cross_jaxili_nbody_tomo.py",
        "--cross-maps-route", "tfds_cross", "--cross-tfds-name", TFDS,
        "--cross-tfds-data-dir", DDIR, "--fiducial-obs-cache-dir", FID,
        "--cross-noise-model", "channel_empirical_global", "--pca-components", "0",
        "--channel-mode", mode, "--nde-perm-split", "5-6", "--nde-val-perm-split", "0-1",
        "--nbins", "4", "--tomo-bin-indices", "1,2,3,4",
        "--field-size", "10", "--field-npix", "80",
        "--n-scales", "5", "--l1-nbins", "40", "--l1-min-snr", "-13", "--l1-max-snr", "13",
        "--cross-map-auto-calibrate-snr", "--cross-snr-percentile", "1.0",
        "--summary-transform", "log1p-zscore", "--clip-value", "5", "--seed", str(seed),
        "--harmonic-obs-perm", str(OBS_PERM), "--harmonic-obs-patch-idx", str(OBS_PATCH),
        "--no-wandb", "--cuda-visible-devices", str(gpu),
        "--save-dir", d, "--cache-dir", f"{OUT}/l1_{mode}_cache",  # SHARED per arm (dedup)
        "--posterior-out", f"{d}/posterior.npy", "--figure-out", f"{d}/corner.pdf",
    ]


def build_jobs(steps):
    """Return list of job dicts. L1 seeds 42/43 depend on seed 41 (datavector cache)."""
    jobs = []
    for mode in ("auto_cross", "auto_only"):
        for seed in SEEDS:
            jobs.append({"name": f"cnn_{mode}_s{seed}", "kind": "cnn", "mode": mode,
                         "seed": seed, "steps": steps, "deps": []})
        # L1: seed 41 builds the per-arm datavector cache; 42/43 depend on it.
        jobs.append({"name": f"l1_{mode}_s41", "kind": "l1", "mode": mode, "seed": 41,
                     "deps": [], "priority": 0})  # priority 0 = start first (long pole)
        for seed in (42, 43):
            jobs.append({"name": f"l1_{mode}_s{seed}", "kind": "l1", "mode": mode,
                         "seed": seed, "deps": [f"l1_{mode}_s41"]})
    return jobs


def cmd_for(job, gpu):
    if job["kind"] == "cnn":
        return cnn_cmd(job["mode"], job["seed"], gpu, job["steps"])
    return l1_cmd(job["mode"], job["seed"], gpu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print commands + schedule, no exec")
    ap.add_argument("--compressor-steps", type=int, default=80000)
    args = ap.parse_args()

    os.makedirs(LOGS, exist_ok=True)
    jobs = build_jobs(args.compressor_steps)
    # Order: L1 seed-41 builds first (priority 0, long pole + unblock 42/43), then the rest.
    jobs.sort(key=lambda j: j.get("priority", 1))

    if args.dry_run:
        print(f"=== Phase C dry-run: {len(jobs)} jobs, GPUs {GPUS}, seeds {SEEDS} ===")
        for j in jobs:
            dep = f" (after {j['deps']})" if j["deps"] else ""
            print(f"\n# {j['name']}{dep}")
            print(" ".join(cmd_for(j, "<GPU>")))
        print("\n(dry-run: nothing executed)")
        return

    os.chdir(SBI)
    pending = list(jobs)
    done, failed = set(), {}
    slots = {g: None for g in GPUS}   # gpu -> (name, Popen, logfile) | None
    t0 = time.time()

    def launch(job, gpu):
        log = open(f"{LOGS}/{job['name']}.log", "w")
        env = dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="3",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.4")
        p = subprocess.Popen(cmd_for(job, gpu), cwd=SBI, env=env, stdout=log,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        print(f"[{time.time()-t0:7.0f}s] LAUNCH {job['name']} on GPU{gpu} (pid {p.pid})", flush=True)
        return (job["name"], p, log)

    while pending or any(slots.values()):
        # Reap finished
        for g in GPUS:
            s = slots[g]
            if s and s[1].poll() is not None:
                nm, p, log = s
                log.close()
                slots[g] = None
                if p.returncode == 0:
                    done.add(nm)
                    print(f"[{time.time()-t0:7.0f}s] DONE   {nm} (GPU{g})", flush=True)
                else:
                    failed[nm] = p.returncode
                    done.add(nm)  # mark resolved so dependents can be skipped/known
                    print(f"[{time.time()-t0:7.0f}s] FAIL   {nm} rc={p.returncode} "
                          f"(see {LOGS}/{nm}.log)", flush=True)
        # Assign ready jobs to free GPUs
        for g in GPUS:
            if slots[g] is not None:
                continue
            for j in list(pending):
                if all(d in done for d in j["deps"]):
                    # skip a job whose dependency FAILED
                    if any(d in failed for d in j["deps"]):
                        print(f"[{time.time()-t0:7.0f}s] SKIP   {j['name']} (dep failed)", flush=True)
                        pending.remove(j)
                        break
                    pending.remove(j)
                    slots[g] = launch(j, g)
                    break
        time.sleep(10)

    elapsed = time.time() - t0
    print(f"\n=== Phase C finished in {elapsed/60:.1f} min ===")
    print(f"  done: {sorted(done - set(failed))}")
    if failed:
        print(f"  FAILED: {failed}")
    else:
        print("  all 12 jobs OK")


if __name__ == "__main__":
    main()
