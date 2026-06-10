#!/usr/bin/env python
"""Flat-local CNN compressor matrix: 4 arms (none/conv/product/both), seed 41, GPU 1+2.

Mirrors run_phase_c_10deg.py but for the DE-LEAKED patch-local cross (CNN side of the
flatsky-cross campaign; FLATSKY_CROSS_RESULT.md is the L1 side). Each arm trains ONE
plain-CNN VMIM compressor on --cnn-map-route flat_local --cross-op <arm> and exits after
building the compressed cache (--exit-after-compress) -> cnn_train.npz / cnn_val.npz /
cnn_cache_meta.npz / cnn_obs.npz per arm. The 3-seed pooling happens DOWNSTREAM at the
common jaxili MAF (population_sweep_flatsky.py --cache-prefix cnn), symmetric with L1 and
removing the NDE-architecture confound (run_phase_d_10deg.py pattern).

Decisions locked w/ Andreas 2026-06-09: plain CNN (NO BatchNorm; resnet50_gn only needed
for the leaky 10-ch harmonic route), 1 compressor/arm, GPU 1+2, train-sample RMS whitening.

Greedy 2-GPU scheduler (GPU 1 + 2 only). --dry-run prints the 4 commands + schedule.
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
OUT = f"{SBI}/results/exploratory/flatsky_cross_2026_06/cnn_phase"
LOGS = f"{OUT}/logs"
GPUS = [1, 2]
SEED = 41
ARMS = ["none", "conv", "product", "both"]
OBS_PERM, OBS_PATCH = 0, 90   # single-obs anchor (matches Phase C 10deg); pop sweep uses 9000


def cnn_cmd(op, gpu, steps):
    d = f"{OUT}/cnn_{op}_s{SEED}"
    return [
        PY, "npe_cnn_nbody_tomo.py", "--train-compressor", "--exit-after-compress",
        "--cnn-map-route", "flat_local", "--cross-op", op,
        "--cross-tfds-name", TFDS, "--cross-tfds-data-dir", DDIR,
        "--fiducial-obs-cache", FID,
        "--harmonic-cache-regime", "nobnt", "--harmonic-normalize-input-channels",
        "--cnn-perm-split", "0-4:5-6",
        "--zero-mean-maps", "--map-kind", "nbody", "--seed", str(SEED),
        "--field-size", "10", "--field-npix", "80",
        "--nbins", "4", "--tomo-bin-indices", "1,2,3,4",
        "--compressor-arch", "plain", "--compressor-dim", "10",
        "--compressor-dense-width", "256", "--compressor-conv-channels", "64,128,256",
        "--compressor-steps", str(steps), "--compressor-batch-size", "128",
        "--compressor-lr", "0.0005", "--compressor-checkpoint-policy", "best_val",
        "--no-wandb",
        "--harmonic-obs-perm", str(OBS_PERM), "--harmonic-obs-patch-idx", str(OBS_PATCH),
        "--cuda-visible-devices", str(gpu),
        "--save-dir", d, "--cache-dir", f"{d}/cache",
        "--posterior-out", f"{d}/posterior.npy", "--figure-out", f"{d}/corner.pdf",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print commands + schedule, no exec")
    ap.add_argument("--compressor-steps", type=int, default=80000,
                    help="Match Phase C 10deg baseline (80000).")
    args = ap.parse_args()

    os.makedirs(LOGS, exist_ok=True)
    jobs = [{"name": f"cnn_{op}_s{SEED}", "op": op} for op in ARMS]

    if args.dry_run:
        print(f"=== Flat-local CNN matrix dry-run: {len(jobs)} arms, GPUs {GPUS}, "
              f"seed {SEED}, steps {args.compressor_steps} ===")
        for j in jobs:
            print(f"\n# {j['name']}")
            print(" ".join(cnn_cmd(j["op"], "<GPU>", args.compressor_steps)))
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
                   XLA_PYTHON_CLIENT_PREALLOCATE="false",
                   XLA_PYTHON_CLIENT_MEM_FRACTION="0.4",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        p = subprocess.Popen(cnn_cmd(job["op"], gpu, args.compressor_steps), cwd=SBI,
                             env=env, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL)
        print(f"[{time.time()-t0:7.0f}s] LAUNCH {job['name']} on GPU{gpu} (pid {p.pid})",
              flush=True)
        return (job["name"], p, log)

    while pending or any(slots.values()):
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
                    done.add(nm)
                    print(f"[{time.time()-t0:7.0f}s] FAIL   {nm} rc={p.returncode} "
                          f"(see {LOGS}/{nm}.log)", flush=True)
        for g in GPUS:
            if slots[g] is not None:
                continue
            if pending:
                slots[g] = launch(pending.pop(0), g)
        time.sleep(10)

    elapsed = time.time() - t0
    print(f"\n=== Flat-local CNN matrix finished in {elapsed/60:.1f} min ===")
    print(f"  done: {sorted(done - set(failed))}")
    if failed:
        print(f"  FAILED: {failed}")
    else:
        print("  all 4 arms OK")


if __name__ == "__main__":
    main()
