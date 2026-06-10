#!/usr/bin/env python
"""Build CNN flat-local fiducial obs summaries (9000/arm) for GATE C + population sweep.

Per arm, reads the arm's cnn_cache_meta.npz to recover the EXACT sha-pinned compressor
checkpoint, then runs build_fiducial_summaries_cnn.py on the flat-local route (--cross-op):
reads RAW autos ch 0..3 of the fiducial cache, builds the patch-local cross on-device + whitens
(same estimator/transform as training), compresses 9000 fiducial obs (perm<50 x 180), and
G1-checks that obs@(perm0,patch90) reproduces the arm's cnn_obs.npz. Output per arm:
  cnn_phase/fiducial_summaries/fiducial_summaries_<op>.npz  (keys S, perm, patch, theta)

These feed population_sweep_flatsky.py --cache-prefix cnn and the CNN L-C2ST. 2-GPU greedy
scheduler, <=2 concurrent (TFDS + fiducial-cache reader cap). --dry-run prints commands.
"""
import argparse, os, subprocess, time
import numpy as np
from pathlib import Path

REPO = "/mnt/home/tersenov/software/cnn_sbi"
SBI = f"{REPO}/scripts/sbi"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
TFDS = "nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180"
DDIR = "/home/tersenov/tensorflow_datasets"
FID = f"{SBI}/results/exploratory/cross_maps_campaign/full_sphere_cache_fiducial_10deg"
CNN = f"{SBI}/results/exploratory/flatsky_cross_2026_06/cnn_phase"
OUTDIR = f"{CNN}/fiducial_summaries"
LOGS = f"{OUTDIR}/logs"
GPUS = [1, 2]
ARMS = ["none", "conv", "product", "both"]
OBS_PERM, OBS_PATCH = 0, 90   # the arm's cnn_obs.npz anchor (matches the matrix obs)


def cmd(op, gpu):
    d = f"{CNN}/cnn_{op}_s41"
    meta = dict(np.load(f"{d}/cache/cnn_cache_meta.npz", allow_pickle=True))
    params = str(meta["compressor_params_path"])
    state = str(meta["compressor_state_path"])
    psha = str(meta["compressor_params_sha256"])
    ssha = str(meta["compressor_state_sha256"])
    nch = int(meta["cnn_input_channels"])
    return [
        PY, "build_fiducial_summaries_cnn.py",
        "--arm-label", f"flat_{op}",
        "--params-pkl", params, "--state-pkl", state,
        "--expect-params-sha", psha, "--expect-state-sha", ssha,
        "--n-channels", str(nch), "--dim", "10",
        "--conv-channels", "64,128,256", "--dense-width", "256",
        "--pool-window", "16", "--pool-stride", "8",
        "--cross-op", op, "--nbins", "4", "--flatsky-roll-frac", "0.10",
        "--cross-tfds-name", TFDS, "--cross-tfds-data-dir", DDIR,
        "--channel-rms-nsample", "8000",
        "--fid-cache-dir", FID, "--regime", "nobnt", "--cosmo-id", "cosmo_fiducial",
        "--perms", "0-49",
        "--g1-obs-npz", f"{d}/cache/cnn_obs.npz",
        "--g1-perm", str(OBS_PERM), "--g1-patch", str(OBS_PATCH),
        "--out", f"{OUTDIR}/fiducial_summaries_{op}.npz",
        "--cuda-visible-devices", str(gpu),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        for op in ARMS:
            print(f"\n# flat_{op}\n" + " ".join(cmd(op, "<GPU>")))
        return
    os.makedirs(LOGS, exist_ok=True)
    os.chdir(SBI)
    pending = list(ARMS)
    slots = {g: None for g in GPUS}
    t0 = time.time()
    done, failed = [], {}

    def launch(op, gpu):
        # cmd() np.load's the arm's cache meta — SKIP this arm rather than crash
        # the driver if the meta is missing/corrupt (e.g. failed compressor run).
        try:
            c = cmd(op, gpu)
        except Exception as exc:
            failed[op] = f"cmd-build: {exc}"
            print(f"[{time.time()-t0:7.0f}s] SKIP flat_{op} (command build failed: {exc})",
                  flush=True)
            return None
        log = open(f"{LOGS}/flat_{op}.log", "w")
        env = dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="3",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.4",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                   CNN_CPU_THREADS="8")
        p = subprocess.Popen(c, cwd=SBI, env=env, stdout=log,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        print(f"[{time.time()-t0:7.0f}s] LAUNCH flat_{op} GPU{gpu} (pid {p.pid})", flush=True)
        return (op, p, log)

    while pending or any(slots.values()):
        for g in GPUS:
            s = slots[g]
            if s and s[1].poll() is not None:
                op, p, log = s
                log.close(); slots[g] = None
                if p.returncode == 0:
                    done.append(op)
                    print(f"[{time.time()-t0:7.0f}s] DONE   flat_{op} (GPU{g})", flush=True)
                else:
                    failed[op] = p.returncode
                    print(f"[{time.time()-t0:7.0f}s] FAIL   flat_{op} rc={p.returncode} "
                          f"(see {LOGS}/flat_{op}.log)", flush=True)
        for g in GPUS:
            if slots[g] is None and pending:
                slots[g] = launch(pending.pop(0), g)
        time.sleep(10)

    print(f"\n=== CNN fiducial summaries done in {(time.time()-t0)/60:.1f} min ===")
    print(f"  done: {done}")
    if failed:
        print(f"  FAILED: {failed}")


if __name__ == "__main__":
    main()
