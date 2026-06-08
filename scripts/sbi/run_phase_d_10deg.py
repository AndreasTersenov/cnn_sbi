#!/usr/bin/env python
"""Phase D orchestrator: per-patch geometry_resample for the 4 arms on GPU 1+2.

Each arm: re-train 3 jaxili MAF NDEs (seeds 41/42/43) from the Phase-C compressed
cache, then sample a posterior for every (patch 0-179 x perm 0-49 = 9000 obs) in the
arm's fiducial summaries -> per_patch_grid (fom3/2D/sigma/bias/pull per obs). Both
arms use the SAME MAF (removes the NDE-architecture confound; CNN Phase-C used RealNVP,
L1 used MAF). Preproc matches the 20deg campaign: L1 log1p-zscore-clip5-mask1e-5,
CNN none-mask1e-12. 4 independent jobs, greedy 2-GPU scheduler. --dry-run prints commands.

After this: geometry_analyze.py + compare_offsets.py (CPU), then SBC + L-C2ST.
"""
import argparse
import os
import subprocess
import time

REPO = "/mnt/home/tersenov/software/cnn_sbi"
SBI = f"{REPO}/scripts/sbi"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
PC = f"{SBI}/results/exploratory/definitive_comparison_10deg/phase_c"
FID10 = f"{SBI}/results/exploratory/cross_maps_campaign/full_sphere_cache_fiducial_10deg"
OUTDIR = f"{PC}/analysis/geometry"
LOGS = f"{PC}/analysis/geometry/logs"
GPUS = [1, 2]
SEEDS = "41,42,43"
PATCHES = "0-179"
PERMS = "0-49"

# (label, train_cache_dir, cache_prefix, summaries_npz, preproc, clip, min_var)
ARMS = [
    ("l1_auto_cross", f"{PC}/l1_auto_cross_cache", "l1",
     f"{PC}/fiducial_summaries/l1_auto_cross.npz", "log1p-zscore", "5", "1e-5"),
    ("l1_auto_only", f"{PC}/l1_auto_only_cache", "l1",
     f"{PC}/fiducial_summaries/l1_auto_only.npz", "log1p-zscore", "5", "1e-5"),
    ("cnn_auto_cross", f"{PC}/cnn_auto_cross_s41/cache", "cnn",
     f"{PC}/fiducial_summaries/cnn_auto_cross.npz", "none", "0", "1e-12"),
    ("cnn_auto_only", f"{PC}/cnn_auto_only_s41/cache", "cnn",
     f"{PC}/fiducial_summaries/cnn_auto_only.npz", "none", "0", "1e-12"),
]


def cmd(arm, gpu):
    label, cache, prefix, summ, transform, clip, minvar = arm
    return [
        PY, "geometry_resample.py",
        "--train-cache-dir", cache, "--cache-prefix", prefix,
        "--summaries-npz", summ, "--arm-label", label,
        "--output-dir", OUTDIR, "--fid-cache-dir", FID10,
        "--preproc-transform", transform, "--clip-value", clip,
        "--min-feature-variance", minvar,
        "--seeds", SEEDS, "--patch-indices", PATCHES, "--perms", PERMS,
        "--cuda-visible-devices", str(gpu),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        for a in ARMS:
            print(f"\n# {a[0]}\n" + " ".join(cmd(a, "<GPU>")))
        return
    os.makedirs(LOGS, exist_ok=True)
    os.chdir(SBI)
    pending = list(ARMS)
    slots = {g: None for g in GPUS}
    t0 = time.time()
    done, failed = [], {}

    def launch(arm, gpu):
        log = open(f"{LOGS}/{arm[0]}.log", "w")
        env = dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="3",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.4")
        p = subprocess.Popen(cmd(arm, gpu), cwd=SBI, env=env, stdout=log,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        print(f"[{time.time()-t0:7.0f}s] LAUNCH {arm[0]} on GPU{gpu} (pid {p.pid})", flush=True)
        return (arm[0], p, log)

    while pending or any(slots.values()):
        for g in GPUS:
            s = slots[g]
            if s and s[1].poll() is not None:
                nm, p, log = s
                log.close()
                slots[g] = None
                (done.append(nm) if p.returncode == 0 else failed.__setitem__(nm, p.returncode))
                print(f"[{time.time()-t0:7.0f}s] {'DONE' if p.returncode==0 else 'FAIL'} {nm} "
                      f"rc={p.returncode}", flush=True)
        for g in GPUS:
            if slots[g] is None and pending:
                slots[g] = launch(pending.pop(0), g)
        time.sleep(10)

    print(f"\n=== Phase D resample finished in {(time.time()-t0)/60:.1f} min ===")
    print(f"  done: {done}" + (f" | FAILED: {failed}" if failed else " | all OK"))


if __name__ == "__main__":
    main()
