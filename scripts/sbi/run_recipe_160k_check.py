#!/usr/bin/env python
"""Recipe-level check: does a HARDER-TRAINED compressor change the CNN cross story?

The multiseed check (MULTISEED_COMPRESSOR_CHECK.md) showed the product/auto effect flips sign
with the compressor seed (0.94/1.10/0.98) and every CNN product seed stays <= 0.85x L1 product —
the seed-level rescue of the optimization-limited hypothesis is falsified, the RECIPE level is
untested. This trains compressor seeds 42, 43 for {none, product} at a heavier recipe —
160k steps (2x) + --compressor-val-batches 16 (de-noised best_val selection) — and runs each
through the identical downstream pipeline (fidsumm -> jit population sweep, 3 MAF seeds,
9000-obs median). Paired with the 80k multiseed results per (op, seed).

Phase-barriered greedy scheduler. Detached: setsid nohup python run_recipe_160k_check.py &.
Writes RECIPE_160K_CHECK.md (data-derived verdict) when done.
"""
import argparse, json, os, subprocess, time
import numpy as np
from pathlib import Path

REPO = "/mnt/home/tersenov/software/cnn_sbi"; SBI = f"{REPO}/scripts/sbi"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
TFDS = "nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180"
DDIR = "/home/tersenov/tensorflow_datasets"
FID = f"{SBI}/results/exploratory/cross_maps_campaign/full_sphere_cache_fiducial_10deg"
CNN = f"{SBI}/results/exploratory/flatsky_cross_2026_06/cnn_phase"
MS = f"{CNN}/multiseed_160k"
LOGS = f"{MS}/logs"
JOBS = [("none", 42), ("none", 43), ("product", 42), ("product", 43)]
OBS_PERM, OBS_PATCH = 0, 90
STEPS = 160000
VAL_BATCHES = 16


def compressor_cmd(op, seed, gpu, steps):
    d = f"{CNN}/cnn_{op}_s{seed}_160k"
    return [PY, "npe_cnn_nbody_tomo.py", "--train-compressor", "--exit-after-compress",
            "--cnn-map-route", "flat_local", "--cross-op", op,
            "--cross-tfds-name", TFDS, "--cross-tfds-data-dir", DDIR, "--fiducial-obs-cache", FID,
            "--harmonic-cache-regime", "nobnt", "--harmonic-normalize-input-channels",
            "--cnn-perm-split", "0-4:5-6", "--zero-mean-maps", "--map-kind", "nbody",
            "--seed", str(seed), "--field-size", "10", "--field-npix", "80",
            "--nbins", "4", "--tomo-bin-indices", "1,2,3,4",
            "--compressor-arch", "plain", "--compressor-dim", "10", "--compressor-dense-width", "256",
            "--compressor-conv-channels", "64,128,256", "--compressor-steps", str(steps),
            "--compressor-batch-size", "128", "--compressor-lr", "0.0005",
            "--compressor-checkpoint-policy", "best_val",
            "--compressor-val-batches", str(VAL_BATCHES), "--no-wandb",
            "--harmonic-obs-perm", str(OBS_PERM), "--harmonic-obs-patch-idx", str(OBS_PATCH),
            "--cuda-visible-devices", str(gpu), "--save-dir", d, "--cache-dir", f"{d}/cache",
            "--posterior-out", f"{d}/posterior.npy", "--figure-out", f"{d}/corner.pdf"]


def fidsumm_cmd(op, seed, gpu):
    d = f"{CNN}/cnn_{op}_s{seed}_160k"
    meta = dict(np.load(f"{d}/cache/cnn_cache_meta.npz", allow_pickle=True))
    return [PY, "build_fiducial_summaries_cnn.py", "--arm-label", f"flat_{op}_s{seed}_160k",
            "--params-pkl", str(meta["compressor_params_path"]),
            "--state-pkl", str(meta["compressor_state_path"]),
            "--expect-params-sha", str(meta["compressor_params_sha256"]),
            "--expect-state-sha", str(meta["compressor_state_sha256"]),
            "--n-channels", str(int(meta["cnn_input_channels"])), "--dim", "10",
            "--conv-channels", "64,128,256", "--dense-width", "256", "--pool-window", "16",
            "--pool-stride", "8", "--cross-op", op, "--nbins", "4", "--flatsky-roll-frac", "0.10",
            "--cross-tfds-name", TFDS, "--cross-tfds-data-dir", DDIR, "--channel-rms-nsample", "8000",
            "--fid-cache-dir", FID, "--regime", "nobnt", "--cosmo-id", "cosmo_fiducial",
            "--perms", "0-49", "--g1-obs-npz", f"{d}/cache/cnn_obs.npz",
            "--g1-perm", str(OBS_PERM), "--g1-patch", str(OBS_PATCH),
            "--out", f"{MS}/fiducial_summaries/fiducial_summaries_{op}_s{seed}.npz",
            "--cuda-visible-devices", str(gpu)]


def sweep_cmd(op, seed, gpu):
    return [PY, "population_sweep_flatsky.py",
            "--train-cache-dir", f"{CNN}/cnn_{op}_s{seed}_160k/cache",
            "--cache-prefix", "cnn", "--arm-label", f"flat_{op}_s{seed}_160k",
            "--fiducial-summaries-npz", f"{MS}/fiducial_summaries/fiducial_summaries_{op}_s{seed}.npz",
            "--output-dir", f"{MS}/population_sweep/{op}_s{seed}",
            "--preproc-transform", "none", "--clip-value", "0", "--min-feature-variance", "1e-12",
            "--seeds", "41,42,43", "--n-obs", "9000", "--max-perm", "50", "--m-samples", "2000",
            "--cuda-visible-devices", str(gpu)]


def run_phase(name, cmd_fn, gpus, steps=None):
    """Greedy scheduler over JOBS for one phase; barrier at the end."""
    print(f"\n===== PHASE {name} =====", flush=True)
    os.makedirs(LOGS, exist_ok=True)
    pending = list(JOBS); slots = {g: None for g in gpus}; t0 = time.time(); failed = {}

    def launch(job, gpu):
        op, seed = job
        try:
            c = cmd_fn(op, seed, gpu, steps) if steps is not None else cmd_fn(op, seed, gpu)
        except Exception as exc:
            failed[f"{op}_s{seed}"] = f"cmd-build: {exc}"
            print(f"[{time.time()-t0:6.0f}s] SKIP {name} {op}_s{seed} "
                  f"(command build failed: {exc})", flush=True)
            return None
        log = open(f"{LOGS}/{name}_{op}_s{seed}.log", "w")
        env = dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="3",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.5",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", CNN_CPU_THREADS="8")
        p = subprocess.Popen(c, cwd=SBI, env=env, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL)
        print(f"[{time.time()-t0:6.0f}s] LAUNCH {name} {op}_s{seed} GPU{gpu} (pid {p.pid})", flush=True)
        return (job, p, log)

    while pending or any(slots.values()):
        for g in gpus:
            s = slots[g]
            if s and s[1].poll() is not None:
                (op, seed), p, log = s; log.close(); slots[g] = None
                tag = f"{op}_s{seed}"
                if p.returncode != 0:
                    failed[tag] = p.returncode
                print(f"[{time.time()-t0:6.0f}s] {'DONE' if p.returncode==0 else 'FAIL'} {name} {tag}",
                      flush=True)
        for g in gpus:
            if slots[g] is None and pending:
                slots[g] = launch(pending.pop(0), g)
        time.sleep(10)
    if failed:
        print(f"  [{name}] FAILED: {failed}", flush=True)
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gpus", default="1,2")
    ap.add_argument("--compressor-steps", type=int, default=STEPS)
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    os.chdir(SBI)
    if args.dry_run:
        for op, seed in JOBS:
            print(f"\n# {op}_s{seed} compressor:\n"
                  + " ".join(compressor_cmd(op, seed, "<GPU>", args.compressor_steps)))
        print("\n(then fiducial summaries + jit population sweep per job)")
        return
    os.makedirs(MS, exist_ok=True)
    t0 = time.time()
    f1 = run_phase("compressor", compressor_cmd, gpus, steps=args.compressor_steps)
    f2 = run_phase("fidsumm", fidsumm_cmd, gpus)
    f3 = run_phase("sweep", sweep_cmd, gpus)

    # --- comparison vs the 80k recipe (paired per (op, seed)) and vs L1 product ---
    def med80(op, seed):
        f = Path(CNN) / "multiseed" / "population_sweep" / f"{op}_s{seed}" / "median_summary.json"
        return json.load(open(f))["fom3"] if f.exists() else None

    def med160(op, seed):
        f = Path(MS) / "population_sweep" / f"{op}_s{seed}" / "median_summary.json"
        return json.load(open(f))["fom3"] if f.exists() else None

    l1p_f = Path(SBI) / "results/exploratory/flatsky_cross_2026_06/population_sweep/flat_product/median_summary.json"
    l1p = json.load(open(l1p_f))["fom3"] if l1p_f.exists() else None

    L = [f"# Recipe-level check — {args.compressor_steps//1000}k steps + val-batches {VAL_BATCHES} "
         "vs the 80k baseline\n",
         "Pooled 3-MAF 9000-obs median FoM3, paired per (arm, compressor seed).\n",
         "| arm/seed | 80k | 160k | 160k/80k | CNN/L1 product (160k) |", "|---|---|---|---|---|"]
    vals = {}
    for op, seed in JOBS:
        a80, a160 = med80(op, seed), med160(op, seed)
        vals[(op, seed)] = (a80, a160)
        r = f"{a160/a80:.2f}×" if (a80 and a160) else "—"
        rl = f"{a160/l1p:.2f}×" if (op == "product" and a160 and l1p) else "—"
        L.append(f"| {op}_s{seed} | {a80:.0f} | {a160:.0f} | {r} | {rl} |"
                 if (a80 and a160) else f"| {op}_s{seed} | {a80 or '—'} | {a160 or '—'} | — | — |")
    ok = {k: v for k, v in vals.items() if v[0] and v[1]}
    if len(ok) == len(JOBS):
        lift = {k: v[1] / v[0] for k, v in ok.items()}
        prod_lift = np.mean([lift[("product", s)] for s in (42, 43)])
        auto_lift = np.mean([lift[("none", s)] for s in (42, 43)])
        rho = max(ok[("product", s)][1] / l1p for s in (42, 43)) if l1p else float("nan")
        verdict = (f"**Observed:** mean 160k/80k lift — auto {auto_lift:.2f}×, product "
                   f"{prod_lift:.2f}×; best CNN/L1(product) at 160k = {rho:.2f}× "
                   f"(80k range was 0.83–0.85× for these seeds). "
                   + ("Product-specific recipe gain (product lift exceeds auto lift)."
                      if prod_lift > auto_lift * 1.05 else
                      "No product-specific recipe gain (product lift ≈/≤ auto lift).")
                   + (" CNN closes on L1 product." if (l1p and rho > 0.95) else
                      (" CNN still well below L1 product." if l1p and rho < 0.9 else "")))
        L += ["", verdict]
    else:
        L += ["", "**Verdict:** INCOMPLETE — missing sweep summaries; no conclusion."]
    L += ["" if not (f1 or f2 or f3) else f"\n⚠ FAILURES: compressor={f1} fidsumm={f2} sweep={f3}",
          "\nNB the 160k recipe bundles TWO changes vs 80k: 2× steps AND de-noised best_val "
          f"(val-batches {VAL_BATCHES} vs 1). If it moves, ablate before attributing."]
    Path(MS, "RECIPE_160K_CHECK.md").write_text("\n".join(L))
    print(f"\n=== recipe check done in {(time.time()-t0)/60:.1f} min -> {MS}/RECIPE_160K_CHECK.md ===")


if __name__ == "__main__":
    main()
