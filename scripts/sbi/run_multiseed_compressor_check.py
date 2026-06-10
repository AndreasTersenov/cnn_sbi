#!/usr/bin/env python
"""Multi-COMPRESSOR-seed robustness check: is the CNN product no-cross-gain robust to the
compressor draw (not just the MAF seed)?

The campaign used ONE compressor (seed 41) per arm. This trains 2 more compressor seeds (42, 43)
for BOTH product and auto-only (none) — the fair per-seed product-vs-auto comparison — and runs each
through the same pipeline as seed 41: compressor (--exit-after-compress) -> fiducial summaries (9000
obs, G1-checked) -> population sweep (retrain 3 MAF seeds + 9000-obs median). Identical recipe to
seed 41 (plain CNN, 80k steps, NO grad-clip — product/none trained clean without it).

Phase-barriered (compressor -> summaries -> sweep), greedy 2-GPU scheduler (GPU 1+2 only; 0/3 have
foreign tenants). Detached: setsid nohup python run_multiseed_compressor_check.py &. --dry-run prints.
Writes MULTISEED_COMPRESSOR_CHECK.md when done.
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
MS = f"{CNN}/multiseed"
LOGS = f"{MS}/logs"
GPUS = [1, 2]
JOBS = [("none", 42), ("none", 43), ("product", 42), ("product", 43)]
OBS_PERM, OBS_PATCH = 0, 90


def compressor_cmd(op, seed, gpu, steps):
    d = f"{CNN}/cnn_{op}_s{seed}"
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
            "--compressor-checkpoint-policy", "best_val", "--no-wandb",
            "--harmonic-obs-perm", str(OBS_PERM), "--harmonic-obs-patch-idx", str(OBS_PATCH),
            "--cuda-visible-devices", str(gpu), "--save-dir", d, "--cache-dir", f"{d}/cache",
            "--posterior-out", f"{d}/posterior.npy", "--figure-out", f"{d}/corner.pdf"]


def fidsumm_cmd(op, seed, gpu):
    d = f"{CNN}/cnn_{op}_s{seed}"
    meta = dict(np.load(f"{d}/cache/cnn_cache_meta.npz", allow_pickle=True))
    return [PY, "build_fiducial_summaries_cnn.py", "--arm-label", f"flat_{op}_s{seed}",
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
    return [PY, "population_sweep_flatsky.py", "--train-cache-dir", f"{CNN}/cnn_{op}_s{seed}/cache",
            "--cache-prefix", "cnn", "--arm-label", f"flat_{op}_s{seed}",
            "--fiducial-summaries-npz", f"{MS}/fiducial_summaries/fiducial_summaries_{op}_s{seed}.npz",
            "--output-dir", f"{MS}/population_sweep/{op}_s{seed}",
            "--preproc-transform", "none", "--clip-value", "0", "--min-feature-variance", "1e-12",
            "--seeds", "41,42,43", "--n-obs", "9000", "--max-perm", "50", "--m-samples", "2000",
            "--cuda-visible-devices", str(gpu)]


def run_phase(name, cmd_fn, steps=None):
    """Greedy 2-GPU scheduler over JOBS for one phase; barrier at the end."""
    print(f"\n===== PHASE {name} =====", flush=True)
    os.makedirs(LOGS, exist_ok=True)
    pending = list(JOBS); slots = {g: None for g in GPUS}; t0 = time.time(); failed = {}
    def launch(job, gpu):
        op, seed = job
        # Command construction can itself fail (e.g. fidsumm_cmd np.load's the
        # compressor cache meta of a job that FAILed in the previous phase).
        # SKIP the chain instead of crashing the whole driver.
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
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                   CNN_CPU_THREADS="8")
        p = subprocess.Popen(c, cwd=SBI, env=env, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL)
        print(f"[{time.time()-t0:6.0f}s] LAUNCH {name} {op}_s{seed} GPU{gpu} (pid {p.pid})", flush=True)
        return (job, p, log)
    while pending or any(slots.values()):
        for g in GPUS:
            s = slots[g]
            if s and s[1].poll() is not None:
                (op, seed), p, log = s; log.close(); slots[g] = None
                tag = f"{op}_s{seed}"
                if p.returncode != 0:
                    failed[tag] = p.returncode
                print(f"[{time.time()-t0:6.0f}s] {'DONE' if p.returncode==0 else 'FAIL'} {name} {tag}",
                      flush=True)
        for g in GPUS:
            if slots[g] is None and pending:
                slots[g] = launch(pending.pop(0), g)
        time.sleep(10)
    if failed:
        print(f"  [{name}] FAILED: {failed}", flush=True)
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--compressor-steps", type=int, default=80000)
    args = ap.parse_args()
    os.chdir(SBI)
    if args.dry_run:
        for op, seed in JOBS:
            print(f"\n# {op}_s{seed} compressor:\n" + " ".join(compressor_cmd(op, seed, "<GPU>", args.compressor_steps)))
        print("\n(then fiducial summaries + population sweep per job)")
        return
    os.makedirs(MS, exist_ok=True)
    t0 = time.time()
    f1 = run_phase("compressor", compressor_cmd, steps=args.compressor_steps)
    f2 = run_phase("fidsumm", fidsumm_cmd)
    f3 = run_phase("sweep", sweep_cmd)

    # --- comparison ---
    def med(op, seed):
        f = Path(MS) / "population_sweep" / f"{op}_s{seed}" / "median_summary.json"
        return json.load(open(f))["fom3"] if f.exists() else None
    s41 = {op: json.load(open(Path(CNN) / "population_sweep" / f"flat_{op}" / "median_summary.json"))["fom3"]
           for op in ("none", "product")}
    L = ["# Multi-compressor-seed check — does the product no-gain survive the compressor draw?\n",
         "Pooled 9000-obs median FoM3 per compressor seed (each = own compressor + 3-MAF-seed pooled sweep).\n",
         "| compressor seed | auto-only | +product | product/auto |", "|---|---|---|---|"]
    ratios = {}
    for seed in (41, 42, 43):
        a = s41["none"] if seed == 41 else med("none", seed)
        p = s41["product"] if seed == 41 else med("product", seed)
        if a and p:
            ratios[seed] = p / a
        r = f"{p/a:.2f}×" if (a and p) else "—"
        L.append(f"| {seed}{' (orig)' if seed==41 else ''} | {a:.0f} | {p:.0f} | {r} |"
                 if (a and p) else f"| {seed} | {a or '—'} | {p or '—'} | — |")
    # Verdict derived from the computed ratios — never asserted unconditionally.
    if len(ratios) < 3:
        verdict = (f"**Verdict:** INCOMPLETE — only {sorted(ratios)} of (41, 42, 43) have sweep "
                   "summaries; no robustness conclusion.")
    elif all(r <= 1.0 for r in ratios.values()):
        verdict = ("**Verdict:** product/auto ≤ 1 across all compressor seeds ⇒ the no-cross-gain is "
                   "robust to the compressor draw (not just the MAF seed).")
    else:
        gain = ", ".join(f"s{s} {r:.2f}×" for s, r in sorted(ratios.items()) if r > 1.0)
        verdict = (f"**Verdict:** product/auto > 1 for {gain} ⇒ the no-cross-gain is NOT robust to "
                   "the compressor draw — consistent with optimization-limited cross extraction; "
                   "interpret per-seed before reframing the headline.")
    L += ["", verdict,
          "" if not (f1 or f2 or f3) else f"\n⚠ FAILURES: compressor={f1} fidsumm={f2} sweep={f3}"]
    Path(MS, "MULTISEED_COMPRESSOR_CHECK.md").write_text("\n".join(L))
    print(f"\n=== multiseed check done in {(time.time()-t0)/60:.1f} min -> {MS}/MULTISEED_COMPRESSOR_CHECK.md ===")


if __name__ == "__main__":
    main()
