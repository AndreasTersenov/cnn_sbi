#!/usr/bin/env python3
"""OVERNIGHT Phase-2 — STAGE 2: fiducial-summary build + RealNVP readout for each trained arch.

For every arch with a trained compressor cache (from stage 1) and no readout yet: build its 9000-obs
fiducial summaries (arch-aware build_fiducial_summaries_cnn.py, with a per-arch G1 reproduction check
as a safety net), then read out with the FIXED best NDE (sbi_lens RealNVP 4x128, screen 2-seed/1000-
obs) -> 9000-obs-style median FoM3. Idempotent (skips done archs). Plain baseline = 3139. Writes
SUMMARY_ARCH.md ranked by FoM3. Run after compressors complete (or incrementally).
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np

REPO = "/mnt/home/tersenov/software/cnn_sbi"; SBI = f"{REPO}/scripts/sbi"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
TFDS = "nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180"
FID_CACHE = f"{SBI}/results/exploratory/cross_maps_campaign/full_sphere_cache_fiducial_10deg"
SW = f"{SBI}/results/exploratory/flatsky_cross_2026_06/cnn_phase/arch_sweep_2026_06_13"
ARCHS = ["resnet50_gn", "resnet_small", "plain_attn", "resnet18"]
PLAIN_BASELINE = 3139


def sh(cmd, log):
    with open(log, "w") as f:
        return subprocess.call(cmd, cwd=SBI, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)


def process(label, gpu, mem):
    d = Path(f"{SW}/cnn_{label}_s41")
    cache = d / "cache"
    if not (cache / "cnn_train.npz").exists():
        return f"{label}: SKIP (no cache yet)"
    read_json = d / "readout" / "median_summary.json"
    if read_json.exists():
        return f"{label}: DONE {json.load(open(read_json))['fom3']:.0f}"
    meta = dict(np.load(cache / "cnn_cache_meta.npz", allow_pickle=True))
    arch = str(meta["compressor_arch"]); nch = int(meta["cnn_input_channels"])
    fid_out = f"{SW}/fidsumm_{label}.npz"
    # --- fidsumm build (arch-aware, with G1 check) ---
    if not Path(fid_out).exists():
        cmd = [PY, "-u", "build_fiducial_summaries_cnn.py", "--arm-label", label,
               "--params-pkl", str(meta["compressor_params_path"]),
               "--state-pkl", str(meta["compressor_state_path"]),
               "--expect-params-sha", str(meta["compressor_params_sha256"]),
               "--expect-state-sha", str(meta["compressor_state_sha256"]),
               "--n-channels", str(nch), "--dim", "10",
               "--conv-channels", "64,128,256", "--dense-width", "256",
               "--pool-window", "16", "--pool-stride", "8", "--compressor-arch", arch,
               "--cross-op", "none", "--nbins", "4", "--flatsky-roll-frac", "0.10",
               "--cross-tfds-name", TFDS, "--cross-tfds-data-dir", "/home/tersenov/tensorflow_datasets",
               "--channel-rms-nsample", "8000", "--fid-cache-dir", FID_CACHE,
               "--regime", "nobnt", "--cosmo-id", "cosmo_fiducial", "--perms", "0-49",
               "--g1-obs-npz", str(cache / "cnn_obs.npz"), "--g1-perm", "0", "--g1-patch", "90",
               "--out", fid_out, "--cuda-visible-devices", str(gpu)]
        import os
        env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false",
                   XLA_PYTHON_CLIENT_MEM_FRACTION=str(mem), PYTHONUNBUFFERED="1")
        rc = subprocess.call(cmd, cwd=SBI, env=env, stdout=open(f"{SW}/fidsumm_{label}.log", "w"),
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        log = Path(f"{SW}/fidsumm_{label}.log").read_text()
        if rc != 0 or "PASS" not in log or not Path(fid_out).exists():
            return f"{label}: FIDSUMM FAIL (rc={rc}, G1 {'PASS' if 'PASS' in log else 'NOPASS'}) — skip"
    # --- RealNVP readout (screen 2-seed/1000-obs) ---
    odir = d / "readout"; odir.mkdir(parents=True, exist_ok=True)
    import os
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_PYTHON_CLIENT_MEM_FRACTION=str(mem), PYTHONUNBUFFERED="1")
    cmd = [PY, "-u", "train_nde_from_compressed.py", "--train-cache-dir", str(cache),
           "--cache-prefix", "cnn", "--fiducial-summaries-npz", fid_out,
           "--arm-label", f"arch_{label}", "--output-dir", str(odir),
           "--nde-family", "sbilens_realnvp", "--nde-layers", "4", "--nde-hidden", "128",
           "--preproc-transform", "none", "--clip-value", "0", "--min-feature-variance", "1e-12",
           "--seeds", "41,42", "--n-obs", "1000", "--m-samples", "2000", "--cuda-visible-devices", str(gpu)]
    rc = subprocess.call(cmd, cwd=SBI, env=env, stdout=open(odir / "run.log", "w"),
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    if read_json.exists():
        return f"{label}: FoM3 {json.load(open(read_json))['fom3']:.0f}"
    return f"{label}: READOUT FAIL (rc={rc})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="2"); ap.add_argument("--mem", default="0.3")
    a = ap.parse_args()
    results = {}
    for label in ARCHS:
        msg = process(label, a.gpu, float(a.mem))
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
        m = Path(f"{SW}/cnn_{label}_s41/readout/median_summary.json")
        if m.exists():
            results[label] = float(json.load(open(m))["fom3"])
    # SUMMARY
    rows = sorted(results.items(), key=lambda kv: -kv[1])
    lines = ["# Phase-2 compressor-arch sweep — RealNVP readout (screen 2-seed/1000-obs)\n",
             f"Baseline: plain conv = **{PLAIN_BASELINE}** (FoM3, the established best CNN). "
             "L1+product = 2875. Each arch = same VMIM recipe, seed 41, 80k steps, best_val, read out "
             "by the FIXED sbi_lens RealNVP 4x128.\n",
             "| arch | FoM3 | vs plain |", "|---|---|---|",
             f"| plain (baseline) | {PLAIN_BASELINE} | — |"]
    for label, f in rows:
        lines.append(f"| {label} | {f:.0f} | {f-PLAIN_BASELINE:+.0f} ({f/PLAIN_BASELINE:.2f}×) |")
    lines.append("\nNOTE: screen-level + UNCALIBRATED. Promote any winner to full 3-seed/9000-obs + GATE C "
                 "(tarp_stratified_val_nde.py) before any claim. Watch train/val gap (over-capacity at 899 cosmos).")
    Path(f"{SW}/SUMMARY_ARCH.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {SW}/SUMMARY_ARCH.md", flush=True)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
