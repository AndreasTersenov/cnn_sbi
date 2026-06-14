#!/usr/bin/env python3
"""OVERNIGHT Phase-2 compressor-architecture sweep — STAGE 1 (compressor training only).

Trains a VMIM compressor per architecture (auto-only, flat-local, seed 41, 80k steps, best_val) via
the VALIDATED npe_cnn_nbody_tomo.py --exit-after-compress path (only --compressor-arch varies). Each
dumps cnn_train/val/obs.npz + cnn_cache_meta.npz. The fiducial-summary build + RealNVP readout (FoM3)
are handled SEPARATELY (the fidsumm builder needs an arch-aware extension; done attended) so a builder
bug can't waste the night. Plain baseline = the existing cnn_none_s41 (FoM3 3139); NOT retrained.
Greedy over GPUs 0,2 (GPU 1 has a foreign tenant; GPU 3 never). Writes STATUS + .done markers.
"""
from __future__ import annotations
import subprocess, time, os
from pathlib import Path

REPO = "/mnt/home/tersenov/software/cnn_sbi"; SBI = f"{REPO}/scripts/sbi"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
TFDS = "nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180"
DDIR = "/home/tersenov/tensorflow_datasets"
FID = f"{SBI}/results/exploratory/cross_maps_campaign/full_sphere_cache_fiducial_10deg"
OUT = f"{SBI}/results/exploratory/flatsky_cross_2026_06/cnn_phase/arch_sweep_2026_06_13"
SEED, STEPS, OBS_PERM, OBS_PATCH = 41, 80000, 0, 90
GPUS = [0, 2]
# (label, arch) — plain baseline reused from cnn_none_s41 (3139), not retrained.
ARCHS = [("resnet50_gn", "resnet50_gn"), ("resnet_small", "resnet_small"),
         ("plain_attn", "plain_attn"), ("resnet18", "resnet18")]


def cnn_cmd(label, arch, gpu):
    d = f"{OUT}/cnn_{label}_s{SEED}"
    return [
        PY, "-u", "npe_cnn_nbody_tomo.py", "--train-compressor", "--exit-after-compress",
        "--cnn-map-route", "flat_local", "--cross-op", "none",
        "--cross-tfds-name", TFDS, "--cross-tfds-data-dir", DDIR, "--fiducial-obs-cache", FID,
        "--harmonic-cache-regime", "nobnt", "--harmonic-normalize-input-channels",
        "--cnn-perm-split", "0-4:5-6", "--zero-mean-maps", "--map-kind", "nbody", "--seed", str(SEED),
        "--field-size", "10", "--field-npix", "80", "--nbins", "4", "--tomo-bin-indices", "1,2,3,4",
        "--compressor-arch", arch, "--compressor-dim", "10",
        "--compressor-dense-width", "256", "--compressor-conv-channels", "64,128,256",
        "--compressor-steps", str(STEPS), "--compressor-batch-size", "128",
        "--compressor-lr", "0.0005", "--compressor-checkpoint-policy", "best_val", "--no-wandb",
        "--harmonic-obs-perm", str(OBS_PERM), "--harmonic-obs-patch-idx", str(OBS_PATCH),
        "--cuda-visible-devices", str(gpu),
        "--save-dir", d, "--cache-dir", f"{d}/cache",
        "--posterior-out", f"{d}/posterior.npy", "--figure-out", f"{d}/corner.pdf",
    ]


def status(msg):
    Path(OUT).mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(f"{OUT}/STATUS_OVERNIGHT.md", "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def main():
    Path(OUT).mkdir(parents=True, exist_ok=True)
    status(f"=== STAGE 1 compressor-arch sweep START: {[a[0] for a in ARCHS]} on GPUs {GPUS} ===")
    pending = list(ARCHS)
    slots = {g: None for g in GPUS}            # gpu -> (label, Popen, logfile)
    while pending or any(slots.values()):
        for g in GPUS:
            if slots[g] is None and pending:
                label, arch = pending.pop(0)
                d = Path(f"{OUT}/cnn_{label}_s{SEED}"); d.mkdir(parents=True, exist_ok=True)
                if (d / "cache" / "cnn_train.npz").exists():
                    status(f"SKIP {label} (cache exists)"); continue
                lg = open(d / "train.log", "w")
                env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false",
                           XLA_PYTHON_CLIENT_MEM_FRACTION="0.85", PYTHONUNBUFFERED="1")
                try:
                    p = subprocess.Popen(cnn_cmd(label, arch, g), cwd=SBI, env=env,
                                         stdout=lg, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                    slots[g] = (label, p, lg); status(f"LAUNCH {label} (arch={arch}) GPU{g} pid {p.pid}")
                except Exception as exc:
                    status(f"FAIL-LAUNCH {label}: {exc}"); lg.close()
        time.sleep(30)
        for g in GPUS:
            if slots[g] is not None:
                label, p, lg = slots[g]
                if p.poll() is not None:
                    lg.close()
                    ok = (Path(f"{OUT}/cnn_{label}_s{SEED}/cache/cnn_train.npz")).exists()
                    Path(f"{OUT}/.done_{label}").write_text(f"rc={p.returncode} cache={ok}\n")
                    status(f"DONE {label} rc={p.returncode} cache_ok={ok}")
                    slots[g] = None
    status("=== STAGE 1 COMPLETE (all compressors trained; fidsumm+readout handled attended) ===")
    Path(f"{OUT}/.STAGE1_DONE").write_text("done\n")


if __name__ == "__main__":
    main()
