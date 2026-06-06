# PLAN — Phase C: the 4-arm × 3-seed definitive run (10°)

Status: **PLAN (awaiting Andreas sign-off).** Created 2026-06-06.
Campaign: `definitive-l1-vs-cnn-10deg-2026-06`. Prereqs: Phase A (data validated),
B-1 (CNN `tfds_cross` + smoke), B-2 (L1 `tfds_cross` + parity + smoke). All 4 arm
configs now runnable on the unified TFDS; L1 auto-only enabled (2026-06-06).

## Goal

Train the **4 arms × 3 seeds = 12 NDEs** at full budget on the unified 10° TFDS,
producing the trained checkpoints + compressed/datavector caches that Phase D's
diagnostics (geometry/spread/bias/SBC/L-C2ST) consume. Lead metric = **σ(w0) +
2D(Ωm,σ8) median over patches; report FoM3 but DON'T headline it** (fragile).

## The arm matrix

| # | arm | route | channel-mode | summary |
|---|-----|-------|--------------|---------|
| 1 | CNN auto+cross | `tfds_cross` | `auto_cross` (10 ch) | learned VMIM (d=10) |
| 2 | CNN auto-only  | `tfds_cross` | `auto_only` (4 ch)  | learned VMIM (d=10) |
| 3 | L1 auto+cross  | `tfds_cross` | `auto_cross` (10 ch) | wavelet ℓ₁ (2000-d) |
| 4 | L1 auto-only   | `tfds_cross` | `auto_only` (4 ch)  | wavelet ℓ₁ (800-d) |

- **Seeds: 41, 42, 43** (match the 20° campaign so Phase E is apples-to-apples).
- **Split (locked, B-1/B-2):** compressor/NDE example-disjoint by **perm** —
  compressor perms 0–4, NDE-train perms 5–6, all 899 train cosmos in both; NDE-val =
  `test` (cosmo 900–1299). Obs held out (fiducial cache).
- **CNN:** each seed = a full (VMIM compressor 80k-step + NDE) train — captures
  compressor seed variance, as at 20° (`phaseA_tfdata` trained per-seed compressors).
- **L1:** the ℓ₁ datavector is parameter-free → compute **once per arm** (cached `.npz`),
  then 3 MAF-NDE seeds reuse it. So 12 "jobs" = 6 CNN (compressor+NDE) + 2 L1 datavector
  builds + 6 L1 MAF seeds.

## Arm commands (full budget; only the data source differs across arms)

CNN (arms 1–2), per seed S ∈ {41,42,43}, `--channel-mode {auto_cross,auto_only}`:
```
python npe_cnn_nbody_tomo.py --train-compressor --cnn-map-route tfds_cross \
  --cross-tfds-name nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180 \
  --cross-tfds-data-dir /home/tersenov/tensorflow_datasets \
  --fiducial-obs-cache <FID> --harmonic-cache-regime nobnt \
  --harmonic-normalize-input-channels --channel-mode <MODE> --cnn-perm-split 0-4:5-6 \
  --zero-mean-maps --map-kind nbody --seed S --field-size 10 --field-npix 80 \
  --nbins 4 --tomo-bin-indices 1,2,3,4 \
  --compressor-arch plain --compressor-dim 10 --compressor-dense-width 256 \
  --compressor-conv-channels 64,128,256 --compressor-steps 80000 \
  --compressor-batch-size 128 --compressor-lr 0.0005 --compressor-checkpoint-policy best_val \
  --npe-samples 100000 --no-wandb --cuda-visible-devices <G> --save-dir <OUT>/cnn_<MODE>_s<S> ...
```
L1 (arms 3–4), per seed S, `--channel-mode {auto_cross,auto_only}`:
```
python npe_l1norm_cross_jaxili_nbody_tomo.py --cross-maps-route tfds_cross \
  --cross-tfds-name ... --cross-tfds-data-dir ... --fiducial-obs-cache-dir <FID> \
  --cross-noise-model channel_empirical_global --pca-components 0 \
  --channel-mode <MODE> --nde-perm-split 5-6 --nde-val-perm-split 0-1 \
  --nbins 4 --tomo-bin-indices 1,2,3,4 --field-size 10 --field-npix 80 \
  --n-scales 5 --l1-nbins 40 --l1-min-snr -13 --l1-max-snr 13 \
  --cross-map-auto-calibrate-snr --cross-snr-percentile 1.0 \
  --summary-transform log1p-zscore --clip-value 5 --seed S --cuda-visible-devices <G> \
  --save-dir <OUT>/l1_<MODE>_s<S> --cache-dir <OUT>/l1_<MODE>_cache ...
```
The L1 `--cache-dir` is **shared across the 3 seeds of an arm** so the ℓ₁ datavector is
computed once and reused (the MAF seed only re-trains the NDE).

## Prep edits before launch (small)
1. **`--nde-val-perm-split` (L1 script):** add a val perm filter (default `all` = current).
   Set **`0-1`** for Phase C → L1 NDE-val = test perms 0–1 (144k) instead of all 504k.
   Val only drives early-stopping (the comparison is at the fiducial), so trimming it is
   free; halves the L1 datavector cost. (Measured: 268 patches/s → train perms 5–6 324k
   ≈ 20 min + val 144k ≈ 9 min ≈ **29 min/arm**, vs ~51 min with full val.)
2. **(optional) bigger `compute_l1_batch`:** the smoke ran GPU at 28% (CPU/Python-bound).
   Benchmark `batch_size` 480→960 for the L1 datavector; adopt only if measured faster.

## Orchestration (GPU 1 + 2)
A small `run_phase_c_10deg.py` (mirrors the existing `run_*_campaign.py` pattern):
- **Phase C-1:** build the 2 L1 datavector caches (arm 3 on GPU 1, arm 4 on GPU 2) — the
  long pole (~29 min each, parallel). In parallel, start CNN compressors.
- **Phase C-2:** the 12 NDE/compressor jobs scheduled across GPU 1+2, ≤1 heavy job/GPU at
  a time (mem-frac 0.4 each; L1 MAF seeds are light and can pack). Pin every job; never 0/3.
- Estimated wall: ~3–5 h total (6 CNN compressor+NDE ≈ 20–40 min each; 6 L1 MAF seeds fast;
  2 L1 datavectors ≈ 29 min). Logs + a STATUS line per job.

## Outputs (per arm/seed) + comparison
- `posterior.npy` (100k samples at the reference obs patch), `.meta.json` (assert
  `pca_applied:false` for L1; `channel_empirical_global` for L1; cnn_input_channels),
  trained checkpoint, compressed/datavector cache, corner PDF.
- **`compare_phase_c.py`** (adapt `compare_cross_only.py`): pool the 3 seeds per arm,
  report per-arm **σ(Ωm,σ8,w0), 2D(Ωm,σ8) area, FoM3 (reported)**, and the L1/CNN ratios
  for auto-only and auto+cross. One overlay corner per probe.
- Reference obs: patch 90 (mid-latitude) for the headline table; the **per-patch population
  (Phase D)** is what the offset/coverage analysis uses, not this single patch.

## Decision metric / done condition
- Primary: **median σ(w0) and 2D(Ωm,σ8)** across the Phase-D patch population (FoM3 reported,
  not led). Per-arm value = 3-seed-pooled.
- Phase C done when all 12 checkpoints + caches exist, `pca_applied:false` verified on L1,
  and `compare_phase_c.py` emits the σ/2D table. Then Phase D (diagnostics) + Phase E (vs 20°).

## Guardrails
- GPU **1 + 2 only**, pinned (never 0/3); check `nvidia-smi` + load before each launch.
- Never PCA L1 (assert `--pca-components 0`). Don't headline FoM3. `--no-wandb` on all
  (the L1 smoke accidentally logged to wandb; Phase C runs pass `--no-wandb`).
- Never `git add`/commit without Andreas's OK.

## After sign-off
Prep edit (1) [+ optional (2) benchmark] → write `run_phase_c_10deg.py` → dry-run 1 arm/1 seed
(reduced steps) to confirm the orchestrator wiring → launch the full 12 → `compare_phase_c.py`.
