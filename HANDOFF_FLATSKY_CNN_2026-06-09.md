# HANDOFF — flat-sky cross, CNN phase (start here)

**Date:** 2026-06-09. **For:** the next Claude Code session.
**Felt fiber:** `.felt/flatsky-cross-2026-06/` (active). **Branch:** see `git status` (continue on it).

## One-paragraph state

We **de-leaked** the tomographic cross-maps: rebuilt them **patch-local (flat-sky)** instead of the
leaky full-sphere harmonic way (`CROSS_MAP_LEAKAGE_FINDING.md`), recomputed **L1**, trained, and
ran the **full calibrated population analysis**. Result (definitive, in `FLATSKY_CROSS_RESULT.md`):
**~92% of the full-sphere L1 auto+cross gain was leakage** — the physically-buildable patch-local
cross retains only **+21% FoM3** (pooled 3-seed, 9000-obs median) vs the full-sphere's **+288%**.
The surviving signal is the **pointwise product** operator (κᵢ·κⱼ = ξᵢⱼ, +20%); the **convolution**
(Zürcher alm-product analog) gives only +4% (~99% leakage for that operator), and is seed/sample-
fragile. Calibrated: **TARP-DRP ✓ + SBC ✓** (L-C2ST N/A — underpowered at high-dim L1). **The L1 side
is DONE.** The open phase is the **CNN side**: same de-leaked cross-maps, CNN-VMIM compressor, for
the physically-defensible **L1-vs-CNN** comparison (the original scientific question).

## Read first (in order)

1. **`FLATSKY_CROSS_RESULT.md`** — the L1 result + full table + leakage accounting + reproduce steps.
2. **`CROSS_MAP_LEAKAGE_FINDING.md`** — why we did this (the leakage).
3. Memory index `…/memory/MEMORY.md` — esp. `project_flatsky_cross_deleaked_result`,
   `reference_lc2st_underpowered_highdim_l1`, `project_resnet_bn_contamination` (CNN!),
   `feedback_gpu1_only`, `project_pool_haircut_invariant_to_architecture`,
   `feedback_benchmark_dont_assume`, `feedback_no_pkill_self_match`.
4. This doc + `.felt/flatsky-cross-2026-06/flatsky-cross-2026-06.md` (live status).

## What is DONE (do not redo)

- **Operators** `scripts/sbi/flatsky_cross.py` — np/torch/**jax** backends for conv + product +
  `none`, bit-matched (GATE A1). The **`build_channels_jax`** backend is ready for the CNN.
- **Frozen L1 noise σ** `freeze_flatsky_cross_noise.py` → `…/flatsky_cross_2026_06/flatsky_cross_noise_sigma.npz`
  (per-(channel,scale), R=48, faithful sphere-SHT noise). *L1-only; CNN uses per-channel RMS instead.*
- **L1 pipeline** wired into `npe_l1norm_cross_jaxili_nbody_tomo.py`:
  `--cross-maps-route flat_local --cross-op {none,conv,product,both}` (+ build-both-slice via
  `--flatsky-both-cache`, per-channel frozen-percentile SNR binning, clamp_overflow). Meta carries
  full provenance.
- **L1 matrix** `run_flatsky_l1_matrix.py` (build-both-once-slice; 4 arms × 3 seeds). Results in
  `…/flatsky_cross_2026_06/l1_matrix/`.
- **GATE C** TARP `run_flatsky_gate_c_tarp.py` (✓), SBC `compute_sbc_from_tarp_dumps.py` (✓), L-C2ST
  `run_flatsky_gate_c_lc2st.py` (N/A high-dim). Figs in `…/gate_c/`.
- **Population sweep** `run_flatsky_population_sweep.py` → `population_sweep/<arm>/median_summary.json`.
  Obs datavectors precomputed: `precompute_fiducial_both_datavectors.py` → `fiducial_both_datavectors.npz`
  (36000×3200; per-arm slices `gate_c/lc2st/fiducial_summaries_{arm}.npz`).
- **Plots** `plot_flatsky_diagnostics.py`, `plot_l1_matrix_corners.py`, `compute_l1_2d_areas.py`,
  `plot_tarp_flatsky_colored.py`, `plot_flatsky_showcase.py`, `representative_corner_flatsky.py` +
  `plot_representative_corners.py`. All in `…/flatsky_cross_2026_06/figs/`.

## NEXT PHASE — CNN arms (the work)

**Goal:** train CNN-VMIM compressors on the same de-leaked patch-local cross-maps, calibrate, and
compare **L1 vs CNN** on the physically-defensible cross. Mirror the L1 phase.

**Wire it (`scripts/sbi/npe_cnn_nbody_tomo.py`):** add a `--cnn-map-route flat_local` +
`--cross-op {none,conv,product,both}` route that:
1. reads **autos ch 0-3** of the cross TFDS (`grid_10deg_80px_nonoverlap180`) — NOT the leaky cross
   channels 4-9; same autos across arms (confound-free).
2. builds the flat cross **ON-DEVICE, batched in JAX** (`flatsky_cross.build_channels_jax`, roll 0.10)
   inside the compressor input step — **NOT** in a CPU `tf.data` map (starves the GPU; GATE A2 lesson).
3. normalizes by **per-channel RMS** (frozen at fiducial, like the harmonic route's `channel_scale` —
   NOT the L1 frozen-σ table; CNN just whitens channel amplitudes). Compute it once from the fiducial
   cross channels and freeze (record in meta).
4. obs = fiducial cache ch 0-3 → build cross → compress → sample (reuse `--fiducial-obs-cache`).

**Key differences from L1 (plan for them):**
- **build-both-slice does NOT transfer.** CNN compresses maps→10-dim; you can't slice the summary, so
  **each arm needs its own compressor** (different channel input). 4 arms × (compressor+NDE) × 3 seeds
  is more expensive than L1. The on-device cross is cheap; the compressor training is the cost.
- **Architecture:** multi-channel input → use **`resnet50_gn` or `plain`**, NEVER stock BatchNorm
  (`project_resnet_bn_contamination`: BN collapses FoM3 on cosmology-mixed batches).
- **Example-disjoint split by perm** (compressor perms 0-4, NDE 5-6) — the CNN convention (`run_phase_c_10deg.py`).
- **L-C2ST WORKS for CNN** (10-dim summary, not high-dim) — so CNN GATE C = full **TARP+SBC+L-C2ST**
  (unlike L1 where L-C2ST was underpowered).
- **NDE confound:** CNN uses sbi_lens RealNVP, L1 uses jaxili MAF (`project_nde_architecture_mismatch`).
  For a clean L1-vs-CNN, consider the common-MAF approach the prior 10° campaign used (Phase D ran both
  arms through the identical jaxili MAF). Decide with Andreas.

**Reuse:** `run_phase_c_10deg.py` is the CNN+L1 orchestrator template (it already does CNN
`--cnn-map-route tfds_cross`); `tarp_stratified_val.py` / `population_sweep_flatsky.py` patterns for
GATE C + sweep. The CNN compressor cache + NDE retrain pattern is in `run_phase_c_10deg.py:cnn_cmd`.

## Hard guardrails (non-negotiable)

- **GPU 1 only** by default (`feedback_gpu1_only`); GPUs 1+2 were granted *this* session — re-confirm
  with Andreas before using 2.
- **Patch-local cross ONLY** (never full-sphere). **Same autos (ch 0-3) across all arms.**
- **On-device cross** (jax.fft), never CPU tf.data map. **≤2 concurrent TFDS readers** — 4 thrash disk
  (throughput collapsed 729→40/s). Build datavectors with a single/few passes.
- **Lead with the POOLED 9000-obs median**, not single-obs (single-obs inflated conv via the per-seed
  metric + a favorable patch). Report σ/2D; FoM3 OK to headline (rule retired 2026-06-09) but quote
  σ/2D alongside.
- **Calibrate (TARP/SBC[/L-C2ST]) BEFORE trusting any contour.**
- **Memory limits:** XLA_MEM_FRACTION ≥0.4 for the high-dim datavectors (0.25 OOMed); `expandable_segments:True`.
- **Detached jobs:** `setsid nohup … &`, poll with `pgrep -f "[b]racket"`; **never `pkill -f` a
  self-matching pattern** (`feedback_no_pkill_self_match` — bit us again this session, exit 144).
- **git:** stage by path, never `git add .`; don't commit results/caches/figures unless asked.

## Open / backlog

- (CNN phase — the main work above.)
- L1 backlog: fixed-[-4,4] SNR binning robustness check; scale-matched product `[ψₛκᵢ][ψₛκⱼ]`;
  the `both` 16-ch MAF high-dim under-training (try L1-VMIM compression, NOT PCA).
- A population corner for L1 needed a re-sample (only metrics were saved) — done via
  `representative_corner_flatsky.py` (typical = perm16/patch23).

## Environment

- `conda run -n jaxili python …`, GPU 1 (`--cuda-visible-devices 1`), `XLA_PYTHON_CLIENT_MEM_FRACTION≈0.4`,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, titan has **no scheduler** (run directly).
- Data: cross TFDS `/home/tersenov/tensorflow_datasets/nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180`;
  fiducial cache `…/cross_maps_campaign/full_sphere_cache_fiducial_10deg` (autos ch 0-3 reusable).
