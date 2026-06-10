# HANDOFF — flat-sky cross, L1-vs-CNN (start here)

**Date:** 2026-06-10. **For:** the next Claude Code session (Fable 5).
**Branch:** `autoresearch/cnn-auto-push-18-20-2026` (continue on it; pushed to origin).
**Felt fiber:** `.felt/flatsky-cross-2026-06/` (active — read its top loop-status stanzas).
**Conda env:** `jaxili` (always `conda run -n jaxili python …` or the direct interpreter
`/home/tersenov/anaconda3/envs/jaxili/bin/python`).

---

## 1. One-paragraph state

We compared two weak-lensing summary statistics — the **wavelet-L1** datavector and a **CNN-VMIM**
learned compressor — for inferring cosmology from tomographic convergence maps, specifically on
**tomographic cross-maps**. Earlier we found the dramatic "CNN ≫ L1 on auto+cross" result was an
artifact of how the cross-maps were built (full-sphere harmonic construction leaks global information
into each patch; `CROSS_MAP_LEAKAGE_FINDING.md`). We rebuilt the cross-maps **patch-local (flat-sky)**
so a real survey could actually make them, recomputed everything, and calibrated it. **Result
(definitive, calibrated):** on the physically-buildable cross, **L1 gains ~+20% FoM3 while the CNN
gains nothing**, so de-leaked **L1 ≳ CNN on the cross; auto-only ties.** The L1 side and the CNN side
are both done, calibrated, written up, figured, committed, and pushed. **The live scientific question**
(opened by Andreas on 2026-06-10) is whether the CNN's no-gain is **optimization-limited** rather than
a real method difference — a multi-compressor-seed experiment is **running right now** to test it.

## 2. Read these first, in order

1. **`FLATSKY_CNN_RESULT.md`** (repo root) — the CNN result table (FoM3/σ/2D), the best-single-seed
   robustness section, and the GATE-C calibration verdicts. The headline.
2. **`FLATSKY_CROSS_RESULT.md`** (repo root) — the L1 side (the other half of the comparison).
3. **`CROSS_MAP_LEAKAGE_FINDING.md`** — *why* we de-leaked; **read §6** (it contains the key caveat
   that a vanilla ReLU CNN does not natively form bilinear channel products — central to the open
   question below).
4. **`scripts/sbi/results/exploratory/flatsky_cross_2026_06/cnn_phase/gate_c/GATE_C_INTERPRETATION.md`**
   — how to read TARP/SBC/L-C2ST and why the result is calibration-trustworthy.
5. Memory index `…/memory/MEMORY.md` — especially `project_flatsky_cnn_no_cross_gain`,
   `project_flatsky_cross_deleaked_result`, `reference_lc2st_underpowered_highdim_l1`,
   `project_nde_architecture_mismatch`, `project_resnet_bn_contamination`, `feedback_gpu1_only`,
   `feedback_benchmark_dont_assume`, `feedback_no_pkill_self_match`.
6. This doc + the felt fiber `.felt/flatsky-cross-2026-06/flatsky-cross-2026-06.md` (live status).

## 3. The result, in numbers (pooled 3-MAF-seed, 9000-obs median; common jaxili MAF)

| arm | CNN FoM3 | CNN vs auto | L1 FoM3 | L1 vs auto | CNN/L1 |
|---|---|---|---|---|---|
| auto-only | 2325 | 1.00× | 2405 | 1.00× | 0.97× |
| +conv | 2192 | 0.94× | 2499 | 1.04× | 0.88× |
| +product | 2181 | 0.94× | 2875 | 1.20× | 0.76× |
| +both | 2306 | 0.99× | 2910 | 1.21× | 0.79× |

- **`product`** = pointwise κᵢ·κⱼ (its mean = ξᵢⱼ); **`conv`** = apodized FFT convolution (Zürcher
  alm-product flat-sky analog, the seed-fragile / mostly-leakage operator); **`both`** = autos+conv+product.
- Cross operators live in `scripts/sbi/flatsky_cross.py` (np/torch/jax backends, bit-matched).
- **Calibration (GATE C):** auto-only + product pass all three tests (TARP global-joint, SBC
  global-marginal, **L-C2ST local-at-fiducial — which works for the CNN's 10-d summary**, unlike
  high-dim L1). conv is locally miscalibrated (L-C2ST 60% reject) but it is the throwaway arm.

## 4. ★ THE LIVE QUESTION (Andreas, 2026-06-10) — read carefully

Andreas's position, which reframes the result: the CNN takes the four tomographic bins **as input
channels**, so its conv filters can mix channels and access cross-bin information that L1 (a strictly
per-map statistic) cannot — therefore explicit cross-maps being redundant *for the CNN* is expected,
and the cross-map trick is fundamentally a device **for per-channel methods** (L1, per-bin peaks).
He does **not** think a 10-d bottleneck is the cause (10 dims is plenty for 6 parameters); he thinks
the gap is **compressor training inefficiency**, and points out that the **best seed already beats L1
on auto-only** (best 2620 vs L1 2405).

Where the current data refine this (state precisely; don't overclaim either way):
- If the CNN extracted cross-bin info from the channels, CNN **auto-only** should *beat* L1 auto-only;
  instead they **tie** (2325 ≈ 2405). And handed the explicit product map it gains nothing while L1
  gains +20%. So at the *pooled* level the CNN is leaving cross-correlation information on the table.
- BUT the per-seed scatter is large (auto 2620/2364/2387; product 2225/2331/2017) — consistent with
  **optimizer-into-better/worse-optima** rather than a capacity wall. **Caveat:** that scatter is over
  the 3 **MAF/NDE** re-trainings on the *single* seed-41 compressor (stage-2 noise), at one obs; the
  "best 2620 > L1 2405" is **best-single-seed (un-haircut) vs L1-pooled (haircut), single-obs** — so
  it is *suggestive, not yet a clean claim* (selection bias + unmatched aggregation).

**The reframing to carry into the writeup IF the running experiment supports it:** not "L1 > CNN as a
property of the methods," but "the CNN's cross extraction is **optimization-limited** — its per-seed
scatter straddles L1, and the pooled number sits below only because pooling averages in under-trained
seeds." Hold the writeup/memory edit until the numbers back it.

## 5. What is RUNNING right now (do not relaunch; check, then continue)

**Multi-COMPRESSOR-seed check** — `run_multiseed_compressor_check.py`, launched detached 2026-06-10,
GPU **1+2** only. Trains 2 more *compressor* seeds (42, 43) for **both** `product` and `auto-only`
(the fair per-seed product-vs-auto), then runs each through the identical seed-41 pipeline:
compressor (`--exit-after-compress`) → fiducial summaries (9000 obs, G1-checked) → population sweep
(retrain 3 MAF seeds + 9000-obs median). Identical recipe to seed 41 (plain CNN, 80k steps, **no**
grad-clip — product/none train clean without it).

- **Status:** `…/cnn_phase/multiseed/driver.out` (phase log) and `…/multiseed/logs/`. Phases run
  compressor → fidsumm → sweep; the **sweeps are the long pole (~2 h each, 2 GPUs → ~4 h)**.
- **When done:** `…/cnn_phase/multiseed/MULTISEED_COMPRESSOR_CHECK.md` (auto-written: product/auto FoM3
  per compressor seed + verdict). If the file is missing, the run failed or is mid-phase — read
  `driver.out`.
- **First thing the next session should do:** check whether it finished; if so, interpret it against
  §4 (does a well-trained compressor lift `product` toward/over L1? does the no-gain hold across
  compressor seeds?), then **reframe FLATSKY_CNN_RESULT.md + the memory accordingly**, commit, push.
- ⚠ GPU note: GPU 0 and 3 had **foreign tenants** (≈18 GB / 12 GB, idle) at launch — stayed off them.
  Re-check `nvidia-smi` and honor **GPU 1 only by default** (1+2 was the campaign grant). Never spill
  onto 0/3 without checking; surface contention to Andreas.

## 6. Next steps (prioritized)

1. **Finish + interpret the multi-compressor-seed check** (§5) and reframe the writeup/memory (§4).
   This is the most important open thread — it settles the "optimization-limited vs real" question.
2. **Principled best-seed comparison.** Currently "best" is post-hoc by FoM3 = selection bias. Redo
   selection by **validation loss** (a held-out criterion), compare best-to-best. L1 best-seed is the
   hard part: L1's ~2000–3200-d datavector **truncates on jaxili checkpoint reload**
   (`reference_jaxili_checkpoint_reload_truncation`), so per-seed L1 needs a **retrain** (the L1
   population-sweep machinery already retrains 3 MAF seeds; a per-seed dump is a small extension of
   `population_sweep_flatsky.py`). Decide with Andreas whether to spend the retrain.
3. **BNT for the flat-sky cross setup** (Andreas's extra ask — scope it, then build). BNT (the nulling
   transform) is implemented for the *CNN auto path* via `--apply-bnt` (shape noise injected BEFORE
   BNT; requires full tomo4 `--nbins 4 --tomo-bin-indices 1,2,3,4`; utils in `scripts/sbi/bnt_utils.py`,
   enforced by `validate_bnt_configuration`). It has **not** been wired into the `flat_local` cross
   route. Open design questions to settle first: do you BNT the autos *then* build the patch-local
   cross from the BNT'd maps, or build the cross then BNT? where does shape noise go relative to BNT
   and relative to the on-device cross build? Produce a calibrated BNT-cross L1-vs-CNN comparison
   mirroring the no-BNT one. Likely a multi-day campaign — scope, get sign-off, then run.
4. **Bug / inefficiency hunt across the analysis** (Andreas's extra ask). Fable 5 is strong at this —
   **use parallel subagents**. High-value targets to audit: (a) train/val example-disjointness across
   the flat_local route (compressor perms 0–4 vs NDE perms 5–6 — is the split truly leak-free end to
   end, including the fiducial-summary obs?); (b) the per-channel RMS whitening (frozen at train-sample
   — is it byte-identical across train/val/obs as claimed? the G1 self-checks say yes, but verify the
   code path); (c) the population-sweep MAF retraining hyperparameters and early-stopping (are arms
   stopped consistently? is the 3-seed pool the right aggregation?); (d) throughput: the sweeps are
   ~2 h/arm — most of it is per-obs sampling in a Python loop (9000 obs × 2000 samples × 3 seeds);
   is there an easy vectorization/batching win? (e) any silent fallbacks (e.g. best_val → last_step
   when save-every > steps, which bit us once). Report findings; only fix after Andreas confirms.

## 7. File / script map (the flat_local CNN campaign)

All live code is in `scripts/sbi/`. Results under `…/results/exploratory/flatsky_cross_2026_06/`
(`cnn_phase/` for the CNN side; the L1 side is the sibling dirs `l1_matrix/`, `population_sweep/`,
`gate_c/`, `representative_corner/`).

**Pipeline engine**
- `npe_cnn_nbody_tomo.py` — the CNN-VMIM pipeline. New this campaign: `--cnn-map-route flat_local`
  `--cross-op {none,conv,product,both}` (reads autos ch 0–3, builds the cross **on-device in JAX**,
  per-channel RMS whitening) and `--compressor-grad-clip` (stabilises the VMIM RealNVP companion;
  `both` NaN'd without it). Shared helpers: `compute_flat_cross_channel_rms`, `make_flat_cross_transform`.
- `flatsky_cross.py` — the cross operators (np/torch/jax, bit-matched). Single source of truth.
- `build_fiducial_summaries_cnn.py` — 9000-obs fiducial summaries (flat_local-aware; G1 self-check).
- `gate_a_flat_cross_cnn.py` — GATE-A construction test (jax vs numpy).

**Orchestrators** (all greedy GPU schedulers, `--dry-run` prints commands)
- `run_flatsky_cnn_matrix.py` — 4 arms × seed-41 compressor.
- `run_flatsky_cnn_fiducial_summaries.py` — fiducial summaries for the 4 arms.
- `run_flatsky_cnn_gate_c_tarp.py` / `run_flatsky_cnn_gate_c_lc2st.py` — GATE-C TARP / L-C2ST.
- `run_flatsky_cnn_population_sweep.py` — the 9000-obs median sweep (`--gpus`, `--mem-fraction`).
- `run_flatsky_cnn_repr_corners.py` — 3-seed representative corners (typical + favorable obs).
- `run_multiseed_compressor_check.py` — **the running experiment** (§5).

**Calibration / analysis / plots**
- `compute_sbc_from_tarp_dumps_cnn.py` — SBC from the TARP dumps (CPU).
- `plot_lc2st_cnn.py` — the L-C2ST figure.
- `cnn_representative_corners.py` — 3-seed pooled posteriors at the representative obs.
- `cnn_per_seed_best.py` / `plot_best_seed.py` — per-MAF-seed (un-pooled) + best-seed numbers/plots.
- `consolidate_cnn_vs_l1.py` — generates `FLATSKY_CNN_RESULT.md` + the figures (reads both sweeps).
- `build_stitched_figure.py` — the A&A double-column paper figure (getdist corner + FoM3-bars inset,
  grayscale-safe). `overnight_cnn_pipeline.sh` — the autonomous overnight driver (already ran).

**Shared (used by both L1 and CNN, do not assume they are CNN-specific)**
- `population_sweep_flatsky.py`, `tarp_stratified_val.py`, `lc2st_diagnostic.py`,
  `train_jaxili_from_compressed.py` (has `compute_fom3`, `fom2d`, `marginal_stats`, `setup_env`),
  `npe_l1norm_cross_jaxili_nbody_tomo.py` (has `preprocess_summaries`, `filter_zero_variance_bins`,
  `train_with_nan_retry`, the jaxili checkpoint-reload helpers).

## 8. Hard guardrails (non-negotiable; from CLAUDE.md + memory)

- **GPU 1 only by default** (`feedback_gpu1_only`); 1+2 was the campaign grant. Re-confirm before any
  other GPU. **Re-check `nvidia-smi` for foreign tenants** (0/3 had them). Never auto-spill; surface
  contention. titan has **no scheduler** — run jobs detached (`setsid nohup … &`, poll with
  `pgrep -f "[b]racket"`). **Never `pkill -f` a self-matching pattern** (`feedback_no_pkill_self_match`).
- **git:** stage by path, **never `git add .`/`-A`**. Do **not** commit large generated artifacts
  (`*.npz` caches, `*.pkl` checkpoints, TARP/posterior dumps — the `cnn_phase/` tree is ~1.2 GB of
  these). Commit code + writeups + figures (png/pdf) + lightweight JSON summaries only. Run a safety
  grep (`git diff --cached --name-only | grep -iE '\.npz$|\.pkl$'`) before every results commit. There
  is a **pre-existing** `gc.log`/loose-objects warning in the repo — leave it (do not `git prune`).
  Lots of **pre-existing dirty files** (paper-synthesis notebooks, other handoffs) are NOT ours — leave
  them; commit only this campaign's files.
- **Metric/calibration conventions:** lead with the **pooled 9000-obs median** (single-obs is noisy
  and patch-dependent); FoM3 is OK to headline but report σ/2D alongside (it is corr-sensitive); never
  trust a contour before GATE C; **never PCA the L1 datavector** (`feedback_never_pca_l1`); CNN preproc
  is `none`/clip 0/min-var 1e-12, L1 is `log1p-zscore`/clip 5/min-var 1e-5.
- **Benchmark, don't assume** for any perf/throughput claim (`feedback_benchmark_dont_assume`); don't
  guess time estimates (`feedback_dont_guess_time_estimates`).

## 9. Notes specific to Fable 5 (this session's model)

- Fable 5 is strong at long-horizon autonomy, bug-finding/code-review recall (good for §6.4), and
  **parallel subagents** — delegate the audit and independent sub-tasks, keep working while they run.
- It follows brief instructions well; you don't need to enumerate behaviors. When you have enough to
  act, act — don't re-survey settled decisions (§4 is settled context, not an open re-litigation).
- Ground progress claims against tool results; report failures with output. Use `high`/`xhigh` effort
  for the science; lower for routine edits.
- The safety classifiers cover offensive-cyber and bio — **irrelevant here** (this is cosmology), so
  no expected refusals. Don't write prompts/skills that ask the model to echo its raw reasoning.
