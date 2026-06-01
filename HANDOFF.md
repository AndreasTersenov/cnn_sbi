# HANDOFF — cross-only L1 vs CNN campaign, v2 (channel-aware σ)

**Date:** 2026-05-15
**Branch:** `l1-cross-maps`
**Latest commit:** `b66b744 diagnostics(sbi): L1 cross-only noise-model investigation scripts`
**Origin in sync:** ✅ pushed through `b66b744` to `origin/l1-cross-maps`.

---

## 1. Objective

The project compares **wavelet-L1 vs CNN-VMIM compressors** as summary
statistics for tomographic weak-lensing SBI. Headline metric is **FoM3 =
1/√det(C₃)** over the 3-D subspace **(Ω_m, σ_8, w_0)** computed from the
posterior samples. Done looks like:

1. A clean, defensible comparison across **3 input configurations** ×
   **3 probes**:
   - Inputs: **auto-only** (4 tomographic auto maps), **cross-only**
     (6 inter-bin harmonic cross maps κ_i × κ_j), **auto+cross**
     (10 channels combined).
   - Probes: **L1** (wavelet starlet ℓ¹ histograms via `wl_stats_torch`),
     **CNN plain** (small VMIM CNN), **CNN resnet50_gn** (ResNet50 with
     GroupNorm — the variant introduced to avoid the BN-running-stats
     contamination on multi-channel inputs documented in
     `memory/project_resnet_bn_contamination.md`).
2. Each cell of that 3×3 should have ≥3 seeds, pooled-and-mean-of-seeds
   FoM3 reported, and contour plots in the project's GetDist style.
3. **TARP joint-coverage** verified on the headline cells (at minimum the
   v2 L1 cross-only and L1 auto+cross arms — v1 TARP exists, v2 does not
   yet).
4. A scientific narrative that resolves the **prior "L1 wins 3× on
   harm-cross"** finding. The v1 number was inflated by a wrong
   cross-channel noise model (see §3); the corrected v2 narrative is the
   one that should appear in any paper/report.

The branch tracks the cross-channel-maps phase of a larger SBI project
(`PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md` is the authoritative synthesis;
read it before any new science work).

---

## 2. Current status

### What's working

- **Channel-aware L1 noise model** is implemented, smoke-tested, and ran
  the full v2 production campaign. Activated via
  `--cross-noise-model channel_empirical_global`. Code:
  `scripts/sbi/npe_l1norm_cross_jaxili_nbody_tomo.py`
  - `calibrate_channel_noise_sigma_from_harmonic_cache()` (~line 430)
    streams the harmonic cache and computes σ_c per channel.
  - `iter_harmonic_examples()` now accepts `channel_scale` (~line 331)
    and applies it after `channel_slice` at read time.
  - Threaded through `calibrate_snr_range_from_harmonic_cache`,
    `compute_l1_dataset_from_harmonic_cache`, `load_observed_from_-
    harmonic_cache`.
  - Saved to `l1_cache_meta.npz` keys
    `cross_noise_model | channel_scale | channel_sigma`.

- **Two-stage shared-compressor flow** for CNN cross-only.
  `scripts/sbi/npe_cnn_nbody_tomo.py`:
  - `--channel-mode {auto_cross, cross_only}` slices the 10-channel cache.
  - `--exit-after-compress` trains compressor + builds compressed-
    dataset cache + writes the cache fingerprint as if loaded
    "pretrained", then exits. Used by orchestrator Phase 1.

- **Cross-only campaign orchestrator** (`scripts/sbi/run_cross_only_-
  campaign.py`) drives a 2-phase pipeline:
  - Phase 1: L1 single-stage seeds + CNN Stage A (shared compressors)
    in parallel across `--gpus 0,1,2`.
  - Phase 2: CNN Stage B (NDE-only per seed) — each takes ~100s on
    cached summaries, vs ~30 min/seed if compressor were retrained.
  - `--skip-existing` is robust (skips on `posterior.npy` exists or
    compressor checkpoint exists).
  - Auto-mem-fraction: per-GPU `XLA_PYTHON_CLIENT_MEM_FRACTION` from
    `nvidia-smi --query-gpu=memory.free`.

- **L1 throughput refactor**:
  - `--l1-realizations-per-batch` (default 10) batches multiple cache
    files per `compute_l1_batch` call.
  - `iter_harmonic_examples()` does async file prefetch via
    `ThreadPoolExecutor` (`prefetch_workers=4, prefetch_depth=12`).
  - Combined: ~52 → ~85 patches/s/worker (+55%) measured live during v2.

- **compare_cross_only.py**: campaign analysis with **GetDist filled
  contours** (PDFs are 0.2–1.1 MB, no longer 60–110 MB). Auto-discovers
  arms/seeds/dims, writes `SUMMARY.md` + figure bundle. Auto-fired by
  the orchestrator's autonomous loop when "All phases done" hits the log.

- **compare_probes_configs.py**: 3×3 overlay corner per probe + FoM3
  table (both pooled and mean-of-seeds). Output dir
  `scripts/sbi/results/exploratory/probes_configs_comparison/`.

### Campaigns finished + on disk

| Campaign root | Status | Noise model | Notes |
|---|---|---|---|
| `cross_only_campaign/` | done | v1 (broken auto-σ for all channels) | 5 L1 + 3+3 CNN, FoM3s in SUMMARY.md |
| `cross_only_campaign_v2_chsigma/` | done | v2 (channel-aware σ) | 5 L1 + 3+3 CNN, **headline result** |
| `auto_cross_v2_chsigma/` | done | v2 channel-aware σ | L1 ONLY (3 seeds), side experiment |
| `cross_maps_campaign/jaxili_auto_nobnt/` | done (older) | v1 (channels are autos, unaffected) | auto-only L1 baseline |
| `cnn_with_harm_cross_normalized/` (plain + resnet50_gn) | done (older) | CNN unaffected by σ | auto+cross CNN |
| `cnn_extended_train_zm/` | done (older) | CNN unaffected | auto-only CNN plain dense512 step240000 |
| `cnn_resnet50_zm_sweep/` | done (older) | CNN, stock BN | auto-only resnet50 (BN, not GN — only choice for this cell) |

### Headline FoM3 from `probes_configs_comparison/fom3_probes_configs.md`
(mean-of-seeds **FoM3** ± std, pooled in parens)

| probe | auto-only | cross-only | auto+cross |
|---|---:|---:|---:|
| **L1** | 11 430 ± 3993 (8182) | 18 120 ± 3251 (16 070) | **38 226 ± 1421** (34 004) |
| **CNN plain** | 22 633 ± 5126 (14 138) | 21 219 ± 1135 (20 104) | 25 466 ± 636 (23 280) |
| **CNN resnet50** | 20 480 ± 2299 (16 948) | **26 614 ± 1269** (25 830) | 18 763 ± 2549 (11 185)¹ |

¹ Pooled is much lower than mean-of-seeds because the 3 seeds disagree
on posterior means (e.g. seed 43 pulls Ω_m to 0.22 while seeds 41/42
are near 0.25). See §4.

### What's half-built / not started

- **TARP coverage on v2**: not run. v1 TARP exists at
  `scripts/sbi/results/diagnostics/tarp_harm_cross/` and the runner
  `scripts/sbi/run_tarp_dumps_campaign.py` is arm-agnostic; a fresh v2
  TARP would just need to point at the v2 campaign dir.
- **Memory hygiene**: the new `memory/project_l1_noise_model_correction.md`
  is comprehensive, but the older `project_harmonic_cross_overturns_-
  flatsky.md` and `project_cross_only_l1_loses.md` files still contain
  the v1 numbers as if they were the science answer. They have been
  flagged in `MEMORY.md` but not rewritten.
- **No paper-style write-up yet**.

### What's broken / questionable

- **CNN resnet50_gn auto+cross seed-to-seed scatter** is wide (std
  ~2549 / mean 18 763 = 13.5% CoV), much higher than the comparable v2
  cross-only resnet50_gn (std 1269 / mean 26 614 = 4.8%). Posterior means
  disagree across seeds in (Ω_m, σ_8, w_0). This is the existing
  `cnn_with_harm_cross_normalized/resnet50_gn/` run, not anything we
  retrained. Per `memory/project_resnet_bn_contamination.md` the GN
  variant should give FoM3 ≈ 22k — our recomputed numbers are in that
  range *per seed* but the pooled FoM3 deflates to 11k. Unclear whether
  the training was suboptimal or this is real seed sensitivity. See §4.
- **σ_8 marginal in v2 L1 cross-only** still looks slightly off-axis
  in the corner overlay (you flagged this in chat — the L1 contours have
  a different degeneracy axis from CNN even in v2). Whether the v2 fix
  fully cured the prior σ_8 inversion is not verified by TARP yet.

---

## 3. Decisions made (non-obvious)

1. **Channel-aware σ_c uses a GLOBAL EMPIRICAL estimator** — one
   number per channel from a calibration sample of ~32 cosmologies,
   fixed for the whole run. We considered:
   - Per-cosmology σ_c (rejected: would leak cosmology info into the
     SNR normalization).
   - Per-realization σ_c (rejected: noisy + same leakage concern).
   - Theoretical product-noise model (rejected: cross map is built in
     harmonic space with apodization, the theoretical pixel-product
     formula doesn't apply directly).

2. **Implementation via pre-scaling, not per-channel noise_sigma.**
   We multiply each channel's map by `σ_auto / σ_c` at read time
   inside `iter_harmonic_examples`. Mathematically equivalent to
   passing per-channel `noise_sigma` into `WLStatistics`, but a much
   smaller code change (one site to edit + threading through the
   pipeline). The L1 absolute values change but the per-feature
   z-score standardization inside the NDE is unaffected.

3. **v1 results kept on disk**, v2 written to a separate directory
   (`cross_only_campaign_v2_chsigma/`, `auto_cross_v2_chsigma/`). Rationale:
   provenance + direct visual comparison. Future TARP on v2 will live
   alongside.

4. **CNN compressor batch reverted from 256 → 128** for v2 (matches
   v1). We initially bumped to 256 expecting a speed gain; it didn't
   help because GPU was already saturated at 128 (per-step time
   scales linearly with batch on a saturated GPU). v1 vs v2 comparison
   stays apples-to-apples.

5. **L1 throughput: file batching + async prefetch** (not "use more
   GPU memory"). Initial attempt was to bump `--ds-batch-size`; that
   had no effect because `compute_l1_batch` processes per-realization
   (48 patches per call) regardless of the `ds_batch_size` flag. The
   real bottleneck was sequential NFS file reads. Async prefetch
   (`ThreadPoolExecutor`, 4 workers, depth 12) overlaps I/O with GPU
   work; file batching (~10 realizations per call) reduces per-call
   overhead. Net: +55% throughput, GPU memory rose 1.2 GB → 9.5 GB.

6. **L1 seeds 44 + 45 launched manually on GPU 1**, outside the
   orchestrator. The orchestrator's auto-launcher (in the autonomous
   loop) would have put them on GPU 2 during Stage A slack; manual
   GPU-1 launch happened because GPU 1 freed up early. Their state
   was injected into `/tmp/cross_only_v2_loop_state.json` so the loop's
   `_extras_launched` flag prevented double-firing.

7. **Two-stage CNN flow** (Stage A trains compressor once, Stage B
   trains NDE per seed) is the only reason Phase 2 takes ~100 s/job
   instead of ~30 min/job. The fingerprint-stamp trick (writing the
   cache as if loaded "pretrained") is what makes the Stage B cache
   hit work after a Stage A.

8. **GetDist filled contours for compare_cross_only.py** — v1 of the
   script used raw `ax.scatter` of 100 k points per arm, producing
   60–110 MB PDFs that PDF viewers couldn't open. Switched to a list
   of `MCSamples` + `gplot.get_subplot_plotter().triangle_plot(
   filled=True, markers=fid_dict, marker_args={"color":"red","lw":1.2})`
   matching the L1 runner's per-run plot style.

9. **GPUs 0, 1, 2 only.** User policy. GPU 3 has been off-limits for
   the entire campaign even when free. Honored throughout.

10. **PNG/PDF cleanup**. A subagent identified 34 PNGs with matching
    PDFs in the same directory and we deleted them (9.14 MB freed).
    Those deletions are staged-unstaged-but-uncommitted in the working
    tree (see §after-handoff).

---

## 4. Open problems

### 4.1 CNN resnet50_gn auto+cross — high seed scatter
- **The problem**: pooled FoM3 = 11 185, mean-of-seeds = 18 763 ± 2549.
  Memory note `project_resnet_bn_contamination.md` says GN variant
  gives ~22k. Our per-seed numbers (15.6k, 18.8k, 21.9k) bracket that
  but the pooled deflates because seeds disagree on posterior means
  (especially seed 43 pulls Ω_m → 0.22 vs 0.25 for the others, w_0
  → −1.09 vs −0.98).
- **What we tried**: examined per-seed FoM3 and means; confirmed the
  meta.json says `compressor_arch=resnet50_gn`.
- **Suspicion**: this is the existing `cnn_with_harm_cross_normalized/-
  resnet50_gn/` run from an earlier session; the training may have been
  unstable. Re-running with the v2 orchestrator's apples-to-apples
  config could fix this.

### 4.2 v2 L1 cross-only σ_8 marginal looks slightly biased
- **The problem**: even with channel-aware σ, the L1 cross-only corner
  plot shows the σ_8 marginal with a tilted degeneracy axis vs CNN.
  User flagged this in chat ("the axis of the degeneracies of the
  contours that we get from L1 and CNN in the cross only case are
  different").
- **What we tried**: re-binning, channel-aware σ. Histograms now fill
  40/40 bins, but the contour shape still differs from CNN.
- **Suspicion**: this might be intrinsic to the L1 statistic (the
  histogram representation loses spatial phase information that the
  CNN keeps) rather than a noise model issue. TARP coverage on v2
  would settle whether the L1 σ_8 inference is calibrated.

### 4.3 No TARP coverage on v2
- Without TARP we can't claim "L1 v2 has the right error bars on σ_8".
- The TARP infrastructure (`run_tarp_dumps_campaign.py`,
  `dump_tarp_posterior_samples.py`) exists and is arm-agnostic.

### 4.4 Auto-only CNN resnet50 is stock BN, not GN
- We didn't run a dedicated auto-only resnet50_gn. The memory says BN
  contamination only matters on multi-channel harmonic input
  (4 autos are unlikely to have the same issue), but it's an
  apples-to-apples mismatch in the 3×3 table. Cell labelled
  "(stock BN)" to be honest.

### 4.5 The dim=20 question
- We dropped `--dims 20` from v2 to save compute. The plan's "tier-2
  expansion" was to add dim=20 only if dim=10 is borderline. dim=10
  is decisive on cross-only (CNN > L1), so dim=20 may not be needed —
  but we never verified.

---

## 5. Immediate next steps (in order)

1. **Compare v1 vs v2 L1 cross-only contours directly** (single figure,
   overlay). Use `compare_probes_configs.py` as a template — adapt for a
   v1-vs-v2 overlay per arm. ~30 min. Output should show the σ_8
   marginal shift and the bin-fill improvement visually.

2. **Run TARP on v2 L1 cross-only AND L1 auto+cross**. Reuse
   `scripts/sbi/run_tarp_dumps_campaign.py`. The L1 posteriors are
   already in `cross_only_campaign_v2_chsigma/l1_cross_only/posteriors/`
   and `auto_cross_v2_chsigma/l1_auto_cross/posteriors/`. Need to
   re-dump per-observation posteriors first (probably the slow part).
   ~3 h on 3 GPUs.

3. **Re-run CNN resnet50_gn auto+cross** to resolve the seed-scatter
   issue. Use the v2 orchestrator with `--channel-mode auto_cross` and
   appropriate output dir. CNN doesn't change with the noise model so
   the only purpose is cleaner training. ~13 h on a clean GPU.

4. **Rewrite the two superseded memory entries** to point to the v2
   results: `project_harmonic_cross_overturns_flatsky.md` and
   `project_cross_only_l1_loses.md`. Keep the historical context but
   add a "**Superseded — see [[project_l1_noise_model_correction]]**"
   header so future sessions don't get confused. (Partially done in
   `MEMORY.md` index, not in the body files.)

5. **Write a 1-2 page summary** for the paper / report: the bug
   discovery story (smoking-gun plot `diagnostics/l1_histograms_signal_-
   check.pdf`), the corrected science narrative, the 3×3 table.

---

## 6. Context the next session needs

### Read these first, in order

1. **`CLAUDE.md`** — project conventions, stack, common commands. Always
   loaded automatically.
2. **`HANDOFF.md`** (this file) — the current state.
3. **`memory/MEMORY.md`** — auto-loaded index of all memory files. The
   two top-priority ones to read:
   - `memory/project_l1_noise_model_correction.md` — the headline
     bug-and-fix from this session.
   - `memory/project_resnet_bn_contamination.md` — why we use
     `resnet50_gn` instead of stock `resnet50` on multi-channel input.
4. **`PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md`** — the long-form science
   knowledge base. More current than `README.md`.
5. **`scripts/sbi/results/exploratory/cross_only_campaign_v2_chsigma/-
   SUMMARY.md`** — the auto-generated v2 results headline.
6. **`scripts/sbi/results/exploratory/probes_configs_comparison/-
   fom3_probes_configs.md`** — the 3×3 FoM3 table.

### Code locations

- L1 cross runner (heaviest single file):
  `scripts/sbi/npe_l1norm_cross_jaxili_nbody_tomo.py` (~2900 lines)
  - Channel-aware noise: `calibrate_channel_noise_sigma_from_harmonic_-
    cache` around line 430, threading via `channel_scale` argument
    through `iter_harmonic_examples`, `calibrate_snr_range_from_-
    harmonic_cache`, `compute_l1_dataset_from_harmonic_cache`,
    `load_observed_from_harmonic_cache`.
  - Throughput: `iter_harmonic_examples` uses `ThreadPoolExecutor`;
    `compute_l1_dataset_from_harmonic_cache` flushes 10-realization
    batches.
  - CLI flags: `--cross-noise-model`, `--channel-sigma-calib-realizations`,
    `--l1-realizations-per-batch`.
- CNN cross runner: `scripts/sbi/npe_cnn_nbody_tomo.py`
  - `--channel-mode`, `--exit-after-compress`, cache-fingerprint stamp
    (search for `compressor_source = "pretrained"`).
- Orchestrator: `scripts/sbi/run_cross_only_campaign.py`
  - `_build_l1_job`, `_build_cnn_stage_a_job`, `_build_cnn_stage_b_job`,
    `build_phase1_jobs`, `build_stage_b_jobs`, `run_jobs_parallel`.
  - `ARMS` dict at line ~371 defines the (arm → config) table.
- Analysis scripts:
  `scripts/sbi/compare_cross_only.py` (per-campaign),
  `scripts/sbi/compare_probes_configs.py` (cross-campaign 3×3).
- Diagnostics:
  `scripts/sbi/diagnose_cross_only_inputs.py`,
  `scripts/sbi/diagnose_cross_only_tighter_snr.py`,
  `scripts/sbi/diagnose_cross_only_channel_aware_noise.py`,
  `scripts/sbi/diagnose_cross_only_signal_check.py`.

### Autonomous loop scripts (in `/tmp`, ephemeral)

- `/tmp/cross_only_loop_check.py` — v1 campaign autonomous monitor (no
  longer needed, can be deleted).
- `/tmp/cross_only_v2_loop_check.py` — v2 campaign autonomous monitor
  (no longer needed — both campaigns done).

These are not committed. If you re-need this pattern, the canonical
reference is the Monitor + ScheduleWakeup setup in the chat history;
re-creating from scratch is faster than salvaging.

### Data locations

- Harmonic cache (10-channel patches, sole source for v2):
  `scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_-
  cache_grid/nobnt/{train,val,obs}/cosmo_NNNNNN_permM.npz`.
  Manifest SHA-256 `0a68ea89...` (recorded in every l1_cache_meta.npz).
- Per-cell L1 cache (built by L1 runner): `<run_dir>/cache/l1_{train,
  val,cache_meta}.npz`. The `cache_meta` includes `cross_noise_model`,
  `channel_scale`, `channel_sigma`.
- Stage A artifacts (shared compressor + compressed-dataset cache):
  `<campaign_root>/_shared_compressor/<arm>/dim_<N>/save_params/...pkl`
  + `<campaign_root>/_shared_compressor/<arm>/dim_<N>/cache/...`.

### Environment

- Conda env: `jaxili` (always run via `conda run -n jaxili python ...`).
- Local PyTorch extension `/home/tersenov/software/wl_stats_torch`
  (sys.path inserted at top of L1 runner).
- TFDS dataset registered via `tf_dataset_nbody_tomo*.py` builders in
  `scripts/sbi/`; the campaign uses
  `NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48`.

### Cosmology conventions

- Parameter order is fixed: `θ = [Ω_m, σ_8, w_0, h_0, n_s, Ω_b]`.
  `h_0 = H_0/100` is applied at preprocessing time (the L1 runner does
  `theta[3] /= 100` in `compute_l1_dataset_*`).
- Fiducial used everywhere: `[0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493]`.
- 4 tomographic bins (`--nbins 4 --tomo-bin-indices 1,2,3,4`). 20 deg
  field, 160 px, full-sphere cache built by
  `scripts/sbi/build_full_sphere_cross_cache.py`.
- FoM3 = `1 / sqrt(det(np.cov(samples[:, :3].T)))` over (Ω_m, σ_8, w_0).
  Definition lives in `npe_l1norm_cross_jaxili_nbody_tomo.py:1737-1759`
  and is duplicated (correctly) in the comparison scripts.

### Code style

- Conventional commit prefixes: `feat(sbi)`, `docs(sbi)`,
  `diagnostics(sbi)`, etc.
- No `git add .` / `git add -A`. Stage by path.
- No mass-edits to notebooks; they're perpetually dirty.
- `py_compile` is the only "test" — no test suite. After non-trivial
  edits run:
  ```bash
  conda run -n jaxili python -m py_compile scripts/sbi/<file>.py
  ```

---

## 7. Things to avoid / gotchas

- **DO NOT use `--cross-noise-model auto_scalar`** for any new L1 cross
  run. It's the broken model, preserved only for reproducing v1.
- **DO NOT directly compare v1 numbers to v2 numbers** without flagging
  the noise model. The v1 FoM3 ~65k for L1 harm-cross is noise-model-
  inflated.
- **CNN Stage A's "exit-after-compress" tail is slow.** After step
  150 000 the worker spends ~10 minutes computing the compressed
  dataset for train + val before exiting (rate ~80 reals/s on 6293
  files). Don't assume the worker is stuck if `rc=0` hasn't hit the
  orchestrator log yet.
- **GPU 3 is off-limits** per Andreas's instruction. Use only GPUs 0/1/2.
- **`alahiry`'s training jobs frequently grab GPUs unannounced** —
  check `nvidia-smi --query-compute-apps=pid,used_memory` before
  launching. Several v1 launch attempts hit OOM because of this.
- **`_shared_compressor/<arm>/dim_<N>/` is the source of truth for
  Stage B**. Do not delete without confirmation — Stage B depends on
  the `_find_latest_compressor_checkpoint` glob there.
- **Notebooks under `notebooks/sbi/` are perpetually dirty.** Never
  commit them.
- **`learn2map/` is a virtualenv, not source code.** Never edit.
- **`compare_cross_only.py` uses pooled FoM3 in its SUMMARY.md but
  mean-of-seeds in the FoM3 column.** Both are reported; cross-check
  before quoting either one as "the number". For CNN resnet50_gn
  auto+cross the discrepancy is large (11k pooled vs 19k mean).
- **L1 v1 vs v2 share theta/posterior shape**, so naive overlay won't
  re-rank seeds (same seeds 41/42/43 in both). Per-seed comparison is
  meaningful.
- **`--skip-existing` skip-on-compressor-checkpoint logic** is happy
  with partial checkpoints. If you kill a Stage A mid-training and
  relaunch with `--skip-existing`, it will incorrectly skip the partial
  arm. Delete the `_shared_compressor/<arm>/dim_<N>/save_params/` first.

---

## 8. Open questions for Andreas

1. **Should we rerun CNN resnet50_gn auto+cross?** Pooled FoM3 = 11k
   is much lower than mean-of-seeds 19k, indicating training
   instability. Per-seed numbers (15.6k / 18.8k / 21.9k) are
   plausible — but the seed-43 posterior pulls Ω_m to 0.22.
   *Assumption I'm proceeding with: yes, this is worth rerunning at
   the v2 orchestrator's config (clean), but you may disagree.*

2. **For the paper/report: do you want the v1 numbers shown alongside
   v2 as "what we thought before"**, or v2 only? *Assumption: show
   both, with the noise-model fix clearly labelled.*

3. **TARP on v2: highest priority before write-up?** *Assumption: yes,
   especially because (a) v1 TARP exists for direct comparison and
   (b) the σ_8 contour shape still looks slightly off and TARP would
   settle whether it's calibrated.*

4. **Auto-only CNN resnet50_gn cell in the 3×3 table**: do we want to
   run a proper resnet50_gn on auto-only to replace the stock-BN
   `cnn_resnet50_zm_sweep` cell, or accept the asymmetry? *Assumption:
   accept the asymmetry and note it; auto-only resnet50_gn is
   ~9 h of compute we don't strictly need.*

5. **dim=20 expansion**: the original plan said run dim=20 only if
   dim=10 was borderline. dim=10 results are quite decisive (CNN > L1
   on cross-only, L1 > CNN on auto+cross). Skip dim=20 entirely?
   *Assumption: yes, skip — the dim=10 result is statistically clean.*

6. **L1 σ_8 contour shape in v2 cross-only**: do you want a deeper
   investigation (e.g. plot the L1-bin importance per parameter to see
   what's driving the σ_8 axis), or is "TARP next" sufficient?
   *Assumption: TARP first; if TARP says σ_8 is mis-calibrated, then
   investigate the bin importance.*

7. **Diagnostic scripts** (`diagnose_cross_only_*.py`): keep them in
   tree as the canonical "this is why we changed the noise model"
   reference, or delete after the write-up? *Assumption: keep — they
   produce the smoking-gun plot referenced from the memory file.*

8. **The two `/tmp/cross_only*_loop_check.py` scripts**: do you want
   them promoted to `scripts/sbi/` as permanent monitoring helpers, or
   left ephemeral? *Assumption: leave ephemeral — they were one-off
   campaign monitors and the pattern is reproducible from the
   conversation transcript.*

9. **`HARMONIC_L1_VS_CNN_INVESTIGATION_NOTES.md`** (in the repo root)
   has stale v1 claims. Rewrite, delete, or leave? *Assumption: leave
   for now and reference the v2 SUMMARY.md / new memory entry as the
   authoritative source.*

---

## Appendix A — useful one-liners

```bash
# Recompute the 3×3 FoM3 table after any new posterior lands
conda run -n jaxili python scripts/sbi/compare_probes_configs.py

# Regenerate the v2 campaign figures + SUMMARY.md
conda run -n jaxili python scripts/sbi/compare_cross_only.py \
  scripts/sbi/results/exploratory/cross_only_campaign_v2_chsigma

# Verify channel-aware noise model on a smoke run
cd /mnt/home/tersenov/software/cnn_sbi
SMOKE=/tmp/cross_only_chsigma_smoke
rm -rf "$SMOKE"; mkdir -p "$SMOKE"
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
  conda run --no-capture-output -n jaxili python \
  scripts/sbi/npe_l1norm_cross_jaxili_nbody_tomo.py \
  --map-kind nbody \
  --tfds-name NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48 \
  --field-size 20 --field-npix 160 \
  --nbins 4 --tomo-bin-indices 1,2,3,4 \
  --full-sphere-cross-cache scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid \
  --channel-mode cross_only --cross-noise-model channel_empirical_global \
  --channel-sigma-calib-realizations 4 \
  --no-wandb --n-scales 5 --l1-nbins 40 --l1-min-snr -13 --l1-max-snr 13 \
  --pca-components 0 --cross-snr-percentile 1.0 \
  --ds-batch-size 32 --l1-realizations-per-batch 4 \
  --learning-rate 1e-4 \
  --total-steps 2 --batch-size 16 --patience 0 --save-every 1 --npe-samples 64 \
  --seed 41 \
  --save-dir "$SMOKE/save_params" --cache-dir "$SMOKE/cache" \
  --posterior-out "$SMOKE/posterior.npy" --figure-out "$SMOKE/posterior.pdf" \
  --harmonic-calibration-realizations 4 --cuda-visible-devices 2
# expected: posterior.npy exists, l1_cache_meta.npz has 40/40 bins non-zero

# Sanity check non-zero bins for any cached L1 dataset
python3 -c "
import numpy as np
d = np.load('PATH/TO/l1_cache_meta.npz', allow_pickle=True)
print('cross_noise_model:', d['cross_noise_model'])
x = np.load('PATH/TO/l1_train.npz')['x']
n_ch = int(d['n_l1_channels']); n_sc = int(d['n_scales']); n_b = int(d['l1_nbins'])
arr = x.reshape(x.shape[0], n_ch, n_sc, n_b)
nz = (arr.mean(0) > 0).sum(-1)
print(f'non-zero bins/(ch,scale): min={nz.min()} mean={nz.mean():.1f} max={nz.max()} /{n_b}')
"
```

## Appendix B — recent commits (latest first)

```
b66b744 diagnostics(sbi): L1 cross-only noise-model investigation scripts
0eb77bb feat(sbi): compare_probes_configs.py — 3 probes × 3 inputs FoM3 overview
881facd feat(sbi): compare_cross_only.py — campaign analysis + GetDist corners
f0b352b feat(sbi): L1 cross runner — channel-aware noise model + throughput
2fc79c3 feat(sbi): cross-only campaign orchestrator
4181631 feat(sbi): CNN runner — cross-only mode + two-stage shared-compressor flow
b8449a4 feat(sbi): TARP joint-coverage infrastructure
08575f6 feat(sbi): SBC runner for harm-cross CNN arms (plain + resnet50_gn)
e94d463 feat(sbi): add --dump-posterior-samples to SBC runners
b859634 docs: refresh CLAUDE.md for l1-cross-maps phase
70c796e docs(sbi): harmonic L1 vs CNN session-2 handoff
```

All 6 commits at the top of this list were pushed to `origin/l1-cross-maps`
during this session.
