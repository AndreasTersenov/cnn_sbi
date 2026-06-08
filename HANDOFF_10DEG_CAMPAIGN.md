# HANDOFF — 10°×10° L1-vs-CNN campaign (start here), 2026-06-04

You are continuing the definitive **L1-wavelet vs CNN-VMIM** weak-lensing SBI comparison in
`/mnt/home/tersenov/software/cnn_sbi` (compressor → jaxili MAF NDE; θ = [Ωm, σ8, w0, h0, ns, Ωb]).
The 20° investigation is **complete and closed**; this session redoes it on a freshly-built **10°
dataset** for the paper, because 10° patches have far better flat-sky validity (gnomonic corner
distortion 6.3% → 1.5%). This is FELT-tracked — drive it via the `felt` CLI.

## READ FIRST, IN ORDER
1. **This file.**
2. `PLAN_10DEG_CAMPAIGN.md` (the build+run plan; Phases 0–4 = dataset DONE, Phases 5–7 = your work).
3. Felt: load `~/.claude/plugins/marketplaces/cailmdaley-felt/claude-plugin/skills/felt/SKILL.md` +
   `FELT_AGENT_GUIDE.md` + CLAUDE.md §"Felt / Ralph conventions". Then `felt show definitive-l1-vs-cnn-10deg-2026-06`.
4. `CLAUDE.md` (project rules). 5. The 20° findings to compare against (below) + the recalled memories.

## STATE: the 10° dataset is BUILT + VERIFIED (overnight 2026-06-04)
- **TFDS (CNN + L1 both read this):** `nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180`
  @ `/home/tersenov/tensorflow_datasets` (LOCAL XFS, not /nas). 391 GB, 2048 tfrecords,
  **1,636,740 examples** (train 1,132,740 = cosmo_idx 1–899; val/`test` 504,000 = 900–1299).
  Features: `map_nbody` (80,80,10) float32 [4 auto + 6 cross], `theta` (6), `cosmo_idx`,`perm`,`patch` int64.
  Verified: count == cache, **bit-exact** vs the SHT cache.
- **Fiducial obs cache (kept, the diagnostics' obs source):**
  `scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_fiducial_10deg` — 200 perms ×
  180 patches; each `.npz`: `patches`(180,80,80,10), `patch_centers`(180,2 lon/lat), `theta`, etc.
- **20° data archived to** `/nas/tersenov/archive_20deg/` (cache + TFDS; recoverable). 20° compressed
  caches + compressor checkpoints kept LOCAL (`definitive_comparison/{compressed,phaseA_tfdata_2026_05_30}`).
- Free disk ~1.6 TB. Nothing running.

## 10° geometry (locked, do not change)
field 10°, **80 px → 7.5 arcmin/px (same res as 20°)**, **180 non-overlap patches, |lat| < 75°**
(polar-safe — the 20° pole bug is FIXED via `--max-abs-lat 75` / `_build_non_overlapping_centers(...,
max_abs_lat)`), nside=512, lmax=1024, σe=0.26, density=30→noise per cache, nobnt, SHT 10-ch route.
**One 10° patch = one "observation"** (100 deg², ~¼ the 20° survey area → per-obs constraints ~2×
weaker; that is the intended flat-sky trade).

## THE 20° SCIENCE FINDINGS (what to reproduce/compare at 10°)
Corrected definitive headline (typical/non-polar obs patch, median over ~300 patches):
- **L1 ≈ CNN on auto+cross**, small L1 edge concentrated in **w0/cross-maps** (σ(w0) L1 0.125 vs CNN
  0.167 ×1.34; σ(Ωm) ×1.18; 2D(Ωm,σ8) ×1.6; FoM3 ×2.17 but **FoM3 amplifies ~20–25% diffs into ~2×**).
  **auto-only = a tie.** Both reasonably calibrated.
- **L1 carries an anti-shrinkage fiducial offset** (pull w0 −0.37σ, Ωm −0.27σ, s8 +0.19σ); CNN unbiased.
  SBC: this offset **cancels globally** (L1 globally calibrated) → it is LOCAL to the fiducial.
- **The offset is CROSS-MAP-SPECIFIC** (flips sign in auto-only) → it's L1's high-gain cross-channel w0
  extraction (tighter-but-overshooting); CNN is regularized (looser-but-centered).
- L1's per-patch FoM3 spread is **realization-driven, not geometry** (the polar tile was a tiling bug).
- Throughline: **never headline FoM3** (fragile; cubes correlation/width changes). Lead with σ / 2D.
Full detail: `fiducial_full200/{SUMMARY_TYPICAL_PATCH.md, geometry_map/GEOMETRY_FINDINGS.md,
calibration/CALIBRATION_FINDINGS.md}` + felt `finding-l1-spread-realization-not-geometry`.

## ★ HEADLINE TEST FOR 10° ★
**Does L1's −0.37σ fiducial w0 offset SHRINK at 10°** (⇒ flat-sky distortion was the cause, 10° is
"more proper") **or PERSIST** (⇒ intrinsic ℓ₁-statistic compression bias, independent of flat-sky)?
This is the scientific payoff of the whole 10° run. (CNN sees the same maps and is unbiased, so the
20° conclusion leaned "intrinsic," but it was unproven — this settles it.)

## YOUR PLAN (Phases 5–7 of PLAN_10DEG_CAMPAIGN.md)
**Phase A — VALIDATE the dataset first** (Andreas wants tests before the campaign):
  - Load the TFDS; assert shapes (80,80,10), finite, channel scales (auto ~1e-2, cross ~1e-7, like 20°).
  - Verify **train/val cosmo_idx disjoint**; spot-check a TFDS example == the fiducial cache patch.
  - Quick smoke posterior (1 arm, few-min NDE) to confirm FoM3 is sane for a 100 deg² survey (expect
    ~½ the 20° per-obs constraint, i.e. wider — that's correct, not a bug).
**Phase B — loaders + caches** (the 20° scripts assume compressed `.npz` caches + cached summaries):
  - **L1-reads-TFDS loader — SCIENCE-CRITICAL, write carefully & verify against a reference.** Past
    bugs cratered FoM 4× here: use `--cross-noise-model channel_empirical_global`, **PCA OFF / never
    PCA L1** (`feedback_never_pca_l1`), the channel-aware route. Verify the ℓ₁ datavector matches a
    datavector computed the proven 20°-cache way on one 10° patch.
  - CNN: existing `tfds.load + tf.data` fast path; retune `read_config(interleave_cycle_length,
    block_length)` for the 2048-shard 10° dataset.
  - **Clean compressor↔NDE split by cosmo_idx** (e.g. compressor 1–630, NDE 631–899) — structurally
    kills the 20° example-slice leakage. The TFDS stores cosmo_idx → filter on it.
  - Build the 10° compressed train/val caches + the per-patch fiducial SUMMARIES (analogues of
    `build_fiducial_summaries_cnn.py` + the L1 `--fiducial-summaries-out` hook), so the diagnostic
    scripts below run summary-only (no map recompute).
**Phase C — run the 4 arms** (L1/CNN × auto/auto+cross), jaxili MAF NDE, 3 seeds.
**Phase D — diagnostics** (reuse, repoint at 10° caches): `geometry_resample.py` + `geometry_analyze.py`
  (geometry/spread/bias/error-budget), `compare_offsets.py` (the offset comparison + auto vs cross),
  `sbc_diagnostic.py` (global calib, has a self-test gate), `lc2st_diagnostic.py` (local calib;
  **use clf-kind logreg** — MLP is underpowered; gate self-tests must pass). All are in `scripts/sbi/`.
**Phase E — compare to 20°** and answer the headline test. Build the typical-patch table + the offset
  shrink/persist verdict.

## INFRASTRUCTURE (scripts/sbi/, all py_compile-clean)
- Dataset build (done; for reference/rebuild): `build_full_sphere_cross_cache.py` (`--field-size 10
  --field-npix 80 --n-centers 180 --center-nside 64 --min-separation-deg 14.2 --max-abs-lat 75`),
  `tf_dataset_nbody_tomo_cross.py` (config `grid_10deg_80px_nonoverlap180`; reads `CROSS_TFDS_CACHE_DIR`;
  skips empty splits), `build_10deg_tfds.py` (programmatic builder — `tfds` CLI is broken: needs
  apache_beam, absent), `run_10deg_{phase0,build,tfds_resume}.sh` (the overnight orchestrators).
- Diagnostics (reuse, repoint): `geometry_resample.py`, `geometry_analyze.py`, `compare_offsets.py`,
  `sbc_diagnostic.py`, `lc2st_diagnostic.py`, `validate_lc2st_power.py`.
- Compressors/NDE: `npe_cnn_nbody_tomo.py` / `npe_cnn_jaxili_nbody_tomo.py` (CNN-VMIM; use
  `resnet50_gn` for 10-ch harmonic), `npe_l1norm_cross_jaxili_nbody_tomo.py` (L1 cross),
  `train_jaxili_from_compressed.py` (the shared NDE + FoM3/2D/σ helpers).

## GOTCHAS / HARD RULES
- **GPUs 0 + 1 only** (Andreas overrode the project GPU-1-only rule for this campaign). Pin every job;
  never touch 2/3; check `nvidia-smi` first. **The node is shared** — `titan` ran an ~80-core job
  tonight; check load and be considerate (the TFDS writer is single-thread-bound ~1 core).
- **Never PCA L1.** Channel-aware noise `channel_empirical_global`. Verify `pca_applied: False`.
- **Don't headline FoM3** — lead with σ(w0) + 2D areas (`feedback_fom3_fragile_use_2d_areas`).
- Build lessons (if you rebuild any TFDS): `tfds` CLI needs apache_beam (absent → use the programmatic
  builder); spawn-Pool workers must NOT touch the GPU (`CUDA_VISIBLE_DEVICES=`) and need
  `OMP_NUM_THREADS=1`; the builder's `__main__` guard is required (spawn re-imports). To go FAST,
  parallelize shard-writing across processes (TFDS's serial writer is the ~9 h bottleneck) — not needed
  unless you rebuild.
- Git: never `git add .`; stage by path; **don't commit without Andreas's OK** (nothing this whole
  campaign is committed). `setsid nohup … &` for detached jobs; `pgrep -f` self-matches (bracket trick).
- Env: `/home/tersenov/anaconda3/envs/jaxili/bin/python` + `PYTHONUNBUFFERED=1`
  `XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30`. Never install packages.

## FELT STATE
- `definitive-l1-vs-cnn-10deg-2026-06` (constitution, OPEN) — YOUR campaign. Read it first.
- `definitive-l1-vs-cnn-2026-05/understand-per-patch-structure-2026-06` (CLOSED) — the 20° investigation.
- `…/finding-l1-spread-realization-not-geometry` — the key 20° finding. `felt check` clean.
Recalled memories to weight: `project_10deg_run_for_paper_decision`, `project_l1_geometry_realization_bias_2026_06`,
`project_l1_patch_sensitivity_full200`, `feedback_never_pca_l1`, `feedback_fom3_fragile_use_2d_areas`,
`feedback_l1_cross_must_use_harmonic_route`, `project_nde_architecture_mismatch`.

## FIRST MOVE
Verify state (nvidia-smi + the TFDS/fiducial-cache exist), `felt show definitive-l1-vs-cnn-10deg-2026-06`,
read PLAN + the 20° FINDINGS, then **propose the Phase A dataset-validation plan and get Andreas's
sign-off before any GPU job.**
