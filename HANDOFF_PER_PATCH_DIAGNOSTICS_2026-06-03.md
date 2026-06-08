# HANDOFF — Per-patch diagnostics (next phase), 2026-06-03

**Read this first**, then the felt constitution + the open fiber (below), then `CLAUDE.md`.
This session moved the L1-vs-CNN comparison from **3 fiducial observations to the full 200
realizations (9600 patches)** — which both corrected the headline and unlocked much stricter
diagnostics. The next session continues with those diagnostics to understand **why** the
results look the way they do.

---

## 0. How to work here (carry over)
- **Check, don't guess. Never fabricate.** Every number below was read off disk. Don't quote a
  perf/time number you haven't measured on-node.
- **Plan before non-trivial code; get Andreas's sign-off.** "Don't start coding" is load-bearing.
- **GPUs:** Andreas overrode the project "GPU-1-only" rule for this campaign → use **GPU 0 + 1**.
  GPU 2/3 have other tenants (bonjean/titan) — never touch. Pin every job; check `nvidia-smi` first.
- **Felt:** drive with the CLI (`felt ls/show/add/edit/history`). `[[wikilinks]]` are for FIBERS
  only; memories/docs in plain prose. `felt check` must stay clean. Load the canonical skill +
  `FELT_AGENT_GUIDE.md` + CLAUDE.md §"Felt / Ralph operating conventions".
- **Git:** never `git add .`; stage by path. **Don't commit without explicit OK.** (Nothing from
  this whole investigation is committed.)
- **Env:** `/home/tersenov/anaconda3/envs/jaxili/bin/python` + `PYTHONUNBUFFERED=1`
  (`XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30-0.40`). Never install pkgs.
- **Memories** updated this session: `project_l1_patch_sensitivity_full200`,
  `project_perm_averaging_overturns_l1_autocross_lead`, `feedback_fom3_fragile_use_2d_areas`,
  `project_patch_center_confound_g8`. Read them.

---

## 1. TL;DR — where we are (settled)
**Corrected definitive headline (typical obs patch):** L1 ≈ CNN on **auto+cross** with a *small* L1
edge concentrated in **w0 / the cross-maps** (σ(w0) ×1.34, σ(Ωm) ×1.18, 2D(Ωm,σ8) ×1.6; FoM3 ×2.17
but **FoM3 amplifies ~20–25% marginal diffs into ~2×**); **auto-only is a tie**. Both calibrated.
The earlier "CNN ≳ L1 auto+cross" was an **artifact of the campaign's fixed obs = patch-0 = the
POLAR patch** (lat 88.5°), atypically low-info for L1's near-polar wavelets; CNN is patch-insensitive.
This is folded into `PHASE_C_2026_05_31/SUMMARY_DEFINITIVE.md` (corrected headline at top).

## 2. The big step (and why it matters)
We had been conditioning everything on **3 fiducial "observations"** (perms 0/1/2 at patch-0).
This session built the **full 200 realizations** of the fiducial cosmology and sliced all 48 patches
→ **9600 per-patch observations**. Consequences:
- **More correct results:** the per-patch *distribution* (not one noisy patch) is the honest unit;
  it revealed patch-0 was a polar outlier and corrected the headline.
- **Stricter diagnostics:** a large sample enables per-geometry maps, spread decomposition, bias
  structure, SBC, and stratified calibration — the next phase.

## 3. Findings established this session (correct & useful)
1. **L1 is highly patch-variable; CNN is stable.** Per-patch FoM3 (single 20 deg² obs): L1 a+c
   median ~53k spanning ~17k–250k; CNN a+c ~24k, tight. (`overlays/fom3_distribution.png`,
   `figures/headline_typical_patch_violins.png`, `posteriors/*/per_patch_fom.csv`.)
2. **patch-0 = polar; OOD ruled out.** `corr(per-patch FoM3, OOD) ≈ 0`; patch-0 is the polar patch
   (most extreme |lat|). (`patch_anomaly_diagnosis.json`, `tarp_per_patch/figures/reversal_B*.png`.)
3. **Tight L1 posteriors are calibrated** (not over-tight). Stratified **varied-θ** TARP-DRP,
   HIGH-FoM3 tercile max|ECP−α| dim3 = 0.068 (L1) vs 0.095 (CNN); both mildly over-confident.
   (`tarp_stratified/figures/tarp_per_arm_dim{3,6}.png`.)
4. **Corrected headline table:** `fiducial_full200/SUMMARY_TYPICAL_PATCH.md`,
   `FIDUCIAL_FULL200_FINDINGS.md`.

## 4. Clean artifact set (after cleanup) — `…/definitive_comparison/fiducial_full200/`
KEEP/valid:
- `SUMMARY_TYPICAL_PATCH.md`, `FIDUCIAL_FULL200_FINDINGS.md`, `patch_anomaly_diagnosis.json`
- `figures/headline_typical_patch_violins.png`, `overlays/fom3_distribution.png`
- `tarp_per_patch/figures/reversal_B_l1_polar_vs_typical.png`
- `tarp_stratified/figures/*` (the valid TARP)
- `summaries/*_S.npz` (per-patch summaries — see §5), `posteriors/*/per_patch_fom.csv` + `step2_*`
- `tarp_per_patch/dumps/{l1,cnn}_autocross/seed_*/n260_m2000/posterior_samples.npz`
  (per-patch posterior SAMPLES, ~260 patches, a+c only — used for corners), `…/coverage/*/coverage_arrays.npz`
- Phase C: `PHASE_C_2026_05_31/SUMMARY_DEFINITIVE.md` (corrected headline at top) + `phase_c.csv`

DELETED this session (wrong/misleading — do NOT regenerate): degenerate fixed-θ TARP-DRP plots,
confounded Mahalanobis coverage + `coverage.json`, OOD mean-datavector posteriors+corners,
single-patch corners (reversal_A/C + 4-arm), the auto-gen `FIDUCIAL_FULL200_SUMMARY.md`.

## 5. Infrastructure (scripts in `scripts/sbi/`, data paths)
**Data:**
- Fiducial cache (200 perms): `…/cross_maps_campaign/full_sphere_cache_fiducial/nobnt/obs/cosmo_fiducial_perm{0..199}.npz`
  — each `patches (48,160,160,10) float32`, `patch_centers (48,2)` (lon,lat deg), `theta`, etc.
- Grid cache (NDE training + L1 calibration): `…/cross_maps_campaign/full_sphere_cache_grid`.
- Per-patch summaries (9600×dim + `perm`,`patch` arrays; **gi = perm*48 + patch**, consistent across
  arms — match by (perm,patch) to be safe): `fiducial_full200/summaries/{l1_autocross,cnn_autocross,
  cnn_maf_autocross,l1_autoonly,cnn_autoonly}_S.npz`. (CNN-std reuses `cnn_autocross_S`.)
- NDE training caches: CNN `…/phaseA_tfdata_2026_05_30/compressed/{autocross,autoonly}_s41/{cnn_train,cnn_val}.npz`;
  CNN-MAF `…/phaseA_maf_2026_05_31/compressed/autocross_s41/`; L1 `…/compressed/l1_{autocross,autoonly}_split70_dv/{l1_train,l1_val}.npz`.

**Scripts:**
- `build_full_sphere_cross_cache.py` — built the 200 perms (fixes: `perm_{p:04d}` zero-pad ×2 sites;
  `--cosmo-id` filter to build only cosmo_fiducial).
- `build_fiducial_summaries_cnn.py` — CNN per-patch summaries (reuses compressor + channel-RMS; G1 self-check).
- `npe_l1norm_cross_jaxili_nbody_tomo.py` — additive hook `--fiducial-summaries-out / --fiducial-perms /
  --fiducial-obs-cache-dir` (after obs_l1; reuses in-scope calibration; G1 + G1-calib gates).
- `fiducial_analyze.py` — per-arm step1(mean-dv; **deprecated, OOD**) + step2(per-patch dist); **G3 gate**
  (reproduce campaign perm-0 FoM3 within 20%). **This is the template for re-sampling per-patch posteriors.**
- `tarp_stratified_val.py` — the VALID varied-θ stratified TARP (use this for calibration questions).
- `diagnose_l1_patch_anomaly.py` — NDE-free geometry + OOD diagnostic (#1). **Template for geometry work.**
- `aggregate_all_arms.py` — Phase C summary generator; now folds in the typical-patch corrected headline
  (`_typical_patch_section` reads `fiducial_full200/posteriors/*/per_patch_fom.csv`).
- `make_reversal_corners.py`, `make_headline_corner.py`, `plot_fiducial.py` — viz.
- Orchestrators: `run_fiducial_full200.sh`, `run_tarp_stratified.sh` (+ `run_tarp_per_patch.sh`,
  the latter produced the now-deprecated fixed-θ coverage).
- `tarp_per_patch_fiducial.py` — **DEPRECATED** (fixed-θ coverage is degenerate/confounded; outputs deleted).

## 6. Gotchas / dead ends (do NOT resurrect)
- **Mean-datavector (step-1)** → OOD over-tightening for L1 (high-dim nonlinear summary; de-noised mean
  off-distribution). Don't use the mean of summaries as an obs for L1.
- **Fixed-θ coverage** (Mahalanobis-at-fiducial; TARP-DRP on a single-θ ensemble) → degenerate (DRP needs
  θ~prior; curves pin to 1.0) / shrinkage-confounded. For "is this posterior over-tight," use a **varied-θ
  stratified TARP**, never coverage-at-a-fixed-point.
- **FoM3 is fragile** — cubes ~20–25% diffs into ~2×. Lead with σ / 2D.
- **patch-0 = polar** — never use as the obs; use a typical/averaged patch.
- **L1 channel_scale MULTIPLIES** the maps (= σ_auto/σ_c), CNN DIVIDES (RMS). Different conventions.
- **L1 NDE preprocessing** = log1p→zscore(train)→clip ±5→mask(var>1e-5). **CNN** = mask(var>1e-12) +
  optional z-score, NO clip. (Needed when re-sampling/analyzing posteriors — see `fiducial_analyze.py`.)
- Per-patch posterior **samples** are saved only for the a+c arms (tarp_per_patch dumps). Diagnostics
  needing samples for other arms/patches must **re-sample** — cheap: summaries are cached, so only NDE
  train + sample (no map recompute). ~4 min/NDE train (50000 epochs), 3 seeds.
- orbax wants ABSOLUTE paths; `mkdir -p $ROOT` before any `> $ROOT/...` redirect; `pgrep -f` self-matches
  (use `[_]` bracket trick / stored PID); detached launches via `setsid nohup … &`.

## 7. NEXT PHASE — understand WHY (felt fiber: `understand-per-patch-structure-2026-06`)
Goal: use the large per-patch sample to understand the *mechanism*, not just measure. Threads:
1. **GEOMETRY map (do first).** Per-patch-index FoM3 + posterior bias (mean−truth) for each of the 48
   patch indices, averaged over perms (geometry-resolved, noise-averaged). Plot vs patch latitude/position.
   Hypothesis: L1's variability + the polar-patch lowness tracks |lat| / gnomonic-projection distortion.
   *First concrete step:* re-sample per-patch-index systematically (all 48 indices × ~30–50 perms each)
   using the `fiducial_analyze.py` pattern (summaries cached → NDE train + sample only); for L1 a+c and
   CNN a+c. Then map FoM3/bias vs `patch_centers[:,1]` (latitude). `diagnose_l1_patch_anomaly.py` is the
   NDE-free starting template (already shows patch-0 is the most-OOD/polar index).
2. **Decompose the spread:** variance of per-patch FoM3 across patch-INDEX (geometry, fixed sky) vs across
   PERM (realization noise) at fixed index. Is L1's huge spread geometry- or noise-driven?
3. **Bias structure:** is L1's posterior-mean center-wander structured (geometry-correlated) or random,
   vs CNN? Map per-patch bias.
4. **The w0 question:** why does L1 extract more w0 from the cross-maps than CNN?
5. **Stricter calibration:** SBC over the population; coverage stratified by geometry (latitude bands).
6. **(optional)** Clean typical-obs definitive table across ALL 10 arms (extend full-200 to native-TFDS auto etc.).

## 8. Felt state
- Constitution `definitive-l1-vs-cnn-2026-05` (OPEN) — outcome + Loop Status updated to the corrected
  headline; original objective settled.
- Closed: `finding-patch-center-confound-g8`, `finding-perm-averaging-overturns-l1-lead`,
  `finding-l1-patch-sensitivity-full200`, `fiducial-full200-meandv`, `maf-companion-not-bottleneck`,
  `bug-multiperm-no-train-flag`, `refine-phase-c-perm-matched`.
- **OPEN (your task): `definitive-l1-vs-cnn-2026-05/understand-per-patch-structure-2026-06`** — the diagnostic threads above.
- `felt check` clean.

## 9. Reading order for the next session
1. This handoff. 2. `felt show definitive-l1-vs-cnn-2026-05` + `felt show …/understand-per-patch-structure-2026-06`.
3. `fiducial_full200/FIDUCIAL_FULL200_FINDINGS.md` + `SUMMARY_TYPICAL_PATCH.md`. 4. The 4 kept plots.
5. `CLAUDE.md` + the updated memories. Then propose a plan for Thread 1 and get sign-off before running.
