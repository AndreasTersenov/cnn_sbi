# HANDOFF — steps 1–4 autonomous block (2026-05-31, Andreas away ~8 h)

Launched ~10:55 UTC. All detached + self-driving; nothing needs babysitting.
**The deliverable to read first:** `results/exploratory/definitive_comparison/PHASE_C_2026_05_31/SUMMARY_DEFINITIVE.md`
(σ/2D primary, FoM3 secondary — auto-refreshes at the end via `aggregate_all_arms.py`).

## What's running (`run_steps_overnight_2026_05_31.sh`, two GPU branches)

**GPU 1 — finish TARP coverage:**
- L1 TARP dumps (auto+cross & auto-only split70, N=200×M=2000) → then `run_tarp_coverage.py`
  re-plots **all** arms together (CNN-RealNVP, CNN-MAF, L1) in **dim-3 and dim-6**.
- Output: `tarp_2026_05_31/figures/tarp_{per_arm,overlay}_dim{3,6}.png`, `tarp_summary.json`.
- (CNN four already done before this block.)

**GPU 0 — steps 3 → 2 → 1 (each independent; a failure is logged, never cascades):**
- **Step 3 (patch-center control, plan arm 4):** native-TFDS-auto compressor (RealNVP, 80k,
  `--cnn-map-route tfds`, 4 ch) + jaxili NDE seeds 41/42/43 → `phaseB_nativeauto_2026_05_31/`.
  If it matches our harmonic-cache auto-only, the G8 patch-center confound is ruled out.
- **Step 2 (standardization, plan arm 6):** jaxili NDE with `--standardize-summary` on the
  existing RealNVP auto+cross cache, seeds 41/42/43 → `phaseB_std_2026_05_31/`. Tests whether
  z-scoring the summary destroys information.
- **Step 1 (CNN multi-perm):** the cheap obs-only recompress (`--no-train --harmonic-obs-perm
  {1,2} --harmonic-*-realizations-limit 1`) to get the fiducial obs at perms 1,2, then NDE with
  `--obs-files p0,p1,p2` → `phaseB_multiperm_2026_05_31/` (3 seeds × 3 perms per arm).
  **GATED:** validates the obs recompress on (autocross, perm1) first; if it produces no
  `cnn_obs.npz`, multi-perm is **skipped** (logged) and CNN stays perm-0 — no wasted compute.

**Phase C (step 4):** when both branches finish, `aggregate_all_arms.py` writes
`SUMMARY_DEFINITIVE.md` + `phase_c.csv` across every arm that produced posteriors.

## Markers / where to look
- `steps_overnight_2026_05_31/steps.log` — live progress; `.GPU0_DONE`, `.GPU1_DONE`, `.STEPS_DONE`.
- Per-step logs in `steps_overnight_2026_05_31/logs/`.
- ETA: GPU 1 (TARP) ~20–30 min; GPU 0 (step3 ~50 min + step2 ~10 min + step1 ~30–60 min) ~2 h.

## Risk notes (honest)
- **Step 1 (multi-perm)** is the least-tested path (the `--no-train` obs recompress). It's gated,
  so worst case it self-skips and we note "CNN perm-0 only" in the writeup.
- **Step 3** uses `--compressor-val-split test` (TFDS-auto convention, matches the 2026-05-28
  native-auto run). If that split name is wrong it fails fast and is logged.
- TARP curves use the existing `run_tarp_coverage.py` (3-D + 6-D), so they match prior repo plots.

## Context (decided earlier today)
- Companion question CLOSED: beefier MAF companion is **worse** (auto+cross FoM3 ~0.45×, never
  better across 5 seeds) → companion is not the CNN bottleneck. See `companion_comparison_2026_05_31/`.
- Clean disjoint rerun **deprioritized** (overlap empirically negligible per Andreas); `.npz`
  loader is GIL-bound so `--harmonic-loader-threads` doesn't help. Constitution updated.
- New code committed: MAF companion (`0d58d5e`). Uncommitted (additive): `--harmonic-loader-*`
  passthrough, `tarp_from_compressed.py`, `aggregate_all_arms.py`, the orchestrators.
