# PLAN — Phase D: per-patch diagnostics + the headline (10°)

Status: **PLAN (awaiting Andreas sign-off).** Created 2026-06-06.
Campaign: `definitive-l1-vs-cnn-10deg-2026-06`. Prereq: Phase C done — 12 NDEs trained,
compressed/datavector caches under `…/definitive_comparison_10deg/phase_c/{cnn,l1}_<mode>_*`.

## Why Phase D (what Phase C could NOT answer)

Phase C gave a **single-patch** (patch 90) read: CNN ≳ L1. But the 20° work proved the
nominal winner **flips with obs-patch and metric**, and L1 has large patch-to-patch spread
(`project_l1_patch_sensitivity_full200`, `feedback_fom3_fragile`). The robust answer needs
the **per-patch population median**. Phase D delivers the two real results:

1. **Robust L1-vs-CNN**: median σ(w0), 2D(Ωm,σ8) over the patch population (does the
   single-patch "CNN ≳ L1" hold, or flip?), + the geometry-vs-realization variance split.
2. **★ THE HEADLINE ★**: does L1's fiducial **w0 offset** (20°: −0.37σ) **SHRINK at 10°**
   (⇒ flat-sky distortion was the cause) **or PERSIST** (⇒ intrinsic ℓ₁ compression bias)?
   Plus SBC (global calibration) + L-C2ST (local calibration at the fiducial).

## The diagnostic scripts (exist; re-train NDE from the Phase-C cache + read obs summaries)

`geometry_resample.py` loads `{cnn,l1}_train.npz` (Phase C cache), re-trains the MAF per
seed, and computes a posterior for every (patch,perm) in a **`--summaries-npz`** → per-patch
`fom3/2D/σ/bias/pull`. `geometry_analyze.py` → variance decomposition + bias. `compare_offsets.py`
→ the offset table. `sbc_diagnostic.py` / `lc2st_diagnostic.py` → calibration (L-C2ST `--clf-kind logreg`).

## D-prep — build the per-patch fiducial summaries + small repoints

The `--summaries-npz` (per-patch obs summaries `S/perm/patch/theta`) does NOT exist yet — it's
the main prerequisite. Build it for all 4 arms over the fiducial cache (180 patches × P perms):

1. **CNN summaries** (`build_fiducial_summaries_cnn.py`, repoint): run each CNN arm's **trained
   compressor** (Phase-C `cnn_<mode>_s41` checkpoint) over the fiducial cache → `S` per patch.
   Verify it reads the `tfds_cross` compressor + the 10° fiducial cache + applies the same
   channel-scale/slice as training.
2. **L1 summaries**: **wire `--fiducial-summaries-out` for `tfds_cross`** (currently raises
   NotImplementedError at `npe_l1norm_cross…:2513`). The block at `:2935` already reads the
   fiducial obs cache + writes `S/perm/patch/theta`; point it at `--fiducial-obs-cache-dir`
   with the **TFDS-calibrated σ_c + SNR range** (compute as in a normal `tfds_cross` run, then
   loop the fiducial perms/patches through `compute_l1_batch`). One summaries npz per L1 arm.
   - **Verify**: the obs patch-90/perm-0 row of the L1 summaries == the Phase-C single-obs L1
     datavector (same σ_c/SNR) → confirms the summary builder matches the arm.
3. **Repoint `geometry_resample.py`**: `FIDCACHE` → `full_sphere_cache_fiducial_10deg`;
   `--patch-indices 0-179`. **Repoint `geometry_analyze.py`**: the hardcoded `48` in the
   η²-null (`:235,240,241`) → read `n_patch_indices` from the data (180).
4. **G3 gate**: set `--expected-fom3` per arm to the Phase-C patch-0/perm-0 3-seed FoM3
   (reproduce within `--g3-tol 0.20` before trusting the sweep; aborts on FAIL).

## D-run

| step | script | arms | output |
|------|--------|------|--------|
| D-1 | `geometry_resample.py` | all 4 | per-patch grid CSV/NPZ (fom3/2D/σ/bias/pull × patch×perm) |
| D-2 | `geometry_analyze.py` | all 4 | median σ/2D/FoM3, η² geometry-vs-realization, bias-vs-lat |
| D-3 | `compare_offsets.py` | L1 vs CNN, auto vs cross | **the w0-offset shrink/persist table** |
| D-4a | `sbc_diagnostic.py` | all 4 | global rank uniformity (the offset cancels over the prior?) |
| D-4b | `lc2st_diagnostic.py --clf-kind logreg` | CNN a+c (+L1 if powered) | local calibration at fiducial |

- **Obs population:** 180 patches × **50 perms = 9000 obs/arm** (≈ the 20°'s 9600; keeps all
  patch geometry, enough perms for the variance split). [decision below]
- **Seeds 41/42/43** (3-seed-pooled), campaign-exact preproc (L1 log1p-zscore-clip5-mask1e-5;
  CNN none-mask1e-12), matching `geometry_resample` defaults.

## Decisions to confirm
1. **Population size:** 180 patches × **50 perms** (9000 obs/arm). [alt: 200 perms = 36000 →
   ~3.75× cost. 50 matches 20° power and keeps runtime ~1–1.5 h/arm.] Build the L1 fiducial
   summaries for the SAME 50 perms (≈33 min/arm at 268 dv/s) — not all 200.
2. **Which arms get full diagnostics:** all 4 for D-1/2/3; **L-C2ST CNN-only** by default
   (20° rationale: L1's 2000-d x is underpowered/flaky for a plain L-C2ST classifier; L1 local
   miscalibration is established directly via the offset + SBC). Confirm or extend to L1.

## Runtime / scheduling (GPU 1+2)
- D-prep summaries: CNN fast; L1 ~33 min/arm (×2). D-1 resample ~1–1.5 h/arm × 4 (the bulk).
  D-2/3 instant (CPU). D-4 SBC/L-C2ST ~30–60 min each. Total ~6–9 h across GPU 1+2.
- A small `run_phase_d_10deg.py` (mirror the Phase-C scheduler) to fan the 4 resample arms +
  summaries-builds across GPU 1+2; `--dry-run` prints commands.

## Deliverables / done condition
- `SUMMARY_PHASE_D.md`: median σ(w0)+2D table (robust L1-vs-CNN, replaces the single-patch
  Phase-C read) + the variance decomposition.
- **`OFFSET_VERDICT.md`**: L1 w0 offset at 10° vs the 20° −0.37σ → **shrinks or persists** (the
  headline), with the auto-vs-cross contrast (is it cross-map-specific as at 20°?).
- Calibration findings (SBC global + L-C2ST local).
- **Phase E** = the 10°↔20° comparison written into these (median-vs-median, offset verdict).

## Guardrails
- GPU 1+2 pinned (never 0/3). Lead with σ/2D, FoM3 reported not headlined. Never PCA L1.
- G3 gate per arm before trusting a sweep. Verify L1 summary builder vs the Phase-C obs datavector.
- Never `git add`/commit without Andreas's OK.

## After sign-off
Wire L1 `--fiducial-summaries-out` (tfds_cross) + repoint geometry_{resample,analyze} →
verify L1 summary vs Phase-C obs → build the 4 summaries → run D-1…D-4 across GPU 1+2 →
`compare_offsets` + the two writeups → Phase E.
