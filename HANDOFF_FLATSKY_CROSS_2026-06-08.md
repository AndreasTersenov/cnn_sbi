# HANDOFF — Flat-sky cross-maps campaign (start here)

**Date:** 2026-06-08. **For:** a fresh Claude Code session continuing the weak-lensing SBI work.
**Felt fiber:** `.felt/flatsky-cross-2026-06/` (active). **One-line mission:** rebuild the
tomographic cross-maps **patch-locally (flat-sky)** instead of the leaky full-sphere way, recompute
the statistics, train L1 + CNN, and produce **calibrated cosmological contours** — the physically
defensible auto+cross result.

---

## Why this campaign exists (the discovery that triggered it)

The previous campaign (`definitive-l1-vs-cnn-10deg-2026-06`) concluded "CNN ≥ L1, decisive on
auto+cross." Then we found a **methodological bug**: the cross-maps are built as a **full-sphere**
harmonic alm-product (`aₗₘ^i·aₗₘ^j → alm2map`) and *then* sliced into 10° patches, so **every
cross-patch pixel is a global functional of the whole sky** — each patch leaks cross-correlation
information from outside its own footprint. Consequence: the auto+cross constraining power (and
especially the CNN's gain) is **partly unphysical** — a real survey patch cannot build these maps.
It is **not** a calibration bug (the leak is self-consistent in train+test, so TARP/SBC pass); it is
a **data-vector realism** problem. Auto-only is unaffected (autos are local). Full write-up:
**`CROSS_MAP_LEAKAGE_FINDING.md`**.

So the auto+cross headline of the prior campaign is **provisional** until redone with patch-local
cross-maps. That is this campaign.

---

## Read first, in this order

1. **`CROSS_MAP_LEAKAGE_FINDING.md`** — why we're doing this (the leakage, quantified).
2. **`FLATSKY_CROSS_REDESIGN_NOTES.md`** — the full design + validation record (§1–14). Load-bearing:
   §7 (what Zürcher 2022 actually does), §8 (flat-sky = convolution; the operator was not the bug),
   §12 (validation; the "ill-posed" claim was RETRACTED — a registration artifact), §13 (agreed plan),
   §14 (ξᵢⱼ recovery).
3. **`FLATSKY_CROSS_BUILD_PLAN.md`** — the concrete build/run plan with gates. Your working doc.
4. The memory index `…/memory/MEMORY.md` (esp. `project_cross_map_leakage_fullsphere`,
   `feedback_l1_cross_must_use_harmonic_route`, `feedback_never_pca_l1`,
   `feedback_fom3_fragile_use_2d_areas`, `feedback_gpu1_only`,
   `reference_jaxili_checkpoint_reload_truncation`).
5. Prior result for comparison: `…/definitive_comparison_10deg/phase_c/analysis/SUMMARY_PHASE_D.md`.

---

## What is DECIDED (do not relitigate)

- **Two cross operators, both patch-local, test both** (they are complementary, not redundant):
  - **Convolution** `Cᵢⱼ = irfft2(rfft2(κᵢ·W)·rfft2(κⱼ·W))` — apodized-circular (Zürcher flat-sky
    analog; smooth/large-scale; cross-info in morphology). **One** definition; the zero-pad+crop
    variant is dropped (differs only by a 39-px crop-offset shift + small edge wrap).
  - **Product** `Pᵢⱼ = κᵢ·κⱼ` — pointwise (its mean = ξᵢⱼ; scale-preserving; strictly local).
- **No sim/dataset rebuild** — cross is computed on-the-fly from **auto channels 0–3** of the existing
  TFDS `grid_10deg_80px_nonoverlap180`. Auto-only baseline uses the *same* autos → clean comparison.
- **Per-channel (ideally per-scale) noise/SNR** — the fix for the old shared-auto-σ bug. For CNN:
  per-channel RMS normalization.
- **Primary metric:** median over typical patches of **σ(w₀) + 2D(Ωm,σ8)**. FoM3 reported, never
  headlined.
- **Simple pointwise product** + existing multiscale ℓ₁ (NOT the per-(pair×scale) scale-matched
  product — that's a backlog item, §10 of the notes).

## What is DONE

- Leakage found, quantified, documented (+ memory + the headline memory caveated).
- Old flat-sky `--cross-maps` route found and critiqued; Zürcher 2022 read; operators analyzed.
- Cheap construction validation (`validate_flatsky_cross.py`): product detects ξᵢⱼ (529× vs
  independent); ξᵢⱼ-recovery matrix is physically correct; convolution mean = 0 (unapodized).
- All committed (`git log`: leakage + redesign-notes + validation commits). Felt fiber created.

## What is NOT done (this campaign's work)

Implement the augmentation → GATE A (construction) → GATE B (cosmology-dependence, NEW & decisive) →
train the matrix → GATE C (calibration) → contours vs auto-only and vs the full-sphere result. See
`FLATSKY_CROSS_BUILD_PLAN.md` §6 for the gated sequence.

---

## Environment & running (titan)

- Env: `conda run -n jaxili python …` (or `/home/tersenov/anaconda3/envs/jaxili/bin/python`),
  `PYTHONUNBUFFERED=1`, `XLA_PYTHON_CLIENT_MEM_FRACTION≈0.35`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- **GPU 1 ONLY** for every job (`--cuda-visible-devices 1`). Never GPU 0/2/3.
- Detached long jobs: `setsid nohup … &`, poll with `pgrep -f "[p]attern"` / log-grep (shell `wait`
  returns early under setsid; **never** `pkill -f` a self-matching pattern).
- titan has **no scheduler** — run directly (no sbatch/srun).
- Data: TFDS at `/home/tersenov/tensorflow_datasets`; fiducial obs cache
  `…/cross_maps_campaign/full_sphere_cache_fiducial_10deg` (200 perms, auto channels reusable).

## Guardrails (non-negotiable — these are how we avoid wrong conclusions)

Patch-local cross ONLY · per-channel noise (not shared auto-σ) · never PCA L1 (`--pca-components 0`) ·
don't headline FoM3 · calibrate BEFORE contours · example-disjoint compressor/NDE split by perm ·
same auto channels across all arms · one apodized-circular convolution definition · stage git files by
path (never `git add .`/`-A`; don't commit artifacts/figures unless asked) · measure perf, don't guess.

## First concrete actions for the new session

1. Read the 4 docs above; recreate env; confirm GPU 1 free.
2. Implement the flat-sky augmentation: L1 `--cross-op {conv,product,both}` (the route exists; add
   product + per-channel noise), and ADD flat-sky support to `npe_cnn_nbody_tomo.py` (reuse
   `_compute_cross_maps_*` + per-channel RMS). Keep tf (train) / np (obs) bit-identical.
3. **GATE A** before any training: bit-match; re-run `validate_flatsky_cross.py` on loader output;
   ξᵢⱼ recovery; per-channel noise sane.
4. **GATE B**: confirm cross statistics move with cosmology across the TFDS (the decisive info test
   the fiducial-only check could not do).
5. Then train the matrix, GATE C calibration, contours. Update felt + memory + write-up as you go.

## What success looks like

A calibrated, patch-local auto+cross result for L1 and CNN with: (a) the honest cross-gain over
auto-only (expected **modest** — cross info is large-scale, a 10° patch samples it poorly); (b) a
clear conv-vs-product (and "both") comparison; (c) a quantified contrast to the inflated full-sphere
number; (d) σ(w₀) + 2D areas as the headline, FoM3 only as support; (e) all calibration clean. That
result is what goes in the paper.
