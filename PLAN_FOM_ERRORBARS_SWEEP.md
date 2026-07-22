# PLAN — FoM error-bar sweep for paper Table 1 (G-03)

Written 2026-07-22 on the laptop, for the first titan session after cluster
access returns. Full rationale (error budget, what the ± means, presentation)
lives in the paper repo: `L1_vs_CNN_Tomographic_SBI/NOTE_FOM_ERROR_BARS.md`.
This file is the compute side only. Mostly CPU; the only possible GPU cost is
CNN compressor retrains (step 3).

## Goal

Per-row ± for the paper's Table 1 (arms: ℓ₁ auto-only, ℓ₁+product, joint ℓ₁,
CNN) = the 3-compressor-seed spread of the robust median FoM₃ (and σ triplet),
under the FINAL recipe, plus a block-bootstrap SE of the median.

## Protocol (identical to RESULT_JOINTL1_SEEDCHECK.md — the template)

For each arm, for each compressor seed c ∈ {41, 42, 43}:
VMIM(seed c) → sbi_lens RealNVP 4×128, NDE seeds 41/42/43 pooled →
robust median FoM₃ + σ(Ωm, σ8, w0) over the SAME n = 9000 fiducial mocks
(180 patches × 50 noise reps) used for the paper's table.

Quote per arm: the three per-seed values, mean ± std, and min–max band.
These are PRE-ensemble singles by design — the ± measures training
stochasticity; do NOT conflate with the single→ensemble de-inflation
(that is the bias term, already in RESULT_BNT_AUTOPROD_ENSEMBLE.md /
RESULT_NOBNT_ENSEMBLE_ROBUSTNESS.md).

## Per-arm status

1. **joint ℓ₁ — DONE.** 3754 / 3761 / 4034 (spread 7%),
   `scripts/sbi/results/exploratory/flatsky_cross_2026_06/analytical_nde_match/RESULT_JOINTL1_SEEDCHECK.md`.
2. **ℓ₁ auto-only and ℓ₁+product — eval only (CPU).** Compressor caches exist,
   built for the 2026-06-27 ensembles: `l1none_vmim_s41`,
   `ens_nobnt_auto_s4{2,3}`, `l1product_vmim_s4{1,2,3}` (see
   `run_bnt_autoprod_ensemble.py`, `ensemble_eval.py`). Run the per-seed
   n=9000 median sweep on each cache separately.
3. **CNN — verify vintage first.** Existing per-seed material
   (`cnn_phase/best_seed/per_seed.json`, cnn seed dirs) is arch-sweep-era,
   NOT the canonical resnet18+RealNVP recipe — unusable for the paper. If no
   final-recipe s42/s43 compressors exist, retrain 2 seeds under the
   canonical recipe (see `HANDOFF_CANONICAL_REFRESH.md`,
   `ESTIMATOR_OPTIMIZATION_RECORD.md`), then the same eval.

## Block bootstrap of the median (per arm, seed 41)

The 9000 mocks are 180 patches × 50 noise reps — not independent. Resample
the 180 patches with replacement, keep all 50 reps of each sampled patch,
recompute the robust median; 10⁴ replicates; report the 68% percentile
interval. Also measure, from the per-obs FoM array: CV_pop (population
coefficient of variation) and the intra-patch correlation ρ (one-way ANOVA:
ρ = τ²/(τ²+σ_w²)). Expected: SE(median) ≈ 0.1–3% depending on ρ —
subdominant to the seed term; the measurement replaces the guess.

## Outputs

- `RESULT_FOM_ERRORBARS.md` — per-arm per-seed table + bootstrap SEs + ρ/CV.
- `per_seed_fom3.json` per arm ({seed: {fom3, sigma:[Om,s8,w0]}}), and
  `bootstrap_median_se.json`.
- Paper-side: the ± column values (mean ± std, 3 seeds, flagged as
  indicative) + one caption sentence — wording already drafted in the paper
  repo's NOTE §6–7.

## Optional extensions (do only if cheap after the above)

- 5 seeds for the analytical arms (2 extra VMIM trainings each, small).
- Per-seed BNT/no-BNT retention ratios for the paper's Table 2 (BNT ensemble
  caches exist; retention per seed, then spread — seeds are shared between
  frames, so this is tighter than naive propagation).

## Other open titan items (context: paper repo REVISION_TRIAGE.md, flag list)

- Pooled TARP/SBC gate on the twopt-split ensembles — REQUIRED before the
  de-inflated ΔNG(conv)=124 / ΔNG(product)=260 may ever be quoted
  (CONV_MAP_SECURE_RESULT.md §8).
- Noiseless-maps BNT run artifact — locate for provenance (run attested by
  Andreas; paper sentence stands).
- ℓ₁+conv BNT arm — does not exist; only re-run if wanted (paper needs
  nothing).
- Transition-band cut–invert experiment (novel) — spec in paper repo
  `NOTE_BNT_CUT_INVERT_MIXING.md` §7–8.
