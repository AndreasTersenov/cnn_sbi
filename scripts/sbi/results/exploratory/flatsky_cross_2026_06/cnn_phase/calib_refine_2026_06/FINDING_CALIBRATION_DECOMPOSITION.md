# Finding — CNN is mildly conservative (safe direction); L1 is the slightly over-confident one; the saved TARP bands were bogus

**Date:** 2026-06-22. **Arm:** resnet18 compressor + sbi_lens RealNVP 4×128, FoM3 3326 (s41) / 3304
(3-seed mean). **Source:** read-only re-analysis + recompute from the existing GATE-C posterior dumps
(no new training). All TARP recomputed with env-`jaxili` `tarp`, same convention as
`run_tarp_coverage.py` (first 3 params; samples→(M,N,3); `references="random"`; `norm=True`).

> This supersedes the first version of this note (which claimed "net ≈ 0, sign-changing, global sharpen
> contraindicated"). Both of those were artifacts of the pipeline's degenerate saved curves; the
> corrected, consistent recompute below is simpler.

## Two bugs in the standing calibration numbers
1. **Bogus uncertainty band.** The pipeline's saved TARP `ecp_bootstrap` resamples only the random
   *reference points* (per-bin std ≈ 1e-4), so every saved band was ~200× too small. The real
   uncertainty is set by the N=600 validation sightlines: 1σ ≈ ±0.020 (binomial SE √(0.25/600)=0.0204).
   Recomputed by bootstrapping the 600 sightlines, our SE@0.5 = 0.0198 — matches. **All plots in
   `calib_refine_2026_06/figs/` use this proper 1σ band.**
2. **Single-flow vs reported (pooled) object.** The saved per-tercile curves were single-NDE-flow and
   carried a ~+0.02 systematic offset vs a transparent recompute. Recomputing on the *reported* pooled
   posterior (3 NDE seeds) removes the spurious structure (see terciles below).

## Calibration, recomputed consistently (same method for every arm)

Un-stratified TARP-DRP net (+ = conservative/over-covers; − = over-confident), 1σ ≈ ±0.020:

| Arm | TARP net | SBC rank-std (Ωm/σ8/w0) | reading |
|---|---|---|---|
| **CNN** (resnet18+RealNVP) | **+0.033** | 0.290 / 0.289 / 0.282 | mildly conservative (safe) |
| **L1 auto+product** (MAF) | **+0.001** | 0.296 / 0.300 / 0.295 | joint-calibrated; marginals slightly over-confident |
| **joint L1** (3-seed ensemble) | **+0.004** | 0.299 / 0.298 / 0.298 | joint-calibrated; marginals ~ideal |

CNN FoM3-stratified terciles (proper 1σ, pooled posterior): **LOW (widest) +0.053, MID +0.002,
HIGH (tightest) +0.021** — mildly conservative across the board, strongest at the wide end, **no
compensating under-coverage** (the earlier "MID under-covers −0.039" was a single-flow/saved-curve
artifact). SBC marginals flat within the 99% binomial band for all arms (ideal rank-std 0.289;
>0.289 = over-confident/narrow, <0.289 = conservative/wide).

Caveat: absolute TARP net carries ~±0.01–0.02 method sensitivity (TARP reference/normalization
conventions); the **relative, same-method** comparison (CNN ~+0.03 more conservative than the L1
variants) is the robust statement.

## Interpretation
- The CNN errs on the **safe** (conservative) side: joint volume slightly too large, marginals
  calibrated (w0 a hair wide). The two **analytical L1** summaries are joint-calibrated but their
  **marginals are slightly over-confident** (SBC ≈ 0.30 > 0.289), L1+product most.
- **Consequence for the headline:** the calibration asymmetry means the reported FoM3 gap
  (CNN 3326/3304 vs L1+product 2875) **under**-states the CNN's lead — perfect calibration would
  *tighten* the (conservative) CNN and *loosen* the (over-confident) L1. So "CNN ≥ L1" is conservative
  as stated. The CNN's FoM3 is effectively a lower bound.
- All three arms PASS GATE C. The CNN's mild conservatism is optional to "fix": a gentle sharpen
  (better-converged / higher-capacity RealNVP) is the right-direction lever and, now that no tercile
  under-covers, is not contraindicated — but it is unnecessary for the paper, since conservative is the
  safe miscalibration and only helps the comparison. Reducing NDE-seed pooling is NOT a lever (pooling
  the near-identical flows slightly *lowers* the net, not raises it).

## Artifacts (this dir)
`figs/tarp_cnn_l1_jointl1_unstratified_1sigma.*` (3-way overlay), `tarp_cnn_vs_l1_unstratified_1sigma.*`
(2-way), `tarp_resnet18_unstratified_1sigma.*` / `tarp_l1product_unstratified_1sigma.*` (per-arm),
`tarp_resnet18_stratified.*` (proper per-tercile bands), `sbc_resnet18_correct.*`. Recompute scripts:
`recompute_tarp_compare.py`, `recompute_tarp_3way.py`, `recompute_stratified_cnn.py`,
`plot_correct_calibration.py`. The paper panel (`nde_sweep_2026_06_13/figs/tarp_cnn_vs_l1_calibrated`,
new `…/tarp_cnn_l1_jointl1_calibrated`) and the in-place gate-C TARP fig were refreshed to these.
