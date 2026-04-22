# Baryonified-observation bias study (tomo4): scientific conclusions

## Scope

We quantified inference bias from **unmodelled baryonic effects** by:

- training/checkpoints fixed to **no-bary** simulations,
- replacing only the observed fiducial map with baryonified realizations (`perm_0000..0019`),
- comparing `CNN`, `L1 (jaxili, no PCA)`, and `L1-VMIM`.

Configuration:

- Variant: `tomo4_20deg160`
- Dataset: `NbodyCosmogridDatasetTomo/grid_20deg_160px`
- Seeds: `41,42,43`
- Baryonified observations: `20 perms`
- Total baryonified runs: `180` (`3 methods × 3 seeds × 20 perms`)
- L1 extraction: `5` scales, SNR `[-13,13]`, `40` bins/scale

## Run integrity

From `scripts/sbi/baryon_bias_tomo4_study/baryon_bias_analysis.json`:

- `present_runs = 180`, `expected_runs = 180`
- `missing_file_count = 0`
- all posteriors have `dim = 6`

So the study matrix is complete and internally consistent.

## Bias metrics used

Primary constrained subspace: `(\Omega_m, \sigma_8, w_0)`.

Per run:

- `FoM3 = 1/sqrt(det(C3))`
- `FoM3 ratio = FoM3_bary / FoM3_nobary`
- `D_truth = sqrt((mu_bary-truth)^T C_nobary^{-1} (mu_bary-truth))`
- `D_shift = sqrt((mu_bary-mu_nobary)^T C_nobary^{-1} (mu_bary-mu_nobary))`
- parameter shifts normalized by no-bary sigmas

## Quantitative summary (means over 60 runs/method)

| Method | FoM3 ratio mean | `D_truth` mean | `D_shift` mean | mean `|ΔΩ_m|/σ` | mean `|Δσ_8|/σ` | mean `|Δw_0|/σ` |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 0.929 | 2.224 | 1.906 | 1.370 | 0.755 | 1.378 |
| L1 (jaxili) | 1.329 | 1.446 | 1.709 | 0.638 | 0.821 | 0.584 |
| L1-VMIM | 1.463 | 1.824 | 1.933 | 0.482 | 0.228 | 1.158 |

Additional robustness snapshots:

- Fraction with `D_shift > 2`:
  - CNN: `0.467`
  - L1: `0.283`
  - L1-VMIM: `0.333`
- Median signed normalized shifts `(Ω_m, σ_8, w_0)`:
  - CNN: `(-1.446, +0.173, -1.421)`
  - L1: `(-0.324, -0.389, -0.410)`
  - L1-VMIM: `(-0.483, -0.244, -1.296)`

## Scientific interpretation

1. **All three summaries are baryon-sensitive** in this mismatch setup (`D_shift` means ~`1.7–1.9`).

2. **L1 (jaxili)** has the smallest average absolute shifts in `w_0` and generally the lowest `D_truth`, i.e. best truth-proximity among the three in this test.

3. **CNN** shows the strongest coherent shift in `Ω_m` and `w_0` (median around `-1.4σ` in both), with the largest fraction of strongly shifted cases (`D_shift > 2`).

4. **L1-VMIM** is intermediate: smaller `Ω_m`/`σ_8` shifts than CNN, but still substantial `w_0` shift.

5. FoM behavior differs by method:
   - CNN baryonified contours are typically broader (`FoM ratio < 1`),
   - L1 and L1-VMIM are often tighter (`FoM ratio > 1`) despite posterior shifts.
   This means contour area alone is not a reliable robustness indicator; shift metrics are essential.

## Practical conclusion for current pipeline

- For this baryon-mismatch test, **no method is bias-free**.
- If prioritizing **truth proximity / smaller net shift**, **L1 (jaxili, no PCA)** is currently the best among tested options.
- If using CNN or L1-VMIM, baryon-mitigation/calibration steps are recommended before scientific interpretation of absolute parameter values.

## Key artifacts

- Study outputs:
  - `scripts/sbi/baryon_bias_tomo4_study/posteriors/*.npy`
- Analysis:
  - `scripts/sbi/baryon_bias_tomo4_study/baryon_bias_analysis.json`
  - `scripts/sbi/baryon_bias_tomo4_study/baryon_bias_per_run.csv`
  - `scripts/sbi/baryon_bias_tomo4_study/baryon_bias_summary.csv`
- Overlays:
  - `scripts/sbi/baryon_bias_tomo4_study/overlays/overlay_*`

