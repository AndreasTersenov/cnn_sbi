# Final Scientific Report: Paper SBI Consolidation

## Scope
This report summarizes generated artifacts under:
- `nobnt_final_matrix` (full no-BNT matrix across methods, bins, seeds)
- `bnt_comparison_tomo4` (tomo4 BNT vs no-BNT comparison)
- `baryonified_appendix` (tomo4 baryon-bias appendix permutations)

No additional inference reruns were performed during this QC/report step.

## QC Status
### 1) Job completion (`job_results.json`)
All three run roots have only zero return codes:
- `nobnt_final_matrix`: 55/55 returncodes are `0`
- `bnt_comparison_tomo4`: 22/22 returncodes are `0`
- `baryonified_appendix`: 180/180 returncodes are `0`

### 2) Posterior-summary sanity (`posterior_summary.json`)
No pathological empty/non-finite indicators were found:
- `nobnt_final_matrix`: 45 entries, no empty entries, no non-finite numeric values
- `bnt_comparison_tomo4`: 18 entries, no empty entries, no non-finite numeric values
- `baryonified_appendix`: 180 entries, no empty entries, no non-finite numeric values

### 3) Key analysis artifacts present
Expected comparison overlays and FoM tables are present (17/17 checked), including:
- `analysis/nobnt_fom/fom3_summary.csv`, `analysis/nobnt_fom/fom3_analysis.json`
- `analysis/baryon_bias_fom/baryon_bias_summary.csv`, `analysis/baryon_bias_fom/baryon_bias_analysis.json`
- Combined overlays for no-BNT tomo/bin comparisons, BNT-vs-noBNT, and baryon-bias comparisons.

**QC verdict: PASS**

## Key quantitative takeaways (FoM)
FoM values below are `fom3_mean` from `analysis/nobnt_fom/fom3_summary.csv`.

| Method | bin1 | bin4 | tomo4 | tomo4 / best single-bin |
|---|---:|---:|---:|---:|
| CNN | 6,001.0 | 52,767.9 | 387,474.6 | 7.34x |
| L1 | 568.1 | 4,641.3 | 9,651.4 | 2.08x |
| L1VMIM | 410.0 | 4,686.0 | 10,650.9 | 2.27x |

CNN remains substantially tighter than L1/L1VMIM in both single-bin and tomographic settings.

## Cross-correlation effect (single-bin vs tomo)
Using the no-BNT matrix:
- Tomographic (`tomo4`) FoM exceeds the **mean** single-bin FoM by:
  - CNN: **13.73x**
  - L1: **3.57x**
  - L1VMIM: **4.30x**
- Attribution in `fom3_analysis.json` reports additional cross-correlation gain factors (`g_corr`) for CNN relative to compressed baselines:
  - vs L1: **3.85**
  - vs L1VMIM: **2.92**

## BNT impact summary (tomo4)
From `bnt_comparison_tomo4/posteriors/*.fom.json` (seed-averaged within-run noBNT controls):

| Method | noBNT mean FoM3 | BNT mean FoM3 | BNT/noBNT | % change |
|---|---:|---:|---:|---:|
| CNN | 221,867.5 | 21,030.4 | 0.0948 | -90.5% |
| L1 | 11,127.0 | 679.9 | 0.0611 | -93.9% |
| L1VMIM | 9,836.8 | 617.9 | 0.0628 | -93.7% |

In this tomo4 setup, BNT strongly reduces FoM3 for all three methods relative to the paired no-BNT controls.

## Baryon-bias appendix summary
From `analysis/baryon_bias_fom/baryon_bias_summary.csv` (`fom3_ratio_mean` is bary/no-bary):

| Method | FoM ratio mean ± std | d_truth mean | d_shift mean |
|---|---:|---:|---:|
| CNN | 1.0147 ± 0.0284 | 2.114 | 1.591 |
| L1 | 1.3142 ± 0.4210 | 1.439 | 1.518 |
| L1VMIM | 1.1696 ± 0.1924 | 2.314 | 2.324 |

Across 60 runs/method (3 seeds × 20 perms), baryonized variants remain finite and complete. Shift diagnostics indicate O(1–2+) Mahalanobis-scale displacements, with largest mean shift for L1VMIM.

## Caveats
- FoM is computed in the 3-parameter subspace (`Omega_m`, `sigma_8`, `w_0`) per analysis assumptions.
- BNT and baryon-bias comparisons are tomo4-focused; absolute FoM scales should be interpreted within each run’s paired design.
