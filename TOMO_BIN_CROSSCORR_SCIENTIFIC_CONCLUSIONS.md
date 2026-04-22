# No-BNT tomography-bin cross-correlation study: scientific conclusions

## Objective

Quantify how much of the CNN advantage over wavelet L1-based summaries comes from implicitly using **cross-bin tomographic correlations** (full tomography) rather than only single-bin information.

## Validated setup

- Regime: **no-BNT** only.
- Dataset: `NbodyCosmogridDatasetTomo/grid_20deg_160px` (`20deg`, `160px`).
- Variants: `bin1`, `bin2`, `bin3`, `bin4`, `tomo4`.
- Methods: `cnn`, `l1`, `l1vmim`.
- Seeds: `41, 42, 43` (independent flow seeds).
- L1 extraction: `5` scales, SNR `[-13, 13]`, `40` bins/scale.
- FoM: `FoM3 = 1/sqrt(det(C3))`, with `C3` from `(\Omega_m, \sigma_8, w_0)` only.

## FoM3 results (mean ± std over seeds)

| Method | bin1 | bin2 | bin3 | bin4 | tomo4 |
|---|---:|---:|---:|---:|---:|
| CNN | 6907.860 ± 110.130 | 19050.623 ± 182.809 | 45248.332 ± 1035.924 | 63488.327 ± 1245.356 | 447603.315 ± 8908.376 |
| L1 | 190.429 ± 23.349 | 451.180 ± 24.849 | 689.543 ± 89.632 | 891.772 ± 97.905 | 2288.207 ± 681.566 |
| L1+VMIM | 231.592 ± 6.562 | 1235.021 ± 458.918 | 4140.103 ± 116.519 | 5438.158 ± 106.892 | 10136.290 ± 338.844 |

## Cross-correlation attribution metrics

Definitions:

- `R_full(X) = FoM3_CNN(tomo4) / FoM3_X(tomo4)`
- `R_bin_avg(X) = mean_b [FoM3_CNN(bin b) / FoM3_X(bin b)]`
- `G_corr(X) = R_full(X) / R_bin_avg(X)`

| Comparator `X` | `R_full(X)` | `R_bin_avg(X)` | `G_corr(X)` |
|---|---:|---:|---:|
| L1 | 195.613 | 53.828 | 3.634 |
| L1+VMIM | 44.158 | 16.964 | 2.603 |

Seed robustness of `G_corr`:

- vs L1: `3.957`, `4.703`, `2.746` (mean `3.802 ± 0.988`)
- vs L1+VMIM: `2.176`, `2.764`, `2.732` (mean `2.557 ± 0.330`)

## Scientific interpretation

1. CNN already outperforms L1-type summaries at single-bin level, but this does **not** explain the full tomographic gap.
2. The full-tomography CNN/L1 advantage (`195.6x`) is much larger than the single-bin averaged advantage (`53.8x`), implying a strong additional gain from cross-bin structure (`G_corr = 3.63`).
3. The same behavior persists against L1+VMIM (`44.2x` full vs `17.0x` bin-average; `G_corr = 2.60`).
4. Therefore, in this no-BNT setup, a substantial fraction of CNN’s final tomographic advantage is consistent with superior implicit treatment of tomographic cross-correlations, not only with better per-bin compression.

## Direct answer to the study question

The data support that CNN’s tomographic edge over L1 methods is **strongly cross-correlation-driven**.  
Relative to its own single-bin advantage baseline, CNN gains an additional factor of:

- about **3.6x** versus raw L1,
- about **2.6x** versus L1+VMIM,

when moving to full `tomo4`.

## Reproducibility anchors

- Manifest/config: `scripts/sbi/nobnt_tomo_bins_crosscorr_study/manifest.json`
- Run integrity: `scripts/sbi/nobnt_tomo_bins_crosscorr_study/job_results.json`
- FoM analysis: `scripts/sbi/nobnt_tomo_bins_crosscorr_study/fom3_analysis.json`
- Aggregated table: `scripts/sbi/nobnt_tomo_bins_crosscorr_study/fom3_summary.csv`
- Per-seed table: `scripts/sbi/nobnt_tomo_bins_crosscorr_study/fom3_per_run.csv`
- Overlays: `scripts/sbi/nobnt_tomo_bins_crosscorr_study/overlays/`
