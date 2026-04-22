# No-BNT tomography-bin cross-correlation study: detailed technical report

## 1. Scope and purpose

This study quantifies how much of CNN's no-BNT advantage over wavelet-L1-based pipelines comes from learning **cross-bin tomographic correlations** (full `tomo4`) rather than only single-bin information.

We compare three methods:

- `cnn` (CNN compressor + NPE),
- `l1` (wavelet L1 summary without learned compressor),
- `l1vmim` (wavelet L1 summary + VMIM compressor + NPE),

across five data variants:

- `bin1_20deg160`, `bin2_20deg160`, `bin3_20deg160`, `bin4_20deg160`, `tomo4_20deg160`.

The regime is **strictly no-BNT**.

## 2. Run matrix and integrity

### 2.1 Matrix definition

- Methods: `3` (`cnn`, `l1`, `l1vmim`)
- Variants: `5` (`bin1..4`, `tomo4`)
- Seeds: `3` (`41, 42, 43`)
- Expected posterior files: `3 x 5 x 3 = 45`

### 2.2 Integrity checks (all passed)

From `scripts/sbi/nobnt_tomo_bins_crosscorr_study/fom3_analysis.json`:

- `present_posterior_count = 45`
- `missing_posterior_count = 0`
- `all_expected_posteriors_present = true`
- `all_method_variant_groups_complete = true`
- `all_posteriors_have_dim6 = true`
- `valid_fom_count = 45`
- `all_fom_valid = true`

From `scripts/sbi/nobnt_tomo_bins_crosscorr_study/job_results.json`:

- Total jobs: `55`
- Non-zero return codes: `0`

## 3. Data, parameters, and metric

### 3.1 Dataset / map setup

- TFDS: `NbodyCosmogridDatasetTomo/grid_20deg_160px`
- Field size / pixels: `20 deg`, `160 px`
- Variants:
  - single-bin: `nbins=1`, `tomo-bin-indices={1,2,3,4}`,
  - tomography: `nbins=4`, `tomo-bin-indices=1,2,3,4`.

### 3.2 FoM definition

We evaluate only lensing-constrained parameters:

- `(\Omega_m, \sigma_8, w_0)` = posterior columns `[0,1,2]`.

For each posterior sample matrix `S`:

- `C3 = cov(S[:, :3])`,
- `FoM3 = 1 / sqrt(det(C3))`.

This is exactly the quantity produced by `scripts/sbi/analyze_nobnt_tomo_bins_fom.py`.

## 4. Exact method configurations

All values below are from:
`scripts/sbi/nobnt_tomo_bins_crosscorr_study/manifest.json`.

### 4.1 CNN pipeline

- Compressor:
  - dim `6`,
  - steps `60000`,
  - save-every `2000`,
  - conv channels `64,128,256`,
  - dense width `128`,
  - pool window/stride `16/8`.
- NPE:
  - flow steps `5000`,
  - batch size `256`,
  - NVP layers `4`,
  - hidden `128`,
  - weight decay `1e-4`,
  - grad clip `1.0`,
  - summary standardization `false`,
  - summary clip `0.0`.

### 4.2 L1 (no compression)

- Wavelet extraction:
  - scales `5`,
  - SNR range `[-13, 13]`,
  - bins per scale `40`.
- NPE:
  - flow steps `5000`,
  - batch size `256`,
  - NVP layers `4`,
  - hidden `128`,
  - weight decay `1e-4`,
  - grad clip `1.0`.

### 4.3 L1+VMIM

- Compressor:
  - conda env `jaxili`,
  - dim `64`,
  - hidden `768,768`,
  - VMIM NF layers/hidden `10/384`,
  - input clip `6.0`,
  - log1p + input standardize enabled,
  - compressor steps `12000`,
  - batch size `128`,
  - lr `3e-4`.
- NPE:
  - flow steps `12000`,
  - batch size `256`,
  - NVP layers `4`,
  - hidden `128`,
  - weight decay `1e-4`,
  - grad clip `1.0`.

## 5. FoM3 results

### 5.1 Per-method, per-variant means (seed-averaged)

| Method | Variant | FoM3 mean | FoM3 std | n_valid/n_total |
|---|---|---:|---:|---:|
| cnn | bin1_20deg160 | 6907.860 | 110.130 | 3/3 |
| cnn | bin2_20deg160 | 19050.623 | 182.809 | 3/3 |
| cnn | bin3_20deg160 | 45248.332 | 1035.924 | 3/3 |
| cnn | bin4_20deg160 | 63488.327 | 1245.356 | 3/3 |
| cnn | tomo4_20deg160 | 447603.315 | 8908.376 | 3/3 |
| l1 | bin1_20deg160 | 190.429 | 23.349 | 3/3 |
| l1 | bin2_20deg160 | 451.180 | 24.849 | 3/3 |
| l1 | bin3_20deg160 | 689.543 | 89.632 | 3/3 |
| l1 | bin4_20deg160 | 891.772 | 97.905 | 3/3 |
| l1 | tomo4_20deg160 | 2288.207 | 681.566 | 3/3 |
| l1vmim | bin1_20deg160 | 231.592 | 6.562 | 3/3 |
| l1vmim | bin2_20deg160 | 1235.021 | 458.918 | 3/3 |
| l1vmim | bin3_20deg160 | 4140.103 | 116.519 | 3/3 |
| l1vmim | bin4_20deg160 | 5438.158 | 106.892 | 3/3 |
| l1vmim | tomo4_20deg160 | 10136.290 | 338.844 | 3/3 |

### 5.2 Per-seed FoM3 table

| Method | Variant | Seed 41 | Seed 42 | Seed 43 | Mean | Std |
|---|---|---:|---:|---:|---:|---:|
| cnn | bin1_20deg160 | 6961.122 | 6981.233 | 6781.224 | 6907.860 | 110.130 |
| cnn | bin2_20deg160 | 18856.828 | 19075.053 | 19219.989 | 19050.623 | 182.809 |
| cnn | bin3_20deg160 | 45330.351 | 44173.836 | 46240.809 | 45248.332 | 1035.924 |
| cnn | bin4_20deg160 | 64335.586 | 64070.941 | 62058.454 | 63488.327 | 1245.356 |
| cnn | tomo4_20deg160 | 446954.938 | 456818.165 | 439036.842 | 447603.315 | 8908.376 |
| l1 | bin1_20deg160 | 166.654 | 191.305 | 213.326 | 190.429 | 23.349 |
| l1 | bin2_20deg160 | 460.908 | 422.940 | 469.694 | 451.180 | 24.849 |
| l1 | bin3_20deg160 | 586.653 | 731.284 | 750.692 | 689.543 | 89.632 |
| l1 | bin4_20deg160 | 828.504 | 1004.543 | 842.271 | 891.772 | 97.905 |
| l1 | tomo4_20deg160 | 1901.423 | 1888.023 | 3075.174 | 2288.207 | 681.566 |
| l1vmim | bin1_20deg160 | 229.232 | 239.007 | 226.536 | 231.592 | 6.562 |
| l1vmim | bin2_20deg160 | 711.835 | 1569.505 | 1423.722 | 1235.021 | 458.918 |
| l1vmim | bin3_20deg160 | 4133.801 | 4026.863 | 4259.645 | 4140.103 | 116.519 |
| l1vmim | bin4_20deg160 | 5485.607 | 5513.111 | 5315.756 | 5438.158 | 106.892 |
| l1vmim | tomo4_20deg160 | 10326.257 | 10337.533 | 9745.082 | 10136.290 | 338.844 |

## 6. Attribution analysis: cross-bin correlation contribution

### 6.1 Raw method ratios by variant

| Variant | CNN/L1 | CNN/L1+VMIM | L1+VMIM/L1 |
|---|---:|---:|---:|
| bin1_20deg160 | 36.275 | 29.828 | 1.216 |
| bin2_20deg160 | 42.224 | 15.425 | 2.737 |
| bin3_20deg160 | 65.621 | 10.929 | 6.004 |
| bin4_20deg160 | 71.193 | 11.675 | 6.098 |
| tomo4_20deg160 | 195.613 | 44.158 | 4.430 |

### 6.2 Cross-correlation gain factor

Using:

- `R_full(X) = FoM3_CNN(tomo4) / FoM3_X(tomo4)`
- `R_bin_avg(X) = mean_b [FoM3_CNN(bin b) / FoM3_X(bin b)]`
- `G_corr(X) = R_full(X) / R_bin_avg(X)`

| Comparator | `R_full` | `R_bin_avg` | `G_corr` |
|---|---:|---:|---:|
| L1 | 195.613 | 53.828 | 3.634 |
| L1+VMIM | 44.158 | 16.964 | 2.603 |

Seed-level robustness:

| Comparator | Seed | R_full | R_bin_avg | G_corr |
|---|---:|---:|---:|---:|
| l1 | 41 | 235.063 | 59.401 | 3.957 |
| l1 | 42 | 241.956 | 51.445 | 4.703 |
| l1 | 43 | 142.768 | 51.996 | 2.746 |
| l1 | mean±std | - | - | 3.802 ± 0.988 |
| l1vmim | 41 | 43.283 | 19.888 | 2.176 |
| l1vmim | 42 | 44.190 | 15.989 | 2.764 |
| l1vmim | 43 | 45.052 | 16.491 | 2.732 |
| l1vmim | mean±std | - | - | 2.557 ± 0.330 |

### 6.3 Tomography-vs-bin accumulation within each method

`FoM3(tomo4) / sum_b FoM3(bin b)`:

| Method | tomo4 / sum(bin1..4) per seed | Mean ± std |
|---|---|---:|
| cnn | 3.299, 3.401, 3.269 | 3.323 ± 0.069 |
| l1 | 0.931, 0.803, 1.351 | 1.028 ± 0.287 |
| l1vmim | 0.978, 0.911, 0.868 | 0.919 ± 0.055 |

Interpretation: CNN gains a strong super-additive tomographic improvement, while L1/L1+VMIM are approximately additive (or sub-additive) in this metric.

## 7. Compressor/evaluation diagnostics

From training logs (`scripts/sbi/nobnt_tomo_bins_crosscorr_study/logs/*train*.log`):

- All compressor jobs completed and checkpoints were found/used.
- CNN compressor training completed at `60000` steps for all variants.
- L1+VMIM compressor training completed at `12000` steps for all variants.
- Evaluation-stage logs show no tracebacks; all posterior outputs were written.

Note on warnings:

- Several logs report: best validation loss at final saved step (flow stage with `--total-steps 1` for compressor-training jobs). This is expected in the study orchestration pattern (compressor training + minimal flow pass) and does not indicate a failed run.

## 8. Figure inventory

Figures were written to:
`scripts/sbi/nobnt_tomo_bins_crosscorr_study/overlays/`

Total overlay figures: `40`

- Seed-wise overlays (`30`): 5 variants x 2 comparisons x 3 seeds
- Combined overlays (`10`): 5 variants x 2 comparisons

Combined figure files:

- `overlay_bin1_20deg160_cnn_vs_l1_combined_nobnt.png`
- `overlay_bin1_20deg160_cnn_vs_l1vmim_combined_nobnt.png`
- `overlay_bin2_20deg160_cnn_vs_l1_combined_nobnt.png`
- `overlay_bin2_20deg160_cnn_vs_l1vmim_combined_nobnt.png`
- `overlay_bin3_20deg160_cnn_vs_l1_combined_nobnt.png`
- `overlay_bin3_20deg160_cnn_vs_l1vmim_combined_nobnt.png`
- `overlay_bin4_20deg160_cnn_vs_l1_combined_nobnt.png`
- `overlay_bin4_20deg160_cnn_vs_l1vmim_combined_nobnt.png`
- `overlay_tomo4_20deg160_cnn_vs_l1_combined_nobnt.png`
- `overlay_tomo4_20deg160_cnn_vs_l1vmim_combined_nobnt.png`

## 9. Scientific conclusions from this study

1. CNN is substantially tighter than both L1 and L1+VMIM in every single-bin variant and in full tomography.
2. The CNN-vs-L1 gap expands dramatically in full tomography (`195.6x`) relative to the single-bin average (`53.8x`), yielding `G_corr = 3.63`.
3. The same pattern holds versus L1+VMIM (`44.2x` vs `17.0x`, `G_corr = 2.60`).
4. Therefore, CNN's no-BNT advantage is not only a per-bin compression effect; a major component is consistent with better exploitation of tomographic cross-bin correlations.

## 10. Reproducibility anchors

- Study manifest: `scripts/sbi/nobnt_tomo_bins_crosscorr_study/manifest.json`
- Job execution record: `scripts/sbi/nobnt_tomo_bins_crosscorr_study/job_results.json`
- FoM analysis JSON: `scripts/sbi/nobnt_tomo_bins_crosscorr_study/fom3_analysis.json`
- FoM per-run table: `scripts/sbi/nobnt_tomo_bins_crosscorr_study/fom3_per_run.csv`
- FoM summary table: `scripts/sbi/nobnt_tomo_bins_crosscorr_study/fom3_summary.csv`
- Posterior quick diagnostics: `scripts/sbi/nobnt_tomo_bins_crosscorr_study/posterior_summary.json`
- Overlay figures: `scripts/sbi/nobnt_tomo_bins_crosscorr_study/overlays/`
