# BNT tomography study (tomo4): final scientific conclusions

## Objective

Assess how applying BNT **after** shape-noise injection changes posterior constraints for three information-extraction strategies:

- CNN-compressed SBI,
- wavelet L1 (no compression),
- wavelet L1 + VMIM compression.

## Validated experimental basis

- Dataset: `NbodyCosmogridDatasetTomo/grid_20deg_160px` (`20deg`, `160px`, bins `1,2,3,4`)
- BNT matrix: `tomo4_bnt_v1` (`scripts/sbi/bnt_utils.py`)
- Operation order in all tested pipelines: **project/load maps -> add noise -> apply BNT (if enabled) -> summary/compression -> NPE**
- L1 extraction settings: `5` wavelet scales, SNR `[-13, 13]`, `40` bins/scale
- Posterior sampling: `100000` samples per run
- Primary comparisons use matched seed sets (`41, 42, 43`); CNN best-case also confirmed on `41..45`

## Final quantitative results

### Core cross-method comparison (3-seed means)

| Method | `std_sum` no-BNT | `std_sum` BNT | Inflation (`BNT/no-BNT`) | `bias_l2` no-BNT | `bias_l2` BNT |
|---|---:|---:|---:|---:|---:|
| CNN (initial configuration) | 0.2189 | 0.3945 | 1.8024 | 0.0690 | 0.0862 |
| L1 (no compression) | 0.3950 | 0.6329 | 1.6024 | 0.0763 | 0.1937 |
| L1 + VMIM (baseline) | 0.3905 | 0.6167 | 1.5793 | 0.0915 | 0.2145 |

### Optimized method outcomes

- **CNN (optimized, Stage-J no-standardize family)**  
  - inflation: **1.0363** (3 seeds), **1.0391** (5 seeds)
- **L1 + VMIM (focused optimized config `cdim64_h768_nf10x384`)**  
  - no-BNT `std_sum`: 0.3791  
  - BNT `std_sum`: 0.5241  
  - inflation: **1.3825**  
  - improvement vs L1+VMIM baseline inflation: **-0.1968** (absolute), BNT width **-15.0%**

## Scientific interpretation

1. **BNT does not inherently destroy information** (linear, invertible transform), but practical performance is summary-limited.
2. **Compressor quality is decisive**: CNN shifts from strongly inflated (1.80) to near-lossless (~1.04) after targeted optimization.
3. **L1-type summaries remain BNT-sensitive** in this setup: raw L1 is strongly inflated (~1.60), and optimized L1+VMIM improves but remains substantially inflated (~1.38).
4. **Cross-bin/correlation modeling is central**: results are consistent with CNN learning BNT-induced tomographic correlation structure more effectively than histogram-based L1 summaries.

## Final project-level conclusion

For BNT-space tomography in this repository, the most robust validated solution is the **optimized CNN pipeline**.  
L1 and L1+VMIM remain useful diagnostics, but they deliver broader and more BNT-sensitive posteriors than optimized CNN under the same experimental protocol.

## Reproducible result anchors

- Core metrics: `scripts/sbi/bnt_tomo4_study/bnt_metrics_summary.json`
- CNN optimized (3 seeds): `scripts/sbi/bnt_tomo4_study/round1_stageJ_postproc/quick_metrics_seed41_42_43_nostd.json`
- CNN optimized (5 seeds): `scripts/sbi/bnt_tomo4_study/round1_stageJ_postproc/quick_metrics_seed41_45_nostd.json`
- L1+VMIM optimized summary: `scripts/sbi/bnt_tomo4_study/l1vmim_opt_round2/best_config_3seed_metrics.json`
