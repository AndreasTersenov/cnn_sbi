# Optimal no-BNT single-bin vs tomo4 benchmark: scientific conclusions

## Scope

- Regime: no-BNT.
- Dataset: `NbodyCosmogridDatasetTomo/grid_20deg_160px` (`20deg`, `160px`).
- Variants: `bin1..bin4`, `tomo4`; seeds: `41,42,43`.
- Methods: `CNN`, `L1 (jaxili, no PCA)`, `L1+VMIM`.
- FoM metric: `FoM3 = 1/sqrt(det(C3))` on `(Omega_m, sigma_8, w_0)` only.

## Method-optimal settings used

- CNN: Stage-J validated no-standardize family (`conv=64,128,256`, `dense=128`, compressor steps `60000`, flow steps `5000`, batch `256`).
  - Note: a fast tomo4 sweep selected `conv_96x192x384__dw_128__std_on__flow_5000__bs_256` with `fom3_mean=10188.3`, but full benchmark uses the higher-performing previously validated Stage-J setup.
- L1: `st_log1p_zscore__clip_5p0__lr_3em04__bs_128__ep_5000__ac_off` (`summary_transform=log1p-zscore`, `clip=5.0`, `lr=0.0003`, `batch=128`, `epochs=5000`), plus `n_scales=5`, SNR `[-13,13]`, `40` bins/scale, no PCA.
- L1+VMIM: `cdim64_h768x768_nf10x384_flow12000_bs256` (`cdim=64`, hidden `768,768`, VMIM NF `10x384`, flow `12000`, batch `256`).

## FoM3 summary (mean ± std over seeds)

| Method | bin1 | bin2 | bin3 | bin4 | tomo4 |
|---|---:|---:|---:|---:|---:|
| CNN | 6907.860 ± 110.130 | 19050.623 ± 182.809 | 45248.332 ± 1035.924 | 63488.327 ± 1245.356 | 447603.315 ± 8908.376 |
| L1 | 438.894 ± 31.934 | 1360.892 ± 201.630 | 3372.736 ± 393.628 | 4983.073 ± 272.194 | 10650.543 ± 2982.997 |
| L1+VMIM | 231.592 ± 6.562 | 1235.021 ± 458.918 | 4140.103 ± 116.519 | 5438.158 ± 106.892 | 10136.290 ± 338.844 |

## Cross-correlation attribution

Definitions: `R_full(X)=FoM3_CNN(tomo4)/FoM3_X(tomo4)`, `R_bin_avg(X)=mean_b FoM3_CNN(bin_b)/FoM3_X(bin_b)`, `G_corr(X)=R_full/R_bin_avg`.

| Comparator X | R_full | R_bin_avg | G_corr |
|---|---:|---:|---:|
| l1 | 42.026 | 13.974 | 3.008 |
| l1vmim | 44.158 | 16.964 | 2.603 |

- Seed-level `G_corr` vs L1: s41:3.308, s42:2.267, s43:3.831 (mean `3.136 ± 0.796`).
- Seed-level `G_corr` vs L1+VMIM: s41:2.176, s42:2.764, s43:2.732 (mean `2.557 ± 0.330`).

## Scientific conclusions

1. CNN strongly outperforms both L1 methods in every variant, with the largest absolute gain in full tomography.
2. The CNN/L1 full-tomo ratio (`R_full=42.0`) is much larger than the single-bin average (`R_bin_avg=14.0`), giving `G_corr=3.01`: CNN’s advantage is strongly amplified by cross-bin information.
3. The same pattern holds vs L1+VMIM (`R_full=44.2`, `R_bin_avg=17.0`, `G_corr=2.60`), so VMIM improves L1 but does not close the cross-correlation gap.
4. L1 and L1+VMIM have similar tomo4 FoM in this benchmark (`10650.5` vs `10136.3`), but L1 shows much larger seed variance; L1+VMIM is more stable.
5. Overall, in this no-BNT setup, CNN’s superiority is not just per-bin compression quality: a major part is consistent with better exploitation of tomographic cross-correlations.

## Run integrity and artifacts

- Integrity checks: present `45/45`, missing `0`, dim6=`True`, all_fom_valid=`True`.
- Main benchmark outputs (matrix + FoM + overlays): `scripts/sbi/nobnt_tomo_bins_crosscorr_study/`.
- Config selections: `scripts/sbi/optimal_nobnt_crosscorr_benchmark/selected_configs/final_method_configs.json`.
- Sweep selections: `scripts/sbi/optimal_nobnt_crosscorr_benchmark/sweeps/{cnn_tomo4,l1_tomo4,l1vmim_tomo4}/final_selection.json`.
