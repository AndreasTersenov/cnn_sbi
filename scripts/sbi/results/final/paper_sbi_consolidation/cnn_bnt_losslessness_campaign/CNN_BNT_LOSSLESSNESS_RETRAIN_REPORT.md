# CNN-BNT losslessness retrain report

## Goal

Reproduce and improve CNN+VMIM BNT results so BNT and no-BNT contours are as similar as possible, while **keeping the existing final-paper baseline run unchanged**.

## What was kept

- Existing baseline (degraded) run was preserved:
  - `scripts/sbi/results/final/paper_sbi_consolidation/bnt_comparison_tomo4/`
- Baseline reference metrics are recorded in:
  - `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign/baseline_metrics.json`

## Retrain campaign

Runner:
- `scripts/sbi/run_cnn_bnt_losslessness_campaign.py`

Legacy CNN script was extended to restore architecture controls used in prior optimization:
- `scripts/sbi/npe_cnn_nbody_tomo.py`
  - `--compressor-conv-channels`
  - `--compressor-dense-width`
  - `--compressor-pool-window`
  - `--compressor-pool-stride`

All optimized runs retrained **separate BNT/noBNT compressors** with matched settings per configuration.

## Results (FoM3 on Omega_m, sigma_8, w_0)

| Setup | Seeds | Compression / flow setup | std inflation (BNT/noBNT) | FoM ratio (BNT/noBNT) | noBNT FoM3 mean | BNT FoM3 mean |
|---|---|---|---:|---:|---:|---:|
| baseline_final_paper | 41,42,43 | default compressor (`32,64,128`, dense 64), 20k compressor steps, standardize ON, flow 5k | 1.8049 | 0.0948 | 221,867.5 | 21,030.4 |
| stagej_repro | 41,42,43,44,45 | `64,128,256`, dense 128, 60k compressor steps, standardize OFF, flow 5k | 1.0392 | 0.7935 | 470,994.3 | 373,741.0 |
| advanced_arch64_dense256_nostd (cdim=6) | 41,42,43,44,45 | `64,128,256`, dense 256, **summary dim 6**, 80k compressor steps, standardize OFF, flow 8k | 1.0452 | 0.8923 | 504,594.1 | 450,251.7 |
| advanced_arch64_dense256_nostd (cdim=8) | 41,42,43,44,45 | `64,128,256`, dense 256, **summary dim 8**, 80k compressor steps, standardize OFF, flow 8k | 1.0423 | 0.7892 | 544,473.3 | 429,711.8 |
| advanced_arch64_dense256_nostd (**cdim=10, current best**) | 41,42,43,44,45 | `64,128,256`, dense 256, **summary dim 10**, 80k compressor steps, standardize OFF, flow 8k | **1.0297** | **0.9065** | **513,673.2** | **465,636.2** |
| NDE capacity test (L10/H320, flow12k) | 41,42,43,44,45 | fixed best cdim=6 compressors, larger NDE (`layers=10`, `hidden=320`, `flow=12k`) | 1.0516 | 0.7437 | 490,416.6 | 364,743.2 |

## Sigma8-focused diagnosis

| Setup | sigma8 std ratio (BNT/noBNT, combined) | sigma8 std ratio (seed mean) | Omega_m-sigma8 area ratio (combined) |
|---|---:|---:|---:|
| advanced_arch64_dense256_nostd (cdim=6) | 1.3428 | 1.1691 | 1.4621 |
| advanced_arch64_dense256_nostd (cdim=8) | 1.3014 | 1.1924 | 1.4388 |
| advanced_arch64_dense256_nostd (**cdim=10**) | **1.2647** | **1.1684** | 1.4506 |
| NDE capacity test (L10/H320, flow12k) | 1.4586 | 1.2252 | 1.6035 |

## Interpretation

1. Keeping the previous baseline-style compression clearly degrades BNT constraints (large inflation, large FoM drop).
2. Reproducing the prior Stage-J strategy recovers near-lossless contour widths.
3. Increasing only NDE capacity does **not** solve the residual sigma8 broadening; it worsens both FoM retention and sigma8 agreement.
4. Compression-side refinement is the effective lever here: with the same architecture/training recipe, moving from cdim=6 to **cdim=10** improves both global BNT/noBNT agreement and sigma8-specific agreement.
5. Current best result is `advanced_arch64_dense256_nostd` with **summary dim 10**, giving near-lossless width behavior (inflation ~1.03) and the strongest FoM retention among tested refined variants.

## Multipatch expansion update (nonoverlap48)

The multipatch campaign (deterministic non-overlapping 48 projections per sphere) is complete, including additional long-training and higher-capacity variants.

| Setup | std inflation (BNT/noBNT) | FoM ratio (BNT/noBNT) | sigma8 std ratio (BNT/noBNT, combined) |
|---|---:|---:|---:|
| old_best_cdim10_random25 (reference) | **1.0297** | **0.9065** | 1.2647 |
| multipatch_stagej_cdim6 | 1.0335 | 0.8476 | 1.4288 |
| multipatch_advanced_cdim10 | 1.0409 | 0.8433 | 1.3549 |
| multipatch_advanced_cdim12 | 1.0280 | 0.7481 | 1.2129 |
| multipatch_advanced_cdim10_long120k | 1.0771 | 0.7494 | **1.1703** |
| multipatch_advanced96_cdim10 | 1.0528 | 0.7783 | 1.2304 |

Interpretation of multipatch results:

1. Multipatch did **not** improve global BNT/noBNT agreement versus the current best random25 cdim10 reference (all multipatch FoM ratios are lower).
2. Longer and/or larger-capacity multipatch training does reduce sigma8 broadening, but this comes with a clear global FoM retention penalty.
3. For paper baseline conclusions, the best retained setup remains `advanced_arch64_dense256_nostd` with cdim=10 on the prior random25 dataset; multipatch is kept as a systematic negative-result diagnostic.

## Independent compressor/NDE split update (multipatch, disjoint train subsets)

Split policy used for all runs in this block:

- compressor train split: `train[:70%]`
- NDE train split: `train[70%:]`
- shared validation split: `test`
- TFDS: `NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48`

| Setup | std inflation (BNT/noBNT) | FoM ratio (BNT/noBNT) | sigma8 std ratio (BNT/noBNT, combined) |
|---|---:|---:|---:|
| old_best_cdim10_random25 (reference) | **1.0297** | **0.9065** | 1.2647 |
| multipatch_advanced_cdim10 (no split) | 1.0409 | 0.8433 | 1.3549 |
| indep_split_advanced_cdim10_long120k | 1.0369 | 0.8462 | **1.0966** |
| indep_split_stagej_cdim6 | 1.0398 | 0.8129 | 1.3046 |
| indep_split_advanced_cdim12 | **1.0169** | 0.7943 | 1.3046 |
| indep_split_advanced_cdim10 | 1.0520 | 0.7699 | 1.4071 |

Interpretation of independent-split results:

1. Split-only runs at standard budget (`indep_split_stagej_cdim6`, `indep_split_advanced_cdim10`, `indep_split_advanced_cdim12`) do **not** outperform the non-split multipatch cdim10 reference in global FoM retention.
2. With increased training budget (`indep_split_advanced_cdim10_long120k`), split training becomes slightly better than non-split multipatch cdim10 on global score (FoM ratio 0.8462 vs 0.8433, with similar inflation) and substantially better on sigma8 broadening (1.0966 vs 1.3549).
3. Best width inflation within split runs is still cdim12 (~1.0169), but best split tradeoff is long120k cdim10.
4. Overall, independent split alone is not a silver bullet, but with enough training it can recover most of the multipatch gap; the random25 best reference remains strongest on global retention (0.9065).

### Split-independence audit (what is and is not independent)

For `train[:70%]` vs `train[70%:]` on `grid_20deg_160px_nonoverlap48`:

- compressor-train examples: **211,445**
- NDE-train examples: **90,619**
- overlap in TFDS record IDs: **0** (example-level disjoint)
- overlap in unique `theta` values: **899 / 899** (all cosmologies appear in both subsets)

So this split is disjoint at **example/patch level**, but **not** disjoint at cosmology/simulation-parameter level.

### No-BNT comparison: non-split vs split overlays

Combined-seed no-BNT overlay (non-split vs split cdim10 variants):

- `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign/figures/overlay_nobnt_nonsplit_vs_split_cdim10_combined.png`
- `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign/figures/overlay_nobnt_nonsplit_vs_split_cdim10_long120k_combined.png`
- `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign/figures/overlay_nobnt_nonsplit_vs_split_cdim10_vs_split_cdim10_long120k_combined.png`

Quantitative no-BNT summary used for that overlay:

- `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign/figures/nobnt_split_vs_nonsplit_summary.json`

## Key artifacts

- Original campaign summary:
  - `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign/comparison_summary.csv`
  - `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign/comparison_summary.json`
- Refinement summary (includes cdim8/cdim10 and NDE test):
  - `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign/comparison_refinement_summary.csv`
  - `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign/comparison_refinement_summary.json`
- Sigma8 diagnostics (updated):
  - `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign/sigma8_diagnosis.json`
- Multipatch comparison/diagnostics (updated with long120k + advanced96):
  - `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign/comparison_multipatch_summary.csv`
  - `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign/comparison_multipatch_summary.json`
  - `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign/sigma8_multipatch_diagnosis.json`
- Independent-split comparison/diagnostics:
  - `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign/comparison_indep_split_summary.csv`
  - `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign/comparison_indep_split_summary.json`
  - `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign/sigma8_indep_split_diagnosis.json`
  - `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign/split_independence_audit.json`
- Full run manifests/results:
  - `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign/campaign_manifest.json`
  - `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign/job_results.json`
- Combined contour overlays:
  - baseline: `.../figures/overlay_baseline_finalpaper_combined_bnt_vs_nobnt.png`
  - stagej: `.../stagej_repro/figures/overlay_stagej_repro_combined_bnt_vs_nobnt.png`
  - advanced (cdim=6): `.../advanced_arch64_dense256_nostd/figures/overlay_advanced_arch64_dense256_nostd_combined_bnt_vs_nobnt.png`
  - advanced (cdim=8): `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign_cdim8/advanced_arch64_dense256_nostd/figures/overlay_advanced_arch64_dense256_nostd_combined_bnt_vs_nobnt.png`
  - advanced (**cdim=10, best**): `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign_cdim10/advanced_arch64_dense256_nostd/figures/overlay_advanced_arch64_dense256_nostd_combined_bnt_vs_nobnt.png`
  - NDE-capacity diagnostic: `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign/nde_capacity_l10h320_flow12k/figures/overlay_ndeL10H320_flow12k_combined_bnt_vs_nobnt.png`
  - multipatch long120k: `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign_multipatch_advanced_cdim10_long120k_v1/advanced_arch64_dense256_nostd_long/figures/overlay_advanced_arch64_dense256_nostd_long_combined_bnt_vs_nobnt.png`
  - multipatch advanced96: `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign_multipatch_advanced96_cdim10_v1/advanced_arch96_nostd/figures/overlay_advanced_arch96_nostd_combined_bnt_vs_nobnt.png`
