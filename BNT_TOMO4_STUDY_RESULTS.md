# BNT Tomo4 Study Results

## Abstract
This report summarizes the tomographic 4-bin BNT study comparing `nobnt` and `bnt` conditions for CNN, L1, and L1-VMIM SBI pipelines. Across all three methods, the BNT condition increases posterior width (`std_sum`) relative to `nobnt`, with inflation factors above 1.57. In this run, `bias_l2` is also higher under BNT for all methods.

## Setup
- Dataset: `NbodyCosmogridDatasetTomo/grid_20deg_160px`
- Field: `20 deg`, `160 px`
- Tomographic bins: `1,2,3,4`
- Conditions: `nobnt, bnt`
- Seeds: `41, 42, 43`
- GPUs used: `0, 1, 2`
- Flow steps (CNN/L1): `5000`
- Flow steps (L1-VMIM): `12000`
- Compressor steps (CNN): `20000`
- Compressor steps (L1-VMIM): `12000`
- Posterior samples per run: `100000`

## Pipeline details (explicit noise->BNT order)
1. For each method (CNN, L1, L1-VMIM), maps are first built in tomographic-bin space and shape noise is injected.
2. The BNT transform is then conditionally applied **after** noise injection when `--apply-bnt` is enabled (`bnt` condition).
3. The `nobnt` condition uses the same pipeline without the BNT transform.
4. Flow training and posterior sampling are then run with matched seeds and sample count.

## Results table
Values are means over seeds `41,42,43` from `bnt_metrics_summary.json`.

| method | nobnt std_sum | bnt std_sum | inflation | nobnt bias_l2 | bnt bias_l2 |
|---|---:|---:|---:|---:|---:|
| cnn | 0.21888271470864615 | 0.39450956384340924 | 1.8023787961901845 | 0.06898132382796522 | 0.08615575197652808 |
| l1 | 0.39498425523440045 | 0.6329077879587809 | 1.602362067782136 | 0.07630503670620892 | 0.1937264428181941 |
| l1vmim | 0.3904571533203125 | 0.6166538794835409 | 1.5793125423358996 | 0.09151079430593062 | 0.2144857202642018 |

## Interpretation
- `std_sum` increases under BNT for all methods, indicating broader posteriors in this configuration.
- Inflation is largest for CNN (`1.8023787961901845`) and remains substantial for L1 (`1.602362067782136`) and L1-VMIM (`1.5793125423358996`).
- `bias_l2` also increases for all three methods under BNT in this study run.

## Caveats
- Results are based on three seeds (`41,42,43`) and one dataset/map configuration.
- Reported metrics (`std_sum`, `bias_l2`) are compact summaries; they do not replace full posterior diagnostics.
- Conclusions are specific to the BNT matrix/version and training setup used in this run.

## Reproducibility (paths)
- Repo root: `/mnt/home/tersenov/software/cnn_sbi/.worktrees/bnt_tomo_study`
- Metrics summary: `/mnt/home/tersenov/software/cnn_sbi/.worktrees/bnt_tomo_study/scripts/sbi/bnt_tomo4_study/bnt_metrics_summary.json`
- Posterior summary: `/mnt/home/tersenov/software/cnn_sbi/.worktrees/bnt_tomo_study/scripts/sbi/bnt_tomo4_study/posterior_summary.json`
- Manifest: `/mnt/home/tersenov/software/cnn_sbi/.worktrees/bnt_tomo_study/scripts/sbi/bnt_tomo4_study/manifest.json`
- Orchestration script: `/mnt/home/tersenov/software/cnn_sbi/.worktrees/bnt_tomo_study/scripts/sbi/run_bnt_tomo4_study.py`
- Method scripts (noise->BNT order):
  - `/mnt/home/tersenov/software/cnn_sbi/.worktrees/bnt_tomo_study/scripts/sbi/npe_cnn_nbody_tomo.py`
  - `/mnt/home/tersenov/software/cnn_sbi/.worktrees/bnt_tomo_study/scripts/sbi/npe_l1norm_nbody_tomo.py`
  - `/mnt/home/tersenov/software/cnn_sbi/.worktrees/bnt_tomo_study/scripts/sbi/npe_l1vmim_nbody_tomo.py`
