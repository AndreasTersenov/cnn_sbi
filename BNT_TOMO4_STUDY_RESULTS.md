# BNT impact on tomographic SBI (CNN, L1, and L1+VMIM): final conclusions

## Scope

This report consolidates the full BNT study on `tomo4_20deg160` and the later CNN optimization round.  
The question was whether BNT (applied after shape-noise injection) inflates contours, and whether a sufficiently good compressor can remain near-lossless under BNT.

## Experimental setup

- Dataset: `NbodyCosmogridDatasetTomo/grid_20deg_160px`
- Field: `20 deg`, `160 px`
- Tomographic bins: `1,2,3,4`
- Seeds (core study): `41, 42, 43`
- Conditions: `nobnt` vs `bnt`
- L1 extraction: `5` wavelet scales, SNR `[-13, 13]`, `40` bins/scale
- Posterior samples/run: `100000`

### BNT definition and order of operations

- BNT matrix version: `tomo4_bnt_v1` (from `scripts/sbi/bnt_utils.py`)
- Matrix:
  - `[1.0, 0.0, 0.0, 0.0]`
  - `[-1.0, 1.0, 0.0, 0.0]`
  - `[0.4521097, -1.4521097, 1.0, 0.0]`
  - `[0.0, 0.25127807, -1.251278, 1.0]`
- In all pipelines: maps are projected, noise is injected, then BNT is applied if `--apply-bnt` is set.

## Run integrity and monitoring

- Core matrix completion: `18/18` expected posterior files were produced.
- L1 rerun recovery after NumPy/TFP compatibility fix: `6/6` rerun jobs succeeded (`rerun_l1_results.json`).
- Log scan over core `bnt_tomo4_study/logs/*.log`: no traceback/error signatures found.

## Core rigorous results (3-seed means)

From `scripts/sbi/bnt_tomo4_study/bnt_metrics_summary.json`:

| Method | `std_sum` no-BNT | `std_sum` BNT | Inflation (`BNT/no-BNT`) | `bias_l2` no-BNT | `bias_l2` BNT |
|---|---:|---:|---:|---:|---:|
| CNN (initial config) | 0.2189 | 0.3945 | 1.8024 | 0.0690 | 0.0862 |
| L1 (no compression) | 0.3950 | 0.6329 | 1.6024 | 0.0763 | 0.1937 |
| L1+VMIM | 0.3905 | 0.6167 | 1.5793 | 0.0915 | 0.2145 |

At this stage, all methods inflate under BNT, and CNN (with the initial compressor setup) inflates the most.

## CNN optimization outcome (fair BNT/no-BNT pairing)

A dedicated CNN optimization sweep then improved compressor architecture and summary preprocessing (`round1_stage*` artifacts).  
Best fair configuration: stronger architecture + `--no-standardize-summary`.

- 3 seeds (`41,42,43`): inflation = `1.0363`
- 5 seeds (`41..45`): inflation = `1.0391`

This is a major reduction from the initial CNN inflation `1.8024`:

- absolute drop: `0.7661`
- relative reduction: `42.5%`

## L1+VMIM focused optimization outcome (new)

You were right to call this out: the earlier report did not include a dedicated L1+VMIM compressor optimization for BNT robustness.  
I ran a focused paired optimization (`l1vmim_opt_round2`) with:

- condition-specific compressor retraining (`nobnt` and `bnt`),
- a screening stage at seed `41`,
- a 3-seed confirmation (`41,42,43`) for the best screened configuration.

### Configurations screened

1. `cdim64_h512_nf8x256`
2. `cdim64_h768_nf10x384`

### Best new L1+VMIM configuration

- `cdim64_h768_nf10x384`
- 3-seed confirmed metrics:
  - no-BNT `std_sum` mean: `0.3791`
  - BNT `std_sum` mean: `0.5241`
  - inflation: `1.3825`
  - no-BNT `bias_l2` mean: `0.0701`
  - BNT `bias_l2` mean: `0.2044`

### Improvement vs previous L1+VMIM baseline

Previous baseline (core study): inflation `1.5793`, BNT `std_sum` `0.6167`.  
New optimized L1+VMIM: inflation `1.3825`, BNT `std_sum` `0.5241`.

- inflation improvement: `-0.1968` (absolute)
- BNT width reduction: `15.0%`

So, L1+VMIM does improve substantially when explicitly optimized, but it still remains notably more BNT-inflated than optimized CNN.

## Final scientific interpretation

1. BNT inflation is real for L1-type summaries in this setup. Raw L1 remains at strong inflation (`~1.60`), and optimized L1+VMIM improves to `~1.38` but still shows clear inflation and BNT-side bias growth.

2. VMIM compression can reduce (but not eliminate) the BNT penalty when explicitly optimized for this setting: `1.579 -> 1.383` inflation in our focused round.

3. The CNN result depends strongly on compressor quality. With the initial configuration, CNN looked worst; after targeted optimization, CNN becomes near-lossless under BNT (`~1.04`).

4. This behavior is physically consistent with BNT being linear/invertible: information can be preserved if the learned summary captures the induced cross-bin structure. A weak/undertrained compressor can fail to do so.

5. Relative to optimized CNN (`1.036`), raw L1 inflation (`1.602`) is about `1.55x` larger, and optimized L1+VMIM inflation (`1.383`) is still about `1.33x` larger.

## Practical conclusion for this project stage

- **Best BNT-robust method currently available in this repository:** optimized CNN (Stage J family).
- **L1 under BNT:** still substantially broader and more biased.
- **L1+VMIM under BNT:** improved after dedicated optimization, but still clearly less robust than optimized CNN.

## Key artifacts

- Core metrics: `scripts/sbi/bnt_tomo4_study/bnt_metrics_summary.json`
- Core posteriors: `scripts/sbi/bnt_tomo4_study/posteriors/*.npy`
- Core overlays (all methods): `scripts/sbi/bnt_tomo4_study/overlays/`
- Optimized CNN metrics: `scripts/sbi/bnt_tomo4_study/round1_stageJ_postproc/quick_metrics_seed41_42_43_nostd.json`
- Optimized CNN 5-seed confirmation: `scripts/sbi/bnt_tomo4_study/round1_stageJ_postproc/quick_metrics_seed41_45_nostd.json`
- Optimized CNN overlays: `scripts/sbi/bnt_tomo4_study/overlays_latest/`
- Optimized L1+VMIM screen: `scripts/sbi/bnt_tomo4_study/l1vmim_opt_round2/screen_results_seed41.json`
- Optimized L1+VMIM 3-seed confirm: `scripts/sbi/bnt_tomo4_study/l1vmim_opt_round2/best_config_3seed_metrics.json`
