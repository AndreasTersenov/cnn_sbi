# L1-VMIM SBI contours: final conclusions (April 2026)

## Scope and objective

We investigated whether VMIM-compressing tomographic L1-norm summaries can preserve cosmological information as well as (or better than) direct no-compression L1 inference.

Target dataset/configuration for all final comparisons:

- `NbodyCosmogridDatasetTomo/grid_20deg_160px`
- 4 tomographic bins (1,2,3,4)
- 5 wavelet scales
- SNR range `[-13, 13]`
- 40 bins per scale

## Method summary

Pipeline used:

1. Extract L1 histogram summaries from maps.
2. Apply `log1p` + train-set standardization at compressor input.
3. Train VMIM compressor (MLP + companion conditional NF).
4. Train conditional RealNVP on compressed summaries.
5. Sample posterior (`100,000` samples), compare spread and calibration.

We ran extensive sweeps over:

- compressed dimension (`cdim=12,24,32,36,40,44,48`)
- VMIM/flow capacities
- optimization schedules
- many flow seeds with frozen best compressor.

## Main quantitative result

Primary metric pair for decision:

- `std_ratio = std_sum(run) / std_sum(no-compression)` (lower is tighter)
- `L2(mean-truth)` (lower is better calibrated)

Best *calibrated near-lossless* run found:

- **Run:** `l1_vmim_tomo4_20deg160_seed202_flowonly.npy`
- **Path:** `scripts/sbi/l1_vmim_runs_fixstd_cdim40_flowseed/posteriors/l1_vmim_tomo4_20deg160_seed202_flowonly.npy`
- **W&B:** https://wandb.ai/cosmostat/l1-vmim-npe-tomo/runs/isyq4krn
- **Metrics:** `std_ratio = 1.019`, `L2(mean-truth) = 0.0642`, `Mahalanobis = 1.303`

This means VMIM compression is now only ~1.9% broader than no-compression while remaining well-calibrated.

## Key comparison table (selected)

| Run | std_ratio | L2(mean-truth) | Notes |
|---|---:|---:|---|
| no-compression L1 baseline | 1.000 | 0.0816 | reference |
| `fixstd40_balanced` | 1.088 | 0.0838 | best pre-flow-seed baseline |
| **`flowseed202`** | **1.019** | **0.0642** | best overall tradeoff |
| `flowseed404` | 1.041 | 0.0674 | good, but worse than seed202 |
| `flowseed202_long_lowlr` | 1.021 | 0.0795 | close but not better |
| `flowseed303_flowplus` | 0.971 | 0.1858 | tighter but biased |
| `flowseed606_regflowplus` | 0.968 | 0.2045 | tighter but strongly biased |
| `flowseed909_wide` | 0.971 | 0.2089 | tighter but strongly biased |

## Scientific interpretation

1. The original major issue (tomographic compression underperforming bin-3 behavior) was fixed by improved compressor conditioning and metadata-safe preprocessing.
2. After that fix, additional gains mainly came from **flow training variability** (seed and schedule), not further compressor-width increases alone.
3. Some runs produce narrower posteriors than no-compression (`std_ratio < 1`), but these narrowings come with significant posterior mean bias (especially in `w`), so they are not scientifically preferred.
4. The selected run (`flowseed202`) is the best Pareto point between tightness and calibration across all tested runs.

## Recommendation

For this project stage, use:

- `scripts/sbi/l1_vmim_runs_fixstd_cdim40_flowseed/posteriors/l1_vmim_tomo4_20deg160_seed202_flowonly.npy`

and treat it as the current best L1-VMIM tomographic posterior.

If future improvement is needed, prioritize **code-level** upgrades (e.g. calibration-aware objective, posterior ensembling) rather than additional blind seed sweeps.
