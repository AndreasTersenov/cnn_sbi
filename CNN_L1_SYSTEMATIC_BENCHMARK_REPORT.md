# CNN vs L1 Systematic Benchmark Report (systematic_runs_24)

## 1) Benchmark matrix and configuration summary

- Artifact root: `/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/systematic_runs_24`
- Methods: `cnn, l1`
- Variants: `tomo4_10deg80, tomo4_20deg160, bin3_10deg80, bin3_20deg160`
- Seeds: `41, 42, 43`
- GPUs used: `0, 1, 2`
- Sweep hyperparameters: compressor_steps=2000, flow_steps=1200, npe_samples=10000

| Variant | TFDS | Field size | Npix | Tomo bins | nbins |
|---|---|---:|---:|---|---:|
| tomo4_10deg80 | `NbodyCosmogridDatasetTomo/grid` | 10 deg | 80 | 1,2,3,4 | 4 |
| tomo4_20deg160 | `NbodyCosmogridDatasetTomo/grid_20deg_160px` | 20 deg | 160 | 1,2,3,4 | 4 |
| bin3_10deg80 | `NbodyCosmogridDatasetTomo/grid` | 10 deg | 80 | 3 | 1 |
| bin3_20deg160 | `NbodyCosmogridDatasetTomo/grid_20deg_160px` | 20 deg | 160 | 3 | 1 |

### Command templates (from executed jobs)

- CNN compressor training (per variant):
```bash
python scripts/sbi/npe_cnn_nbody_tomo.py --no-wandb --map-kind nbody --tfds-name <tfds> --field-size <10|20> --field-npix <80|160> --nbins <1|4> --tomo-bin-indices <bins> --cache-dir <...> --save-dir <...> --train-compressor --compressor-steps 2000 --compressor-save-every 2000 --total-steps 1 --save-every 1 --no-sample --cuda-visible-devices <gpu>
```
- CNN evaluation (per variant, per seed):
```bash
python scripts/sbi/npe_cnn_nbody_tomo.py --no-wandb --map-kind nbody --seed <41|42|43> --tfds-name <tfds> --field-size <10|20> --field-npix <80|160> --nbins <1|4> --tomo-bin-indices <bins> --cache-dir <...> --save-dir <...> --compressor-params <...> --compressor-state <...> --total-steps 1200 --save-every 300 --npe-samples 10000 --posterior-out <...>.npy --ds-batch-size 500 --cuda-visible-devices <gpu>
```
- L1 evaluation (per variant, per seed):
```bash
python scripts/sbi/npe_l1norm_nbody_tomo.py --no-wandb --map-kind nbody --seed <41|42|43> --tfds-name <tfds> --field-size <10|20> --field-npix <80|160> --nbins <1|4> --tomo-bin-indices <bins> --cache-dir <...> --save-dir <...> --total-steps 1200 --save-every 300 --npe-samples 10000 --posterior-out <...>.npy --ds-batch-size 256 --cuda-visible-devices <gpu>
```

## 2) Run completion accounting

- Expected jobs: 28 (train=4, eval=24); observed jobs: 28
- Successful jobs (returncode=0): 28/28
- Job categories observed: train_compressor=4, cnn_eval=12, l1_eval=12
- Posterior files expected: 24; observed `.npy`: 24; summary rows: 24
- Missing method/variant/seed combinations: none
- Total wall time (sum of job durations): 52.96 min
- Runtime by category (min): train=5.87, cnn_eval=6.65, l1_eval=40.45
- Average eval runtime per job: cnn=33.2s, l1=202.2s (L1 is 6.1x slower)

## 3) Metrics: bias and spread comparisons

Definitions from sweep script: `bias_l2 = ||mean(posterior) - truth||_2`, `std_sum = sum(std(posterior, axis=0))`. Lower is better for both.

### 3.1 Seed-level paired results (CNN minus L1 deltas)

| Variant | Seed | CNN bias_l2 | L1 bias_l2 | Δ bias (CNN-L1) | CNN std_sum | L1 std_sum | Δ spread (CNN-L1) |
|---|---:|---:|---:|---:|---:|---:|---:|
| tomo4_10deg80 | 41 | 0.1686 | 0.1355 | +0.0331 | 0.5913 | 0.5652 | +0.0261 |
| tomo4_10deg80 | 42 | 0.1428 | 0.1375 | +0.0053 | 0.6225 | 0.6639 | -0.0415 |
| tomo4_10deg80 | 43 | 0.1666 | 0.2282 | -0.0616 | 0.6070 | 0.5730 | +0.0340 |
| tomo4_20deg160 | 41 | 0.1631 | 0.0600 | +0.1031 | 0.6000 | 0.4280 | +0.1720 |
| tomo4_20deg160 | 42 | 0.1593 | 0.0782 | +0.0811 | 0.5888 | 0.4195 | +0.1694 |
| tomo4_20deg160 | 43 | 0.1717 | 0.1784 | -0.0067 | 0.6030 | 0.4163 | +0.1867 |
| bin3_10deg80 | 41 | 0.1617 | 0.1232 | +0.0385 | 0.6010 | 0.6817 | -0.0807 |
| bin3_10deg80 | 42 | 0.1450 | 0.2333 | -0.0883 | 0.6166 | 0.7668 | -0.1502 |
| bin3_10deg80 | 43 | 0.1816 | 0.1765 | +0.0051 | 0.6021 | 0.6015 | +0.0006 |
| bin3_20deg160 | 41 | 0.1804 | 0.2845 | -0.1041 | 0.6000 | 0.5335 | +0.0666 |
| bin3_20deg160 | 42 | 0.1603 | 0.1913 | -0.0310 | 0.6164 | 0.4684 | +0.1480 |
| bin3_20deg160 | 43 | 0.1618 | 0.0615 | +0.1003 | 0.5998 | 0.4455 | +0.1543 |

- Pairwise wins over 12 variant×seed pairs: lower bias -> CNN 5, L1 7; lower spread -> CNN 3, L1 9.

### 3.2 Variant-level summary across seeds (mean ± sd)

| Variant | CNN bias_l2 | L1 bias_l2 | CNN vs L1 bias % | CNN std_sum | L1 std_sum | CNN vs L1 spread % |
|---|---:|---:|---:|---:|---:|---:|
| tomo4_10deg80 | 0.1593 ± 0.0144 | 0.1671 ± 0.0530 | -4.6% | 0.6069 ± 0.0156 | 0.6007 ± 0.0549 | +1.0% |
| tomo4_20deg160 | 0.1647 ± 0.0063 | 0.1055 ± 0.0638 | +56.1% | 0.5973 ± 0.0075 | 0.4213 ± 0.0061 | +41.8% |
| bin3_10deg80 | 0.1628 ± 0.0183 | 0.1777 ± 0.0551 | -8.4% | 0.6066 ± 0.0087 | 0.6833 ± 0.0827 | -11.2% |
| bin3_20deg160 | 0.1675 ± 0.0112 | 0.1791 ± 0.1120 | -6.5% | 0.6054 ± 0.0095 | 0.4825 ± 0.0456 | +25.5% |

Interpretation of % columns: negative means CNN is lower/better, positive means CNN is higher/worse relative to L1.

### 3.3 Overall summary across all variants and seeds

| Method | Runs | bias_l2 (mean ± sd) | std_sum (mean ± sd) |
|---|---:|---:|---:|
| CNN | 12 | 0.1636 ± 0.0118 | 0.6040 ± 0.0101 |
| L1 | 12 | 0.1573 ± 0.0713 | 0.5469 ± 0.1161 |

- Overall, CNN vs L1: bias_l2 +4.0% and std_sum +10.4%.

## 4) Practical conclusions and recommendation

1. **No single method dominates all regimes.** CNN is better on both metrics for `bin3_10deg80`, slightly better bias for `bin3_20deg160` and `tomo4_10deg80`, while L1 is clearly best on `tomo4_20deg160` (both lower bias and tighter spread).
2. **L1 is usually tighter (lower spread), but less seed-stable.** Overall spread is lower for L1, yet L1 seed-to-seed variability is much larger (e.g., bias SD 0.0713 for L1 vs 0.0118 for CNN).
3. **CNN is much faster.** Average eval runtime is ~33s for CNN vs ~202s for L1 (~6.1x slower for L1).
4. **Recommendation:** for production where throughput and run-to-run stability matter, prefer **CNN** as default. For `tomo4_20deg160` where max posterior sharpness/accuracy is priority and extra runtime is acceptable, prefer **L1**. A hybrid policy (method by variant) gives the best aggregate performance.

