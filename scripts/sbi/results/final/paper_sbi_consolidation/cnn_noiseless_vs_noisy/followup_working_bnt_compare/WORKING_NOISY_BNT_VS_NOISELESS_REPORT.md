# Working noisy-BNT vs noiseless comparisons

- Working noisy BNT source: `cnn_bnt_losslessness_campaign_cdim10/advanced_arch64_dense256_nostd`
- Seed subset used for fair comparison: `41,42,43`

## Ratios

| comparison | std ratio | fom ratio | sigma8 std ratio | Om-s8 area ratio |
| --- | ---: | ---: | ---: | ---: |
| noiseless BNT / noisy working BNT | 0.7674 | 3.9877 | 0.5166 | 0.3596 |
| noisy working BNT / noisy working no-BNT | 1.0361 | 0.9079 | 1.1722 | 1.2194 |
| noiseless BNT / noiseless no-BNT | 0.9932 | 1.0528 | 0.9386 | 0.9299 |

## Figures

- `figures/overlay_noisy_working_bnt_vs_noiseless_bnt_combined.png`
- `figures/overlay_noiseless_nobnt_vs_noiseless_bnt_combined.png`