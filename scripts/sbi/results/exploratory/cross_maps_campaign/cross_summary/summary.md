# Cross-maps campaign summary

**Apples-to-apples comparison: `cross_*` vs `auto_zm_*`** — both arms use `grid_20deg_160px_nonoverlap48` (multipatch, ~150k maps) with `--zero-mean-maps`, run via the same script. The only difference is the channel set (4 auto vs 4 auto + 6 cross).

**Legacy reference: `auto_bnt` / `auto_nobnt`** — single-patch `grid_20deg_160px`, no zero-mean. Kept for context only; do NOT use for the cross-information contrast.

## Aggregate FoM3 (mean over seeds 41/42/43)

| arm | std_sum_3par | sigma_omega_m | sigma_sigma_8 | sigma_w_0 | FoM3 |
|---|---|---|---|---|---|
| cross_bnt_pct1 | 0.44521 | 0.06539 | 0.13249 | 0.24733 | 1155.67 |
| cross_nobnt_pct1 | 0.25620 | 0.03472 | 0.04613 | 0.17536 | 11545.21 |
| cross_bnt | 0.48703 | 0.06648 | 0.14345 | 0.27710 | 392.98 |
| cross_nobnt | 0.31320 | 0.04031 | 0.06257 | 0.21032 | 1962.01 |
| auto_zm_bnt | 0.48570 | 0.06672 | 0.14100 | 0.27798 | 789.27 |
| auto_zm_nobnt | 0.24461 | 0.03388 | 0.04410 | 0.16663 | 13130.75 |
| auto_bnt | 0.50173 | 0.07414 | 0.15412 | 0.27346 | 640.27 |
| auto_nobnt | 0.23923 | 0.03553 | 0.04678 | 0.15692 | 11429.87 |
| harm_cross_bnt | 0.32641 | 0.04174 | 0.07075 | 0.21392 | 5160.65 |
| harm_cross_nobnt | 0.19346 | 0.02452 | 0.04083 | 0.12811 | 59243.09 |

## Per-seed FoM3

| arm | s41 | s42 | s43 |
|---|---|---|---|
| cross_bnt_pct1 | 875.67 | 1462.87 | 1128.47 |
| cross_nobnt_pct1 | 10932.42 | 14302.26 | 9400.95 |
| cross_bnt | 429.75 | 377.72 | 371.47 |
| cross_nobnt | 2444.26 | 1936.81 | 1504.96 |
| auto_zm_bnt | 722.76 | 890.41 | 754.62 |
| auto_zm_nobnt | 12804.82 | 16622.61 | 9964.84 |
| auto_bnt | 554.13 | 596.24 | 770.45 |
| auto_nobnt | 9786.53 | 16930.84 | 7572.24 |
| harm_cross_bnt | 5627.26 | 5003.13 | 4851.57 |
| harm_cross_nobnt | 58654.29 | 63533.80 | 55541.20 |

## Cross-vs-auto-zm ratios (matched comparison)

| metric | BNT pct1 / auto_zm | BNT min/max / auto_zm | no-BNT pct1 / auto_zm | no-BNT min/max / auto_zm |
|---|---|---|---|---|
| std_sum_3par | 0.917 | 1.003 | 1.047 | 1.280 |
| omega_m_std | 0.980 | 0.997 | 1.025 | 1.190 |
| sigma8_std | 0.940 | 1.017 | 1.046 | 1.419 |
| w0_std | 0.890 | 0.997 | 1.052 | 1.262 |
| fom3 | 1.464 | 0.498 | 0.879 | 0.149 |

Interpretation: ratio > 1 for FoM3 means cross channels add information; ratio < 1 for std means tighter constraints (good).
