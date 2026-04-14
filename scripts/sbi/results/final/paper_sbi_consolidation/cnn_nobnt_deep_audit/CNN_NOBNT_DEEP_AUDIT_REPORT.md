# CNN no-BNT deep audit report

- Campaign root: `/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/final/paper_sbi_consolidation/cnn_nobnt_deep_audit`
- Baseline run: `baseline_fulltrain`
- Metrics: width (`std_sum`), FoM3, `sigma8` std, and Omega_m-sigma_8 covariance proxies.

## Static pipeline checks

- train/test tfds_id overlap: `0`; theta overlap: `0`.
- split_a/split_b tfds_id overlap: `0`; theta overlap: `899`.
- h0 rescale check passed: `True`; augmentation stochastic check passed: `True`.

## Dynamic control summary (mean across seeds)

| run | std_sum | FoM3 | sigma8_std | corr(Om,s8) | std_ratio | fom_ratio | sigma8_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_fulltrain` | 0.1899 | 536358.5398 | 0.0140 | -0.2084 | 1.0000 | 1.0000 | 1.0000 |
| `baseline_fulltrain_shuffle` | 0.8321 | 153.2595 | 0.2577 | -0.6934 | 4.3813 | 0.0003 | 18.3468 |
| `split70_disjoint` | 0.1862 | 537097.3657 | 0.0138 | -0.2523 | 0.9802 | 1.0014 | 0.9801 |
| `split70_disjoint_shuffle` | 0.8333 | 153.2297 | 0.2578 | -0.6993 | 4.3881 | 0.0003 | 18.3572 |
| `split70_small_nde10` | 0.1867 | 532790.3965 | 0.0146 | -0.2936 | 0.9832 | 0.9933 | 1.0386 |
| `split70_long12000` | 0.1881 | 522701.7136 | 0.0145 | -0.2622 | 0.9906 | 0.9745 | 1.0352 |

## Interpretation highlights

- `baseline_fulltrain_shuffle`: std_ratio=4.3813, fom_ratio=0.0003, sigma8_ratio=18.3468.
- `split70_disjoint_shuffle`: std_ratio=4.3881, fom_ratio=0.0003, sigma8_ratio=18.3572.
- `split70_disjoint`: std_ratio=0.9802, fom_ratio=1.0014, corr_delta=-0.0439.
- `split70_small_nde10`: std_ratio=0.9832, fom_ratio=0.9933, corr_delta=-0.0852.
- `split70_long12000`: std_ratio=0.9906, fom_ratio=0.9745, corr_delta=-0.0538.

Overlay figures were written to `figures/overlay_<baseline>_vs_<run>_combined.png`.