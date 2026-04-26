# jaxili auto-only consistency vs reference

Reference: `final/paper_sbi_consolidation/bnt_comparison_tomo4/posteriors/l1_tomo4_20deg160_{regime}_s{41,42,43}.npy`.
Current arms: `jaxili_auto_{regime}/posteriors/l1_tomo4_20deg160_{regime}_s{41,42,43}.npy`,
built with the exact reference config (`npe_l1norm_jaxili_nbody_tomo.py`, no PCA, grid_20deg_160px, lr=1e-4,
batch 256, --total-steps 5000, SNR [-13, 13], n_scales=5, l1_nbins=40).

## BNT

| metric | current | reference | ratio (cur/ref) |
|---|---|---|---|
| std_sum_3par | 0.50173 | 0.49257 | 1.019 |
| omega_m_std | 0.07414 | 0.06985 | 1.062 |
| sigma8_std | 0.15412 | 0.15108 | 1.020 |
| w0_std | 0.27346 | 0.27164 | 1.007 |
| fom3 | 640.27445 | 679.86411 | 0.942 |

| param | current mean | ref mean | current std | ref std |
|---|---|---|---|---|
| Omega_m | 0.2581 | 0.2452 | 0.0741 | 0.0698 |
| sigma_8 | 0.8707 | 0.8861 | 0.1541 | 0.1511 |
| w_0 | -0.8281 | -0.8536 | 0.2735 | 0.2716 |
| h_0 | 0.7270 | 0.7264 | 0.0513 | 0.0515 |
| n_s | 0.9626 | 0.9697 | 0.0558 | 0.0560 |
| Omega_b | 0.0451 | 0.0444 | 0.0084 | 0.0085 |

## NOBNT

| metric | current | reference | ratio (cur/ref) |
|---|---|---|---|
| std_sum_3par | 0.23923 | 0.23939 | 0.999 |
| omega_m_std | 0.03553 | 0.03584 | 0.991 |
| sigma8_std | 0.04678 | 0.04665 | 1.003 |
| w0_std | 0.15692 | 0.15690 | 1.000 |
| fom3 | 11429.87292 | 11127.02182 | 1.027 |

| param | current mean | ref mean | current std | ref std |
|---|---|---|---|---|
| Omega_m | 0.2686 | 0.2710 | 0.0355 | 0.0358 |
| sigma_8 | 0.8409 | 0.8387 | 0.0468 | 0.0467 |
| w_0 | -1.0475 | -1.0188 | 0.1569 | 0.1569 |
| h_0 | 0.7146 | 0.7161 | 0.0491 | 0.0492 |
| n_s | 0.9559 | 0.9570 | 0.0490 | 0.0497 |
| Omega_b | 0.0461 | 0.0458 | 0.0081 | 0.0081 |
