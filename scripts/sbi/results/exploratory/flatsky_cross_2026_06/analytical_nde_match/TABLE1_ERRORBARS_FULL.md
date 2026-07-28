# Table 1 — FoM3 error bars (retrained on Jean-Zay, 2026-07-28)

## Headline

| Row | published | **retrained** (quoted) | Δ | **±** (n seeds) | n | band |
|---|--:|--:|--:|--:|--:|---|
| l1 auto, no-BNT | 2448 | **2776.9** (single) | +13.4% | **±31 (1.1%)** | 3 | 2739–2800 |
| l1 auto, BNT | 388 | **390.7** (ensemble) | +0.7% | **±49 (11.1%)** | 3 | 389–481 |
| l1 +product, no-BNT | 3045 | **3231.8** (single) | +6.1% | **±199 (6.0%)** | 3 | 3176–3546 |
| l1 +product, BNT | 718 | **758.3** (ensemble) | +5.6% | **±38 (4.1%)** | 3 | 891–964 |
| joint l1, no-BNT | 3371 | **3379.5** (ensemble) | +0.3% | **±108 (2.8%)** | 3 | 3723–3927 |
| joint l1, BNT | 2424 | **2405.2** (ensemble) | -0.8% | **±260 (8.6%)** | 3 | 2740–3251 |
| CNN, no-BNT | 3326 | **3427.6** (single) | +3.1% | **±14 (0.4%)** | 3 | 3401–3428 |
| CNN, BNT | 3186 | **3147.1** (single) | -1.2% | **±19 (0.6%)** | 3 | 3147–3183 |

## Full detail

| Row | published | retrained quoted | retrained single s41 | retrained ensemble | singles mean | per-seed values | ± (std) | bias single→ens | median SE (68%) | ρ | CV |
|---|--:|--:|--:|--:|--:|---|--:|--:|--:|--:|--:|
| l1 auto, no-BNT | 2448 | 2776.9 (single) | 2776.9 | 2511.1 | 2772.1 | 2776.9 / 2739.5 / 2800.0 | ±31 (1.1%) | -9.6% | ±4.95 | 0.000 | 0.171 |
| l1 auto, BNT | 388 | 390.7 (ensemble) | 388.9 | 390.7 | 445.2 | 388.9 / 481.0 / 465.6 | ±49 (11.1%) | +0.4% | ±1.12 | 0.002 | 0.257 |
| l1 +product, no-BNT | 3045 | 3231.8 (single) | 3231.8 | 3053.9 | 3317.8 | 3231.8 / 3175.9 / 3545.6 | ±199 (6.0%) | -5.5% | ±6.09 | 0.002 | 0.180 |
| l1 +product, BNT | 718 | 758.3 (ensemble) | 912.1 | 758.3 | 922.3 | 912.1 / 890.7 / 964.0 | ±38 (4.1%) | -16.9% | ±2.98 | 0.008 | 0.231 |
| joint l1, no-BNT | 3371 | 3379.5 (ensemble) | 3762.3 | 3379.5 | 3804.1 | 3762.3 / 3723.2 / 3926.7 | ±108 (2.8%) | -10.2% | ±7.60 | 0.005 | 0.169 |
| joint l1, BNT | 2424 | 2405.2 (ensemble) | 3075.1 | 2405.2 | 3022.3 | 3075.1 / 3251.4 / 2740.4 | ±260 (8.6%) | -21.8% | ±6.52 | 0.000 | 0.258 |
| CNN, no-BNT | 3326 | 3427.6 (single) | 3427.6 | 3324.2 | 3411.8 | 3427.6 / 3400.8 / 3406.9 | ±14 (0.4%) | -3.0% | ±8.37 | 0.003 | 0.178 |
| CNN, BNT | 3186 | 3147.1 (single) | 3147.1 | 3109.5 | 3163.1 | 3147.1 / 3183.4 / 3158.9 | ±19 (0.6%) | -1.2% | ±6.59 | 0.004 | 0.171 |

**± = spread over independently trained compressors (pre-ensemble singles)**, per `NOTE_FOM_ERROR_BARS.md` §5.3–5.4 — quoted for ALL rows, including the
ensemble-estimated ones, so it is comparable across rows as training stochasticity.

`retrained quoted` follows each row's published estimator convention: the SINGLE (seed 41) for ℓ1 auto/+product no-BNT and both CNN rows, the 3-compressor ENSEMBLE for the BNT ℓ1
rows and both joint ℓ1 rows (the ensemble is the quoted estimator only where the single failed the calibration battery — `RESULT_NOBNT_ENSEMBLE_ROBUSTNESS.md`).

The **single→ensemble shift is the BIAS term**, reported separately and never summed with the ± (§1, §5.4). Median SE = block bootstrap over the 180 patches keeping all 50 noise
reps, 10⁴ replicates, 68% percentile interval, at seed 41 (§4, §5.5). ρ = intra-patch correlation by one-way ANOVA; measured ≈0 everywhere, which is why the median term is
at the bottom of the note's predicted 0.1–3% range.
