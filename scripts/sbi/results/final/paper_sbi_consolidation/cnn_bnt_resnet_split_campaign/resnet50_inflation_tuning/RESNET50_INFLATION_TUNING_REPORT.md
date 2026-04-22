# RESNET50 Inflation-Tuning Report

## Objective
Consolidate ResNet50 inflation-tuning experiments and recommend the best next configuration to reduce BNT/no-BNT inflation with acceptable information retention.

## Inputs analyzed
- `campaign_summary.json`
- baseline `resnet50_split/metrics.json`
- control `control_plain_split/metrics.json`
- `nde_variants_summary.json` + per-variant NDE `metrics.json`
- `long_compressor_summary.json` + per-variant long-compressor `metrics.json`

## Transparent tradeoff metric
`rank_score_tradeoff = abs(inflation_std_sum_bnt_over_nobnt - 1) + abs(fom3_ratio_bnt_over_nobnt - 1)` (lower is better for parity only).

## Consolidated comparison (required + family bests)
| Config | inflation_std_sum_bnt_over_nobnt | fom3_ratio_bnt_over_nobnt | nobnt_fom3_mean | bnt_fom3_mean | no-BNT retention vs baseline | no-BNT retention vs control | tradeoff score | inflation < 1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline ResNet50 split (`resnet50_split`) | 1.0703 | 0.5791 | 165789.9 | 96008.9 | 100.0% | 41.9% | 0.4912 | no |
| Best NDE by tradeoff (`std10k_l10h320`) | 0.9370 | 0.8588 | 109490.2 | 94035.5 | 66.0% | 27.6% | 0.2042 | yes |
| Best NDE by retention (`std6k_l8h256`) | 1.0785 | 0.5905 | 184630.2 | 109018.7 | 111.4% | 46.6% | 0.4880 | no |
| Best long-compressor by tradeoff (`long15k_nostd6k_l8h256`) | 1.0258 | 0.8084 | 323659.9 | 261642.9 | 195.2% | 81.7% | 0.2174 | no |
| Best long-compressor by retention (`long15k_std6k_l8h256`) | 1.0418 | 0.7561 | 342639.5 | 259055.9 | 206.7% | 86.5% | 0.2858 | no |
| Control plain split (`control_plain_split`) | 1.0592 | 0.6099 | 396125.4 | 241591.1 | 238.9% | 100.0% | 0.4493 | no |

## Full ranking (all evaluated configurations)
| Rank | Config | Family | Inflation | FoM ratio | no-BNT retention vs baseline | no-BNT retention vs control | tradeoff score |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `std10k_l10h320` | nde_variant | 0.9370 | 0.8588 | 66.0% | 27.6% | 0.2042 |
| 2 | `long15k_nostd6k_l8h256` | long_compressor_variant | 1.0258 | 0.8084 | 195.2% | 81.7% | 0.2174 |
| 3 | `long15k_std6k_l8h256` | long_compressor_variant | 1.0418 | 0.7561 | 206.7% | 86.5% | 0.2858 |
| 4 | `nostd10k_l10h320` | nde_variant | 1.0248 | 0.6635 | 78.9% | 33.0% | 0.3612 |
| 5 | `control_plain_split` | control_reference | 1.0592 | 0.6099 | 238.9% | 100.0% | 0.4493 |
| 6 | `std6k_l8h256` | nde_variant | 1.0785 | 0.5905 | 111.4% | 46.6% | 0.4880 |
| 7 | `resnet50_split` | baseline | 1.0703 | 0.5791 | 100.0% | 41.9% | 0.4912 |

## Key findings
1. **NDE parity best (`std10k_l10h320`) over-corrects inflation** (0.9370 < 1) and has weak no-BNT retention (66.0% vs baseline), so parity gain comes with substantial information loss.
2. **Long-compressor variants dominate retention-aware tradeoff**: both long variants move inflation closer to 1 and substantially improve FoM ratio versus baseline while retaining/enhancing no-BNT information.
3. **Best retention-aware tradeoff** is `long15k_nostd6k_l8h256` (inflation 1.0258, FoM ratio 0.8084, no-BNT retention 195.2% vs baseline).
4. Campaign baseline context (`campaign_summary.json`): historical best among non-tuning split configs is `control_plain_split`, but within this ResNet50 tuning sweep the strongest retention-aware parity option is `long15k_nostd6k_l8h256`.

## Recommendation (next run priority)
**Prioritize `long15k_nostd6k_l8h256`.**

Rationale:
- Best tradeoff score among variants that (a) improve inflation closeness to 1 vs baseline, (b) improve BNT/no-BNT FoM ratio vs baseline, and (c) keep no-BNT retention at/above baseline.
- Strong inflation reduction relative to baseline (1.0703 → 1.0258) without over-correction.
- Strong parity improvement in FoM ratio (0.5791 → 0.8084) while preserving high no-BNT information.

Secondary option if maximizing absolute no-BNT FoM is prioritized over parity balance: `long15k_std6k_l8h256`.
