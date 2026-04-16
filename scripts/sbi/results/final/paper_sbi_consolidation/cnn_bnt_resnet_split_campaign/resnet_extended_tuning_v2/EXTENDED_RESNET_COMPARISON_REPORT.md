# Extended ResNet vs Plain-CNN Consolidated Comparison (v2)

## Scope
- Consolidates prior split campaign, prior ResNet50 inflation tuning, and new v2 ResNet sweeps.
- Includes required references: `control_plain_split`, `advanced_arch64_dense256_nostd_long`, and baseline `resnet50_split`.
- Seed-matched check uses seeds: 41, 42, 43.

## Consolidated comparison (full available seeds)
| Config | Category | Seeds | inflation_std_sum_bnt_over_nobnt | fom3_ratio_bnt_over_nobnt | nobnt_fom3_mean | bnt_fom3_mean | rank_score | retention_vs_control_plain | retention_vs_advanced_plain_long |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control_plain_split | prior_split_reference | 41,42,43,44,45 | 1.0592 | 0.6099 | 396,125.4 | 241,591.1 | 0.4493 | 1.0000 | 0.7639 |
| advanced_arch64_dense256_nostd_long | plain_cnn_reference | 41,42,43,44,45 | 1.0369 | 0.8462 | 518,590.2 | 438,810.8 | 0.1907 | 1.3092 | 1.0000 |
| resnet50_split | prior_split_reference | 41,42,43 | 1.0703 | 0.5791 | 165,789.9 | 96,008.9 | 0.4912 | 0.4185 | 0.3197 |
| std10k_l10h320 | prior_resnet50_tuning | 41,42,43 | 0.9370 | 0.8588 | 109,490.2 | 94,035.5 | 0.2042 | 0.2764 | 0.2111 |
| long15k_nostd6k_l8h256 | prior_resnet50_tuning | 41,42,43 | 1.0258 | 0.8084 | 323,659.9 | 261,642.9 | 0.2174 | 0.8171 | 0.6241 |
| long15k_std6k_l8h256 | prior_resnet50_tuning | 41,42,43 | 1.0418 | 0.7561 | 342,639.5 | 259,055.9 | 0.2858 | 0.8650 | 0.6607 |
| r50_long30k_std10k_l10h320 | resnet50_v2 | 41,42,43 | 0.9832 | 0.8028 | 321,064.7 | 257,745.4 | 0.2140 | 0.8105 | 0.6191 |
| r50_long30k_nostd6k_l8h256 | resnet50_v2 | 41,42,43 | 1.0512 | 0.7811 | 507,302.5 | 396,232.7 | 0.2702 | 1.2807 | 0.9782 |
| resnet18_long15k_std10k_l10h320 | backbone_v2 | 41,42,43 | 1.0348 | 1.0049 | 231,794.4 | 232,926.1 | 0.0397 | 0.5852 | 0.4470 |
| resnet34_long15k_std10k_l10h320 | backbone_v2 | 41,42,43 | 0.8318 | 1.6840 | 164,492.6 | 277,009.9 | 0.8522 | 0.4153 | 0.3172 |

## Seed-matched comparison (41/42/43)
| Config | matched_seeds | inflation_std_sum_bnt_over_nobnt | fom3_ratio_bnt_over_nobnt | nobnt_fom3_mean | bnt_fom3_mean | rank_score | retention_vs_control_plain | retention_vs_advanced_plain_long |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control_plain_split | 41,42,43 | 1.0580 | 0.6196 | 396,073.4 | 245,409.7 | 0.4384 | 1.0000 | 0.7601 |
| advanced_arch64_dense256_nostd_long | 41,42,43 | 1.0405 | 0.8129 | 521,088.6 | 423,611.1 | 0.2276 | 1.3156 | 1.0000 |
| resnet50_split | 41,42,43 | 1.0703 | 0.5791 | 165,789.9 | 96,008.9 | 0.4912 | 0.4186 | 0.3182 |
| std10k_l10h320 | 41,42,43 | 0.9370 | 0.8588 | 109,490.2 | 94,035.5 | 0.2042 | 0.2764 | 0.2101 |
| long15k_nostd6k_l8h256 | 41,42,43 | 1.0258 | 0.8084 | 323,659.9 | 261,642.9 | 0.2174 | 0.8172 | 0.6211 |
| long15k_std6k_l8h256 | 41,42,43 | 1.0418 | 0.7561 | 342,639.5 | 259,055.9 | 0.2858 | 0.8651 | 0.6575 |
| r50_long30k_std10k_l10h320 | 41,42,43 | 0.9832 | 0.8028 | 321,064.7 | 257,745.4 | 0.2140 | 0.8106 | 0.6161 |
| r50_long30k_nostd6k_l8h256 | 41,42,43 | 1.0512 | 0.7811 | 507,302.5 | 396,232.7 | 0.2702 | 1.2808 | 0.9735 |
| resnet18_long15k_std10k_l10h320 | 41,42,43 | 1.0348 | 1.0049 | 231,794.4 | 232,926.1 | 0.0397 | 0.5852 | 0.4448 |
| resnet34_long15k_std10k_l10h320 | 41,42,43 | 0.8318 | 1.6840 | 164,492.6 | 277,009.9 | 0.8522 | 0.4153 | 0.3157 |

## Does any ResNet now beat advanced plain long on both parity and retention?
- **No (full-seed criterion).** No ResNet has both lower parity rank score than advanced plain long and >1.0 retention_vs_advanced_plain_long.
- **No (seed-matched criterion).** Same tradeoff remains under seeds 41/42/43.

**Remaining tradeoff:**
- Best parity ResNet is `resnet18_long15k_std10k_l10h320` with rank_score=0.0397, but retention_vs_advanced_plain_long=0.4470.
- Best retention ResNet is `r50_long30k_nostd6k_l8h256` with retention_vs_advanced_plain_long=0.9782, but rank_score=0.2702.

## Recommendations
- **Parity-first:** `resnet18_long15k_std10k_l10h320` (rank_score=0.0397, infl=1.0348, fom_ratio=1.0049).
- **Retention-aware:** `r50_long30k_nostd6k_l8h256` (retention_vs_advanced_plain_long=0.9782, rank_score=0.2702).
- **Practical default for next production run:** `r50_long30k_nostd6k_l8h256` (best parity-retention balance score=0.2919; retention_vs_advanced_plain_long=0.9782, rank_score=0.2702).
