# Final Noise Curriculum Report

- Generated UTC: 2026-04-17T04:04:26.527549+00:00
- Campaign root: /mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_noise_curriculum_campaign
- Follow-up root: /mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_noise_curriculum_campaign/followup_resnet18
- Rank score formula: abs(inflation_std_sum_bnt_over_nobnt-1)+abs(fom3_ratio_bnt_over_nobnt-1)
- FoM parity formula: abs(fom3_ratio_bnt_over_nobnt-1)

## Consolidated metrics across six configs

| config | family | scope | steps | stage_fracs | inflation_ratio | fom3_ratio | abs(fom3-1) | sigma8_std_ratio | rank_score |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| plain_ref | plain | primary | 120000 | - | 1.0200 | 0.9137 | 0.0863 | 1.1290 | 0.1063 |
| resnet18_curriculum | resnet18 | primary | 15000 | - | 0.9615 | 0.8684 | 0.1316 | 0.9076 | 0.1701 |
| plain_curriculum | plain | primary | 120000 | - | 1.1117 | 0.7570 | 0.2430 | 1.2721 | 0.3548 |
| resnet18_curriculum_long22k | resnet18 | followup | 22500 | - | 1.1072 | 0.6088 | 0.3912 | 1.2058 | 0.4984 |
| resnet18_ref | resnet18 | primary | 15000 | - | 1.3329 | 0.4331 | 0.5669 | 2.5558 | 0.8999 |
| resnet18_curriculum_slowramp | resnet18 | followup | 15000 | - | 1.3261 | 0.3413 | 0.6587 | 2.1775 | 0.9847 |

## Required conclusions

- Plain family curriculum helped? NO. plain_ref abs(fom3-1) 0.0863 vs plain_curriculum 0.2430; rank 0.1063 vs 0.3548.
- ResNet18 family curriculum helped? YES. resnet18_ref abs(fom3-1) 0.5669 vs resnet18_curriculum 0.1316; rank 0.8999 vs 0.1701.
- Follow-up variants beat primary resnet18_curriculum? NO. primary abs(fom3-1) 0.1316; slowramp 0.6587; long22k 0.3912.
- Best config by user-priority FoM parity (minimum abs(fom_ratio-1)): plain_ref with abs(fom3-1) 0.0863.

## Path validation

The following referenced files were validated to exist before consolidation:
- /mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_noise_curriculum_campaign/plain_ref/metrics.json
- /mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_noise_curriculum_campaign/plain_curriculum/metrics.json
- /mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_noise_curriculum_campaign/resnet18_ref/metrics.json
- /mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_noise_curriculum_campaign/resnet18_curriculum/metrics.json
- /mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_noise_curriculum_campaign/followup_resnet18/resnet18_curriculum_slowramp/metrics.json
- /mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_noise_curriculum_campaign/followup_resnet18/resnet18_curriculum_long22k/metrics.json
- /mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_noise_curriculum_campaign/followup_resnet18/FOLLOWUP_RESNET18_SUMMARY.md
- /mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_noise_curriculum_campaign/followup_resnet18/followup_resnet18_comparison_summary.json
- /mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_noise_curriculum_campaign/followup_resnet18/followup_resnet18_comparison_summary.csv
