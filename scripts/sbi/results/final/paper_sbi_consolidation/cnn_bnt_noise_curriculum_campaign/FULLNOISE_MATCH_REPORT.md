# Full-noise-match curriculum report

- Generated: 2026-04-17T16:19:40.460145+00:00
- Scope: compare full-noise-matched curriculum runs against prior family baselines.
- Rank score formula: abs(infl-1)+abs(fom_ratio-1).
- FoM parity formula: abs(fom3_ratio_bnt_over_nobnt-1).

## Ranked configs

1. plain_ref - infl=1.0200, fom_ratio=0.9137, sigma8_ratio=1.1290, rank=0.1063
2. plain_curriculum_fullnoise_match - infl=1.0404, fom_ratio=0.9049, sigma8_ratio=1.2096, rank=0.1354
3. resnet18_curriculum - infl=0.9615, fom_ratio=0.8684, sigma8_ratio=0.9076, rank=0.1701
4. resnet18_curriculum_fullnoise_match - infl=1.1334, fom_ratio=0.7467, sigma8_ratio=1.5245, rank=0.3867; note: bnt seed 42 low FoM3 (6858.2 vs median 275217.4)

## Family conclusions

- Plain family: full-noise match did not improve FoM parity (abs err 0.0951 vs 0.0863) and did not improve rank score (0.1354 vs 0.1063).
- ResNet18 family: full-noise match did not improve FoM parity (abs err 0.2533 vs 0.1316) and did not improve rank score (0.3867 vs 0.1701).
- ResNet18 full-noise run outlier detected: bnt seed 42 low FoM3 (6858.2 vs median 275217.4).

- Best overall by FoM parity: plain_ref (abs err=0.0863, fom_ratio=0.9137).
