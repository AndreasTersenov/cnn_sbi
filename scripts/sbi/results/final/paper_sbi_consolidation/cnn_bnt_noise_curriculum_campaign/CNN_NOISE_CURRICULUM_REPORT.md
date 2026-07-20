# CNN noise-curriculum campaign report

- Generated: 2026-04-17T02:49:42.041679+00:00
- Best by rank score: `plain_ref`
- Rank score formula: `abs(infl-1)+abs(fom_ratio-1)`

## Ranked variants

1. `plain_ref`
   - arch=plain, curriculum=False, infl=1.0200, fom_ratio=0.9137, sigma8_ratio=1.1290, rank=0.1063
2. `resnet18_curriculum`
   - arch=resnet18, curriculum=True, infl=0.9615, fom_ratio=0.8684, sigma8_ratio=0.9076, rank=0.1701
3. `plain_curriculum`
   - arch=plain, curriculum=True, infl=1.1117, fom_ratio=0.7570, sigma8_ratio=1.2721, rank=0.3548
4. `resnet18_ref`
   - arch=resnet18, curriculum=False, infl=1.3329, fom_ratio=0.4331, sigma8_ratio=2.5558, rank=0.8999
