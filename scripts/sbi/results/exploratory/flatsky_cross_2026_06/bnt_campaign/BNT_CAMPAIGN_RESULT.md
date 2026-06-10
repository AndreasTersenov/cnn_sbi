# BNT flat-local campaign — inflation ratios (FoM3_BNT / FoM3_noBNT)

Pooled 3-MAF 9000-obs medians; no-BNT references read from the existing campaign dirs. Predictions (plan §B): L1-auto inflates ≪1, L1+product less so, CNN ≈ 1.

| arm | no-BNT FoM3 | BNT FoM3 | inflation (BNT/noBNT) |
|---|---|---|---|
| L1 none | 2405 | 364 | 0.15× |
| L1 product | 2875 | 637 | 0.22× |
| CNN none s41 | 2325 | 2174 | 0.94× |
| CNN none s42 | 2170 | 2172 | 1.00× |
| CNN none s43 | 2480 | 2137 | 0.86× |
| **CNN none mean-of-seeds** |  |  | **0.93×** |
| CNN product s41 | 2181 | 2054 | 0.94× |
| CNN product s42 | 2393 | 2004 | 0.84× |
| CNN product s43 | 2433 | 2072 | 0.85× |
| **CNN product mean-of-seeds** |  |  | **0.88×** |

**Prediction ladder (derived):**
1. L1-auto inflates (ratio 0.15 < 0.9): HOLDS
2. L1+product inflates less than L1-auto (0.22 vs 0.15): HOLDS
3. CNN ≈ lossless (auto ratio 0.93 > 0.9): HOLDS

Caveat: FoM3 is correlation-sensitive; check σ/2D in the median_summary.json files before headlining, and GATE C the load-bearing BNT arms.