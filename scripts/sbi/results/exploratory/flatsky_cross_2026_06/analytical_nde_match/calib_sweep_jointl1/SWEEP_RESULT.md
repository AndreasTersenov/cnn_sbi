# jointl1 calibration sweep — RealNVP capacity × 5-seed ensemble

Goal: shave marginal over-confidence (SBC std ~0.31) toward ideal (0.289) without losing FoM3.
Baseline jointl1 4×128 / 3-seed = FoM3 3754, pooled SBC 0.313/0.316/0.304, TARP net -0.003.
Reference: l1+product 3045 (SBC ~0.30), CNN 3326 (SBC ~0.29).

| config | seeds | FoM3 | worst dev | net bias | SBC std (Om/s8/w0) | verdict |
|---|---|---|---|---|---|---|
| 4×128 | 5 | 3753 | 0.085 | -0.008 | 0.313/0.316/0.305 | **PASS-with-caveat** |
| 3×128 | 5 | 3821 | 0.076 | -0.013 | 0.312/0.313/0.305 | **PASS-with-caveat** |
| 4×64 | 5 | 3918 | 0.050 | -0.007 | 0.312/0.315/0.305 | **PASS-with-caveat** |
| 3×64 | 5 | 3857 | 0.111 | -0.019 | 0.311/0.314/0.304 | **FAIL** |
| 2×64 | 5 | 3998 | 0.060 | -0.031 | 0.316/0.316/0.302 | **PASS-with-caveat** |
