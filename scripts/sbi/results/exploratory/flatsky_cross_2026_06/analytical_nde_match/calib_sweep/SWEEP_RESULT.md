# Calibration sweep — RealNVP capacity × 5-seed ensemble (levers #1+#3)

Goal: clean PASS (worst dev ≤0.05 AND SBC std ∈[0.275,0.305]) at ~CNN FoM3.
Baseline l1+product 4×128 / 2-3 seeds = 3270, PASS-with-caveat (net ~−0.02, SBC std ~0.30).
CNN ref 3293 (PASS). Raw l1+product-MAF 2875 (PASS-clean).

| config | seeds | FoM3 | worst dev | net bias | SBC std (Om/s8/w0) | verdict |
|---|---|---|---|---|---|---|
| 3×128 | 5 | 3173 | 0.071 | +0.016 | 0.300/0.305/0.300 | **PASS-with-caveat** |
| 3×64 | 5 | 3172 | 0.077 | +0.010 | 0.300/0.304/0.299 | **PASS-with-caveat** |
| 4×128 | 5 | 3084 | 0.094 | +0.003 | 0.301/0.305/0.301 | **PASS-with-caveat** |
| 4×64 | 5 | 3133 | 0.089 | -0.010 | 0.299/0.303/0.299 | **PASS-with-caveat** |
