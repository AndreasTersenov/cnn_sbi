# GATE C — lane-B post-cut arms (derived verdicts; PLAN_M4_GATE_C.md)

Does the post-cut recombination gain (B1/B2 ~1.8x the uniform-cut noBNT B3) survive
calibration? TARP-stratified-val (600 pts, FoM3 terciles, 3 seeds) + SBC. preproc
log1p-zscore/clip5/min-var1e-5; GPU 2.

| arm | FoM3 | TARP HIGH/MID/LOW (dim3) | net bias | SBC std (Om,s8,w0) | verdict |
|---|---|---|---|---|---|
| B0_bntcut_l1 | 268 | -0.072/+0.058/+0.052 | +0.006±0.037 | 0.300,0.294,0.293 | **PASS-with-caveat** |
| B1_bntcut_sums | 596 | +0.058/+0.057/-0.078 | -0.000±0.056 | 0.303,0.305,0.292 | **PASS-with-caveat** |
| B2_bntcut_deep2 | 613 | -0.090/+0.074/+0.048 | +0.003±0.029 | 0.300,0.294,0.291 | **PASS-with-caveat** |
| B3_nobnt_unicut | 337 | —/—/— | — | nan,nan,nan | **INCOMPLETE** |

(+ net bias = conservative/over-covers; − = over-confident. Uniform=0.289.)

## Reading (registered band P-B, PLAN_M4_GATE_C.md)
- B1/B2 pass-with-caveat -> gain is broadly real but carry the calibration caveat (named tercile/direction); quote calibrated marginals alongside FoM3.
