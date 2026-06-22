# Shear-aware (rotated-grid) joint ℓ1 in BNT space — gated

Per-(pair,scale) 2-D PCA-whitened binning (shear-aware transport). Matched pipeline.
Baseline (axis-aligned adaptive-ranges): jointl1_nobnt 3754 PASS-caveat (SBC 0.31); jointl1_bnt 3232, raw ret 0.861, FAIL (SBC 0.33, dev 0.110).

| arm | FoM3 n=9000 | gate | worst dev | SBC std |
|---|---|---|---|---|
| jointl1_nobnt_rot | 4565 | PASS-with-caveat | 0.080 | 0.318/0.320/0.313 |
| jointl1_bnt_rot | 3221 | FAIL | 0.115 | 0.338/0.343/0.312 |

**rotated retention (n=9000) = 3221/4565 = 0.705**  (axis-aligned was 0.861, FAIL)
