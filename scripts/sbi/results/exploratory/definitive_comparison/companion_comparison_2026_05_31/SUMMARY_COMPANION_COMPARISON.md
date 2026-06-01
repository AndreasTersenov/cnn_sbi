# Companion comparison — L1 vs CNN-RealNVP vs CNN-MAF (perm0, cs41, NDE seeds pooled)

⚠️ CNN FoM are tf.data-route leak-inflated (~1.6×). Compare **CNN-MAF vs CNN-RealNVP** (same leakage → companion delta is clean) and relative gains.

## FoM3 (pooled cs41)

| input | method | FoM3 | σ(Ωm) | σ(σ8) | σ(w0) | n |
|---|---|---|---|---|---|---|
| autocross | L1 | 34607 | 0.0273 | 0.0423 | 0.1245 | 3 |
| autocross | CNN-RealNVP | 26748 | 0.0268 | 0.0378 | 0.1508 | 3 |
| autocross | CNN-MAF | 11984 | 0.0346 | 0.0416 | 0.2126 | 3 |
| autoonly | L1 | 10560 | 0.0393 | 0.0519 | 0.2039 | 3 |
| autoonly | CNN-RealNVP | 9125 | 0.0351 | 0.0420 | 0.2163 | 3 |
| autoonly | CNN-MAF | 6679 | 0.0433 | 0.0590 | 0.2165 | 3 |

## Companion effect (CNN-MAF / CNN-RealNVP FoM3)

- autocross: MAF/RealNVP = **0.45×** (MAF 11984 vs RealNVP 26748)
- autoonly: MAF/RealNVP = **0.73×** (MAF 6679 vs RealNVP 9125)
