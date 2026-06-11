# Whitening test — decomposing the L1 BNT inflation

Per-channel L1 in Q = (BB^T)^(-1/2) B (noise-whitened BNT = orthogonal rotation of the original basis). Pooled 3-MAF 9000-obs median FoM3. DIAGNOSTIC basis (remixes the nulled kernels — not a practical recipe).

| arm | no-BNT | whitened | BNT | whiten/noBNT | recovered fraction* |
|---|---|---|---|---|---|
| L1 none | 2405 | 2524 | 364 | 1.05× | 106% |
| L1 product | 2875 | 2897 | 637 | 1.01× | 101% |

*recovered fraction = (whiten − BNT) / (noBNT − BNT) in FoM3.

**Verdict:** whitening recovers most of the collapse ⇒ the BNT inflation is DOMINANTLY the noise-correlation / per-map-S/N (basis) component; the irreducibly-joint share is small.