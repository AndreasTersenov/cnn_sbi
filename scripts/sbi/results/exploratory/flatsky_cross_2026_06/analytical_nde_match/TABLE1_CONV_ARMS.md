# ℓ1+conv and ℓ1+conv+product — FoM₃ + marginals, both frames

Paper convention throughout. no-BNT rows quoted as SINGLE (compressor seed 41); BNT rows
quoted as ENSEMBLE (three pooled compressors). FoM₃ central = published value where one
exists, else the retrained quoted estimator. ± always = (central) × (relative std over
the three retrained compressor seeds); identical composition to every row in `TABLE1_PAPER.md`.

| Summary | frame | FoM₃ | σ(Ω_m) | σ(σ₈) | σ(w₀) | bias single→ens |
|---|---|--:|--:|--:|--:|--:|
| l1 +conv | no-BNT | **2671 ± 133** (5.0%) | 0.0518 ± 0.0011 | 0.0814 ± 0.0021 | 0.2366 ± 0.0030 | -9.5% |
| l1 +conv | BNT | **458 ± 15** (3.3%) | 0.0836 ± 0.0010 | 0.1643 ± 0.0021 | 0.3155 ± 0.0028 | -7.8% |
| l1 +conv+product | no-BNT | **3255 ± 200** (6.1%) | 0.0452 ± 0.0007 | 0.0714 ± 0.0014 | 0.2202 ± 0.0041 | -12.2% |
| l1 +conv+product | BNT | **704 ± 76** (10.8%) | 0.0755 ± 0.0024 | 0.1349 ± 0.0051 | 0.3030 ± 0.0077 | -7.6% |

**Notes.**  σ centrals track the row's quoted estimator (s41 for no-BNT, 3-pooled
ensemble for BNT) — no published σ triplets exist for these four rows to defer to.
The two BNT rows have no published FoM₃ counterpart either (the operator ladder was
no-BNT only), so both central and ± are from the retrained pipeline; flag these two
as new results in the caption.

## Backup detail (retrained, 3 compressor seeds each)

### l1 +conv, no-BNT
- per-seed FoM₃: 2720.2 / 2623.5 / 2463.0   → mean 2602.2, std 129.9 (4.99%)
- ensemble FoM₃: 2461.0   bias single→ens -9.53%
- σ(Ω_m): per-seed 0.0518 / 0.0513 / 0.0534   → ensemble 0.0532, spread 2.07%
- σ(σ₈): per-seed 0.0814 / 0.0807 / 0.0846   → ensemble 0.0835, spread 2.52%
- σ(w₀): per-seed 0.2366 / 0.2413 / 0.2355   → ensemble 0.2428, spread 1.28%
- central source: published

### l1 +conv, BNT
- per-seed FoM₃: 496.4 / 495.4 / 524.7   → mean 505.5, std 16.7 (3.30%)
- ensemble FoM₃: 457.6   bias single→ens -7.82%
- σ(Ω_m): per-seed 0.0820 / 0.0826 / 0.0807   → ensemble 0.0836, spread 1.21%
- σ(σ₈): per-seed 0.1605 / 0.1581 / 0.1565   → ensemble 0.1643, spread 1.26%
- σ(w₀): per-seed 0.3036 / 0.3091 / 0.3062   → ensemble 0.3155, spread 0.90%
- central source: new (retrained ensemble)

### l1 +conv+product, no-BNT
- per-seed FoM₃: 3564.9 / 3154.9 / 3414.8   → mean 3378.2, std 207.4 (6.14%)
- ensemble FoM₃: 3130.4   bias single→ens -12.19%
- σ(Ω_m): per-seed 0.0452 / 0.0463 / 0.0450   → ensemble 0.0467, spread 1.51%
- σ(σ₈): per-seed 0.0714 / 0.0741 / 0.0736   → ensemble 0.0746, spread 2.00%
- σ(w₀): per-seed 0.2202 / 0.2266 / 0.2190   → ensemble 0.2265, spread 1.84%
- central source: published

### l1 +conv+product, BNT
- per-seed FoM₃: 761.7 / 879.8 / 715.1   → mean 785.6, std 84.9 (10.81%)
- ensemble FoM₃: 703.9   bias single→ens -7.60%
- σ(Ω_m): per-seed 0.0737 / 0.0704 / 0.0750   → ensemble 0.0755, spread 3.24%
- σ(σ₈): per-seed 0.1307 / 0.1227 / 0.1314   → ensemble 0.1349, spread 3.79%
- σ(w₀): per-seed 0.2978 / 0.2904 / 0.3055   → ensemble 0.3030, spread 2.53%
- central source: new (retrained ensemble)

