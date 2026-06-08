# Phase C — 4-arm × 3-seed (10°), 3-seed-pooled @ obs patch 90

Lead metrics: **σ(w0), 2D(Ωm,σ8)** (FoM3 reported, NOT headlined — fragile).

| arm | n_seed | σ(Ωm) | σ(σ8) | σ(w0) | 2D(Ωm,σ8) | FoM3 | max\|pull\| |
|---|---|---|---|---|---|---|---|
| l1_auto_cross | 3 | 0.0379 | 0.0770 | 0.1600 | 1153 | 10689 | 1.05σ |
| cnn_auto_cross | 3 | 0.0301 | 0.0455 | 0.1567 | 2172 | 22163 | 0.61σ |
| l1_auto_only | 3 | 0.0521 | 0.0873 | 0.2321 | 403 | 2201 | 1.27σ |
| cnn_auto_only | 3 | 0.0397 | 0.0732 | 0.2215 | 578 | 3481 | 1.04σ |

## L1-vs-CNN ratios (CNN/L1 for σ → >1 means L1 tighter)
- **auto_cross**: σ(w0) L1 0.160 vs CNN 0.157 (×0.98); σ(Ωm) ×0.80; 2D ×0.53; FoM3 ×0.48
- **auto_only**: σ(w0) L1 0.232 vs CNN 0.221 (×0.95); σ(Ωm) ×0.76; 2D ×0.70; FoM3 ×0.63

## vs 20° (typical patch, for Phase E)
- 20° L1 a+c: σ(w0) 0.125, 2D 3343, FoM3 53069 | CNN a+c: σ(w0) 0.167, 2D 2085, FoM3 24453
- 20° L1/CNN a+c: σ(w0) ×1.34, 2D ×1.60, FoM3 ×2.17
