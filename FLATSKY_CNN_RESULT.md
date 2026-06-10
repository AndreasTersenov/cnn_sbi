# Flat-sky (patch-local) cross — de-leaked L1-vs-CNN (2026-06-09)

**Pooled 3-seed 9000-obs median, common jaxili MAF (NDE confound removed).** CNN-VMIM compressor on the same de-leaked patch-local cross as the L1 side (`FLATSKY_CROSS_RESULT.md`). Calibrated: TARP ✓ / SBC / L-C2ST (see cnn_phase/gate_c).

## FoM3 (pooled median)

| arm | CNN FoM3 | CNN vs auto | L1 FoM3 | L1 vs auto | CNN/L1 |
|---|---|---|---|---|---|
| auto-only | 2325 | 1.00× | 2405 | 1.00× | 0.97× |
| +conv | 2192 | 0.94× | 2499 | 1.04× | 0.88× |
| +product | 2181 | 0.94× | 2875 | 1.20× | 0.76× |
| +both | 2306 | 0.99× | 2910 | 1.21× | 0.79× |

*Full-sphere (leaky) reference: L1 auto 2200, L1 auto+cross 8530 (3.88×), CNN auto+cross ~17251 (~7.4×). The leaky CNN crushed L1; de-leaked they should be comparable.*

## Marginal sigma + 2D(Om,s8) (pooled median)

| arm | CNN sig(Om,s8,w0) | CNN 2D(Om,s8) | L1 sig(Om,s8,w0) | L1 2D(Om,s8) |
|---|---|---|---|---|
| auto-only | 0.051,0.077,0.244 | 447 | 0.053,0.082,0.245 | 471 |
| +conv | 0.051,0.081,0.244 | 425 | 0.052,0.081,0.245 | 484 |
| +product | 0.053,0.085,0.244 | 421 | 0.048,0.075,0.238 | 522 |
| +both | 0.051,0.080,0.247 | 444 | 0.046,0.075,0.232 | 528 |

## GATE C
- TARP-DRP: see `cnn_phase/gate_c/tarp_drp/` (per-arm/tercile).
- SBC: see `cnn_phase/gate_c/sbc/sbc_summary.json`.
- L-C2ST: see `cnn_phase/gate_c/lc2st/` (works for CNN 10-dim).
