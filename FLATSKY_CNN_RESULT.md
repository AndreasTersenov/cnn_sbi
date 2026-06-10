# Flat-sky (patch-local) cross — de-leaked L1-vs-CNN (2026-06-09)

**Pooled 3-seed 9000-obs median, common jaxili MAF (NDE confound removed).** CNN-VMIM compressor on the same de-leaked patch-local cross as the L1 side (`FLATSKY_CROSS_RESULT.md`). Calibration: GATE C section below + cnn_phase/gate_c.

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

## Robustness — best single (MAF) seed, un-pooled
Pooling 3 MAF seeds applies a haircut, so the best single seed is the CNN at its most favorable. Reloaded the trained MAF checkpoints, sampled each seed at the typical obs. **The no-cross-gain survives un-pooling** — every cross arm stays ≤ auto-only, so it is not a pool-haircut artifact. (MAF seeds, not compressor seeds; one compressor.)

| arm | s41 | s42 | s43 | best | best vs-auto |
|---|---|---|---|---|---|
| auto-only | 2620 | 2364 | 2387 | **2620** (s41) | 1.00× |
| +conv | 2418 | 1968 | 2491 | **2491** (s43) | 0.95× |
| +product | 2225 | 2331 | 2017 | **2331** (s42) | 0.89× |
| +both | 2475 | 2436 | 2205 | **2475** (s41) | 0.94× |

Figures: `cnn_phase/best_seed/` (FoM3 bars, per-arm CNN-best-seed vs L1-pooled overlays). Caveat: best-vs-L1-*pooled* is best-vs-haircut, not best-vs-best (L1's 2000-d datavector can't be reloaded per-seed); the robust claim is the within-CNN no-gain.


## GATE C — calibration
Full interpretation: **`cnn_phase/gate_c/GATE_C_INTERPRETATION.md`** (TARP/SBC verdicts documented there from `gate_c/{tarp_drp,sbc}/`). The L-C2ST verdicts below are derived from the per-arm summaries.

| arm | L-C2ST reject@p<0.05 | median p | verdict |
|---|---|---|---|
| auto-only | 0% | 0.27 | calibrated |
| +conv | 60% | 0.05 | MIScalibrated |
| +product | 0% | 0.22 | calibrated |
| +both | 13% | 0.12 | mild |

L-C2ST self-test (power gate): ST_H0 p=0.71 (no false alarm), ST_H1 p=0.02 (planted 0.5σ w0 DETECTED) ⇒ the test has power, so the calibrated verdicts are real (unlike high-dim L1).
Figures: `gate_c/{tarp_drp/figures, sbc, lc2st}/`.
