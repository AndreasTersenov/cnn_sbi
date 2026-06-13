# VMIM compressor multi-seed robustness (Lane A check)

Question: is A1's FoM3 3822 robust to the VMIM compressor seed, and is its joint-coverage calibration (TARP net bias, + = conservative) stable? 3 NDE seeds per arm; compressor seed varied.

| comp. seed | FoM3 | σ(Ωm) | σ(σ8) | σ(w0) | TARP net bias |
|---|---|---|---|---|---|
| 41 | 3822 | 0.0421 | 0.0607 | 0.2161 | +0.021±0.028 |
| 42 | 3441 | 0.0417 | 0.0634 | 0.2184 | +0.009±0.022 |
| 43 | 3408 | 0.0455 | 0.0714 | 0.2239 | -0.022±0.015 |

**FoM3 across compressor seeds: 3557 ± 188 (min 3408, max 3822); spread 5%.** Baselines: l1+product 2875 (gate-C clean), pair2d 2794, l1-auto 2405.

Reading (derived): if the band stays well above l1+product 2875 AND the TARP net bias stays >= 0 (conservative) across seeds, the joint-PDF gain is robust and calibrated -> third pillar confirmed. A large spread or a seed dropping to ~2875 would mark it compressor-fragile (quote a band).

**ADJUDICATION (2026-06-13): band IS robustly > 2875 (3408-3822) BUT the calibration
condition FAILS — net bias +0.021/+0.009/-0.022 is seed-dependent, NOT reliably >= 0
(seed 43 mildly over-confident). Combined with the DPI argument + fiducial marginals
(tied-to-~10% over l1+product, not the ×1.33 FoM3), "third pillar" is NOT confirmed.
Full synthesis: LANE_A_CONCLUSION.md.**

## Leakage control (architecture argument + empirical)
- The VMIM compressor trains ONLY on the pair2d TRAIN split. TARP/SBC are measured on the held-out VAL split (never seen by the compressor); FoM3 on the independent fiducial sims. So calibration and constraining power are evaluated out-of-sample.
- Harmful overfitting would DEGRADE out-of-sample FoM3 (noisy features on unseen obs), not inflate it; a high held-out FoM3 + net-conservative TARP is evidence AGAINST harmful leakage.
- The multi-seed band above is the empirical robustness test; a stricter compressor-train / NDE-disjoint split is a possible follow-up if the band warrants.
