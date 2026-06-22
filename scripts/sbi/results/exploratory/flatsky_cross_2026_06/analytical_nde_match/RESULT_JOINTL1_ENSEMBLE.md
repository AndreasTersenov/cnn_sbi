# Joint ℓ1 calibration via a compressor ensemble — the properly-calibrated result (2026-06-22)

The single-compressor joint ℓ1 was mildly over-confident (noBNT SBC ~0.31, caveat) and frankly
over-confident under BNT (SBC ~0.34, gate FAIL). This documents the fix and the corrected numbers.

## What did NOT fix it: shear-aware (rotated-grid) binning
Per-(pair,scale) 2-D PCA-whitened binning (`build_flatsky_joint_arm.py --rotated-binning`) —
rotating each pair's grid onto its BNT covariance eigen-axes — left the BNT over-confidence
UNCHANGED (SBC 0.338/0.343, dev 0.115; vs axis-aligned 0.333/0.335, dev 0.110). It DID raise the
noBNT FoM3 (3754→4565, still caveat) by extracting more pairwise structure, but did not transfer to
BNT. Conclusion: the over-confidence is NOT the binning shear — a pairwise grid can't follow BNT's
full-4-D cross-channel rotation anyway. (RESULT_JOINTL1_ROTATED.md.)

## What fixed it: compressor deep-ensemble (principled, non-conformal)
The over-confidence has the fingerprints of amortized-SBI leakage (flat across flow capacity —
calib_sweep_jointl1/SWEEP_RESULT.md; present in noBNT too; VMIM + RealNVP train on the same maps).
Pooling the posteriors of **3 compressor-seed arms** (41/42/43) per obs — a deep ensemble over
compressors — diversifies the 10-D summary and washes the over-confidence out. Gate dumps pooled
per-obs (the gate's own pooling), TARP-DRP + SBC:

| basis | arm | TARP worst | net | SBC std (Om/s8/w0) | gate |
|---|---|---|---|---|---|
| noBNT | single (s41) | +0.028 | −0.004 | 0.312/0.315/0.305 | caveat |
| noBNT | **ensemble (×3)** | +0.016 | +0.007 | **0.299/0.298/0.298** | **clean PASS** |
| BNT | single (s41) | +0.110 | — | 0.333/0.335/0.313 | **FAIL** |
| BNT | **ensemble (×3)** | +0.019 | +0.003 | **0.304/0.304/0.298** | **clean PASS** |

Both bases clean-PASS after the ensemble. (run_jointl1_bnt_ensemble.py; the noBNT seeds are from the
seed-check; per-obs ensemble TARP curve in plot_calibration_ensemble.py.)

## Properly-calibrated FoM3 (ensemble per-obs over the 9000-fiducial population)
`ensemble_eval.py` pools the 3 compressor flows (×3 NDE seeds) per physical obs and recomputes FoM3:

| arm | noBNT FoM3 | BNT FoM3 | BNT retention | calibration |
|---|---|---|---|---|
| ℓ1+product | 3045 | 779 | **0.26** | caveat |
| **joint ℓ1 (ensemble)** | **3371** (σ 0.044/0.072/0.223) | **2424** (σ 0.051/0.086/0.235) | **0.72** | **clean PASS (both)** |
| CNN ResNet18 | 3326 | 3186 | 0.96 | PASS |

## Headlines (calibrated, corrected)
- **Q1:** the properly-calibrated analytical joint ℓ1 (**3371**) is a **calibrated TIE with the CNN
  (3326)**, σ matched — *no caveat*. The single-arm 3754 was ~10% inflated by over-confidence
  (removing it gives 3371, dead-on the CNN — physically sensible since the CNN is ~optimal).
- **Q2:** under BNT the joint ℓ1 retains **0.72 (calibrated)** vs ℓ1+product's **0.26** — ~3× more of
  the surviving cross-correlation — but below the CNN's **0.96**. The remaining 0.72→0.96 gap is the
  full-4-D channel mixing the CNN learns and a fixed (pairwise) analytical statistic cannot reach.
  **NB the calibrated retention is 0.72, not the single-arm raw 0.86** (the over-confidence, larger in
  BNT, inflated the raw ratio; both arms calibrated gives 0.72).

Figures: violins_ensemble_3arm, violin_fom3_ensemble_3arm, tarp/sbc_pooled_ensemble_3arm,
contour_ensemble_{nobnt,bnt}_3arm. Supersedes the single-arm calibration framing in
RESULT_JOINT_MATCHED.md (FoM3 3754 / retention 0.86).
