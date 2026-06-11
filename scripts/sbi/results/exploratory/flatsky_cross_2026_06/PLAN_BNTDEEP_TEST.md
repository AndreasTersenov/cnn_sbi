# PLAN — §5.4 one-extra-deep-channel test (GO from Andreas 2026-06-11)

## ADDENDUM — two-deep-channel test (GO from Andreas 2026-06-11, after the 0.730 result)

**Registered BEFORE any data:** 6-channel arm = 4 untouched BNT maps + the bin average +
the deepest bin κ₄ alone (mix mode `deep2`). The span-calibrated account (deep-dive §5.3/5.4)
predicts recovered_2 strictly between 0.73 and 1. Derived verdict ladder:
- recovered_2 ≤ 0.75: a second depth-distinct deep direction adds ~nothing — the span
  account is refuted at the margin (the residual 27% is NOT among-deep-kernel structure);
- 0.75 < recovered_2 < 0.95: monotone span increment — account supported;
- recovered_2 ≥ 0.95: two directions essentially span the usable signal-rich subspace.
Same concat machinery (BNT 800 cols bit-identical; 2-channel deep block, 400 cols; all
alignment hard-asserted); deep-block σ rows = √(Σⱼ Mᵢⱼ²σⱼ(s)²) from the verified no-BNT
table (exact: avg row = ¼√Σσⱼ², e₄ row = σ₄). Output: `bntdeep2_campaign/`.
NB the deep-channel ranges are recalibrated per arm (reservoir draws differ with 2 channels)
— each arm is self-consistent; binning parity caveat carried as in the 1-channel test.

**Pre-registered prediction** (BNT_THEORY_DEEP_DIVE.md §5.4, before any data): appending ONE
deep channel — the plain bin average (κ₁+κ₂+κ₃+κ₄)/4 of the original noisy demeaned autos —
to the four untouched BNT maps restores ≥ 0.8 of the L1's BNT loss:
recovered_deep = (FoM3_deep5 − FoM3_BNT)/(FoM3_noBNT − FoM3_BNT) ≥ 0.8, with
FoM3_noBNT = 2405, FoM3_BNT = 364 (pooled 3-MAF 9000-obs medians, l1 auto arm).
Verdict ladder (derived, not asserted): ≥0.8 prediction PASSES (deep-direction account
supported) / 0.4–0.8 partial (account incomplete) / <0.4 prediction REFUTED (joint share
hiding elsewhere). Either outcome is paper material. NB this is the MECHANISM test in the
uncut information-accounting setting — NOT a survey recipe (the deep channel would need
conservative cuts in a real analysis; deep-dive §1.7 item 2 caveat).

## Implementation: per-channel block concatenation (no 5-channel plumbing)

The L1 datavector is per-channel blocks by construction, so the 5-channel arm's datavector
EQUALS [cached BNT-auto blocks (800 cols) | fresh deep-channel block (200 cols)] — ceteris
paribus with the measured BNT arm by construction (identical BNT columns, identical ranges,
identical noise model; the ONLY change is the appended channel). Pieces:

1. `flatsky_cross.py`: mix mode `deep` = 1×4 matrix (¼,¼,¼,¼) + `n_built_channels()` helper
   (mode-aware channel count). `flatsky_cross_l1.calibrate_snr_range_flat_local` counts
   channels mode-aware.
2. `build_flatsky_bntdeep_arm.py` (new): deep-channel σ(s) DERIVED from the verified no-BNT
   frozen table — σ_deep(s) = ¼·√(Σⱼ σⱼ(s)²), exact by wavelet linearity + verified inter-bin
   noise independence (corr +0.0026, table GATE) — then per-channel SNR-range calibration
   (entry-script protocol: 3600 maps, train perms 5–6, q=0.5/99.5, margin 0.05, seed 0),
   train pass (split=train, perms 5–6, flip=True, seed=1001, batch 512, clamp=True) and val
   pass (split=test, perms 0–1, flip=False, seed=2001) — EXACTLY the BNT cache's build
   parameters so row order matches; **hard assertions: theta(train/val) bit-equal to the BNT
   cache's theta; fiducial perm/patch arrays bit-equal to fiducial_summaries_l1_none.npz**.
   Outputs: `bntdeep_campaign/l1_matrix/.../flat_local_none_bntdeep/{l1_train,l1_val}.npz`
   (x = [bnt 800 | deep 200]), ranges (5,2), meta; fiducial
   `bntdeep_campaign/fiducial_summaries/fiducial_summaries_l1_none.npz`
   (S = [S_bnt 800 | S_deep 200], 36000 rows).
3. `run_flatsky_bntdeep_campaign.py` (new): build (solo GPU) → jitted population sweep
   (3 MAF seeds, 9000 obs, m=2000; identical sweep flags to the whiten campaign) → derived
   `BNTDEEP_RESULT.md` (numbers read from median_summary.json files on disk).

Wall estimate from measured analogs (whiten campaign): build ≈ the loader-bound pass
(~50 min) + fiducial (~2 min) + sweep (~5–10 min). GPU: one of {0,1,2}, tenant-checked.

## Why this is clean

- The BNT 800 columns are bit-identical to the measured 0.15× arm — the comparison isolates
  the appended channel exactly.
- The deep σ is exact given the verified table (no new freeze; derivation documented in the
  build log).
- Alignment is asserted, not assumed (NaN-batch skipping makes row order parameter-dependent;
  identical parameters + bit-exact theta checks).
- All verdicts derived from artifacts on disk.
