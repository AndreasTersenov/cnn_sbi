# Lane-B M-vs-L robustness (PLAN_M4_GATE_C.md 2nd schedule = L, light/realistic)

Does the post-cut rescue (cut-BNT + recombinations beats the uniform-cut noBNT analysis) hold across cut schedules? M = moderate (shallow channel keeps 2 coarsest scales), L = light (shallow channels drop only their finest scale).

| arm | M FoM3 | L FoM3 | L net-bias | L SBC std | L verdict |
|---|---|---|---|---|---|
| B0L_bntcut_l1 | 268 | 350 | +0.037±0.019 | 0.303,0.299,0.294 | **PASS** |
| B1L_bntcut_sums | 596 | 1286 | -0.011±0.042 | 0.302,0.300,0.292 | **PASS** |
| B2L_bntcut_deep2 | 613 | 2007 | +0.010±0.019 | 0.304,0.305,0.287 | **PASS** |
| B3L_nobnt_unicut | 337 | 1880 | -0.002±0.052 | 0.295,0.299,0.286 | **PASS** |

**L ratios: B1L/B3L = 0.68, B2L/B3L = 1.07** (M was 1.77 / 1.82). Robust-across-schedules if the L ratios are also clearly > 1 with B1L/B2L calibrated; schedule-dependent otherwise.

---

## Combined M + L verdict (2026-06-13) — derived

**Calibration:** all four L arms PASS clean (net bias |.| <= 0.04, SBC std 0.286-0.305);
M rescue arms B1/B2 PASS-with-caveat. B3-M (uniform-cut noBNT, schedule M) gate is
INCOMPLETE — its NDE sampled all-NaN at prior-edge val points (the near-prior-flat,
heavily-cut regime where the MAF tails diverge); its FoM3 337 is a valid point estimate
(population sweep, n=8997 finite at fiducial), and B3L's clean PASS + the precedent that
noBNT-l1 arms gate clean make 337 reasonably calibrated-by-inference. So the gain is NOT
over-confidence (contrast A1) — the FoM3 numbers are trustworthy.

**Magnitude is strongly schedule-dependent** (FoM3 ratios vs the uniform-cut noBNT analysis):
| arm | M (moderate) | L (light, realistic) |
|---|---|---|
| B0 cut-BNT per-channel l1 | 0.79x | 0.19x |
| B1 + pairwise sums | 1.77x | 0.68x |
| B2 + reconstructed-deep | 1.82x | 1.07x |

**Reading:**
1. Per-channel L1 in cut-BNT space COLLAPSES (B0L = 0.19x the uniform-cut noBNT analysis).
2. Adding two reconstructed-deep channels (cut-then-mix; preserves the per-slice cuts)
   RESCUES it to parity-plus (B2L = 1.07x), calibration-clean. The rescue is real: 350 -> 2007.
3. But the ADVANTAGE over a standard uniform-cut analysis is schedule-dependent: ~1.8x for
   the aggressive cut (which craters the uniform analysis to 337), only ~7% for the realistic
   light cut (where the uniform analysis keeps 1880 ~= 78% of uncut 2405, leaving little to
   win back). So the strong "1.8x" does NOT generalize.
4. "Plain sums suffice" was a moderate-cut artifact: under L, reconstructed-deep (2007) >>
   sums (1286, BELOW the uniform analysis). If the rescue matters, the RECONSTRUCTION matters.

**M4 paper message (honest):** recombination of the cut BNT channels (specifically the
B^-1-reconstructed deep directions) rescues the cut-BNT L1 from catastrophic collapse and
restores it to ~the standard uniform-cut analysis level (within ~10%) WHILE preserving BNT's
clean per-slice systematics control — the value of BNT is the clean cuts, not a raw FoM3 win.
Calibration-clean; schedule-robust in DIRECTION (rescue works) though not in MAGNITUDE
(advantage shrinks from 1.8x to ~1.07x as the cut becomes realistic).
