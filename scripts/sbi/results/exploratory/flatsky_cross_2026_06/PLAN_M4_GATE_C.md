# PLAN — M4 GATE C on the lane-B post-cut arms (+ second cut schedule decision)

**Date:** 2026-06-13. Continues paper message **M4** (make L1 work in BNT space via map
combinations). The overnight-2 lane-B arms have FoM3 but NO calibration; per the standing
rule, an uncalibrated FoM3 gain is not a result. This gate is the first half; the second cut
schedule (a physics choice for Andreas) is the second half.

## The arms (schedule M, already built; preproc log1p-zscore/clip5/min-var1e-5)
| arm | what | FoM3 | dims |
|---|---|---|---|
| B0_bntcut_l1 | cut BNT per-channel l1 (collapsed baseline) | 268 | 560 |
| B1_bntcut_sums | + 6 pairwise sums of cut channels | 596 | 1600 |
| B2_bntcut_deep2 | + 2 B⁻¹-reconstructed deep channels | 613 | 960 |
| B3_nobnt_unicut | noBNT under the uniform cut (the comparator) | 337 | 320 |
Headline (uncalibrated): B2/B3 = 1.82, B1/B3 = 1.77, B1≈B2 ("plain sums suffice").

## GATE C (this run, GPU 2)
`tarp_stratified_val.py` per arm (600 val pts, FoM3 terciles, 3 seeds) → `run_tarp_coverage`
(dim 3) → SBC from dumps → `GATE_C_LANEB.md` with DERIVED verdicts. Packed 2-at-a-time on
GPU 2 (mem 0.42, light foreign tenant). Same machinery as run_joint_gate_c / run_bnt_gate_c.

## Registered bands (BEFORE data)
- **P-B (the load-bearing one):** the rescue arms B1/B2 calibrate like the gated L1 arms —
  |net TARP bias| ≤ 0.05 AND SBC std ∈ [0.275, 0.305]. If so → the ~1.8× gain over the
  uniform-cut noBNT analysis is CALIBRATION-CLEAN and real ⇒ "BNT + cleaned linear
  recombinations of the kept cut channels beats the uniform-cut noBNT analysis" is a
  paper-ready M4 result (pending the 2nd-schedule robustness check).
- If B1/B2 are over-confident (net < −0.05 or SBC std > 0.305) → the 1.8× is partly
  inflated; downgrade to "comparable" and quote calibrated marginals only.
- B3 (comparator) must also be ~calibrated for the ratio to be meaningful; B0 gated for
  completeness (its calibration is moot — collapsed).
- Verdict per arm: PASS (|dev| ≤ 0.05, SBC std in band) / PASS-with-caveat (worst |dev|
  ∈ (0.05, 0.10]) / FAIL (> 0.10 or SBC std off ≥ 0.02), as in the joint/BNT gates.

## Second cut schedule — DECISION FOR ANDREAS (the physics choice; not run here)
The 1.8× is schedule-conditional (one toy schedule M). To make M4 robust we need ≥1 more.
Options for the second schedule (keep-masks per BNT channel shallow→deep, scale 0=finest):
- **M′ (aggressive):** schedule M + DROP κ̃₁ entirely (the contaminated shallow map) — tests
  whether the rescue survives when a whole channel is cut, the most survey-realistic case.
- **L (light):** cut only the finest scale of κ̃₁/κ̃₂ — a gentler systematics scenario.
- **Your own keep-masks.**
Recommendation: **M′** — it's the survey-relevant stress test and the most likely referee
ask. On go I build the masked caches + recombination arms for the 2nd schedule and gate them
the same way; M4 is then "robust across {M, M′}" or honestly "schedule-dependent".

## Folds (post-gate)
GATE_C_LANEB.md (derived); fold verdict into PAPER_MESSAGES.md M4 + a felt stanza; commit by
path (report + coverage figs; never dumps/.npz).
