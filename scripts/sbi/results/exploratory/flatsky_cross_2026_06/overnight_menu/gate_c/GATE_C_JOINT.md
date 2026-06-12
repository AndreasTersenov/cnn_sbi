# GATE C — joint-statistic arms (derived verdicts)

Validates the overnight joint-stat posteriors (OVERNIGHT_RESULT.md addenda) per
PLAN_GATE_C_JOINT.md. Machinery = the BNT gate's TARP+SBC; L-C2ST skipped
(3000-dim, underpowered). NDE retrains mirror the full-rigor sweeps exactly
(log1p-zscore / clip 5 / min-var 1e-5 / MAF seeds 41,42,43 / epochs 50000 /
batch 256 / lr 1e-4). 600 val points, m=2000, FoM3-tercile stratified.

## SBC (ranks pooled from the TARP dumps; science params)
| arm | n | mean(Om,s8,w0) | std (uniform=0.289) | min KS p |
|---|---|---|---|---|
| pair2dq_nobnt | 1800 | 0.500,0.496,0.488 | 0.300,0.305,0.300 | 0.000 |
| jointl1q_nobnt | 1800 | 0.497,0.492,0.491 | 0.301,0.307,0.299 | 0.000 |
| pair2dq_bnt | 1800 | 0.493,0.501,0.485 | 0.307,0.309,0.298 | 0.000 |
| jointl1q_bnt | 1800 | 0.497,0.497,0.481 | 0.306,0.306,0.298 | 0.000 |

## TARP (dim-3 science subspace; signed max ECP − α, bootstrap-mean curve,
worst seed per FoM3 tercile; positive = conservative, negative = over-confident)
| arm | HIGH (tightest) | MID | LOW |
|---|---|---|---|
| pair2dq_nobnt | -0.134 | +0.040 | -0.051 |
| jointl1q_nobnt | -0.080 | -0.048 | +0.076 |
| pair2dq_bnt | -0.050 | +0.075 | -0.061 |
| jointl1q_bnt | -0.059 | +0.038 | -0.066 |

(no-BNT l1 reference from the flat-local gate: load-bearing arms |dev| <= 0.037;
dim-6 curves in gate_c/tarp_drp/curves/.)

## Verdicts (registered bands, PLAN_GATE_C_JOINT.md §5 — derived, not asserted)
Bands: PASS = all terciles |dev| <= 0.05 AND SBC std in [0.275, 0.305];
PASS-with-caveat = worst |dev| in (0.05, 0.1] or std outside by < 0.02;
FAIL = |dev| > 0.1 or std off by >= 0.02.

| arm | worst |dev| | SBC std excess | verdict |
|---|---|---|---|
| pair2dq_nobnt | 0.134 | 0.000 | **FAIL** |
| jointl1q_nobnt | 0.080 | 0.002 | **PASS-with-caveat** |
| pair2dq_bnt | 0.075 | 0.004 | **PASS-with-caveat** |
| jointl1q_bnt | 0.066 | 0.001 | **PASS-with-caveat** |

## Comparative check for the noBNT headline (registered sensitivity note)
The 'marginals equal-or-better than l1+product' claim rests on a ~4% sigma_s8 edge
over a comparator gated at |dev| <= 0.037. Downgrade trigger (derived): min signed
dev <= -0.05 (systematic over-confidence) OR any science-param SBC std >= 0.30.
- pair2dq_nobnt: DOWNGRADE to comparable (min signed dev -0.134, max std 0.305)
- jointl1q_nobnt: DOWNGRADE to comparable (min signed dev -0.080, max std 0.307)

## Registered-prediction adjudication
- P-G1 (noBNT arms land like the gated l1 noBNT arms, PASS clean): DOES NOT HOLD — see verdict table
- P-G2 (BNT-side arms PASS-with-caveat, worst |dev| in (0.05, 0.10]): HOLDS

Corners (pre-existing, morning session): overnight_menu/corners/ + figures/.

## Addendum — per-seed spread on the deciding terciles (derived from the curve npz files)

| arm | tercile | seed 41 | seed 42 | seed 43 |
|---|---|---|---|---|
| pair2dq_nobnt | HIGH | -0.108 | -0.134 | -0.092 |
| jointl1q_nobnt | HIGH | -0.074 | -0.080 | -0.075 |
| pair2dq_bnt | HIGH | -0.050 | -0.035 | +0.035 |
| jointl1q_bnt | HIGH | -0.059 | -0.044 | -0.040 |

Reading: the noBNT over-confidence in the tightest tercile is SEED-ROBUST (all three seeds
past -0.05 for both arms; pair2d all past -0.09) — not a worst-seed fluke. The failure mode
is exactly the registered sensitivity concern: apparent tightness in the highest-FoM3
posteriors is partly miscalibration. Magnitude: ECP deficit 0.09-0.13 at the worst alpha in
the tightest tercile + SBC std 0.300-0.307 (~4-6% global under-coverage) vs a claimed edge
of ~4% on sigma_s8 and +16% FoM3 (~5%/axis) — same order as the claim. The "equal-or-better
than l1+product" headline does NOT survive; "joint stats reach at least the l1-auto level
and are broadly comparable to l1+product, with mild over-confidence concentrated in their
tightest posteriors" is what the artifacts support.
