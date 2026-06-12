# Overnight menu — derived results (PLAN_OVERNIGHT_MENU.md)

Screening = 1 MAF seed, 3000 obs; full = 3 seeds, 9000 obs. Baselines: L1 noBNT auto 2405, L1 BNT auto 364 (pooled medians).

| arm | kind | screening FoM3 | recovered* | full FoM3 | full recovered |
|---|---|---|---|---|---|
| A1_cov_bnt | rescue | 1194 | 0.407 | 1134 | 0.377 |
| A2_unions6_bnt | rescue | 2931 | 1.258 | 2768 | 1.178 |
| pair2d_nobnt | joint | 2842 | 1.214 | 2803 | 1.195 |
| pair2d_bnt | joint | 1590 | 0.600 | FAIL | — |
| full4d_nobnt | joint | FAIL | — | — | — |
| full4d_bnt | joint | FAIL | — | — | — |
| jointl1_nobnt | joint | 2864 | 1.225 | 2795 | 1.191 |
| jointl1_bnt | joint | 1644 | 0.627 | FAIL | — |

*recovered = (arm − BNT)/(noBNT − BNT); for joint/nobnt arms read it as FoM3 relative positioning, the meaningful number is FoM3 itself vs 2405.

- **pair2d basis-invariance ratio (BNT/noBNT, screen): 0.559** (P4b predicts ≈1 for full4d only)
- **jointl1 basis-invariance ratio (BNT/noBNT, screen): 0.574** (P4b predicts ≈1 for full4d only)

Registered readings: A1 recovered = Gaussian-sector share of the l1's loss; A2 ≥0.95 expected (span); full4d ratio ≈1 is the exact-covariance test; pair2d/jointl1 ratios measure the pairwise approximation's basis fragility.

NaN/failure notes and logs: overnight_menu/logs/.

## ADDENDUM — full4d retry (K=4, dequantized; the K=5 arms NaN'd the MAF on quasi-discrete sparse cells)

| arm | screening FoM3 | full FoM3 |
|---|---|---|
| full4dq_nobnt | 2442 | 2401 |
| full4dq_bnt | 1123 | 1078 |
| jointl1q_nobnt | 2773 | 2788 |
| jointl1q_bnt | 1557 | 1517 |
| pair2dq_nobnt | 2788 | 2794 |
| pair2dq_bnt | 1520 | 1460 |

**full4dq basis-invariance ratio (BNT/noBNT, full): 0.449** (P4b predicts ≈1)

**jointl1q basis-invariance ratio (BNT/noBNT, full): 0.544**

**pair2dq basis-invariance ratio (BNT/noBNT, full): 0.522**

## ADDENDUM — full4d retry (K=4, dequantized; the K=5 arms NaN'd the MAF on quasi-discrete sparse cells)

| arm | screening FoM3 | full FoM3 |
|---|---|---|
| full4da_nobnt | 2101 | 2085 |
| full4da_bnt | 1458 | 1455 |

**full4da basis-invariance ratio (BNT/noBNT, full): 0.698** (grid-transport test: ≥0.75 supports, ≤0.55 refutes)

## Night synthesis (2026-06-12 00:05, written after all runs; numbers from the tables above)

**Full-rigor headline numbers** (3 MAF seeds, 9000 obs; baselines L1-auto 2405 / L1+product
2875 / L1-BNT 364):

| statistic | noBNT | BNT | ratio | note |
|---|---|---|---|---|
| pair2d joint PDF (K=10, dq) | 2794 | 1460 | 0.52 | +16% over l1-auto; ≈ l1+product |
| joint wavelet l1 (K=10, dq) | 2788 | 1517 | 0.54 | ≈ pair2d: weighting adds nothing |
| full4d joint PDF (K=4, dq, fixed grid) | 2401 | 1078 | 0.45 | = l1-auto; resolution-limited |
| full4d, adaptive percentile grid | 2085 | 1455 | 0.70 | the grid-transport test |
| A2: BNT-L1 + 6 union channels | — | 2768 | rec 1.18 | survey practice = full rescue |
| A1: BNT-L1 + 50 wavelet (co)variances | — | 1134 | rec 0.38 | Gaussian share of the loss |

**Conclusions (screening-grade where only screened; full-rigor as marked):**
1. *Joint one-point statistics work.* The pairwise joint PDF and the joint wavelet l1 both
   beat the marginal l1 by ~16% on autos and land at the l1+product level — consistent with
   the theory (the pairwise joint contains the product's one-point information). Plain
   counts suffice; l1-weighting is informationally idle here.
2. *Rescue menu closed.* Survey-practice unions fully rescue the BNT l1 (1.18, matching the
   deep/deep2 span results); the P7 second-moment block measures the Gaussian share of the
   l1's BNT loss at 38% — i.e. ~62% of what per-channel analysis loses under nulling is
   non-Gaussian content.
3. *"BNT-robust by construction" needs a qualifier.* P4b covariance holds for the
   DISTRIBUTION; a binned estimator is only as invariant as its grid is transported. Fixed
   noise-scaled grid: ratio 0.45. Axis-adapted (percentile) grid: 0.70. The residual is the
   SHEAR part of the cell transport (B-images of cells are parallelepipeds), which no
   axis-aligned histogram can represent — the learned compressor implements exactly this
   shear in its first layer. Registered bands (≥0.75 / ≤0.55): landed between, toward
   support; "grid-transport explains the majority, axis-aligned binning caps the rest."
4. *NDE practicalities:* count histograms need dequantization (quasi-discrete sparse cells
   NaN the MAF — three arms diagnosed and fixed); no dimensionality limitation observed up
   to 3200 dims ⇒ the VMIM compression step was NOT warranted (per the pre-registered
   last-if-warranted rule) and was not run.

Not run (judged diminishing-returns; cheap if wanted): pair2da (decompose the pairwise 0.52
into grid vs incompleteness); K=15 pair2d resolution scaling; VMIM-compressed variants.

## GATE C (2026-06-12; `gate_c/GATE_C_JOINT.md`, bands registered in `../PLAN_GATE_C_JOINT.md`)

TARP (600 val pts, FoM3 terciles, 3 seeds) + SBC (n=1800) on the four q-arms:

| arm | worst tercile dev (dim-3) | SBC std (sci params) | verdict |
|---|---|---|---|
| pair2dq_nobnt | **−0.134 HIGH** (seed-robust −0.092/−0.108/−0.134) | 0.300–0.305 | **FAIL** |
| jointl1q_nobnt | −0.080 HIGH (seed-robust) | 0.299–0.307 | PASS-with-caveat |
| pair2dq_bnt | +0.075 MID | 0.298–0.309 | PASS-with-caveat |
| jointl1q_bnt | −0.066 LOW | 0.298–0.306 | PASS-with-caveat |

**Consequence for conclusion 1 above (registered comparative check — TRIGGERED):** both
noBNT joint arms are mildly over-confident, concentrated in their TIGHTEST posteriors
(~4–6% global under-coverage by SBC; ECP deficit up to 0.13 in the HIGH tercile), while the
l1/l1+product comparators were gated at |dev| ≤ 0.037. The over-confidence is the same
order as the claimed edge (σ_s8 0.072 vs 0.075; +16% FoM3 ≈ 5%/axis). **Downgrade:**
"marginals equal-or-better than l1+product" → "reach at least the l1-auto level and are
broadly comparable to l1+product"; the +16%-over-auto FoM3 edge is partly calibration,
quote only with this caveat. The directional theory point (joint occupancy carries the
product's one-point information) stands; the quantitative parity claim is not
calibration-clean. Basis-invariance RATIOS (0.52/0.54): both numerator and denominator err
in the same direction (over-confident), so the ratios are less affected, but pair2d's uses
a FAIL-grade denominator — carry the flag.
