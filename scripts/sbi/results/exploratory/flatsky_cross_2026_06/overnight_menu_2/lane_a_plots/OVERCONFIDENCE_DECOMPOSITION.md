# Lane A — is FoM3 3822 real information or over-confidence?

Per-param width-inflation k_i inferred from the SBC rank std (rank = Φ(N(0,k²))); 
FoM3 inflation = Π k_i (FoM3 ∝ 1/Π σ_i, correlations preserved); 
FoM3_calibrated = FoM3_obs / Π k_i.

CAVEAT: SBC std also absorbs bias / non-Gaussianity, so Π k_i is an UPPER bound on width over-confidence and FoM3_calibrated a LOWER bound on true calibrated power.

| arm | FoM3_obs | SBC std (Om,s8,w0) | k (Om,s8,w0) | Π k | FoM3_calibrated |
|---|---|---|---|---|---|
| A1 VMIM 10-d (3822) | 3822 | 0.317,0.325,0.297 | 1.20,1.26,1.06 | 1.61 | 2381 |
| A2 pair2d K=8 (2874) | 2874 | 0.301,0.301,0.298 | 1.08,1.08,1.06 | 1.23 | 2328 |
| pair2d K=10 raw (2794) | 2794 | 0.300,0.305,0.300 | 1.08,1.11,1.08 | 1.28 | 2182 |

Calibrated baselines for comparison: l1+product 2875 (gate-C clean, |dev|≤0.037), l1-auto 2405.

Reading (derived): compare FoM3_calibrated to 2875. If above, real information survives the over-confidence correction; if ≈ or below, the boost is mostly over-confidence.

---

## CORRECTED READING (the SBC-deflation above OVER-corrects — TARP is the right arbiter)

The table above used the per-param SBC rank std as if its excess over 0.289 were all WIDTH
over-confidence, and deflated FoM3 by Πk. That is wrong for two measured reasons, and the
corrected conclusion flips toward "the boost is largely real":

**1. The joint metric that actually governs FoM3 says A1 is the BEST-calibrated arm.**
Net signed TARP-DRP bias (dim-3, pooled seeds+terciles; + = conservative/over-covers):
- A1 VMIM 10-d : **+0.021** (slightly conservative)
- A2 pair2d K=8 : −0.007 (≈calibrated)
- pair2d K=10 raw : **−0.044** (over-confident — the worst)
FoM3 is a joint-VOLUME quantity, so TARP-DRP (joint coverage), not the per-param SBC std,
is the correct calibration arbiter for it. On that metric A1's 3822 is NOT inflated by joint
over-confidence; if anything A1 mildly UNDER-states its precision ⇒ 3822 is a defensible,
possibly conservative number.

**2. A1's high SBC std is marginal NON-GAUSSIANITY, not over-tight width.** Excess kurtosis
of the science marginals (0 = Gaussian; negative = platykurtic / prior-bounded):
A1 −0.82, A2 −0.62, pair2d −0.56. A1's marginals are the MOST non-Gaussian (flat-topped,
prior-bounded). The SBC rank-std metric is calibrated to a Gaussian-uniform expectation
(0.289), so it misreads platykurtic shape as dispersion and inflates k. The Πk deflation
therefore over-corrects (it would also wrongly deflate the gate-C-clean pair2d to ~2180).

**Corrected conclusion:** Lane A flips toward the third-pillar hypothesis. VMIM compression
of the joint statistic appears to extract genuine extra information (FoM3 3822 vs pair2d
2794, l1+product 2875) AND calibrate the joint volume at least as well as — better than —
the raw histogram. The overnight one-line verdict ("pathology-not-dimensionality") and the
Πk table above are SUPERSEDED by this joint-metric reading.

**Two checks before banking 3822 (NOT yet done):**
- (a) Multi-seed the VMIM compressor. This is a single compressor seed (41); our own CNN
  work shows single-compressor-seed FoM3 is fragile (feedback: never quote single-seed
  cross-gains). 3 compressor seeds would tell us the ±band on 3822.
- (b) Rule out compressor↔NDE leakage. The compressor trained on the pair2d TRAIN split;
  confirm the FoM3 isn't inflated by the compressor learning features the same-split NDE
  then exploits (best-val at step 7000 suggests limited overfit, but verify with a
  compressor-train / NDE-disjoint split or a fresh-sim fiducial check — the latter already
  holds: fiducial obs are independent sims).
Caveat on significance: the TARP gap A1(+0.021) vs pair2d(−0.044) is ~1 per-curve σ; the
direction is consistent across all 9 curves but not high-significance on its own.
