# Getting a *directly* calibrated CNN posterior — principled routes (research note)

**Date:** 2026-06-23. **Goal:** remove the CNN's residual mild over-coverage (un-stratified TARP net
+0.033 ± 0.020; ~1.6σ) **by a principled method**, not a hand-tuned recalibration, so we can state
"the optimal CNN is calibrated and ≥ the best analytical L1 statistic" directly rather than via a
matched-calibration argument. Diagnosis (see `FINDING_CALIBRATION_DECOMPOSITION.md`): marginals are
calibrated (SBC 0.290/0.289/0.282), the over-coverage is in the **joint** (correlation under-tightening
by the affine RealNVP); the implied calibrated FoM3 is ~3700–3850 (≥ joint-ℓ1 3371).

## Target (pre-registered, to avoid overshoot)
Calibrated = TARP net within ±0.02 of 0 (i.e. ≤ ~1σ) **and** SBC rank-std ∈ [0.282, 0.296] on all
three params, on a **held-out** gate. Hard guard: never accept SBC > 0.305 or net < −0.02
(over-confident is worse than conservative for a cosmology result). Report FoM3 of the *same*
recalibrated/retrained posterior we gate. FoM3 is fragile, so the headline claim stays "calibrated and
≥ L1 / tie within tolerance," not a horse-race number.

## Wrong-direction methods (ruled out, for the record)
Balanced NRE/NPE (Delaunoy et al., arXiv:2208.13624, 2304.10978) and conservative/distributionally-
robust NPE (arXiv:2605.28516) all *increase* conservativeness to fix over-confidence. We are already
conservative → these move us the wrong way. Do not use.

## Three right-direction routes, ranked for this paper

### Route A (try first — cheapest, most elegant): a more expressive NDE (Neural Spline Flow)
The over-coverage is most consistent with the **affine** RealNVP coupling under-modelling the
posterior's correlations/tails (affine maps → a too-round joint). A **rational-quadratic Neural Spline
Flow** (Durkan et al. 2019) is strictly more expressive and the modern SBI default; deeper / more
coupling blocks are the same lever. If the over-coverage is an expressivity artifact, a better-specified
flow calibrates the joint **by construction**, with a likely FoM3 gain.
- Paper story: "we used a properly-expressive estimator" — fits the existing estimator-optimisation
  narrative and the referee-defense goal exactly.
- Effort: low — a flow swap on the frozen resnet18 summaries; gate each (TARP+SBC, proper bands).
- Risk / honest caveat: not guaranteed. Prior evidence that flow family matters a lot here (jaxili MAF
  on the same summary gave FoM3 2312 vs RealNVP 3139), so "more expressive" ≠ automatically calibrated
  *and* sharper. NSF couplings differ from MAF (flexible monotonic transforms, better tails), so it is
  worth testing, but the sweep — not a guarantee — decides. Candidates: NSF (rq-spline) couplings;
  RealNVP with 8/12 coupling blocks; spline-MAF.

### Route B (most direct "by construction"): calibration-aware training
Add a **differentiable coverage-probability term** to the NDE training loss so the flow is calibrated
during training while preserving sharpness ("Calibrating Neural SBI with Differentiable Coverage
Probability", arXiv:2310.13402, NeurIPS 2023). It relaxes the coverage/calibration error into a
differentiable penalty, is model-agnostic, and the paper optimises **coverage *and* expected posterior
density** jointly (so it is meant to calibrate without throwing away sharpness). It targets exactly the
diagnostic we gate on.
- Paper story: "the NDE is trained with a calibration objective and is calibrated by construction."
- Effort: medium — implement the differentiable coverage term in the sbi_lens RealNVP training loop
  (or jaxili). Directionality: framed for over-confidence, but the penalty is a (symmetric) deviation-
  from-diagonal error, so it should pull our over-coverage down too — verify on a screen.

### Route C (guaranteed backstop / robustness check): post-hoc conformal calibration
**CP4SBI** ("Local Conformal Calibration of Credible Sets in SBI", arXiv:2508.17077, 2025): a post-hoc,
model-agnostic conformal procedure that takes the trained flow's samples and produces credible sets
with **finite-sample coverage guarantees** (LoCart variant = local finite-sample; CDF variant =
asymptotic conditional). For an over-covering posterior it *tightens* the calibrated HPD region →
sharper, calibrated, **guaranteed** — directly converting our +0.033 into a calibrated (and tighter)
result, fit on a held-out calibration split and validated on a disjoint one.
- Why it is NOT "by hand": it is a principled, citable algorithm with coverage guarantees, not a manual
  temperature knob.
- Effort: medium — implement HPD-conformal scoring; define FoM3 on the calibrated credible set
  (e.g. covariance of the conformally-reweighted/trimmed sample) consistently with the other arms.
- Best used as the **rigour layer**: even if Route A/B gives a calibrated estimator, reporting a
  conformal coverage guarantee on top is a strong referee-defense.

## Recommended staged plan
1. **Route A sweep** on frozen resnet18 summaries (NSF + deeper RealNVP), 3 NDE seeds, gate each with
   the proper-band TARP+SBC. Accept the calibrated variant with highest FoM3. Likely lands net≈0,
   FoM3 ~3.6–3.8k ≥ 3371. *If it works, this alone supports the claim.*
2. **If A leaves residual miscalibration** → **Route B** (coverage-calibrated training).
3. **Route C (CP4SBI)** as a coverage-guarantee robustness check on the chosen estimator, regardless —
   it is the strongest single "calibrated by a principled method with guarantees" statement.

All stages: vary one factor, pre-register the predicted direction, gate on a held-out split, never
overshoot into over-confidence, recompute FoM3 on the gated posterior. Honest fallback if none cleanly
beats the guard: keep the current conservative CNN (FoM3 = lower bound) and report "tie within
calibration tolerance" — already true and publishable.

## Sources
- CP4SBI — Local Conformal Calibration of Credible Sets in SBI: arXiv:2508.17077
- Calibrating Neural SBI with Differentiable Coverage Probability (NeurIPS 2023): arXiv:2310.13402
- Balanced NRE / Balancing SBI for Conservative Posteriors (Delaunoy et al.): arXiv:2208.13624, 2304.10978
- Conservative NPE via distributionally robust training: arXiv:2605.28516
- Neural Spline Flows (Durkan et al. 2019, NeurIPS)
- TARP expected-coverage diagnostic (Lemos et al. 2023) — already the project gate
