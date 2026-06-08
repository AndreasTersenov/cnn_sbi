# OFFSET VERDICT — does L1's w0 bias shrink at 10°? → YES (flat-sky artifact)

The headline question of the whole 10° campaign. At 20°, L1 carried a coherent fiducial
w0 offset of **−0.37σ** (Ωm −0.27σ, σ8 +0.19σ) that CNN lacked — interpreted (unproven) as
possibly intrinsic to the ℓ₁ statistic. The 10° patches have far better flat-sky validity
(gnomonic corner distortion 6.3%→1.5%), so they are the decisive test:
**flat-sky cause ⇒ shrinks; intrinsic ℓ₁ cause ⇒ persists.**

## Result (population mean pull over 9000 fiducial obs/arm)

| param | L1 a+c 20° | L1 a+c 10° | CNN a+c 10° |
|---|---|---|---|
| w0 | **−0.37σ** | **−0.10σ** | −0.10σ |
| Ωm | −0.27σ | +0.17σ | +0.01σ |
| σ8 | +0.19σ | −0.18σ | −0.03σ |

## Verdict: **SHRINKS — the 20° L1 w0 bias was largely a flat-sky-distortion artifact.**

Two independent lines:
1. **Magnitude:** L1's w0 offset drops −0.37σ → −0.10σ going from 20° to 10°.
2. **No longer L1-specific:** at 10° the CNN auto+cross has the *same* −0.10σ w0 pull. The
   20° signature was "L1 biased, CNN not" — that L1-vs-CNN *difference* is gone at 10°.

So the offset is **not** an intrinsic ℓ₁ compression bias; it tracked the flat-sky
approximation and largely vanishes on the more-valid 10° patches.

## Caveats / nuance
- **Auto-only shared offset:** the auto-only arm shows ~+0.35σ w0/Ωm pull in *both* L1 and
  CNN — shared ⇒ a property of the 10° auto-only observation/projection, not a method bias.
  It **cancels globally** (SBC ranks uniform → globally unbiased), so it's a fiducial-local,
  information-content effect, not an ℓ₁ artifact.
- **Globally consistent:** SBC shows w0 ranks uniform (mean_rank ≈ 0.50, KS p ≈ 0.5) for all
  arms — i.e. globally unbiased, consistent with the small residual local offset.
- The residual −0.10σ is small and shared by both methods (within the calibrated spread per
  TARP/L-C2ST), so it does not compromise the constraints.
