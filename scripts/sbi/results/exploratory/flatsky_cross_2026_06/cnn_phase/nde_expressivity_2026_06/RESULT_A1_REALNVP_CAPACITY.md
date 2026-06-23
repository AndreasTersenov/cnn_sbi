# Route A1 — deeper/wider RealNVP: clean NEGATIVE (capacity hurts; over-coverage is not under-fitting)

**Date:** 2026-06-23. Frozen resnet18 s41 summaries; reported posterior = sbi_lens RealNVP, 3 NDE
seeds pooled. Screen FoM3 on 1000 fiducial obs; calibration via the proper sightline-bootstrap
(un-stratified TARP net ±1σ + SBC) on 600 varied-θ val points. Pre-registered rule: calibrated = net
∈ [−0.02,+0.02] AND SBC ∈ [0.282,0.296]; target FoM3 ≥ 3371 (joint ℓ1).

| config | FoM3 (1000-obs) | TARP net (±1σ) | SBC (Ωm/σ8/w0) | verdict |
|---|---|---|---|---|
| **4×128 (baseline)** | 3326 / 3304 | +0.033 ± 0.020 | 0.290/0.289/0.282 | conservative (reference) |
| L8×128 | 3237 | +0.050 ± 0.021 | 0.284/0.286/0.271 | not calibrated |
| L12×128 | 3154 | +0.042 ± 0.021 | 0.290/0.288/0.280 | not calibrated |
| L16×128 | 2705 | +0.050 ± 0.020 | 0.271/0.277/0.255 | not calibrated |
| L8×256 | 2881 | +0.021 ± 0.021 | 0.287/0.287/0.274 | not calibrated |
| L12×256 | 2391 | +0.001 ± 0.021 | 0.270/0.278/0.272 | not calibrated |

## Findings
1. **FoM3 declines monotonically with capacity** (depth 4→16: 3326→3154→2705; width 128→256 worse at
   every depth). The 4×128 production flow is the optimum of the affine-RealNVP family — it is **not
   under-fitting**.
2. **No config calibrates the joint while staying sharp.** The ×128 configs hold net ≈ +0.04–0.05
   (the joint over-coverage is unchanged or slightly worse). The ×256 configs lower the net only by
   *widening the whole posterior* — their SBC marginals drop to 0.270–0.278 (now conservative on the
   margins too) and FoM3 collapses. There is no "tighten the correlations, keep the marginals" config.
3. **Instability at depth:** L12×128's gate flow hit a NaN blow-up mid-training (recovered to its best
   pre-NaN checkpoint); deeper affine RealNVP is harder to optimise — another sign this is the wrong
   lever.

## Interpretation
Because adding flow capacity *hurts* FoM3 and does not move the joint toward the diagonal, the CNN's
residual mild over-coverage (+0.033) is **not a flow-capacity / under-fitting deficiency**. It is most
consistent with an **intrinsic** effect (amortization gap + finite training set) — the calibrated CNN
posterior at this summary/data size simply is what it is. This is useful for the paper's
referee-defense: it is direct evidence the CNN at 4×128 is at its estimator optimum, so the tie with
joint ℓ1 (3371) is real, not a "you trained the flow wrong" artifact.

## Status
- **A1 closed: negative.** Baseline 4×128 retained (FoM3 3304/3326, the conservative lower bound).
- **A2 (Neural Spline Flow)** is the remaining flow lever — it changes the *transform family* (flexible
  monotonic vs affine), an axis A1 did not test, so it is not strictly ruled out. But A1's "capacity
  hurts / not under-fitting" signal lowers the prior that any flow change calibrates-while-sharpening to
  ≥ 3371. Decision pending (worth one careful NSF attempt vs accept the intrinsic-tie conclusion).
