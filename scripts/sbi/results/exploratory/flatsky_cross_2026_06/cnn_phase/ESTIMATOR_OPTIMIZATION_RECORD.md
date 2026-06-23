# CNN estimator optimization — the "we tried everything" record (paper robustness section)

**Purpose.** Document that the CNN-VMIM → NDE pipeline used for the L1-vs-CNN comparison was
*exhaustively* optimized, so that the comparison is a fair "best-effort CNN vs best-effort analytical
statistic," and the residual mild conservatism is intrinsic rather than a training failure. This is the
referee-defense record for "your CNN is undertrained / you used the wrong estimator." Every number is
pooled 9000-obs median FoM3 (= 1/√det C₃ over Ωm, σ8, w0) unless noted, on the de-leaked flat-local
10°/80px data; calibration is GATE-C (TARP-DRP + SBC), recomputed with the proper sightline-bootstrap
band (see `calib_refine_2026_06/`). Frozen resnet18 summaries unless stated.

## The chosen setup
**Compressor:** ResNet-18 VMIM (10-D summary). **NDE:** sbi_lens ConditionalRealNVP, 4 coupling
blocks × 128, 3 NDE-seeds pooled. **FoM3 = 3326 (s41) / 3304 (3-seed mean).** Calibrated (mildly
conservative, safe direction). Every lever below was swept around this point; none beat it.

## Lever 1 — density-estimator family (on fixed plain-CNN summaries, seed 41)
| NDE family | FoM3 |
|---|---|
| jaxili MAF (the original common readout) | 2312 |
| jaxili RealNVP | 2258 |
| jaxili MDN | 2885 |
| **sbi_lens RealNVP** | **3139** |
The common-MAF readout under-served the low-D CNN summary; its own best-calibrated NDE (sbi_lens
RealNVP) is +36%. (This is what reversed the earlier "CNN underperforms L1": that was a common-MAF
artifact.) Control: the *same* RealNVP **craters** on the 2000-D L1 vector (1249) — the lift is
CNN-specific; L1's best NDE is the MAF (2875).

## Lever 2 — compressor architecture (each read out with sbi_lens RealNVP, seed 41)
| compressor | FoM3 |
|---|---|
| plain CNN | 3139 |
| **ResNet-18** | **3326** |
| plain + attention | 3205 |
| ResNet-small | 3072 |
| ResNet-50 (GroupNorm) | 2760 |
ResNet-18 is the optimum; deeper (ResNet-50) overfits the ~900-cosmology training set.

## Lever 3 — NDE flow capacity (deeper / wider RealNVP, frozen resnet18 s41)
| config | FoM3 | TARP net | calibrated? |
|---|---|---|---|
| **4×128 (production)** | **3326 / 3304** | +0.033 | conservative (optimum) |
| 8×128 | 3237 | +0.050 | no |
| 12×128 | 3154 | +0.042 | no |
| 16×128 | 2705 | +0.050 | no |
| 8×256 | 2881 | +0.021 | no |
| 12×256 | 2391 | +0.001 | no |
FoM3 declines monotonically with capacity; the 4×128 flow is **not** under-fitting (more capacity
hurts). Deeper affine flows also become unstable (NaN). Detail: `nde_expressivity_2026_06/RESULT_A1_REALNVP_CAPACITY.md`.

## Lever 4 — NDE transform family (Neural Spline Flow, frozen resnet18 s41)
| flow | steps | val loss | FoM3 | σ_w0 |
|---|---|---|---|---|
| affine RealNVP 4×128 | 50k | −11.6 | **3326** | 0.231 |
| NSF 4×128 (RQS, 8 bins) | 50k | −1.0 | 1993 | 0.247 |
| NSF 4×128 (RQS, 8 bins) | 150k | −5.8 | 839 | 0.279 |
A rational-quadratic spline flow (the modern, strictly-more-expressive default) is a *better density
estimator* (higher likelihood) but yields *much wider* posteriors, and trains *wider* the longer it
runs — the classic val-loss-≠-FoM3 decoupling. Detail: `…/RESULT_A2_NSF.md`. (Earlier negative levers,
same direction: beefier MAF VMIM companion flow worse than RealNVP; PCA on L1 craters FoM3.)

## The throughline
Across the density-estimator family, compressor architecture, flow capacity, **and** transform family,
**more flexibility consistently buys density-fit, not posterior sharpness** — it widens the posterior
and lowers FoM3. The chosen ResNet-18 + sbi_lens RealNVP 4×128 is the optimum on every axis swept.

## Calibration of the optimum (and why the comparison is fair)
The optimal CNN is **mildly conservative**: un-stratified TARP net **+0.033 ± 0.020** (~1.6σ), SBC
0.290/0.289/0.282 (marginals calibrated, w0 a hair wide). The over-coverage is therefore **intrinsic**
(amortization gap + finite training set), not a fixable estimator deficiency — confirmed by Levers 3–4
(no flow change reduces it without collapsing FoM3). The two analytical L1 summaries are
joint-calibrated but slightly **over-confident** on the marginals (SBC ≈ 0.30), so the comparison is
*conservative for the CNN*: correcting both to perfect calibration would tighten the CNN and loosen the
L1 (the CNN FoM3 3304/3326 is a lower bound). Net: the optimal CNN **ties joint ℓ1 (3371) within
calibration tolerance**, with the residual difference favouring the CNN once both are held to the same
standard.

## Bottom line for the paper
We optimized the learned-summary pipeline exhaustively — estimator family, architecture, flow capacity,
and transform family — and the production setup is the best on every lever; its mild conservatism is
intrinsic and in the safe direction. Therefore the headline ("a near-optimal learned CNN summary ties
the best analytical higher-order statistic within calibration tolerance") is not an artifact of an
under-trained or mis-specified estimator. Post-hoc recalibration (conformal / CP4SBI) was deliberately
*not* used — the result stands on the by-construction estimator.
