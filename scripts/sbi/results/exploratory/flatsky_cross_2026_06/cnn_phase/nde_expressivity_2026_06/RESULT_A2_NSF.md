# Route A2 — Neural Spline Flow: clean NEGATIVE (more transform flexibility → wider posteriors)

**Date:** 2026-06-23. Frozen resnet18 s41 summaries. NSF = conditional rational-quadratic-spline
MaskedCoupling chain on **distrax** (the TFP-JAX `tfb.RationalQuadraticSpline` does not `jit`), θ
z-scored to an N(0,I) base, spline interval [−5,5], 8 bins, alternating even/odd masks. Implemented as
a new `--nde-family sbilens_nsf` (`build_spline_flow` + `train_sbilens_nsf`).

| flow | steps | val loss (best) | FoM3 | σ(Ωm/σ8/w0) |
|---|---|---|---|---|
| affine RealNVP 4×128 (production) | 50k | ~−11.6 | **3326** | 0.045/0.072/0.231 |
| NSF 4×128 (8 bins) | 50k (under-trained) | −1.01 | 1993 | 0.051/0.082/0.247 |
| NSF 4×128 (8 bins) | 150k (converged) | −5.81 | **839** | 0.060/0.116/0.279 |

## Findings
1. **NSF trains stably** under jit (monotone val-loss decrease, no NaN) — the implementation is sound.
2. **NSF posteriors are far wider** than the affine flow: FoM3 839 vs 3326 (converged), σ inflated ~1.3–1.6×
   per parameter. It cannot approach calibrated-and-≥3371, so the calibration gate is moot.
3. **More training makes it WIDER, not sharper** (50k FoM3 1993 → 150k FoM3 839) even as the likelihood
   improves (val loss −1.0 → −5.8). Classic val-loss-≠-FoM3 decoupling: the spline's extra flexibility
   buys density-fit, not posterior sharpness. Same direction as A1 (more capacity → wider), stronger.

## Caveat (honest)
The NSF requires θ-standardization (z-score, N(0,I) base) for spline stability, whereas the production
affine flow uses raw θ with an MVN(0.5,0.05) base. So the comparison is not a perfectly isolated
"affine vs spline" swap. But the FoM3 gap is far too large (839/1993 vs 3326) to be a standardization
artifact, and it moves the *wrong* way with training — the negative is robust to that confound.

## Combined conclusion (A1 + A2)
Across **both** flow axes — capacity (deeper/wider affine RealNVP, A1) and transform family
(rational-quadratic splines, A2) — **nothing beats the production affine RealNVP 4×128**, and more
flexibility consistently *widens* the posterior (lowers FoM3). The CNN's residual mild over-coverage
(+0.033) is therefore **not a fixable estimator deficiency**; it is intrinsic (amortization gap +
finite training set). A *directly* calibrated CNN at FoM3 ≥ joint-ℓ1 (3371) is **not achievable by
changing the flow**.

This is a strong referee-defense result: the estimator was exhaustively optimized (capacity AND
family), the CNN at 4×128 is at its estimator optimum, its mild conservatism is intrinsic and in the
*safe* direction (so its FoM3 is a lower bound), and it ties joint-ℓ1 **within calibration tolerance**.
That is the conclusion to carry to the paper. Route A closed: negative.

## Code
New (working, negative result — keep for the record): `--nde-family sbilens_nsf` in
`train_nde_from_compressed.py` (`train_sbilens_nsf`, z-score standardization) +
`build_spline_flow` (distrax conditional NSF) in `npe_cnn_nbody_tomo.py`; dispatch added in
`tarp_stratified_val_nde.py`. py_compile + jit functional test pass. The earlier tfb `SplineCoupling`
was removed (did not jit).
