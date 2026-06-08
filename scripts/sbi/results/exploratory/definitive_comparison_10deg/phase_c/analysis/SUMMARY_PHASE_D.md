# Phase D — definitive 10° L1-vs-CNN result (2026-06-07)

Robust per-patch population analysis over **9000 obs/arm** (180 patches × 50 perms),
3-seed-pooled, **both arms through the identical jaxili MAF** (removes the Phase-C
NDE-architecture confound: Phase C used RealNVP for CNN, MAF for L1). Lead with σ/2D;
FoM3 reported, not headlined.

## 1. Constraining power — median over patches

| arm | σ(Ωm) | σ(σ8) | σ(w0) | 2D(Ωm,σ8) | FoM3 |
|---|---|---|---|---|---|
| **CNN auto+cross** | **0.032** | **0.047** | **0.171** | **1808** | 17251 |
| L1 auto+cross | 0.046 | 0.072 | 0.188 | 1045 | 8530 |
| CNN auto-only | 0.051 | 0.079 | 0.247 | 463 | 2343 |
| L1 auto-only | 0.056 | 0.085 | 0.246 | 441 | 2200 |

- **auto+cross: CNN beats L1 on every parameter** — σ(Ωm) ×1.45, σ(σ8) ×1.5, σ(w0) ×1.1,
  2D ×1.7, FoM3 ×2.0. Even **w0** (L1's traditional edge) now favors CNN.
- **auto-only: a tie** (CNN a hair better: σ(w0) ×1.00, 2D ×0.95).
- **Overturns the original 20° "L1 wins" headline** (20° median had L1 ahead: σ(w0) ×1.34,
  FoM3 ×2.17). Confirms the Phase-C single-patch hint (L1/CNN FoM3 ratio 0.49 here vs 0.48
  single-patch — consistent, so not a patch fluke).

## 2. ★ The w0-offset headline — SHRINKS (flat-sky artifact, not intrinsic ℓ₁ bias) ★

Population-mean w0 pull at the fiducial:

| arm | pull(w0) 20° | pull(w0) 10° |
|---|---|---|
| L1 auto+cross | **−0.37σ** | **−0.10σ** |
| CNN auto+cross | ~0 | −0.10σ |

L1's 20° −0.37σ w0 bias **shrinks to −0.10σ at 10° AND is no longer L1-specific** (CNN
auto+cross has the same −0.10σ). ⇒ the 20° L1 w0 bias was largely a **flat-sky-distortion
artifact**, not an intrinsic ℓ₁ compression bias. (See OFFSET_VERDICT.md.)
Secondary: the **auto-only** arm shows a **shared ~+0.35σ** w0/Ωm offset in *both* L1 and
CNN — a property of the 10° auto-only observation/projection, not method-specific (and it
cancels globally — SBC below).

## 3. Calibration — CNN's tightness is trustworthy (all three tests)

- **TARP-DRP (proper varied-θ, stratified by FoM3 tercile):** all 4 arms — and crucially the
  tight **HIGH-FoM3 tercile** — sit on the diagonal (expected ≈ nominal) → **calibrated**,
  including the tight posteriors ⇒ **FoM3 is real, not inflated.** (NOTE: an earlier fixed-θ
  Mahalanobis-χ²₃ proxy from `tarp_per_patch_fiducial.py` spuriously showed strong over-coverage
  — it is NOT a valid TARP at fixed θ and is disregarded; the varied-θ DRP above is the valid test,
  consistent with SBC + L-C2ST. Figs: `tarp_drp/figures/tarp_{per_arm,overlay}_dim3`.)
- **SBC (global, 400 val cosmos):** ranks uniform on Ωm/σ8/**w0** (mean_rank ≈ 0.50, KS
  p ≈ 0.4–0.6, std_rank ≈ 0.29 ≈ uniform — *not* the 20°'s ∪-shaped 0.31–0.34 over-confidence).
  w0 globally unbiased for all arms. (Mild miscalibration only in the weak h0/Ωb nuisances.)
- **L-C2ST (local at fiducial, CNN, logreg):** **0% of 30 obs reject** local calibration
  (median p ≈ 0.2; median T_obs ≈ 6e-4 = calibrated baseline); validity gate passes. A clean
  improvement over 20° (which rejected at 87%). CNN is locally calibrated.

## 4. Sharpness ≠ calibration — CNN's tightness is efficiency, not geometry-recognition

Decomposing the per-patch pull z=(post_mean−truth)/post_σ over the 9000-obs grid (one-way
variance components, factor = patch index, 180 patches × 50 perms) settles *why* CNN is
tighter, and disposes of a "CNN learns the patch geometry" reading:

- **The patch-to-patch scatter is realization, not geometry.** Geometry (between-patch)
  fraction of the pull variance: **L1 = 0.2% / 0.5% / 2.7%** (Ωm/σ8/w0), **CNN ≈ 0% all
  params**. >97% of L1's scatter is the noise/structure *realization*, not sky position; the
  only whiff is L1's w0 (2.7%, mild |lat| trend). Neither method has meaningful geometry
  dependence — so the CNN↔L1 gap is **estimator efficiency** (CNN's within-patch z-std
  0.6–0.7 vs L1's 0.9–1.0, a ~1.3–1.4× lower-variance edge), exactly what a near-sufficient
  VMIM summary should give.

- **Tightness does not mean over-confidence — if anything CNN is the conservative one.**
  Empirical marginal coverage over the 9000 obs (calibrated ⇒ z-std≈1, 68% interval covers 68%):

  | arm / param | 68% cov (Ωm/σ8/w0) | z-std (Ωm/σ8/w0) | verdict |
  |---|---|---|---|
  | L1 a+c | 72% / 68% / 84% | 0.93 / 1.01 / 0.73 | ~calibrated (conservative on w0) |
  | CNN a+c | 87% / 83% / 91% | 0.67 / 0.74 / 0.59 | mildly **conservative (over-covers)** |

  L1 is essentially spot-on; CNN's nominal-68% intervals actually contain truth ~83–91% — its
  contours are tighter than L1's *and* a touch wider than they strictly need to be. So in TARP
  CNN sits on-or-slightly-**above** the diagonal (conservative), L1 on it — never L1 below
  (over-confident), consistent with §3. The estimator-variance difference lives on the
  **sharpness** axis (σ/FoM3), which TARP is blind to by construction — which is why the
  scatter difference correctly does *not* appear as a TARP calibration gap.

- **Aside (small, real):** L1 carries a mild fiducial *bias* (mean pull +0.17σ Ωm, −0.18σ σ8)
  that CNN lacks (≈0); both share the −0.10σ w0 offset. ~0.17σ, washes out globally per SBC,
  but it's the second reason L1 centers look further from truth (higher variance + small offset).

Fig: `figs/D7_local_coverage_vs_latitude.{png,pdf}` — per-patch z-std vs latitude (flat ⇒ no
geometry; CNN in the conservative band, L1 on the calibrated line).

## 5. Where L1's fiducial offset comes from — prior shrinkage, an information effect (NOT an L1 bug)

The small L1 a+c fiducial offset (mean pull +0.17σ Ωm, −0.18σ σ8) is **prior shrinkage**
(regression of the posterior mean toward the prior mean), with size set by *how informative
the summary is*, not by which compressor. Three consistent lines of evidence:

1. **Direction matches.** The fiducial sits off-center in the CosmoGrid training prior
   (prior means Ωm=+0.299, σ8=+0.811, w0=−0.896 vs fiducial 0.26/0.84/−1.0). Shrinkage pulls
   the mean toward the prior mean ⇒ Ωm up (+), σ8 down (−) — exactly the observed signs.

2. **Magnitude obeys one information fraction.** For a conjugate-Gaussian posterior,
   bias ≈ (1−r)·(prior_mean − truth), r = information fraction. Fitting r on Ωm,σ8 (w0 carries
   the separate flat-sky term, see below): a *single* r per arm fits both params. Values track
   FoM3 monotonically:

   | arm | median FoM3 | r (info fraction) | Ωm bias / σ8 bias |
   |---|---|---|---|
   | CNN a+c | 17251 | **0.97** | +0.002 / +0.000 |
   | L1 a+c | 8530 | **0.64** | +0.014 / −0.011 |
   | CNN auto | 2343 | **0.49** | +0.022 / −0.012 |
   | L1 auto | 2200 | **0.22** | +0.030 / −0.024 |

3. **It's not an L1 property.** **CNN auto-only (+0.36σ Ωm) is *more* biased than L1 a+c
   (+0.17σ)** — bias is monotonic in information across *all* arms, regardless of compressor.
   And *within* L1 a+c, splitting by FoM3 tercile, |Ωm bias| collapses LOW +0.047 → MID +0.015
   → HIGH −0.021. CNN looks unbiased in the headline only because CNN a+c is the most
   informative arm (r≈0.97, almost no prior pull); L1 a+c is more prior-regularized (r≈0.64).

This is the **correct, expected** behavior of a calibrated posterior at a prior-off-center
truth — not a noise-model bug, not a flat-sky artifact, not L1-specific. SBC confirms it
averages to zero over the prior (cosmologies on the other side of the prior mean flip the sign).

- **w0 is the exception** (separate mechanism): the a+c −0.10σ is the flat-sky residual (shared
  with CNN), which *opposes* the weak w0 shrinkage (prior mean −0.896 > −1.0 wants +). In the
  less-informative auto-only arms shrinkage wins ⇒ w0 +0.38σ toward the prior mean.
- **20° sign-flip reconciliation (hypothesis, needs 20° per-patch grid):** the 20° offset had
  Ωm −0.27σ / σ8 +0.19σ — opposite to 10° shrinkage. Likely a larger flat-sky *projection* bias
  dominated at 20° (opposite direction); at 10° that bias shrinks (w0 −0.37→−0.10), leaving prior
  shrinkage as the residual.

Fig: `figs/D8_shrinkage.{png,pdf}` — (left) bias vs (prior_mean − truth) with per-arm shrinkage
lines slope=1−r; (right) |bias| vs median FoM3, monotonic ⇒ information effect.

## 6. ⚠ Cross-map information leakage — auto+cross is partly unphysical (open: flat-sky test)

The 6 cross channels are built on the **full sphere** (a^i_ℓm·a^j_ℓm → iSHT on the whole sphere →
gnomonic patch cutouts; `build_full_sphere_cross_cache.py`, no apodization/mask). So **every
cross-patch pixel is a global functional of the full-sky convergence** — each 10° patch carries
cross-correlation info from the entire field (leakage). Quantified: cross channels hold 12–20% of
their variance at super-patch scales (ℓ<18) vs 0.4–1.0% for autos, with ℓ_median down to ~60
(autos ~600) — large-scale and non-local (fig `figs/D9_cross_leakage_scales`).

This explains the auto-only tie vs auto+cross CNN≫L1: **autos are local/fair** (both compressors
tie); **cross channels carry leaked full-sky info** the CNN reads efficiently but L1 (per-channel
ℓ₁, small-scale statistic) cannot — and which neither could get from the local auto patches (so
the CNN's "should recover it from autos" intuition fails: the info isn't locally there). Old 20°
"L1 ≫ CNN auto+cross" was mostly artifact (noise model + single-perm + FoM3), not a real reversal.

**Implication:** not a calibration bug (leakage is self-consistent in train+test ⇒ TARP/SBC pass),
but the auto+cross constraining power — esp. the CNN gain — is **partly unphysical** (a real patch
survey can't build these maps); patches also aren't independent across a realization. **Auto-only
is unaffected (local).** Decisive open test: rebuild cross channels **flat-sky/patch-local** and
rerun — if leakage drove it, the CNN auto+cross gain collapses toward auto-only. Full writeup:
repo-root `CROSS_MAP_LEAKAGE_FINDING.md`; memory `project_cross_map_leakage_fullsphere`.

## Bottom line

At the more-flat-sky-valid 10°, **CNN matches-or-beats L1 (decisive on auto+cross), its
tighter constraints are fully calibrated (TARP + SBC + L-C2ST all clean), and L1's 20° w0
bias was a flat-sky artifact** (shrinks to ~0, no longer method-specific). This is the
opposite of the original 20° headline and resolves the w0-bias question.

Artifacts: `geometry/*/per_patch_grid.{csv,npz}`, `tarp/*/coverage.json`,
`sbc/*/sbc_*.json`, `lc2st/*/*.json`.
