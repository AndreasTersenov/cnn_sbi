# Lane A — corrected conclusion (supersedes "third pillar reopens")

**Question:** is A1 (VMIM-compressed pairwise joint PDF, FoM3 ~3.5k) a genuine new best
summary statistic with more cosmological information?

**Answer: No — not as a clean information claim.** A1 is comparable-to-slightly-better than
l1+product on marginals, with seed-dependent borderline calibration; its FoM3 headline is
inflated by FoM3 fragility and by neural-estimator-path effects, not by extra physics. The
robust, real finding underneath is that **our learned compressors/NDEs are suboptimal** —
which is the methodological issue worth chasing.

## The evidence, assembled

**1. Multi-seed band (VMIM_ROBUSTNESS.md).** Compressor seeds 41/42/43 → FoM3 3822 / 3441 /
3408 (mean 3557 ± 188, 5% spread). Seed-41's 3822 was the favorable draw; robust central
~3420–3560. TARP net bias VARIES with seed: +0.021 / +0.009 / −0.022 — i.e. NOT reliably
conservative (seed 43 mildly over-confident). Registered condition for "third pillar
confirmed" was *band > 2875 AND net bias ≥ 0 across seeds* → the calibration half FAILS.

**2. Fiducial marginals (the apples-to-apples; fiducial_corner_a1_vs_product_typical).** At
the canonical typical obs (perm16/patch23), A1 vs l1+product: σ(Ωm) 0.042 vs 0.041 (tied),
σ(σ8) 0.060 vs 0.067 (~10% tighter), σ(w0) 0.214 vs 0.241 (~11% tighter). A MODEST marginal
edge — the population FoM3 ×1.22–1.33 is fragility compounding small per-param gains plus a
correlation-structure change. Contours overlap heavily in the corner.

**3. Data-processing inequality (the structural argument).** A1 is a deterministic
compression of pair2d, so I(θ;A1) ≤ I(θ;pair2d): compression cannot add information. Yet
measured FoM3(A1) ~3.5k > FoM3(pair2d-raw) 2794. The only resolution: the MAF estimates a
10-d density better than a 3000-d one, so A1's gain over pair2d-raw is an *estimation-path*
effect, not physics. Corroborated by the K-trend: K=8/10/15 → 2874/2794/2455, FoM3 DECREASES
as the grid refines (a finer grid contains ≥ the information by the same DPI) — pure
curse-of-dimensionality in the estimator.

**4. The robust core (independent of A1).** l1+product is gate-C CLEAN (|dev| ≤ 0.037), so
its 2875 is trustworthy and near-optimal *for that statistic*. The flat-sky CNN sits at
~2300 (FLATSKY_CNN_RESULT). A calibrated hand-built statistic beating the CNN ⇒ **the CNN is
genuinely suboptimal** (documented: unstable sbi_lens RealNVP companion —
[[project_maf_companion_not_bottleneck]], [[project_nde_architecture_mismatch]]; flat-sky
"remaining rung = architecture"). This is the real result and the right thing to fix.

## What survives as physics
- Joint one-point statistics reach ~l1+product level from AUTO maps alone (the cross-map
  information is accessible from the pairwise joint histogram of the autos). [GATE-C-caveated;
  the pair2d arms are mildly over-confident — see GATE_C_JOINT.md.]
- BNT P4c grid-transport story (lane C): finer K does NOT close the shear gap (r 0.53→0.55
  K=10→15), confirming only a learned linear front-end transports.

## What does NOT survive
- "A1 = new best statistic, third pillar." The 3.5k is not a defensible information claim
  (DPI + fragility + seed-dependent calibration). Do not headline it.

## The clean next test (Andreas's idea, the adjudicator): the 2×2
{pair2d, l1+product} × {raw→MAF, VMIM→MAF}. Three cells exist:
l1+product raw = 2875 (clean), pair2d raw = 2794 (over-confident), pair2d+VMIM = A1 ~3.5k
(borderline). The 4th cell = **l1+product + VMIM** (Andreas's proposal). Pre-registered:
- If l1+product+VMIM IMPROVES FoM3 *and* stays calibrated → even a cleanly-calibrated
  high-dim arm leaves usable information for the MAF; compress-before-flow is a generic win
  (partial vindication of the estimation-path view).
- If it does NOT improve → l1+product's MAF was already optimal; A1's gain was pair2d-raw-
  specific + its own miscalibration (the joint-PDF "win" is an artifact).
- If it improves but miscalibrates like A1 → artifact signature; discount.
Gate the new arm (TARP+SBC) exactly as A1. The *pattern across the 2×2* — not any single
FoM3 — is the result.

**Deeper recommendation:** before ranking ANY statistics by FoM3, fix one NDE architecture +
training budget + convergence diagnostic and run every arm (CNN features, l1+product,
pair2d, A1) through it. Current FoM3 differences of ~20–30% between methods are as likely to
be estimator quality as physics.
