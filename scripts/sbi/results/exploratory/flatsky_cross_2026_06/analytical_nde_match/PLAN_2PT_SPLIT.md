# PLAN — is the cross-map information only two-point? (2026-07-01)

## Objective
Decide, per operator, whether the **conv** and **product** cross-maps carry cosmological
information *beyond the complete two-point sector* — i.e. genuinely non-Gaussian cross content.
This adjudicates a tension: CLAUDE.md's headline says the product carries "genuine non-Gaussian
joint moments ⟨κᵢⁿκⱼⁿ⟩," while `06-bnt.tex` l.111 says "products = the Gaussian sector."
(Also re-grounds the stale 0.38 figure, which came from a MAF-era auto-only run — see
`BNT_THEORY_DEEP_DIVE.md` §5.4 / `overnight_menu` A1_cov_bnt.)

## The idea
By P7 (`BNT_THEORY_DEEP_DIVE.md`), the **auto+cross wavelet (co)variance vector `cov`** *is* the
complete two-point sector, exactly. So anything a cross-map adds **on top of a datavector that
already contains `cov`** is non-Gaussian by construction. Append `cov` to each L1 arm and compare.

## Arms (no-BNT frame, matched pipeline = same as RESULT_ANALYTICAL_NDE_MATCH.md)
Build with `build_flatsky_joint_arm.py --stat cov --basis nobnt`, appended to the existing L1
caches (`l1_matrix/l1_{none,conv,product}_cache`), then VMIM→10-D→sbilens_realnvp 4×128 → FoM₃.

| arm | datavector | build |
|---|---|---|
| `cov` | complete 2pt sector alone | `--stat cov` (no append) |
| `auto_cov` | auto-ℓ₁ ⊕ cov | `--append-to l1_none_cache` |
| `conv_cov` | (auto+conv)-ℓ₁ ⊕ cov | `--append-to l1_conv_cache` |
| `product_cov` | (auto+product)-ℓ₁ ⊕ cov | `--append-to l1_product_cache` |

Reference (committed, RealNVP): auto 2448, +conv 2671, +product 3045, joint ℓ₁ 3371, CNN 3326.

## Decision metric
FoM₃ (median over the fiducial population), **same VMIM→RealNVP pipeline for every arm**.
- **Positive control (test sensitivity):** `auto_cov` − `cov` > 0 ⟹ the pipeline can see
  non-Gaussian content (the auto 1-point moments beyond variance). If ≈0, the whole test is blind
  and inconclusive — stop.
- **conv non-Gaussian:** ΔNG(conv) = `conv_cov` − `auto_cov`.
- **product non-Gaussian:** ΔNG(product) = `product_cov` − `auto_cov`.
  - ≈0 (within seed/calibration noise) ⟹ the operator is **two-point only**.
  - > 0 significantly ⟹ the operator carries **non-Gaussian** cross information.

Prediction: `product_cov` > `auto_cov` (product has non-Gaussian) and `conv_cov` ≈ `auto_cov`
(conv is a two-point re-encoding). If so, `06-bnt.tex` "products = Gaussian sector" is wrong.

## Staging
1. **Screen** (this run): 1 compressor seed (s41), n=1000, GPU 0. Read the ΔFoM₃ signs.
2. **Escalate** only if the screen is clear/interesting: 3 seeds (41/42/43), n=9000, + TARP/SBC gate.

## Config fingerprint
- Pipeline: `vmim_from_cache.py --summary-dim 10` → `train_nde_from_compressed.py
  --nde-family sbilens_realnvp --nde-layers 4 --nde-hidden 128 --flow-total-steps 50000`.
- Data: TFDS grid_10deg_80px_nonoverlap180 autos, no-BNT, frozen nobnt sigma table.
- Output dir: `analytical_nde_match/twopt_split/` (FRESH — touches no final result/figure).
- GPU 0 only, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`.

## Done condition
Screen finishes all 4 arms → report ΔFoM₃(conv), ΔFoM₃(product), and the positive control to
Andreas. Do not touch the draft until Andreas rules on the outcome.

---
## SCREEN result (2026-07-01, s41, n=1000, ungated)
| arm | FoM3 | vs auto_cov |
|---|---|---|
| cov (2pt sector) | 987 | −2008 |
| auto_cov | 2994 | baseline |
| conv_cov | 3355 | +361 |
| product_cov | 3604 | +610 |

- Positive control auto_cov−cov = +2008 → test detects non-Gaussian (PASS).
- ΔNG(product) = +610 → **product carries cross non-Gaussian info beyond the complete 2pt sector**
  ⇒ §6.2 "products = the Gaussian sector" is wrong on the physics (not just the stale 0.38).
- ΔNG(conv) = +361 → ~1σ of screen scatter; conv 2pt-only is NOT cleanly established (ambiguous).

Caveats: n=1000 single-seed UNGATED (absolute FoM3 inflated; deltas can be inflated by
over-confidence). Escalate auto_cov/conv_cov/product_cov to 3-seed/n=9000/gated to confirm
product's ΔNG and resolve conv. Do NOT revise §6.2 until the gated deltas are in.

---
## GATED result (2026-07-01, 3 seeds, n=9000, TARP+SBC)
| arm | FoM3 | gate |
|---|---|---|
| cov | 982 | - |
| auto_cov | 2916 | PASS-with-caveat |
| conv_cov | 3221 | PASS-with-caveat |
| product_cov | 3624 | **FAIL** (over-confident: MID TARP −0.116, net −0.028) |

- positive control auto_cov−cov = +1934 (calibrated).
- ΔNG(conv) = +305 (both arms calibrated; but conv_cov slightly more over-confident than auto_cov,
  SBC std 0.31 vs 0.30 → true residual ≤305). ⇒ conv is 2pt-DOMINATED, small/negligible NG residual
  — matches theory (lag-space ξ estimator, trispectrum suppressed).
- ΔNG(product) = +708 but product_cov FAILS calibration ⇒ inflated by over-confidence, an UPPER
  BOUND not a clean measurement. Clean ΔNG(product) needs the 3-compressor deep ensemble (as joint-ℓ₁).

NEXT: run product_cov (+auto_cov/conv_cov) through the 3-compressor ensemble for a calibrated ΔNG.

---
## ENSEMBLE (de-inflated) result (2026-07-01, 3 VMIM seeds x 3 NDE seeds pooled, n=9000)
| arm | single (gated) | 3-compressor ensemble |
|---|---|---|
| auto_cov | 2916 (PASS-caveat) | 2897 |
| conv_cov | 3221 (PASS-caveat) | 3021 |
| product_cov | 3624 (FAIL) | 3157 |

- ΔNG(conv)    = +124 (~4% of auto_cov)
- ΔNG(product) = +260 (~9%)  [product FAIL arm de-inflated most: 3624->3157]

CONCLUSION (matches theory): both operators carry a small NG residual beyond the complete 2pt
sector; conv ~4% (two-point-DOMINATED), product ~9% (~2x conv, genuine pointwise NG moments).
Screen's 305/708 were over-confidence inflation. NOT yet gated on pooled posteriors (follow-up),
but ensemble de-inflation is the calibration fix (joint-l1 / BNT-autoprod precedent).
