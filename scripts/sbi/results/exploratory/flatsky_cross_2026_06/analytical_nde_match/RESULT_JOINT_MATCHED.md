# Joint ℓ1 / full4d through the matched best-NDE pipeline — gated (2026-06-21)

**Thesis (Andreas):** the wavelet *joint* ℓ1 — the histogram of the across-channel coefficient
vector, with cells holding the ℓ1 sum — is the *complete* cross-correlation statistic; products
κ_iκ_j are only its 2nd-moment (ξ_ij) slice. So it should (Q1) beat ℓ1+product on FoM3 if
calibrated, and (Q2) be far more BNT-robust (full4d is exactly basis-covariant, P4b) — ℓ1+product's
BNT collapse (0.26×) being the direct evidence products miss most of the cross-correlation.

**Pipeline (every arm):** build datavector → VMIM-MLP 10-D → sbi_lens RealNVP 4×128 (NDE seeds
41,42,43 pooled) → n=9000 median FoM3 → GATE C (TARP-DRP 600 + SBC). All arms adaptive-ranges
(transported binning). full4d K=5 + dequantize; jointl1 K=10. Reference: ℓ1+product noBNT 3045 /
BNT 779 (0.26×); pair2d→RealNVP 4864 GATE FAIL (the cautionary baseline).

## Q1 — better statistic? (noBNT FoM3 vs ℓ1+product 3045; PASS-gate required)
| arm (noBNT) | FoM3 n=9000 | σ(Ωm,σ8,w0) | gate |
|---|---|---|---|
| ℓ1+product (2nd-moment cross only) | 3045 | 0.048/0.077/0.229 | PASS-with-caveat |
| **joint ℓ1** (full pairwise joint, ℓ1-weighted) | **3754** | **0.043/0.069/0.220** | **PASS-with-caveat** ✅ |
| CNN ResNet18 (reference) | 3326 | 0.045/0.072/0.231 | PASS (net +0.030) |
| full4d (full 4-D joint, counts) | 4501 | 0.037/0.057/0.209 | **FAIL** (SBC 0.35) |
| pair2d (pairwise joint, counts) | 4864 | — | **FAIL** |

**A clean monotonic completeness/calibration trade-off:** as the joint statistic gets more complete
(products → joint ℓ1 → full4d/pair2d), raw FoM3 rises but calibration degrades. **joint ℓ1 is the
calibrated sweet-spot** — a genuine +23% FoM3 / ~10% marginal gain over ℓ1+product *at the same
calibration bar* (both PASS-with-caveat, SBC ~0.31). The continuous ℓ1-weighted cells keep it
honest where the count-histograms (full4d, pair2d) over-fit into over-confidence (fool's gold).
joint ℓ1 sits at/above the CNN; its small FoM3 edge over the CNN is partly its milder calibration
(SBC 0.31 vs CNN 0.29), so the fair statement is **joint ℓ1 ≈ the CNN**.

**Pooled calibration (Andreas's preferred judge — overall, not per-tercile).** Pooling all
terciles+seeds, the pooled-TARP coverage is clean for joint ℓ1: **net −0.003, worst dev +0.027 —
within the ±0.05 PASS zone**, as well-centered as ℓ1+product (net +0.002) and tighter-centered than
the CNN (net +0.035, conservative). The "PASS-with-caveat" above is the stricter *per-tercile* read;
pooled, joint ℓ1 is calibrated. Figures: tarp_pooled_jointl1_3arm.png, sbc_pooled_jointl1_3arm.png,
violins_jointl1_3arm.png, violin_fom3_jointl1_3arm.png.

### Q1 seed-robustness (the binding caveat — RESOLVED)
The Q1 win is **not** a favorable compressor-seed draw (contrast pair2d/A1, which were seed-sensitive
*and* failed calibration). 3 compressor seeds (each VMIM(s)→RealNVP, NDE 41,42,43 pooled, gated):

| compressor seed | FoM3 n=9000 | gate |
|---|---|---|
| 41 | 3754 | PASS-with-caveat |
| 42 | 3761 | PASS-with-caveat |
| 43 | 4034 | PASS-with-caveat |

**Band 3754–4034 (mean 3850, ~7% spread), all PASS-with-caveat.** Every seed beats ℓ1+product by
≥23% and sits at/above the CNN, with consistent calibration. → **seed-robust and calibrated.**

### Mean-observation (noiseless, symmetric) cross-check
At the noiseless mean of ~9000 patches: joint ℓ1 **3433** (σ 0.045/0.072/0.232) ≈ CNN **3299**
(0.044/0.071/0.230), both above ℓ1+product **2839** (0.050/0.080/0.235). Same ranking as the
population median; unbiased (truth-centered). Figure: contour_jointl1_vs_l1product_vs_cnn.png.

## Q2 — BNT-lossless? (BNT/noBNT ratio vs ℓ1+product's 0.26×)
| statistic | noBNT | BNT | raw BNT/noBNT | BNT gate |
|---|---|---|---|---|
| ℓ1+product | 3045 | 779 | **0.26×** | (PASS) |
| joint ℓ1 | 3754 | 3232 | **0.861** | **FAIL** (SBC 0.33, dev 0.110) |
| full4d | 4501 | 3163 | 0.703 | **FAIL** (SBC 0.34, dev 0.130) |

The raw-retention contrast (0.86 vs 0.26) is exactly the prediction — **the joint captures the
cross-correlation that products miss; the collapse-vs-no-collapse is the direct evidence.** BUT every
BNT joint arm FAILs the gate (over-confident in the BNT frame, SBC ~0.33). So we **cannot** quote a
*calibrated* BNT-lossless number: the joint is far more BNT-robust in raw FoM3, but the BNT-frame
estimate is over-confident. This is the P4c residual-shear problem made empirical — the fixed
adaptive-ranges (transported) binning does not fully transport the pairwise/4-D joint, and the
mis-transport surfaces as BNT-frame over-confidence ("only a *learned* linear front-end transports",
lane-C). full4d's lower raw ratio (0.70) is between two over-confident numbers (its noBNT arm also
FAILed) and is not a meaningful "retention".

## Discarded / not-the-answer
- `deep2` (BNT auto + avg + bin4): **inadmissible** — it builds deep channels from the *original*
  (pre-BNT) maps, so it cannot survive a real cut-in-BNT-space pipeline; and uncut it is the trivial
  rotate-back (B invertible). RealNVP "over-recovered" it to 1.53× but the gate FAILed (over-confident).
- `full4d`, `pair2d`: over-fit (FAIL) — completeness without calibration.

## Headline for the paper (refines M1)
With the matched best NDE and the standard 10-D compression, a **fixed analytical statistic — the
wavelet joint ℓ1 — robustly matches the optimized CNN and beats ℓ1+product by ~25% on FoM3 (~10% on
the σ marginals), calibrated and seed-robust.** This sharpens M1 from "CNN ≥ analytical by ~9%" to
"analytical (joint ℓ1) ≈ CNN, calibrated" — the joint captures the cross-correlation that products
miss. Under BNT the joint is far more robust in raw FoM3 (0.86 vs 0.26) but not yet *calibratedly*
lossless (BNT-frame over-confidence; open).

Figures: violins_jointl1_3arm.png, violin_fom3_jointl1_3arm.png, tarp_pooled_jointl1_3arm.png,
sbc_pooled_jointl1_3arm.png, contour_jointl1_vs_l1product_vs_cnn.png.
Seed-check: RESULT_JOINTL1_SEEDCHECK.md.
