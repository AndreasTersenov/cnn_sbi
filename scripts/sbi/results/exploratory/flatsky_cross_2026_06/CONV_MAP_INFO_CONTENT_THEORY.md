# What information do the convolution ("Zürcher") cross-maps contain? — THEORY (D1)

**Answer, one line.** The patch-local convolution cross-map is a *bilinear (degree-2) functional
of the fields whose signal is purely two-point*; it re-encodes the bin-pair two-point information
(a window-folded transform of ξ_ij, **not** the cross-correlation itself), and its only channel to
non-Gaussian information — the trispectrum imprint on the scatter of a few large-scale mode
products — carries negligible signal-to-noise on a 10° patch. The pointwise product, by contrast,
hands the inference a pointwise sample of the *joint one-point PDF* p(κ_i, κ_j), whose higher
moments ⟨κ_iⁿκ_jⁿ⟩ are genuine non-Gaussian inter-bin information.

Created 2026-07-01 by the theory-verification session (answers `HANDOFF_CONV_MAP_INFO_CONTENT.md`).
Qualitative claims **proved**; the residual magnitude is **measured** with the gated 2-pt-split arm.
**Adversarially re-verified 2026-07-01** (`CONV_MAP_SECURE_RESULT.md`): all claims K1–K8 confirmed
(none broken); the fold formula closed to roundoff; T4 quantified (N_eff, T/Wick); §8 paragraph
updated in place (few-mode numbers + ensemble-de-inflated ΔNG).

---

## 0. Two corrections this document establishes (read first)

The prior sketch (handoff §3.2, T3) and the current manuscript (`07-discussion.tex` §7.1, l.23–26)
both contain a precise statement that is **wrong**, though the *conclusion* they draw from it is
right. The certainty standard requires stating the correct version.

- **C1 — the convolution is NOT the cross-correlation, "up to a reflection" or otherwise.**
  The operator computes a *convolution* (product of two forward FFTs, no conjugate):
  κ^ij_conv[x] = Σ_r a_i[r] a_j[x−r], a ≡ κ·W. Its mean is a window-*folded* linear functional of
  ξ_ij, **E[κ^ij_conv(x)] = Σ_r W[r] W[x−r] ξ_ij[2r−x]**, which does not reproduce ξ_ij(x). The
  clean "mean = ξ_ij(x)·(W⋆W)(x)" holds only for the *correlation* operator (conjugate one factor),
  which the code does not use. Measured (see §7): the correlation-operator mean matches ξ with
  Pearson r = 0.9996; the code's convolution mean has r = −0.35 with ξ (opposite sign at zero-lag).
  → The convolution's *signal is still purely two-point* (a linear functional of ξ_ij), so the
  information conclusion is unchanged; only the identification with the cross-correlation is wrong.
- **C2 — the convolution is not "strongly Gaussianised."** Its non-Gaussian *information* is
  suppressed by the **few-effective-modes** regime (§T4), not by central-limit Gaussianisation —
  with a handful of dominant low-k modes the map is, if anything, *less* Gaussian in shape. Shape is
  irrelevant to information content (T1): a map can look strongly non-Gaussian and still carry only
  two-point information. "Strongly Gaussianised" conflates shape with information and should be
  dropped.

---

## 1. Setup and the two operators

Tomographic convergence fields κ_i(x), i = 1..4, on flat-sky patches (10°, 80 px, 7.5′/px).
W is a fixed separable cosine apodisation window; a_i ≡ κ_i·W. Per bin pair i<j the code
(`flatsky_cross.py`, verified line-by-line) builds:

- **Convolution** κ^ij_conv = F⁻¹[ F(κ_iW)·F(κ_jW) ]  ⇔  circular convolution (a_i ∗ a_j)[x] =
  Σ_r a_i[r] a_j[x−r]. Inputs apodised. Flat-sky analogue of the Zürcher harmonic a_lm-product.
- **Product** κ^ij_prod = κ_i·κ_j. Raw, strictly local; spatial mean = ξ_ij(0) (verified, §7).

Both are **quadratic (bilinear) functionals** of the fields. This is the entire crux: the
2-pt-vs-non-Gaussian question is a question about a *degree-2 statistic*, and degree-2 statistics
have a rigid, provable moment structure.

---

## 2. T1 — Gaussian baseline (PROVED): on a Gaussian field, both operators carry *exactly* two-point information

**Theorem (quadratic forms of a Gaussian vector are two-point-complete).**
Let x ∈ ℝ^n be a centred Gaussian vector, x ~ N(0, Σ). Stack the two fields' pixel values into x;
then each output pixel of either cross-map is a quadratic form Q_p = xᵀA_p x with A_p a *fixed*,
known matrix (for the product, A_p is diagonal in the coincident-pixel block coupling κ_i and κ_j;
for the convolution, A_p is the fixed circulant/window operator implementing Σ_r W[r]W[x_p−r]·(κ_iκ_j)).
The joint law of the whole map {Q_p} is a function of Σ alone.

*Proof.* The joint moment generating function is a Gaussian integral,
E[exp(Σ_p t_p Q_p)] = E[exp(xᵀ(Σ_p t_p A_p)x)] = det(I − 2Σ^{1/2}(Σ_p t_p A_p)Σ^{1/2})^{−1/2},
finite for small t, depending only on the fixed {A_p} and on Σ. Equivalently every joint cumulant
is a trace polynomial in {A_pΣ}: for a single zero-mean quadratic form the r-th cumulant is
κ_r(Q) = 2^{r−1}(r−1)! · tr((AΣ)^r), and cross-cumulants are tr of products of A_pΣ. All are
functions of Σ. ∎

Σ is exactly the set of auto- and cross-**two-point** functions (the P7 "complete two-point sector":
`cov` = auto+cross wavelet covariance transforms as Ĉ ↦ BĈBᵀ, a closed invertible set — see
`BNT_THEORY_DEEP_DIVE.md` P7). Hence:

> **Corollary.** On a Gaussian field with given auto/cross spectra, the convolution map *and* the
> product map carry **exactly** the two-point information — no more (their entire law is a function
> of the spectra) and no less. Therefore **any** genuinely non-Gaussian information either operator
> can carry originates *solely* in the field's connected ≥3-point functions (bispectrum,
> trispectrum, …).

This is the load-bearing certainty. It also disposes of the recurring trap (handoff §Pitfalls,
"shape vs information"): a convolution map has a χ²/Wishart-like one-point PDF *even on a Gaussian
field*. That non-Gaussian **shape** is two-point-*determined* and carries **zero** non-Gaussian
information. "The map looks non-Gaussian" and "the map carries non-Gaussian information" are
different statements; only T1 makes the second one precise.

---

## 3. T2 — Moment/cumulant structure (DERIVED): where the trispectrum first enters

Because both operators are degree-2, their cumulants couple to the fields' n-point functions in a
fixed ladder. Writing the connected field correlators as P (power/2-pt), B (bispectrum/3-pt),
T (trispectrum/4-pt):

- **Mean** ⟨κ^ij_conv⟩, ⟨κ^ij_prod⟩ ~ ⟨a_i a_j⟩ → **2-point only** (P). No bispectrum, no
  trispectrum. (Odd/3-pt cannot enter the mean of a product of two fields.)
- **Variance / map covariance** ~ ⟨a_i a_j a_i a_j⟩_c → **Wick (P·P) + connected 4-point (T)**.
  The trispectrum enters *first* here, as a correction on top of the Gaussian (Wick) term.
- **Third cumulant** ~ ⟨(a_i a_j)³⟩_c → connected 6-point and its reducible pieces (including B²
  and T·P products). Bispectrum can appear here, at sixth order.

So for **either** operator: mean = pure two-point; the leading non-Gaussian entry is the
**trispectrum in the variance**; higher connected functions enter only at third cumulant and above.
Neither operator is "two-point only" on the real field — but the *signal* (the mean, the
parameter-dependent quantity inference latches onto most strongly) is two-point for both, and the
non-Gaussian content lives entirely in the higher cumulants sourced by T and up. The operators
differ **not in which correlators they touch** (both: T at the variance) but in *how much
independent, high-S/N access* they give to those correlators — which is T4.

---

## 4. T3 — The convolution's signal is two-point, but it is a *folded* transform of ξ_ij, not ξ_ij (CORRECTED)

**Convolution (the code).** κ^ij_conv[x] = Σ_r a_i[r] a_j[x−r] (proved from the DFT identity
irfft2(F_i·F_j) = circular convolution; derivation in §7 header). Its mean is
**E[κ^ij_conv(x)] = Σ_r W[r] W[x−r] ξ_ij[2r−x]** — a window-weighted *fold* of ξ_ij (substitute
u = 2r−x: = ¼ Σ_u W[(x+u)/2] W[(x−u)/2] ξ_ij[u]). This is a linear functional of the two-point
function, hence a **purely two-point signal**, but it is **not** ξ_ij(x): at zero lag it is
Σ_r W[r]²ξ_ij[2r], a broad window-weighted average over all lags, not ξ_ij(0). In Fourier space the
same fact reads ⟨â_i(k)â_j(k)⟩ = ∫ (d²q/(2π)²) P^×_ij(q) W̃(k−q) W̃(k+q): the convolution couples
mode k of field i to mode −k of field j (different modes), which for a homogeneous field vanishes in
the mean except near k = 0 — so it does **not** estimate the cross-spectrum P^×_ij(k) (that is the
job of the *correlation* â_iâ_j*, which couples same modes). Measured confirmation (§7): the
convolution mean has Pearson r = −0.35 with ξ_ij, the correlation-operator mean has r = 0.9996.

**Correlation (contrast, not used by the code).** X_ij[x] = Σ_r a_i[r] a_j[r+x] has
E[X_ij(x)] = ξ_ij(x)·(W⋆W)(x) — the clean lag-space cross-correlation modulated by the window
autocorrelation. This is the operator the sketch/paper *described*; it is not the one *built*.

**Product.** Spatial mean (1/N²)Σ_x κ_i κ_j = ξ_ij(0) exactly (measured ratio 1.0000, §7): a single
two-point number. Its *higher* one-point moments are the non-Gaussian part (T4).

**Upshot.** "The convolution re-encodes two-point information in lag space" is correct. "The
convolution *is* the empirical cross-correlation, up to a reflection" is **not** — a reflection of
one field turns convolution into correlation, but that reflection is not a symmetry of the joint
statistic and changes the mean from ξ-shaped to folded (an r = 0.9996 map into an r = −0.35 map).
The convolution is a *lossy, folded* re-encoding of the same two-point content the `cov` sector
already holds — which is exactly why it adds almost nothing on top of it (§6).

---

## 5. T4 — Why the convolution's non-Gaussian *information* is suppressed (RESOLVED: few-mode, not CLT)

The handoff flagged two competing stories: (i) CLT-Gaussianisation from summing many lag pairs, and
(ii) few large-scale modes on a 10° patch. They pull in opposite directions **as statements about
PDF shape**, and the manuscript currently asserts both ("strongly Gaussianised *and* a handful of
effective modes"). The resolution separates shape from information (T1):

**The operative regime is few-mode, and CLT-Gaussianisation is not active.** κ^ij_conv[x] =
Σ_k â_i(k)â_j(k) e^{ik·x} is dominated by the low-k mode products (κ has a red spectrum, and the
product â_iâ_j doubles the low-k weighting); on a 10° patch only a handful of large-scale modes carry
the bulk of the weight (independently established: the full-sphere cross power piles up at ℓ ≈ 60–90
with 12–20% of its variance at ℓ < 18, `07-discussion.tex` §7.1). A sum dominated by a few
non-Gaussian-in-distribution mode-bilinears is *not* pushed to a Gaussian by the CLT — CLT would
require many comparably-weighted independent terms, which is precisely what the patch lacks. So
"strongly Gaussianised" is the wrong mechanism.

**Why the information is nonetheless suppressed.** Two independent reasons, both robust:
1. **The signal is two-point by construction** (T3): ⟨κ^ij_conv⟩ is a functional of ξ_ij that the
   `cov` sector (P7) already contains in full. The convolution's *mean* therefore adds nothing to a
   datavector that already has the complete two-point sector.
2. **Its only non-Gaussian channel has negligible S/N.** The trispectrum enters only through the
   *scatter* of the convolution map (T2), i.e. through the connected 4-point of a few large-scale
   mode-bilinears. That channel is (a) amplitude-suppressed — the trispectrum/Wick ratio is small at
   these scales — and (b) sampled by very few independent modes, so its *parameter dependence* is
   measured with tiny signal-to-noise. Non-Gaussian information ∝ (parameter sensitivity of the
   trispectrum-sourced moments) × (number of independent modes); both factors are small.

**Contrast with the product, made precise.** κ^ij_prod(x) = κ_i(x)κ_j(x) is strictly local, so its
sample moments over the ~N_pix pixels directly estimate the *joint one-point moments*
⟨κ_iᵖ κ_jᵍ⟩ — which for p+q ≥ 3 are the *collapsed* poly-spectra (bispectrum at order 3, trispectrum
at order 4, evaluated at coincident points). The product hands the inference the joint one-point PDF
p(κ_i, κ_j), sampled at ~N_pix quasi-independent locations, so its non-Gaussian moments have real
S/N. The product's non-Gaussian channel is **wide** (N_pix samples) and **direct** (the collapsed
poly-spectra *are* its higher moments); the convolution's is **narrow** (few modes) and **faint**
(small trispectrum/Wick). This is the entire mechanism behind ΔNG(product) ≫ ΔNG(conv).

---

## 6. Mapping to the gated measurement (E1): the residual is measured, and it is negligible

Gated 2-point-split (`twopt_split/RESULT_TWOPT_SPLIT_FULL.md`; n=9000, 3 seeds, TARP+SBC).
`cov` = complete two-point sector (P7); adding a cross-map's ℓ₁ *on top of* a vector that already
contains `cov` isolates what is genuinely non-Gaussian:

| arm | FoM₃ | gate |
|---|---|---|
| cov (2-pt only) | 982 | — |
| auto_cov (auto-ℓ₁ ⊕ cov) | 2916 | PASS-with-caveat |
| conv_cov (conv-ℓ₁ ⊕ cov) | 3221 | PASS-with-caveat |
| product_cov (product-ℓ₁ ⊕ cov) | 3624 | **FAIL (over-confident)** |

- Positive control auto_cov − cov = **+1934**: the test *can* see non-Gaussianity (the autos' own).
- **ΔNG(conv) = conv_cov − auto_cov = +305** (~1.3σ of scatter). Both arms calibrated, but conv_cov
  is *slightly more* over-confident than auto_cov, so the true residual is **≤ 305 and marginal**.
  → The convolution cross-channels add essentially **no** genuine non-Gaussian inter-bin information
  beyond the complete two-point sector plus the autos' own non-Gaussianity. **Consistent with the
  theory (T3+T4): two-point signal, negligible-S/N trispectrum channel.**
- ΔNG(product) = +708, but product_cov **FAILS** calibration, so 708 is inflated by over-confidence
  (an **upper bound**, not a clean number). The *sign and clear excess* — product carries genuine
  non-Gaussian content — are unambiguous and match theory; the clean calibrated magnitude is the one
  **open** item (needs the 3-compressor deep ensemble, `run_bnt_autoprod_ensemble.py` pattern;
  `driver_ensemble.log` shows that run was underway — no `RESULT_TWOPT_SPLIT_ENSEMBLE.md` yet). This
  open item does **not** affect the convolution conclusion.
  **UPDATE 2026-07-01: the ensemble finished** (`RESULT_TWOPT_SPLIT_ENSEMBLE.md`): de-inflated
  ΔNG(conv) = **124**, ΔNG(product) = **260** (pooled-posterior TARP/SBC gate still to run). The
  de-inflation asymmetry (conv −200 vs auto −19) empirically confirms the "conv_cov more
  over-confident ⇒ ΔNG(conv) ≤ 305" direction.

Screen (`RESULT_TWOPT_SPLIT.md`; s41, n=1000, ungated) agrees: ΔNG(conv) = 361, ΔNG(product) = 610.

**E2 (Gaussianisation cross-check)** — the independent, decisive test (run conv-ℓ₁ on matched-spectra
GRFs vs N-body; a 2-point-only operator gives FoM₃(GRF) ≈ FoM₃(N-body)) — is **not required** for the
convolution claim: the qualitative answer is *proved* (T1–T3) and the residual is *gated-bounded*
(≤ 305, marginal). It remains the cleanest further hardening if an airtight *number* is ever wanted,
and would double as the clean product control. Not run here.

---

## 7. T5, T6, and the numerical verification

**T5 — Apodisation W.** W is a fixed, parameter-independent linear taper. The Gaussian baseline (T1)
is W-independent: a_i = Wκ_i is Gaussian whenever κ_i is, so κ^ij_conv is still a quadratic form in a
Gaussian vector — only Σ changes, not the fact that the law is a function of Σ. W re-weights the mode
coupling (controls edge/mode-coupling artefacts and ringing) but neither creates nor removes the
trispectrum coupling. So the 2-pt-vs-non-Gaussian answer is **W-independent**; W is a nuisance
control, not an information lever. If anything, apodisation slightly *reduces* the already-few
effective modes, mildly reinforcing the T4 suppression.

**T6 — Patch-local flat-sky vs full-sphere Zürcher.** The flat-sky κ^ij_conv is patch-local (each
cross pixel is a functional of the patch's own auto-maps only; docstring l.5–6). The full-sphere
a_lm-product is non-local — each cross pixel is a functional of the whole sky — and ≈92% of its
apparent cross gain is **super-patch leakage**, not physical patch information (`FLATSKY_CROSS_RESULT.md`;
`project_flatsky_cross_deleaked_result`; `07-discussion.tex` §7.1). The *kind* of information is the
same two-point-dominated bilinear content; the full-sphere excess is leakage, **not** extra
non-Gaussian signal. The paragraph describes the *patch-local* operator, and this is why.

**Numerical verification** (`scratchpad/verify_conv_mean.py`, run with the actual `flatsky_cross.py`
operators on correlated stationary GRFs, N=80, 4000 realisations, imposed cross-corr r=0.6):
- product spatial mean / ξ_ij(0) = **1.0000** (product mean = zero-lag two-point ✓);
- correlation-operator mean vs ξ_ij: Pearson **r = 0.9996** (peaks at zero-lag ✓);
- correlation-operator mean vs ξ_ij·(W⋆W): **r = 0.9999** (window-modulation prediction ✓);
- **code's convolution mean vs ξ_ij: r = −0.35**, peak off zero-lag, opposite sign at zero-lag
  (convolution ≠ cross-correlation ✓ — establishes C1);
- convolution mean sign-matches the fold E[C(x)] = Σ_r W[r]W[x−r]ξ[2r−x] at all sampled pixels.

---

## 8. Deliverable paragraph (D2) — proved vs measured

Drop-in replacement for `07-discussion.tex` §7.1, l.23–30 (and a shorter form can seed
`03-statistics.tex` §3.4). Uses only proved/measured statements.

> The two patch-local operators then explain the result of Sect.~\ref{sec:power}. Both are
> quadratic (bilinear) functionals of the convergence, so on a Gaussian field their entire
> statistics — every cumulant of every pixel — are fixed by the auto- and cross-spectra alone; any
> information beyond the two-point can therefore only come from the fields' connected
> higher-order correlators, entering first as the trispectrum in the maps' variance. The
> convolution's signal is purely two-point: its mean is a window-folded transform of the bin-pair
> correlation ξ_ij (not ξ_ij itself, and not the cross-correlation — the operator is a convolution,
> not a correlation), so its one-point statistics re-encode two-point information that the
> auto-maps and the power spectra already constrain. Its only route to non-Gaussian information is
> the trispectrum imprint on its fluctuations, and on a 10° patch — where the mode products are
> dominated by a few tens of effective large-scale modes — that channel carries negligible
> signal-to-noise; adding the convolution channels on top of the complete two-point sector
> improves FoM₃ by an amount consistent with zero within the calibration scatter (ΔNG ≲
> 3\times10^2 single-compressor, ≈1.2\times10^2 after ensemble de-inflation). The pointwise
> product is the local complement: its spatial mean is ξ_ij(0), but its higher one-point moments
> are the joint moments ⟨κ_i^n κ_j^n⟩ of the two bins — genuinely non-Gaussian inter-bin
> information, sampled pointwise across the patch — and it is this that the ℓ_1-norm turns into
> the +24\% gain.

**Proved:** quadratic ⇒ Gaussian-sector two-point completeness (T1); mean = two-point functional,
convolution ≠ cross-correlation (T3, C1); trispectrum-first moment ladder (T2); product moments =
joint one-point moments (T4). **Measured:** ΔNG(conv) ≤ 305 (~1.3σ, gated, both arms calibrated;
124 after ensemble de-inflation, pooled gate pending); product carries a clearly larger, genuinely
non-Gaussian gain (ensemble 260, pooled gate pending). T4 quantified in `CONV_MAP_SECURE_RESULT.md`
§6: conv N_eff = 30–208 modes with a +5–9% connected-4pt variance share, vs product N_eff ≈
1450–1860 with +36–75% and exkurt 608–961.

---

## 9. Downstream: what to fix in the manuscript

1. **`07-discussion.tex` §7.1 l.23–26** — replace "up to a reflection, the lag-space empirical
   cross-correlation" (C1: false) and "strongly Gaussianised" (C2: wrong mechanism) with the §8
   paragraph. The +9% / +24% headline gains and the leakage discussion are unaffected.
2. **`03-statistics.tex` §3.4 l.170–172** — "the convolution is a smooth, large-scale statistic,
   while the product … reduces, in the mean, to the standard two-point cross-correlation" is
   **fine** (it correctly attributes mean = ξ to the *product*). Optionally add one clause that the
   convolution's signal is likewise two-point (a folded transform of ξ_ij), for symmetry.
3. **Related open item I1** (`AUDIT_OPEN_ITEMS.md`, `06-bnt.tex` §6.2): the "0.38 Gaussian sector"
   wording is a *separate* issue (MAF-era auto-only number; see handoff §Pitfalls). This document
   does not resolve I1, but it does provide the clean, RealNVP-era statement of "what the two-point
   sector is" (P7 + the gated `cov` arm) that I1 should be reconciled against — do not reuse 0.38.
