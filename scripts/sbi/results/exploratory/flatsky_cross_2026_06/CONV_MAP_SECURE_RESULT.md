# SECURE result — adversarial re-verification of the convolution cross-map claims

**Answers `HANDOFF_CONV_MAP_SECURE.md`. Created 2026-07-01 by the adversarial-verification
session.** Object under audit: `CONV_MAP_INFO_CONTENT_THEORY.md` (D1) and its §8 paragraph (D2).
Stance was adversarial: every load-bearing claim re-derived or re-measured independently; the
prior session's numeric was NOT trusted (its known loose end is autopsied in §3).

**Bottom line: the theory writeup SURVIVES. No claim breaks.** Two claims are
CONFIRMED-WITH-CORRECTION (K5's "handful of modes" becomes measured numbers that are larger than
"a handful"; K6/K7 get better numbers from the now-finished ensemble). The D2 paragraph is safe
for the paper after the small wording updates in §10 (final version there, and applied to D1 §8).
The two manuscript corrections C1 ("up to a reflection…": **false, confirmed**) and C2
("strongly Gaussianised": **wrong mechanism, confirmed**) are both secured.

**Provenance.** Code: `cnn_sbi` git `4fee7d0` (dirty tree, but `flatsky_cross.py` unmodified).
Verification scripts + outputs archived in `conv_map_secure/`
(`secure_conv_mean.py`, `slope_seed_test.py`, `quantify_t4.py`, `t4_results.npz`).
Data: gate JSONs `analytical_nde_match/twopt_split/<arm>/gate/verdict.json`; fiducial ensemble
`results/exploratory/cross_maps_campaign/full_sphere_cache_fiducial_10deg/nobnt/obs/` (200 files
× 180 patches = 36 000); numpy 1.26.4, backward-FFT convention throughout.

---

## 1. Verdict table

| # | Claim | Verdict | Evidence (section) |
|---|---|---|---|
| K1 | Both operators are quadratic (bilinear) forms in the Gaussian field vector, A_p fixed & parameter-independent | **CONFIRMED** | §2 (re-derived; A_p written explicitly) |
| K2 | On a Gaussian field the map's entire law — hence any deterministic summary (ℓ₁, VMIM) — is a function of Σ only | **CONFIRMED** | §2 (MGF re-derived; ℓ₁/VMIM path audited for extra randomness — none that is parameter-dependent) |
| K3 | Non-Gaussian info only from connected ≥3-pt; trispectrum first in the variance; no bispectrum in the mean | **CONFIRMED** | §2 (ladder re-derived) + §4 check H (pure-3-pt pair → conv mean consistent with zero) + §6 (autos' variance NB=GRF to 0.01%, conv's variance excess = connected 4-pt) |
| K4 | Conv mean = window-fold Σ_r W[r]W[x−r]ξ_ij[2r−x], NOT ξ_ij, not the cross-correlation "up to a reflection" | **CONFIRMED** (loose end closed) | §3: operator ≡ circular convolution at 4.6e−16; fold formula verified by two independent exact implementations at ~1e−14; production-code MC slope 0.9966±0.0095; prior 0.75–2.7 ratios fully attributed to MC noise |
| K5 | Conv's NG info suppressed by the few-mode regime, not CLT Gaussianisation | **CONFIRMED-WITH-CORRECTION** | §6: measured. N_eff(conv) = 30–208 (pair-dependent) — the right regime but not literally "a handful"; conv map exkurt on N-body 0.30–0.57 (NOT Gaussianised, comparable to the autos' 0.29–0.97); nonzero exkurt 0.05–0.26 even on GRFs (shape ≠ info, demonstrated) |
| K6 | Gated ΔNG(conv) ≤ 305 and marginal | **CONFIRMED** (and strengthened) | §5: conv_cov sbc_std > auto_cov on all 3 seeds (0.3097/0.3089/0.3004 vs 0.2998/0.3033/0.2975; ideal 0.2887) ⇒ conv more over-confident ⇒ ≤305 direction correct. Ensemble de-inflation confirms empirically: conv −200 vs auto −19 → ΔNG(conv) = **124** |
| K7 | ΔNG(product)=708 inflated (arm FAILs); clean magnitude open, pending ensemble | **CONFIRMED-WITH-CORRECTION** (open item now closed) | `RESULT_TWOPT_SPLIT_ENSEMBLE.md` exists since the handoff: de-inflated ΔNG(product) = **260** (pooled gate still to run). Sign/excess unambiguous |
| K8 | Answer W-independent (T5); full-sphere excess is leakage, same kind of info (T6) | **CONFIRMED** | T5 follows from K2 re-derivation (Wκ is Gaussian when κ is; only Σ changes). T6 quote spot-checked: `FLATSKY_CROSS_RESULT.md` — ~92% of full-sphere ℓ₁ cross gain was leakage; ≈99% for the conv operator; matches `07-discussion.tex` §7.1 |

No claim is BROKEN.

---

## 2. P0.1 — T1/T2 re-derived from scratch (independent of D1)

**K1 (quadratic forms).** Stack x = (κ_i-pixels, κ_j-pixels) ∈ ℝ^{2N²}. Each conv output pixel
C[p] = Σ_r W[r]W[p−r] κ_i[r]κ_j[p−r] = xᵀA_p x with the symmetrized fixed matrix
[A_p]_{(1,r),(2,s)} = [A_p]_{(2,s),(1,r)} = ½·W[r]W[s]·δ_{s,(p−r) mod N}: entries built from W
(fixed, LOCKED roll_frac=0.10) and index bookkeeping only — no cosmology anywhere. The product's
A_p is the coincident-pixel block δ_{r,p}δ_{s,p}. ✓

**K2 (Σ-completeness).** For x ~ N(0,Σ) and any finite set {A_p},
E[exp(Σ_p t_p Q_p)] = det(I − 2 Σ^{1/2}(Σ_p t_p A_p)Σ^{1/2})^{−1/2} (Gaussian integral, small t):
a function of Σ and the fixed A_p alone, so the JOINT law of the whole map is Σ-determined.
Any deterministic map applied afterwards (starlet transform → frozen-σ SNR binning → ℓ₁; or the
trained VMIM compressor, which is a FIXED function at evaluation time) preserves this: a fixed
measurable function of a Σ-determined law has a Σ-determined law. Audit of the actual path for
sneaked-in randomness: training-time flip augmentation and compressor seeds have
parameter-independent laws (and are frozen at eval); frozen sigma tables are constants;
dequantization noise is used only by the `full4d` arm, not conv/product ℓ₁. ✓ Σ is exactly the
auto+cross two-point sector (P7, reused not re-proved).

**K3 (moment ladder).** Mean: E[C[p]] = Σ_r W[r]W[p−r] E[κ_i[r]κ_j[p−r]] — linearity of
expectation, only the cross two-point enters; a bispectrum CANNOT appear (it would need a third
field factor). Variance: Cov(C[p],C[q]) = Σ W·W·W·W ⟨κ_iκ_jκ_iκ_j⟩, and
⟨κ_iκ_jκ_iκ_j⟩ = Wick(P_ii P_jj + P_ij P_ij terms) + T_conn: the connected 4-pt (trispectrum)
enters FIRST here. Third cumulant: connected 6-pt and reducible B², T·P pieces. Same ladder for
the product (A_p diagonal). ✓ Numerical falsification test (§4, check H) agrees.

---

## 3. P0.2 — the fold formula CLOSED (the K4 loose end)

`conv_map_secure/secure_conv_mean.py`, checks A–D. Chain of evidence:

1. **[A] The operator IS the circular convolution** (this makes the fold formula one line of
   algebra): `irfft2(rfft2(a)·rfft2(b))` vs brute-force Σ_r a[r]b[x−r] double loop:
   max rel err **4.6e−16**. Production `build_channels_np` conv channel vs float64 replica:
   **4.4e−08** (float32 roundoff). PASS.
2. **[B] The discrete fold prediction is exact algebra**, verified by two INDEPENDENT
   implementations: real-space E[C(x)] = Σ_u W[u]·W[x−u]·ξ[x−2u] (roll-sum) vs Fourier-space
   E[Ĉ(k)] = (1/N²)Σ_q P(q)Ŵ(k−q)Ŵ(k+q) → ifft2. Max rel err **≤1.4e−14** on red, white and
   bump spectra. PASS. (This is the "to roundoff" closure the handoff asked for — it lives at
   the algebra level, where "roundoff" is meaningful.)
3. **[C] The production code + ensemble conventions have no systematic**, MC at N=80 with the
   real apodisation: M=200 000, per-pixel z-scores rms 0.965, max|z| 3.02 over 6400 pixels —
   consistent with pure MC noise. Regression slope of measured mean on prediction, over
   **10 independent seeds** (`slope_seed_test.py`): **0.9966 ± 0.0095** (r=0.6) and
   **1.0035 ± 0.0107** (r=1 same-field, high signal). Slope = 1 within ~1%. PASS.
4. **[D] The prior session's 0.75–2.7 ratios: fully attributed, no formula problem.** At the
   prior M=4000 and its 5 sample pixels, the per-pixel prediction SNR is only 1.2–3.2, so ratios
   of 0.73, 1.74, even −0.32 correspond to z = +0.8, −0.9, +1.5 — ordinary MC noise (plus its ξ
   was itself MC-estimated). Subtlety worth recording: the conv mean-error map is dominated by a
   few low-k modes, so per-pixel ratios and even the global slope swing coherently by ±2–3% at
   fixed M — the 6400 pixels are nowhere near 6400 independent checks. (This correlated-noise
   structure is itself the few-mode regime of T4, showing up inside the verification.)

**Two small corrections to D1's presentation of T3 (conclusions unaffected):**
- D1 §4's parenthetical substitution "= ¼ Σ_u W[(x+u)/2] W[(x−u)/2] ξ_ij[u]" is a
  continuum-only identity. On the discrete torus, r ↦ 2r−x mod N is 2-to-1 onto a half-lattice
  (for even N), so the ¼-sum with half-integer indices is not the discrete statement. The primary
  formula Σ_r W[r]W[x−r]ξ_ij[2r−x] is the exact one (and is what was verified).
- D1 §4's zero-lag special case should read Σ_r W[r]W[−r mod N]ξ[2r] (W[−r] ≈ W[r] only up to
  the one-pixel grid offset of the symmetric window).

---

## 4. P0.3 — C1 is structural and spectrum-independent

Sweep (`secure_conv_mean.py` [E/F/G/H]; M=60 000 each; N=80, real window):

| spectrum | r | z_rms (conv vs fold) | Pearson(conv mean, ξ) | Pearson(corr-op mean, pred ξ·(W⋆W)) | sign conv(0)/ξ(0) |
|---|---|---|---|---|---|
| red | 0.3 | 0.91 | **−0.30** | +0.9999 | −/+ |
| red | 0.9 | 0.84 | **−0.33** | +1.0000 | −/+ |
| white | 0.6 | 0.99 | **+0.005** | +1.0000 | +/+ |
| bump | 0.3 | 0.98 | −0.03 (pred ≈ 0) | +0.9999 | −/+ |
| bump | 0.9 | 1.01 | −0.06 (pred ≈ 0) | +1.0000 | −/+ |

- **The invariant claim:** conv mean = the window-fold of ξ (z-consistent everywhere); it is
  never ξ-shaped (Pearson −0.33 … +0.005, spectrum-dependent), while the correlation operator's
  mean tracks ξ·(W⋆W) at 0.9999+ in every case. The **−0.35 of D1 is red-spectrum-specific**
  (D1 already treats it as a measurement, correctly; it must not be quoted as universal, nor the
  zero-lag sign, which flips between spectra).
- **Strongest counterexample to the paper's identification:** for the band-pass (bump) spectrum
  the conv mean nearly VANISHES (prediction SNR ≈ 0 at M=60 000) while ξ is O(1) — because
  E[â_i(k)â_j(k)] couples modes k and −k and only survives within a window-width of k=0. A map
  whose mean can be ~zero while the cross-correlation is large is not "the cross-correlation up
  to a reflection" in any useful sense.
- **No linear rescue exists** ([G] + proof): conv map Fourier content is {â_i(k)â_j(k)}; the
  correlation map's is {â_i(k)â_j(−k)}. These are linearly independent monomials in the field
  variables, so NO fixed linear map (reflection, roll, fftshift, any matrix) turns the conv MAP
  into the correlation MAP realization-by-realization; the reflection identity conv(a, flip b) =
  corr requires flipping an INPUT before convolving, which the code never does. Measured: best
  normalized match of the corr map over all 4 flips × all 6400 rolls of the conv map =
  0.63–0.70 (control: self-match 1.000).
- Falsification check [H]: for κ_j = κ_i²−mean (zero cross-2-pt, maximal dependence, strong
  3-pt), conv mean and product spatial mean are consistent with zero (z_rms 1.19 given
  correlated pixels; product z=+2.05 ≈ noise) — the means are blind to everything beyond the
  cross two-point, as K3 requires.

---

## 5. P0.4 + P0.5 — downstream transforms and the K6 inequality

**P0.4 (reflection hunt): none exists.** Audit of the consumption path:
`build_channels_{np,torch,jax}` concatenates [autos | conv | product] with no shift/roll/
reflection (`flatsky_cross.py:255–274, 316–336, 373+`); the ℓ₁ path goes
`build_and_l1 → build_channels_torch → starlet wavelet → frozen-σ SNR ℓ₁`
(`flatsky_cross_l1.py:151–166`) with no spatial transform in between; the docstring's 39-px
lag-registration shift belongs to the DROPPED zero-pad+crop variant (`flatsky_cross.py:11–13`) —
the kept variant has none. Training-time random flips (loader `flip=True`;
`_harmonic_random_flip`) flip BOTH input autos identically, which maps the conv map to a
rolled flip of itself (still a convolution) — they cannot manufacture a correlation. And by the
§4 monomial argument no linear transform could rescue the wording anyway.
**⇒ C1 stands at full strength: "not the cross-correlation", not a softened "fold vs reflection".**

**P0.5 (K6 direction): settled, twice over.**
- Gate JSONs (read raw): sbc_std conv_cov = [0.30972, 0.30894, 0.30038] vs auto_cov =
  [0.29984, 0.30328, 0.29750]; ideal 1/√12 = 0.28868. Both arms slightly over-confident
  (U-shaped ranks); **conv_cov more so on all 3 seeds** ⇒ conv_cov's FoM₃ more inflated ⇒
  ΔNG(conv) = 305 is an upper bound. Direction as claimed.
- Empirical confirmation: the 3-compressor ensemble (`RESULT_TWOPT_SPLIT_ENSEMBLE.md`, completed
  after the handoff was written) de-inflates conv_cov by −200 (3221→3021) vs auto_cov by only
  −19 (2916→2897): exactly the asymmetry the SBC comparison predicts. **De-inflated
  ΔNG(conv) = 124.** (Ensemble ΔNG(product) = 260; pooled-posterior TARP/SBC gate still to run,
  precedent: joint-ℓ1 and BNT-autoprod ensembles calibrated.)
- Provenance of "~1.3σ" (undocumented in D1): reconstructed as the joint-ℓ1 compressor-seed
  band {3754, 3761, 4034} ⇒ σ(FoM) ≈ 160, σ(ΔFoM) ≈ √2·160 ≈ 226 ⇒ 305/226 ≈ 1.35. Borrowed
  from a richer arm of the same pipeline — defensible as an order of magnitude, but the better
  quotable statement is now: **ΔNG(conv) = 124 after ensemble de-inflation (≈ 0.5σ of that
  scatter), ≤ 305 single-compressor.**

---

## 6. P1.2 — T4 quantified (adjectives → numbers)

`conv_map_secure/quantify_t4.py` on the full fiducial ensemble (36 000 patches — the same obs
maps the gated FoM used), pushed through the REAL `build_channels_np(op='both')`, against
36 000 matched GRF patch sets with exactly the measured 4×4 spectral matrix S_ij(k)
(eigen-factor mixing, Hermitian-mirrored; match validated: auto spectra ratio 1.0000, cross
1.0001). The GRF is the Wick baseline; N-body excess = connected ≥4-pt. Internal control: the
auto channels' variance matches GRF to <0.02% (z≈0) — a linear channel's variance is pure 2-pt,
and the matching is exact.

| channel | conn-4pt share of Var (T/Wick) | z | exkurt N-body | exkurt GRF | N_eff (PR) | n90 |
|---|---|---|---|---|---|---|
| auto0–auto3 | −0.005% … +0.013% | ≈0 | 0.29–0.97 | 0.000 | 720–1491 | 1242–1421 |
| conv01–conv23 | **+5.2% … +9.3%** | 25–33 | 0.30–0.57 | 0.05–0.26 | **208 → 30** (01→23) | 494–989 |
| prod01–prod23 | **+36% … +75%** | 69–96 | **608–961** | 6.5–9.6 | 1442–1861 | 2185–2454 |

(N_eff = participation ratio of the map-variance over unique Fourier modes, of 3240; n90 =
modes holding 90% of the variance. Maps are the pipeline's own fiducial patches; their spectra
show no white-noise plateau at 7.5′ resolution — noise, where present in the pipeline, is baked
in upstream of these caches and any Gaussian noise only dilutes both NG channels.)

**Readings.**
- **Few-mode, quantified:** conv N_eff = 30–208 (deepest pairs fewest) vs 700–1500 for the autos
  and 1450–1860 for the product. "A handful" in the old wording is an overstatement — the right
  phrase is "a few tens to ~two hundred effective modes" — but the REGIME (an order of magnitude
  fewer effective modes than either autos or product) is confirmed.
- **CLT-Gaussianisation is not active, demonstrated both ways:** the conv map on N-body inputs
  keeps exkurt 0.30–0.57 (comparable to the raw autos — NOT "strongly Gaussianised"), and on
  exactly-Gaussian inputs it has exkurt up to 0.26 (χ²-like shape with zero non-Gaussian
  information — the shape≠information pitfall made concrete). C2's correction stands.
- **Faint vs direct, quantified:** conv's only NG channel (connected-4pt in its fluctuations) is
  +5–9% of map variance on those few modes; the product's is +36–75% of variance with
  three-orders-of-magnitude kurtosis excess (608–961 vs 6.5–9.6) sampled at ~N_pix pixels. This
  is the measured version of "narrow & faint vs wide & direct", and it is consistent with the
  gated ΔNG(conv) = 124–305 (marginal) vs the product's clear excess.
- Note the conv connected share is NOT zero (z≈30 with 36 000 patches) — T2 requires it to be
  nonzero (trispectrum enters the variance). The paper claim is about its negligible
  *information* at patch S/N, which the gated ΔNG measures directly; do not quote T4 as "conv is
  exactly Gaussian-equivalent".

---

## 7. P1.1 — E2 go/no-go

**No matched-spectra GRF generator existed in the repo** (grep: no synfast/grf/lognormal
generation anywhere in `scripts/`). **One now exists at map level**: `quantify_t4.py` contains a
validated 4-bin stationary GRF sampler matched to an arbitrary measured S_ij(k) (validated to
1e−4 on 36 000 patches). What full E2 still needs on top:
1. per-cosmology S_ij(k) from the TFDS train grid (~hours, CPU; needs a binning/smoothing choice
   for the per-cosmology spectral estimates);
2. a GRF training+fiducial suite mirroring the TFDS layout (generator exists; plumbing ~a day);
3. conv_cov + product_cov (+auto_cov control) retrained on GRF: ≈2 GPU-days of the
   twopt_split-style runs incl. gates.

**Recommendation: DEFER.** The conv conclusion is already (i) proved at the mean level (§2–4),
(ii) gated at the information level (ΔNG ≤ 305, ensemble 124), and (iii) mechanism-quantified
(§6, which is itself a map-level Gaussianisation test: the GRF-vs-N-body comparison at the
moment level). Full E2 adds an inference-level airtight number but does not change any
conclusion; run it only if a referee (or the BNT story) demands the number. Scope if wanted:
~1 day setup + ~2 GPU-days.

---

## 8. What was NOT verified here (honest residuals)

- The **ensemble gate** (TARP/SBC on the pooled twopt-split posteriors) has not run; ΔNG=124/260
  are de-inflated but not yet gate-stamped. (Precedent says they will pass; still, run it before
  quoting 124/260 as *calibrated* numbers anywhere.)
- P2.1 (finer band-power `cov` stability) not run — for conv it can only shrink ΔNG further.
- The exact provenance of the original "~1.3σ" phrasing (§5) is reconstructed, not documented.
- T6's leakage numbers were spot-checked against `FLATSKY_CROSS_RESULT.md` (92% cross-gain
  leakage; ≈99% for the conv operator), not re-measured.

---

## 9. Manuscript corrections — final wording (do NOT edit the paper without Andreas's sign-off)

- **C1** (`07-discussion.tex` §7.1 l.23–24). "The convolution map is, up to a reflection, the
  lag-space empirical cross-correlation of the two bins" — **false; replace at full strength**
  ("not the cross-correlation"; the relation to ξ is a window-fold). Secured by §3+§4+§5: the
  operator is a convolution; no downstream transform, and provably no linear transform at all,
  rehabilitates the reflection wording; the mean can even vanish for band-limited spectra where
  ξ is O(1).
- **C2** (§7.1 l.26). "strongly Gaussianised and compresses to a handful of effective modes" —
  drop "strongly Gaussianised" (wrong mechanism, §6); replace "handful" with the measured
  few-tens-to-~200 N_eff, or a non-numeric "a few tens of effective modes" if numbers are
  unwanted there.
- `03-statistics.tex` §3.4 l.163–165 & 170–172: correct as-is (mean=ξ is attributed to the
  product). Optional tightening only: "spatial mean is the bin-pair correlation ξ_ij" →
  "…the zero-lag bin-pair correlation ξ_ij(0)"; optionally add the one clause that the conv's
  mean is likewise two-point (a window-fold of ξ_ij).
- I1 (0.38 wording in §6.2) remains out of scope, as instructed; nothing here uses 0.38.

---

## 10. D2 paragraph — final verified version (applied to D1 §8 in place)

Changes vs the audited version: "a handful of large-scale modes" → measured range; ΔNG updated
to quote the ensemble de-inflation; one clause tightened ("consistent with zero" kept, now
0.5–1.3σ). Everything else survives verification verbatim.

> The two patch-local operators then explain the result of Sect.~\ref{sec:power}. Both are
> quadratic (bilinear) functionals of the convergence, so on a Gaussian field their entire
> statistics — every cumulant of every pixel — are fixed by the auto- and cross-spectra alone;
> any information beyond the two-point can therefore only come from the fields' connected
> higher-order correlators, entering first as the trispectrum in the maps' variance. The
> convolution's signal is purely two-point: its mean is a window-folded transform of the
> bin-pair correlation ξ_ij (not ξ_ij itself, and not the cross-correlation — the operator is a
> convolution, not a correlation), so its one-point statistics re-encode two-point information
> that the auto-maps and the power spectra already constrain. Its only route to non-Gaussian
> information is the trispectrum imprint on its fluctuations, and on a 10° patch — where the
> mode products are dominated by a few tens of effective large-scale modes — that channel
> carries negligible signal-to-noise; adding the convolution channels on top of the complete
> two-point sector improves FoM₃ by an amount consistent with zero within the calibration
> scatter (ΔNG ≲ 3×10² single-compressor, ≈1.2×10² after ensemble de-inflation). The pointwise
> product is the local complement: its spatial mean is ξ_ij(0), but its higher one-point moments
> are the joint moments ⟨κ_iⁿκ_jⁿ⟩ of the two bins — genuinely non-Gaussian inter-bin
> information, sampled pointwise across the patch — and it is this that the ℓ₁-norm turns into
> the +24% gain.

**Safe to insert into the paper: YES**, in this v2 form (or the audited v1 with only the
"handful"→"few tens" and ΔNG-number updates), after Andreas's sign-off. If the ensemble numbers
are to be quoted, run the pooled gate first (§8, first bullet).
