# Flat-sky cross-map redesign — notes

**Purpose:** design a *physically valid, real-data-applicable* flat-sky tomographic cross-map
construction to replace (a) the leaky full-sphere harmonic construction
([[CROSS_MAP_LEAKAGE_FINDING]]) and (b) the flawed previous flat-sky attempt analyzed below.
Reference for the cross-map idea: Zürcher et al. 2022 (arXiv:2206.01450).

Status: **design discussion in progress** (2026-06-08). Nothing rebuilt yet.

---

## 1. Where the previous flat-sky implementation lives

The `--cross-maps` TFDS route (distinct from the `--full-sphere-cross-cache` harmonic route):
- Generation: `_compute_cross_maps_tf` / `_compute_cross_maps_np` + `_apod_window_np`/`_make_apod_window`
  in `npe_l1norm_nbody_tomo.py` (L171–183 flags, L394–442) and
  `npe_l1norm_cross_jaxili_nbody_tomo.py` (L215–267). Core: `irfft2(rfft2(κᵢ·w)·rfft2(κⱼ·w))`.
- Analysis/SNR: `compute_wavelet_transform(img, noise_sigma)` + `--cross-map-min/max-snr`,
  `--cross-map-auto-calibrate-snr`, `cross_snr_percentile`.
- Diagnostics: `diagnose_cross_maps.py`.

## 2. What it actually computes

`X[i]*X[j]` (product of FFTs, NO conjugate) → `irfft2` = **circular convolution κᵢ⊛κⱼ** — not the
cross-correlation, not the pointwise product. In Fourier it is `X̃ᵢ(k)·X̃ⱼ(k)`; since convergence
power falls steeply, the result piles at low-k ⇒ a **smooth, large-scale field**.

## 3. Numerical evidence (cached fiducial patches, 10°/80px)

**Scale (mean radial wavenumber, cycles/patch; lower = smoother):**
`auto = 18.3 | flat-sky cross = 9.3 | harmonic cross = 11.4` → flat-sky cross is ~2× smoother than
auto ⇒ structurally mismatched to the small-scale wavelet ℓ₁ statistic.

**Amplitude (correction to an earlier mistaken claim):** the ~10⁴× suppression is the HARMONIC
route, NOT flat-sky. Flat-sky cross std / auto std = 0.68×–1.77× (per pair); cross_std/σ_auto =
0.54–1.42. So the SNR is **not** collapsed — but the amplitude is physically arbitrary
(∝ √N·σ², FFT-convention-dependent) and varies ~2.6× across pairs.

**Construction (decisive):** Pearson r between flat-sky (convolution) and harmonic (alm-product)
cross-maps, same patch/pair = **≈ 0** (−0.03…+0.01). On a 10° patch the flat FFT *is* the local
harmonic transform, so if the harmonic cross were local these would nearly coincide. r≈0 ⇒ the
harmonic cross is dominated by non-local content ⇒ independent confirmation of leakage, AND the old
"flat-sky" maps were never a flat-sky *approximation* of the harmonic maps.

## 4. Re-contextualizing the old "flat-sky didn't help" verdict

Memory `feedback_l1_cross_must_use_harmonic_route` recorded flat-sky ~10k FoM3 vs harmonic ~40k and
called flat-sky "broken." But the harmonic 40k is **leakage-inflated**, so flat-sky 10k may be
closer to honest — yet it is *also* suppressed by the implementation flaws below. The old null
result cannot separate "little physical cross info" from "measured badly." A clean reimplementation
closes that gap.

## 5. Consolidated issue list

**Generation**
1. Operator choice: convolution (κᵢ⊛κⱼ) instead of pointwise product κᵢ·κⱼ or a proper
   cross-correlation/aperture statistic — least HOS-friendly choice; smears to large scales.
2. No physical normalization: amplitude ∝ √N·σ², per-pair spread ~2.6×.
3. Edge treatment: only an 8% cosine taper, no zero-padding ⇒ circular-convolution wrap (the doc's
   own warned-against pitfall).
4. Inconsistency: cross built from apodized inputs but raw (un-apodized) autos concatenated; cross
   channels not re-demeaned.
5. Two hand-synced implementations (tf train / np obs) — silent-divergence risk.

**Analysis**
6. Noise model: single auto pixel-σ for all channels; channel-aware σ exists but wired only to the
   harmonic route.
7. SNR calibration: percentile re-ranging treats binning, not the noise denominator.
8. Statistic–scale mismatch: small-scale ℓ₁ on a smooth field; coarse ℓ₁ bins sample-variance-limited
   at 80px.
9. Efficiency: FFTs recomputed inside the tf.data map every epoch (not cached).

## 6. Open design questions (to resolve next)

- **Operator:** pointwise product κᵢ·κⱼ vs convolution vs aperture-mass / filtered cross-correlation.
  Must be physically motivated AND computable on real (masked, finite) survey patches.
- **Normalization:** to physical units, channel-by-channel; consistent with a channel-aware noise model.
- **Flat-sky specifics:** apodization window choice, zero-padding to kill circular wrap, treatment of
  the mask/footprint, pixel window / beam, mode-coupling from windowing.
- **Statistic matching:** wavelet scales appropriate to the (smoother) cross field; or a different
  statistic for the cross channels.
- **Cross-check:** the flat-sky cross on a patch should agree with a masked-curved-sky computation on
  the same patch (sanity), and must NOT depend on data outside the patch (no leakage).

## 7. What Zürcher et al. 2022 actually do (arXiv:2206.01450, read 2026-06-08)

- **Cross map (their Eq. 12, verbatim):** κ_{i,j}(θ,φ) = Σ_ℓ Σ_m κ̂_{ℓm,i} · κ̂_{ℓm,j} · Y_{ℓm}(θ,φ),
  via HEALPix `alm2map`. Element-wise **product of the two bins' aₗₘ**, full-sky, NSIDE=1024,
  ℓmax=3071. i=j → (nonlinear) single-bin map; i≠j → cross map.
- **Multiscale filtering:** 12 Starlet (preferred) or Gaussian scaling functions, FWHM ∈
  [31.6, 29.0, 26.4, 23.7, 21.1, 18.5, 15.8, 13.2, 10.5, 7.9, 5.3, 3.3] arcmin; filtering done in
  harmonic space (ĉ_s(ℓ,m) = ψ̂_s(ℓ)·κ̂(ℓm)·√(4π/(2ℓ+1))).
- **Statistics:** peak counts, minima counts, Minkowski functionals on the **SNR-normalized**
  filtered maps. **SNR ≡ κ_filtered / σ(κ_filtered)** (σ = std of the filtered map); measured over
  SNR ∈ [−4, 4] in 10 bins.
- **E-modes only** (mask → E/B mode-mixing); full-sky maps cut to the survey footprint with the mock
  mask, rotated to make several realizations. Shape noise = random galaxy positions to target n_gal +
  intrinsic ellipticities (their Eq. 23) added to the shear, then convergence reconstructed.

## 8. The exact flat-sky correspondence — the convolution operator was RIGHT

In the flat-sky limit Yₗₘ→e^{iℓ·θ}, Σₗₘ→∫d²ℓ, κ̂ₗₘ→κ̃(ℓ), so Eq. 12 becomes
κ_{ij}(θ) = ∫d²ℓ κ̃ᵢ(ℓ)κ̃ⱼ(ℓ) e^{iℓ·θ} = (κᵢ ⊛ κⱼ)(θ) — a **convolution**, i.e.
`IFFT(FFT κᵢ · FFT κⱼ)`. Because the FFT of a real field is Hermitian, the product is Hermitian and
the inverse is automatically real (cleaner than the spherical case, where reality must be imposed by
hand on the aₗₘ product).

**⇒ RETRACTION of §5.1:** the repo's flat-sky FFT-product (convolution) is the *faithful* flat-sky
analog of Zürcher's alm-product. The operator was not the bug. (My earlier "use the pointwise product
instead" was wrong as a *Zürcher analog* — though see §10 for why a scale-matched product is still
worth considering on physical grounds.) The real flat-sky defects were §5 items 2–9
(normalization, edge/wrap, noise model, apodization-consistency, statistic-scale), plus the
full-sphere-slice leakage of the *harmonic* route.

**What the convolution measures:** its power spectrum ∝ Cᵢ(ℓ)·Cⱼ(ℓ) (product of the AUTO spectra),
not the cross-spectrum Cᵢⱼ. The genuine i–j cross-information enters through the **phase coherence**
of κ̃ᵢ and κ̃ⱼ (both trace the same LSS) → the map's morphology (peaks where both bins light up
together). So it is a non-Gaussian cross-*morphology* probe, not a 2-pt cross estimator — which is
the intended role (the 2-pt cross-info is in the power spectrum, measured separately).

## 9. Proper flat-sky construction — what the sphere→patch move forces us to add

The sphere is compact (no boundary, all modes present); a flat patch is finite (edges, missing
large scales). Going to flat-sky requires:

1. **Zero-padding → linear (not circular) convolution.** `IFFT(FFT·FFT)` on an N×N grid is a
   *circular* convolution (wraps opposite edges → spurious). Pad both maps to ≥2N−1, multiply,
   IFFT, crop the valid central region. (Old code: 8% taper only, no padding → wrap contamination.)
2. **Apodization + mask, forward-modeled.** Real patches are masked with hard edges → Fourier
   ringing. Apodize the field/mask with a smooth window before FFT. Apodization multiplies by W(θ),
   which couples modes and biases amplitude — but for SBI the clean fix is **consistency**: apply
   the *same* mask+apodization to sims and data (it's part of the deterministic forward model), and
   calibrate noise/normalization on the apodized sims. No MASTER-style deconvolution needed. This is
   directly real-data-applicable (you apply the survey footprint mask to both).
3. **Missing large-scale modes — the fundamental, honest limit.** The convolution cross map is
   large-scale-dominated (we measured ⟨k⟩≈9 cycles/patch ≈ scales ~67′, coarser than the autos'
   ~18). A 10° patch cannot measure modes larger than itself (ℓ≲18). So a patch intrinsically holds
   only the *sub-patch* slice of the cross information; the full-sky version (Zürcher) sees all the
   large-scale cross modes a patch cannot. **Expect modest cross gains over auto-only — that is the
   physically correct answer, and exactly what the leaky full-sphere route was hiding** (it smuggled
   the large-scale modes into every patch).
4. **Pixel window + multiscale Starlet filtering** on the cross channels too, same physical FWHM
   scales as the autos (3.3–31.6′ all fit inside 10°). Flat-sky starlet (à-trous / FFT) instead of
   harmonic.
5. **Noise propagation (the real fix for the old noise-model bug).** With noisy bins,
   (κᵢ+nᵢ)⊛(κⱼ+nⱼ) = κᵢ⊛κⱼ + κᵢ⊛nⱼ + nᵢ⊛κⱼ + nᵢ⊛nⱼ. The cross-map "noise" is **not** the auto
   pixel-σ — it is a convolution of noise with signal/noise, with its own (correlated) spectrum and
   amplitude. The SNR denominator must be the **cross channel's own** noise, estimated **per scale,
   per channel** from noise realizations (rotate galaxy ellipticities → recompute the cross map →
   take the std of each filtered noise cross map). Standard, real-data-applicable. The old code's
   shared auto-σ was wrong precisely because it ignored this.
6. **Normalization is then automatic.** Zürcher's SNR = κ_filtered/σ(κ_filtered): the arbitrary
   convolution amplitude (pixel-area × FFT convention) **cancels** as long as each channel is
   normalized by *its own* per-scale σ. The old bug was a *shared* σ; per-channel SNR fixes both the
   normalization and the noise-model issues at once.

## 10. A physically-motivated alternative worth a sensitivity test (NOT a unilateral change)

The literal convolution's 2-pt content is Cᵢ·Cⱼ and it smears to large scales (patch-limited).
A **scale-matched filtered product** — filter both bins with the same starlet ψ_s, then pointwise
multiply, [ψ_s*κᵢ](x)·[ψ_s*κⱼ](x) — is local, scale-preserving, and its mean is the scale-s
cross-correlation (directly Cᵢⱼ-like), capturing small-scale joint structure the convolution loses.
It is closer to third-order aperture-mass cross statistics and equally real-data-friendly. Downside:
it deviates from Zürcher's literal Eq. 12 (defensibility/referees). **Recommendation:** implement the
faithful convolution as the baseline (validates against Zürcher/harmonic) AND test the scale-matched
product as a possible improvement — decide together.

## 11. Validation / back-pressure (before any retrain)

The properly padded+apodized flat-sky cross on a patch should agree with a **masked-sphere**
computation on the *same footprint* (mask the sphere to the patch, SHT, alm-product, alm2map, read
the patch) — both are local. And both must DIFFER from the current full-sphere cross (Pearson r≈0,
§3) — that simultaneously validates the flat-sky approximation and re-confirms the leakage. Only
after this agreement do we rebuild + retrain.

## 12. CHEAP VALIDATION RESULTS (2026-06-08, `validate_flatsky_cross.py`)

Ran no-training construction checks on cached fiducial auto patches (10°/80px).

- **V1 — RETRACTED claim.** I initially reported the convolution as "boundary-dependent / ill-posed
  on a patch" (circular vs padded r=0.51, differ ~2× everywhere). **That was wrong — a registration
  artifact.** Convolution outputs are indexed by *lag*, and circular vs linear put lag-zero in
  different places, so I was correlating *shifted* versions of the same operation (a centered 'same'
  crop gives r≈0 purely from the shift). The gallery's faint upper-left in the "padded" panel was
  likewise my crop choice (the partial-overlap corner of the linear conv), not a defect. **Corrected
  conclusion: the apodized (optionally zero-padded) convolution is a perfectly usable patch operator.**
  The old convolution code's real problems were the *other* items in §5 (shared auto-σ noise model,
  no principled edge handling, ℓ₁ scale-mismatch), NOT the operator.
- **V2 — product mean is a direct cross-correlation probe (stands, with a fair reading).** Pointwise
  product mean: true bin pair +4.6e−5 vs bin × *independent*-patch bin +9e−8 → **529×** (the product
  mean *is* ξᵢⱼ). Convolution *variance* moved only 1.59× — fair reading: the convolution stores its
  cross-information in **morphology/phase**, which a variance doesn't capture (NOT "the convolution
  is blind"). ⇒ the two operators encode the correlation *differently* → complementary.
- **V3 — cross-map noise amplitude ≠ auto pixel-σ** (0.33×σ here) ⇒ SNR must use a per-channel
  (per-scale, on real coloured noise) estimate, not a shared auto-σ. (Stands.)
- The leakage finding (full-sphere → patch, [[CROSS_MAP_LEAKAGE_FINDING]]) is unaffected and stands.

## 13. AGREED PLAN (with Andreas, 2026-06-08)

- **Both operators are valid patch-local constructions and are likely complementary** (convolution =
  multiply in Fourier / mode-by-mode; product = multiply in real space / direct cross-correlation).
- **Operator = simple POINTWISE product** `κᵢ(x)·κⱼ(x)` (one cross map per pair), then the **existing
  multiscale wavelet-ℓ₁** on it — NOT the per-(pair×scale) scale-matched product. (Scale-matched is
  cleaner in principle but judged too complex for likely-small gain; **kept as a backlog item** to
  test if time permits.)
- **Experiment matrix** (all vs the auto-only baseline; patch-local cross only, no leakage):
  1. **auto + convolution-cross** (apodized κᵢ⊛κⱼ, one map/pair)
  2. **auto + product-cross** (κᵢ·κⱼ, one map/pair)
  3. **auto + BOTH cross sets** — run only if (1) and (2) show complementary information.
  Each arm for both L1 and CNN.
- **Per-channel, per-scale SNR/noise normalization** (noise from shape-rotations) for every cross
  set — fixes the old shared-auto-σ bug (V3).
- **Cheap correctness check still wanted:** the product map's mean reproduces the known tomographic
  cross-correlation ξᵢⱼ on sims (no training).
- **Still TODO before rebuild+retrain:** wavelet scale set; mask/apodization forward-model; the
  ξᵢⱼ-recovery check; consistent tf (train) / np (obs) implementations of both operators.
