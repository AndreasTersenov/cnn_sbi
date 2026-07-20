# Paper scientific synthesis — L1-norm vs CNN-VMIM for tomographic weak-lensing SBI

**Stage-2 deliverable; the primary input to the `paper-draft` skill.** Written 2026-06-09 from the
Stage-1 evidence base (`PAPER_FILE_TRIAGE.md`, `PAPER_FIGURE_INVENTORY.md`) under *newest-wins with
provenance*. Companion: `PAPER_NARRATIVE_AND_PITFALLS.md` (the full journey + pitfalls catalog).
Trust screen: 3 hard invalidators (mass-sheet leak, L1 cross noise-model bug, full-sphere cross-map
leakage) + 3 flags (FoM3-fragility, NDE-architecture, train/test overlap); see
`PAPER_TRIAGE_WORKFLOW_SPEC.md`.

**Two framing principles for the paper (Andreas):**
1. **FoM3 may be headlined** (the "use 2D instead" rule is retired); we still report σ and 2D areas
   alongside, and acknowledge FoM3's correlation-amplification as a caveat.
2. **The paper showcases the journey, not just the final numbers.** The thorough ruling-out of
   confounds and bugs is itself a contribution — it is *why* the final result is trustworthy, and it
   is a service to the community (a pitfalls catalog). The "what we tried / why it failed" material
   lives in the companion doc and is surfaced in this paper as (a) woven "what we ruled out" subtext,
   (b) a standalone methodological-reversal section, and (c) an appendix/pitfalls catalog.

---

## 1. Scope, objectives, goals

**One sentence.** Using simulation-based inference on CosmoGridV1 weak-lensing convergence maps, we
compare the cosmological constraining power of two summary statistics — the **wavelet ℓ₁-norm** and a
**CNN compressor trained with the VMIM objective** — in a **tomographic** setting, determine the best
architecture for each, and ask how best to build **cross-bin maps** to extract extra information; and
we use the same machinery to revisit the long-standing claim that **BNT pre-processing inflates
higher-order-statistic contours**.

The work has **two pillars** that share one mechanism — *inter-bin (cross-bin) tomographic
information, and whether a given compressor recovers it*:

### Pillar 1 — L1-norm vs CNN-VMIM constraining power, and the cross-map strategy
- Compare ℓ₁ vs CNN-VMIM summaries for inferring **θ = [Ωm, σ8, w0, h0, ns, Ωb]** (the
  lensing-constrained subspace is **[Ωm, σ8, w0]**), at fixed survey assumptions, via SBI.
- Find the architecture that gives each compressor its best contours.
- Determine the **best strategy to construct cross-maps** (κᵢ×κⱼ between tomographic bins) that add
  *physical* cosmological information beyond the per-bin auto-maps.

### Pillar 2 — BNT + higher-order statistics: the contour-inflation question
- Prior work (including the authors') found that applying the **BNT (Bernardeau–Nishimichi–Taruya)
  nulling transform** before a higher-order statistic (e.g. the ℓ₁-norm) **inflates** the cosmological
  contours.
- **Thesis:** BNT is an **invertible linear** transform, so no information can be lost in principle.
  The inflation comes from a **per-channel statistic failing to recover the inter-bin
  cross-correlations** — BNT decorrelates the *signal* across bins while *correlating the noise*,
  which lowers the SNR of a statistic that does not model the cross-bin structure.
- **Prediction:** a **CNN compressor fed the tomographic auto-maps as channels**, trained with VMIM,
  recovers the implicit cross-correlations itself ⇒ **no information loss ⇒ no contour inflation**.

**In/out of scope.** In: the methods comparison on simulations, calibration, the cross-map
construction question, the BNT mechanism. Out (future, see §7): the physically-defensible flat-sky
auto+cross numbers, and the clean BNT-CNN no-inflation demonstration.

---

## 2. The narrative arc (the throughline)

The scientific spine of the paper is a single throughline — **which compressor recovers inter-bin
information, and how that interacts with map construction and survey geometry** — told through a
genuine reversal that is the reason to trust the final answer:

1. Early tomographic results (20° patches, full-sphere harmonic cross-maps) appeared to show **L1
   beating CNN by ~2.5–3× on auto+cross**, concentrated in w0 — a striking "simple statistic wins"
   story.
2. That headline did not survive scrutiny. It dissolved through a sequence of corrections — a
   cross-channel **noise-model bug**, **FoM3** correlation-amplification, a single-perm favorable
   draw, and an **NDE-architecture confound** (the two compressors were compared through *different*
   density estimators).
3. Putting both compressors through a **common density estimator**, at a **more flat-sky-valid 10°
   patch size**, with a robust per-patch population analysis, **overturns the headline**: **CNN ≥ L1**
   — a tie on auto-only, CNN clearly ahead on auto+cross — and **L1's apparent w0 advantage was a
   flat-sky-geometry artifact**, not an intrinsic property of the ℓ₁ statistic.
4. The remaining puzzle — *why does the CNN gain so much from explicit cross-maps when, in principle,
   it could compute cross-correlations from the auto channels itself?* — has a clean answer that is a
   contribution in its own right: **the cross-maps, as built, leak full-sky information**, so they
   carry non-local modes that are simply not present in a local auto-patch. This makes the auto+cross
   constraining power **partly unphysical** and motivates the decisive follow-up (flat-sky rebuild).

The same throughline carries Pillar 2: BNT inflation is the inter-bin-information question viewed
through a linear transform, and the CNN's channel-wise access to the bins is predicted to neutralize
it. (Full chronology and every dead-end in `PAPER_NARRATIVE_AND_PITFALLS.md`.)

---

## 3. Methods (what the paper needs to describe)

**Simulations & data.** CosmoGridV1 N-body weak-lensing convergence maps (`stage3_forecast`), 4
tomographic bins. Maps are projected to flat-sky patches; **shape noise is injected** at the map
level. The production geometry for the final comparison is **10°×10° patches at 80 px** (~7.5
arcmin/pix), non-overlapping, restricted to **|lat| < 75°** (180 polar-safe patches/sphere); an
earlier 20°×20° / 160 px geometry was used through most of the development and is retained for
diagnostics. **Per-pixel spatial demeaning (`--zero-mean-maps`)** is applied — removing the
mass-sheet (mean-convergence) signal that is not observable in real data (see Pillar-1 pitfall).

**Parameters.** θ = [Ωm, σ8, w0, h0, ns, Ωb], with h0 = H0/100 applied in preprocessing. FoM and 2D
areas are reported on the lensing-constrained subspace **[Ωm, σ8, w0]**; the nuisances h0/ns/Ωb are
weakly constrained.

**Summary 1 — the wavelet ℓ₁-norm.** Starlet wavelet transform (5 scales), per-channel ℓ₁ datavector
built as histograms of wavelet-coefficient SNR (≈40 bins/scale). The per-channel noise σ used to set
the SNR scale is estimated **per channel** from the data (`channel_empirical_global`) — *critical* for
the cross channels, whose amplitudes are ~10⁴× smaller than the autos (see Pillar-1 pitfall). **No
PCA** on the ℓ₁ datavector (a hard project rule — PCA craters the information). For the auto-map case
this is a per-bin statistic; with cross-maps it is computed on the 4 auto + 6 cross channels.

**Summary 2 — the CNN-VMIM compressor.** A CNN maps the tomographic maps (auto bins as channels, and
optionally the cross channels) to a low-dimensional summary (cdim ≈ 10), trained with the
**Variational Mutual-Information Maximization (VMIM)** objective (a companion normalizing flow
provides the variational bound during compression). Architecture family ∈ {plain, resnet*,
resnet50_gn}; on multi-channel (10-channel) harmonic input, **GroupNorm (`resnet50_gn`) or a plain
CNN** must be used — stock BatchNorm ResNets collapse because BN running statistics average across
cosmology-mixed batches (see pitfall).

**Cross-map construction (the strategy question).** Two routes:
- **Full-sphere harmonic** (used in the campaigns): κ^{ij} = iSHT(aⁱ_ℓm · aʲ_ℓm) on the whole HEALPix
  sphere, then gnomonic patch cutout. *This is the route behind all current auto+cross numbers, and it
  leaks full-sky information into every patch* (§5.5).
- **Flat-sky / patch-local** (the proposed, physically-defensible route; future work): build the cross
  map from the patch's own two auto maps, e.g. apodized κᵢ⊛κⱼ (convolution) and/or κᵢ·κⱼ (pointwise
  product). Strictly local; what a real patch survey could actually compute.

**Density estimator (NDE).** The inference uses Neural Posterior Estimation. Two estimators appear in
the project: an in-repo conditional **RealNVP** (`sbi_lens`) and a **jaxili Masked Autoregressive Flow
(MAF)**. **The definitive comparison puts both compressors through the *same* jaxili MAF** so the
compressor — not the flow — is what is compared.

**Calibration battery.** Three independent tests, all run on the final comparison: **TARP-DRP**
(varied-θ distance-to-random-point coverage, stratified by FoM3 tercile), **SBC** (simulation-based
calibration rank uniformity over validation cosmologies), and **L-C2ST** (local classifier
two-sample test at the fiducial). Calibration is verified *before* any constraining-power claim.

**Reproducibility discipline.** zero-mean maps; example-disjoint compressor/NDE split (by noise
permutation); per-channel cross noise; no-PCA on ℓ₁; GPU-pinned runs; per-run posterior `.npy` +
`.meta.json` + metrics. (The project's own SBI protocol, `skills/sbi/SKILL.md`.)

---

## 4. Pillar 1 results — L1-norm vs CNN-VMIM (final, trustworthy)

**Source of record:** the definitive 10° campaign,
`scripts/sbi/results/exploratory/definitive_comparison_10deg/phase_c/analysis/SUMMARY_PHASE_D.md`
(+ `OFFSET_VERDICT.md`); memory `project_10deg_definitive_cnn_geq_l1`. Both compressors through a
common jaxili MAF; robust **median over 9000 obs/arm** (180 patches × 50 noise perms), 3-seed-pooled.
**CITE** in the triage. Numbers verified to clear the hard invalidators (zero-mean maps; channel-aware
cross noise via the harmonic cache; common MAF; example-disjoint split). The **auto+cross** rows are
flagged **PROVISIONAL** under the cross-map leakage finding (§5.5).

### 4.1 Constraining power (median over patches)

| arm | σ(Ωm) | σ(σ8) | σ(w0) | 2D(Ωm,σ8) | FoM3 |
|---|---|---|---|---|---|
| **CNN auto+cross** | **0.032** | **0.047** | **0.171** | **1808** | **17251** |
| L1 auto+cross | 0.046 | 0.072 | 0.188 | 1045 | 8530 |
| CNN auto-only | 0.051 | 0.079 | 0.247 | 463 | 2343 |
| L1 auto-only | 0.056 | 0.085 | 0.246 | 441 | 2200 |

- **Auto-only: a tie.** L1 ≈ CNN on every metric (CNN a hair tighter on 2D, ×0.95; σ(w0) ×1.00). This
  is the **clean, leakage-free** result — both compressors extract comparable information from local
  auto-patches.
- **Auto+cross: CNN ahead on every parameter** — σ(Ωm) ×1.45, σ(σ8) ×1.5, σ(w0) ×1.1, 2D ×1.7, **FoM3
  ×2.0**. Even w0, historically L1's edge, favors CNN. **PROVISIONAL** (see §5.5).
- This **overturns** the original 20° headline (which had L1 ahead, σ(w0) ×1.34, FoM3 ×2.17).

### 4.2 Calibration — the tightness is trustworthy

All three tests pass on the final arms (this is the back-pressure that makes the constraining-power
claim safe):
- **TARP-DRP** (proper varied-θ, stratified by FoM3 tercile): all 4 arms — *including the tight
  HIGH-FoM3 tercile* — lie on the diagonal ⇒ calibrated, so the tight posteriors' FoM3 is **real, not
  inflated**. (A naïve fixed-θ Mahalanobis-χ² proxy spuriously shows over-coverage and is **not** a
  valid TARP — a documented pitfall.)
- **SBC** (400 validation cosmologies): ranks uniform on Ωm/σ8/w0 (mean rank ≈ 0.50, KS p ≈ 0.4–0.6,
  rank-std ≈ 0.29), i.e. globally unbiased — not the 20°'s ∪-shaped over-confidence. Mild
  miscalibration only in the weak h0/Ωb nuisances.
- **L-C2ST** (local, at the fiducial, CNN): **0/30 obs reject** (median p ≈ 0.2), a clean improvement
  over the 20° run (which rejected at 87%).

### 4.3 The w0 offset — a flat-sky artifact, not an ℓ₁ bias (resolved headline question)

The 10° run existed to answer: *is L1's fiducial w0 offset an intrinsic ℓ₁ bias or a geometry
artifact?* Population-mean w0 pull at the fiducial:

| arm | pull(w0) 20° | pull(w0) 10° |
|---|---|---|
| L1 auto+cross | **−0.37σ** | **−0.10σ** |
| CNN auto+cross | ~0 | −0.10σ |

L1's 20° −0.37σ w0 bias **shrinks to −0.10σ at 10° and is no longer L1-specific** (CNN shares the same
−0.10σ) ⇒ it was a **flat-sky-distortion artifact**, not an ℓ₁ compression bias. (Auto-only arms share
a separate ~+0.35σ w0/Ωm offset in *both* methods — a 10°-auto-only projection effect that cancels
globally per SBC.)

### 4.4 Why CNN is tighter — efficiency, not over-confidence or geometry

Decomposing the per-patch pull over the 9000-obs grid settles *why* CNN is tighter and rules out "CNN
learned the patch geometry":
- **Scatter is realization, not geometry.** Between-patch (geometry) fraction of the pull variance:
  L1 = 0.2/0.5/2.7% (Ωm/σ8/w0), CNN ≈ 0% — >97% of the scatter is the noise/structure realization,
  not sky position.
- **CNN is the *conservative* one.** Within-patch z-std: CNN 0.6–0.7 vs L1 0.9–1.0 (a ~1.3–1.4×
  lower-variance edge, what a near-sufficient VMIM summary should give). Empirical 68% coverage: CNN
  87/83/91% (mildly over-covers), L1 72/68/84% (≈calibrated). So CNN's contours are tighter **and** a
  touch wider than they strictly need to be — sharper but not over-confident. Sharpness ⊥ calibration.
- **L1's small fiducial offset is prior shrinkage** (an information effect, not an ℓ₁ pathology): the
  fiducial sits off-center in the training prior, and a less-informative summary regresses its
  posterior mean further toward the prior mean. A single information-fraction r per arm fits the bias
  on Ωm,σ8, monotonic in FoM3 (CNN a+c r≈0.97, L1 a+c 0.64, CNN auto 0.49, L1 auto 0.22). CNN
  *auto-only* is **more** biased than L1 auto+cross ⇒ not an L1 property; it washes out globally in
  SBC. (memory `project_l1_fiducial_bias_is_prior_shrinkage`.)

### 4.5 The cross-map strategy — and why auto+cross is currently PROVISIONAL

The auto-only tie vs the auto+cross CNN-lead is explained by **how the cross-maps are built**, and this
is a genuine methodological finding (`CROSS_MAP_LEAKAGE_FINDING.md`, **CITE**):

- The 6 cross channels are constructed on the **full sphere** (κ^{ij} = iSHT(aⁱ_ℓm·aʲ_ℓm), no
  apodization/mask) and only then cut into patches. Because each a_ℓm integrates over the whole sky,
  **every cross-patch pixel is a global functional of the full-sky convergence** — the patch carries
  cross-correlation information from the entire field. Autos do not leak (they are a local SHT→iSHT
  roundtrip).
- Quantified (structural angular-power decomposition, no NDE/noise dependence): cross channels hold
  **12–20% of their variance at super-patch scales (ℓ<18)** vs **0.4–1.0% for autos**, with cross
  ℓ_median crashing to ~60–90 (autos ~600). The cross field is large-scale and non-local.
- **Consequences:** (a) it explains the puzzle — the CNN can't reconstruct full-sphere a_ℓm from a
  cutout, so the explicit cross channels add information genuinely *unavailable* in the local autos
  (the "CNN should get it from the autos" intuition fails because the info isn't locally there); (b)
  the CNN reads these large-scale modes efficiently while the per-channel small-scale ℓ₁ does not,
  which is *why* CNN ≫ L1 on auto+cross specifically; (c) **the auto+cross constraining power is
  partly unphysical** — a real survey observing a 10° patch cannot build these maps. It is **not a
  calibration bug** (the leak is self-consistent across train and test, so TARP/SBC/L-C2ST pass); it
  is a data-vector *realism* problem.
- **Auto-only is unaffected** (local), so the auto-only tie stands clean.

**The cross-map strategy conclusion (current):** explicit cross-maps do add information, but the
full-sphere construction overstates the physically-achievable gain and *differentially* favors the CNN
(it reads the leaked large-scale modes that small-scale ℓ₁ cannot). The physically-defensible gain is
the **flat-sky / patch-local** construction — the decisive open experiment (§7).

---

## 5. Pillar 2 results — BNT contour inflation (direction established; clean run owed)

**Status:** every quantitative BNT result in the repo is from the **April Phase-1** campaigns, which
predate `--zero-mean-maps` and are therefore **mass-sheet-contaminated (hard #1)** and FoM3-only — so
the **absolute numbers are not citable**, but the **directional findings survive as within-era
relative comparisons** and motivate the thesis. (Sources: `BNT_TOMO4_*`, `TOMO_BIN_CROSSCORR_*`,
`OPTIMAL_NOBNT_CROSSCORR_*`, the `bnt_tomo4_study` / `nobnt_tomo_bins_crosscorr_study` /
`cnn_bnt_*` campaigns; all **BACKGROUND/WRONG** in the triage — *direction only*.)

### 5.1 The mechanism (the paper's argument)
BNT is an invertible linear mixing of the tomographic bins. Information is conserved; a sufficient
summary of the BNT maps is a sufficient summary of the original maps. Inflation under BNT therefore
indicates a **summary that fails to capture the BNT-induced cross-bin structure** — consistent with
BNT decorrelating signal across bins while correlating the noise, which lowers the SNR of a
per-channel statistic.

### 5.2 What the (contaminated) evidence shows — directionally
- **BNT inflates the ℓ₁ contours; a good compressor does not.** Within-era BNT/no-BNT inflation
  ratios (3-seed means, *direction only*): CNN initial-config ~1.80, raw L1 ~1.60, L1+VMIM ~1.58;
  after compressor optimization, **CNN → ~1.04 (near-lossless)** while optimized L1+VMIM only → ~1.38.
  ⇒ the inflation is a *compressor* property, and a sufficiently good compressor removes it.
- **The decisive lever is turning OFF summary standardization** — it moves CNN BNT/no-BNT FoM3
  retention from ~0.095 (catastrophic) to ~0.79–0.91. (NDE capacity was *not* the bottleneck.)
- **A multi-channel CNN extracts inter-bin cross-correlation gain far better than per-channel ℓ₁.**
  Cross-correlation attribution G_corr = (CNN/X full-tomo ratio)/(CNN/X single-bin-avg ratio): **G_corr
  ≈ 3.0 vs L1, ≈ 2.6 vs L1+VMIM** (the authoritative on-disk JSON values; the older markdown reports
  quote 3.6/195.6 — *stale, do not use*). Equivalently, tomo4/Σ(single-bin) FoM3 is **super-additive
  for CNN (~3.3)** but ~additive for L1/L1+VMIM (~0.9–1.0). Per-bin ordering bin4>bin3>bin2>bin1 is
  robust across methods.
- **Plain CNN beats stock-BN ResNets** on BNT parity (architecture depth does not help; data-limited).

### 5.3 What must be re-run clean (the "proof")
The clean demonstration — a **zero-mean, common-MAF, disjoint-split CNN(auto-maps-as-channels)-VMIM
run showing NO BNT contour inflation, against an ℓ₁ baseline that does inflate** — **does not yet
exist**. The repo got close (the ~1.04 near-lossless CNN ratio), but only under the contaminated
pipeline. This is the headline Pillar-2 result the paper will assert and must produce cleanly (§7).

---

## 6. Pitfalls woven into the paper (pointers; full catalog in the companion)

These are surfaced as "what we ruled out" subtext where they bear on a result; the full,
community-facing catalog (signatures, magnitudes, fixes) is in `PAPER_NARRATIVE_AND_PITFALLS.md` and
becomes the paper's pitfalls appendix.

- **Mass-sheet leak** (Methods/why-zero-mean): un-demeaned maps let the CNN exploit the
  mean-convergence level — unobservable in real data — inflating CNN FoM3 ~25–30×. Demeaning is
  mandatory; it is *why* all pre-2026-04 numbers are excluded.
- **L1 cross-channel noise model** (Methods/ℓ₁): using the auto pixel-σ for the ~10⁴× smaller cross
  channels collapses their wavelet SNR to ~0 and *inflated* the original "L1 wins 3×." Per-channel
  noise is required, and the TFDS route silently falls back to the broken model.
- **NDE-architecture confound** (Methods/NDE): comparing compressors through *different* flows
  conflates compressor with flow (a ~47% FoM3 swing on the *same* compressor). The common-MAF design
  is the fix and is what makes the final comparison fair. *(Flag, not disqualifier.)*
- **FoM3 fragility** (Discussion/metrics): for the strongly correlated Ωm–σ8 posterior (ρ≈−0.93) a
  ~5% marginal change can swing FoM3 ~50%; we report FoM3 *with* σ and 2D areas. *(Flag; FoM3 retained
  as a headline metric.)*
- **Cross-map leakage** (Results §4.5): the full-sphere construction makes auto+cross partly
  unphysical — a first-class finding, not just a caveat.
- **Calibrate before believing tightness** (Results §4.2): tight ≠ correct; the TARP/SBC/L-C2ST battery
  is the back-pressure (and a TARP done at fixed-θ is itself a trap).

---

## 7. Open / to-be-run — clearly future, never presented as done

1. **Flat-sky (patch-local) cross-map rebuild** — the decisive Pillar-1 experiment: rebuild the 6
   cross channels per patch from the patch's own autos (apodized convolution and/or pointwise
   product), recompute ℓ₁ and CNN summaries, retrain, and re-run the Phase-C/D comparison +
   calibration. Isolates the *physical* cross-information from the full-sphere leakage. Prediction:
   the CNN's auto+cross gain shrinks toward auto-only but does not vanish (a flat-sky cross is a real,
   if modest, local feature); the L1-vs-CNN ordering on the physical cross information is **genuinely
   open** (could stay CNN-favored, tie, or move toward L1). Plan: `FLATSKY_CROSS_BUILD_PLAN.md` /
   `FLATSKY_CROSS_REDESIGN_NOTES.md`; fiber `flatsky-cross-2026-06`. **The auto+cross headline stays
   PROVISIONAL until this runs.**
2. **Clean BNT-CNN no-inflation run** — the Pillar-2 proof: zero-mean, common-MAF, disjoint-split
   CNN(auto-channels)-VMIM showing no BNT inflation vs an inflating ℓ₁ baseline. Re-run of the
   April demonstration in the corrected pipeline.

---

## 8. Figures

See `PAPER_FIGURE_INVENTORY.md` for the full mapping (PDF-preferred), tagged KEEPER / NEAR-FINAL /
PLACEHOLDER, with a slot→file plan. Highlights: the data-vector gallery and full-sphere map panel
(KEEPERS); the D1–D9 + TARP-DRP result figures from the 10° campaign (NEAR-FINAL; auto+cross panels
regenerate post-flat-sky); the cross-construction and BNT-inflation overlays (PLACEHOLDERS to swap
when final). Until the flat-sky and clean-BNT runs land, the paper is laid out with the PLACEHOLDER
figures so its structure is visible.

---

## 9. Key references (for the bibliography — verify before citing)

- **BNT nulling:** Bernardeau, Nishimichi & Taruya (2014), *Cosmic shear full nulling* (arXiv:1312.0430).
- **Lossless / information-maximizing compression:** Heavens, Jimenez & Lahav (2000, MOPED,
  astro-ph/9911102); Alsing & Wandelt / Alsing et al. (2018, 2019, score compression / NDE +
  active learning); Charnock, Lavaux & Wandelt (2018, IMNN, arXiv:1802.03537); Makinen et al. (2021,
  arXiv:2107.07405).
- **SBI / NPE:** Cranmer, Brehmer & Louppe (2020, *The frontier of SBI*, arXiv:1911.01429); RealNVP
  (Dinh, Sohl-Dickstein & Bengio 2017, arXiv:1605.08803); MAF (Papamakarios et al. 2017).
- **Calibration:** TARP (Lemos et al. 2023); SBC (Talts et al. 2018); L-C2ST (Linhart et al. 2023).
- **Wavelet ℓ₁ / peaks in WL:** the starlet ℓ₁-norm WL literature (Ajani et al.); higher-order WL
  statistics reviews. **Sims:** CosmoGridV1 (Kacprzak et al. / Fluri et al.). Harmonic cross-maps:
  Zürcher et al. (2022).

*(The Stage-1 docs already collected the BNT-inflation reference set in
`BNT_NO_BNT_CONTOUR_INFLATION_NOTE.md`; reuse it. Verify all arXiv ids/years in `paper-draft`.)*

---

## 10. Provenance & trust (so `paper-draft` never over-claims)

- **Every CITE number here traces to the 10° campaign or the leakage finding** (the only citable
  core). Auto+cross numbers carry the PROVISIONAL-leakage tag.
- **Pillar-2 numbers are directional only** (mass-sheet-contaminated); state them as relative/within-era
  and never as clean absolutes.
- **Do not resurrect** the 20° "L1 wins" numbers as results — they are SUPERSEDED/WRONG (they remain
  in the paper only as the *narrative* of what was overturned).
- **DONE vs TO-BE-RUN** (§7) must stay separate; the flat-sky and clean-BNT results are forthcoming.
- Full per-file trust: `PAPER_FILE_TRIAGE.md`. The journey behind these conclusions:
  `PAPER_NARRATIVE_AND_PITFALLS.md`.
