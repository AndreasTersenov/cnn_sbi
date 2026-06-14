# Tomographic 2D-1D wavelet ℓ1-norm — proposal, theory, and phased plan

**Date:** 2026-06-13. **Author:** Claude (Opus 4.8), with Andreas. **Origin:** Jean-Luc Starck
suggested generalizing the starlet ℓ1-norm with an extra 1D axis = the tomographic-bin dimension
(a "2D-1D wavelet"), "with the absolute values."

**Status:** PROPOSAL + THEORY. No code written, no GPU run yet. This is the canonical document for
the direction; it consolidates a research deep-dive (3 agents, primary sources) and a design
discussion. The phased plan in §8 is the actionable part and waits for sign-off.

---

## 0. One-paragraph summary

We want a wavelet ℓ1-norm that captures cross-bin (tomographic) correlations — the information that
per-bin ℓ1 misses and that collapses catastrophically under BNT (per-bin ℓ1 → 0.15× the no-BNT FoM),
while the CNN, which can re-mix bins internally, does not (0.93×). The 2D-1D wavelet adds a 1D
transform along the bin axis. Two versions are on the table, and we will build them in order:
**Approach A** — the *pure* 2D-1D wavelet ℓ1-norm (2D starlet ⊗ 1D Haar along bins, then the
S/N-binned ℓ1). It is faithful, cheap, and interpretable, but linear, so it is mathematically a
standard wavelet ℓ1-norm applied to a fixed set of Haar (sum/difference) maps — bounded by the
"linear-recombination ceiling." **Approach B** — insert a *modulus* between the 2D and 1D transforms.
That nonlinearity is the only way to capture cross-information beyond the linear ceiling, but it makes
the statistic scattering-*structured* (no longer a pure ℓ1-norm). Plan: build A, measure where it
lands (that fixes the linear ceiling), then draft B to chase the cross-information above it.

---

## 1. Motivation — the problem we are trying to solve

Tomographic weak lensing splits sources into redshift bins; a large share of the cosmological
information lives in the *cross*-correlations between bins. Our analytic summary, the starlet wavelet
ℓ1-norm, is computed per bin and concatenated — cross-bin correlation enters only through the
covariance, never the summary itself. Two symptoms of the gap, both measured in this campaign:

1. **BNT collapse.** The BNT transform recombines bins to null low-z lenses (clean scale cuts). In
   BNT space the per-bin ℓ1 FoM collapses to ~0.15× the no-BNT value, because BNT scatters the deep
   coherent signal across thin nulled channels and a per-channel statistic never reforms it. The CNN
   does not collapse (0.93×) — it can learn the inverse recombination. (`project_flatsky_bnt_losslessness`.)
2. **Cross-maps only modestly help the ℓ1.** Our hand-built cross channels gain +20% (product =
   ξ_ij), conv ≈ 0 after de-leaking. (`FLATSKY_CROSS_RESULT.md`, M2.)

Jean-Luc's idea: instead of hand-building cross maps, put a *wavelet on the bin axis* — the natural,
principled, multiscale way to mix bins. The two goals Andreas set: **(1) tightest contours (recover
the most cross-information)** and **(2) BNT robustness (contours that do not inflate in BNT space).**

---

## 2. Background (verified from primary sources)

### 2.1 The 2D-1D wavelet transform (Starck/CosmoStat lineage)
- **Starck, Fadili, Digel, Zhang, Chiang (2009)**, A&A **504**, 641, arXiv:0904.3299 — the canonical
  paper. Built for Fermi cubes (2 angular + 1 energy/time). Core idea, verbatim: a 3D *isotropic*
  wavelet "does not make sense" when the third axis is not spatial, so use a **partially separable**
  transform — an undecimated isotropic 2D starlet (B3-spline à trous) in space, composed with a
  *separate* 1D wavelet along the third axis. A coefficient `w_{j1,j2}` is indexed by spatial scale
  `j1` and third-axis scale `j2`. The linear transform is a redundant, exactly-invertible frame.
- **Schmitt et al. (2012)**, arXiv:1206.2787 — same construction on the sphere (MS-VSTS).
- **Flöer & Winkel (2012)**, PASA, arXiv:1112.3807 — independent re-use for HI spectral cubes.
- **Software:** CosmoStat/**Sparse2D** (C++): `mr2d1d_trans` (linear transform), `msvst_2d1d`
  (Poisson denoising), class `Atrous2D1D`; bi-Haar variant `bihaar_*_2d1d` in the `mwir` module; on
  the sphere via **MRS-MSVSTS** (iSAP/IDL); Python binding `mr_2d1d.hpp` → pySAP.
- **The 1D-axis wavelet in the canonical method is the B3-spline starlet, NOT Haar.** Haar is a
  documented but non-default variant. The 2009 paper argues Haar is *worse spatially* (isotropic
  sources prefer the smooth starlet) — but that argument is about the 2D plane and does not transfer
  to a short ordinal bin axis. So Haar on *our* axis is justified on its own merits (§2.2), not by
  citing the Fermi papers.

### 2.2 Haar on a length-4 bin axis (verified, numerically checked)
Orthonormal Haar on `(κ1,κ2,κ3,κ4)`:
```
        ┌ 1/2    1/2    1/2    1/2  ┐   m=0: global mean   ¼Σκ            (deep mode, high S/N)
  H  =  │ 1/2    1/2   -1/2   -1/2  │   m=1: coarse difference (12)-(34)
        │ 1/√2  -1/√2   0      0    │   m=2: fine difference κ1-κ2
        └ 0      0      1/√2  -1/√2 ┘   m=3: fine difference κ3-κ4
```
Decimated Haar on length 4 = exactly 2 levels (log₂4). The unnormalized 1-level Haar is literally
sum/difference maps `(κ1+κ2, κ3+κ4, κ1-κ2, κ3-κ4)`. For a short, non-smooth, ordinal axis Haar is the
right minimal choice (a B-spline barely fits 4 points and is boundary-dominated at scale 2).
**Recommendation: decimated orthonormal Haar, applied once** — 4 maps in, 4 maps out, lossless.

### 2.3 The wavelet ℓ1-norm, and where this idea sits (verified)
- **Ajani, Starck, Pettorino (2021)**, A&A **645**, L11, arXiv:2101.01542 — the starlet ℓ1-norm:
  `ℓ1^{(j,i)} = Σ_u |S_{j,i}[u]|`, the sum of `|wavelet coefficients|` at scale `j` within S/N bin
  `i` (29 S/N bins/scale). A rich one-point distribution (peaks AND voids). Already tomographic (4
  bins), but **combines bins by concatenation** — no cross-bin statistic.
- **The wavelet ℓ1-norm is NOT the scattering transform.** ℓ1-norm = S/N-binned histogram of
  `|wavelet coeffs|` (terminal `|·|` only). Scattering (Cheng/Ménard; Gatti) = `wavelet → |·| →
  wavelet → |·| → mean` — its power is re-wavelet-transforming the modulus field, read out as a mean.
  Structurally different statistics. This distinction is the whole point of §3.
- **Cross-bin HOS that exist:** Zürcher+2022 (DES Y3) cross-*peaks* on harmonic-product bin-pair maps
  (arXiv:2110.10135); Gatti+2023 (DES Y3) "cross-maps" for scattering/WPH/moments (arXiv:2310.17557).
  Both use *pairwise products* of bins.
- **Closest "wavelet along the line of sight":** Leistedt+2015 flaglets (arXiv:1509.06750) — but a
  *continuous radial* coordinate of a 3D field, not a transform across discrete tomographic maps.
- **Novelty:** a wavelet (or Haar) transform on the *discrete tomographic-bin axis*, feeding a
  starlet-ℓ1 HOS, appears unclaimed. Cite Zürcher/Gatti as the pairwise-cross precedent and Leistedt
  as the radial-wavelet precedent. (Caveat: do a 30-min direct read of Gatti+2023 §methods and
  Ajani+2023 arXiv:2211.10519 before asserting novelty in print.)

---

## 3. The two approaches

### Approach A — the pure 2D-1D wavelet ℓ1-norm (the faithful, simpler version)
```
κ_b  --2D starlet S_{j1}-->  --1D Haar H across bins-->  w_{j1,m}  --S/N-binned ℓ1-->  datavector(j1, m, S/N)
```
The 2D-1D coefficients are `w_{j1,m}`, indexed by spatial scale `j1` and Haar bin-mode `m`. The
statistic is the ordinary wavelet ℓ1-norm of `|w_{j1,m}|`, now carrying the extra bin-mode index `m`.
The "absolute values" Jean-Luc mentioned are the ℓ1-norm's own `Σ|w|`; **there is no extra modulus.**
This is a wavelet ℓ1-norm, faithfully generalized.

### Approach B — insert a modulus between the two transforms (the generalized, scattering-like version)
```
κ_b  --2D starlet S_{j1}-->  --|·|-->  |S_{j1}κ_b|  --1D Haar H across bins-->  --S/N-binned ℓ1-->
```
An intermediate modulus sits between the spatial and bin transforms. The Haar then mixes the *modulus
fields* across bins. This is scattering-*structured* (wavelet → modulus → wavelet), read out with an
ℓ1 rather than a mean — a scattering/ℓ1 hybrid, not a pure ℓ1-norm.

---

## 4. What each captures — the information analysis

### 4.1 Approach A is linear, and reduces to an ℓ1-norm on Haar maps
Both the 2D starlet `S_{j1}` and the 1D Haar `H` are linear, so they commute:
$$w_{j1,m} = \sum_b H_{m,b}\,S_{j1}[\kappa_b] = S_{j1}\!\Big[\sum_b H_{m,b}\,\kappa_b\Big] = S_{j1}\big[\kappa^{\mathrm{Haar}}_m\big].$$
The 2D-1D wavelet coefficient is just the 2D starlet of the `m`-th Haar combination of the maps. The
ℓ1-norm then histograms `|w_{j1,m}|` — **so Approach A is mathematically identical to applying the
ordinary wavelet ℓ1-norm to the four Haar maps** `{¼Σκ, (κ1+κ2)−(κ3+κ4), κ1−κ2, κ3−κ4}`.

This is **not** "captures no cross-information." The ℓ1-norm of `κ1−κ2` depends on `Cov(κ1,κ2)`
(its variance is `Var(κ1)+Var(κ2)−2Cov`), and `¼Σκ` is a high-S/N coherent deep mode. So Approach A
genuinely sees cross-bin correlation — but only the amount carried by the **one-point distributions
of fixed linear combinations of bins.** That is the *linear-recombination ceiling*: the same family
as the sum/union maps we already ran (~1.2–1.4× over auto-only). The specific *orthonormal-Haar* set,
with the proper 4-bin deep mode (2× S/N vs a single bin) and a per-mode noise model (§6), is a
cleaner and possibly-better member than anything we tested — worth pinning down as the best
linear-family point — but it will not break out above the ceiling.

### 4.2 Approach B escapes the ceiling because the modulus does not commute
`|S_{j1}κ_b|` is nonlinear in `κ_b`, so `Σ_b H_{m,b} |S_{j1}κ_b|` is *not* the starlet of any linear
combination — the §4.1 reduction breaks, and the statistic can access cross-bin structure the linear
form cannot. This is the only way for a 2D-1D wavelet ℓ1-norm to capture cross-information beyond the
linear-recombination ceiling. The cost is that it is no longer a pure ℓ1-norm; structurally it is a
first-order scattering field re-mixed across bins by a Haar and read out with an ℓ1.

**The fork is genuine:** "stay a pure ℓ1-norm" and "beat the linear cross-correlation ceiling" are
partly mutually exclusive, and the intermediate modulus is exactly the boundary between them.

---

## 5. BNT robustness — the analysis

### 5.1 The principle (from our own M3 whitening result)
A statistic is BNT-robust iff it is insensitive to BNT's bin-axis rotation. M3 quantified the room:
`Q = (BBᵀ)^{-1/2}B` is the *orthogonal polar factor* of BNT, and per-channel ℓ1 in *that* orthonormal
frame recovers the full no-BNT FoM (1.06/1.01). Read literally: **BNT's damage to a per-channel
statistic is entirely its non-orthogonal (shear) part; there exists an orthonormal bin frame in which
per-channel ℓ1 loses nothing.** (`project_flatsky_bnt_losslessness`.)

### 5.2 Approach A on BNT
A Haar recombination of BNT maps is a fixed linear combination of the original maps — still linear.
Whether it rescues the collapse depends on whether the Haar combinations reconstruct a deep mode.
This is M4 territory (cut-then-mix / reconstructed-deep): the rescue works but is **schedule-dependent
— 1.82× at an aggressive cut, only ~1.07× at a realistic light cut** — because the cut removes
information in the sheared frame *before any statistic sees it*. Approach A cannot escape M4's physics.
(`M_VS_L_ROBUSTNESS.md`, M4 PARKED.)

### 5.3 Approach B on BNT — plausibly better, but an empirical question
The Haar *sum* mode of the modulus fields is `Σ_b |S_{j1}κ_b|` — a sum of **positive** moduli (total
wavelet power across bins). Summing magnitudes does **not** suffer the sign cancellation that nulls
the linear deep mode in BNT space. So Approach B is plausibly *much* less BNT-fragile than per-bin
linear ℓ1 — not because it is invariant (it is not; the modulus bakes in the frame), but because the
specific failure mechanism that kills the linear version does not apply. **This is not provable a
priori; it must be measured (BNT vs no-BNT).**

### 5.4 The fresh BNT angle either approach enables — a better cut *basis*
BNT cuts per-channel-then-by-spatial-scale. The 2D-1D wavelet produces a **2D scale-space (spatial
scale `j1` × bin-mode `m`)** in which to define the systematics cut. Cutting the specific `(j1, m)`
cells carrying baryonic/nonlinear contamination — rather than whole channels below a scale — is a
*more surgical* cut that could remove the same systematics while keeping more constraining cells.
This generalizes the HOWLS-2 "BNT for scale cuts" idea to a wavelet domain, is genuinely new, and is
independent of the A-vs-B choice. It reframes goal 2 from "rescue what the cut destroyed" to "design a
less destructive cut." Untested; the part of the BNT story I am most optimistic about.

---

## 6. Honest expectations, registered predictions, and the noise-model tripwire

### 6.1 Two ceilings the design cannot move
- **Goal 1 (tighter contours) is ceiling-limited.** M1 (CNN does not beat L1+product), M2 (cross gain
  +20%, conv ≈ 0 de-leaked) and the 10° de-leak all say the accessible cross-bin information on a 10°
  patch is *small* and set by the physics, not the statistic. The design decides whether we *reach*
  the CNN/L1+product ceiling (~2900–3200), not where it is. **A clean tie with the CNN is the win we
  want** (`project_cnn_optimization_goal_referee_defense`): it would show a principled analytic
  statistic captures the accessible cross info and the CNN's theoretical edge does not materialize.
- **Goal 2 (BNT) is fully solvable only UNCUT (M3) — and uncut BNT has no purpose; the realistic cut
  case is M4-limited.** The fresh lever is the surgical-cut basis (§5.4), not raw FoM3 rescue.

### 6.2 Registered predictions (before any run — so we score honestly)
- **Approach A, no-BNT:** FoM3 ≈ **2900–3300** (vs auto-only 2405, L1+product 2875, unions6 ~2840);
  ~1.2–1.4× over auto-only, driven mainly by the deep `¼Σκ` mode. P(beats 3300 clean) ≈ 25%.
- **Approach A, BNT:** rescues the collapse to ~parity at a realistic cut (inherits M4); direction-
  robust, magnitude schedule-dependent.
- **Approach B, no-BNT:** ≥ A; whether it clears the CNN/L1+product ceiling is the open question (the
  modulus accesses genuinely-joint structure, but the physics ceiling caps it).
- **Approach B, BNT:** plausibly substantially more robust than A (§5.3) — the headline test.
- A large overshoot of these triggers a hunt for the hole in the analysis (most likely an
  S/N-binning / noise-model interaction), not a victory lap.

### 6.3 The one tripwire: per-bin-mode noise model
The ℓ1-norm bins by S/N, so it needs the per-channel noise level. The Haar modes have *different*
noise: `¼Σκ` has ~σ/2 (independent per-bin shape noise), differences have ~√2·σ. Reusing one per-bin
σ for all modes reproduces *exactly* the 2026-05-15 cross-channel noise bug
(`feedback_l1_cross_must_use_harmonic_route`, `project_l1_noise_model_correction`): the S/N maps to
the wrong bins, histogram bins zero out, FoM3 craters. **Mandatory:** propagate analytically
(`σ²_m = Σ_b H_{m,b}² σ²_b`) or estimate per-mode empirically (`channel_empirical_global`). Verify the
per-mode σ table prints, not a fallback warning. For Approach B the noise model is on the *modulus*
fields' Haar combinations — estimate empirically. Also: `pca_applied: False` always
(`feedback_never_pca_l1` — note the bin-axis Haar/PCA on the 4 maps is a *different*, allowed
operation from PCA on the ℓ1 datavector).

---

## 7. Novelty and paper value

Even if the FoM3 lands at the ceiling (§6.1), the value is real: (i) it replaces our ad-hoc cross-maps
zoo (sum / product / conv / unions) with **one principled, citable, Jean-Luc-endorsed statistic** —
the natural tomographic generalization of the wavelet ℓ1-norm — which pre-empts "why these particular
cross maps?"; (ii) it cleanly characterizes the linear ceiling (Approach A) and, with Approach B,
isolates the irreducibly-joint / nonlinear part — tying M2, M3, M5 into one frame; (iii) the bin-axis
wavelet for an ℓ1 HOS appears genuinely novel (§2.3); (iv) the surgical-cut basis (§5.4) is a new
contribution to BNT scale-cut methodology. A clean negative ("the principled multiscale bin-axis
statistic matches but does not beat the product channel") is still publishable and exactly the kind of
thoroughness the paper's journey-narrative wants (`project_paper_narrative_includes_journey`).

---

## 8. Phased plan (Andreas's sequencing: simpler first, measure, then generalize)

### Phase 1 — Approach A (build, measure, gate). The actionable next step.
Reuses the L1 pipeline almost verbatim; the new piece is the 4×4 Haar between the 2D starlet and the
ℓ1, plus the per-mode noise model.
- **A-arms (one factor at a time; same MAF + TARP/SBC gate as every other arm):**
  - `A1` — 2D-1D-Haar ℓ1, **no-BNT**, per-mode noise. Compare to per-bin auto (2405), L1+product
    (2875), unions6. *Registered: 2900–3300.*
  - `A2` — 2D-1D-Haar ℓ1, **BNT** maps (Haar across BNT channels). Compare to M4 reconstructed-deep.
    *Registered: rescue-to-parity at realistic cut.*
- **First-class output: the BNT-vs-no-BNT comparison** (the goal-2 measurement — do not assume it).
- **Decision gate:** A1's landing point fixes the measured linear-recombination ceiling. If A ties or
  beats L1+product cleanly → strong principled result, possibly enough on its own. If A confirms the
  ceiling and we want the cross-info above it → Phase 2.

### Phase 2 — Approach B (draft after A). Chase the cross-info above the linear ceiling.
- `B1` — modulus-inserted (`2D starlet → |·| → Haar → ℓ1`), no-BNT. Gain measured *against A1's known
  baseline* — this is the payoff of doing A first.
- `B2` — same, BNT. The headline goal-2 test (§5.3).
- Only opened if Phase 1 confirms the ceiling and the modulus is worth the added complexity.

### Phase 3 (optional, later) — the surgical-cut basis (§5.4)
Use the `(j1, m)` 2D scale-space to define a targeted BNT systematics cut; compare its FoM3-at-fixed-
systematics-control against BNT's per-channel cut. The most novel BNT contribution; orthogonal to A/B.

### Back-pressure (how we know it worked)
Per-mode σ table prints (no noise fallback); `pca_applied: False`; FoM3 reported *with* σ(Ωm,σ8,w0)
marginals-first; TARP net-bias + SBC std in the registered band before any claim; score against the
§6.2 registered predictions.

---

## 9. The one open question for Jean-Luc
The distinction that changes what we build and what we can promise: did "absolute values" mean the
ℓ1-norm's own `Σ|w|` (→ **Approach A**, pure ℓ1-norm, linear ceiling) — Andreas's current read — or an
**intermediate modulus** between the two transforms (→ Approach B, escapes the ceiling, scattering-
structured)? We will build A regardless (it is the faithful simpler version and its result is
diagnostic); the answer decides how hard we push toward B and how we frame it.

---

## Appendix A — Bibliography (verified from primary sources)

**2D-1D wavelet / sparse cubes**
- Starck, Fadili, Digel, Zhang, Chiang (2009), A&A 504, 641, arXiv:0904.3299. [VERIFIED — PDF read]
- Schmitt, Starck, Casandjian, Fadili, Grenier (2012), A&A, arXiv:1206.2787. [VERIFIED — abstract/PDF; confirm final vol/page on ADS]
- Flöer & Winkel (2012), PASA, arXiv:1112.3807. [VERIFIED]
- Zhang, Fadili, Starck (2008), IEEE TIP 17(7), 1093 — MS-VST foundation. [VERIFIED]
- Starck, Fadili, Murtagh (2007), IEEE TIP 16(2), 297 — IUWT/UWT. [PARTIAL — verify page nos.]
- Starck, Murtagh, Fadili, *Sparse Image and Signal Processing*, CUP (2010/2015). [PARTIAL — textbook]

**Wavelet ℓ1-norm / WL HOS**
- Ajani, Starck, Pettorino (2021), A&A 645, L11, arXiv:2101.01542 — starlet ℓ1-norm. [VERIFIED — full text]
- Ajani, Starck et al. (2023), arXiv:2211.10519 — starlet HOS clustering+lensing. [PARTIAL — read before novelty claim]
- Martinet et al. (2018), MNRAS, arXiv:1709.07678 — KiDS-450 tomographic peaks. [VERIFIED]
- Zürcher et al. (2022), MNRAS, arXiv:2110.10135 — DES Y3 cross-peaks (harmonic-product bin pairs). [VERIFIED — abstract]
- Cheng, Ting, Ménard, Bruna (2020), MNRAS 499, 5902, arXiv:2006.08561 — scattering transform WL. [VERIFIED]
- Cheng & Ménard (2021), MNRAS 507, 1012, arXiv:2103.09247 — scattering sensitivity. [VERIFIED]
- Gatti et al. (2023), arXiv:2310.17557 — DES Y3 cross-maps (scattering/WPH/moments). [PARTIAL — read §methods before novelty claim]
- Valogiannis & Dvorkin (2022), PRD 105, 103534, arXiv:2108.07821 — WST on Quijote LSS. [VERIFIED]

**BNT / scale cuts / radial wavelets**
- Bernardeau, Nishimichi, Taruya (2014), MNRAS 445, 1526 — BNT (full nulling). [VERIFIED]
- Euclid Collab. (2026), "Euclid prep. LXXXV / HOWLS-2", A&A 707, A235, arXiv:2510.04953 — BNT scale cuts + ℓ1-norm. [PARTIAL — abstract]
- Euclid Collab., Ajani et al. (2023), "Euclid prep. XXVIII", A&A 675, A120, arXiv:2301.12890 — HOWLS-1 ten HOS. [VERIFIED — listing]
- Leistedt, McEwen, Kitching, Peiris (2015), PRD 92, 123010, arXiv:1509.06750 — flaglets (continuous radial). [VERIFIED]

**Software:** CosmoStat/Sparse2D — `mr2d1d_trans`, `msvst_2d1d`, `Atrous2D1D`, `bihaar_*_2d1d`;
MRS-MSVSTS (sphere); pySAP binding `mr_2d1d.hpp`.

## Appendix B — Provenance / decision log (compact)
- **Research deep-dive** (3 parallel agents, primary sources): bibliography + technical construction
  verified; correction that the canonical 1D-axis wavelet is B3-spline starlet, not Haar (Haar is a
  justified-on-merits choice for our short axis); ℓ1-norm third author is Pettorino not Pires; novelty
  (bin-axis wavelet for ℓ1 HOS) appears clean. Agent IDs below.
- **Design discussion (Andreas):** confirmed the statistic is a *2D-1D wavelet ℓ1-norm*, NOT a
  scattering transform (retraction of an earlier "this is Gatti scattering" mischaracterization).
  Jean-Luc said "absolute values" → most likely Approach A's terminal `|·|`; explicitly did **not**
  mean a whitening frame (an earlier "whiten-before-modulus / C2" proposal is therefore retired as a
  literal reading, though §5.1's principle remains the lens for interpreting BNT behaviour). Settled
  sequencing: Approach A first (measure the linear ceiling), then Approach B.
- **Agent IDs (for follow-up):** 2D-1D construction `ae7aa522e16fad347`; Haar / 1D-axis choice
  `a5887a84fb2f4aa73`; WL HOS landscape / novelty `a80a70b75fa493ed8`.
