# The 2D-1D wavelet ℓ1-norm for tomographic weak lensing — method, implementation, and first results

*A working note (2026-06-14) on the idea Jean-Luc suggested — a generalization of the starlet
ℓ1-norm with an extra 1D axis along the tomographic-bin dimension, "with the absolute values." It
describes exactly what we built, the two ways one can read the suggestion, what we found for the
first (simpler) one, and the question we'd like Jean-Luc to settle. Figures referenced are in
`plots_2d1d/`.*

---

## 1. The problem we are trying to solve

We have four flat-sky tomographic convergence maps `κ_1..κ_4` (10° patches, 80², ~7.5′/pixel, from
CosmoGridV1) and infer `(Ω_m, σ_8, w_0, …)` by simulation-based inference (a normalizing-flow NDE on
a summary statistic). Our analytic summary is the **starlet wavelet ℓ1-norm**, computed per bin and
concatenated. Two symptoms motivate this work:

1. **A lot of tomographic information lives in the cross-correlations between bins**, which a per-bin
   ℓ1-norm sees only through the data covariance, not in the statistic itself.
2. **Under the BNT transform the per-bin ℓ1-norm collapses.** BNT recombines the bins to null low-z
   lenses (it buys clean, physically-localized scale cuts). In BNT space our ℓ1-norm FoM falls to
   ~0.15× the no-BNT value, while a CNN compressor — which can re-mix the bins internally — does not
   degrade. So the ℓ1-norm is missing the cross-bin information that the CNN captures.

**Jean-Luc's suggestion:** instead of hand-building cross-maps, put a *wavelet along the bin axis* — a
"2D-1D wavelet," the 2D starlet in the sky plane composed with a 1D wavelet over the four tomographic
bins (the natural analogue of the 2D-1D transform of Starck, Fadili, Digel, Zhang & Chiang 2009, where
the third axis was photon energy) — and take "the absolute values." The two goals: **(1)** recover
more cross-bin information (tighter contours); **(2)** be robust to BNT (contours that do not inflate
in BNT space).

---

## 2. The statistic we build on (starlet ℓ1-norm)

For one map: the **starlet (isotropic undecimated B3-spline à trous) transform** gives wavelet
coefficient maps `W_j` at scales `j = 0..4` (j=0 finest). The **ℓ1-norm** (Ajani, Starck & Pettorino
2021) is the sum of `|W_j|` binned by signal-to-noise: for each scale `j` and S/N bin `i`,
`ℓ1^{(j,i)} = Σ_pixels∈bin i |W_j[pixel]|`. The S/N denominator is a frozen per-(channel, scale) noise
σ measured from noise realizations; we use 40 S/N bins per scale with per-channel S/N ranges. The
datavector is the concatenation over `(channel, scale, S/N-bin)`. **The only nonlinearity is the
ℓ1-norm's own absolute value.** This is implemented in `wl_stats_torch.WLStatistics` (GPU/torch) and
the per-channel-noise machinery in `flatsky_cross_l1.py`.

---

## 3. The 2D-1D extension — and the one place where "with the absolute values" matters

The 2D-1D coefficient is `w_{j,m}` indexed by the spatial starlet scale `j` and a **bin-axis index
`m`** produced by a 1D transform over the four bins. For the (very short, 4-sample, non-smooth) bin
axis the natural minimal 1D wavelet is **Haar**; the orthonormal 4-point Haar over the bins is
```
H = [[ 0.5,  0.5,  0.5,  0.5],   # m0  deep mode   ¼Σκ  (the bin average; high S/N)
     [ 0.5,  0.5, -0.5, -0.5],   # m1  coarse difference (κ1+κ2) − (κ3+κ4)
     [ 1/√2,-1/√2, 0,    0  ],   # m2  fine difference κ1 − κ2
     [ 0,    0,    1/√2,-1/√2]]] # m3  fine difference κ3 − κ4
```
(unnormalized, these are exactly the familiar sum/difference combinations of the bins; the deep mode
`m0` is the bin average). **There are two genuinely different ways to assemble "the 2D-1D wavelet
ℓ1-norm with absolute values," and they are not equivalent — this is the crux question for Jean-Luc.**

### 3.1 Approach A — the *pure* 2D-1D wavelet ℓ1-norm (the linear reading)
Take the four Haar combinations of the maps, then the ordinary starlet ℓ1-norm of each:
```
κ_b  --1D Haar across bins-->  κ^Haar_m   --2D starlet-->  --S/N-binned ℓ1-->   datavector(m, j, S/N)
```
Here "the absolute values" are the ℓ1-norm's own `|·|`; there is **no extra modulus**. This is the
faithful generalization of the ℓ1-norm: the same statistic, now carrying the extra bin-mode index `m`.

### 3.2 The subtlety that makes A a re-parameterization
The 2D starlet `S_j` is linear and the Haar `H` is linear, so they commute:
`w_{j,m} = S_j[Σ_b H_{m,b} κ_b] = S_j[κ^Haar_m]`. The ℓ1-norm then histograms `|w_{j,m}|` per channel.
**So Approach A is mathematically identical to applying the ordinary starlet ℓ1-norm to the four fixed
Haar maps** `{¼Σκ, (κ1+κ2)−(κ3+κ4), κ1−κ2, κ3−κ4}`. It does capture cross-correlation (e.g. the
ℓ1-norm of `κ1−κ2` depends on `Cov(κ1,κ2)`, and the deep `¼Σκ` is a high-S/N coherent combination) —
but only the amount carried by the *one-point distributions of fixed linear combinations of the bins*.
It cannot exceed that "fixed-linear-recombination" ceiling, and a fixed rotation does not, in general,
recover the BNT-scrambled information (only a whitening/orthonormalizing rotation does).

### 3.3 Approach B — insert a modulus between the two transforms (the scattering reading)
```
κ_b  --2D starlet-->  S_j[κ_b]  --|·| (modulus)-->  |S_j[κ_b]|  --1D Haar across bins-->  --ℓ1-->
```
Now "the absolute values" is an **intermediate modulus** sitting *between* the 2D and 1D transforms.
Because `|S_j κ_b|` is nonlinear in the maps, `Σ_b H_{m,b} |S_j κ_b|` is no longer the starlet of any
linear combination — the §3.2 reduction breaks, so this can capture cross-bin structure the linear form
cannot. Structurally this is a (first-order) scattering transform with a Haar second step along the bin
axis, read out with an ℓ1-norm rather than a plain mean. It is the only version that can (i) exceed the
linear ceiling and (ii) potentially survive BNT — the Haar *sum* of moduli `Σ_b |S_j κ_b|` is a sum of
*positive* quantities, so it does not suffer the sign cancellation that nulls the deep mode under BNT.

**The question for Jean-Luc:** did you mean Approach A (the ℓ1-norm's own absolute value, a pure
wavelet ℓ1-norm — what we ran first) or Approach B (an intermediate modulus, scattering-structured)?
Our reading of "with the absolute values" now points to B, and B is where any real gain must live (A is
bounded as in §3.2). We ran A first because it is the cheap, faithful baseline; we are now running B.

---

## 4. What we implemented for Phase 1 (Approach A) and the three arms

**Implementation.** In our flat-sky pipeline, applying a mix matrix `M` over the four autos and then
the per-channel starlet ℓ1-norm is a single existing code path (`flatsky_cross_l1.build_and_l1(autos,
op="none", bnt=M)`), so Approach A is literally "pass `H` as the mix." The per-channel noise is the
correct quadrature of the (independent) per-bin shape noise, `σ²_m = Σ_b H_{m,b}² σ²_{auto,b}`; per-
channel S/N ranges are calibrated from the data percentiles. Every arm is then run through **the same**
normalizing-flow NDE (jaxili MAF), the same preprocessing, the same 9000-observation fiducial
population (median FoM3 + marginals), and the same calibration gate (TARP + SBC) as our existing
baselines — so the comparison is apples-to-apples. Build script `build_flatsky_haar_arm.py`, orchestrated
by `run_haar_2d1d_phase1.py`.

**Three arms** (FoM3 = 1/√det C₃ over (Ω_m, σ_8, w_0); baselines, same pipeline: auto-only **2405**,
L1+product (= ℓ1 on autos + the ξ_ij product cross-maps) **2875**, L1+conv+product **2910**):

| arm | what it is | mix `M` |
|---|---|---|
| `haar_nobnt` | the pure 2D-1D Haar ℓ1-norm (§3.1) | `H` |
| `autohaar_nobnt` | the four autos' ℓ1 **plus** the four Haar-mode ℓ1 (augmented) | autos ⊕ `H` |
| `haar_bnt_uncut` | the *same* 2D-1D Haar ℓ1-norm computed **in BNT space** (the goal-2 test) | `H·B` |

(`B` is the BNT matrix; "Haar across the BNT channels" = the single combined mix `H·B` over the autos,
since both are linear. "uncut" = no scale cuts applied — the clean frame-robustness test.)

---

## 5. Phase-1 results (Approach A) — the linear form underdelivers on both goals

| arm | FoM3 | σ(Ω_m, σ_8, w_0) | calibration gate |
|---|---|---|---|
| auto-only (baseline) | 2405 | 0.053, 0.082, 0.245 | clean |
| L1+product (bar) | 2875 | 0.048, 0.075, 0.238 | clean |
| L1+conv+product | 2910 | 0.046, 0.075, 0.232 | clean |
| **haar_nobnt** (pure 2D-1D Haar) | **2676** | 0.049, 0.078, 0.235 | **FAIL** (mild tail over-confidence) |
| **autohaar_nobnt** (autos ⊕ Haar) | **2954** | 0.046, 0.074, 0.231 | PASS-with-caveat |
| **haar_bnt_uncut** (Haar in BNT space) | **885** | 0.082, 0.128, 0.303 | PASS-with-caveat |

**Goal 1 (tighter contours): the Haar modes add nothing beyond what we already had.** The pure 2D-1D
Haar (2676) sits between auto-only and the product bar but is mildly over-confident in the high-S/N tail
(so its number is partly inflated). The augmented autos⊕Haar (2954) is well-calibrated but lands
statistically on top of the existing best linear arm (L1+conv+product 2910, identical marginals). The
contour overlay (`contours_2d1d.png`) shows the four no-BNT arms essentially coincident. This is exactly
the §3.2 ceiling: a fixed linear recombination + per-channel ℓ1 cannot beat the cross-maps we already had.

**Goal 2 (BNT robustness): the linear Haar does NOT survive BNT.** In BNT space the same statistic
collapses to FoM3 885 (0.33× of no-BNT; σ(w0) inflates 0.235→0.303), and it is *calibrated* — a real
information loss, not over-confidence. The datavector plot (`datavector_bnt_collapse.png`) shows the
mechanism cleanly: the **deep channel** loses its high-S/N reach under BNT (its S/N support shrinks from
≈[−12,+14] to ≈±7), while the thin difference channels are unchanged. A generic fixed (Haar) rotation
does not reconstruct the deep coherent mode that BNT scrambles — consistent with our separate finding
that only a *whitening* rotation fully recovers the BNT information. (`datavector_sensitivity_haar_nobnt.png`
shows where the statistic's constraining power lives — the deep and coarse channels at the finer scales —
and hence why losing the deep mode hurts.)

**Bottom line for Phase 1:** the *linear* 2D-1D Haar ℓ1-norm reproduces the existing cross-map ceiling
(goal 1) and collapses under BNT (goal 2). This is the clean confirmation of §3.2 — and it means the
modulus (Approach B) is the only remaining lever for *either* goal.

---

## 6. Phase 2 (Approach B, the modulus) — also tested; it underperforms

We built and ran the modulus version (§3.3): 2D starlet → |·| → Haar across bins → S/N-binned ℓ1,
with the per-(mode,scale) noise σ estimated empirically (the modulus folds the Gaussian, so quadrature
is invalid), through the same common NDE + TARP+SBC gate. We pre-validated the transform (the deep mode
`½Σ_b|W_b|` is ≥0 everywhere; einsum exact; a cosmology-sensitivity proxy looked promising). **The gated
result is negative:**

| arm | FoM3 | gate |
|---|---|---|
| modulus Haar, no-BNT | **2234** | PASS-with-caveat (slightly conservative) |
| modulus Haar, BNT space | **706** | PASS-with-caveat |

The modulus form lands **below auto-only (2405)** on goal 1 — worse than the linear form (2676) — and
**collapses under BNT (706)** on goal 2, the same ~3× drop the linear form showed. It is calibrated
(if anything slightly conservative), so this is a real information loss, not over-confidence.

**Why (the mechanism):** the ℓ1-norm *already* uses the absolute value optimally — it bins by *signed*
S/N, so peaks (+) and voids (−) sit in different bins and their asymmetry (cosmologically informative)
is preserved. An *extra* modulus `|W_b|` before the bin-axis Haar **destroys that sign** (a peak and a
void map to the same |W_b|), and the cross-bin power-asymmetry it adds does not compensate. So the
modulus-Haar ℓ1 is a strictly *less* informative use of the wavelet coefficients than the ordinary
signed ℓ1-norm. This is why it falls below even auto-only.

## 7. Combined verdict and the question for Jean-Luc
Across both natural readings of "the 2D-1D wavelet ℓ1-norm with the absolute values":
- **Goal 1:** the linear Approach A ties the existing cross-map ceiling (~2900; no gain over the
  product/conv cross channels we already had); the modulus Approach B underperforms auto-only.
- **Goal 2:** both collapse under BNT (A → 885, B → 706). Only a *whitening* rotation recovers the
  BNT information (separate result), which neither Haar reading is.

So the natural 2D-1D Haar ℓ1-norm, in both forms, does not beat what we already have and does not solve
the BNT inflation — and we understand mechanistically why (A is bounded by the linear-recombination
ceiling per §3.2; B throws away the peak/void sign the ℓ1 relies on). The data also resolve the §3
ambiguity in favour of **Approach A being what "with the absolute values" most likely meant** (B's
extra modulus is counter-productive).

**For Jean-Luc:** does either implementation match what you intended? And given that "linear ties the
ceiling / modulus loses the sign," did you have a specific construction in mind that we are missing — a
different 1D operation along the bins, a signed (peak/void-preserving) cross-bin combination, or a
different placement of the absolute value? We would rather check with you than conclude prematurely
that the idea doesn't help.

## 8. First-principles addendum (2026-06-23): why it failed, and when it *would* work very well

Stepping back from the two specific readings, the negative has a structural cause that also says
exactly when a 2D-1D-style approach shines. A "transform along the bin axis" must satisfy two things:
**G1** read genuine *cross-bin* structure beyond fixed linear combinations; **G2** be *covariant* under
the linear BNT relabelling `B`. The 4-bin construction fails both for intrinsic reasons:

1. **At 4 bins the "1D wavelet" is not a wavelet — it is a fixed 4×4 rotation.** A wavelet needs a
   multiscale axis; on 4 samples Haar just gives a deep mode + three differences, no scale hierarchy.
   So "2D starlet ⊗ 1D Haar" = ℓ1 of four *fixed linear combinations* of the maps, which is bounded by
   the fixed-linear-recombination ceiling (exactly the ~2900 plateau it hit) and adds nothing over the
   conv/product channels. No mother wavelet escapes this — the bin axis is too short.
2. **A fixed basis can never be BNT-covariant.** `H·B` is just another fixed frame, and per-channel ℓ1
   of any fixed frame collapses when `B` rotates the signal-rich deep direction out of every channel.
   Only a `B`-*adaptive* transform survives: whitening `(BBᵀ)^{-1/2}B` (= rotate-back, recovers fully
   but undoes the cuts) or a *learned* mixing (= the CNN). A fixed wavelet is neither — BNT-fragile by
   construction, not by tuning. (This is a theorem, consistent with the M3 whitening result.)

The two failure causes point at the two regimes where it *would* work very well:

- **Many tomographic bins.** This is the real one. With ~15-30+ bins (high tomographic resolution / a
  continuous photo-z weighting) the redshift axis acquires genuine 1D scale structure — the regime
  where the Starck et al. 2009 2D-1D transform shines (their finely-sampled energy axis). A
  2D(space)×1D(redshift-scale) ℓ1 then captures *how the non-Gaussian structure evolves across redshift
  scales* (peaks/voids coherent across z vs localized in z) — real information a 4-bin statistic cannot
  hold. The approach is **bin-count-limited**; at 4 bins it degenerates.
- **A signed, joint, covariant readout instead of a fixed-linear/modulus one — i.e. the joint ℓ1.**
  Both A (linear, bounded) and B (modulus, sign-destroying) fail to read the *joint signed* across-bin
  structure. The construction that does — histogramming the signed across-bin coefficient *vector* — is
  the joint ℓ1: covariant under `B` (BNT-robust, climbs to ×0.72), sign-preserving, and joint (beats
  the linear ceiling). See `JOINT_L1_DEFINITION_AND_THEORY.md`.

**Synthesis.** The 2D-1D *wavelet* and the *joint ℓ1* are two routes to the same goal (generalise the
wavelet ℓ1 across the tomographic axis). At 4 bins the wavelet route degenerates to a fixed rotation
and the joint readout wins; **the joint ℓ1 is the small-bin-count limit of "doing the cross-bin part
right," and a redshift-resolved 2D-1D ℓ1 is what it would become at high bin count** — the wavelet
handling the now-long redshift axis, a signed joint readout handling the cross-structure. That
many-bin, joint-readout version is a genuine future direction (it needs more tomographic bins than
CosmoGridV1's 4), not a rescue for the present 4-bin setup, where the joint ℓ1 already captures
everything the 2D-1D Haar could, and more.

## Figures (full paths)
All under `/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/flatsky_cross_2026_06/plots_2d1d/`:
- **Contour overlay** at a matched cosmology (the 5 arms; Haar-in-BNT-space visibly widest):
  `/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/flatsky_cross_2026_06/plots_2d1d/contours_2d1d.png`
  (PDF: `…/plots_2d1d/contours_2d1d.pdf`)
- **Datavector, no-BNT vs BNT** (the goal-2 collapse; deep channel loses its high-S/N reach under BNT):
  `/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/flatsky_cross_2026_06/plots_2d1d/datavector_bnt_collapse.png`
- **Datavector, the statistic itself** (linear 2D-1D Haar, 4 channels × 5 scales, colored by σ_8):
  `/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/flatsky_cross_2026_06/plots_2d1d/datavector_sensitivity_haar_nobnt.png`
- **Phase-2 modulus-field sensitivity check** (σ_8/Ω_m response of the modulus-Haar field, no-BNT vs BNT):
  `/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/flatsky_cross_2026_06/plots_2d1d/validate_scatter_sensitivity.png`

Full method/spec: `TOMO_2D1D_WAVELET_RESEARCH.md`, `PLAN_2D1D_PHASE_1_2.md`; results detail:
`RESULT_2D1D_PHASE1.md` (linear) and `RESULT_2D1D_PHASE2.md` (modulus) — all in the same
`…/flatsky_cross_2026_06/` directory.
