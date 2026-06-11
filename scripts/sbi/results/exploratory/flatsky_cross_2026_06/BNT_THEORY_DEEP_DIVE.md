# BNT and per-channel statistics: where the information goes — the deep-dive (v2, 2026-06-11)

Canonical theory treatment for paper pillar 2. **v2** after Andreas's review of v1: single
smoothing scale throughout (no wavelet machinery — the pipeline applies the same logic per
band, which changes nothing conceptual); a plain-language layer (§1); walked-through math
(§2–§3); the whitening test explained and re-interpreted after a correctness re-pass that
found and fixed an error in v1's post-mortem (§5); the joint PDF made concrete with a
computability verdict (§4.3); the practical menu for rescuing the l1 on BNT maps (§1.7, §4.4);
and a new proposition (P7) that *predicts* the reported literature result that auto+cross
power spectra are unaffected by BNT. Measured numbers from FLATSKY_BNT_RESULT.md and
whiten_campaign/WHITEN_RESULT.md. v1 is in git history (commit dbd2471).

**Notation, once:** four tomographic convergence maps; at each pixel p the maps give a
4-vector κ(p) = (κ₁,κ₂,κ₃,κ₄)(p). Data x = κ + n, shape noise n independent between bins and
pixels, equal variance σ² (injected before BNT, as in the pipeline). BNT: x' = Bx applied
pixel by pixel, with the fixed lower-triangular matrix (tomo4_bnt_v1)

      ⎡  1       0       0      0 ⎤
  B = ⎢ −1       1       0      0 ⎥
      ⎢  0.452  −1.452   1      0 ⎥
      ⎣  0       0.251  −1.251  1 ⎦

All maps are smoothed at one analysis scale before any statistic; "marginal of channel c"
means the 1-d distribution of that one smoothed map's pixel values; "the l1" is the pipeline's
statistic: per channel, histogram the pixel SNR values u = x_c/σ_c (σ_c = that channel's known
noise std) into 40 bins over frozen ranges, and record the sum of |pixel value| in each bin —
a pure function of that single channel's marginal.

================================================================================
## §0 — Executive summary and claims ledger
================================================================================

BNT is invertible, so it destroys no information (P1–P2) and cannot reduce what a
channel-mixing compressor can extract (P3). It devastates the per-channel l1 (0.15× FoM3
auto; 0.22× with the product channel) while the CNN is lossless within seed scatter
(0.93×/0.88×). The whitening test then showed the l1's loss has **no irreducibly-joint
component at all**: one fixed rotation of the nulled maps returns the full no-BNT figure of
merit (recovered fraction 1.06 / 1.01, every marginal). The account that survives our own
correctness re-pass (two earlier explanations were tested against the transform's geometry
and discarded — §5) is physical:

> **BNT trades four deep, overlapping — hence mutually redundant — lensing kernels for one
> shallow map plus three thin lens-redshift slices. Each nulled map alone is signal-starved
> (thin slice vs amplified noise) and carries little of the non-Gaussian structure, which
> lives in the deep common modes that nulling removes from every channel by design. Jointly
> the slices still hold everything (invertibility), but a statistic that looks at one map at
> a time starves. Any frame that contains a deep direction again — the whitened rotation, a
> generic rotation, or the original bins — feeds per-channel statistics; the nulled frame is
> the special frame that contains none.**

Claims ledger (tags: PROVED here / MEASURED in the campaign / MECHANISM = physically grounded
but not derived end-to-end):

| claim | tag | where |
|---|---|---|
| Posterior and information exactly invariant under fixed invertible B | PROVED | P1, P2 |
| CNN class closed under channel mixing ⇒ achievable info basis-invariant | PROVED | P3 |
| BNT moves information only across channels, never across pixels | PROVED | P4a |
| The joint one-point PDF dominates every per-channel statistic in every basis | PROVED | P4b |
| Hierarchy marginals < few projections < joint one-point < field is strict | PROVED | P5 |
| In the Gaussian sector the l1 carries exactly the per-channel variance | PROVED | P6 |
| Auto+cross second moments (power spectra) are EXACTLY BNT-invariant; auto-only are not | PROVED | P7 |
| Honest Gaussian toy: per-channel variances do NOT collapse under nulling (the trap) | PROVED | F3 |
| Mixing of INDEPENDENT non-Gaussian fields contracts standardized cumulants | PROVED | F5 |
| Combinations of signal-poor directions stay signal-poor (slice bound) | PROVED | F5b |
| Union-catalog maps = count-weighted combos of bin maps (no new field info) | PROVED | M1 |
| Pairwise union maps complete at 2nd order, incomplete at 3rd | PROVED | M2 |
| l1 collapse 0.15×/0.22×; CNN 0.93×/0.88×; whitening recovers 1.06/1.01 | MEASURED | campaign |
| σ8 hit hardest / w0 mildest: w0's room capped by the prior (max 1.9× vs measured 1.32×) | MEASURED+derived | F6 |
| The no-deep-direction account of the collapse | MECHANISM | §5 |
| Proposed decisive check: append ONE deep channel to the 4 nulled maps → near-full recovery | PREDICTION (not run) | §5.4 |

================================================================================
## §1 — The story in plain language
================================================================================

### 1.1 Pixels as points, statistics as shadows

Smooth the four maps at the analysis scale. Each pixel now contributes a 4-vector — its value
in bin 1, 2, 3, 4 — so the whole map stack is a cloud of ~6400 points (80×80) in a
4-dimensional space. Every one-point statistic we use is some view of this cloud:

- The **histogram (or l1) of one map** is the cloud's *shadow on one axis*: project every
  point onto the bin-1 axis, look at the resulting 1-d distribution.
- The histogram of a **combined map** like (κ₁+κ₂)/2 is the shadow on a *tilted axis* — a
  direction pointing diagonally between the bin-1 and bin-2 axes.
- The **joint PDF** (§1.8) is the full 4-d shape of the cloud itself.

So "which linear combinations of maps do we histogram?" = "from which directions do we look
at the cloud?". A classical theorem (Cramér–Wold; the CT-scanner principle) says that the
shadows from *all* directions determine the cloud's shape completely — exactly as X-ray
projections from all angles reconstruct a 3-d body. Four axes are four angles: informative,
but partial. Everything in this document is about which angles a statistic uses and what the
cloud looks like from them.

### 1.2 What BNT does to the cloud — and to the kernels

BNT replaces the maps by fixed linear combinations, pixel by pixel: κ'₂ = κ₂ − κ₁, etc. Two
complementary descriptions:

*Geometrically:* the cloud itself is only relabeled — B is invertible, so the cloud's shape
(and all the information) is exactly preserved (P1/P2). What changes is the set of axes we
shadow it onto: the new "per-map" axes are the rows of B.

*Physically:* each original map κ_i sees all lensing structure from z = 0 up to its sources —
a deep kernel. The bins overlap heavily, so the four maps share their dominant structure
(they are strongly correlated — mostly four copies of the same deep field plus increments).
Nulling is designed to cancel the shared part: each nulled map sees only a thin slice of lens
redshift. That is its *purpose* — a systematic at one lens z contaminates one nulled map, not
all four. The price: a thin slice has little signal. And the shape noise, independent
between the original bins, gets combined too: the nulled maps' noise is amplified (×1, 1.41,
1.82, 1.62 — the row norms of B) and *correlated between maps* (e.g. −0.71 between maps 1
and 2, because both contain bin-1's noise with opposite signs).

*Why four or more bins?* The efficiency seen by a source plane at χ_s for a lens at χ_l is
q ∝ 1 − χ_l/χ_s — a TWO-parameter family in lens distance (constant + slope). Nulling the
whole foreground therefore imposes two conditions, Σp_j = 0 and Σp_j/χ_{s,j} = 0, so a fully
nulled combination needs THREE bins; N bins give N−2 fully nulled maps. N = 3 yields one
(legal but nearly pointless); **N = 4 is the first genuinely tomographic nulled set** — the
origin of the "BNT needs ≥4 bins" folklore — and more bins mean thinner slices and a sharper
ℓ↔k mapping. Honest detail of our own matrix: row 2 = (−1, 1, 0, 0) has only two
coefficients, so it satisfies Σp = 0 but cannot satisfy the slope condition — it is a
PARTIALLY nulled slice (rows 3–4 are the proper nulls; their coefficients sum to zero and
encode the slope condition through the bin distances). So the nulled frame is precisely: one
shallow map, one partially-nulled slice, two proper slices — "three thin slices" elsewhere in
this document is shorthand for the latter three.

### 1.3 Why the l1 collapses — and what is conserved

The l1 looks at one nulled map at a time. Each nulled map is a thin, signal-poor slice under
amplified noise: its histogram is nearly the histogram of pure noise, and barely moves when
cosmology moves (this is visible directly in the measured datavectors — the
`datavectors_bnt_vs_nobnt_s8_relative` figure: the auto blocks lose almost all σ8 response
under BNT). The one map BNT leaves untouched is bin 1 — the *shallowest*, weakest bin. So the
nulled frame offers per-map statistics: one weak deep map + three noise-dominated slices.
Measured price: FoM3 falls to 0.15× (σ(σ8) doubles).

Nothing was destroyed: the slices plus their correlated noise can be recombined into the
original maps exactly. The information has moved into *relations between* maps — which
fluctuations appear coherently across slices — where no per-map histogram can see it. The
fraction the l1 keeps (15%) is a *measurement* of how little of its information was ever
per-map-accessible in this frame.

### 1.4 Why the CNN doesn't care

The first thing a multichannel CNN does is form learned linear combinations of its input
channels. Undoing BNT (applying B⁻¹) is therefore literally available to the network inside
its first layer, at zero capacity cost, before any nonlinearity acts (P3). The BNT test
measures whether training finds this in practice: it does (0.93×/0.88×, within compressor-seed
scatter). The CNN is not "smarter" — on friendly bases it does not beat the l1 given the
explicit product channel (0.83–0.85×, pillar 1) — it is *basis-robust* where per-channel
statistics are basis-fragile.

### 1.5 The two-point fact from the field — predicted by this framework

It has been reported that BNT leaves power-spectrum contours unchanged *when both auto- and
cross-spectra are used* (we do not re-derive the literature here; [REF when assembling the
paper]). In this framework that is a two-line theorem, not a coincidence (P7): the full set
of auto+cross second moments transforms among themselves invertibly under B (Ĉ' = BĈBᵀ, and
B is known), so the BNT-basis datavector is a lossless repackaging of the original one —
*identical* posteriors, exactly, for any field, Gaussian or not. The same two lines show why
auto-spectra alone are NOT protected: the diagonal of BĈBᵀ cannot be inverted back without
the off-diagonals. The rule of thumb this gives cosmologists: **a statistic survives BNT iff
the transformed statistic can be reassembled from the measured one. Auto+cross 2-pt: yes,
exactly. Per-map histograms/l1/peaks (any order, autos only): no.** Our pillar-2 measurement
is the higher-order, autos-only member of this family, and 0.15× is its price tag.

### 1.6 The whitening test, in plain language

*What it is.* Take the nulled maps and apply one more fixed matrix, Q = (BBᵀ)^(−1/2)B,
chosen so that the shape noise becomes again independent and equal between the four channels.
Algebra gives a bonus: this particular combination is exactly a pure *rotation* of the
original (un-nulled) basis — mixed signal, but pristine noise.

*Why not simply B⁻¹?* Good question with an instructive answer. ANY matrix W that restores
independent equal noise to the nulled maps must make the net transform WB a pure rotation of
the original maps — so the whiteners form a family, W = O·B⁻¹ over rotations O, and **B⁻¹
itself is the member with O = I**: it lands exactly back on the original maps. Using it would
have re-measured the no-BNT arm and proven nothing. Q is the symmetric member of the family,
landing on a genuinely rotated frame (19°–37° from the original axes, all channels mixed) —
which makes the test falsifiable: "is there anything special about the original axes, or
does ANY clean-noise frame feed the l1?" Measured answer: any.

*Why we ran it.* To split the l1's collapse into two conceptually different parts:
(a) damage from the nulled frame's *geometry* (signal-starved axes, amplified/correlated
noise) — which a change of frame can undo; and (b) information that genuinely requires
looking at several channels *jointly* — which NO frame's per-map histograms can reach. The
recovered fraction (whiten − BNT)/(noBNT − BNT) measures the share of type (a).

*The result and its reading.* Recovered = 1.06 (auto) and 1.01 (+product): **complete
recovery — type (b) is empty.** Every single marginal recovers too (σ(σ8): 0.080 whitened vs
0.082 no-BNT vs 0.176 BNT). In words: everything the l1 ever measured is still available
from four *single-map histograms*, provided the four maps are taken in a sane frame. The
collapse was never about needing joint statistics; it was the nulled frame itself. (Values
marginally above 1 are read as "complete to within a few percent" — re-train repeatability
for these pooled arms was not separately measured.)

*The catch (important):* Q re-mixes the nulled maps, destroying exactly the slice-locality
BNT was applied for. So whitening is *information accounting*, not an analysis recipe. The
practical lesson is the decoupling: use BNT to clean (nulling-informed cuts), then choose
whatever basis feeds your statistic (§1.7).

*Honesty note.* Our pre-registered theory expectation was partial recovery at best; the data
said full. Two successive explanations we wrote down were checked against the transform's
geometry and discarded before the present account survived — the full post-mortem chain is
kept in §5, deliberately, as part of the paper's "journey" material.

### 1.7 What would rescue the l1 on BNT maps — the practical menu

Ranked; "measured" = this campaign, "exact" = proved, "predicted" = registered, not yet run.

0. *The survey workflow these all live in:* BNT's purpose is that each nulled map sees a
   thin lens slice, so ℓ ↔ k is sharp and k-cuts (where baryons/nonlinearity live) become
   clean per-map ℓ-cuts with no leakage between scales. Every item below APPLIES THE CUTS IN
   BNT SPACE FIRST; they differ only in where the statistic is then computed.
1. **Decouple cleaning from measuring** (exact, free): null, make the nulling-informed cuts,
   then rotate back (B⁻¹ on the cut maps — or any fixed rotation; post-cut they are all
   information-equivalent) and run the l1 in the original frame. This null → cut → invert →
   measure pipeline has been proposed in the literature [REF]; our results are its
   quantitative justification for higher-order statistics: skipping the rotate-back step
   costs 85% of the l1's FoM3 (measured), performing it costs nothing (measured, whitening),
   and P1/P4b are the proof-level statements that the post-cut basis is a free choice.
2. **Append ONE deep channel** (predicted near-full recovery; cheap decisive test, §5.4):
   keep the four nulled maps untouched — preserving per-slice cuts — and add the plain bin
   average (κ₁+κ₂+κ₃+κ₄)/4 as a fifth l1 channel. If the no-deep-direction account is right,
   most of the lost FoM3 returns with a single fixed extra channel and no re-mixing of the
   nulled maps. Not yet run; would be a ~3 h mini-campaign with existing machinery.
   **Real-survey caveat:** the appended deep map is exactly the object with ℓ↔k leakage, so
   in a cut analysis it must itself be cut conservatively — eroding its gain. It is the
   right MECHANISM test (uncut, information-accounting setting), not a survey recipe.
3. **Append the auto+cross second moments** (exact for the Gaussian sector): 10 numbers
   (the 4 variances + 6 covariances of the smoothed maps; per scale if multi-scale). By P7
   this restores the entire two-point content exactly, in any basis, for ~free. It cannot
   restore non-Gaussian information.
4. **Append product maps** κ'ᵢκ'ⱼ (measured: 0.15× → 0.22× only): specific quadratic
   combinations; partial by construction (§4.2 says exactly which orders they carry).
5. **Append more linear-combination channels** (union-style, several weight ratios): each
   new direction adds shadows; M2 quantifies exactly which mixed cumulants each ratio pins
   down. Diminishing returns per channel; the limit of all directions is item 6.
6. **Pairwise joint histograms** (the principled completion at one-point level; §1.8/§4.3):
   replaces shadows by 2-d views; basis-covariant, hence BNT-robust by construction.

### 1.8 The joint PDF, in plain language — and can we actually use it?

*What it is.* Instead of histogramming each smoothed map separately (four 1-d histograms),
histogram the maps *together*: each pixel is a 4-vector; bin the 4-vectors in a 4-d grid of
cells; the normalized cell counts are the joint PDF estimate. It answers "how often is map 1
high WHILE map 2 is low AND map 3 is high..." — the cloud's shape, not its shadows. Because
B just relabels the cloud's coordinates, the joint histogram in one basis determines it in
any other (P4b): it is BNT-robust *by construction*, the canonical fixed statistic with the
CNN's invariance property.

*How to compute it (concretely, our pipeline).* Smooth the 4 noisy maps at the analysis
scale; standardize each channel by its known noise σ_c (as the l1 already does); choose bin
edges per channel (frozen percentiles, as the l1 already does); call a 4-d histogram routine
(`np.histogramdd` / `torch` bucketize+scatter_add — GPU-trivial) per patch; flatten counts
into the datavector; feed the same MAF. It is the same plumbing as the l1 with a different
reduction. Cost: negligible vs the loader pass.

*Is it computable for our analysis — honestly?* The sizing is the only real question. A
patch has 80×80 = 6400 pixels. Full 4-d: 5 bins/axis → 625 cells (~10 px/cell — workably
sparse); 6/axis → 1296 (~5 px/cell — thin). Pairwise 2-d (all 6 pairs, 15×15) → 1350 numbers
(~28 px/cell) and captures everything up to pairwise dependence — strictly more than all our
product channels combined. Either variant is *smaller* than the L1-both datavector (3200) the
MAF already digests. There is NO covariance-matrix obstacle in SBI (the classical blocker for
joint PDFs — inverting a 10³–10⁴-cell covariance — simply does not arise; the NDE consumes
the vector directly). Sparse counts just mean noisy features, which SBI tolerates; coarser
binning trades resolution for stability.

*Does it defeat BNT's purpose?* No — the opposite. The purpose of BNT is the sharp ℓ↔k
mapping of thin lens slices, which makes per-map scale cuts physical. The joint PDF respects
that completely: apply the per-slice cuts in BNT space exactly as designed, then histogram
the CUT NULLED MAPS jointly — **you never leave BNT space at all**, because the joint
histogram is frame-indifferent (P4b: same information in any invertibly-related frame). The
cut structure stays manifest; only the basis-fragility of per-channel reductions evaporates.
Survey practicality is that of any one-point statistic (masks = drop pixels; varying
noise/depth = per-channel standardization — the l1's existing machinery applies verbatim).

**Verdict: yes, we could just test it** — a
third-pillar mini-campaign (build pass + slices + sweeps, the whiten-campaign template,
~2–3 h/arm) with pairwise-2-d as the default and full-4-d as a variant. Needs your explicit
go; the interesting arms are (noBNT vs BNT) × (joint-hist vs l1) — the prediction is
BNT-invariance of the joint-hist arms.

================================================================================
## §2 — Formal core (single scale; proofs walked through)
================================================================================

Per-channel statistic class: t(x) = (t₁(x_{c₁}),…,t_K(x_{c_K})) — every component a function
of ONE channel's marginal (the l1, per-bin PDFs, per-bin peaks/Minkowski). Channel-mixing
class F: networks whose first layer is linear in channels (our CNNs).

**P1 (posteriors don't see invertible transforms).** p(θ|Bx) = p(θ|x).
*Proof.* The likelihood of the transformed data is p'(x'|θ) = p(x|θ)/|det B|^{N_pix}
(change of variables; one factor per pixel). The Jacobian does not depend on θ, so it cancels
between numerator and evidence in Bayes' theorem. ∎
*In words:* relabeling data invertibly cannot change what they say about θ — the entire BNT
inflation must therefore be a property of the summary statistic, never of BNT.

**P2 (information doesn't either).** I(θ; Bx) = I(θ; x).
*Proof.* Mutual information never increases under a deterministic map (data-processing
inequality): I(θ;Bx) ≤ I(θ;x). Apply the same to B⁻¹ acting on Bx: I(θ;x) ≤ I(θ;Bx). ∎

**P3 (the CNN class absorbs B).** For every f in F, f∘B is also in F; hence
max_{f∈F} I(θ; f(Bx)) = max_{f∈F} I(θ; f(x)).
*Proof.* The first layer computes, at each location, Σ_i K_{oi} x_i over channels i (K = its
kernels). Feeding Bx instead gives Σ_i K_{oi}(Bx)_i = Σ_j (Σ_i K_{oi}B_{ij}) x_j — the same
layer with kernels K·B: another member of F with the same parameter count. B invertible makes
f ↦ f∘B a bijection of F onto itself, and a supremum over a set is invariant under a
bijection of that set. Pipeline preprocessing preserves this: demeaning is linear (commutes
with B); per-channel standardization is a diagonal matrix D per basis, and D'⁻¹BD is again
one absorbable channel map. ∎
*In words:* "undo the nulling" is a configuration of the network's first layer; the
hypothesis class is identical in both bases, so only optimization difficulty can differ —
the measured 0.93× is that optimization residual, not an information loss.

**P4a (BNT never moves information across pixels).** For any pixels p₁…p_k and channels
i₁…i_k: ⟨x'_{i₁}(p₁)···x'_{i_k}(p_k)⟩ = Σ_{j₁…j_k} B_{i₁j₁}···B_{i_kj_k}
⟨x_{j₁}(p₁)···x_{j_k}(p_k)⟩.
*Proof.* x' = Bx at each pixel separately; expand the product, take expectations. ∎
*In words:* a transformed correlation function at given pixel positions is a combination of
original correlation functions at the SAME positions. BNT shuffles information between
channel-marginal and channel-joint structure only — never between scales or positions. So
"where does the information go?" has an exact answer: into the joint one-point structure,
and nowhere else; recovering it never requires multi-point cross statistics.

**P4b (the joint one-point PDF dominates all per-channel statistics, in every basis).**
Let P = the distribution of the 4-vector x(p) over pixels. (i) P' = the image of P under B —
knowing P in one basis is knowing it in all. (ii) Each channel's marginal is a projection of
P, and every per-channel statistic is a function of these marginals. Hence (iii) whatever any
per-channel statistic extracts in ANY basis is computable from P, whose information content
is basis-independent. ∎
*In words:* the joint histogram is an information envelope over the whole per-channel family
— the "fixed statistic that survives BNT" of §1.8.

**P5 (the hierarchy is strict).** marginals < finitely many directions < joint one-point <
full field. *Witnesses for each gap:* (a) two correlated-vs-uncorrelated bivariate Gaussians
with identical marginals; (b) two distinct 4-d distributions agreeing on any finite set of
projection directions (classical non-uniqueness of finite Radon data); (c) two Gaussian
fields with the same pixelwise covariance Σ at the analysis scale but different spatial
correlation shapes: identical joint one-point PDFs, different fields. ∎
*In words:* shadows < a few angles < the cloud's shape < the cloud plus how it is woven in
space. BNT-induced losses live entirely below level 3 (P4a); field-level information beyond
the one-point cloud is the CNN's (and only the CNN's) territory here.

**P6 (in the Gaussian sector, the l1 is the variance).** If a channel's marginal is Gaussian
with std σ_tot, the expected content of each SNR bin is an explicit error-function expression
whose only free parameter is σ_tot/σ_c. So the expected l1 datavector of that channel is a
deterministic curve through the single number σ_tot — no more information than the variance.
*In words:* everything the l1 adds beyond a power spectrum is non-Gaussian marginal shape;
that is the sector where its BNT damage must (and does) live.

**P7 (auto+cross second moments are exactly BNT-invariant; autos alone are not). NEW.**
Let Ĉ be the 4×4 sample covariance of the smoothed maps over pixels (4 autos + 6 crosses;
the single-scale stand-in for auto+cross power spectra — per ℓ-bin the argument is verbatim).
Under BNT, Ĉ' = (1/N)Σ_p (Bx)(Bx)ᵀ = BĈBᵀ — *exactly*, realization by realization, not just
on average. The map Ĉ ↦ BĈBᵀ on symmetric matrices is invertible (apply B⁻¹·Bᵀ⁻¹). So the
BNT-basis datavector is an invertible function of the original one, and by P1-at-the-summary-
level the posteriors are identical — for any field, Gaussian or not, and for any noise
(noise spectra transform congruently and are known). The autos alone are diag(BĈBᵀ): four
numbers depending on all ten originals — not invertible, no protection. ∎
*In words:* this *predicts* the reported lossless-2-pt-with-crosses result [REF], and locates
our measurement precisely: the l1-auto arm is the (higher-order, autos-only) configuration
maximally exposed to the basis, and auto+cross 2-pt is the configuration provably immune.
Practical corollary: appending the 10 numbers of Ĉ to any BNT-basis datavector restores the
complete Gaussian sector for free (§1.7 item 3).

================================================================================
## §3 — The worked Gaussian toy (all closed-form, walked through)
================================================================================

**Fisher information in three sentences.** For data with distribution p(data|θ), the Fisher
information I(θ) measures how distinguishable nearby θ values are; 1/√I is the best error
bar any unbiased analysis can reach. For a *summary* statistic T with mean μ(θ) and
covariance Σ_T, the usable information is I_T = (∂μ/∂θ)ᵀ Σ_T⁻¹ (∂μ/∂θ): "how fast the
expectation moves, measured against how much it scatters." Comparing I_T to the full-data I
tells us what the summary wastes.

**The model.** One scale, N i.i.d. pixels (the idealization that makes everything explicit).
Per pixel y ~ N(0, C(θ)), C = S(θ) + σ²I (signal + noise). Full-data Fisher per parameter
pair: I_full = (N/2)·tr(C⁻¹C_{,a}C⁻¹C_{,b}). Substituting C → BCBᵀ leaves this unchanged
(all B's cancel cyclically) — F-side restatement of P1/P2. **(F1)**

**The per-channel summary.** T̂_c = the sample variance of channel c (by P6, the Gaussian-
sector stand-in for the l1). Its statistics: E[T̂] = diag(C); Cov(T̂_i,T̂_j) = (2/N)·C_ij²
(a Wick-theorem identity). Note the one place cross-channel structure touches a per-channel
statistic: correlated CHANNELS make the two sample variances scatter TOGETHER (the C_ij²),
even though neither looks across channels. Summary Fisher: I_diag = (N/2)·rᵀM⁻¹r with
r = ∂θ diag(C) and M_ij = C_ij². **(F2)**

**The 2-bin nulling toy and the trap. (F3)** Caricature of nested kernels: bin 2 sees bin 1's
lenses plus an increment, κ₂ = κ₁ + β, with κ₁ ⊥ β, Var(κ₁) = Au, Var(β) = Av; A is an
amplitude parameter (σ8-like). Nulling: B = [[1,0],[−1,1]], so the nulled pair is (κ₁, β) —
the toy nuller exactly isolates the increment. (Bin-count honesty: a pure difference enforces
only the Σp = 0 condition, which acts as an exact nuller here ONLY because the toy's
foreground is a one-parameter family — pure common mode. Real kernels are the two-parameter
family of §1.2, which is why true nulling needs three bins per row and the folklore says BNT
needs ≥4 bins overall; none of the toy's conclusions depend on this, since the toy's job is
to test the Gaussian-geometry story, not the nulling construction.) Covariances:

  original C = [[Au+σ², Au],[Au, A(u+v)+σ²]]      nulled C' = [[Au+σ², −σ²],[−σ², Av+2σ²]]

(note the nulled off-diagonal is PURE NOISE — the anti-correlated shape noise of §1.2).
Work in the regime where nulling matters: strong shared signal, weak increment, Au ≫ σ² ≫ Av.
Plugging into F2 and keeping leading terms (expansions verified twice):

  I_diag^original ≈ (N/2)·[ 1/A² + v²/(4Auσ²) ]
  I_diag^nulled   ≈ (N/2)·[ 1/A² + v²/(4σ⁴)   ]  ≈  I_full

Since Au ≫ σ², the nulled-basis term v²/4σ⁴ is the LARGER one: **in this honest Gaussian
toy, per-channel variances are MORE informative after nulling, and essentially efficient.**
Why: in the original basis the two sample variances are almost perfectly correlated (both
dominated by the same κ₁), and the increment hides in their difference, suppressed by the
huge common variance; nulling un-buries it. The correlated noise costs only a sub-leading
term. Three lessons, all load-bearing:
 (i) the folk story "BNT correlates the noise and lowers per-map S/N, hence per-channel
     statistics fail" is *backwards* at the Gaussian one-point level — it cannot be the
     mechanism of the measured 0.15×;
 (ii) the toy is "too kind" to BNT in one identifiable way: its nulled basis still contains
     the deep direction (κ₁ — the toy's common mode survives as channel 1). The real B keeps
     only bin 1, the *shallowest* map, and no channel retains the deep common structure —
     see §5; the toy thus brackets the real case from the favorable side;
 (iii) where real damage CAN enter even at this level: if the response of the nulled
     marginals to θ degenerates (∂μ losing rank) — and the measured σ8-flat BNT datavector
     blocks show the real case does sit in that regime.

**The whitening operator, formally. (F4)** Q = (BBᵀ)^(−1/2)B satisfies QQᵀ = I (orthogonal —
verified numerically to 4·10⁻⁹ in the campaign code): noise N(0,σ²I) again, signal rotated.
What whitening can restore: anything attributable to the nulled frame's noise/signal
geometry. What it cannot: information absent from EVERY frame's marginals (the genuinely
joint residue). Hence the decomposition logic of §1.6. Pre-registered expectation was
partial recovery; measured: full (see §5 for the post-mortem).

**When does mixing Gaussianize? — done honestly. (F5)** For INDEPENDENT fields κ_j with
variances λ_j and standardized k-th cumulants γ_j, the mixture m = Σ b_jκ_j has
  γ_k(m) = Σ_j b_j^k λ_j^{k/2} γ_j / (Σ_j b_j² λ_j)^{k/2},
and for k ≥ 3 one shows |γ_k(m)| ≤ max_j|γ_j| with equality only if a single component
contributes (norm-monotonicity ‖z‖_k ≤ ‖z‖₂ applied to z_j = |b_j|√λ_j) — mixing independent
non-Gaussian fields is a one-step CLT. **(F5, proved.)** BUT — and v1 of this document
misused this — tomographic bins are strongly CORRELATED, and the lemma's independence is
essential: averaging a field with itself changes nothing (γ(κ+κ) = γ(κ)). The true
statements for our stack: same-sign combinations of strongly correlated maps preserve the
shared non-Gaussian structure; *differencing* them removes it — and the nulling rows are
precisely the differencing combinations that remove the shared (deep) structure from every
channel. What survives per nulled channel is the thin slice: small against amplified noise,
and intrinsically more Gaussian (higher-z, more linear structure). This — not generic
"mixing Gaussianizes" — is the non-Gaussian half of the collapse mechanism.
**(F5b, the slice bound, proved):** if a set of unit directions each sees signal variance
≤ s (the slices), any unit combination w of them sees w^TSw ≤ (Σ|c_i|√s)² ≤ (dim)·s
(Cauchy–Schwarz on the cross terms). So recombining slices cannot manufacture a deep
direction — escaping the starved frame requires a component OUTSIDE the nulled span, which
is exactly what Q's first row has (§5.3).

**The σ8/w0 anisotropy, adjudicated from artifacts. (F6)** Measured inflation of the l1-auto
marginals: σ(σ8) 0.082→0.176 (2.15×), σ(Ωm) 0.053→0.090 (1.70×), σ(w0) 0.245→0.323 (1.32×).
Two mechanisms, both real: (i) *prior geometry* — the effective prior is the training-grid
support, with uniform-equivalent widths σ_prior ≈ (0.115, 0.288, 0.462) for (Ωm, σ8, w0).
The maximum inflation the prior wall allows is σ_prior/σ_noBNT ≈ (2.2×, 3.5×, 1.9×): w0's
posterior was already within a factor ~2 of the prior before BNT acted, so its mild 1.32× is
substantially a ceiling effect rather than resilience; as a fraction of the *available* room,
all three parameters lose a comparable 61–78%. (ii) *sector exposure* — σ8 is the amplitude
of exactly the deep non-Gaussian structure that nulling removes from every channel (F5b/§5),
so it pays the largest physical share; kernel-shape parameters keep some response in the
surviving slice structure (and B, frozen at the fiducial, re-lights the nulled channels at
linear order when kernel-moving parameters shift — a leakage signal per-channel statistics
CAN see). (i) is derived from artifacts; (ii) is MECHANISM.

================================================================================
## §4 — Survey practice, completeness order-by-order, and the joint PDF
================================================================================

### 4.1 Union catalogs add bookkeeping, not field information (M1)

Surveys often build "cross" statistics by merging catalogs of two bins and re-running the
map-maker. Because the map estimator is linear in galaxies, the union map is exactly the
count-weighted average of the per-bin maps, NOISE INCLUDED — same galaxies, same noise
realizations, regrouped:
  κ̂_{i∪j} = (N_iκ̂_i + N_jκ̂_j)/(N_i+N_j),
and its noise variance comes out identical either way (σ_e²/(N_i+N_j); proof is two lines of
variance bookkeeping). So everything a union-catalog analysis extracts is computable from the
per-bin maps already in hand — the BNT inflation is NOT a data-access problem, and "we would
need the catalogs" is never the blocker. (Caveat: spatially varying counts/weights/masks make
the combination position-dependent — bookkeeping, not new information.)

### 4.2 Which combinations recover which cross-information (M2) — the accounting

A union/combination map is a direction w in channel space (§1.1). Its k-th cumulant expands
multilinearly — for a pair (i,j):
  cum_k(w_iκ_i + w_jκ_j) = Σ_{m=0}^{k} C(k,m)·w_i^m w_j^{k−m}·cum(κ_i:m, κ_j:k−m),
a polynomial in the weights whose coefficients are the order-k mixed cumulants. The two autos
give the m = 0 and m = k terms. Each distinct weight RATIO gives one new equation. Worked
order 3 (the first non-Gaussian order): unknowns cum(κ_i²κ_j) and cum(κ_iκ_j²) — two of
them; one equal-weight union gives ONE equation (3·cum(κᵢ²κⱼ) + 3·cum(κᵢκⱼ²) = known) —
**underdetermined**. A second ratio (say 2:1) closes it (the 2-equation system has a
Vandermonde-type determinant, nonzero for distinct ratios). General rule: order k needs k−1
distinct ratios per pair; three-bin mixed cumulants (cum(κᵢκⱼκ_l)) need three-bin unions.
So: **pairwise equal-weight unions are provably complete at second order and provably
incomplete at third** — survey practice samples the cross-information, order by order, and
the limit of all directions is the joint PDF (Cramér–Wold, §1.1). Two useful extras: Gaussian
noise has no cumulants beyond k = 2, so order-≥3 mixed cumulants from combination maps are
noise-bias-free in expectation; and w·κ' = (Bᵀw)·κ — the same combination family is
constructible from BNT maps (another face of "the statistics basis is free").

### 4.3 The joint PDF as a statistic — design sheet (M3)

§1.8 gives the plain-language version and the computability verdict (yes; pairwise-2-d
default at 6 pairs × 15×15 = 1350 numbers, full-4-d 5⁴ = 625 as variant; GPU-trivial; same
MAF; no covariance obstacle in SBI). Design choices if/when tested: channel standardization
by known noise σ_c (matrix whitening by Σ'^{1/2} optional in the BNT basis); frozen
percentile bin edges (the l1's convention); counts vs counts-of-|value| (plain counts =
the PDF; the l1-analogue weighting is also available). Prediction to register at launch:
joint-hist arms are BNT-invariant up to estimator noise (P4b); pairwise-2-d on no-BNT maps
≥ l1+product (it strictly contains the product's information at one point, §4.2). Status:
NOT launched; needs explicit go (third pillar).

### 4.4 The practical menu

Consolidated in §1.7 (items 1–6 with status tags); the supporting results are P7 (item 3),
M2 (item 5), P4b/M3 (item 6), and §5.4 (item 2).

================================================================================
## §5 — The whitening post-mortem (kept in full, deliberately) and the final account
================================================================================

### 5.1 What was registered and what was measured

Registered before the run: recovery LOW-to-MID (reasoning: Q is itself a mixing; v1's
reading of F5 said mixing per se contracts the non-Gaussian content the l1 lives on).
Measured: recovery COMPLETE (1.06/1.01, every marginal). The registered prediction was
falsified.

### 5.2 First correction — also wrong, and how we caught it

v1's post-mortem blamed sign structure: "Q's rows are same-sign averages, B's are signed
differences; F5 bites only along signed rows." Check against the actual matrices: Q's rows
2–4 are NOT same-sign (row 4 = (0.00, −0.04, −0.59, 0.81) is a near-difference of adjacent
bins). The explanation died on contact with the transform it claimed to explain.

### 5.3 The account that survives (and the small lemmas that frame it)

Compute the geometry (constants of the fixed transform): the unit-noise nulling directions
are the normalized rows b̂_i of B; Q's rows lie 19°–37° away from the corresponding b̂_i, and
— the killer for any pure "alignment" story — Q's rows 3–4 lie 95–99% INSIDE the span of the
nulling directions b̂₂..b̂₄. What actually distinguishes the frames:

- Each *individual* nulling direction sees only a thin lens-redshift slice: little signal
  (per-direction starvation), little of the deep non-Gaussian structure (F5 differencing).
- By the slice bound (F5b), recombining within the nulled span cannot manufacture a deep
  direction — directions mostly inside that span stay slice-like.
- The decisive asset is Q's FIRST row, (0.875, 0.392, 0.221, 0.180): 70% of its power lies
  OUTSIDE the nulled span — it is, to good approximation, the deep common mode (≈ the plain
  bin average: the single most signal- and non-Gaussianity-rich direction in channel space,
  the one direction nulling is built to remove from everywhere). The nulled frame's only
  un-nulled channel is bin 1 alone — the shallowest, weakest map.
- The toy (F3, lesson ii) confirms the pattern from the other side: ITS nulled basis kept
  the deep direction, and showed no collapse.

So the durable statement: **the l1's BNT collapse is the starvation of a frame that contains
no deep direction; restore one deep direction (Q does; a generic rotation does; the original
axes do fourfold-redundantly) and the per-channel information returns in full — measured.**
Tagged honestly: the invariances and the bounds are PROVED; the full-recovery and the 0.15×
are MEASURED; "the deep direction carries the bulk" is MECHANISM — strongly supported
(datavector response figure; the toy; the geometry above) but not derived end-to-end.

### 5.4 The decisive cheap test (registered, not run)

If §5.3 is right, appending ONE deep channel — the plain average (κ₁+κ₂+κ₃+κ₄)/4, fixed, no
learning — to the four untouched nulled maps should restore most of the lost FoM3, while
keeping the nulled channels available for per-slice cuts (unlike whitening, which destroys
them). Five-channel l1, one extra build slice, whiten-campaign template (~3 h). PREDICTION:
recovery ≥ 0.8 of (noBNT − BNT). If it fails, §5.3 is wrong and the joint share is hiding
somewhere subtler — either outcome is paper material.

### 5.5 Synthesis (the two pillars, final form)

Per-channel statistics are statistic-strong, basis-fragile: in a friendly basis with the
right explicit channel the l1 beats the trained compressor (CNN/L1 = 0.83–0.85× on product,
seed- and recipe-robust); in the nulled frame it keeps 15%, and the whitening test proves the
loss is frame geometry, not a need for joint statistics. The CNN is basis-robust, not
statistic-optimal: class closure (P3) makes its reachable information exactly invariant —
measured residual 0.93×/0.88×. The 2-pt sector is provably frame-immune once crosses are
included (P7 — matching the reported literature result), and the practical resolution for
everything above 2-pt is the decoupling: **null to clean, then measure in a frame with deep
directions** — or hand the statistic the one deep channel it is missing (§5.4).

================================================================================
## Appendix — exact constants of tomo4_bnt_v1 (derived from B; GATE A1b verified the
## noise rows empirically)
================================================================================

Row norms (noise amplification): (1.000, 1.414, 1.820, 1.621).
Post-BNT noise correlations: ρ₁₂ = −0.707, ρ₂₃ = −0.740, ρ₃₄ = −0.548, ρ₁₃ = +0.248,
ρ₂₄ = +0.110, ρ₁₄ = 0.
Noise-ellipsoid (BBᵀ) eigenvalues: (0.088, 0.838, 2.417, 5.599) — condition number 63
(noise std 0.30σ to 2.37σ across channel-space directions).
Whitener Q = (BBᵀ)^(−1/2)B (orthogonal to 5·10⁻¹⁵ here; 4·10⁻⁹ in the float32 campaign code):
  ⎡ 0.875  0.392  0.221  0.180 ⎤
  ⎢−0.482  0.648  0.460  0.369 ⎥
  ⎢ 0.047 −0.652  0.628  0.423 ⎥
  ⎣ 0.002 −0.042 −0.588  0.808 ⎦
Angles ∠(Q_i, b̂_i): 29.0°, 37.0°, 28.8°, 19.0°. Fraction of each Q row inside
span(b̂₂,b̂₃,b̂₄): 0.305, 0.753, 0.950, 0.992 — Q row 1 is the deep direction (§5.3).
