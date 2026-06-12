# BNT and per-channel statistics: where the information goes — the deep-dive (v2.2, 2026-06-12)

**v2.2 (GATE-C + theory-discussion fold, 2026-06-12 afternoon):** GATE C (TARP+SBC) has now
RUN on the joint-stat q-arms (`overnight_menu/gate_c/GATE_C_JOINT.md`): pair2dq_nobnt FAILS
its registered band (tightest-tercile TARP −0.134, seed-robust), the other three arms
pass-with-caveat; all four are ~4–6% globally over-confident — the registered comparative
downgrade triggered, so the joint-stat headline is "reach at least l1-auto, broadly
comparable to l1+product", never "equal-or-better" (§1.8, §4.3 updated). Also folded: the
plain-language tilt picture for P4c (§1.8), the cut-then-mix resolution of item 2's caveat
and the post-cut frontier (§1.7), and the conv-operator account + Zürcher reconciliation
(in FLATSKY_CROSS_RESULT.md — cross-bin literature gains are IA-self-calibration-dominated,
absent from this forecast).

**v2.1 (overnight-menu fold, 2026-06-12):** the §1.7 rescue menu and the §1.8/§4.3 joint-PDF
program are now MEASURED (16 arms; PLAN_OVERNIGHT_MENU.md has every registered prediction,
overnight_menu/OVERNIGHT_RESULT.md the derived tables). Three additions: the joint one-point
statistics results (§1.8, §4.3), the Gaussian-share decomposition of the l1's BNT loss
(§5.4), and a new geometric remark P4c — binned joint estimators are only as basis-invariant
as their grid is transported, and axis-aligned grids cannot transport under B.

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

Span calibration (§5.4, both rungs run after registration): ONE appended deep channel
recovers 0.730 of (noBNT − BNT) — below its registered 0.8, refuting the single-direction
strong form; TWO depth-distinct deep channels (average + deepest bin) recover 1.082 — the
registered SPANNING branch. The residual of the first rung was tomographic structure among
the four deep kernels, fully retrieved by the second direction: per-channel-accessible
information saturates at about two depth-distinct deep directions for these parameters.

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
| The no-deep-direction account of the collapse (dominant; span-calibrated) | MECHANISM | §5 |
| §5.4 ladder: +1 deep → 0.730 (registered ≥0.8 refuted); +2 deep → 1.082; +6 unions → 1.178 | MEASURED | §5.4 |
| Gaussian (two-point) share of the l1's BNT loss = 0.38 (P7 block appended; rest is non-Gaussian) | MEASURED | §5.4 |
| Joint one-point statistics reach l1-auto +16%, broadly comparable to l1+product, autos alone (GATE C 06-12: noBNT joint arms over-confident in their tightest tercile, pair2d FAILS its band — "equal-or-better" downgraded) | MEASURED + GATED | §4.3, §1.8 |
| Binned joint estimators: invariance requires grid TRANSPORT; axis-aligned grids cannot transport under B | PROVED (geometry) + MEASURED (0.45→0.70) | P4c, §4.3 |

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
2. **Append deep channels** (MEASURED 2026-06-11, both rungs, §5.4): keep the four nulled
   maps untouched — preserving per-slice cuts — and add fixed deep combination(s) as extra
   l1 channels. ONE (the bin average): recovers 0.73 (FoM3 364 → 1854 of 2405). TWO (average
   + the deepest bin κ₄): recovers 1.082 — complete, every marginal at or better than noBNT
   (FoM3 2573; σ(σ8) 0.079 vs 0.082). Two fixed extra channels, no learning, no re-mixing of
   the nulled maps.
   **Real-survey caveat:** the appended deep map is exactly the object with ℓ↔k leakage, so
   in a cut analysis it must itself be cut conservatively — eroding its gain. It is the
   right MECHANISM test (uncut, information-accounting setting), not a survey recipe.
   **Cut-then-mix resolution of this caveat (2026-06-12 discussion):** the caveat applies to
   mixes of RAW maps. Build the deep combinations from the ALREADY-CUT nulled maps instead —
   since κ_k = Σ_{j≤k}(B⁻¹)_{kj}κ̃_j, the deep directions are linear combinations of the BNT
   channels, so "sums of cut BNT maps" reconstruct deep channels that INHERIT the per-slice
   cleaning. Ordering is everything: cut-then-mix is frame-compatible, mix-then-cut is not.
   Uncut, this is identical to item 2 by the span identity (so 1.082 is its uncut limit);
   its new content is entirely post-cut (item 7). Note the sharpening this rests on:
   spanning is NOT the criterion — the four BNT channels already span everything yet
   collapse; what rescues is restoring ~2 SIGNAL-RICH (depth-extended) directions, and
   nonlinear pointwise combos (products: 0.22×; conv+product `both`: 751 single-run) do not
   restore directions.
3. **Append the auto+cross second moments** (MEASURED 2026-06-12: recovers 0.38): the 50
   per-scale wavelet (co)variances appended to the BNT-l1 give FoM3 364 → 1134 of 2405.
   By P7 these restore the entire two-point content exactly, in any basis — so this number
   doubles as a measurement: **38% of what the l1 loses under BNT is Gaussian (two-point)
   content; the remaining 62% is non-Gaussian** (the F5/§5 sector). By construction it can
   do no more.
4. **Append product maps** κ'ᵢκ'ⱼ (measured: 0.15× → 0.22× only): specific quadratic
   combinations; partial by construction (§4.2 says exactly which orders they carry).
5. **Append more linear-combination channels** (MEASURED 2026-06-12: recovers 1.178): the
   six equal-weight pair averages (κᵢ+κⱼ)/2 — the survey-practice union-map analogs, M2 —
   appended to the BNT-l1 give FoM3 2768 (σ(σ8) 0.078 vs noBNT 0.082): full rescue,
   consistent with the §5.4 span account (six deep-ish directions over-span the signal-rich
   subspace). Survey practice, validated.
6. **Joint histograms** (MEASURED 2026-06-12, §1.8/§4.3): as STATISTICS they reach the
   l1-auto+16% level from the auto maps alone, broadly comparable to l1+product (GATE C
   downgraded "equal-or-better"; §1.8); as BNT-robust objects they carry the P4c
   caveat — the law is basis-covariant, a fixed binning of it is not.
7. **The post-cut frontier (registered direction, NOT run — the survey-relevant open
   question):** every rescue above is measured in the UNCUT setting, where rotate-back makes
   the answer trivial. The scientific point of BNT is dropping contaminated channels/scales;
   then B⁻¹ is unavailable and the well-posed question becomes: which linear recombinations
   of the KEPT, CUT nulled channels recover most of the per-channel-accessible information,
   while keeping the systematics rejection that motivated the cuts? Design: choose a
   physical cut schedule (Andreas's input — a physics choice); arms (a) l1 on kept cut BNT
   channels (predicted: collapsed), (b) + pairwise sums of cut channels (predicted: partial
   — two thin slabs are still not deep), (c) + two B⁻¹-weighted reconstructed-deep channels
   from kept channels only (predicted: best obtainable), vs (d) noBNT with the crude uniform
   cut a non-BNT analysis would need. If (c) ≈ (d)'s information WITH the cleaning (d)
   lacks, that is the constructive survey punchline of the pillar. All machinery (mix modes,
   σ-table freezing, sweep) exists.

### 1.8 The joint PDF, in plain language — and can we actually use it?

*What it is.* Instead of histogramming each smoothed map separately (four 1-d histograms),
histogram the maps *together*: each pixel is a 4-vector; bin the 4-vectors in a 4-d grid of
cells; the normalized cell counts are the joint PDF estimate. It answers "how often is map 1
high WHILE map 2 is low AND map 3 is high..." — the cloud's shape, not its shadows. Because
B just relabels the cloud's coordinates, the joint histogram in one basis determines it in
any other (P4b): it is BNT-robust *by construction*, the canonical fixed statistic with the
CNN's invariance property.

*The tilt picture (P4c, in one paragraph).* The robustness above is a property of the
DISTRIBUTION; the estimator inherits it only if the grid follows the basis change. Picture
two channels' values at one position as a point (u₁, u₂); the joint PDF is a cloud. The
histogram lays a checkerboard of upright squares over the cloud and counts points per
square. BNT moves every point by the same linear map: the cloud is tilted and stretched,
and the image of each upright square is a slanted parallelogram. With slanted cells you
would lay the tilted grid and count IDENTICAL occupancies — perfect invariance. But a
histogram only has upright cells: adapting the per-axis ranges fixes the stretch (measured:
ratio 0.45 → 0.70), while no choice of upright cells reproduces the tilt — and the tilt is
where the between-channel correlation lives. Finer K helps only the way a staircase
approximates a diagonal. The learned compressor's first layer applies a linear mix BEFORE
any nonlinearity — it tilts the coordinates first, the one move a fixed per-channel
histogram cannot make. For PAIRWISE statistics there is a second, harder fact: the tilt
that undoes B for the pair (i,j) involves the other channels too (B mixes all four), so
even a tilted 2-D grid in the (i,j) plane could not be exact; only the full joint can.

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

**TESTED (overnight 2026-06-12; full rigor = 3 MAF seeds, 9000-obs pooled medians.
GATE C ran 2026-06-12 — see verdict below the findings):**

| statistic (original basis) | FoM3 | σ(σ8) | σ(w0) | σ(Ωm) |
|---|---|---|---|---|
| wavelet l1, autos | 2405 | 0.082 | 0.245 | 0.053 |
| wavelet l1, autos + product map | 2875 | 0.075 | 0.238 | 0.048 |
| pairwise joint PDF (K=10, autos only) | 2794 | 0.072 | 0.228 | 0.045 |
| joint wavelet l1 (K=10, autos only) | 2788 | 0.072 | 0.229 | 0.045 |
| full-4D joint PDF (K=4) | 2401 | 0.081 | 0.231 | 0.050 |

Three findings. (i) *The joint statistics work:* from the auto maps alone — no cross-map
construction — the pairwise joint PDF nominally matches the l1+product arm (σ(σ8) 0.072 vs
0.075, σ(w0) 0.228 vs 0.238, σ(Ωm) 0.045 vs 0.048; FoM3 −3%). This is the constructive
endpoint of the P5 hierarchy, measured — but see the GATE C verdict below for how to quote
it. (ii) *Counts suffice:* the joint wavelet l1 (amplitude-weighted cells) is statistically
identical to plain counts — the information is in joint occupancy. (iii) *Resolution beats
joint order at fixed budget:* the full-4D histogram at K=4 lands exactly on the marginal-l1
baseline; K=10 pairwise wins. Practicalities: count features REQUIRE dequantization
(+U(0,1), seeded) or the MAF NaNs on quasi-discrete sparse cells; in the BNT basis the same
statistics carry the P4c grid-transport caveat (measured ratios 0.45 fixed-grid → 0.70
axis-adapted; §4.3).

**GATE C verdict (2026-06-12, `overnight_menu/gate_c/GATE_C_JOINT.md`; bands registered
before data):** the noBNT joint arms are mildly over-confident, concentrated in their
TIGHTEST posteriors — pair2dq HIGH-tercile TARP signed dev −0.134 (seed-robust; FAILS its
band), jointl1q −0.080 (pass-with-caveat); all arms SBC std 0.30–0.31 (~4–6% global
under-coverage), while the l1/l1+product comparators were gated at |dev| ≤ 0.037. The
miscalibration is the same order as the nominal edge in (i), so the quotable claim is:
**joint one-point statistics reach at least the l1-auto level (+16% nominal) and are
broadly comparable to l1+product — NOT "equal-or-better".** A pitfall in its own right:
count-histogram datavectors are harder to CALIBRATE through a MAF than weighted reductions
(pure counts −0.134 vs weighted cells −0.080), not just harder to train.

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

**P4c (remark: estimators inherit only as much invariance as their grid transports). NEW
v2.1.** P4b is a statement about the DISTRIBUTION P. A practical estimator is a histogram
T_G(x) = (counts of x(p) in the cells of a fixed partition G). T_G in the new basis carries
the same information as T_G' in the old iff G' = B⁻¹G (the transported partition) — but the
B-image of an axis-aligned cell is a sheared parallelepiped, and axis-aligned partitions are
closed only under DIAGONAL channel maps. So per-channel rescaling of the grid (our adaptive
percentile ranges) implements exactly the diagonal part of the transport and no axis-aligned
grid can implement the rest. ∎ (geometry)
*Measured (overnight, full-4D joint histogram, dequantized, identical treatment per basis):*
fixed noise-scaled grid — BNT/noBNT FoM3 ratio **0.45**; per-axis adapted grid — **0.70**;
the residual is the un-implementable shear plus finite-resolution effects. The learned
compressor's first layer applies precisely this shear (P3) — the basis-adaptivity advantage,
visible at the estimator level. *In words:* even the canonical invariant object is only as
basis-free as its reduction; "BNT-robust by construction" holds for the joint LAW, not
automatically for any fixed binning of it.

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
*Empirical anchor (2026-06-12):* the six equal-weight pair averages appended to the BNT-l1
recover 1.178 of (noBNT − BNT) — survey practice fully rescues the per-channel statistic in
the nulled basis (§1.7 item 5).

### 4.3 The joint PDF as a statistic — design sheet (M3), MEASURED 2026-06-12

Built and measured overnight (per-scale joint histograms of the wavelet coefficients in SNR
units; both registered predictions adjudicated honestly):

- *Registered:* "pairwise-2-d on no-BNT maps ≥ l1+product (strict information containment
  at one point)." *Measured:* every science marginal equal or tighter (σ(σ8) 0.072 vs
  0.075, σ(w0) 0.228 vs 0.238, σ(Ωm) 0.045 vs 0.048); FoM3 2794 vs 2875 (−3%). Verdict: a
  statistical tie read on the marginals — consistent with strict containment minus
  finite-binning losses; the FoM3 deficit is within that statistic's correlation-structure
  sensitivity and should not be over-read in either direction.
- *Registered:* "joint-hist arms are BNT-invariant up to estimator noise (P4b)." *Measured:
  FALSIFIED as operationalized* — full-4D BNT/noBNT ratio 0.45 on the fixed noise-scaled
  grid, 0.70 with per-axis adapted (percentile) grids. Resolution: P4b covariance is a
  property of the LAW; the binned estimator inherits it only through grid transport, and
  axis-aligned grids cannot implement the sheared part of B (P4c, proved). The adapted-grid
  measurement isolates the diagonal part of the transport (0.45 → 0.70); the residual is the
  shear (plus finite resolution). The basis-adaptive (learned) reduction is the object that
  closes this gap — P3's first-layer mix is exactly the missing shear.
- Design lessons for any future deployment: dequantize count features (+U(0,1), seeded;
  three MAF-NaN incidents diagnosed to quasi-discrete sparse cells); prefer pairwise K=10
  over full-4D K=4 at fixed feature budget (resolution beats joint order: full-4D landed on
  the marginal-l1 baseline); plain counts ≈ amplitude-weighted cells; grids in a nulled
  basis must be adapted at minimum, learned ideally.
- Calibration (TARP/SBC/L-C2ST) has not been run on these arms; all numbers are
  constraining-power comparisons at matched pipeline rigor.

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
no deep direction; restore the signal-rich subspace and the per-channel information returns.**
Quantified by the §5.4 ladder: ONE appended deep direction restores 73%; TWO depth-distinct
deep directions (average + deepest bin) restore everything (1.082) — the residual of the
1-deep rung was among-deep-kernel tomographic structure, fully retrieved by the second
direction. Tagged honestly: the invariances and the bounds are PROVED; 0.15×, 1.06/1.01,
0.730 and 1.082 are MEASURED; "per-channel information scales with the spanned signal-rich
subspace, saturating at ~2 depth-distinct directions for these parameters" is MECHANISM —
calibrated at four points but not derived end-to-end.

### 5.4 The decisive cheap test — REGISTERED ≥ 0.8, MEASURED 0.73 (partial)

Registered before any data: appending ONE deep channel — the plain average (κ₁+κ₂+κ₃+κ₄)/4,
fixed, no learning — to the four untouched nulled maps should restore ≥ 0.8 of (noBNT − BNT).
Run 2026-06-11 (`bntdeep_campaign/BNTDEEP_RESULT.md`; BNT columns bit-identical to the
measured 0.15× arm, theta/perm/patch alignment hard-asserted):

| arm | FoM3 | σ(σ8) | σ(w0) |
|---|---|---|---|
| noBNT auto | 2405 | 0.082 | 0.245 |
| BNT auto | 364 | 0.176 | 0.323 |
| BNT + deep (5ch) | 1854 | 0.096 | 0.256 |

**recovered = 0.730 — the registered threshold was NOT met.** Honest reading, both ways:
- *Support:* a single fixed appended channel undoes 73% of the FoM3 loss, ~85% of the σ(σ8)
  damage, and nearly all of the σ(w0) damage. The deep direction is confirmed as the
  DOMINANT carrier of what per-channel statistics lose under nulling.
- *Refutation of the strong form:* "the deep common mode carries essentially all of it" is
  too strong by ~27%. The natural refinement: the original basis contains FOUR deep kernels
  of different depths — a 4-dimensional signal-rich structure — and one average direction
  cannot carry the tomographic information AMONG the deep modes. The full-recovery frames
  (whitening, original axes) span the signal-rich subspace with several channels; the
  bnt+deep frame spans one direction of it. Account update: per-channel-accessible
  information scales with how much of the signal-rich subspace the frame's channels span —
  one direction ≈ 73%, a spanning set = 100% (measured at both ends).
- *Caveat biasing recovered DOWN (unquantified, small):* the deep channel inherits the
  standard 40-bin protocol over its calibrated range, which is wide and heavy-tailed
  ([−12.1, +14.4] in SNR; raw max 52) — coarser binning of the core than the other channels
  get. Some of the missing 27% may be binning efficiency, not missing structure.
- *Second rung (RUN 2026-06-11, registered ladder: ≤0.75 refutes-at-margin / 0.75–0.95
  span-supported / ≥0.95 spanning):* appending TWO depth-distinct deep channels (the average
  AND the deepest bin κ₄ alone; 6-channel arm) gives **recovered = 1.082 — the SPANNING
  branch**: FoM3 2573 (vs noBNT 2405), σ(σ8) 0.079 (vs 0.082), σ(w0) 0.241 (vs 0.245) —
  every marginal at or slightly better than no-BNT (`bntdeep2_campaign/BNTDEEP_RESULT.md`).
  The 1-deep residual is confirmed as among-deep-kernel structure: the second direction
  retrieves all of it. The completed span curve:

  | signal-rich directions in the frame | recovered |
  |---|---|
  | 0 (nulled frame) | 0.00 (the 0.15× collapse) |
  | 1 (+ average) | 0.730 |
  | 2 (+ average + κ₄) | 1.082 |
  | 4 (orthonormal Q) | 1.06 |
  | 6 (+ pairwise unions; overnight) | 1.178 |

  Orthogonal calibration of what the ladder measures (overnight, not a direction count):
  appending the COMPLETE two-point sector instead of directions (the 50 per-scale wavelet
  (co)variances, exact by P7) recovers only **0.38** — so the directions are carrying
  predominantly NON-Gaussian information: ~62% of the l1's BNT loss lies beyond the entire
  Gaussian sector, quantifying F5.

  Values above 1 (deep2, whiten) read as "complete, plausibly a mildly better-conditioned
  frame than the original bins" — the standard per-bin frame is itself not an optimal
  one-point direction sampling (four heavily redundant deep kernels); we do not over-read
  the +7–8% beyond noting it recurs across two independent frames.
Survey-recipe caveat unchanged: this is the uncut information-accounting setting (§1.7
item 2 — in a cut analysis the two deep channels need conservative cuts).

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

### 5.6 The overnight completion (2026-06-12)

Three threads closed at once. (i) The per-channel family's constructive endpoint is
measured: joint one-point histograms reach the l1+product level (broadly — GATE C 06-12
caps the comparison at "comparable", §1.8) from the auto maps alone —
the cross-map construction was, in retrospect, a workaround for not histogramming jointly.
(ii) The BNT loss is decomposed by sector: 38% Gaussian (recoverable by 50 appended
covariance numbers, exact by P7), 62% non-Gaussian (recoverable only through signal-rich
directions or joint structure — the F5 sector, now quantified). (iii) The basis-adaptivity
story closes at the estimator level (P4c): even the provably covariant joint law loses
invariance through any fixed axis-aligned binning, because grids do not transport under B's
shear — per-axis adaptation buys 0.45 → 0.70, and only a learned linear front-end (P3)
implements the rest. One sentence for the paper: *per-channel statistics fail under nulling
because they cannot follow the basis; joint histograms fail more subtly because their grids
cannot either; the learned compressor is the unique member of this family whose reduction
transports.* (GATE C 2026-06-12: joint q-arms gated — noBNT arms over-confident in their
tightest tercile, "equal-or-better" downgraded to "broadly comparable"; §1.8.)

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
