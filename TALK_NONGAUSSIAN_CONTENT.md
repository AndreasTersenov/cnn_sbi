# Talk content: "The Non-Gaussian Universe" (2026-06-16)

**Speaker:** Andreas Tersenov · **Format:** ~30 min, two acts · **This is a content package, not a deck.**
Narrative → slide-by-slide content → figure-per-slide manifest. Figures live in `talk_figures/`.

**Design standard:** follow `TALK_BEST_PRACTICES.md` when building the deck and reworking plots,
assertion-evidence headlines (the slide titles below become full-sentence messages; drafts in that
doc's §D), one message per slide, KISS/colorblind-safe figures, ABT spine, conclusions stated early.

Built 2026-06-14 from the vetted sources (`HANDOFF_TALK_NONGAUSSIAN_2026-06.md`, `PAPER_MESSAGES.md`,
the `FLATSKY_*_RESULT.md` docs, the `project_cnn_nde_swap_resolves_m1` memory, `SUMMARY_ARCH.md`,
`PAPER_NARRATIVE_AND_PITFALLS.md`) and from the submitted baryonic-feedback paper
(`/home/tersenov/papers/Impact_of_Baryonic_Feedback_Submission/main.tex`).

**Confirmed design decisions (Andreas, 2026-06-14):**
- **Registered title (locked): "Do Baryons Break Higher-Order Statistics?"**, Tue 16 Jun, 12:30–13:00
  (30 min), Dougalis room, FORTH. The abstract on file covers **only Part 1** (baryonic feedback); Part 2
  (this SBI work) is an unannounced extension. So **the whole talk answers the title question**, and the
  answer spans both parts: *no, HOS are resilient to cheap scale cuts, and even the BNT "break" is an
  analysis artifact a learned compressor (or one rotation) undoes.* Part 2 is the deeper resolution.
- ~30 min total. **This SBI work is the second half**; the first half is the baryonic-feedback paper.
  The two are one connected story. Build the whole arc.
- Spine: a **mix of "non-Gaussian information: analytical vs deep learning, and can we trust it"**
  (arc 1) and **"where inter-bin / cross-bin information lives"** (arc 3). Reliability gets *one*
  slide, not the lead, but framed as the meeting's central validation question (§1.5).
- Journey/pitfalls: **one dedicated pitfalls slide + woven "what we ruled out" subtext.**
- M1 framing **[UPDATED 2026-06-15, matched-NDE result]:** **the analytical ℓ1+product *almost
  reaches* the optimal CNN.** Given L1+product the CNN's *own* density estimator (VMIM→10-D→sbi_lens
  RealNVP) it hits FoM3 ~3045 (population, n=9000) vs CNN ~3293, **within ~7%**, calibrated-with-caveat,
  σ marginals near-identical. ⇒ **the apparent CNN edge was the density estimator, not the physics;
  ℓ1+product is near-sufficient.** *(Supersedes the earlier "+15% CNN modestly beats" framing, which
  compared CNN-RealNVP vs L1-MAF, i.e. an unmatched NDE. Quote ~7%, the population number, NOT the
  noiseless-obs ~15%; per-parameter the gap is smaller still.)* The escalation still holds: standard
  space ≈ near-sufficient → **BNT space the channel-mixing CNN wins decisively** (L1 collapses 0.15×,
  CNN lossless 0.97×) → but a whitening rotation recovers L1 (frame artifact). Perfect answer to the
  round-table's interpretable-vs-optimal / physics-vs-learned questions.
- **No embargo concerns** (Andreas confirmed). The **BNT→baryon-robustness punchline stays
  forward-looking** (not a finished measurement).
- *Open with Andreas:* "we'll work on the plots", the figures here are curated/working versions;
  styling iteration comes later.

**Confirmed design decisions (Andreas, 2026-06-15) that supersede the slide budget above:**
- **Budget: ~25 min of talking** (assume Q&A is separate). Target ~17 of "my" slides plus Andreas's
  own BNT-intuition block (below). Act 0 expands to 3 (the review-style trust-agenda framing); Act 1 is
  trimmed hard (8 → 5 slides); Act 2 merges the foldable beats.
- **Review-style intro (Andreas, 2026-06-15):** open by situating HOS in the field, lots of effort to
  optimize statistics, but a two-point community that does not yet trust the contours (a "2pt skeptic"
  cartoon as the laugh line). Then a checklist of what it would take to make HOS flagship-grade
  (blinding, covariance, emulators, systematics, analytical cross-checks, non-Gaussian likelihood,
  method limits, null/validation tests, simplicity), and which of these the talk takes on. This is the
  bigger-picture framing that makes the talk a contribution to the trust agenda, not a one-off method.
- **Cost-benefit verdict slide (Andreas, 2026-06-15), closes the CNN-vs-ℓ1 thread (S15):** an open
  question for the round table, is a ~7% FoM gain worth going neural? It cost extensive architecture
  search, a very large dataset (899 cosmologies, ~324k maps; VMIM needs the scale or it biases), and a
  string of *unphysical-information* traps (patch geometry, map-mean / mass-sheet mode, 20° per-patch
  projection features) that tighten contours dishonestly **and escape TARP/SBC**. ℓ1 is simple,
  interpretable, inspectable; CNNs are powerful but treacherous. Counterweight: BNT is where the CNN
  genuinely earns its keep. This *replaces* the old standalone pitfalls catalog (which moves to backup),
  bookending the trust-agenda intro.
- **Structure consolidation (Andreas, 2026-06-15), now the FULL content version (19 slides + block):**
  the optimal-*tomographic*-strategy thesis is the talk's spine (S3 dedicated, §0/§1 reframed); Act 1
  adds wavelet peak-count (S5) and ℓ1-norm (S6) explainer slides (the peaks → ℓ1 build) and opens with
  the LAM "no longer statistics-limited, now systematics-limited" hook (S4); the two Part-1 BNT slides
  merge into one bridge (S9: kernels + before/after maps + inflation); Act 2 adds a neural-compression
  explainer (S11: MSE → VMIM). BNT stays a single Part-1 bridge (the tomography spine makes it one
  thread, not a cross-part mix). 19 slides ≈ 30 min; the trim path to ~16 (25 min) is in the §2 intro.
  Part-1 presentation is upgraded from the LAM_2026 deck (reuse its assets and the systematics framing).
- **Andreas has built an intuition sequence** (~4-5 min, his own slides) showing *intuitively why BNT
  destroys the ℓ1 contours but is lossless for the CNN.* It is a dedicated block that sits **between the
  M3 quantitative result (S13) and the whitening clincher (S14)**: result → intuition (his slides) →
  quantitative confirmation. My S14 must *confirm* his intuition, not re-explain it.
- **New figures, main-line:** the **pipeline schematic** (`p2_pipeline_schematic`, now built and
  transparent) goes on the methods slide (S9); the **BNT before/after maps** (`p1_maps_before_noisy` /
  `p1_maps_after_bnt_noisy`) go on the BNT-setup slide (S7).
- **Backup, not main-line:** the **summary-embedding** (`p2_M1_summary_embedding`) and **CNN saliency**
  (`p2_saliency_cnn`) showpieces. Saliency rests on a modest r=0.30 correlation; in this
  interpretability-wary crowd keep it in the back pocket for "what does the CNN actually look at?"
- **Style:** follow the no-em-dash rule (commas/colons/parens; arrows → fine) in all slide text and
  prose; "projection" not "shadow"; "cross-bin / common signal", avoid colliding terms.

**Division of labour in this doc:** Part 1 (baryonic feedback) is outlined at the level needed to set
up Part 2 and the bridge, its figures already exist in the paper. Part 2 (this repo) is the detailed
deliverable. New plotting effort went into Part 2 + the one new M2 figure.

---

## 0. The one-sentence spine

> This talk is about the optimal *tomographic* weak-lensing strategy, the under-explored half of the
> optimal-summary question (not just which statistic, but how to use the redshift bins). Weak lensing's
> non-Gaussian information is real and valuable even on baryon-safe scales (the ℓ1-norm
> beats the power spectrum 3× there), but the natural baryon-mitigation move, the BNT transform,
> *breaks* analytical higher-order statistics; a calibrated, channel-mixing **learned** compressor
> both matches the analytical ℓ1-norm in the standard basis and **rescues the information BNT appears
> to destroy**, because seeing the tomographic bins as channels is exactly what per-bin statistics
> cannot do.

The field has optimized the *statistics* (peaks, ℓ1, scattering, learned summaries) but under-explored
the *tomography*, so this talk's angle is the **optimal tomographic strategy**. The unifying mechanism
that ties both halves together: **inter-bin (cross-bin) tomographic information, and whether a given
summary can access it.** A per-channel statistic (ℓ1 on each bin)
cannot model the cross-bin structure that BNT makes essential; a CNN that takes the bins as input
channels can. That single idea explains (i) why ℓ1 inflates under BNT and the CNN does not, (ii) why
the auto-only CNN already matches L1+product, and (iii) why explicit cross-maps add little for the CNN.

---

## 1. The full narrative arc (prose)

**Act 0: why go beyond the power spectrum.** Stage IV surveys (Euclid, Rubin/LSST, Roman) push weak
lensing into the non-linear regime, where most of the signal lives and where the field is markedly
non-Gaussian. The power spectrum captures everything only for a Gaussian field, so two-point analyses
are fundamentally incomplete on small scales. This is the meeting's premise; I'll make it concrete and
lay out the two questions the talk answers: *(1) how much non-Gaussian information can we actually use,
given that small scales are also where baryons and theory uncertainty bite? (2) can a learned summary
do better than our best analytical non-Gaussian statistic, and can we trust either's contours?*

But there is a prior question that frames the whole talk. Everyone at this meeting is optimizing
statistics, going to higher order, squeezing out more information. The two-point community, meanwhile,
stays skeptical: the flagship Stage-III and Stage-IV analyses are still run with standard two-point
statistics, and our higher-order results, however striking, are treated as proofs of concept. So what
would it actually take to make higher-order statistics trustworthy enough to lead a flagship analysis?
Not one thing, but a checklist the community keeps raising: blinding, robust covariance, emulators,
systematics (how is each statistic biased, and at what level?), analytical cross-checks, dropping the
Gaussian-likelihood assumption, knowing the limits of each method, null and validation tests, and
simplicity. This talk does not solve all of them, but it takes several head-on (systematics, the limits
of each method, a concrete validation standard, and a simple analytical statistic measured against the
optimal learned one), and the simulation-based-inference framework removes the covariance and
Gaussian-likelihood worries for free. That is the bigger picture the two halves below fit into.

**Act 1: baryonic feedback and higher-order statistics (Paper I).** Here is the tension that defines
the small-scale regime: the scales with the most constraining power are exactly where baryonic feedback
(AGN, supernovae) redistributes matter, and where feedback models disagree most. Using CosmoGridV1
with a baryon-correction model and SBI (NPE + MAF), I show that **unmodeled baryons bias cosmology more
the larger the survey**, manageable for Stage III, but >2σ at Stage-IV area and catastrophic (>3σ) in
the full-sky limit. The standard fixes are unappealing: model the feedback (but the models disagree) or
cut scales (but the power spectrum needs aggressive cuts that throw away most of the signal).

The good news is the talk's first real result: **higher-order statistics are far more resilient.** The
starlet ℓ1-norm is cleaned of baryonic bias by removing a *single* finest wavelet scale, and (the
headline of Paper I) **even restricted to those baryon-safe, quasi-linear scales the ℓ1-norm gives
constraints ~3× tighter than the power spectrum.** Non-Gaussianity is not just a deep-non-linear
phenomenon; there is usable non-Gaussian information at intermediate scales the feedback barely
touches. So HOS are a robust path forward: *prioritize HOS on conservative scales.*

Then the cautionary half of Act 1, which is also the bridge. The BNT (Bernardeau–Nishimichi–Taruya)
transform is an elegant third option: a linear, invertible mixing of the tomographic bins that nulls
the lensing efficiency for low-redshift lenses, localizing the baryonic sensitivity to one transformed
bin so you can cut scales only there. For the power spectrum it works (it's effectively lossless if you
keep the full cross-covariance). **But applied to map-based HOS it backfires:** the same linear mixing
that nulls the signal correlates the originally-independent shape noise across bins, the noise floor
rises, and the HOS contours *inflate dramatically.* And here is the puzzle I leave Act 1 on: BNT is
invertible, so no information can truly be lost, yet recovering the lost signal-to-noise for HOS is
"highly non-trivial," and even a recent Euclid Stage-IV forecast that explicitly modeled cross-bin HOS
still saw inflated BNT contours. *Is the BNT information really lost for higher-order statistics, or are
we just analyzing it wrong?*

**Act 2: learned vs analytical, and can we trust it (Paper II / this work).** Paper I establishes the
ℓ1-norm as a strong analytical non-Gaussian statistic. Act 2 asks the two questions the conference
cares about. *First: is it optimal?* In principle a CNN trained to maximize mutual information with the
parameters (VMIM) is the "optimal learned compressor." Does it beat the hand-designed ℓ1-norm? I make
this an apples-to-apples SBI comparison (same simulations, both calibrated) on flat-sky 10° patches
(so the cross-map construction is physically defensible, unlike a full-sphere build). Crucially, the
dataset is ample (324k patches, 899 cosmologies), so any gap is the *compressor/estimator*, not data
scarcity. And the key control is the **density estimator**: when each summary is read out with the
*same* flow the CNN uses (VMIM-compress to 10-D, then sbi_lens RealNVP), **the analytical ℓ1+product
almost reaches the optimal CNN: FoM3 ~3045 vs ~3293, within ~7% (population median), calibrated, with
the marginal errors near-identical (σ(w0) exact).** The mechanism is clean: the NDE is the lever, and the
same summary jumps MAF→RealNVP by ~30% for *both* L1 and the CNN. So **the apparent CNN advantage was
the density estimator, not the physics**: the hand-built non-Gaussian statistic is *near-sufficient*.
That is the result for this audience. (Honest caveat: the matched-NDE L1 is calibrated-with-caveat; the
fully-clean L1 number is 2875; and on the single noiseless observation the CNN keeps ~10–15%, but the
robust population statement is the ~7% near-match.)

A one-line methods aside (the referee-defense beat): getting the CNN there required reading its
low-dimensional summary out with an expressive flow (RealNVP, +36%) and a modestly better architecture
(resnet18, +6%); the deeper net over-fits at 899 cosmologies. We optimized the learned side properly,
so "you just trained it wrong" doesn't hold.

*Second: can we trust the contours?* This is the one reliability slide. Tight contours are worthless if
they're wrong, so every headline arm passes a calibration battery: varied-θ TARP-DRP coverage and
simulation-based calibration. The best CNN sits on (or just inside) the diagonal: its tightness is
real, if anything mildly conservative.

Then *where the inter-bin information lives*, the part that resolves Act 1. **Cross-maps (M2):** if you
want to add physical cross-bin information to the ℓ1-norm, the pointwise product κ_i·κ_j (whose mean is the
cross-correlation ξ_ij) buys ~+20%, while a convolution buys nothing, and a warning, the full-sphere
cross construction that we and others have used inflates the apparent gain ~4× because every cross-patch
secretly encodes the whole sky. **BNT (M3), the payoff:** rerun the BNT experiment with the learned
compressor. The per-bin ℓ1-norm collapses exactly as in Paper I (FoM3 × 0.15; σ8 width doubles), but
**the CNN is lossless** (× 0.96, matched NDE, even given the CNN's own density estimator the per-channel
L1 still collapses, so the BNT gap is the representation, not the estimator). And the mechanism: a single fixed *whitening* rotation of the nulled
maps recovers the full no-BNT FoM3 for the ℓ1-norm too (× 1.06). So the information BNT appeared to
destroy was never lost: the collapse is a per-channel *frame* artifact, and either a channel-mixing
compressor or one linear re-rotation recovers it. **This is the decisive CNN win and the answer to
Act 1's puzzle:** the learned compressor doesn't just slightly beat ℓ1 in the standard basis; in the
BNT basis (the basis you actually want for baryon mitigation) it wins outright, because it does the
one thing a per-bin statistic can't: mix the bins.

**Act 3: synthesis.** The whole talk is one escalating story about non-Gaussian, cross-bin
information: power spectrum → ℓ1-norm (much more, even on safe scales) → learned compressor (a bit more
still, and provably calibrated). And BNT, the natural baryon-mitigation tool that Paper I showed breaks
analytical HOS, becomes viable again once the summary can mix bins. The forward-looking punchline (flag
as such): this points to **baryon-robust, non-Gaussian SBI** that keeps BNT's clean per-bin scale cuts
without paying the contour-inflation tax. And a mini-conclusion before the round table: the CNN's win
over the ℓ1-norm is real but small (~7%), and it was expensive and treacherous to earn (architecture
search, a very large dataset, and a string of unphysical-information traps that escape TARP and SBC), so
whether that 7% is worth the loss of interpretability is a genuine open question. BNT is the clearest
case where the learned compressor earns its keep. This comparison is trustworthy precisely because we
ruled out that catalog of real traps along the way (the pitfalls double as a gift to the community).

---

## 1.5 Tuning to this audience (who's in the room, and how to land it)

This is a small, expert, **Starck-lineage** meeting (wavelets/sparsity, HOS for weak lensing, SBI/NDE
methodology, one-point statistics, and a large 21cm/EoR contingent), with a **strong reliability /
"can we trust beyond-2pt?" undercurrent** that surfaces explicitly in the Wednesday round table. The
talk should read as *central to the meeting's core debate*, not a side contribution. Concrete moves,
by person/session, weave these as one-line nods, not separate slides:

**The Wednesday round table is the single biggest opportunity.** Chaired by Natalia Porqueres
(Heavens, Uhlemann, Camera, Cuesta-Lazaro), its posted questions are almost a description of this talk:
*"How much of the non-Gaussian gains are robust to systematics? Model all systematics or rely on
conservative scale cuts? What is the minimum validation standard for a new statistic (coverage tests)?
Do we want optimal summaries, interpretable summaries, or both? If NN statistics aren't interpretable
but pass every validation test, should we care? Ten years from now, physics-based statistics or
learned summaries?"* **Tee these up explicitly** (Andreas speaks Tue, before the Wed round table, he
can seed it): Part 1 answers *model-vs-scale-cuts* (scale cuts are robust and cheap); Part 2's **M1 is
a concrete data point on "interpretable vs optimal" and "physics-based vs learned"** (the interpretable,
hand-built ℓ1-norm *essentially matches* the learned optimum, within ~7%, once both get the same
density estimator, both calibrated); and the **TARP-DRP + SBC + L-C2ST battery is a concrete proposal for the "minimum validation
standard"** the panel is asking about. Closing line option: *"these are exactly Wednesday's round-table
questions, here's one data point."*

**Simone Vinciguerra** (Wed, "Explicit vs. Implicit Likelihood Inference… CNN-based map-level
estimators… posterior calibration"): he is in the room, and **his Euclid forecast (Vinciguerra et al.
2026) is the paper Part 1 cites and Part 2 resolves**: it found BNT inflation persists for HOS *even
with explicitly modeled cross-components*, and that recovering the SNR is "highly non-trivial." On the
bridge/M3 slides, **name it generously**: *"a recent Euclid forecast showed BNT inflation survives even
with explicit cross-bin HOS. I'll show the information isn't actually lost; a channel-mixing compressor,
or a single fixed rotation, recovers it."* His explicit-vs-implicit and calibration themes also overlap
our analytical-vs-learned + TARP/SBC framing, a friendly forward-reference to his talk.

**Alan Heavens** (opening keynote + round table; "hybrid summary statistics… extreme data compression
of physics-based and neural-network summaries in a Bayesian SBI framework," DES Y3 beating 3×2pt):
this is the meeting's framing keynote and it's *exactly* our paradigm. Frame M1 as a contribution to
the analytical-vs-learned-compression question Heavens opens with: ℓ1-norm (analytical) vs CNN-VMIM
(learned) compressors, both read out by the same flow NDE.

**Cora Uhlemann** (Tue + round table; "one-point statistics compress higher-order correlations… WL,
clustering, and their joint distribution"): **frame the starlet ℓ1-norm as a (multiscale) one-point
statistic** (it is the PDF of wavelet coefficients), so it sits squarely in her framing. (Back-pocket:
our parked M5 joint-one-point work connects directly if she or anyone asks.)

**Giovanni Aricò** (Wed; "Higher-order baryonic modelling for Euclid," baryonification + bispectrum
emulator): he is a **baryonification author: we *use* the shell-baryonification BCM (Schneider/Aricò)
in Part 1.** Position modeling (Aricò) and conservative scale cuts (us) as **complementary** mitigation
strategies, not rivals. Acknowledge his is the path to *using* the small scales we choose to cut.

**Aurelio Amerio / GenSBI** (Thu; JAX-native flow-matching/diffusion NDE backends with SBC/TARP/LC2ST):
same JAX/SBI ecosystem as our jaxili pipeline. Our **+36% NDE-swap finding (MAF→RealNVP)** is a clean
argument that *the density-estimator choice matters materially*, supports the value of flexible NDE
libraries like his.

**David Gebauer / SBi3PCF** (Tue; SBI on the integrated 3PCF, **same CosmoGridV1, same MAF, same
Ωm-σ8-w0 FoM**, +63.8% over 2pt): a direct methodological sibling in the *same currency*. Keep our FoM
framing consistent with his; we can note results are comparable-apples (same sims, same flow family).

**Maria Marinichenko** (Wed; scattering transform vs 2pt, FLAMINGO, baryons+shape noise) and **Bhuvnesh
Jain** (Wed; 3pt + wavelet stats for systematics, diffusion mass maps): scattering transforms and
wavelet statistics are siblings of the ℓ1-norm in the analytical-HOS family. Light nod that the
analytical-vs-learned question generalizes across these. (Back-pocket: our parked 2D-1D Haar/modulus
excursion found the scattering-style *modulus* doesn't help the ℓ1, relevant if scattering comes up.)

**Nicolas Martinet** (Tue; Euclid HOS review): our work fits the Euclid HOS program (Andreas is
HOWLS-connected), a natural place to position the contribution.

**Systematics humility (preempt the obvious critique).** Several talks center systematics we do *not*
model: **Seung-gyu Hwang** (source clustering measurably affects the wavelet ℓ1-norm), **Casper Vedder**
(3-point intrinsic alignments), photo-z, shear calibration (Gebauer models all of these). Be explicit:
Part 2 is a **controlled methods comparison** (specific systematics held fixed); real-world robustness,
the round-table's worry, needs exactly the IA/source-clustering/photo-z work these speakers are doing.
This honesty plays *well* with this crowd and aligns the talk with the meeting's central concern.

**The lessons generalize (closing nod to the 21cm half of the meeting).** The starlet ℓ1-norm and
one-point PDF are a through-line across the whole program, WL *and* HI/21cm (Gorbatchev's "Starlet ℓ1
for HI," Vos's "HI one-point statistics," the EoR bispectrum talks). Our **basis-dependence insight (M3
whitening: a one-point statistic's information depends on the frame; one rotation undoes the BNT
collapse)** is a general lesson for anyone applying these statistics after a linear transform, a clean,
memorable note to land in the synthesis for the broader room.

**One-line register check.** This crowd *loves* interpretable, physically-motivated statistics and is
wary of "DL beats your statistic" triumphalism. Lead the learned-vs-analytical result with the
pro-interpretability reading (analytical ≈ optimal), then let the BNT twist (CNN wins decisively *in a
frame where per-channel stats structurally fail*) be the interesting, earned surprise.

---

## 2. Slide-by-slide content

This is the **full content version: 19 of my slides + Andreas's ~4-5 min BNT-intuition block** (after
S15), which is ~30 min of material. **To hit 25 min, trim in this order:** (a) merge S2 (trust) and S3
(tomography) into one "two gaps" framing slide; (b) keep S5/S6 (peaks, ℓ1) fast, this crowd knows them;
(c) fold the *(fold if tight)* beats below (the P(k) panel on S4, the M1 corner and referee aside on
S12). That lands ~16 slides. Each slide: **what it says** (the spoken point) and **figure** (file in
`talk_figures/`). Speaker-note numbers are in §4.

### ACT 0: framing + the trust agenda (3)

**S1: Title.**
- *Says:* **"Do Baryons Break Higher-Order Statistics?"** plus name/affiliations. One-line hook that
  previews the full arc: *"...and when our cleanest fix seems to break them, is the information really
  lost?"* (Part 1 = the title's question, Part 2 = the deeper resolution.)
- *Figure:* a κ map or the flat-sky inputs panel as a backdrop (`p2_methods_flatsky_inputs`), optional.

**S2: Everyone is optimizing the statistics, but the two-point camp does not trust the contours.**
- *Says:* This whole meeting is one shared effort: extract more from the lensing field than the power
  spectrum can reach (peaks, the starlet ℓ1-norm, scattering, the one-point PDF, the bispectrum, learned
  summaries). We are all optimizing statistics, going to higher order. But here is the uncomfortable
  truth: the two-point community is not convinced. HOS are still not mainstream; the flagship Stage-III
  and Stage-IV analyses run on standard two-point statistics, and our results, however pretty, are still
  proofs of concept. So what would it take to make HOS trustworthy enough to lead a flagship analysis? A
  non-exhaustive checklist the community keeps raising: blinding; robust covariance; emulators;
  **systematics** (the big one, how is each statistic biased and at what level?); analytical
  cross-checks; dropping the Gaussian-likelihood assumption; knowing the **limits** of each method;
  **null / validation tests**; and **simplicity**. This talk takes several head-on, and SBI removes a
  couple for free (no covariance to invert, no Gaussian-likelihood assumption).
- *The laugh line:* a "two-point person" cartoon, *"I don't believe any of your contours."* That skeptic
  is the antagonist the whole talk is built to answer.
- *Audience hook (big):* this checklist *is* the Wednesday round table (minimum validation standard,
  robustness to systematics, interpretable vs optimal); it frames the talk as a contribution to that
  agenda, not a one-off method. (Echoes Heavens's opening on analytical-vs-NN compression.)
- *Figure:* the 2pt-skeptic cartoon + the checklist with the addressed items highlighted (systematics,
  limits, validation/null, simplicity, plus the SBI-inherent covariance and likelihood). Build in deck.

**S3: Optimal statistics are well studied; optimal *tomography* is not.**
- *Says:* Within that agenda, here is the specific under-explored axis this talk attacks. The community
  has worked hard on *which statistic* to use (peaks, ℓ1, scattering, learned summaries), but far less on
  *how to use the tomographic bins*: the inter-bin information and the choice of basis. So the question is
  not just the optimal statistic, it is the **optimal tomographic strategy**. The structure it exploits:
  the lensing kernels are broad and overlap across redshift, so the bins share information (that overlap
  *is* the cross-bin signal), and BNT is just a transform that re-mixes those kernels. Hold this picture;
  it returns for the cross-maps and for BNT.
- *Audience hook:* this is the talk's novelty handle, and it unifies the later results (cross-maps, BNT,
  whitening) into one question: how should a summary use the bins?
- *Figure:* the tomography viz (the n(z) bins and the broad, overlapping lensing kernels / projection;
  build in deck). Callback at the BNT bridge (S9), where the same kernels are re-mixed and nulled.

### ACT 1: baryonic feedback and HOS (Paper I) (6)

**S4: Stage IV is no longer statistics-limited, it is systematics-limited.**
- *Says:* The hook (reused from the LAM deck, it lands): Stage IV surveys are no longer limited by
  statistics but by **systematics**, and the sharpest one on small scales is baryonic feedback (AGN,
  supernovae), which redistributes matter exactly where the constraining power lives and where the
  feedback models disagree most. Two mitigation options, both unappealing: model the feedback (models
  diverge) or cut scales (lose signal). The toolkit: CosmoGridV1 N-body, **4 tomographic bins**, paired
  DMO and "baryonified" maps (baryon-correction model), Euclid-like shape noise, and SBI (treat the
  pipeline as a simulator, no likelihood; NPE with a MAF).
- *Figure:* tomographic maps `p1_methods_tomo_maps.pdf` (+ optional `p1_setup_nz_bins.pdf`); reuse the
  **Illustris feedback video** from the LAM deck for "what baryons do." The P(k) suppression curve
  (`p1_baryon_impact_ps.pdf`) is a *(fold if tight)* panel/backup.

**S5: Wavelet peak counts.**
- *Says:* The first higher-order statistic: run the starlet wavelet transform on the κ maps, then count
  the **local maxima as a function of SNR in each scale**. Simple and well-established, but it only uses
  the high-SNR peaks.
- *Figure:* a starlet-decomposition + peak-count illustration (the LAM "Summary Statistics" slide has the
  assets). Keep it fast, this crowd knows peaks.

**S6: The wavelet ℓ1-norm uses the *whole* distribution.**
- *Says:* The ℓ1-norm generalizes peaks: per scale, ℓ1 is the **sum of absolute starlet coefficients in
  SNR bins**, so it uses *every* pixel and captures **voids as well as peaks**, the full convergence PDF,
  with no discrete-feature definition. That "peaks plus voids plus everything in between" is exactly why
  it beats peaks later. This is our analytical hero.
- *Figure:* the ℓ1 definition / histograms (build in deck). `p2_methods_l1_vs_cosmology` (the ℓ1 shifts
  systematically with cosmology) is a strong *(fold if tight)* anchor that ℓ1 is information-rich.

**S7: Baryonic bias scales with survey area.** ★
- *Says:* Quantify the *parameter* bias from unmodeled baryons with a difference-of-means tension Q_DM
  (in σ; robust threshold <0.3σ), as a function of footprint, since more area means smaller errors and
  higher sensitivity to bias. It grows: fine for Stage III, ~2σ at Stage-IV area (~14,000 deg²), and >3σ
  full-sky for *all* statistics, already at ℓ_max = 1024 (HOS are if anything more sensitive than P(k),
  they live on the contaminated small scales). This is the systematic that forces the issue.
- *Figure:* `p1_bias_vs_survey_area.pdf` (n_σ vs survey area, all statistics); the bias triangle plot is
  backup.

**S8: HOS are resilient *and* constraining (Paper I headline).** ★
- *Says:* The fix that works. The power spectrum needs an aggressive cut (ℓ ≤ 400 full-sky) for unbiased
  inference, throwing away most of the signal, because its baryon contamination is spread across scales.
  For the ℓ1-norm the contamination is **isolated in the single finest wavelet scale**, so removing just
  that band suffices across all survey areas. And on those baryon-safe, quasi-linear scales the ℓ1-norm
  still constrains ~**3× tighter** than P(k). So HOS are not only deep-non-linear probes; there is robust
  non-Gaussian information at intermediate scales feedback barely touches. *Prioritize HOS on
  conservative scales: robust and constraining, no baryon modeling needed.*
- *Figure:* `p1_PSvsHOS_safe_scales.png` (PS vs peaks vs ℓ1 on safe scales, ℓ1 tightest). The per-scale
  contamination plot (`p1_baryon_impact_l1.pdf`, the bias sits in the finest band) supports the "one cut"
  point; `p1_l1_constraints_vs_area.pdf` is backup.
- *Audience hook:* answers the round-table's *model-vs-scale-cuts* question (cuts are robust and cheap);
  positions modeling (Aricò) and conservative cuts as complementary.

**S9: The natural next step is BNT, but it breaks HOS (the bridge).** ★
- *Says:* The natural way to exploit even smaller scales is the BNT transform: a linear, invertible
  re-mixing of the tomographic bins that nulls the low-z lensing efficiency, so baryon sensitivity
  localizes to one transformed bin and you can cut scales only there (recall the overlapping kernels from
  S3, BNT re-mixes them). In Paper I we tried it, and for map-based HOS it **backfires**: the same mixing
  correlates the originally-independent shape noise, the noise floor rises, and the HOS contours
  **inflate dramatically** (you can see it in the maps). The puzzle, and the hinge of the whole talk: BNT
  is *invertible*, so no information can truly be lost, yet recovering the lost SNR for HOS is "highly
  non-trivial," and even a Euclid Stage-IV forecast with explicit cross-bin HOS still saw inflated BNT
  contours. *Is the information really lost, or are we just analyzing it wrong?* That carries us into
  Part 2.
- *Audience hook:* **name Vinciguerra et al. (2026), Simone is in the room** (Wed, explicit-vs-implicit
  likelihood): "a recent Euclid forecast found this persists even with explicit cross-components, and
  recovering the SNR is highly non-trivial; I'll come back to that." A direct, generous setup for the
  Part-2 payoff.
- *Figure:* the kernels `p1_bnt_kernels.pdf` (broad → nulled, the S3 callback) + the before/after maps
  `p1_maps_before_noisy.pdf` → `p1_maps_after_bnt_noisy.pdf` (the SNR blow-up) + `p1_BRIDGE_bnt_inflates_l1.pdf`
  (the inflated ℓ1 contours: gray = BNT, red = safe-scale ℓ1, blue = all-scale). One slide carries
  kernels + maps + inflation; if too dense, split the maps off as a *(fold if tight)* beat. **The hinge.**

### ACT 2: learned vs analytical, and reliability (Paper II / this repo)

**S10: The comparison, done fairly.**
- *Says:* Two summaries of the same tomographic κ maps go through the *same* flow density estimator to a
  posterior: (a) the analytical starlet ℓ1-norm; (b) a CNN compressor trained with VMIM (variational
  mutual-information maximization), the "optimal learned compressor." Flat-sky **10° patches** so any
  cross-maps are physically buildable, both arms calibrated, and it is not undertraining: **324k patches
  / 899 cosmologies**, ample for the compressor. *(fold if tight: the ℓ1 histograms shift systematically
  with cosmology, `p2_methods_l1_vs_cosmology`, anchoring that ℓ1 is a real information-rich statistic,
  not a toy baseline.)*
- *Figure:* **`p2_pipeline_schematic`** (the built schematic: κ maps → {ℓ1-norm | CNN-VMIM} → flow NDE →
  posterior, calibration-gated). Detail/backup: `p2_methods_flatsky_inputs` (the auto + cross channels
  the compressors see).

**S11: How a CNN learns a summary: from MSE regression to VMIM.**
- *Says:* What is inside the "learned compressor" box. The simplest neural summary is a **regression /
  MSE compressor**: train a network to predict the parameters from the map and use its output (or
  penultimate layer) as the summary; easy, but it only targets the posterior mean. **VMIM** (variational
  mutual-information maximization) goes further, learning the summary that **maximizes the mutual
  information between the summary and the parameters**, capturing the full posterior shape with no
  Gaussian assumption. VMIM is the stronger, "optimal" compressor we use, and (foreshadow S17) the one
  that needs a large dataset or it biases.
- *Figure:* a small MSE-vs-VMIM schematic (build in deck); the LAM **NDE.gif** is a reusable asset for
  the flow side.

**S12: Headline (M1): the analytical ℓ1-norm almost reaches the optimal CNN.** ★
- *Says:* Give the ℓ1+product summary the CNN's *own* density estimator (VMIM compress to 10-D, then
  sbi_lens RealNVP) and it reaches **FoM3 ~3045** vs the CNN's **~3293**, within **~7%** (population
  median), calibrated, with σ marginals near-identical (**σ(w0) exact**). The mechanism is the point:
  the NDE is the lever. The *same* 10-D summary jumps MAF 2426 → RealNVP 3270, exactly mirroring the
  CNN's own MAF 2312 → RealNVP 3293. So the apparent CNN edge was the *density estimator, not the
  physics*: the hand-built ℓ1+product is *near-sufficient*. *(Quote ~7% population, not the noiseless-obs
  ~15%. Honest caveat: matched-NDE L1 is calibrated-with-caveat; the fully-clean L1 number is raw-MAF
  2875.)*
- *Audience hook (the big one):* the **direct answer to Wednesday's round table**, "optimal vs
  interpretable summaries? physics-based vs learned?" Here the *interpretable* hand-built statistic
  essentially *matches* the learned optimum once you control the estimator, both calibrated. The
  strongest pro-HOS / pro-interpretability reading, complementing Heavens's hybrid-compression program.
- *Figure:* **`p2_M1_matched_nde`** (the 3-bar headline, all matched-NDE: L1 auto-only 2448 → L1
  auto+product 3045 → CNN 3293, "only ~7% below the CNN"). Optional second beat *(fold if tight)*:
  **`p2_M1_corner_matched`** (the matched-NDE contours nearly coincide). Rigor backup:
  **`p2_M1_nde_matrix`** (the gated representation×NDE matrix; pair2d's over-confident fool's-gold
  rejected). Referee-defense aside (verbal, or a *(fold if tight)* half-slide): getting the CNN there
  took an expressive flow (RealNVP, +36%) and a better architecture (resnet18, +6%); the deeper net
  overfits at 899 cosmologies. We optimized the learned side properly, so "you trained it wrong" does
  not hold.

**S13: Can we trust it? (the one reliability slide).** ★
- *Says:* Tight is not the same as correct. Every headline arm passes a calibration battery: varied-θ
  **TARP-DRP** coverage (on the diagonal) and **SBC** rank uniformity (flat within the 99.5% band). Both
  the analytical and the learned arm are calibrated; the CNN is, if anything, mildly conservative (net
  coverage bias +0.033, L1 +0.002). Its tightness is real.
- *Audience hook:* a concrete answer to the round-table's *"what is the minimum validation standard a
  new statistic should meet?"* We apply the *same* battery (TARP/SBC, plus L-C2ST where dimension
  allows) to *every* arm, analytical and learned alike. (Same JAX/SBI calibration stack as GenSBI; same
  explicit-vs-implicit calibration concern as Vinciguerra.)
- *Figure:* **`p2_M1_calibration_tarp` + `p2_M1_calibration_sbc`** (matched-NDE overlays, both arms on
  the diagonal). Fallback: the old CNN-only `p2_reliability_tarp_sbc`.

**S14: Where the cross-bin information lives (M2).** ★
- *Says:* Can explicit cross-maps add *physical* cross-bin information to the ℓ1-norm? A little: the
  pointwise product κ_i·κ_j (whose mean is the cross-correlation ξ_ij) buys **+20%**; a convolution buys ~0. And a
  community warning: the full-sphere cross construction (ours and others') inflates the apparent gain ~4×
  because every cross-patch pixel is a global functional of the whole sky, so it is partly *unphysical*
  (~92% of that gain is leakage). The robust, physical cross gain is ~+20%.
- *Figure:* `p2_M2_cross_deleaking` (L1 arms auto/conv/product/both + the leaky full-sphere reference).
  Main-talk alt without the leakage bar: `p2_M2_cross_deleaking_nofullsphere`.

**S15: BNT revisited with a learned compressor (M3, the quantitative payoff).** ★
- *Says:* Rerun Paper I's BNT experiment with both summaries, through the *same* matched NDE (apples to
  apples with M1). Even given the CNN's own NDE, the per-channel ℓ1+product **collapses** (FoM3 3045 →
  779 = ×**0.26**; σ8 width +65%), while the channel-mixing CNN is **lossless** (3326 → 3186 = ×**0.96**).
  And the collapse is *calibrated* (the wide BNT contours honestly report a real loss, not
  over-confidence). The beautiful parallel with M1: in the standard basis the gap was the *estimator*
  (give L1 the CNN's NDE and it catches up); under BNT the gap is *not* the estimator (give L1 the same
  NDE and it *still* collapses), it is the per-channel *representation*. *This is where the learned
  compressor structurally wins.* Then hand off to the intuition block for *why*.
- *Figure:* `p2_M3_bnt_inflation` (matched-NDE FoM3 bars: L1+product 0.26× vs CNN 0.96×). Optional:
  `p2_M3_corner_bnt_4way` (L1-BNT balloons, CNN-BNT stays tight) or the cleaner `p2_M3_corner_l1_collapse`.
  Calibration of the collapse: `p2_M3_calibration_tarp/sbc`.

**[ANDREAS'S BNT-INTUITION BLOCK: ~4-5 min, his own slides, inserted here]**
- *Role:* the conceptual heart, *why* BNT destroys the per-channel ℓ1 but is lossless for the CNN. The
  intuition: BNT trades 4 deep, redundant lensing kernels for 1 shallow map + 3 thin slices; a per-bin
  statistic reads each channel alone and cannot see the cross-channel structure BNT makes essential,
  while a CNN that takes the bins as input *channels* can. S15 gave the number, this block gives the
  intuition, S16 gives the quantitative proof. Keep S16 framed as *confirmation* of this block, not a
  re-explanation. *(Make sure the palette/notation here matches the deck: CNN blue, ℓ1 vermillion;
  no em-dashes; "cross-bin / common signal", not "shadow".)*

**S16: The clincher: a frame artifact, not lost information.** ★
- *Says:* The decisive diagnostic. A single fixed *whitening* rotation Q of the nulled maps recovers the
  full no-BNT FoM3 for the ℓ1-norm too (×**1.06**). So the information BNT "destroys" for the ℓ1 was
  never lost; the collapse is a per-channel *frame* artifact, and either a channel-mixing compressor *or*
  one linear re-rotation recovers it. This quantitatively confirms the intuition you just saw, and it is
  a principled version of the BNT-smoothing/denoising hybrids the field has speculated about.
- *Audience hook:* **close the Vinciguerra loop**, "their forecast said recovering the BNT SNR for HOS is
  highly non-trivial; here it is, in one fixed rotation." And it is a *frames* result, the kind this
  Starck-lineage wavelet/sparsity room will appreciate: a one-point statistic's information content is
  basis-dependent, and BNT is simply a poor frame for a per-channel statistic.
- *Figure:* `p2_M3_bnt_whitening` (noBNT / whitened ≈ full / BNT ≈ collapsed).

**S17: Is the +7% worth it? The honest cost of going neural (open question for the round table).**
- *Says:* A mini-conclusion before tomorrow's round table. The optimized CNN does beat the ℓ1-norm, but
  by only ~7% in FoM3, both calibrated. And earning that 7% *honestly* was expensive. To get contours
  that are tight *and* true I had to search architectures extensively, tune hyperparameters hard, and
  train on a very large dataset (**899 cosmologies, ~324,000 maps**); the stronger VMIM compressor in
  particular needs that scale, or it biases. Worse, the CNN kept finding *unphysical* shortcuts that
  tightened the contours dishonestly: the geometry of how the patches are cut, features in the *means* of
  the maps (the mass-sheet mode), and, with the 20° patches, projection features that vary with position
  on the sphere, all of which the network happily used as constraining power that real data would never
  provide. The dangerous part: these failures largely *escape* TARP and SBC. The contours look perfectly
  calibrated and are still wrong. The ℓ1-norm has none of this: it is simple, interpretable, and you can
  go back and inspect the data vectors. CNNs are powerful but hard and treacherous. So, an open question
  for the panel: is a ~7% FoM gain worth that cost and that risk? The honest counterweight, and where the
  answer leans yes: **BNT**, where the channel-mixing CNN keeps the full information for free while the
  per-channel ℓ1 collapses. There the CNN clearly earns its keep.
- *Audience hook (the panel tee-up):* the explicit hand-off to Wednesday's round table ("if NN
  statistics aren't interpretable but pass every validation test, should we care?"), with a sharp, honest
  twist: *some failure modes pass every validation test and are still wrong.* The pitfalls are also a
  gift to the community (full 6-trap catalog in the backup pile).
- *Figure:* a cost/benefit balance: on one pan the M1 ~7% bar, on the other the cost (architecture
  search, huge dataset, unphysical-information traps that escape TARP and SBC), with BNT as the thumb on
  the scale. Build in deck. *(Dataset line to confirm on slide: 899 cosmologies, ~324k maps; add exact
  realizations-per-cosmology and batch size if wanted.)*

### ACT 3: synthesis (2)

**S18: One story about the optimal tomographic strategy.**
- *Says:* Pull it together. Power spectrum → ℓ1-norm (much more information, even on baryon-safe scales)
  → learned compressor (a bit more still, and provably calibrated). The thread is cross-bin information:
  a per-bin statistic cannot access it (so it breaks under BNT); a channel-mixing learned compressor can
  (so it is BNT-lossless and needs no explicit cross-maps). **BNT, the baryon-mitigation tool Paper I
  showed breaks analytical HOS, becomes viable once the summary mixes bins.** Forward punchline (*flag as
  forward-looking, see §4*): a route to **baryon-robust, non-Gaussian SBI** that keeps BNT's clean
  per-bin scale cuts without the contour-inflation tax.
- *Figure:* the bridge pair side by side, `p1_BRIDGE_bnt_inflates_l1.pdf` (ℓ1 inflates) next to
  `p2_M3_bnt_whitening` (recovered): the visual "problem → resolution."

**S19: Takeaways and what's next.**
- *Says:* Back to the title, **do baryons break HOS? No.** (i) Usable non-Gaussian information persists
  on baryon-safe scales, prioritize HOS, cleaned by a single scale cut; (ii) a well-built *interpretable*
  analytical statistic (ℓ1) essentially matches the learned optimum (within ~7%; the gap is the density
  estimator, not the physics; ℓ1+product near-sufficient), and both are calibrated; (iii) even BNT's
  apparent HOS "information loss" is a frame artifact a channel-mixing compressor (or one rotation)
  undoes. Next: the physically-buildable flat-sky cross for the CNN; a clean end-to-end baryon-mitigation
  demo with CNN+BNT; calibration at scale for Stage IV.
- *Audience hooks:* (a) **round-table tee-up (broaden from S17)**, S17 already posed the sharp "is it
  worth it" question; here recap the full set, "these are exactly Wednesday's questions (optimal vs
  interpretable summaries, physics-based vs learned, what validation standard), and the talk gave a
  concrete data point on each." (b) **systematics humility**, this is a controlled methods study; real-world
  robustness needs the IA / source-clustering / photo-z work others here present (Vedder, Hwang,
  Gebauer). (c) **generalization nod to the 21cm half**, the ℓ1-norm and one-point PDF, and the
  basis-dependence lesson, carry across to HI/EoR (Gorbatchev, Vos). *(Acknowledgements: Guerrini,
  Starck, Kilbinger; CosmoGridV1; based on Zeghal's Learn2Map.)*
- *Figure:* none / summary slide.

**(Backup pile):** summary-embedding (`p2_M1_summary_embedding`, the interpretable-vs-optimal showpiece)
and CNN saliency (`p2_saliency_cnn`, "what the CNN looks at", with the r=0.30 caveat) if interpretability
comes up; the lever-decomposition bar (NDE +36% / arch +6% / resnet50_gn −12%); the M1 corner at a
representative patch; the cross-only NDE-architecture confound; the prior-shrinkage and
sharpness-vs-calibration diagnostics; the full pitfalls table.

---

## 3. Figure-per-slide manifest

All paths relative to `talk_figures/`. PDF preferred (vector); PNG present for the Part-2 result
figures. ★ = the figures that carry the talk.

| slide | figure file | source | status |
|---|---|---|---|
| S1 | `p2_methods_flatsky_inputs.{pdf,png}` (backdrop, optional) | flatsky `figs/maps_examples` | keeper |
| S2 | 2pt-skeptic cartoon + trust checklist (addressed items highlighted) | build in deck | new (framing) |
| S3 | tomography viz (n(z) + broad overlapping lensing kernels / projection) | build in deck | new (framing) |
| S4 | `p1_methods_tomo_maps.pdf` (+ `p1_setup_nz_bins.pdf`; Illustris video from LAM; fold-in `p1_baryon_impact_ps.pdf`) | Paper I / LAM | keeper |
| S5 | wavelet peak-count illustration (starlet decomposition + SNR peaks) | build in deck (LAM assets) | new |
| S6 | wavelet ℓ1-norm definition/histograms (fold-in `p2_methods_l1_vs_cosmology`) | build in deck | new |
| S7 ★ | `p1_bias_vs_survey_area.pdf` | Paper I `nsigma_vs_mask_area_all_stats` | keeper |
| S8 ★ | `p1_PSvsHOS_safe_scales.png` (fold-in `p1_baryon_impact_l1.pdf`; backup `p1_l1_constraints_vs_area.pdf`) | Paper I | keeper |
| S9 ★ | `p1_bnt_kernels.pdf` **+** `p1_maps_before_noisy.pdf` → `p1_maps_after_bnt_noisy.pdf` **+** `p1_BRIDGE_bnt_inflates_l1.pdf` | Paper I | keeper (the hinge; split maps if dense) |
| S10 | `p2_pipeline_schematic.{pdf,png}` (detail/backup `p2_methods_flatsky_inputs`) | this repo | LOCKED (built, transparent) |
| S11 | MSE-vs-VMIM compression schematic (+ LAM `NDE.gif`) | build in deck | new |
| S12 ★ | `p2_M1_matched_nde` (3-bar: L1 auto 2448 → L1+product 3045 → CNN 3293, ~7%) + opt `p2_M1_corner_matched`; rigor backup `p2_M1_nde_matrix` | session A | FINAL [matched-NDE] |
| S13 ★ | `p2_M1_calibration_tarp` + `p2_M1_calibration_sbc` (matched-NDE, both arms calibrated; 99.5% SBC band) | session A | FINAL/gated |
| S14 ★ | `p2_M2_cross_deleaking.{pdf,png}` (main-talk alt: `..._nofullsphere`) | this repo | LOCKED |
| S15 ★ | `p2_M3_bnt_inflation.{pdf,png}` (+ opt `p2_M3_corner_bnt_4way` / `p2_M3_corner_l1_collapse`; calib `p2_M3_calibration_tarp/sbc`) | session A | LOCKED [matched-NDE] |
| (block) | Andreas's BNT-intuition slides + animation (his own deck, ~4-5 min) | Andreas | his |
| S16 ★ | `p2_M3_bnt_whitening.{pdf,png}` | this repo (whiten_campaign) | LOCKED |
| S17 | cost/benefit balance (the ~7% bar vs the cost + the traps); → round table | build in deck | new (verdict) |
| S18 ★ | `p1_BRIDGE_bnt_inflates_l1.pdf` + `p2_M3_bnt_whitening.pdf` | both | keeper |
| S19 | (summary) | n/a | n/a |
| backup | `p2_M1_summary_embedding`, `p2_saliency_cnn` (r=0.30 caveat), `p2_M1_stitched` (demoted +15% view), `p2_M1_fom3_distribution`/`p2_M1_violin_fom3`, lever bar, full 6-trap pitfalls table | this repo | backup |

**Generators + per-figure provenance:** `talk_figures/INDEX.md` (the locked palette block, the
Extra/showpiece section, and the `_new_figs/make_*.py` list). The pipeline schematic, generic
posterior, and BNT before/after maps were built/finalized 2026-06-15.

---

## 4. Vetted numbers (with provenance and flags): marginals shown alongside every FoM3

**Parameters:** θ = [Ω_m, σ8, w0, h0, n_s, Ω_b]; lensing-constrained subspace [Ω_m, σ8, w0]. FoM3 =
1/√det(C₃) on that subspace. Part-2 numbers are flat-sky 10° patches, pooled 3-seed 9000-obs median,
all GATE-C (TARP-DRP + SBC) calibrated.

### Part 2: M1 (learned vs analytical), FINAL [matched-NDE resolution, 2026-06-15]
Sources: memory `project_analytical_matches_cnn_via_nde` (the matched-NDE result, supersedes the M1
framing in `project_cnn_nde_swap_resolves_m1`); `…/analytical_nde_match/RESULT_ANALYTICAL_NDE_MATCH.md`
+ `HANDOFF_MORNING.md`; `…/analytical_nde_match/fom3_matrix.png`. Branch `analytical-nde-match-2026-06`.
All cells GATE-C gated (TARP+SBC); population (9000-obs) median FoM3.

**The headline comparison (matched NDE, both summaries through the CNN's own VMIM→sbi_lens RealNVP):**
- **CNN (resnet18 + RealNVP): FoM3 ≈ 3293** (population). *(The arch-sweep also quotes 3326 for the
  resnet18 n=9000/3-seed; use ~3300 consistently. 3293 is the matched-comparison run, used for the gap.)*
- **L1+product, matched NDE (VMIM→10-D→RealNVP): FoM3 3045** (n=9000) / 3270 (n=1000, 3-seed band
  {3146,3265,3399}). **Calibrated-with-caveat** (SBC std ~0.30, net −0.022/−0.011/+0.004, centered,
  mildly over-confident; pooled TARP net +0.001 = near-perfect joint coverage).
- **L1+product, best fully-clean NDE (raw→MAF): FoM3 2875** (clean PASS, the honest clean baseline).
- **Headline gap: ~7%** = (3293−3045)/3293. **Quote ~7% (population), NOT the noiseless-mean-obs ~15%**
  (Andreas). **Per-parameter the gap is smaller still:** σ = 0.047/0.077/0.227 (matched L1) vs CNN
  0.045/0.072/0.229, **σ(w0) exact**, σ(Ω_m,σ8) within ~7%.
- **Mechanism (the point):** the NDE is the lever, isolated: on the *same* 10-D L1+product summary, MAF
  2426 → RealNVP 3270 (+30%), **exactly mirroring the CNN's own MAF 2312 → RealNVP 3293**. RealNVP on
  the *raw* 2000-D L1 craters (1111), so the VMIM compression is what unlocks it.
- **Controls (why it's not fool's gold):** l1-auto (no cross) →VMIM→RealNVP = 2448, calibrated, does
  NOT jump to CNN levels (so the gain is the cross ξ_ij info, not a generic inflator); pair2d →RealNVP
  = 4864 but GATE **FAIL** (over-confident, DPI fool's gold, correctly rejected). The matrix figure
  (`p2_M1_nde_matrix`) shows this gating.
- **Scientific statement:** *"the analytical ℓ1+product almost reaches the optimal CNN (within ~7%,
  calibrated) once given the same density estimator: the gap is the estimator, not the physics ⇒
  ℓ1+product is near-sufficient."*
- **⚠ Honest caveats (must travel):** matched-NDE L1 is PASS-**with-caveat** (mild over-confidence), not
  fully clean; the truly-clean analytical number is raw-MAF 2875. State "matches within calibration
  tolerance," not "gains information." On the noiseless mean-observation the CNN keeps ~10–15% (CNN
  3239 vs L1 2808), a different question; lead with the population ~7%, mention the mean-obs only if asked.
- **Superseded framing:** the earlier "+15% CNN modestly beats" (CNN-RealNVP 3326 vs L1-**MAF** 2875)
  compared *unmatched* NDEs. It is the "each probe its own best *clean* NDE" view, defensible but it
  attributes to the CNN a gap the matched-NDE result shows is the estimator. Figure `p2_M1_stitched`
  (the +15% stitched corner) is **demoted to backup**; the headline is now `p2_M1_matched_nde`.

### Part 2: M2 (designed cross-maps), LOCKED
Source: `FLATSKY_CROSS_RESULT.md`. L1, population median, calibrated.

| arm | FoM3 | vs auto | σ(Ω_m) | σ(σ8) | σ(w0) |
|---|---|---|---|---|---|
| flat-local auto-only | 2405 | 1.00× | 0.053 | 0.082 | 0.245 |
| + conv | 2499 | 1.04× | 0.052 | 0.081 | 0.245 |
| + product (mean = ξ_ij) | 2875 | **1.20×** | 0.048 | 0.075 | 0.238 |
| + both | 2910 | 1.21× | 0.046 | 0.075 | 0.232 |
| full-sphere auto+cross (LEAKY) | 8530 | 3.88× | 0.046 | 0.072 | 0.188 |

~92% of the full-sphere gain is leakage. Physical cross gain ≈ **+20%** (the product). Leakage lived
mostly in w0 (full-sphere σ(w0) 0.246→0.188; physical only →0.232).

### Part 2: M3 (BNT), LOCKED [matched-NDE, 2026-06-15c]
Source: `analytical_nde_match/RESULT_ANALYTICAL_NDE_MATCH.md` Addendum C (both probes through the SAME
VMIM→RealNVP NDE, consistent with the M1 framing); `FLATSKY_BNT_RESULT.md` (whitening + raw-NDE history).

**Matched-NDE BNT (the headline, consistent with M1):**
| arm | no-BNT FoM3 | BNT FoM3 | ratio | calibration |
|---|---|---|---|---|
| **L1+product → RealNVP** | 3045 | **779** | **0.26× (COLLAPSE)** | calibrated (TARP net +0.005, SBC std 0.31), a *real*, honestly-reported loss |
| **CNN resnet18 → RealNVP** | 3326 | **3186** | **0.96× (LOSSLESS)** | calibrated |

- Marginals (matched-NDE L1, BNT): σ(σ8) 0.077 → 0.127 (+65%), σ(w0) 0.229 → 0.296. CNN flat.
- **The parallel to M1 is the point:** standard-space gap = the *estimator* (M1); BNT gap = the
  *representation* (per-channel L1 discards cross-channel info before the NDE; same NDE, still collapses).
- *(Raw-NDE history, for context, L1 auto 0.15× / +product 0.22×, from `FLATSKY_BNT_RESULT.md`; the
  matched-NDE 0.26× is the consistent number to quote. The mixed-NDE figure 0.97× is superseded/archived.)*
- **Whitening (mechanism, M3):** one fixed rotation Q=(BBᵀ)^(−1/2)B recovers the per-channel L1 fully
  (raw-L1 demo: auto 1.06×, +product 1.01×) ⇒ the collapse is a *frame* artifact, not lost info, only a
  channel-mixing (CNN) or whitened frame recovers it. (`p2_M3_bnt_whitening` is the raw-L1 frame demo.)

- **⚠ FLAG (calibration):** in the matched-NDE setup the L1-BNT *collapse* is itself **calibrated**
  (pooled TARP net +0.005, SBC std ~0.31), the wide BNT contours honestly report a real information
  loss, not over-confidence. FoM3 fragility is irrelevant at this effect size (3.9×) and the marginals
  agree. (The earlier common-MAF CNN-BNT+product L-C2ST-40%-reject caveat applied to the superseded
  mixed-NDE arms and no longer governs the headline.)

### Part 1: baryonic feedback, from the submitted paper
Source: `/home/tersenov/papers/Impact_of_Baryonic_Feedback_Submission/main.tex` (Conclusions).

- Baryonic bias grows with survey area: manageable Stage III, **>2σ** at ~14,000 deg² (Stage IV),
  **>3σ** full-sky, at ℓ_max = 1024.
- HOS cleaned of baryonic bias by removing a **single finest wavelet scale**; PS needs aggressive
  area-dependent cuts (ℓ ≤ 400 full-sky).
- **On baryon-safe scales the starlet ℓ1-norm improves the FoM by ≈3× over the PS** (full-sky).
- BNT: ~lossless for the PS (with full cross-covariance); for map-based HOS the linear mixing correlates
  shape noise → **drastic contour inflation**; recovering the SNR is "highly non-trivial even if
  cross-components are modeled" (cites the Euclid Vinciguerra et al. 2026 forecast).

### ⚠ Honesty flags to respect on stage
1. **M1 = matched-NDE ~7% (population), not +15%, not the noiseless-obs ~15%.** Quote the population
   gap (~7%); say "ℓ1 almost matches the optimal CNN, the gap is the estimator." Carry the
   calibrated-with-caveat note. (The old "+15%"/"+25%" were the unmatched-NDE / fiducial-corner numbers.)
2. **The "CNN+BNT → baryon-robust HOS" punchline is FORWARD-LOOKING.** Paper II's M3 demonstrates the
   CNN is **BNT-lossless** and that whitening recovers the ℓ1 information, on (DMO) CosmoGridV1 maps.
   It does **not** re-run Paper I's baryon-bias-mitigation test with the CNN. So "the learned compressor
   makes BNT viable for baryon mitigation" is a *natural implication / next step*, not a demonstrated
   end-to-end result. Present it as the synthesis/forward-looking claim; don't state it as a finished
   measurement.
3. **Don't merge Paper I and Paper II into one FoM ladder.** Paper I = full-sky HEALPix with ℓ_max
   cuts; Paper II = flat-sky 10° patches. The "PS → ℓ1 → CNN" progression is *conceptual*, not a single
   quantitative axis. Keep "ℓ1 ≈ 3× PS" (Paper I) and "ℓ1 ≈ optimal CNN, matched-NDE" (Paper II) as separately-scoped
   statements.
4. **Always show marginals (σ, 2D) with any FoM3**: FoM3 is fragile (pitfall #5).
5. **Embargo: cleared** (Andreas confirmed 2026-06-14, no concerns). Paper I is submitted; Paper II is
   unpublished thesis work, fine to present.
6. **Do NOT present the historical inflated numbers** ("L1 wins 3–4×", the 20° full-sphere campaign,
   the 10° "definitive" full-sphere-cross result) as results; they're superseded/leaky. They appear
   only as the *journey* (the dissolved headline) on the pitfalls slide.

---

## 5. Still to build / open (in the deck, not blocking)
- **DONE (matplotlib, transparent):** pipeline schematic (`p2_pipeline_schematic`, S10), M1 headline
  (`p2_M1_matched_nde`, S12) + `p2_M1_nde_matrix` + the M1 corner (`p2_M1_corner_matched`), BNT
  before/after maps (`p1_maps_before_noisy` → `p1_maps_after_bnt_noisy`, S9), and the backup showpieces
  (`p2_M1_summary_embedding`, `p2_saliency_cnn`).
- **NEW deck assets to build (no data, pure illustration):** the 2pt-skeptic cartoon + trust checklist
  (S2); the **tomography viz** (n(z) + broad overlapping lensing kernels / projection, S3, callback at
  S9); the **wavelet peak-count** illustration (S5); the **ℓ1-norm definition** (S6); the
  **MSE-vs-VMIM** compression schematic (S11); the **cost/benefit balance** (S17). All "build in deck."
- **Reuse from the LAM_2026 deck** (`/home/tersenov/software/talks/LAM_2026/`): the **Illustris feedback
  video** (S4), the **NDE.gif** flow animation (S11), and, if the new deck is dark-themed, the `_dark`
  PDF versions of the Part-1 figures (`../assets/figures/...`), which also settles the projector-font
  pass for Part 1.
- **Andreas's BNT-intuition block + animation** (~4-5 min, his own slides): sits between S15 and S16.
  Match the deck palette/notation (CNN blue, ℓ1 vermillion; no em-dashes; "cross-bin"/"common signal").
- **Lever-decomposition bar** (optional, the S12 referee-defense aside): NDE +36% / arch +6% /
  resnet50_gn −12%. Trivial to make; currently text/backup.
- **Optional restyle** of `p2_M1_nde_matrix` to slide fonts (keep the green/orange/red verdict coloring,
  it encodes calibration, not method).

## 6. Parked (not in the talk unless asked)
M4 (BNT post-cut rescue), M5 (joint one-point statistics), the 2D-1D Haar wavelet ℓ1 excursion, clean
but in-the-weeds; mention only if a methods question invites them. Sources: `PAPER_MESSAGES.md` M4/M5,
`RESULT_2D1D_PHASE{1,2}.md`.
