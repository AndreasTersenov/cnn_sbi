# Paper draft — BNT losslessness: main-text + appendix versions (2026-06-11)

Two registers of the same argument, per Andreas's plan: the cosmologist-oriented version for
the MAIN TEXT, the formal version (post red-team pass) for an APPENDIX. Numbers from
FLATSKY_BNT_RESULT.md; informal precursor BNT_CROSS_INFO_ARGUMENT.md; supersedes
PAPER_BNT_THEORY_SECTION_DRAFT.md (kept for history).

**2026-06-11 (deep-dive pass):** the canonical theory treatment is now
`BNT_THEORY_DEEP_DIVE.md` (same directory) — all claims there are derived, with a claims
ledger (PROVED / MEASURED / MECHANISM). Parts I–II below were REVISED against it: the worked
Gaussian toy (deep-dive F3) shows the "suppressed per-map S/N + correlated noise" account,
taken alone, predicts the OPPOSITE of the measurement — the causal weight now sits on the
Gaussianization-of-marginals mechanism (F5), residual joint response (F3.4), and SNR-grid
flattening (F3.5). Former Parts III–IV are superseded by the deep-dive (L4) and removed.

================================================================================
## PART I — MAIN TEXT (cosmologist-oriented): "Where does the information go under BNT?"
================================================================================

The BNT transform is a fixed, invertible linear recombination of the tomographic maps,
kappa' = B kappa. Invertibility has an immediate consequence worth stating plainly: no
information about cosmology is created or destroyed by applying it. Whatever constraints are
obtainable from the original maps are obtainable, exactly, from the nulled maps. The contour
inflation reported for higher-order statistics on BNT maps — which we reproduce dramatically
here, the wavelet l1 retaining only 15% of its figure of merit — therefore cannot be a
property of the transform. It must be a property of the STATISTIC.

The resolution is that BNT moves information, and some statistics cannot follow it. Because
the transform acts pixel by pixel, everything it rearranges stays at zero lag and equal
scale: information moves between what each map looks like on its own and how the maps relate
jointly — never across scales or positions. A statistic computed map-by-map — the wavelet l1,
peak counts, Minkowski functionals, any statistic that reduces each channel separately before
comparison — keeps only the per-map share, and nulling minimizes exactly that share.
Physically: the original bins are four deep, heavily overlapping lensing kernels — four maps
that largely share their dominant structure. Nulling, by design, cancels the shared part,
trading them for one shallow map (bin 1, untouched — the weakest of the four) plus three thin
lens-redshift slices under amplified, inter-map-correlated noise. Each slice alone is
signal-starved and carries little of the non-Gaussian structure — the features that make a
higher-order statistic worth more than a power spectrum live in the deep common modes that
nulling removes from every map simultaneously. The noise side of the story is, perhaps
surprisingly, NOT sufficient: an honest Gaussian accounting of map-by-map variances predicts
no collapse at all (in the idealized nulling limit it even favors the nulled basis; Appendix
ref). A whitening test localizes the damage exactly: re-analysing the nulled maps after one
fixed orthogonal rotation — the noise-whitened BNT basis, which restores a deep direction,
no learning involved — returns the full no-BNT figure of merit (recovered fractions 1.06 and
1.01 for the two l1 configurations), marginal by marginal. Nothing the statistic lost is
irreducibly joint: the collapse is entirely a property of the nulled frame — the unique
frame, by construction, with no deep direction anywhere. (The rotation re-mixes the nulled
kernels, so this is information accounting, not an analysis recipe; what it licenses is the
separation of the cleaning basis from the statistics basis — and a cheaper fix still,
appending a single deep combination map to the otherwise untouched nulled set.)

A convolutional network fed the tomographic bins as input channels evades this for a simple,
almost mechanical reason: the first operation of a multichannel CNN is a learned linear
combination of its input channels. Undoing the nulling — applying B^-1 — is therefore an
operation the network can absorb into its first layer at no cost, before any nonlinearity
acts. Its constraining power is consequently insensitive to any invertible re-mixing of the
input maps; the inflation test measures whether training actually finds this in practice, and
it does (BNT/no-BNT figure-of-merit ratios 0.93 and 0.88, within the compressor seed-to-seed
scatter of our pipeline). We emphasise what this does and does not show: the network is not
extracting MORE cross-bin information than a well-built summary statistic — on the un-nulled
maps it does not outperform the l1 given an explicit cross-map (Sect. X) — it is BASIS-ROBUST,
where per-channel statistics are basis-fragile.

The explicit cross-maps occupy the middle ground. Supplying the pointwise products
kappa_i kappa_j alongside the autos injects genuine cross-bin information and lifts the nulled
l1 from 0.15 to 0.22 of its no-BNT figure of merit — but only partially, because the products
are a few combinations fixed in advance, blended across scales, and still reduced map-by-map
afterwards. Cross-maps are, in this precise sense, a device FOR per-channel statistics:
pillar 1 of this work shows the network neither needs nor uses them, and pillar 2 shows the
per-channel statistic without them loses most of its power the moment the basis turns hostile.

None of this diminishes BNT. Its purpose — localising the lensing kernels so that small-scale
systematics can be excised bin by bin — is a property of the nulled BASIS, and it is retained.
The lesson is that the cleaning basis and the statistics basis need not be the same: once
nulling has served its purpose, the information that survives the cuts can be extracted in any
basis — by a compressor that learns the rotation, or by a statistic handed the right
combinations explicitly.

================================================================================
## PART II — APPENDIX (formal; post red-team pass 2026-06-11)
================================================================================

LaTeX-ready prose; convert math with the paper-draft pipeline.

## Setup

kappa(p) = (kappa_1(p),...,kappa_N(p))^T tomographic maps; BNT acts pixel- and channel-wise:
kappa'(p) = B kappa(p), B in GL(N,R) fixed, lower-triangular, invertible (kernel-dependent only).
Noise injected BEFORE nulling: pre-BNT white, independent, equal variance sigma^2; post-BNT
Sigma' = sigma^2 B B^T — white within each map, correlated ACROSS bins.

## Exact invariance at the data level

For fixed invertible B: p(theta | Bx) = p(theta | x) (theta-independent Jacobian);
I(theta; Bx) = I(theta; x) (data-processing inequality saturated by invertible maps).
=> ANY constraint degradation under BNT is a property of the summary, not the transform.

## Two classes of summaries

Per-channel statistic: t_pc(x) = (t_1(kappa_{c_1}),...,t_K(kappa_{c_K})), each component a
functional of ONE (possibly derived) channel. The wavelet l1 datavector is in this class
(binned functionals of the MARGINAL distribution of W_s kappa_c per channel/scale). Two joint
fields with identical per-channel marginals but different cross-channel dependence are
indistinguishable to t_pc — blind to the inter-channel copula at every scale.

Channel-mixing compressor f_phi (CNN+VMIM): first operation is a learned linear channel mix,
so the hypothesis class is CLOSED under channel-basis changes:
    F o (I (x) B) = F  for all B in GL(N)
(kernel-level: K'_{oj}(q) = sum_i K_{oi}(q) B_ij — same shape, absorbed before the first
nonlinearity at zero capacity cost; requires only that the first layer is linear in channels,
true for the plain CNN). NB the per-channel RMS whitening preprocessing is diagonal-linear and
per-basis, so closure survives it: D B^-1 D'^-1 is one absorbable channel map; demeaning
commutes with B. Hence
    max_{f in F} I(theta; f(Bx)) = max_{f in F} I(theta; f(x))
— achievable information is exactly basis-invariant; only optimization can differ. Measured
residual: 0.93x (auto) / 0.88x (auto+product), within compressor-seed scatter.

## Why nulling is adversarial for the per-channel class

t_pc is NOT closed under mixing: marginal reductions do not commute with linear recombination
(t(a kappa_i + b kappa_j) is not a function of t(kappa_i), t(kappa_j)). The defensible causal
chain (each link derived in BNT_THEORY_DEEP_DIVE.md): total information invariant [exact,
P1–P2]; per-channel statistics read only the per-channel marginals [definition, P4b]; the
nulling rows remove the SHARED deep structure — the carrier of the signal's non-Gaussianity —
from every channel simultaneously (differencing of strongly correlated maps; Gaussianization
lemma F5 holds exactly for the independent caricature, and the slice bound F5b shows
recombinations within the nulled span stay slice-like); and the theta-response of the
SNR-binned datavector flattens against its fixed bin grid as per-channel S/N drops.
Measured: l1 on nulled autos 0.15x, sigma(sigma_8) doubles. IMPORTANT NEGATIVE RESULT
(deep-dive F3, worked closed-form): the naive Gaussian one-point account — "per-map S/N drops
and the noise becomes correlated, hence per-channel statistics fail" — does NOT survive
honest analysis; in the idealized-nulling Gaussian toy the per-channel variances become MORE
efficient in the nulled basis (nulling diagonalizes the signal covariance; I_diag^BNT ~
I_full > I_diag^orig). The Gaussian-geometry story is therefore not the mechanism; the
collapse lives in the non-Gaussian sector and the response structure. (Even at Gaussian
level, SCALE-RESOLVED cross-spectra restore invariance in any basis — indicting the pixel
product's scale-blending specifically.) Control: the BNT arm used a RE-FROZEN
per-(channel,scale) sigma in the BNT basis (GATE A1b PASS), so the collapse cannot be blamed
on mis-normalization — it isolates the statistic's inability to follow the channel mixing.

## Why appended quadratic channels recover only part (0.15x -> 0.22x)

kappa'_i kappa'_j = sum_kl B_ik B_jl kappa_k kappa_l — the 10 quadratic monomials (incl.
squares) span a B-invariant LINEAR subspace. Neither condition for basis-invariance holds in
the arm as built: we feed only the 6 i<j products (a sub-basis), and the subsequent reduction
is nonlinear per-channel. Losses in practice:
 (i)  non-commutation again: per-channel histograms of W_s[kappa'_i kappa'_j]; histograms of
      fixed components do not determine histograms of recombinations;
 (ii) scale blending: W_s[kappa_i kappa_j] mixes all Fourier pairs beating into band s
      (convolution theorem); scale-resolved cross-coherence and RELATIVE PHASE of W_s kappa_i,
      W_s kappa_j — where signal is distinguished from correlated noise — are marginalised;
 (iii) noise standardisation: per-(channel,scale) scalar sigma_c(s) cannot represent the
      inter-channel covariance Sigma'.

## The two-point sector is exactly protected once crosses are included (P7)

The sample covariance of the maps transforms congruently and EXACTLY, realization by
realization: C-hat' = B C-hat B^T (per ell-bin for spectra, verbatim). This map is invertible
on symmetric matrices, so the BNT-basis auto+cross second-moment datavector is a lossless
repackaging of the original — identical posteriors for any field, Gaussian or not (the
summary-level instance of the data-level invariance). The autos alone are diag(B C-hat B^T):
not invertible, no protection. This PREDICTS the reported result that BNT leaves
power-spectrum contours unchanged when both auto- and cross-spectra are used [REF], and
locates our measurement as the maximally exposed configuration (higher-order, autos-only).
Practical corollary: appending the 10 auto+cross second moments to a BNT-basis datavector
restores the complete Gaussian sector for free.

## Diagnostic decomposition (whitening identity) — DIAGNOSTIC ONLY, not a practical rescue

Equal per-bin noise => M = (B B^T)^(-1/2) satisfies (M B)(M B)^T = I: noise-whitening the
nulled maps = an ORTHOGONAL rotation Q = M B of the original basis. Per-channel statistics on
M kappa' see independent equal-variance noise again; the residual deficit vs no-BNT isolates
the genuinely joint (rotation-mixed) information. (Footnote: unequal per-bin n_gal =>
Sigma' = B diag(sigma_i^2) B^T; whitener still defined, no longer a pure rotation.)

IMPORTANT FRAMING CORRECTION (red-team pass 2026-06-11): whitening — like B^-1 — REMIXES the
nulled kernels and therefore destroys the very structure BNT is applied for. It is NOT a way
to keep BNT's systematics benefits while rescuing a per-channel statistic. Two honest
framings: (a) DIAGNOSTIC — the whitened run decomposes the inflation into a noise-basis
component vs an irreducibly-joint component (pure information accounting); (b) PIPELINE
DECOUPLING — in a realistic analysis BNT serves the CLEANING step (nulling-informed scale
cuts); after cleaning, the STATISTICS basis is a free choice, and our result says the
post-cleaning information is recoverable in any basis (learned or constructed). (b) is the
paper-level point; state it as its own claim, not as a property of whitening.

MEASURED OUTCOME (2026-06-11, whiten_campaign/): recovered fraction
(whiten − BNT)/(noBNT − BNT) = 1.06 (auto) / 1.01 (+product) — FULL recovery, marginal by
marginal. The irreducibly-joint component is ≈ 0: the l1's BNT loss is entirely a frame
artifact. Mechanism resolution per BNT_THEORY_DEEP_DIVE.md §5: the nulled frame is, by
construction, the unique frame with no deep-kernel direction — per-channel signal and the
deep non-Gaussian structure are minimized in every channel simultaneously; Q recovers because
its leading row restores ≈ the deep common mode (70% of its power outside the nulled span,
appendix table). Noise correlation is invisible to marginals and noise amplification is
absorbed by the SNR normalization, so neither was ever a complete mechanism. (Two earlier
explanations — the pre-registered partial-recovery expectation and a first post-mortem based
on mixing sign structure — were falsified against the data and the transform's own geometry;
the chain is preserved in deep-dive §5 as journey material. Registered next test, §5.4:
appending ONE deep channel to the four untouched nulled maps should recover ≥0.8 — a
practical rescue that, unlike whitening, preserves the nulled channels for per-slice cuts.)

## Interpretation (the two pillars as one statement)

Friendly basis: the TRAINED compressor did not match the hand-crafted statistic on the
explicit cross channel (0.83-0.85x, robust to compressor seed and doubled training budget).
NB the asymmetry: the BNT-invariance argument compares the CNN TO ITSELF across bases (class
closure = exact; 0.93x = measured optimization residual), while the friendly-basis comparison
is between ACHIEVED ESTIMATORS (10-d summary, finite data; independent evidence the CNN is
data-limited) — class capability is exactly what it cannot establish. What the compressor has
— provably by class closure, empirically by the nulling test — is BASIS ADAPTIVITY. Per-channel statistics:
statistic-strong, basis-fragile. Channel-mixing compressors: basis-robust, not
statistic-optimal. The reported BNT inflation of higher-order statistics is an artefact of the
analysis basis, not information loss — removable by learning the basis (CNN) or constructing
it (whitening rotation; scale-resolved cross-coherence channels W_s[kappa_i] W_s[kappa_j];
phase-aware cross statistics, e.g. wavelet phase harmonics).

## Empirical anchors (this work)

- CNN BNT/noBNT: 0.93x auto / 0.88x product (3 compressor seeds; marginals <= 3%); 6/6
  chains at or below 1.0 (mean ~0.90) => quote as 'lossless within seed scatter; residual
  <~10% optimization-in-harder-basis cost not excluded'.
- l1 BNT/noBNT: 0.15x auto (sigma_s8 x2), 0.22x with explicit product channel.
- No-BNT CNN/l1 on product: 0.83-0.85x, robust to seed and 2x recipe.

## Care points (claims hygiene)

- Closure/invariance is stated for the HYPOTHESIS CLASS and achievable MI (exact); the trained
  network's 0.93x is the measured optimization residual.
- Gaussian S'/Sigma' paragraph = mechanism intuition, flagged as such.
- Whitening = pure rotation only for equal per-bin noise variance (true here; footnote else).


================================================================================
## PARTS III–IV — SUPERSEDED (2026-06-11)
================================================================================

The joint-PDF / Cramér–Wold / union-catalog discussion (former Part III) and the point-cloud
picture with the sharp per-scale-joint-one-point statement (former Part IV) are absorbed,
with full derivations and a claims ledger, into `BNT_THEORY_DEEP_DIVE.md`:
- formal core (posterior/MI invariance, CNN class closure, configuration-preserving
  information flow, joint one-point envelope, strict hierarchy, Gaussian-sector l1 lemma): L2;
- worked Gaussian Fisher analysis incl. the F3 trap result, the F5 Gaussianization lemma, the
  whitening (F4) pre-registered reading, and the sigma8/w0 anisotropy adjudication (F6): L3;
- union-catalog identity, constructive order-by-order Cramér–Wold completeness, joint-PDF-as-
  statistic design: L4. Git history retains the original Parts III–IV text.
