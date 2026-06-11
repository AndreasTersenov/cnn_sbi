# Paper draft — BNT losslessness: main-text + appendix versions (2026-06-11)

Two registers of the same argument, per Andreas's plan: the cosmologist-oriented version for
the MAIN TEXT, the formal version (post red-team pass) for an APPENDIX. Numbers from
FLATSKY_BNT_RESULT.md; informal precursor BNT_CROSS_INFO_ARGUMENT.md; supersedes
PAPER_BNT_THEORY_SECTION_DRAFT.md (kept for history).

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

The resolution is that BNT moves information, and some statistics cannot follow it. By
construction, nulling concentrates the signal into fewer maps and leaves the remaining maps
with little signal over their noise; at the same time, because each nulled map is a
combination of the original bins, the shape noise — independent between bins before nulling —
becomes correlated BETWEEN the nulled maps. The total information is conserved, but it now
resides less in the appearance of each individual map and more in the relationships between
maps: which fluctuations appear coherently across bins, and how the correlated noise can be
told apart from signal. A statistic computed map-by-map — the wavelet l1, peak counts,
Minkowski functionals, any statistic that reduces each channel separately before comparison —
sees only the individual maps, and so inherits only the diminished per-map share. This is why
the inflation occurs, and why it is largest exactly when nulling works as designed.

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
(t(a kappa_i + b kappa_j) is not a function of t(kappa_i), t(kappa_j)). Gaussian-sector
intuition (mechanism, not theorem): S' = B S B^T against Sigma' = sigma^2 B B^T — nulling
suppresses per-map S/N of the nulled bins while conserving total information, so the
information share in inter-channel structure grows by exactly what the marginals lose.
Measured: l1 on nulled autos 0.15x, sigma(sigma_8) doubles. (NB info does not decompose
additively into diagonal/off-diagonal shares — the defensible chain is: total invariant
[exact] + per-map S/N of nulled bins drops [by construction] + per-channel statistic loses 85%
[measured]; the 'migration' sentence is connective intuition. Even at Gaussian level,
SCALE-RESOLVED cross-spectra restore invariance in any basis — indicting the pixel product's
scale-blending specifically.) Control: the BNT arm used a RE-FROZEN per-(channel,scale) sigma
in the BNT basis (GATE A1b PASS), so the collapse cannot be blamed on mis-normalization — it
isolates the statistic's inability to use cross-channel CORRELATION.

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
## PART III — What would it take to fix per-channel statistics? (discussion material,
## from Andreas's Q&A 2026-06-11)
================================================================================

Q1 — completeness of "PDF" statistics:
- Per-bin one-point PDFs are per-channel => BNT-fragile (Part II argument).
- The MULTIVARIATE cross-bin one-point PDF P(kappa_1(p),...,kappa_N(p)) per smoothing scale is
  BASIS-COVARIANT under pixelwise mixing: P'(k') = P(B^-1 k')|det B|^-1 — knowing it in one
  basis = knowing it in any. The canonical 'normal' statistic inheriting the CNN's invariance.
  Smoothing commutes with B => holds scale by scale.
- It is NOT full information: one-point objects (even joint, even across a scale ladder)
  discard spatial morphology/phase; full info = field level (what CNN+SBI approximates).
  Ladder: per-bin PDFs < pairwise joint PDFs < N-dim joint PDF per scale < field level.
- The product map's one-point PDF is a STRICT functional of the pairwise joint PDF (the law of
  the product does not determine the joint) => the principled 'normal' upgrade over our
  product channel is the pairwise joint PDF itself. Practical obstacle: curse of dim.

Q2 — survey practice (cross catalogs) and data access:
- A union-catalog map is a DETERMINISTIC count-weighted linear combination of the noisy
  per-bin maps (same galaxies, same noise realizations regrouped). Catalogs add bookkeeping
  (weights/masks/varying noise), not field information => THE INFLATION IS NOT A DATA-ACCESS
  LIMITATION (auto maps suffice to build everything cross catalogs build).
- The practice = feeding per-channel statistics LINEAR-combination channels (vs our QUADRATIC
  product/conv). CRAMER-WOLD: the 1-d laws of ALL linear combinations t.kappa determine the
  full multivariate law => per-channel one-point PDFs on a sufficiently rich family of
  combination maps are EQUIVALENT to the joint cross-bin PDF (the BNT-robust object). Surveys'
  pairwise equal-weight unions = a finite sample of directions => partial.
- Constructive version: cum_k(w_i k_i + w_j k_j) is a polynomial in (w_i,w_j) with
  binomial-weighted mixed-cumulant coefficients => k+1 weight ratios determine ALL pairwise
  mixed cumulants of order k. >=3-bin mixed cumulants need >=3-bin unions (directions must
  grow with the joint structure sought).
- Basis-agnostic: w^T kappa' = (B^T w)^T kappa — the same combination family is constructible
  from BNT maps; another face of 'the statistics basis is a free choice'.
- Taxonomy of cross-map constructions for the paper: LINEAR (survey practice;
  Cramer-Wold-complete in the limit) | QUADRATIC (our product/conv; specific mixed moments,
  scale-blended) | LEARNED (CNN; adaptive, empirically sufficient). The measured inflation is
  the bottom rung (zero cross channels, hostile basis), not a ceiling of map-level statistics.
- Honesty cap: Cramer-Wold completeness is a ONE-POINT-level statement; full-field claims
  still rest on the Part II field-level argument.


================================================================================
## PART IV — The point-cloud picture (pedagogical core; Andreas Q&A 2026-06-11)
================================================================================

THE PICTURE: smooth at scale s; each pixel = a 4-VECTOR => the map stack = a cloud of N_pix
points in 4D. Per-channel PDF/l1 = axis SHADOWS of the cloud; joint PDF = the cloud's SHAPE;
cross-correlations (all orders) = the ways the cloud is not the product of its shadows; BNT =
a linear distortion of the cloud (info unchanged, axes changed); Martinet union map = a
DIAGONAL shadow; product map = shadow of a quadratically WARPED cloud.

WHERE THE INFO GOES UNDER BNT (geometric): the noise blob is spherical pre-BNT, an ELONGATED
ELLIPSOID (sigma^2 B B^T) post-BNT; the signal directions are no longer axis-aligned while the
noise is fat along the axes => every axis shadow is noise-dominated (the sigma8-flat histogram
grid) even though in 4D the signal still stands out along directions where the noise is thin.
2-bin toy: k2' = k2 - k1 = B + (n2 - n1): doubled noise, anti-correlated with channel 1.
SHARP STATEMENT: BNT is pixelwise => everything it moves is at ZERO LAG and EQUAL SCALE; the
per-scale joint ONE-POINT PDF captures in principle all BNT-displaced information (multi-point
cross stats are not needed for BNT recovery). Non-Gaussian sector: post-BNT the surviving
non-Gaussianity is in JOINT tail events (several maps simultaneously extreme in the right
pattern vs noise coincidences) — invisible to any per-map histogram.
HONESTY NOTE: in degenerate toys the diagonal of C can still determine the params and the loss
is estimator INEFFICIENCY (noise-correlation inflates diag-estimator variance); in the real
4-bin non-Gaussian case both effects act; 0.15x is the combined measurement; the whitening
test separates the noise-geometry share.

JOINT PDF, CONCRETELY: histogramdd of the per-pixel 4-vectors per smoothing scale (wavelet
version: joint histogram of W_s kappa_1..4 = the strict joint generalization of l1). Classical
obstacle (10^4-cell covariance) EVAPORATES IN SBI: coarse-bin (6^4 cells or 6 pairwise 15x15
2D histograms ~ 1300 numbers, comparable to our 800-3200-d datavectors), feed to the same MAF.
Basis-covariant => BNT-robust by construction (Part II). LDT lensing-PDF programme is moving
toward joint tomographic PDFs at low order; the wavelet-domain joint version would be new.

MARTINET MAPPING (exact): union-catalog map = count-weighted average of the bin maps, noise
included: k_{i u j} = (n_i k_i + n_j k_j)/(n_i + n_j); pooled-noise variance identical either
way => EVERYTHING Martinet extracts is extractable from the auto maps (catalogs add
bookkeeping, not field information). What a union map adds vs per-channel-on-autos: ONE
diagonal shadow; its k-th cumulant = binomial-weighted sum of order-k MIXED cumulants.
COMPLETENESS = CRAMER-WOLD (CT-scan metaphor): 1-d laws along ALL directions reconstruct the
joint law; per-channel = 4 axis projections; Martinet pairs = a few diagonal projections (a
finite Radon sampling); more weight ratios pin individual mixed cumulants order-by-order;
triplet unions reach 3-bin cumulants; the all-directions limit IS the joint PDF. Product maps
= a different (nonlinear-warp) probe family; neither contains the other; both < joint PDF.

HIERARCHY (one line): axis shadows (per-channel PDF/l1) < finitely many shadows (Martinet
unions, products) < cloud shape (joint one-point PDF per scale) < full field (multi-point /
field level). BNT-induced losses live entirely within level 3 (pixelwise transform).

WHITENING TEST (running 2026-06-11, whiten_campaign/): per-channel L1 in Q = (BB^T)^-1/2 B
(orthogonal; noise blob spherical again). Recovered fraction (whiten-BNT)/(noBNT-BNT)
decomposes the inflation into noise-geometry vs irreducibly-joint components.
