# Paper section draft — why nulling is lossless for channel-mixing compressors (2026-06-11)

Companion to BNT_CROSS_INFO_ARGUMENT.md (informal version) and FLATSKY_BNT_RESULT.md (numbers).
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
