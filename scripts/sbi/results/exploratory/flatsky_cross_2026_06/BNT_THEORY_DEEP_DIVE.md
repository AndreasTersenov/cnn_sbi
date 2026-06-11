# BNT and per-channel statistics: where the information goes — the deep-dive (2026-06-11)

Canonical theory treatment for paper pillar 2. Supersedes Parts III–IV of
PAPER_BNT_MAIN_AND_APPENDIX_DRAFT.md (Parts I–II remain the paper-facing prose/appendix and
are revised against this doc). Plan + sign-off: PLAN_BNT_THEORY_DEEP_DIVE.md. Measured numbers
from FLATSKY_BNT_RESULT.md / FLATSKY_CNN_RESULT.md; whitening decomposition from
whiten_campaign/WHITEN_RESULT.md (slot §L5.2). All math here is analytic; the only numerical
inputs are exact constants of the fixed transform `tomo4_bnt_v1` and already-measured campaign
artifacts.

================================================================================
## L0 — Executive statement and claims ledger
================================================================================

BNT is a fixed invertible pixelwise channel mix. It cannot change the information content of
the data (P1–P2), and it cannot change what a channel-mixing compressor class can extract (P3).
It CAN devastate a per-channel statistic, and measurably does: the wavelet l1 keeps 0.15× of
its FoM3 (auto), 0.22× with the explicit product channel; the CNN keeps 0.93×/0.88× (within
compressor-seed scatter). This document derives, rather than asserts, where the damage comes
from. The single most important derived result is a negative one:

> **The Gaussian one-point variance toy, worked honestly (F3), predicts the OPPOSITE of the
> measurement** — perfect nulling diagonalizes the signal covariance, making per-channel
> variances MORE efficient in the nulled basis. The measured collapse therefore cannot be
> Gaussian noise-vs-signal one-point geometry alone. It must live in the three ingredients the
> toy lacks: (i) the non-Gaussian marginal content that mixing provably contracts (F5 — the
> Gaussianization lemma); (ii) residual cross-channel signal response (the real B nulls
> KERNELS, not covariances — F3.4); (iii) response flattening against the fixed SNR bin grid
> (F3.5). The whitening test measures the complementary noise-geometry share (F4).

Claims ledger (every claim in the paper maps to one row; tags: PROVED here, MEASURED in the
campaign, MECHANISM = honest flagged intuition):

| claim | tag | where |
|---|---|---|
| Posterior/MI exactly invariant under fixed invertible B | PROVED | P1, P2 |
| CNN hypothesis class closed under channel GL(N); achievable MI basis-invariant | PROVED | P3 |
| BNT moves info only across channels at fixed spatial configuration & scale | PROVED | P4a |
| Per-scale joint one-point PDF is a basis-invariant envelope of ALL per-channel statistics | PROVED | P4b |
| Hierarchy marginals < finite projections < joint one-point < field is STRICT | PROVED | P5 |
| Gaussian-sector l1 ⟺ per-channel variance | PROVED | P6 |
| Zero-lag Gaussian toy: nulled-basis variances are MORE efficient (the trap) | PROVED | F3 |
| Mixing strictly contracts standardized cumulants (Gaussianization) | PROVED | F5 |
| Union-catalog maps = count-weighted combos of auto maps (no new field info) | PROVED | M1 |
| Pairwise unions complete at 2nd order, incomplete at 3rd (order-by-order accounting) | PROVED | M2 |
| l1 collapse 0.15×/0.22×; CNN 0.93×/0.88× | MEASURED | campaign |
| σ8 hit hardest / w0 mildest; w0 substantially prior-capped | MEASURED+derived | F6 |
| Real-case damage split: noise-geometry vs irreducibly-joint | MEASURED | whitening, L5.2 |
| Residual cross-signal response asymmetry between bases | MECHANISM | F3.4 |

================================================================================
## L1 — Main-text register (the prose the paper carries)
================================================================================

The argument in five sentences, each load-bearing on a result below. (1) BNT is invertible, so
the inflation cannot be a property of the transform — it is a property of the statistic (P1).
(2) Because BNT acts pixel-by-pixel, everything it rearranges stays at zero lag and equal
scale: information moves between channel-marginal and channel-joint structure, never across
positions or scales (P4a) — so "where the information goes" has an exact answer: into the
joint one-point structure of the maps, scale by scale (P4b). (3) A statistic that reduces each
channel separately keeps only the marginal share; the measured collapse calibrates how large
the joint share is for the wavelet l1: 85% of FoM3 (MEASURED). (4) The damage is not mainly
about noise becoming correlated — the worked Gaussian toy shows per-channel variances would
actually IMPROVE under perfect nulling (F3); it is about signal mixing Gaussianizing each
channel's marginal (F5), exactly the non-Gaussian content a higher-order statistic exists to
exploit. (5) A multichannel CNN absorbs any channel re-mix into its first linear layer at zero
capacity cost (P3) — basis-robust, while per-channel statistics are basis-fragile; the
explicit cross-map channels are a partial, fixed-direction patch (M2) for a statistic that
cannot follow the mixing on its own.

================================================================================
## L2 — Formal core (definitions and propositions, with proofs)
================================================================================

**Setup.** Maps κ(p) ∈ R^N (N=4 bins), data x = κ + n with per-pixel noise n ~ N(0, σ²I_N),
iid across pixels and bins (shape noise; injected BEFORE BNT in the pipeline). BNT: x' = Bx
pixelwise, B = tomo4_bnt_v1, lower-triangular, fixed at the fiducial cosmology. Wavelet bank
W_s (starlet, n_scales=5) acts spatially, per channel. The l1 datavector as the pipeline
computes it: per (channel c, scale s), SNR u = (W_s x_c)/σ_c(s) with σ_c(s) the frozen
propagated NOISE std; 40 bins over frozen per-channel percentile SNR ranges; bin content
= Σ |W_s x_c(p)| over pixels with u(p) in the bin. Per-channel class: any t_pc whose every
component is a functional of ONE channel's empirical marginal at one scale (l1, peaks per
bin, per-bin Minkowski, per-bin PDF). Channel-mixing class F: plain CNN whose first layer is
linear in channels (true for every arch used here).

**P1 (posterior invariance).** p(θ | Bx) = p(θ | x) for fixed invertible B.
*Proof.* p(Bx|θ) = p(x|θ)·|det B|^(−N_pix): the Jacobian is θ-independent and cancels in
Bayes' normalization. ∎ (No Gaussianity, no stationarity used.)

**P2 (information invariance).** I(θ; Bx) = I(θ; x).
*Proof.* Data-processing: I(θ; Bx) ≤ I(θ; x) since Bx is a deterministic function of x;
apply again with B⁻¹ to Bx for the reverse inequality. ∎

**P3 (class closure ⇒ basis-invariant achievable information).** For the plain-CNN class F,
F∘(I_pix ⊗ B) = F; hence max_{f∈F} I(θ; f(Bx)) = max_{f∈F} I(θ; f(x)).
*Proof.* First-layer kernels K_{oi}(q) acting on channels i: replacing input x by Bx is
absorbed by K'_{oj}(q) = Σ_i K_{oi}(q)(B)_{ij} — same parameter count, before any
nonlinearity; B invertible makes the map F→F a bijection. Pipeline preprocessing preserves
closure: demeaning commutes with B (linear); per-channel RMS standardization is
diagonal-linear per basis, so D'⁻¹BD is again one absorbable channel map. The supremum over a
set is invariant under a bijection of the set. ∎
*Scope:* exact for the CLASS and achievable MI; the trained network's 0.93× is the MEASURED
optimization residual in the harder basis — class capability vs achieved estimator must not
be conflated (the friendly-basis CNN-vs-l1 comparison is between achieved estimators only).

**P4a (BNT's information flow is exactly channel-space, configuration-preserving).** For any
k points p_1..p_k, scales s_1..s_k, channels i_1..i_k:
⟨(W_{s1}x'_{i1})(p_1)···(W_{sk}x'_{ik})(p_k)⟩ = Σ_{j1..jk} B_{i1 j1}···B_{ik jk}
⟨(W_{s1}x_{j1})(p_1)···(W_{sk}x_{jk})(p_k)⟩.
*Proof.* W_s convolves spatially per channel and B mixes channels pointwise, so they commute:
W_s(Bx) = B(W_s x); expand the product and use linearity of expectation. ∎
*Reading:* the transformed correlator at a given spatial/scale configuration is a linear
combination of original correlators at the SAME configuration. BNT never moves information
across positions, lags, or scales — only across channel indices. "Where does it go" has this
exact answer; everything else below is about which statistics can follow it.

**P4b (joint one-point envelope).** Let P_s = the joint law of (W_s x_1,…,W_s x_N)(p) at scale
s. Then (i) P'_s = B_* P_s (pushforward), so knowing {P_s} in one basis = knowing it in any;
(ii) every per-channel statistic in every basis is a functional of {P_s}; hence (iii)
information attainable from any t_pc in any channel basis ≤ information in {P_s}, which is
basis-invariant. *Proof.* (i) is W_s(Bx) = B(W_s x) at a point, plus invertibility of B_* for
invertible B. (ii): the (c,s) marginal is the c-th coordinate projection of P_s, and t_pc is
by definition a functional of these marginals. (iii) chains (i)+(ii) with P2. ∎
*Corollary:* no multi-point or cross-scale statistic is needed to recover BNT-displaced
information — the per-scale joint one-point PDF already dominates the entire per-channel class
in every basis. (It is NOT full information: see P5(c).)

**P5 (the hierarchy is strict).** Axis marginals < finitely many linear projections <
per-scale joint one-point < field level. *Witnesses.* (a) Two bivariate Gaussians with equal
marginals and ρ = 0 vs ρ ≠ 0: identical to all per-channel statistics, different joint.
(b) Finite-Radon non-uniqueness: for any finite set of directions there exist distinct laws
with identical projections along all of them (classical; the constructive cumulant accounting
in M2 shows exactly which orders go missing). (c) Two Gaussian fields with different C_ij(ℓ)
inside a wavelet band but equal band averages: identical {P_s} (each P_s is N(0, Σ(s)) with
Σ(s) = band-averaged covariance), different fields — joint one-point misses within-band shape
and all phase information. ∎

**P6 (Gaussian-sector l1 ⟺ variance).** If the (c,s) marginal is N(0, σ_tot²), the expected
content of SNR bin [u_1,u_2) is N_pix σ_c(s) · E[|U| 1{U∈[u_1,u_2)}] with
U ~ N(0, r²), r = σ_tot/σ_c(s): explicitly N_pix σ_c(s) r [φ(u_1/r)−φ(u_2/r)]·… a closed
truncated-Gaussian integral that depends on the marginal ONLY through r. The expected l1
datavector at (c,s) is a deterministic curve in the single parameter σ_tot — i.e. in the
Gaussian sector the l1 carries exactly the per-channel variances {C_ii(s)}, no more.
*Consequence:* the Gaussian variance analysis of L3 is the exact Gaussian-sector skeleton of
the real statistic, and everything the l1 adds beyond it is non-Gaussian marginal shape —
precisely the sector attacked by F5. ∎

================================================================================
## L3 — The worked Gaussian analysis (closed form throughout)
================================================================================

Zero-lag per-pixel model (one scale; identical structure per scale): y = κ + n per pixel,
κ ~ N(0, S(θ)), n ~ N(0, σ²I), y ~ N(0, C = S+σ²I), N i.i.d. pixels. BNT: C' = BSBᵀ + σ²BBᵀ.

**F1 (full-data Fisher; invariance shown explicitly).**
I_ab = (N/2) tr(C⁻¹C_{,a}C⁻¹C_{,b}). Under C → BCBᵀ: C'⁻¹ = B⁻ᵀC⁻¹B⁻¹ and C'_{,a} = BC_{,a}Bᵀ;
substituting, all B's cancel cyclically inside the trace. ∎ (F-side restatement of P1–P2.)

**F2 (per-channel-variance summary Fisher).** Summary T̂_i = (1/N)Σ_p y_i(p)².
E[T̂] = d ≡ diag(C); Cov(T̂_i,T̂_j) = (2/N) C_ij² (Wick: Cov(y_i², y_j²) = 2C_ij²).
Asymptotically (CLT over pixels) T̂ is Gaussian and the summary Fisher is mean-dominated:
  I_diag = (N/2) (∂_a d)ᵀ M⁻¹ (∂_b d),  M_ij = C_ij².
The covariance's own θ-dependence contributes O(1) vs O(N) — stated, not hidden. Note where
the cross-channel structure enters a statistic that never looks across channels: through
M_12 = C_12² — correlated CHANNELS make correlated SAMPLING NOISE of the per-channel
variances. This is the precise form of "correlated noise penalizes per-channel statistics."

**F3 (the 2-bin nulling toy, worked to the end).**
Nested-kernel caricature: κ_2 = κ_1 + β with β ⊥ κ_1 (bin 2 sees bin 1's lenses plus an
increment), Var(κ_1) = Au, Var(β) = Av; A = amplitude parameter (σ8-like). B = [[1,0],[−1,1]].
Then S = [[Au, Au],[Au, A(u+v)]] and — the toy's defining feature — nulling EXACTLY
diagonalizes the signal: S' = diag(Au, Av), while the noise picks up the correlation:
σ²BBᵀ = σ²[[1,−1],[−1,2]], ρ'_noise = −1/√2 ≈ −0.707 (the real tomo4_bnt_v1 value for the
(1,2) pair — see F3.4). So:
  C  = [[Au+σ², Au],[Au, A(u+v)+σ²]]        (original basis)
  C' = [[Au+σ², −σ²],[−σ², Av+2σ²]]          (nulled basis; off-diagonal = PURE noise)
Regime of interest ("nulling worked"): Au ≫ σ² ≫ Av, v ≪ u.

*F3.1 — original basis.* With a = Au+σ², b = A(u+v)+σ², c = Au, ∂_A d = (u, u+v):
  I_diag^orig = (N/2)·[u²b² − 2u(u+v)c² + (u+v)²a²] / (a²b² − c⁴)
Expanding to leading order (the x² coefficient is exactly v² by (u+v−u)²):
  I_diag^orig ≈ (N/2)·[1/A² + v²/(4Auσ²)] → (N/2)/A².
The buried increment contributes ~v²/(4Auσ²) ≈ nothing: in the original basis the two sample
variances are almost perfectly correlated (both dominated by the same κ_1), and their
difference — where β lives — is suppressed by the common-mode variance. The amplitude is
measured at the self-calibration (cosmic-variance) limit (N/2)/A².

*F3.2 — nulled basis.* a' = Au+σ², b' = Av+2σ², c'² = σ⁴, ∂_A d' = (u, v):
  I_diag^BNT = (N/2)·[u²b'² − 2uvσ⁴ + v²a'²] / (a'²b'² − σ⁸) ≈ (N/2)·[1/A² + v²/(4σ⁴)].

*F3.3 — full Fisher and the punchline.* From C' (noise off-diagonal kept):
  I_full = (N/2)·[u²b'² + 2uvσ⁴ + v²a'²] / (a'b' − σ⁴)² ≈ (N/2)·[1/A² + v²/(4σ⁴)].
Since Au ≫ σ²: v²/(4σ⁴) ≫ v²/(4Auσ²), hence
  **I_diag^BNT ≈ I_full > I_diag^orig.**
In the honest Gaussian one-point toy, per-channel variances are MORE informative in the
nulled basis: nulling un-buries the increment by removing the common mode, and the noise
correlation costs only the sub-leading ∓2uvσ⁴ term. The often-told story "BNT correlates the
noise and that's why per-channel statistics fail" is, at the Gaussian one-point level,
backwards. THE TRAP IS THE RESULT: the measured 0.15× collapse cannot be Gaussian one-point
variance geometry. What the toy lacks, the real case has — three derivable ingredients:

*F3.4 — residual cross-signal response (real B nulls kernels, not covariances).* tomo4_bnt_v1
nulls the lensing KERNEL below the bin edge at the fiducial cosmology; the signal covariance
is not diagonalized (only the foreground share cancels), so S' retains off-diagonals carrying
θ-response that no per-channel statistic can read in EITHER basis — but the bases are not
symmetric: in the original basis the off-diagonal response is largely REDUNDANT with the auto
response (overlapping kernels), in the nulled basis the autos' response is reduced by the
designed cancellation while the residual joint response is not. Perturbing F3 with S'_12 ≠ 0
moves I_diag^BNT down while I_full is invariant (F1). [MECHANISM — direction derivable,
magnitude not, without the full kernel model.] Exact constants of the real transform (from
B alone, verified empirically by GATE A1b): per-bin noise amplification √(BBᵀ)_ii =
(1.000, 1.414, 1.820, 1.621); noise correlations ρ'_12 = −0.707, ρ'_23 = −0.740,
ρ'_34 = −0.548, ρ'_13 = +0.248, ρ'_24 = +0.110, ρ'_14 = 0.000; noise-ellipsoid eigenvalues
of BBᵀ = (0.088, 0.838, 2.417, 5.599) — condition number 63: there are directions in channel
space where the noise std is 0.30σ against 2.37σ — an 8× spread in noise std (63× in
variance) that only joint (channel-mixing) analyses can exploit.

*F3.5 — SNR-grid response flattening.* The l1 bins coefficients by SNR with the (correctly
re-frozen) per-channel σ_c(s). Post-BNT the noise is amplified (×1.41–1.82) while the
per-channel signal share drops, so a given physical fluctuation moves fewer SNR bins: the
datavector's θ-response flattens against the FIXED bin grid. Per channel this is a monotone
reparametrization (no information destroyed in principle) but a real efficiency loss for a
fixed finite binning — and it compounds with F3.4 and F5. [MECHANISM; the empirical witness
is the σ8-flat relative datavector, below.]

**F4 (whitening, analyzed before reading the result).** Q = (BBᵀ)^(−1/2)B is orthogonal
(QQᵀ = I, exact for equal per-bin noise; verified to 4e-8 in the campaign), so x'' = Qx has
noise N(0, σ²I) again — independent, equal variance — while the signal becomes
S'' = QSQᵀ = (BBᵀ)^(−1/2) S' (BBᵀ)^(−1/2): the whitener un-correlates the noise at the price
of RE-MIXING the (toy-diagonalized) signal. What whitening can and cannot restore:
- restores: everything attributable to noise GEOMETRY (amplification + correlation): the F2
  penalty M'_12, the F3.5 SNR-grid crushing from the noise side.
- cannot restore: per-channel access to joint signal response (F3.4 — Q is a basis like any
  other), and the non-Gaussian marginal content already contracted by ANY mixing — F5 applies
  to orthogonal Q exactly as to B.
Pre-registered reading of recovered = (whiten − BNT)/(noBNT − BNT): HIGH (>0.8) ⇒ the
collapse was dominantly noise-ellipsoid geometry (Gaussian-sector mechanics; would partially
rehabilitate the "correlated noise" story); MID (0.4–0.8) ⇒ comparable shares; LOW (<0.4) ⇒
dominantly irreducibly-joint / mixing-destroyed structure (the F3-trap + F5 prediction).
Given F3's result — at the Gaussian one-point level there is nothing for whitening to give
back that BNT took — F5-type damage should dominate and we expect LOW-to-MID. [Written
before WHITEN_RESULT.md; the measured number goes in L5.2 untouched either way.]
NB whitening, like B⁻¹, destroys the nulled kernels — it is information accounting, not an
analysis recipe; the pipeline-level statement stays "cleaning basis ≠ statistics basis."

**F5 (the Gaussianization lemma — mixing contracts standardized cumulants).**
Let κ_j be independent with variances λ_j and standardized cumulants γ_k(j) = cum_k/λ_j^(k/2),
and m = Σ_j b_j κ_j. Cumulant additivity under independence gives
  γ_k(m) = Σ_j b_j^k λ_j^(k/2) γ_k(j) / (Σ_j b_j² λ_j)^(k/2).
*Lemma.* For k ≥ 3: |γ_k(m)| ≤ max_j |γ_k(j)|, with equality iff a single component
contributes. *Proof.* With z_j = |b_j|√λ_j: |γ_k(m)| ≤ Σ_j z_j^k max|γ| / (Σ_j z_j²)^(k/2)
= max|γ| · (‖z‖_k/‖z‖_2)^k ≤ max|γ|, by ℓ^p-norm monotonicity ‖z‖_k ≤ ‖z‖_2 for k > 2;
equality iff z has one nonzero entry. ∎
Signed mixing (BNT rows alternate signs by design) can do far better than the bound — odd-k
terms b_j^k cancel directly. *Reading:* mixing maps the signal's per-channel marginals toward
Gaussianity (a one-step CLT) while the noise stays Gaussian — so EXACTLY the content by which
the l1 beats a variance (P6: everything beyond r = σ_tot/σ_n) is contracted in every mixed
channel. For correlated components (the real κ_j) additivity fails but the cancellation
mechanism is the same; the nulling rows are BUILT to cancel the common (correlated) share.
[Lemma PROVED for the independent caricature; real-case version MECHANISM.] The empirical
witness that the real case sits in this regime: `datavectors_bnt_vs_nobnt_s8_relative` — under
BNT the auto and conv blocks lose almost all σ8 response while the product block retains the
most. A variance-level collapse would flatten everything equally; the selective survival of
the explicitly-joint channel is the F5 signature.

**F6 (the σ8/w0 anisotropy, adjudicated with existing artifacts only).**
Measured (noBNT → BNT, L1 auto): σ(σ8) 0.082→0.176 (2.15×), σ(Ωm) 0.053→0.090 (1.70×),
σ(w0) 0.245→0.323 (1.32×). Two mechanisms, both real:
(i) *Prior geometry.* The effective prior is the CosmoGrid training support: σ8 ∈ [0.40,1.40],
Ωm ∈ [0.10,0.50], w0 ∈ [−1.93,−0.33]; uniform-equivalent σ_prior ≈ (0.288, 0.115, 0.462).
Posterior/prior width ratios: σ8 0.28→0.61, Ωm 0.46→0.78, w0 0.53→0.70. The maximum inflation
the prior wall allows is ≈ (3.5×, 2.2×, 1.9×) respectively — w0 was HALF prior-limited before
BNT ever acted, so its mild 1.32× is substantially a ceiling effect, not BNT-resilience; as a
FRACTION of available room all three parameters lose comparably (61%, 78%, 70%). The dumbbell
ordering largely reflects how much room each parameter had. [Derived from artifacts.]
(ii) *Sector exposure.* σ8 is the amplitude of the non-Gaussian signal: its constraint above
the Gaussian floor rides exactly on the marginal non-Gaussianity that F5 contracts — maximal
exposure. Shape/kernel parameters (w0) keep response in surviving structure; additionally B is
pinned at θ_fid, so kernel-moving parameters re-light the nulled channels at linear order in
δθ (the nulling residual is itself a w0-sensitive signal a per-channel statistic CAN see),
while a pure amplitude change keeps nulled combinations nulled. [MECHANISM.]

================================================================================
## L4 — Survey practice and the joint-PDF program
================================================================================

**M1 (union catalogs add bookkeeping, not field information).** Per-bin map from catalog i
(linear mass-mapping estimator, e.g. KS): κ̂_i = κ_i + n_i, n_i = per-pixel average of N_i
galaxy ellipticities, Var = σ_e²/N_i. The union catalog i∪j re-runs the SAME linear estimator
on the pooled galaxies:
  κ̂_{i∪j} = (N_i κ̂_i + N_j κ̂_j)/(N_i+N_j),
because a linear estimator of a count-weighted pooled catalog is the count-weighted
combination of the per-catalog estimators — same galaxies, same noise REALIZATIONS, regrouped.
Noise check: Var = (N_i²·σ_e²/N_i + N_j²·σ_e²/N_j)/(N_i+N_j)² = σ_e²/(N_i+N_j) — identical to
running on the pooled catalog directly. ∎ So every union-catalog statistic is computable from
the per-bin maps; catalog access adds nothing at the field level (caveats: per-pixel varying
counts, weights, masks make the combination spatially varying — bookkeeping, not new
information). The BNT inflation is NOT a data-access limitation.

**M2 (constructive Cramér–Wold: order-by-order completeness of combination maps).**
Cramér–Wold: the 1-d laws of w·κ for ALL directions w determine the joint law — so per-channel
one-point statistics on a sufficiently rich family of LINEAR combination maps are equivalent
to the joint one-point PDF (P4b's envelope), and union catalogs are exactly such maps with
count-weights (M1). Basis-agnostic: wᵀκ' = (Bᵀw)ᵀκ — the same family is constructible from
nulled maps. How rich is "sufficiently"? Cumulants give the exact accounting. For a pair
(i,j), order k:
  cum_k(w_i κ_i + w_j κ_j) = Σ_{m=0..k} C(k,m) w_i^m w_j^(k−m) · cum(κ_i^[m], κ_j^[k−m])
— a homogeneous degree-k polynomial in (w_i, w_j) whose k+1 coefficients are the mixed
cumulants. The two autos supply m = 0, k; the k−1 mixed cumulants need k−1 distinct weight
ratios (Vandermonde in (w_i/w_j)). Hence:
- order 2: ONE pairwise union (any ratio) completes the cross-covariance — equal-weight
  pairwise unions are provably COMPLETE at second order;
- order 3: two unknowns (cum(κ_i²κ_j), cum(κ_iκ_j²)) vs one equation — a single equal-weight
  union per pair is provably INCOMPLETE at third order; a second ratio per pair would close it;
- order k: k−1 ratios per pair; ≥3-bin mixed cumulants (e.g. cum(κ_iκ_jκ_l)) require ≥3-bin
  unions — directions must grow with the joint structure sought.
Two bonuses: Gaussian noise contributes nothing to cum_{k≥3}, so mixed cumulants of order ≥3
estimated from union maps are noise-bias-free in expectation (only their variance feels the
noise); and the pixelwise PRODUCT maps probe a different (quadratically warped) family —
E[κ_iκ_j] = ξ_ij at zero lag — neither family contains the other, both are strictly below the
joint one-point PDF (P5). Honesty cap: all of this is one-point-level completeness; field-level
claims rest on P4a/P5(c), not Cramér–Wold.

**M3 (the joint PDF as a statistic, concretely).** The P4b envelope is constructible: per
scale s, the joint histogram of (W_s x_1,…,W_s x_4)(p). Binning arithmetic at our sizes: 6
bins/axis → 6⁴ = 1296 cells × 5 scales ≈ 6.5k numbers; or the pairwise-2D reduction (six
15×15 histograms × 5 scales ≈ 6.8k; complete only to pairwise structure — P5(b)) — both within
a factor ~2–8 of the current 800–3200-d l1 datavectors, and the classical obstruction (a
10⁴-cell covariance matrix) does not exist in SBI: the NDE consumes the histogram directly,
as it already consumes the l1. Noise handling generalizes the scalar σ_c(s) to the matrix
whitener Σ'(s)^(−1/2) (known analytically per basis from B). Basis-covariant by P4b ⇒
BNT-robust by construction. Wavelet-domain joint one-point histograms as an SBI datavector
would be new (the LDT lensing-PDF program is moving toward joint tomographic PDFs at low
order); this is the principled "fixed statistic that survives BNT" — a candidate third pillar,
NOT launched (needs its own campaign + explicit go).

================================================================================
## L5 — Synthesis
================================================================================

**L5.1 The two pillars as one statement.** Per-channel statistics are statistic-strong but
basis-fragile: given a friendly basis and the right explicit channel they beat the trained
compressor (CNN/L1 = 0.83–0.85× on product, seed- and recipe-robust), but a hostile basis
removes 85% of their power (0.15×) because they can only read marginals (P4b) and the mixing
has Gaussianized those (F5). Channel-mixing compressors are basis-robust but not
statistic-optimal: class-closure makes their achievable information exactly basis-invariant
(P3), measured residual 0.93×/0.88× = optimization-in-harder-basis cost. The asymmetry of the
two comparisons matters: the BNT claim compares the CNN TO ITSELF across bases (exact closure
+ measured residual); the friendly-basis claim compares two achieved estimators. Cross-maps
are a device FOR per-channel statistics (M2's finite-direction patch): pillar 1 shows the CNN
neither needs nor uses them; pillar 2 shows the per-channel statistic without them loses most
of its power when the basis turns hostile — and with them (product, 0.22×) recovers only the
share their fixed directions carry.

**L5.2 The whitening decomposition (slot — filled when WHITEN_RESULT.md lands).**
recovered = (whiten − BNT)/(noBNT − BNT) = ____ . Reading per F4's pre-registered ladder:
>0.8 noise-geometry-dominated / 0.4–0.8 mixed / <0.4 irreducibly-joint-dominated; F3+F5
predict LOW-to-MID. [TO FILL]

**L5.3 What would rescue a fixed statistic (ranked by principle).** (1) The per-scale joint
one-point PDF (M3) — the canonical BNT-robust object (P4b). (2) Scale-resolved cross-coherence
channels W_s[κ_i]·W_s[κ_j] (restore Gaussian-sector invariance scale by scale; indict the
pixel product's scale blending specifically). (3) Richer linear-combination families (M2,
Cramér–Wold-complete in the limit, order-by-order in practice). (4) The pixel product as
deployed — one fixed quadratic family, measured at 0.15→0.22. The cleaning basis and the
statistics basis need not coincide: BNT keeps its systematics-localization purpose; after the
cuts, extract in whatever basis the statistic can actually read.
