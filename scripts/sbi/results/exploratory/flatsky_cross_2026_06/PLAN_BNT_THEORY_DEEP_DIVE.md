# PLAN — BNT theory deep-dive (consolidated, layered, proof-grade)

**Date:** 2026-06-11. **Status:** SIGNED OFF (Andreas, 2026-06-11): go as written; Q-A = state
the union-catalog completeness accounting generically, do NOT attribute the specific
construction to Martinet et al. (no fetch); Q-B = prior-bound read allowed as artifact;
Q-C = **hard cap ~400 lines** (dense complete proofs, sketches where unavoidable).
**Deliverable:** `BNT_THEORY_DEEP_DIVE.md` (this directory) — the canonical standalone treatment.
`PAPER_BNT_MAIN_AND_APPENDIX_DRAFT.md` afterwards gets trimmed to the paper-facing layers
(Parts I–II revised) + a pointer; Parts III–IV are absorbed here and retired.
**Constraints from the interview:** derivations not assertions ("nothing proved, just stated"
is the failure mode to fix); NO numerics — all math analytic/closed-form, all numbers from the
measured campaign; layered register (main-text prose → formal → worked math → extensions).

---

## Document structure (6 layers)

**L0 — Executive statement** (~1 page). The claims, the measured anchors (0.15×/0.22×/0.93×/
0.88×; whitening recovered fraction once landed), and a **claims-classification table**: every
claim tagged PROVED / MEASURED / MECHANISM (honest intuition, flagged). This table is the
contract for the rest of the doc — anything tagged PROVED must have a proof below.

**L1 — Main-text prose** (revised Part I, ~1.5 pages). Cosmologist register, no equations
beyond κ' = Bκ. Every sentence backed by a labeled proposition or measured number from L2–L4.

**L2 — Formal core: definitions + propositions WITH PROOFS.**
Notation: maps κ(p) ∈ R^N, noise model, wavelet bank W_s, the two statistic classes
(per-channel t_pc; channel-mixing class F), the actual l1 datavector defined exactly as the
pipeline computes it (SNR-binned sums of |W_s κ_c|, per channel/scale — so later statements
attach to the real object, not an idealization).

- **P1 (posterior invariance).** p(θ|Bx) = p(θ|x) for fixed invertible B. Proof: θ-independent
  Jacobian cancels in Bayes' normalization. Exact; no Gaussianity.
- **P2 (MI invariance).** I(θ;Bx) = I(θ;x). Proof: data-processing inequality applied in both
  directions (B, B⁻¹ deterministic).
- **P3 (CNN-class closure).** F∘(I⊗B) = F for the plain-CNN class; hence
  max_{f∈F} I(θ;f(Bx)) = max_{f∈F} I(θ;f(x)). Proof at kernel level (first-layer channel mix
  absorbs B), **including the actual preprocessing chain**: demeaning commutes with B;
  per-channel RMS standardization is diagonal-linear per basis, so D'⁻¹·B·D is one absorbable
  channel map. States precisely what is class-level (exact) vs trained-network (measured 0.93×).
- **P4 (per-channel statistics ⊂ marginal functionals ⊂ joint one-point).** (i) t_pc is a
  functional of the per-(channel,scale) marginal laws; (ii) marginals are projections of the
  per-scale **joint** one-point law P_s; (iii) P_s is basis-covariant: P'_s = B_* P_s, because
  BNT is pixelwise and commutes with W_s. **Corollary (the sharp statement, now proved):** the
  per-scale joint one-point PDF is a basis-invariant information envelope over all per-channel
  statistics in all bases — whatever any t_pc extracts in any basis is recoverable from
  {P_s} in any other basis. BNT-displaced information never needs multi-point cross statistics.
- **P5 (strict hierarchy with counterexamples).** Axis marginals < finitely-many projections
  (unions/products) < per-scale joint one-point < field level. Each gap witnessed by an
  explicit pair of distributions: (a) bivariate Gaussians with equal marginals, different ρ;
  (b) classical finite-Radon non-uniqueness (two laws agreeing on any finite direction set);
  (c) Gaussian fields with different C_ij(ℓ) but equal band averages over the wavelet bands —
  identical {P_s}, different fields.
- **P6 (l1 under Gaussian marginals).** Closed form for the expected SNR-binned l1 datavector
  under a Gaussian marginal: each bin's content is an explicit function of the single scale
  parameter σ_c(s) (truncated-Gaussian integrals). Establishes: in the Gaussian sector the l1
  carries exactly the information of the per-channel variances — the bridge lemma that makes
  the L3 Gaussian analysis the *exact* Gaussian-sector skeleton of the real statistic.

**L3 — The worked Gaussian analysis (the analytic backbone; all closed-form).**
Per-pixel zero-lag model: y = κ + n, κ ~ N(0,S(θ)), n ~ N(0,σ²I_N), y ~ N(0,C=S+σ²I),
N_pix i.i.d. pixels (idealization flagged; per-scale version identical in structure).
BNT: C' = BSB^T + σ²BB^T.

- **F1 (full-data Fisher, basis-invariant).** I_full = (N_pix/2)tr(C⁻¹C_{,a}C⁻¹C_{,b});
  explicit two-line proof of invariance under C → BCB^T.
- **F2 (diag-summary Fisher).** Summary T̂ = sample variances of the channels. Asymptotic
  sampling law: mean μ = diag(C'), Cov(T̂) = (2/N_pix)·[C'²]_{ij} (Wick; the off-diagonal
  C'_12² term is **exactly where correlated noise penalizes a per-channel statistic** even
  though the statistic never looks at the cross-channel). Summary Fisher
  I_diag = ∂μ^T Cov⁻¹ ∂μ; the O(1) θ-dependence of Cov vs O(N_pix) mean term stated honestly.
- **F3 (the 2-bin worked example — the heart).** B = [[1,0],[−1,1]], S(θ) with one amplitude
  parameter (σ8-like) and one shape parameter (tilts relative bin power, w0-like). Derive
  **closed-form I_diag/I_full in both bases** as functions of the nulled channel's S/N and the
  induced noise correlation ρ' = (BB^T)_12/√((BB^T)_11(BB^T)_22). Show explicitly:
  (i) original basis — diag near-efficient;
  (ii) BNT basis — same statistic, efficiency collapses as nulled-channel S/N → 0 and |ρ'|→1;
  (iii) **identifiability vs efficiency made precise** (the prior session's honest catch): in
  this toy diag(C') still *identifies* θ (no information loss at the consistency level) — the
  collapse is summary-Fisher efficiency. True summary-level information *loss* enters exactly
  when the marginal response degenerates (∂μ rank-deficient); connect to the measured
  σ8-flat datavector blocks (`datavectors_bnt_vs_nobnt_s8_relative` — the empirical witness
  that the real case is in the compound regime: response suppression AND noise penalty).
- **F4 (whitening, analyzed before the result is read).** Q = (BB^T)^{−1/2}B orthogonal ⇒
  noise restored to σ²I; diag-summary Fisher in the Q basis = diag Fisher of a *rotated*
  signal: recovered fraction < 1 whenever the rotation moves signal power off-axis. Derive the
  toy's recovered fraction (whiten−BNT)/(noBNT−BNT) in closed form ⇒ an **interpretation
  framework written down before today's WHITEN_RESULT.md lands** (pre-registered reading of
  high/mid/low recovered fractions in terms of noise-geometry vs rotation-mixed signal).
- **F5 (Gaussianization of mixed marginals — the non-Gaussian mechanism, derived).** For the
  independent-component caricature: standardized cumulant of order k of a mix Σ_j b_j κ_j is
  γ_k(mix) = Σ_j b_j^k λ_j^{k/2} γ_k(j) / (Σ_j b_j² λ_j)^{k/2} — signed mixing *shrinks*
  standardized cumulants (CLT-in-miniature; alternating-sign nulling rows cancel the common
  correlated part). Consequence: BNT doesn't just bury per-channel signal under correlated
  noise — it **Gaussianizes the per-channel marginals of the signal**, destroying precisely
  the non-Gaussian content that makes the l1 worth more than a power spectrum. Real-case
  (correlated κ_j) version stated as mechanism, flagged honestly.
- **F6 (σ8/w0 anisotropy — honest treatment, no numerics).** Two candidate mechanisms,
  derived where possible, adjudicated only with existing artifacts: (i) amplitude parameters
  live in per-channel scale (what nulling+noise crushes — F3/F5); shape parameters retain
  response in surviving channel ratios; (ii) **prior saturation**: σ(w0) 0.245→0.323 may
  simply be approaching the prior wall (inflation capped by prior width — check against the
  pipeline's prior, an existing artifact, not a new run). Also derive the fiducial-pinned-B
  observation: B is built at θ_fid, so for kernel-moving parameters the nulled channels
  re-light at linear order in δθ (a leakage signal per-channel stats CAN see), while pure
  amplitude scaling keeps the nulled combination nulled. Stated with explicit first-order
  expansion; classified MECHANISM unless the existing artifacts settle it.

**L4 — Survey practice & the joint-PDF program (Part III/IV deepened).**
- **M1 (union-catalog identity, derived with noise).** κ_{i∪j} = (n_iκ_i + n_jκ_j)/(n_i+n_j)
  including the noise bookkeeping: catalog-level pooling and map-level combination give the
  *same* noise realization and variance σ_e²/(n_i+n_j) — full proof ⇒ "catalogs add
  bookkeeping, not field information" becomes a proposition, not a slogan. Masks/weights
  caveat stated.
- **M2 (Cramér–Wold, constructive, order-by-order).** cum_k(Σ w_iκ_i) is a degree-k
  homogeneous polynomial in w with binomial-weighted mixed-cumulant coefficients; Vandermonde
  argument ⇒ k+1 distinct weight ratios determine all order-k pairwise mixed cumulants;
  ≥3-bin mixed cumulants need ≥3-bin unions. **Sharp new accounting of Martinet-style
  practice:** equal-weight pairwise unions + autos determine all order-2 cross structure
  exactly, but leave order-3 pairwise mixed cumulants underdetermined (1 equation, 2 unknowns
  per pair) — survey practice is provably complete at second order and provably incomplete at
  third, with the exact missing directions identified. (Verify the Martinet construction
  details against the paper before asserting which unions they use.)
- **M3 (joint PDF in SBI, concretely).** The estimator (per-scale joint histogram of wavelet
  coefficients), binning arithmetic vs current datavector dims, noise standardization in the
  joint setting (scalar σ_c(s) → matrix Σ'^{1/2} whitening), relation to the LDT joint-PDF
  program, what would be novel. Design-level only (third-pillar campaign needs a separate go).

**L5 — Synthesis.** The two pillars as one statement (basis-adaptivity vs statistic-strength,
asymmetry of the two comparisons kept from Part II); the whitening-test slot (F4 framework +
measured number when it lands); the claims-classification table closing the loop with L0.

---

## Process

1. Plan sign-off (this file) — **waiting on Andreas**.
2. Draft `BNT_THEORY_DEEP_DIVE.md` top-to-bottom; every PROVED tag gets a real proof; every
   formula derived in-line (no "it can be shown").
3. Cross-check every measured number against FLATSKY_BNT_RESULT.md / FLATSKY_CNN_RESULT.md /
   WHITEN_RESULT.md (when it lands; the F4 slot is written to absorb it either way).
4. Then trim PAPER_BNT_MAIN_AND_APPENDIX_DRAFT.md: Parts I–II revised against the deep-dive,
   Parts III–IV replaced by a pointer. Felt stanza + commit (docs only, by path).

## Open questions for Andreas (answer at sign-off)

- **Q-A:** M2 wants one factual check against Martinet et al. (which union catalogs they
  actually build) — OK to WebSearch/fetch the paper for that single detail? (No other
  external sourcing planned.)
- **Q-B:** F6's prior-saturation check reads the prior bounds from the pipeline config — fine
  as an "existing artifact," or do you consider that a numeric and want it prose-only?
- **Q-C:** target length — my default is "as long as the proofs need, no padding" (~600–900
  lines). Cap it?
