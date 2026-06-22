# Paper messages — the spine (living doc; iterate as work proceeds)

Andreas's framing (2026-06-13), with status flags. Two sessions feed this: THIS one
(analytical summary stats — L1, BNT rescue) and a SEPARATE one (CNN optimization,
HANDOFF_CNN_OPTIMIZATION.md). Status legend: **LOCKED** (result + explanation solid,
doc-backed) / **OPEN** (live work) / **VULN** (referee-exposed, needs shoring up).

---

## M1 — With matched best NDE, analytical ℓ1+product ≈ the optimal CNN; the CNN's lead is the estimator, not physics. [to-LOCK]
**REWRITTEN 2026-06-15 (NDE-matched, every arm gated). Supersedes the common-MAF numbers below
and the definitive_comparison violin figures (now stale — see "Outdated plots").**

The earlier "CNN does not outperform L1" rested on a COMMON jaxili-MAF NDE that under-served the
CNN (CNN-MAF ~2300–2620). With the CNN's NDE properly optimized — **ResNet18 + sbi_lens RealNVP
4×128 → FoM3 3293** (σ 0.045/0.072/0.229), calibrated (TARP+SBC) — the optimized CNN DOES beat the
common-MAF L1+product (2875) by ~15%. **But that gap is an ESTIMATOR effect, not physics:** give the
calibration-clean ℓ1+product the SAME pipeline (VMIM-compress to 10-D → the SAME sbi_lens RealNVP)
and it reaches **FoM3 ~3100–3270**, calibrated:
- **Population-median FoM3 (robust, matched n=9000):** l1+product **3045** vs CNN **3326** — CNN ~9%
  higher (with the 5-seed ensemble at n=1000, l1 ≈ 3173 → ~5%). σ(Ωm,σ8,w0) l1 0.048/0.077/0.229 vs CNN
  0.045/0.072/0.231 ⇒ **σ(w0) matched, Ωm/σ8 within ~7%.** (The earlier "tie at 3270≈3293" was l1's
  noisier n=1000 screens running high; the n=9000 medians are the reliable ones — quote these.)
- **Noiseless mean-observation:** **CNN ~3250–3280 vs l1 ~2770–2840** (CNN ~10–15% tighter); both
  UNBIASED. l1's lower noiseless constraint (≈ clean raw-MAF 2875) vs its higher median reflects its
  larger per-patch realization scatter ([[project_l1_patch_sensitivity_full200]]).
- **Calibration (Andreas 2026-06-15 — judge OVERALL, not per-tercile):** l1+product→RealNVP is
  CALIBRATED for the paper — **pooled TARP within ±0.05 (net +0.001 = perfect joint coverage), SBC
  marginal-std 0.30 in-band**; the CNN is comparably imperfect (net +0.030, conservative). The
  per-tercile HIGH worst-dev (~0.07) is a stricter diagnostic, NOT a paper blocker; an optional
  disjoint-split / conformal pass would erase it but isn't needed. Everything is within the bands.

**Mechanism (the lever, isolated):** on the IDENTICAL 10-D ℓ1+product summary, jaxili-MAF = 2426 vs
sbi_lens-RealNVP = 3146 (+30%) — exactly mirroring the CNN's own MAF 2312 → RealNVP 3293. RealNVP on
the RAW 2000-D ℓ1 craters (1111), so the VMIM compression is what unlocks it. Controls: ℓ1-auto (no
cross) → RealNVP = 2448, calibrated, does NOT inflate ⇒ the gain is the cross ξ_ij info, not a generic
artifact; pair2d (over-confident raw) → RealNVP = 4864 but GATE **FAIL** (fool's gold, correctly
rejected). **Different NDEs per probe are fine given each is independently gated** (Andreas).

**Headline (referee-proof):** with best-effort training on BOTH sides — we optimized the CNN
(arch+NDE+convergence) AND gave L1 the same NDE — the optimal CNN's advantage over the analytical
wavelet ℓ1+product is **small (~5–9% on the population-median FoM3, ~10–15% on the expected
observation; σ(w0) matched), calibrated, and is an estimator/full-field effect, not a representation
gap ⇒ ℓ1+product is near-sufficient.** This
RESOLVES the VULN (the gap is small, calibrated, and understood). Backing:
analytical_nde_match/RESULT_ANALYTICAL_NDE_MATCH.md (matrix + sweep + overlays), fom3_matrix.png,
tarp/sbc/contour overlay PNGs; FLATSKY_CNN_RESULT.md; HANDOFF_CNN_OPTIMIZATION.md.
Memory: [[project_analytical_matches_cnn_via_nde]], [[project_cnn_nde_swap_resolves_m1]].

### Joint ℓ1 update (2026-06-21) — analytical now MATCHES the CNN, calibrated + seed-robust
A richer analytical statistic closes the small remaining gap. The wavelet **joint ℓ1** (histogram
of the across-channel coefficient vector, cells holding the ℓ1 sum — the *complete* cross-correlation
statistic; products κ_iκ_j are only its 2nd-moment ξ_ij slice) through the SAME matched pipeline
(VMIM→10-D→sbi_lens RealNVP) reaches **FoM3 3754 (n=9000), gate PASS** — vs ℓ1+product 3045 and CNN
3326. So **analytical (joint ℓ1) ≈ the optimized CNN, calibrated** (σ 0.043/0.069/0.220 ≈ CNN
0.045/0.072/0.231; pooled TARP net −0.003, within ±0.05). Seed-robust: 3 compressor seeds = 3754 /
3761 / 4034, all PASS-with-caveat (band mean 3850, ~7%). A clean *completeness/calibration*
trade-off bounds it: pushing the joint further (full4d 4501, pair2d 4864 — count histograms) buys raw
FoM3 but FAILs the gate (over-confidence); the continuous ℓ1-weighted joint ℓ1 is the calibrated
sweet-spot. ⇒ the joint captures the cross-correlation that products miss, and the CNN's earlier
~9% edge over ℓ1+product is closed by a *fixed analytical statistic*. (BNT/Q2: joint ℓ1 is far more
BNT-robust in raw FoM3 — 0.86 vs ℓ1+product's 0.26 — but the BNT-frame estimate is over-confident
(gate FAIL); captures the cross-corr, not yet *calibratedly* BNT-lossless.) Backing:
analytical_nde_match/RESULT_JOINT_MATCHED.md (+ RESULT_JOINTL1_SEEDCHECK.md); figures
violins_jointl1_3arm, violin_fom3_jointl1_3arm, tarp/sbc_pooled_jointl1_3arm,
contour_jointl1_vs_l1product_vs_cnn. Memory: [[project_joint_l1_matches_cnn]].

#### Calibrated update 2026-06-22 (compressor ensemble — the number to quote)
The single-compressor joint ℓ1 (3754) was ~10% over-confident (SBC 0.31, caveat). A **3-compressor
deep ensemble** (pool the 41/42/43 compressor posteriors per obs — principled, non-conformal)
calibrates it to a **clean PASS** (TARP net −0.005, SBC 0.30) and gives the corrected FoM3 **3371**
(σ 0.044/0.072/0.223) = a **calibrated TIE with the CNN (3326), σ matched, no caveat.** Removing the
over-confidence brings the inflated 3754 down onto the CNN — physically sensible (the CNN is ~optimal;
"analytical > CNN" was the over-confidence, now gone). **Quote 3371 ≈ CNN, not 3754.** The capacity
sweep does NOT fix the over-confidence (flat across flow capacity) — it's amortization leakage, fixed
by the ensemble. Backing: RESULT_JOINTL1_ENSEMBLE.md; figs violins_ensemble_3arm,
tarp/sbc_pooled_ensemble_3arm, contour_ensemble_nobnt_3arm.

### Outdated plots (regenerate / drop before the paper)
The pre-NDE-swap comparison figures show the stale "CNN ~2300, L1 wins/ties on autos" picture and
must NOT go in the paper. Known stale: `definitive_comparison/fiducial_full200/figures/
headline_typical_patch_violins.png` (and the other definitive_comparison/violin/per-patch figures
built on the common-MAF CNN). The CURRENT, PROPER comparison figures are the
analytical_nde_match overlays: `fom3_matrix.png`, `tarp_overlay_l1_vs_cnn.png` /
`tarp_pooled_l1_vs_cnn.png`, `sbc_overlay_l1_vs_cnn.png` / `sbc_pooled_l1_vs_cnn.png`,
`contour_overlay_meanobs_l1_vs_cnn.png`. Audit `figs/` for any FoM3-bar/violin that cites CNN<2900.

## M2 — Designed cross-maps: convolution doesn't help, product modestly does, and we know why. [LOCKED]
Conv +4% (≈0 de-leaked), product +20% (L1). Three-leg explanation: (i) the conv map is a
lag-space re-encoding of TWO-point information; (ii) CLT-compressed to a few effective modes
on a 10° patch; (iii) Zürcher reconciliation — their cross-bin gains are
IA-self-calibration-dominated (Table 3: σ(A_IA) −104%..−430% without cross-bins) and our
forecast has NO IA, so the dominant literature channel is absent by construction. Lead with
the IA leg. Backing: FLATSKY_CROSS_RESULT.md.

## M3 — BNT: L1 contours inflate, CNN is lossless, both explained. [LOCKED]
L1-auto BNT/noBNT 0.15× (collapse); CNN 0.93× (lossless within seed scatter). Whitening test:
one fixed rotation of the nulled maps recovers the full no-BNT FoM (1.06/1.01) ⇒ no
irreducibly-joint loss; the collapse is a per-channel-shadow / no-deep-direction frame
artifact. Backing: FLATSKY_BNT_RESULT.md, BNT_THEORY_DEEP_DIVE.md (proofs P1–P7, P4c).
**Matched best-NDE confirmation (2026-06-15):** in the M1 setup (VMIM→sbi_lens RealNVP, gated),
l1+product BNT/no-BNT = **0.26×** (COLLAPSE; 779 vs 3045, calibrated — pooled TARP net +0.005, SBC
std ~0.31, so the wide BNT contours are a REAL info loss not over-confidence) while CNN ResNet18
BNT/no-BNT = **0.96×** (LOSSLESS, 3186 vs 3326). The collapse is NDE-independent: the per-channel L1
discards the cross-channel info BEFORE the VMIM MLP, so even the CNN's own NDE can't recover it; the
channel-mixing CNN mixes channels before the summary. Figures: analytical_nde_match/contour_bnt_*,
bnt_fom3_bars_l1_vs_cnn, tarp_pooled_bnt_l1_vs_cnn. RESULT_ANALYTICAL_NDE_MATCH.md Addendum c.
**Joint ℓ1 under BNT (2026-06-22, calibrated ensemble):** the wavelet joint ℓ1 is far more
BNT-robust than the per-channel ℓ1/products — calibrated retention **0.72** (BNT 2424 / noBNT 3371,
3-compressor ensemble, clean PASS both bases) vs ℓ1+product's **0.26** and CNN's **0.96**. So the
joint captures ~3× more of the BNT-surviving cross-correlation than products, but is NOT fully
BNT-lossless: the 0.72→0.96 gap is the full-4-D channel mixing the CNN learns, which a fixed
(pairwise) analytical statistic cannot reach (the shear-aware rotated-grid binning did not close it —
RESULT_JOINTL1_ROTATED.md). The single-arm raw retention 0.86 was over-confidence-inflated; calibrated
it is 0.72. Backing: RESULT_JOINTL1_ENSEMBLE.md; figs contour_ensemble_bnt_3arm,
tarp/sbc_pooled_ensemble_3arm.

## M4 — Make L1 work in BNT space via map combinations. [PARKED 2026-06-13 — NOT in paper for now]
**Decision (Andreas 2026-06-13):** result is gated + schedule-tested but modest in the
realistic regime (B2/B3 ≈ 1.07× at the light cut) — not interesting enough to feature yet.
Keep it in the back of the mind (the one nugget worth recalling: per-channel L1 collapses
under cut-BNT and the B⁻¹-reconstructed-deep recombination restores it to ~the standard
uniform-cut level while PRESERVING clean cuts — i.e. BNT's value is the clean cuts, not a
FoM3 win). Do NOT write into the paper or point compute here for now. Full record below +
M_VS_L_ROBUSTNESS.md. Memory: [[project_flatsky_bnt_losslessness]] addendum.

Rescue ladder (uncut): +deep 0.73, +deep2 1.08, +unions6 1.18, whiten 1.06. Post-cut frontier
(lane B), now GATE-C'd and tested across two cut schedules (M moderate, L light/realistic):
- Per-channel L1 in cut-BNT space COLLAPSES (B0: 0.79× the uniform-cut noBNT analysis at M,
  0.19× at L). Adding two B⁻¹-reconstructed deep channels (cut-then-mix, preserves the
  per-slice cuts) RESCUES it — calibration-clean (all L arms PASS; M rescue arms PASS-w-caveat).
- BUT the ADVANTAGE over a standard uniform-cut analysis is schedule-dependent: B2/B3 = 1.82×
  at the aggressive cut M (which craters the uniform analysis to 337), only **1.07× at the
  realistic light cut L** (uniform analysis = 1880 ≈ 78% of uncut, little to win back).
  The strong "~1.8×" does NOT generalize.
- "Plain sums suffice" was a moderate-cut artifact: at L, reconstructed-deep (2007) ≫ plain
  sums (1286, BELOW the uniform analysis) — the reconstruction matters, not generic sums.
**Honest message:** the reconstructed-deep recombination rescues the cut-BNT L1 from collapse
to ~the standard uniform-cut level (within ~10%) WHILE preserving BNT's clean per-slice
systematics control — BNT's value is the clean cuts, not a raw FoM3 win; schedule-robust in
direction (rescue works), not in magnitude. The gain is NOT over-confidence (gated clean,
contrast M5/A1). Backing: gate_c_laneB/GATE_C_LANEB.md, gate_c_laneB_L/M_VS_L_ROBUSTNESS.md,
PLAN_M4_GATE_C.md, BNT_THEORY_DEEP_DIVE.md §1.7 item 7.

## M5 — Joint one-point statistics (A1 etc.) — keep in mind, NOT priority. [parked]
Defensible version: pairwise joint PDF of wavelet coefficients reaches ~l1+product level
from AUTO maps alone (cross-map info accessible without building cross maps). The FoM3 ~3.5k
"new best" headline does NOT survive (DPI: A1 = compression of pair2d can't add info, yet
higher FoM3 ⇒ estimator-path artifact; fiducial marginals only tied-to-~10% over l1+product;
mild Ωm/σ8 over-confidence). At most a discussion/appendix note. Backing: LANE_A_CONCLUSION.md.

---

## Cross-cutting methodological note (applies to M1, M4, M5)
FoM3 differences of ~20–30% between methods in our setup are partly NDE/estimator quality,
not physics (DPI argument; the K-trend 2874/2794/2455 decreasing with finer binning is the
clean demonstration). Robust claims are the **calibrated, marginals-level** ones. For any
"statistic A vs statistic B" ranking that matters, fix one NDE + training budget + a
convergence diagnostic and run both through it.

## Dataset facts (settles the "undertraining?" question, 2026-06-13)
Flat-local training cache: **323,640 patch examples / 899 distinct cosmologies / 360
patches per cosmology** (180 patches × 2 perms). Val: 144,000 / 400 cosmologies. Grid total
= 2500 cosmologies (CosmoGridV1). Reading: 324k patches is ample for the COMPRESSOR (feature
learning / augmentation) ⇒ data scarcity is NOT the compressor's problem. The 899 cosmologies
cap the theta-dependence the NDE can learn, but L1 and CNN face the SAME 899, and L1 does
better ⇒ the CNN's deficit is architecture/optimization/flow, not data. (Corrects an earlier
"~70k sims" mis-statement.)
