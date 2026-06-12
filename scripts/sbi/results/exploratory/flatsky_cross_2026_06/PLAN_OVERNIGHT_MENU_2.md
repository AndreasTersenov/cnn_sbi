# PLAN — Overnight menu 2 (2026-06-12 → 13): every arm pre-commits a sentence per branch

**Status: awaiting Andreas's sign-off** (§8). **Revision 2** after Andreas's anti-circling
directive: 20° lane DROPPED (his call); full4d-K5 DROPPED (failed the branch test below);
lane C reframed as one curve with a message; anti-stuck fences added.

**The admission filter (applied to every arm):** an arm runs tonight ONLY if the verdict
sentence for EVERY outcome branch is written in this plan before launch. If a branch
wouldn't change what we say in the paper or what we do next, the arm is aimless and is cut.
Each arm below lists its branch sentences — they are the pre-registered verdict templates;
the night's report fills in numbers, never invents readings.

**Pattern:** as PLAN_OVERNIGHT_MENU.md (slot workers GPUs 0/1/2, politeness probes,
screening 1-seed/3000-obs → auto-escalation 3-seed/9000-obs, derived verdicts only).
Baselines (full rigor): l1-auto 2405 (σs8 .082) | l1+product 2875 (.075) | l1-BNT 364 |
pair2dq_nobnt 2794 | jointl1q_nobnt 2788 | pair2dq ratio 0.52 | full4da ratio 0.70.
GATE-C state: pair2d HIGH-tercile −0.134 (FAIL), jointl1 −0.080; SBC std 0.30–0.31.

**Phase 0 (standing decision):** Tier-1 packing benchmarks ride this campaign first
(PLAN_PACKING_BENCHMARKS.md, 3-pack only); result configures the night's own packing.

Priority: **A > B > C > D.**

---

## Lane A — Is the joint-stat miscalibration a feature-space artifact?

Decides third-pillar vs discussion-section. Not circular by construction: it does not
re-measure anything — it explains this morning's FAIL and decides a paper-structure choice.

**A1 `pair2d_vmim`** — VMIM MLP on the pair2dq_nobnt cache → 10-d summary → MAF (3 seeds)
→ full sweep + AUTO-GATE (tarp_stratified_val on the 10-d arm).
Branch sentences (pre-committed):
- calibrates (worst-tercile |dev| ≤ 0.05) AND FoM3 ≥ 0.9×2794 → *"the joint PDF's
  miscalibration is a high-dimensional count-feature artifact; compressed, the statistic is
  calibration-clean at full constraining power — the parity claim is rehabilitated (quote
  from the VMIM arm)"* → third-pillar option REOPENS.
- calibrates but FoM3 < 0.9×2794 → *"the nominal edge was partly miscalibration; the
  calibrated version of the statistic sits at [X]; downgrade is final"* → discussion section.
- still |dev| > 0.05 at 10-d → *"the miscalibration is not dimensionality; it tracks the
  statistic/posterior geometry itself"* → discussion section + pitfall paragraph.
Fallback fence: if screening (1 seed) yields no usable summary in the 2 h time-box, arm
auto-skips; A2+A3 carry the lane (their branches still decide the mechanism question).

**A2 `pair2d_k8`** — rebuild pair2d K=8 (1920-d, dq) → sweep + AUTO-GATE.
- worst dev shrinks ≥30% (toward jointl1's −0.080) → *"the tail miscalibration is driven by
  sparse-cell occupancy; coarser grids are the calibratable regime of joint histograms"* (a
  practical recipe, quotable).
- dev unchanged → *"sparsity is not the driver; count features through a MAF carry an
  intrinsic tail-calibration cost"* (sharpens the pitfall).
Bonus read: K=8 joins K=10 (measured) and C2's K=15 in the K-trend (lane C).

**A3 `pooled_tarp`** — CPU-only, zero GPU: reprocess the EXISTING gate dumps, pooling the
3 seeds per point (the pooled posterior IS the quoted estimator) → one TARP curve/tercile.
- pooled HIGH dev ≤ −0.05 still → *"the over-confidence survives pooling; the FAIL stands
  for the quoted estimator"* (verdict hardens, no ambiguity left).
- pooled clean (|dev| < 0.05) → *"the worst-seed gate over-penalized; the quoted pooled
  posterior is calibrated — the downgrade softens to a per-seed caveat"* (report gains a
  pooled-estimator column; GATE_C_JOINT.md gets an addendum).
Either way the gate report's wording changes — the rare arm where both branches edit a doc.

## Lane B — The post-cut frontier (the genuinely new science)

The survey question none of our uncut results answer. Every branch lands a novel, quotable
statement, so the lane passes the filter at every rung.

**Cut schedule (NEEDS SIGN-OFF — the one physics input).** Wavelet scales s1(finest)…s5;
BNT channels shallow→deep. **Schedule M:** κ̃₁ keeps {s4,s5}; κ̃₂ {s3..s5}; κ̃₃ {s2..s5};
κ̃₄ all. **Comparator U** (the uniform cut a noBNT analysis needs to reject the same low-z
small-scale contamination): all 4 original maps keep {s4,s5}. Variant M′ = M + drop κ̃₁.
Anti-arbitrariness note: the qualitative conclusion (collapse persists / sums partial /
reconstructed-deep recovers-or-not vs U) is direction-robust to the schedule; the NUMBER is
schedule-conditional and will be quoted as such. Schedule-robustness check rides a future
session only if the result is interesting — not tonight (fence).

| arm | construction | branch sentences |
|---|---|---|
| B0 `bntcut_l1` | feature-mask BNT l1 cache to M | baseline/denominator; if NOT collapsed (≥0.5×B3) → *"cuts remove mostly what per-channel analysis couldn't read anyway"* — itself interesting |
| B1 `bntcut_sums` | + 6 pair-sums of M-cut channels | lands between B0 and B2 → *"two-slab kernels are partially deep"* ; ≈B2 → *"plain sums suffice — no B⁻¹ weighting needed"* (simpler survey recipe); ≈B0 → *"pair sums are not deep enough; direction design matters"* |
| B2 `bntcut_deep2` | + 2 B⁻¹-weighted reconstructed-deep channels from kept cut content | THE test — read against B3 |
| B3 `nobnt_unicut` | feature-mask noBNT cache to U | comparator; its gap to uncut 2405 is also a quotable number ("what the uniform cut costs") |

B2/B3 branch sentences: ≥ 0.9 → *"BNT + two cleaned recombinations costs ≤10% of the
information while retaining per-slice systematics rejection — the constructive resolution
of the BNT trade-off"*; 0.75–0.9 → *"the trade-off is real and now quantified: [X]% info
for clean cuts"*; < 0.75 → *"post-cut recombination cannot rescue per-channel statistics;
the BNT information cost in survey practice is substantial"* — honest negative, still the
paper's discussion anchor. Primary metric: FoM3(B2)/FoM3(B3), marginals-first read.

Deferred (named): CNN-on-cut-BNT ceiling arm (map-space band-cut machinery; a follow-up
once B's direction is known).

## Lane C — one curve, one rebuttal (was: P4c decimal-polishing; re-justified)

Reframed after the filter. C is NOT more P4c decimals; it is (i) the K-trend curve that
turns three point measurements into one statement, and (ii) the measured answer to the
inevitable referee question "why not just bin finer?".

**C1 `pair2d_bnt_ar`** — pair2dq BNT with `--adaptive-ranges` (the REGISTERED-unrun band:
0.52 < r < ~0.75).
- r ≈ 0.70 → *"placement explains the same majority for pairwise as for full-4D"* (P4c
  table complete).
- r ≈ 0.55 → *"pairwise statistics are structurally more basis-fragile: marginal
  incompleteness, not placement, dominates"* (sharpens the not-closed-under-mixing point).

**C2 `pair2d_k15_nobnt`** (6750-d, dq) — is the noBNT ceiling resolution-saturated?
- within ±5% of 2794 → *"K=10 is the saturated regime; the parity comparison is
  resolution-robust"* (robustness line for the joint-stat section).
- > +5% → *"the joint-stat ceiling is higher than quoted"* → re-opens the parity question
  upward; lane A's calibration answer then applies to the K=15 arm tomorrow.
With A2 (K=8) and the measured K=10: the **K-trend** [FoM3(K), worst-dev(K)] for K∈{8,10,15}
— the info-vs-calibration trade-off of histogram resolution as one quotable curve.

**C3 `pair2d_k15_bnt_ar`** — the staircase test (same K=15 build as C2; marginal cost).
- r(C3) − r(C1) ≥ 0.05 → *"finer K visibly staircase-approximates the shear; finite-K is
  the binding constraint"*.
- flat → *"the shear residual is K-stubborn; only a learned linear front-end transports"*
  (the referee rebuttal, measured).

CUT by the filter: full4d K=5 adaptive (no branch changes any claim — both outcomes are
"mildly nice"); GM-vs-product screen (information argument already airtight); any K > 15;
second cut schedules; VMIM-jointl1 (rides tomorrow ONLY if A1 rehabilitates).

## Lane D — order-3 closure (Andreas's triplets, with a fence)

**D1 `l1_product3`** — 4 triple-product maps + 1 quadruple as l1 channels appended to
auto+product, noBNT; new `product3` op + empirical σ freeze (established pattern).
- within seed noise of 2875 → *"the accessible cross-bin information is pairwise-saturated:
  measured through third order"* (the completeness sentence his Q3 deserves — closes the
  triplet idea with a number, not an argument).
- > +5% with every science marginal ≤ → *"order-3 pointwise information is accessible"* —
  genuinely new thread; goes to the morning session as a finding, NOT extended tonight.
Fence: no triplet variants, no order-4 follow-ups tonight regardless of outcome.

---

## Anti-stuck fences (the meta-rules for the night)

1. Branch sentences above are the ONLY verdicts the night may emit (numbers filled in).
2. No new arms after launch; anything interesting goes to the morning list.
3. Time-boxes: A1 screening 2 h (then auto-skip, lane carried by A2/A3). Everything else
   uses measured anchors (build 200–400 s, screen ~5 min, full ~15 min, gate ~10 min/arm).
4. Auto-skip on any builder assert failure — zero overnight debugging; failed arm = logged
   + morning item.
5. Done condition: all arms verdict-or-skipped OR 07:00 UTC. No loops, no retries beyond
   `train_with_nan_retry`'s built-ins.

## Execution

Build phase (~2–3 h, before launch, py_compile + unit asserts + theta/perm/patch
bit-equality hard asserts against parent caches): masked-cache script (B0/B3), mix modes
`cutsum6`/`cutdeep2` with per-channel scale masks + σ rows √(M²σ²) (B1/B2), `vmim_from_cache.py`
glue (A1), `product3` op + σ freeze (D1), pooled-TARP reprocessor (A3, CPU). Arms via
`run_flatsky_overnight_menu.py` ARMS-dict extension (slot workers, 40% caps, 12 GB
foreign-tenant back-off, ≥1680 screening-FoM3 escalation; AUTO-GATE chained for A1/A2).
Fresh tenant check at launch; GPUs 0/1/2, GPU 3 never, ≤50 CPU workers. Detached + Monitor.
Verdicts → OVERNIGHT2_RESULT.md (derived); felt stanzas at launch + completion; doc folds
wait for the morning.

Arm count: 10 GPU arms + 1 CPU arm + phase-0 benchmarks — comfortable for 3 GPUs at the
anchors; the only unmeasured item left is VMIM (fenced).

## §8 — Sign-off

1. **Cut schedule:** M + U as specified? M′ (drop κ̃₁)? Or your own keep-masks?
2. **A1 time-box** 2 h OK?
3. Anything above you'd cut further?

**SIGNED OFF (Andreas, 2026-06-12 evening): "perfect, do all that autonomously" + drop
the 20° lane.** Defaults adopted: schedule M + U; A1 time-box 110 min in-script (+3 h
driver guard).

**Execution addenda (autonomous judgment calls, logged for the morning):**
- **A3 ran pre-launch (CPU)** and RESOLVED: pooled-estimator TARP HIGH devs −0.106 /
  −0.079 (pair2d/jointl1 noBNT) ⇒ over-confidence SURVIVES pooling; the gate verdict
  stands for the quoted estimator (GATE_C_JOINT.md Addendum 2).
- **Packing bench embedded on real workload**: B1-spec (3 solo reps of the B3 screen, then
  the same arm ×3 concurrent on one GPU); B2/B3 compressor benchmarks DEFERRED (no
  compressor workload tonight). Identical thread caps (5) in solo, pack, and deployment.
- **C2 also gated** (tarp_stratified_val) — the K-trend table [FoM3(K), worst-dev(K)] for
  K ∈ {8, 10, 15} needs the K=15 calibration point; plan text listed gates only for A1/A2.
- **Always-escalate**: every arm that screens OK goes to full rigor — each arm's branch
  sentences need full numbers; screening = smoke test + bench workload (the ≥1680 rule
  would never escalate the deliberately-collapsed B0).
- **Sweep thread caps 5** (9 packed jobs × 5 = 45 ≤ 50 CPU budget); builds at 8.
