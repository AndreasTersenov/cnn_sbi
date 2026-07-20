# Paper narrative & pitfalls — the journey, the dead ends, and the traps

**Stage-2 companion to `PAPER_SCIENTIFIC_SYNTHESIS.md`.** This is the *journey* document: the
chronology of what was tried, the approaches that failed and **why**, and a community-facing **pitfalls
catalog**. Andreas's framing (memory `project_paper_narrative_includes_journey`): the unusually
thorough ruling-out of confounds is itself a contribution — it is why the final result is trustworthy,
and it warns others about real traps. Material is mined from the triage's SUPERSEDED/WRONG rows
(`PAPER_FILE_TRIAGE.md`) and the chronology in `EXPERIMENT_AUDIT.md` (whose *journey* content is the
backbone even though its final science *verdict* is superseded).

**How this feeds the paper (per Andreas's hybrid choice):** some of this is **woven** into Methods/
Results as "what we ruled out"; the **reversal** (§2) is a candidate **standalone methodological
section**; the **pitfalls catalog** (§4) and **exhaustive-search evidence** (§5) are candidate
**appendices**. Final placement is a paper-draft decision; this doc supplies the raw material for all
three.

---

## 1. Timeline — how the project actually unfolded

Five development phases (from `EXPERIMENT_AUDIT.md`, which cataloged 1,517 runs across 190 dirs), plus
the June definitive work that postdates the audit.

- **Phase 0 — origins (Jul–Aug 2025).** Fork of Justine Zeghal's Learn2Map. The experimental template
  is set: CosmoGridV1 maps → CNN-VMIM compressor → conditional RealNVP NDE → posterior contours. The
  20°/160px field and 4-bin tomography are established; the **first BNT analysis** appears in the first
  week of systematic work. Then a ~6-month gap (thesis/teaching).
- **Phase 1 — systematic pipelines & the BNT campaigns (Feb–Apr 2026).** The L1-norm script
  (2026-02-18), the CNN script (2026-03-17), and crucially the **jaxili MAF** scripts (2026-03-29) —
  *this is where the NDE architecture silently splits*: CNN defaults to sbi_lens RealNVP, L1 to jaxili
  MAF. L1-VMIM follows (2026-04-01). The BNT-losslessness campaign runs. **2026-04-22: `--zero-mean-
  maps` is introduced** — the dividing line of the whole project: **1,225 runs (81%) predate it and are
  mass-sheet-contaminated.**
- **Phase 2 — L1 development & benchmarking.** The no-BNT tomographic cross-correlation studies (the
  G_corr attribution), the optimal-no-BNT benchmark. (All pre-zmm; directional only.)
- **Phase 3 — cross-maps (May 2026).** Harmonic full-sphere cross-maps; the **"L1 wins ~2.5–4.6×
  auto+cross"** headline; the **noise-model bug** found and fixed (2026-05-15,
  `channel_empirical_global`); the cross-only campaigns (v1/v2); the **BatchNorm-contamination**
  discovery and the `resnet50_gn` fix; the three falsified hypotheses about why the CNN underperforms
  on cross (§5).
- **Phase 4 — autoresearch campaigns (from 2026-05-18).** `cnn-auto-push` (auto-only; a certified
  ceiling) and `cnn-auto-cross-push` (**115 iterations that never closed the gap to L1** — §5).
- **Phase 5 — canonical refresh & the NDE-swap session (2026-05-26/27).** The **NDE-architecture
  mismatch** and **FoM3 fragility** are named for the first time; the first fully-clean (zmm+disjoint)
  CNN posteriors are produced.
- **June 2026 — the definitive run & the leakage finding.** The 10° common-MAF campaign (Phase C/D,
  2026-06-07) **overturns the L1-wins headline**; the prior-shrinkage / sharpness-vs-calibration
  diagnostics; and the **cross-map leakage finding** (2026-06-08) that makes auto+cross provisional.

---

## 2. The reversal — "L1 wins 3×" → "CNN ≥ L1" (the paper's spine)

The single most important narrative: a strong-looking result that was dismantled step by step. Each
step is a SUPERSEDED/WRONG row in the triage; together they are *why* the final answer is believable.

1. **The headline (≈ early–mid May, 20°, full-sphere harmonic cross, v1 noise).** L1 appears to beat
   CNN by **~2.5–4.6×** on auto+cross, concentrated in w0 (e.g. FoM3 56,602 vs 12,421;
   `HARMONIC_L1_VS_CNN_INVESTIGATION_BRIEF.md`, `…SESSION2_HANDOFF.md` — both **WRONG** in the triage).
2. **Noise-model bug (2026-05-15).** The L1 cross datavector used the **auto pixel-σ for the ~10⁴×
   smaller cross channels**, collapsing their wavelet SNR to ~0 and *inflating* L1 auto+cross. With the
   channel-aware fix: L1 auto+cross ~65k → ~38k (−48%); cross-only 12k → ~16–18k (+33%). The "wins 3×"
   shrinks toward ~1.5×. (memory `project_l1_noise_model_correction`.)
3. **Cross-only control.** On the 6 cross channels *alone*, **CNN beats L1** (~26.6k vs ~18.1k) ⇒ L1's
   auto+cross advantage was **combinatorial**, not an intrinsic superiority on cross-map signal.
4. **FoM3 fragility & the NDE confound (NDE-swap, 2026-05-27).** The *same* compressor through RealNVP
   vs MAF gives a **47% FoM3 gap** but only ~5% marginal-width and ~15–20% 2D-area differences ⇒ FoM3
   amplifies correlation noise; and L1-vs-CNN had been compared through **different flows** all along.
5. **Perm-averaging (2026-06-01).** The favorable perm-0 L1 draw (FoM3 34,607) averages down; per-perm
   L1 25,808 vs a matched CNN 28,093 ⇒ **CNN ≳ L1**, L1 keeps only a perm-fragile σ(w0) edge.
6. **Patch population & geometry (2026-06-02/03).** Over many patches L1 ≈ CNN on robust marginals
   (the 2× FoM3 is amplification); L1's spread is ~92% realization / ~8% geometry (just polar patch-0);
   L1 carries a coherent **−0.37σ w0 offset** at 20°. The 10° run is set up as the decisive test.
7. **The definitive 10° run (2026-06-07).** Both compressors through a **common MAF**, robust 9000-obs
   median: **CNN ≥ L1** (auto-only tie; CNN ahead on auto+cross), and **L1's w0 offset shrinks
   −0.37σ→−0.10σ and is no longer L1-specific** ⇒ a **flat-sky artifact**. All calibration clean.
8. **The leakage resolution (2026-06-08).** *Why* does CNN gain so much from explicit cross-maps? Because
   the full-sphere construction **leaks global modes** into each patch — info the local autos don't
   contain and the small-scale ℓ₁ can't read. This makes auto+cross **partly unphysical** ⇒ PROVISIONAL.

**The point for the reader:** none of the "L1 wins" supersession rests on the relaxed flags — it was
driven by a hard noise-model bug plus a fair, common-NDE, more-geometrically-valid re-measurement.
That is the rigor that makes "CNN ≥ L1" trustworthy.

---

## 3. Approaches tried and why they failed or were abandoned

A structured catalog (each is paper-usable as "we tried X; it failed because Y"). Most are
methodology lessons; a few are the dead-ends that, having been ruled out, harden the final result.

| approach | what happened | why it failed / lesson | status |
|---|---|---|---|
| **PCA on the ℓ₁ datavector** | FoM3 cratered ~5× (16k → 3k) | The ℓ₁ histogram bins are not redundant; PCA discards cosmology-bearing tail structure. **Never PCA the ℓ₁ vector.** | Abandoned (hard rule) |
| **Stock BatchNorm ResNet on 10-ch harmonic input** | FoM3 collapsed to ~700 | BN running statistics average across cosmology-mixed batches, erasing the signal. **Use GroupNorm (`resnet50_gn`) or plain CNN** on multi-channel input. | Fixed (GroupNorm) |
| **L1 cross via the TFDS `--cross-maps` route** | 4× FoM3 crater; a warning, then silent fallback | The channel-aware noise model is only wired to the harmonic-cache route; TFDS silently reverts to the broken auto-σ. **Use `--full-sphere-cross-cache`.** | Fixed (route discipline) |
| **Deeper/bigger CNN (resnet50_gn @120k steps)** | FoM3 11.8k vs plain-CNN 19.5k; val-loss argmin at 35%, then drift | The CNN is **data-limited, not capacity-limited** (~70k training cosmologies). Depth overfits. | Falsified (ceiling) |
| **Cross-channel attention block (H1)** | Pool/MoS haircut 0.684 vs 0.685 (Δ≈0) | Inductive-bias tweak does not help the CNN extract cross-channel info; seed-to-seed mode drift is structural. | Falsified |
| **More cosmologies (H2)** | Determined untestable with the current sim suite | Scoping only; the data-limit could not be probed without new sims. | Scoped out |
| **Higher summary dim (cdim=100, H3)** | FoM3 12.2k vs 24.0k anchor (−49%) | A bigger bottleneck *hurt* — the compressor is not bottlenecked by dimensionality. | Falsified (opposite) |
| **Capacity-matched CNN arch on harm-cross** | FoM3 floor 15–17k vs ~25k baseline (multi-checkpoint) | Matching capacity to the auto-only winner did not recover auto+cross performance. | Falsified |
| **Beefier MAF VMIM companion flow** | Worse than the default RealNVP companion across all seeds | The companion-flow quality is **not** the CNN bottleneck (lower VMIM loss yet worse FoM3 — val-loss ≠ FoM3). | Closed |
| **Harmonic-TFRecord CNN training path** | ~1 it/s under node load; over-engineered | CPU-contention-bound; not a drop-in (global shuffle shifts VMIM). | Judged not worth it |
| **Summary standardization ON (BNT)** | BNT/no-BNT FoM3 retention 0.095 (catastrophic) | Standardization destroys the BNT-specific summary structure. **Turn it OFF** → ~0.79–0.91. | Lever found |
| **Noise curriculum (BNT)** | Hurts plain CNN (0.757 vs 0.914); helps ResNet18 | Best config is **plain CNN, no curriculum**. | Mapped |
| **The 115-iteration `cnn-auto-cross-push` campaign** | ~500 GPU-h; FoM3 never moved off ~25k toward the L1 ~38k target | It was **chasing a phantom** — the "gap" was the noise-model + leakage + FoM3 artifacts, not a tuning deficit. Its *failure to close* is strong evidence the gap was not real. | Superseded by the reversal |

---

## 4. The pitfalls catalog (community-facing — the paper's appendix)

Each entry: **the trap → its signature → its magnitude → the fix.** These are real, mechanistically
understood traps in SBI weak-lensing pipelines; most cost the project days-to-weeks. (The first three
are *hard* invalidators in the triage screen; the rest are flags or process traps.)

### P1 — Mass-sheet (mean-convergence) leakage *(hard)*
- **Trap:** if convergence maps are not spatially demeaned, a CNN compressor learns the
  **mean-convergence level**, which is **not observable** in a real survey (mass-sheet degeneracy).
- **Signature:** `zero_mean_maps` absent/False; CNN posteriors implausibly tight; CNN inflated far
  more than ℓ₁ (which responds to texture, not the mean).
- **Magnitude:** CNN FoM3 inflated **~25–30×** (pre-fix ~400k → post-fix ~15–20k); ℓ₁ ~30% only.
- **Fix:** per-pixel spatial demeaning (`--zero-mean-maps`) before compression. Excludes 81% of the
  project's runs from quantitative use.

### P2 — Cross-channel noise model *(hard)*
- **Trap:** applying the **auto-map pixel-σ** to cross channels when setting wavelet SNR thresholds —
  but cross-map amplitudes are **~10⁴× smaller**, so their SNR collapses to ~0 and ~95% of the ℓ₁
  histogram bins zero out.
- **Signature:** a single `noise_sigma` scalar logged for all 10 channels; near-empty cross-channel ℓ₁
  histograms; *inflated* L1 auto+cross FoM3.
- **Magnitude:** the original "L1 wins 3×" (L1 auto+cross ~65k) → ~38k with the fix (−48%); cross-only
  +33%.
- **Fix:** estimate noise **per channel** from the data (`channel_empirical_global`); beware silent
  fallback on the wrong data route.

### P3 — Cross-map leakage from full-sphere construction *(hard)*
- **Trap:** building cross-maps as iSHT(aⁱ_ℓm·aʲ_ℓm) on the **whole sphere** and then cutting patches
  makes every cross-patch pixel a **global functional of the full sky** — the patch encodes
  information a real patch survey cannot access.
- **Signature:** cross channels carry **12–20%** of variance at super-patch scales (ℓ<18) vs 0.4–1%
  for autos; cross ℓ_median ~60–90 vs ~600; auto+cross gain that the autos "should" reproduce but
  cannot.
- **Magnitude:** drives the entire auto-only-tie vs auto+cross-CNN-lead split; makes the auto+cross
  constraining power **partly unphysical** (not a calibration bug — TARP/SBC/L-C2ST still pass).
- **Fix:** build cross-maps **patch-locally** (flat-sky) from the patch's own autos. *(Future work.)*

### P4 — NDE-architecture confound *(flag)*
- **Trap:** comparing two compressors through **different** density estimators (here RealNVP for CNN,
  MAF for L1) conflates the compressor with the flow.
- **Signature/magnitude:** the *same* compressor gives a **47% FoM3 swing** (RealNVP 25.9k vs MAF
  17.6k) at only ~5% marginal / ~15–20% 2D difference; RealNVP is also unstable under small-data
  splits.
- **Fix:** put both compressors through the **same** NDE (the common-MAF design). *Flag, not a
  disqualifier — but the only fair comparison controls it.*

### P5 — FoM3 fragility *(flag)*
- **Trap:** FoM3 = 1/√det(C₃) amplifies tiny correlation-structure changes for strongly correlated
  posteriors (Ωm–σ8 ρ≈−0.93).
- **Signature/magnitude:** a ~5% marginal-width change → ~50% FoM3 swing; pooled vs mean-of-seeds vs
  per-seed-min differ by a ~0.69 "haircut."
- **Fix:** report FoM3 **with** σ and 2D areas; never mix FoM3 variants across campaigns. *(FoM3 is
  retained as a headline metric — the fragility is acknowledged, not used to suppress it.)*

### P6 — Compressor↔NDE train/test overlap *(flag)*
- **Trap:** training the compressor and the NDE on the **same** examples.
- **Magnitude:** disjoint split costs ~18–24% FoM3 — i.e. overlap *inflates*.
- **Fix:** example-disjoint split (here, by noise permutation). *Flag; annotate.*

### P7 — The fixed-θ "TARP" trap *(process)*
- **Trap:** evaluating coverage at a **single fixed θ** (a Mahalanobis-χ² proxy) and calling it TARP —
  it spuriously shows strong over-coverage.
- **Fix:** TARP/DRP requires **θ ~ prior** (varied-θ). The valid varied-θ DRP agrees with SBC and
  L-C2ST; the fixed-θ proxy disagreed and was wrong.

### P8 — Route / patch-center confound (G8) *(process)*
- **Trap:** comparing arms that draw **different sky patches** (random-center TFDS vs deterministic
  non-overlapping cache) silently changes the effective data and inflates apparent cross-gain.
- **Fix:** identical patch centers across arms; quote cross-gain only against a route-matched
  auto-only baseline.

*(Other documented traps usable as appendix entries: PCA-on-ℓ₁ craters FoM3 5×; stock-BN ResNet on
multi-channel input collapses to ~700; the TFDS-route silent fallback for the cross noise model;
val-loss is not a reliable cross-architecture FoM3 proxy.)*

---

## 5. Exhaustiveness — the evidence that we tried everything

This is the material that converts "CNN ≥ L1" from a claim into a hardened result, and is the strongest
"thoroughness as contribution" content.

- **The auto-only ceiling (`cnn-auto-push`).** A focused campaign certified a CNN auto-only ceiling
  from **three independent falsifiers** (variance/drift at 120k, bound-widening, and a deeper
  resnet50_gn that overfit). The ceiling is real and architecture-robust.
- **The auto-cross-push campaign (115 iterations, ~500 GPU-h).** Architecture variants, capacity
  matching, learning-rate schedules, data-prep toggles, and VMIM-companion changes were all tried; the
  CNN auto+cross FoM3 **never moved off ~25k** toward the (then-believed) L1 ~38k target. We now know
  *why it couldn't*: the target itself was inflated (noise model) and leakage-driven. **A negative
  result that became evidence** — the gap was not a tuning deficit.
- **Three falsified hypotheses for "why CNN underperforms on cross"** (run as formal hypothesis
  fibers): **H1** cross-channel attention (no improvement), **H2** data-limit (untestable with the
  current suite), **H3** higher cdim=100 (49% *worse*). All three candidate explanations were
  eliminated — which, in hindsight, pointed at the data vector (leakage), not the compressor.
- **The NDE-swap matrix.** Systematically swapping RealNVP↔MAF on the *same* compressor isolated the
  flow's contribution and exposed FoM3 fragility — the experiment that reframed every prior
  comparison.
- **The calibration battery on the final run.** TARP-DRP (stratified) + SBC + L-C2ST all pass,
  including on the tight HIGH-FoM3 tercile — the back-pressure proving the tight contours are real.

---

## 6. Lessons / recommendations (the paper's "what the community should take away")

1. **Demean your maps** (or you measure the mass sheet, not lensing texture) — and check it the way we
   did (the asymmetric CNN-vs-ℓ₁ inflation is the tell).
2. **Match the density estimator** when comparing summaries; otherwise you compare flows.
3. **Calibrate before you believe tightness** — and use a *valid* (varied-θ) coverage test; a fixed-θ
   "TARP" will lie to you.
4. **Set per-channel noise scales** for multi-channel/cross statistics; a shared scale silently
   destroys the small-amplitude channels.
5. **Know what your cross-maps actually measure** — a full-sphere construction injects non-local
   information; for patch-survey realism, build them patch-locally.
6. **Report FoM3 with σ and 2D areas** — it is a fine headline scalar but it amplifies correlation
   structure; show the marginals so the reader can see when a FoM3 gap is amplification.
7. **A campaign that cannot close a gap may be telling you the gap is not real** — treat persistent
   negative results as evidence, not just as failed tuning.

---

*Cross-references: final results & methods in `PAPER_SCIENTIFIC_SYNTHESIS.md`; per-file trust in
`PAPER_FILE_TRIAGE.md`; figures in `PAPER_FIGURE_INVENTORY.md`; the run catalog/chronology in
`EXPERIMENT_AUDIT.md` (journey source; final verdict superseded by June).*
