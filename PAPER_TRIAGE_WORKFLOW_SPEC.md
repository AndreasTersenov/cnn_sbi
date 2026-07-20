# Stage-1 dynamic-workflow spec — paper file triage (read-only fan-out + adversarial cross-check)

**Purpose.** This is the method record for how `PAPER_FILE_TRIAGE.md` was produced: a CITE / BACKGROUND
/ SUPERSEDED / WRONG verdict for every major doc, results-dir, and memory file in the repo, with a
provenance trail, screened against 6 invalidators, **reconciling the May trust-catalog
(`EXPERIMENT_AUDIT.md`) with the June work**. Stage 1 of the paper-synthesis task (fiber
`paper-synthesis-triage-2026-06`). Stage 2 (writing `PAPER_SCIENTIFIC_SYNTHESIS.md`) is a separate,
non-workflow session.

Executed 2026-06-08 as two read-only dynamic-workflow passes (Claude Code `ultracode`):
- **Pilot** (Part 1, Pillar-1 cross-maps + 10°): 39 records → 38 rows, 124 agents, ~6.24M tokens.
- **Sweep-2** (Part 2, Pillar-2 BNT + synthesis + memory): 57 rows, 184 agents, ~7.24M tokens
  (resumed once after a mid-run credential expiry; cached gather results made the resume cheap).

Each pass: **Phase 0** enumerate (1 read-only Sonnet agent) → **Phase 1** gather (1 read-only Sonnet
agent per item, pipelined) → **Phase 2** adversarial screen (2 independent session-model screeners per
item + a third-agent tiebreaker on disagreement) → **Phase 3** a single writer agent (the only agent
that writes a file). Scripts persisted under the session's `workflows/scripts/` dir; resumable by
`{scriptPath, resumeFromRunId}`.

---

## 0. Hard rules enforced in every agent prompt

- **READ-ONLY** for gather + screen agents (no edits/writes/state-changing git); only the Phase-3
  writer writes, and only the triage file. (Workflow subagents run in `acceptEdits` with auto-approved
  edits — the instruction is the only guard, so it was repeated verbatim into every agent.)
- **Newest-wins with provenance.** Precedence: (1) June docs override earlier cross-map / L1-vs-CNN
  conclusions; (2) `EXPERIMENT_AUDIT.md` trust status as a **prior, not gospel** — where June work or a
  memory contradicts it, June/memory wins and the row flags "audit needs correction"; (3) memory files
  are the tie-breaker.
- **Never invent a number** (`not computed` if absent). **Never headline FoM3** — σ(w₀) + 2D(Ωm,σ8)
  area are primary; FoM3 is support only.

## 1. The invalidators — 3 HARD + 3 FLAGS (with signatures)

The triage was first run with all six as hard disqualifiers. **Andreas relaxed three on 2026-06-08:**
#3 FoM3-fragility, #4 NDE-architecture mismatch, and #5 compressor↔NDE overlap are now **flags to note
and carry as a caveat, not disqualifiers.** A number is non-citable only if a *hard* invalidator fires
(#1, #2) or — for auto+cross — it is rendered PROVISIONAL by #6. (Source: `EXPERIMENT_AUDIT.md` bug
timeline.) The relaxation does **not** flip the WRONG/SUPERSEDED tally: those rest on the hard
invalidators + newest-wins, both independent of the three relaxed flags.

**HARD (disqualifies):**

1. **Mass-sheet-degeneracy leak** — any run without `zero_mean_maps=True` (pre 2026-04-22, commit
   `deb5ee0`). Inflates CNN FoM3 ~25–30×; barely touches L1. ~81% of runs (1,225/1,517). *Signature:*
   `zero_mean_maps` absent/False in meta.json.
2. **L1 cross-channel noise-model bug** (`auto_scalar`) — auto-map pixel-σ applied to cross channels
   (~30,000× smaller) → cross wavelet SNR ≈ 0. Fixed 2026-05-15 (`channel_empirical_global`, `f0b352b`);
   **silently ignored on the TFDS `--cross-maps` route** (must use the harmonic cache). Inflated the
   "L1 wins 3×" headline (v1 ~65k → v2 ~38k auto+cross). *Signature:* a single `noise_sigma` scalar for
   all 10 channels in the log, or a warning + fallback to `auto_scalar`.
6. **Cross-map leakage (full-sphere)** — the 6 cross channels are aᵢ_ℓm·aⱼ_ℓm → iSHT on the WHOLE
   sphere → patch cutout, so every cross-patch pixel is a global functional of the full sky (12–20%
   super-patch variance vs 0.4–1% for autos). Makes **all auto+cross constraining power partly
   unphysical** ⇒ renders auto+cross **PROVISIONAL** (not WRONG). NOT a calibration bug
   (TARP/SBC/L-C2ST pass). Auto-only unaffected. (`CROSS_MAP_LEAKAGE_FINDING.md`, 2026-06-08.)

**FLAGS (note and carry; do NOT disqualify) — relaxed 2026-06-08:**

3. **FoM3 reporting** *(relaxed — FoM3 is fine)* — FoM3 = 1/√det(C₃) may be cited as a primary metric.
   Keep the fragility caveat in mind (~5% marginal change → ~50% FoM3 swing; pooled vs mean-of-seeds vs
   per-seed-min differ by the ~0.69 haircut, so don't mix variants across campaigns) and report σ + 2D
   alongside where available — but a FoM3-based number is no longer downgraded for being FoM3.
4. **NDE-architecture mismatch** *(relaxed — flag only)* — CNN = sbi_lens RealNVP (567k params,
   unstable) vs L1 = jaxili MAF; a non-common-NDE comparison conflates compressor with NDE (47% FoM3
   gap on the same compressor). The June 10° campaign uses a common MAF and is flag-free. Annotate
   cross-architecture comparisons; do not disqualify.
5. **Compressor↔NDE train/test overlap** *(relaxed — flag only)* — compressor on `train`, NDE also on
   `train` (~86% of runs); disjoint split costs ~18–24% FoM3. Annotate; do not disqualify.

## 2. June overrides (the newest-wins layer)

- `…/definitive_comparison_10deg/phase_c/analysis/SUMMARY_PHASE_D.md` (2026-06-07) — current Pillar-1
  anchor (common-MAF, 9000 obs/arm, 10°): auto-only tie (clean); auto+cross CNN ahead but provisional
  (leakage); L1's 20° −0.37σ w₀ offset shrinks to −0.10σ & is no longer L1-specific ⇒ flat-sky artifact.
- `CROSS_MAP_LEAKAGE_FINDING.md` (2026-06-08) — invalidator #6.
- `FLATSKY_CROSS_BUILD_PLAN.md` / `FLATSKY_CROSS_REDESIGN_NOTES.md` (2026-06-08) — the decisive FUTURE
  test (patch-local cross rebuild); never a result.
- Memory: `project_10deg_definitive_cnn_geq_l1`, `project_cross_map_leakage_fullsphere`,
  `project_l1_fiducial_bias_is_prior_shrinkage`, `project_cnn_tightness_calibrated_not_geometry`,
  `feedback_fom3_fragile_use_2d_areas`, `project_l1_noise_model_correction`, `feedback_never_pca_l1`,
  `feedback_l1_cross_must_use_harmonic_route`, `project_nde_architecture_mismatch`,
  `project_resnet_bn_contamination`, `project_cnn_vmim_mass_sheet_leak`.

## 3. Pillar-2 (BNT) screening addendum

The paper's Pillar-2 thesis: BNT is an invertible linear transform ⇒ no information is truly lost;
contour inflation comes from a per-channel statistic (wavelet ℓ₁) failing to recover inter-bin
cross-correlations (BNT decorrelates signal across bins but correlates noise); a CNN fed the tomographic
auto-maps **as channels** under VMIM recovers them ⇒ no inflation. Every quantitative BNT result in the
repo is April Phase-1, **mass-sheet-contaminated (hard #1)** — that, not the now-relaxed FoM3 flag, is
the disqualifier. Therefore:
- Absolute BNT FoM3 / "near-lossless 1.04" presented as a clean result ⇒ **WRONG** (or **SUPERSEDED**),
  on the hard #1 leak (absolute CNN FoM3 inflated ~25–30×), independent of the FoM3-fragility relaxation.
- The same finding framed as within-era **directional** (inflation summary-limited; standardization
  lever 0.095→0.794; CNN out-extracts L1 on inter-bin cross-corr / G_corr; plain CNN > stock-BN ResNet
  on parity) ⇒ **BACKGROUND** motivation, not a citable number.
- The clean BNT-CNN no-inflation run is **FUTURE** (to-be-run), never a result.
- Keep distinct: Pillar-2 *implicit inter-bin cross-correlations* (recovered by a multi-channel CNN
  from auto-maps) ≠ Pillar-1 *explicit cross-maps* (κᵢ×κⱼ channels, the leakage issue).

## 4. Verdicts

- **CITE** — survives all 6 invalidators + newest-wins; paper-usable; provenance named.
- **BACKGROUND** — methods/definition doc, OR a directional/relative finding valid only within its era;
  not a citable absolute number.
- **SUPERSEDED** — correct in its day, a newer doc/run overrides it (named).
- **WRONG** — a number/claim invalidated by a named bug (named).

## 5. Output

`PAPER_FILE_TRIAGE.md` (canonical, both parts merged): a citable-numbers ledger (only
all-invalidator survivors), verdict tables grouped by category, a Pillar-2 "directional findings usable
as motivation" section, audit-correction notes (every June-vs-May conflict found), and a DONE
vs TO-BE-RUN section. The full-repo per-run catalog is deliberately **not** re-derived — the triage
works at the doc/campaign-directory level and cites `EXPERIMENT_AUDIT.md`'s per-run trust status, which
already cataloged all 1,517 runs.
