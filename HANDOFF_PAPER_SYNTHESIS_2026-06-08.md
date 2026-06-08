# HANDOFF — Paper scientific synthesis & good/bad triage (for the paper-draft session)

**Date:** 2026-06-08. **Goal:** this repo holds *months* of work across many branches, hundreds of
runs, and dozens of partly-contradictory documents. Produce **one clean, trustworthy scientific
synthesis** + a **good/bad/important file triage**, so that the `paper-draft` skill can write the
manuscript from solid ground instead of getting lost in the contradictions.

**This is a read-only archaeology + synthesis task. NO GPU, NO training, NO new experiments.** The
job is to *understand and organize what already exists*, separate what's trustworthy from what's
superseded/wrong, and write it up for the paper.

---

## The single most important starting point

**`EXPERIMENT_AUDIT.md`** (repo root, 870 lines) already did most of the good/bad separation: it
catalogs **1,517 runs across 190 dirs**, assigns each a trust status (trustworthy / partial /
invalidated), verifies a **10-bug timeline from git**, and has a "clean comparison table" + "open
questions". Built by felt fiber `comprehensive-experiment-audit-2026-05` (status: done). Raw data:
`audit_inventory.json` (1.5 MB); script: `scripts/sbi/tools/audit_inventory.py`; briefing:
`HANDOFF_COMPREHENSIVE_AUDIT.md`.

**BUT it is dated 2026-05-27** — it predates the two most important recent developments:
1. the **definitive 10° L1-vs-CNN campaign** (June; `…/definitive_comparison_10deg/phase_c/analysis/
   SUMMARY_PHASE_D.md`), and
2. the **cross-map leakage finding** (2026-06-08; `CROSS_MAP_LEAKAGE_FINDING.md`) which makes the
   auto+cross results **provisional** pending the flat-sky rebuild (`FLATSKY_CROSS_BUILD_PLAN.md`).

So **do not treat EXPERIMENT_AUDIT.md as final** — it's the backbone; reconcile it with the June work
and the memory files, and flag where the newest findings overturn earlier ones.

---

## The paper's scope — two pillars (Andreas's framing, capture faithfully)

### Pillar 1 — L1-norm vs CNN constraining power, + the cross-map strategy
Compare the constraining power of the **wavelet ℓ₁-norm** vs **CNN-based (VMIM) summaries** in a
**tomographic** weak-lensing setting on simulations, via **SBI**. Investigate architectures to get
the *best* contours from both compressors. **Additionally**, determine the **best strategy to
generate cross-maps** (vs the auto-maps the sims give) to extract *extra* cosmological information —
this is where the harmonic-vs-flat-sky construction, the leakage finding, and the upcoming flat-sky
rebuild live. Current state: CNN ≳ L1 at 10° auto-only (tie) and on auto+cross (CNN ahead) — **but
the auto+cross part is contaminated by the cross-map leakage and must be redone patch-local**
(separate campaign, `flatsky-cross-2026-06`).

### Pillar 2 — BNT + higher-order statistics: the contour-inflation question (the paper's "proof")
Prior work (incl. one of Andreas's own papers) found that applying **BNT** before a **higher-order
statistic** (e.g. the ℓ₁-norm) **inflates** the cosmological contours. Andreas's argument: BNT is an
**invertible linear** transform, so information *cannot* be lost in principle; the inflation comes
from **not fully extracting the cross-correlations between the tomographic bins**. BNT decorrelates
the *signal* across bins but **correlates the noise** (making it complicated), which lowers the
signal-to-noise → contour inflation for a statistic that doesn't recover the cross-information.

**The thesis this paper wants to prove:** a **CNN-based compressor does NOT suffer this inflation**,
because feeding the CNN the tomographic auto-maps **as channels** lets it extract the implicit
cross-correlations on its own (the **VMIM** objective drives it to capture all θ-relevant info). So a
CNN compressor on BNT maps → **no information loss → no contour inflation**. The repo got *close* to
demonstrating this; it must be **re-run cleanly in the current setup** to confirm. (BNT materials:
`BNT_NO_BNT_CONTOUR_INFLATION_NOTE.md`, `BNT_TOMO4_FINAL_SCIENTIFIC_CONCLUSIONS.md`,
`BNT_TOMO4_STUDY_RESULTS.md`, `TOMO_BIN_CROSSCORR_DETAILED_REPORT.md`,
`OPTIMAL_NOBNT_CROSSCORR_SCIENTIFIC_CONCLUSIONS.md`; branches `bnt-parity-techniques`, `bnt_tomo_study`
+ worktree `.worktrees/bnt_tomo_study`.)

The two pillars share one mechanism: **cross-correlation information between tomographic bins**, and
whether a given compressor recovers it. That's the throughline to foreground in the paper.

---

## Where the project's history lives (survey these)

**Branches** (the project arc): `l1norm` → `l1_compressor` → `l1-cross-maps`, the `l1-jax*` family
(`l1-jax`, `l1-jax-cnn-audit`, `l1-jax-indep-split`, `l1-jax-multipatch`, `l1-jax-resnet`),
`bnt-parity-techniques`, `bnt_tomo_study`, `dev`, `main`, and the `autoresearch/cnn-auto-push-*` /
`cnn-auto-cross-push-*` campaigns (current HEAD = `autoresearch/cnn-auto-push-18-20-2026`).
**Worktrees:** `.worktrees/bnt_tomo_study`, `.worktrees/cnn-auto-cross-push-18-20-2026`.
Use `git log --oneline` per branch and the dated HANDOFF_*.md docs to reconstruct the timeline.

**Synthesis / knowledge docs** (read, but cross-check dates — older ones cite superseded numbers):
`EXPERIMENT_AUDIT.md` (trust catalog), `PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md` (569L long-form),
`INVENTORY.md`, `SBI_L1_CNN_PIPELINE_DETAILED.md`, `SBI_PIPELINE_BEST_PRACTICES.md`,
`scripts/sbi/results/final/paper_sbi_consolidation/FINAL_SCIENTIFIC_REPORT.md`.

**Pillar-1 cross-maps:** `CNN_CROSS_MAPS_INFORMATION_NOTE.md`, `Harmonic_cross_maps.md`,
`Flat-Sky_Tomographic_Cross_Maps.md`, `HARMONIC_L1_VS_CNN_*`, `CROSS_MAP_LEAKAGE_FINDING.md`,
`FLATSKY_CROSS_{REDESIGN_NOTES,BUILD_PLAN}.md`, `SUMMARY_PHASE_D.md`.

**Memory** (`…/memory/MEMORY.md`) — the most curated, current one-liners; treat as the tie-breaker
when docs disagree. Key: the 10° definitive result, the leakage finding, fom3-fragility, the
noise-model correction, mass-sheet leak, never-PCA-L1, NDE-architecture-mismatch.

---

## The "self-contradiction" problem (the core difficulty)

Numbers in this repo were repeatedly **revised** as bugs were found. Same headline ("L1 wins 3×")
appears with different values across docs. Rule of thumb for **which to trust**, newest-wins with
provenance:
1. **`EXPERIMENT_AUDIT.md` trust status** for any run predating June.
2. **June docs** (`SUMMARY_PHASE_D.md`, `CROSS_MAP_LEAKAGE_FINDING.md`, the FLATSKY notes) override
   earlier cross-maps/L1-vs-CNN conclusions.
3. **Memory files** as the curated tie-breaker.
4. Known invalidators to screen every cited number against: **mass-sheet-degeneracy leak**
   (pre-`--zero-mean-maps`; killed ~81% of runs), **L1 cross-channel noise-model bug** (auto-σ on
   cross), **FoM3 fragility** (never headline FoM3 — use σ + 2D areas), **NDE-architecture mismatch**
   (CNN RealNVP vs L1 jaxili MAF), **train/test (compressor↔NDE) overlap**, **cross-map leakage**
   (full-sphere). A number is only citable if it survives all of these.

---

## Deliverables for this session

1. **`PAPER_SCIENTIFIC_SYNTHESIS.md`** (new, repo root) — the input for `paper-draft`. Sections:
   - **Project arc & scope** (timeline across branches; the two pillars).
   - **Pillar 1 results**: L1 vs CNN (auto-only, auto+cross) — current trustworthy numbers (σ, 2D
     areas; FoM3 only as support), calibration status, with provenance; the cross-map-strategy story
     (auto vs harmonic-leaky vs flat-sky-to-come).
   - **Pillar 2 results**: BNT inflation for L1 vs the no-inflation hypothesis for CNN — what was
     shown, how close we got, what must be re-run.
   - **Methods** the paper needs: sims (CosmoGridV1), tomography, ℓ₁ wavelet statistic, CNN-VMIM
     compressor, cross-map construction, SBI/NDE, calibration (TARP/SBC/L-C2ST).
   - **Figures that exist / are needed** (point to the real files), and **key references**.
   - **Open items / still-to-run** (flat-sky cross rebuild; clean BNT-CNN no-inflation run) — clearly
     marked as future/forthcoming so the draft doesn't claim them as done.
2. **`PAPER_FILE_TRIAGE.md`** (new) — a good/bad/important table: for each major doc and results dir,
   mark **CITE / BACKGROUND / SUPERSEDED / WRONG**, with the one-line reason and the superseding item.
   Seed it from `EXPERIMENT_AUDIT.md` and extend with the June work.
3. Update the felt index (a fiber for this synthesis task) and the memory if any durable fact emerges.

## Guardrails

- **Read-only.** No GPU/training/new runs. (If a number is missing, say "not computed", don't make
  one up.)
- **Newest-wins with provenance** (the rule of thumb above). Every cited result needs a trust trail.
- **Don't headline FoM3** anywhere; lead with σ(w0) + 2D areas.
- **Separate done vs to-be-done**: the flat-sky cross result and the clean BNT-CNN run are *future*;
  the synthesis must not present them as established.
- **Don't overwrite/delete** existing docs; produce the two NEW synthesis docs.
- Faithful to Andreas's two-pillar framing above (it's the paper's thesis).

## First actions
1. Read `EXPERIMENT_AUDIT.md`, `HANDOFF_COMPREHENSIVE_AUDIT.md`, this handoff, `MEMORY.md`, and the
   two pillars' key docs. `git log --oneline --all | head -200` + per-branch logs for the timeline.
2. Reconcile the May audit with the June work (SUMMARY_PHASE_D, leakage, flat-sky).
3. Draft `PAPER_FILE_TRIAGE.md`, then `PAPER_SCIENTIFIC_SYNTHESIS.md`. Get Andreas's review before
   handing to `paper-draft`.

## Recommended execution — dynamic workflow (Stage 1 only)

This task is a strong fit for Claude Code **dynamic workflows** (research preview; needs ≥ v2.1.154,
enabled in `/config`). Why: it is *width-shaped* (sweep 1,517 runs / 190 dirs / 13 branches / 2
worktrees in parallel) and the hard part — resolving self-contradictions — is exactly the
**adversarial cross-check / vote-and-filter** pattern workflows provide. Workflows also defeat the
two failure modes this task is most prone to: **agentic laziness** (a single session will quietly
triage only part of 1,517 runs) and **goal drift** (intermediate state lives in script variables,
not a degrading context).

**Stage it** (workflows take no mid-run user input → run each stage separately for sign-off):

- **STAGE 1 = a workflow (read-only fan-out + cross-check + triage).** Invoke with `ultracode` /
  "use a workflow". Phases: (a) enumerate branches/worktrees/results-dirs/docs; (b) one **read-only**
  agent per chunk extracts {what it is, dates, claimed numbers, experiments}; (c) **adversarial
  verification** — agents screen every claimed number against `EXPERIMENT_AUDIT.md` + the 6
  invalidators + the June docs, vote, filter, and assign **CITE / BACKGROUND / SUPERSEDED / WRONG**.
  Output: `PAPER_FILE_TRIAGE.md` + the verified evidence base. **Pilot on one branch/phase first** to
  gauge token cost; tell the gather agents "read-only, do not edit any file"; make sure the
  verification agents are pointed at the bug timeline + invalidators (or the cross-check is toothless).
- **→ Andreas reviews the triage.**
- **STAGE 2 = a normal single session** (NOT a workflow): write `PAPER_SCIENTIFIC_SYNTHESIS.md` from
  the verified evidence, two-pillar structure. The narrative is judgment-heavy and Andreas will steer
  it — keep it in one coherent, reviewable pass.

(Most tasks don't need a workflow; this one does because of scale + cross-check. Don't use a workflow
for Stage 2.)

## Copy-paste prompt for the new session

```
I'm continuing a weak-lensing SBI project (cnn_sbi repo on titan). This session is a separate,
READ-ONLY task: I'm about to write the paper for this project, but the repo is huge and messy after
months of work, with lots of superseded and self-contradictory material. Your job is to produce a
clean scientific synthesis and a good/bad file triage so that my paper-writing skill can work from
solid ground.

START by reading `HANDOFF_PAPER_SYNTHESIS_2026-06-08.md` in the repo root — it's the entry point and
tells you where everything is (incl. the recommended dynamic-workflow execution). Then read
`EXPERIMENT_AUDIT.md` (the May trust-catalog of 1,517 runs — the backbone, but it predates the June
work and must be reconciled), `MEMORY.md`, and the two pillars' key docs. Use `git log` across
branches/worktrees to reconstruct the project timeline.

READ-ONLY: no GPU, no training, no new experiments. Understand and organize what exists; separate
trustworthy from superseded/wrong; write it up for the paper.

For STAGE 1 (gather + cross-check + triage), USE A DYNAMIC WORKFLOW (`ultracode`): fan out read-only
agents across the branches/worktrees/results-dirs/docs, then run an adversarial cross-check that
screens every claimed number against EXPERIMENT_AUDIT.md + the known invalidators (mass-sheet leak,
L1 cross noise-model bug, FoM3 fragility, NDE-architecture mismatch, compressor↔NDE overlap,
cross-map leakage) + the June docs, votes, and assigns CITE/BACKGROUND/SUPERSEDED/WRONG. Pilot on one
branch/phase first to gauge cost; keep the gather agents strictly read-only. Output `PAPER_FILE_TRIAGE.md`.
Then STOP for my review. STAGE 2 (writing `PAPER_SCIENTIFIC_SYNTHESIS.md`, the input for paper-draft)
is a normal single session, NOT a workflow — the two-pillar narrative is judgment-heavy and I want to
steer it.

The paper has two pillars (full framing in the handoff): (1) L1-norm vs CNN-VMIM constraining power
in tomographic weak-lensing SBI, incl. the best strategy to build cross-maps for extra cosmological
info; (2) the BNT contour-inflation question — prior work (incl. mine) found BNT + higher-order
statistics inflates contours; the thesis is this is from failing to extract inter-bin
cross-correlations (BNT is invertible/linear, so no info is truly lost — it decorrelates signal but
correlates noise), and that a CNN compressor (auto-maps as channels, VMIM) does NOT inflate because
it recovers those cross-correlations. We came close to showing this; it must be re-run cleanly later.

Trust rule — newest-wins with provenance: every cited number must survive the invalidators above;
June docs override earlier cross-map/L1-vs-CNN conclusions; memory files are the tie-breaker; if a
number was never computed, say so — do NOT invent one. Don't headline FoM3 (use σ + 2D areas).
Clearly separate DONE from TO-BE-RUN (the flat-sky cross rebuild and the clean BNT-CNN run are future
work, not results). Don't overwrite existing docs; create the two new ones. Track the task in a felt
fiber.

Read everything first, then give me a short plan + the proposed structure of the synthesis doc, and
get my sign-off before writing. Begin by reading the handoff and confirming you understand the two
pillars, the trust rule, and the staged workflow plan.
```
