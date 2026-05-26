---
name: Audit — why did canonical CNN auto+cross drop 47% from the iter-108-Q6ON-60k anchor?
tags:
    - investigation
    - canonical-anchors-refresh-2026-05
    - ship-blocker
    - suspicious
created-at: 2026-05-26T13:43:07.472081755Z
status: open
outcome: 'OPEN AUDIT (2026-05-26). Andreas distrusts the canonical CNN auto+cross pool 12,615 vs stale 23,986 (-47%). Train/train compressor-NDE overlap typically inflates FoM3 by 10-30%, not 2x. The L1 cross arm reproduces v2_chsigma to 0.5% under canonical methodology, validating the data + L1 pipeline. The CNN auto+cross 2x drop is therefore NOT just methodology — there must be another change. Hypotheses to investigate, in priority order: (H-A) re-run iter-108-Q6ON-60k EXACTLY today (train/train, no audit flag) — does it still give ~24k? If YES the drop is methodology+unknown; if NO there is a code regression. (H-B) maybe the NDE on 30% of train (train[70%:], ~90k examples) is under-training relative to the cdim=10 summary capacity; test by running canonical CNN cross with --nde-train-split train (full) to see if it recovers. (H-C) checkpoint policy difference: iter-108 used last_step (old default), canonical used best_val — but best_val typically improves not degrades. (H-D) the saved-compressor recovery path used the iter-2 compressor checkpoints from a buggy run; rerun cross with fresh compressor from scratch to confirm. Primary metric for the audit: CNN auto+cross 3-seed pooled FoM3. Decision rule: if H-A reproduces 24k, drop is real but unexplained; if not, find the code regression. NO COMPUTE until Andreas approves the audit plan in the new session. All 12 canonical posteriors remain on disk for comparison.'
---

## Objective

Resolve a serious anomaly in the canonical-anchors-refresh campaign:
**CNN auto+cross dropped from a stale anchor of 23,986 (iter-108-Q6ON-60k)
to a canonical 12,615 (-47%)**, while L1 auto+cross under the canonical
methodology reproduced the v2_chsigma anchor (33,820 → 34,004, +0.5%).

Andreas's read: train/train compressor-NDE contamination typically inflates
FoM3 by 10-30%, not 2×. The asymmetry between "L1 cross reproduces stale
exactly" and "CNN cross drops 2×" is not explained by the methodology fix.
**There must be another change we haven't isolated.** The previous
cnn-auto-cross-push Ralph campaign spent significant effort to push CNN
auto+cross to ~25k; losing that to "splits" without a deeper explanation
is suspicious.

## Primary metric

**3-seed pooled FoM3 on (Ωₘ, σ₈, w₀)**, CNN auto+cross, harmonic-cache
nobnt regime, plain CNN compressor 64/128/256/dense=256/cdim=10 with
60k compressor steps + 50k NDE steps. Same as `[[canonical-anchors-refresh-2026-05]]`
parent fiber.

## Hypotheses (ranked priority)

### H-A — Reproduce iter-108-Q6ON-60k EXACTLY (highest priority)

Run the iter-108-Q6ON-60k config today, *not* with the canonical methodology:
- `--compressor-train-split train` (not `train[:70%]`)
- `--nde-train-split train` (not `train[70%:]`)
- NO `--require-disjoint-train-examples`
- Otherwise everything matches iter-108 manifest

**Decision**:
- If we get ~24k pool again → the original iter-108 number is reproducible
  with current code. The drop to 12.6k under canonical is real and
  attributable to methodology change. The magnitude (~2×) is then a
  surprising scientific finding that needs explaining.
- If we get something significantly different from 24k → there's been a
  code regression between iter-108's generation and today. Find and fix.

Resource estimate: ~3-4h on GPU 1 sole tenant, 3 seeds parallel-3.

### H-B — NDE undertrained on train[70%:]

The canonical NDE saw only 30% of train (~90k examples) vs iter-108's
100% (~302k). For cdim=10 summary that's still ~9k examples per dim,
but maybe at this regime more data helps. Test: run canonical CNN cross
with `--nde-train-split train` (full data, but compressor still on
train[:70%] to keep the disjointness). If FoM3 lifts toward 24k,
NDE-data-quantity was the dominant factor.

This would break the "fair" comparison story but would also be a real
scientific insight (NDE training data > 90k matters for CNN cdim=10).

### H-C — Checkpoint policy

iter-108 was generated *before* the best-val checkpoint fix landed on
the cnn-auto-cross-push branch (commit 5c5a6d9 from 2026-05-19 +). It
may have used the last_step compressor checkpoint, not best-val.
Canonical uses best-val. Best-val typically *improves* FoM3 vs last_step
(less overfit), so this hypothesis predicts canonical > iter-108, not
the observed canonical < iter-108. Probably not the cause, but verify.

### H-D — Compressor checkpoint reuse from buggy iter-1 run

The canonical CNN cross posteriors came from the NDE-only recovery
that re-used the iter-1 compressor checkpoints. The iter-1 run had a
KeyError crash AFTER compressor training, so checkpoints survived —
but maybe the best-val snapshot was saved at a suboptimal step due to
some pre-crash state corruption. Test: re-run CNN cross from scratch
(fresh compressor training) and see if results match the recovery
posteriors. If different, the recovery path was contaminated.

### H-E — Something else

There may be a code change between iter-108's era and today that we
haven't identified. Check git log for npe_cnn_nbody_tomo.py + relevant
shared utilities since iter-108-Q6ON-60k's date. Cross-reference any
changes against compressor behavior, NDE training, summary standardization.

## Investigation sequence

1. **First**: re-compute FoM3 from the iter-108-Q6ON-60k saved posteriors
   on disk. Confirm we still get 23,986 from the SAVED data (cheap, no
   compute, ~1 min). This rules out any analysis-script regression.

2. **Then**: examine the saved compressor weights at
   `iter-108-Q6ON-60k/compressor/...` — do they exist? If yes, do a
   NDE-only re-run using those weights with canonical 70/30 split.
   If we get ~24k → the compressor matters (iter-108 compressor better
   than what canonical produces). If we get ~12.6k → it's NDE-side.

3. **Then H-A**: full reproduction of iter-108 config today (no
   canonical splits). Decision per above.

4. **Then H-B / H-D / H-E** as needed, based on outcomes.

## What NOT to do until this audit clears

- Do NOT trust the canonical CNN auto+cross 12.6k number for any paper
  writeup.
- Do NOT close the parent `canonical-anchors-refresh-2026-05` fiber.
- Do NOT update `CNN_CROSS_MAPS_INFORMATION_NOTE.md` §1 with the
  canonical numbers.
- Do NOT propose CLAUDE.md convention #8 enforcing the 70/30 split
  until we understand whether the split is doing what we expect.

The canonical numbers for the THREE other arms (CNN auto, L1 auto, L1
cross) may still be usable independently (their methodologies are less
disputed and L1 cross reproduces v2_chsigma exactly), but the
cross-arm-derived ratios (cross/auto for CNN, CNN/L1 at auto+cross) all
depend on resolving this anomaly.

## Sub-fibers

(To be filed as the audit proceeds. Each H-* gets its own sub-fiber if
significant compute or analysis is involved.)

## Connections

- Parent: `[[canonical-anchors-refresh-2026-05]]`
- Reference anchor: iter-108-Q6ON-60k at
  `/nas/tersenov/claude-notes/runs/cnn-auto-cross-push-18-20-2026/iter-108-Q6ON-60k/`
- Related: `[[project_pool_haircut_invariant_to_architecture]]`,
  `[[feedback_val_loss_not_reliable_fom3_proxy]]`,
  `[[project_resnet50gn_120k_overfits]]`.
