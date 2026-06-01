---
name: Comprehensive experiment audit — catalog every run, trace every config, identify what's trustworthy
status: done
tags:
    - investigation
    - audit
    - meta
created-at: 2026-05-27T12:57:14.479539277Z
outcome: 'DONE 2026-05-27 (deep pass). Audited 1,517 runs across 190 dirs. 27 fully clean (1.8%). 1,225 (81%) invalidated by mass-sheet leak. Bug timeline verified from git (10 bugs, exact commit dates). Critical discovery: 6 fully-clean CNN posteriors exist (zmm+disjoint, 0 overlap verified) in canonical_anchors_refresh — FoM3 never computed, posteriors on disk. NDE-swap test quantified: 70/30 split costs 18-24% FoM3, RealNVP/jaxili gap is 47% FoM3 but only 5% marginal σ. L1 auto+cross advantage confirmed real via w₀ (σ=0.133 vs CNN 0.166-0.194). 6 open experiments proposed (2 zero-GPU-time).'
---

## Objective

Read every experiment ever run in this project (~1,200 posteriors across
5 phases spanning months of work), extract the full configuration of each,
cross-reference with the known bug timeline, and produce a single readable
document (`EXPERIMENT_AUDIT.md`) that catalogs what we ran, what we can
trust, and what scientific conclusions survive.

## Primary metric

Completeness: every `.meta.json` in the project must appear in the audit
catalog. Trust status (trustworthy / partially trustworthy / invalidated)
must be assigned to every run with a one-line justification.

## Done condition

- `EXPERIMENT_AUDIT.md` exists at repo root with all 5 phase sections complete.
- Every experiment directory listed in `HANDOFF_COMPREHENSIVE_AUDIT.md` §1
  has been scanned and cataloged.
- The "Clean comparison table" section contains only runs verified against
  the 10-bug checklist, with 2D areas and marginal σ (not just FoM3).
- The "Open questions" section lists what experiments are needed to resolve
  the L1-vs-CNN comparison cleanly.

## Loop Status (live)

**Phase tracker** (update as each phase completes):
- [x] Phase 0: Inventory script + bug timeline (pre-computed inventory: 1,517 runs across 190 dirs)
- [x] Phase 1: BNT-era — 1,170 runs, ALL invalidated by mass-sheet leak. 118 CNN+jaxili runs found but also contaminated.
- [x] Phase 2: L1 development — covered as part of Phase 1 and Phase 3 (no standalone Phase 2 directories)
- [x] Phase 3: L1-vs-CNN cross-maps — 214 runs. Best L1 auto+cross: 38k (v2 noise model, verified clean). Best CNN auto+cross: 25.5k (harm-norm, train/train overlap).
- [x] Phase 4: Autoresearch campaigns — 241 runs. auto-push plateaued at MoS 19.9k. auto-cross-push ran 112+ iters, never closed gap to L1.
- [x] Phase 5: Canonical refresh + NDE-swap — 25 runs. NDE mismatch confirmed. FoM3 fragility confirmed.
- [x] Phase 6: Executive summary + clean comparison table + open questions — written.

**Audit complete.** Output: `EXPERIMENT_AUDIT.md` at repo root.

No GPU compute needed. This is a read-only audit. `/clear` between
phases to manage context.

## Methodology

### Phase 0: Bootstrap

1. Read `HANDOFF_COMPREHENSIVE_AUDIT.md` (the full briefing).
2. Read `CLAUDE.md` (project conventions, known bugs).
3. Read `MEMORY.md` (accumulated project knowledge).
4. Write `scripts/sbi/tools/audit_inventory.py` — a script that walks
   every experiment directory and extracts key fields from every `.meta.json`
   into a single `audit_inventory.json`. Fields to extract:

   ```
   path, method, compressor_arch, compressor_dim, compressor_train_split,
   nde_train_split, require_disjoint_train_examples, cnn_map_route,
   cnn_input_channels, zero_mean_maps, apply_bnt, summary_standardized,
   pca_applied, cross_noise_model, total_steps, patience, batch_size,
   flow_training_summary.best_val_loss, npe_epochs, npe_learning_rate,
   npe_batch_size, tfds_name, seed
   ```

   Also extract which SCRIPT was used (infer from `method` field:
   `cnn` → `npe_cnn_nbody_tomo.py` with sbi_lens RealNVP;
   `l1norm_cross_jaxili` → `npe_l1norm_cross_jaxili_nbody_tomo.py` with jaxili MAF;
   `cnn_jaxili` → `npe_cnn_jaxili_nbody_tomo.py` with jaxili MAF).

5. Run the inventory script. Save output.
6. Write the bug timeline section of `EXPERIMENT_AUDIT.md`.

### Phases 1-5: Per-phase audit

For each phase:

1. Load the inventory for that phase's directories.
2. Group runs by experiment/campaign.
3. For each group:
   - Read the launch script or campaign driver (if it exists).
   - Read 1-2 representative `.meta.json` files in full.
   - Read the relevant log files for anomalies (NaN, divergence, crashes).
   - Cross-reference each run against the 10-bug checklist.
   - Record: what was tested, what the result was, whether it's trustworthy.
4. Write one section of `EXPERIMENT_AUDIT.md`.
5. `/clear` context before starting the next phase.

### Phase 6: Synthesis

1. Read back all phase sections from `EXPERIMENT_AUDIT.md`.
2. Write the executive summary.
3. Build the clean comparison table (only trustworthy runs, with 2D areas).
4. Write the open questions section.
5. Update this fiber's outcome and close it.

## The 10-bug checklist (apply to every run)

1. PCA default=50 on L1 → must be `--pca-components 0`
2. L1 cross TFDS route silently uses broken `auto_scalar` noise model
3. RealNVP NDE divergence (check val loss curves if available)
4. Train/train compressor-NDE overlap (check `require_disjoint_train_examples`)
5. Missing `--zero-mean-maps` (mass-sheet degeneracy leak)
6. BatchNorm on 10-channel harmonic input (need GroupNorm)
7. Compressor checkpoint policy (`last_step` vs `best_val`)
8. NDE architecture (RealNVP vs jaxili MAF — which was used?)
9. Channel normalization mismatch between CNN and L1 routes
10. FoM3 fragility — report 2D areas alongside FoM3

## Key references

- `HANDOFF_COMPREHENSIVE_AUDIT.md` — the full briefing with file paths and chronology
- `CLAUDE.md` §"Felt / Ralph operating conventions" — project rules
- `MEMORY.md` — accumulated knowledge (16 entries including NDE mismatch and FoM3 fragility)
- `scripts/sbi/results/exploratory/canonical_anchors_refresh/nde_swap_test/SESSION_RESULTS.md` — verified NDE-swap results
- `PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md` — older synthesis (may cite stale numbers)
- `SBI_L1_CNN_PIPELINE_DETAILED.md` — pipeline documentation

## Connections

- Parent: none (top-level investigation)
- Triggered by: the NDE-swap-test session (2026-05-26/27) which found the
  NDE architecture mismatch and FoM3 fragility
- Blocks: any further L1-vs-CNN paper claims
- Related fibers: `[[canonical-anchors-refresh-2026-05]]`,
  `[[cnn-auto-push-18-20-2026]]`, `[[cnn-h1-inductive-bias-2026-05]]`,
  `[[cnn-h2-data-limit-scoping-2026-05]]`
