---
name: Pooled FoM3 confirms mode drift dominates (audit A4)
status: closed
tags:
    - finding
    - audit-A4
    - cnn-auto-push-18-20-2026
created-at: 2026-05-18T22:33:13.570522363Z
outcome: 'Mode-drift A3 hypothesis CONFIRMED empirically: CNN kept/tie iters have pooled/mos FoM3 = 0.63-0.69 vs L1 reference 0.89. iter-0 (d=10 baseline) was 0.82; tightening to iter-5 (best mean) dropped it to 0.69; wider-conv arms went to 0.63. The autoresearch +30% MoS gain since iter-0 is only +10% in pooled (11700 -> 12894). σ_8 mode drift dominates (R=0.78-0.99 in wider-conv arms vs 0.20 for L1). The loop has been climbing a misleading hill.'
---

# Pooled FoM3 confirms mode drift dominates (audit A4)

A3 (adversarial peer review, iter-4) filed [[cnn-auto-bug-fom3-rewards-mode-drift]] as a *possibility*. This audit (iter-5, Ralph) **confirms it empirically** by computing the diagnostic the A3 fiber requested.

## What was measured

For each kept/tie iter and the L1 auto+cross reference, over 3 seeds (41/42/43), in the FoM3 subspace (Ω_m, σ_8, w_0):

- `mos_FoM3 = mean_i 1/sqrt(det C_i)` (the autoresearch metric — what we have been optimizing).
- `pooled_FoM3 = 1/sqrt(det C_pooled)` (the metric science actually cares about).
- `pooled / mos` (≤ 1; 1 = seeds fully agree; ≪ 1 = mode drift dominates).
- `R_j = std_seeds(<mean_seed_j>) / mean_seeds(std_seed_j)` per parameter (R ≫ 1 means centroid scatter ≫ posterior width — drift dominates that axis).
- `joint_R = ‖centroid_scatter‖ / ‖avg_width‖` in (Ω_m, σ_8, w_0).

Recipe: `audits/2026-05-18_A_mode_drift/compute_mode_drift.py` (in `$NOTES_DIR/runs/cnn-auto-push-18-20-2026/`). Posteriors are the per-seed `.npy` files; pooled is concatenation across 3 seeds (100k samples each → 300k pooled).

## Result table

| iter | mos | pooled | pooled/mos | R(Ω_m) | R(σ_8) | R(w_0) | joint_R |
|---|---:|---:|---:|---:|---:|---:|---:|
| iter-0 (d=10 lr=5e-4) | 14295 | 11700 | **0.82** | 0.40 | 0.57 | 0.07 | 0.16 |
| iter-1 (d=16) | 16149 | 11116 | 0.69 | 0.55 | 0.74 | 0.25 | 0.31 |
| iter-5 (BEST, d=16 lr=1e-3) | 18568 | 12894 | **0.69** | 0.53 | 0.78 | 0.23 | 0.31 |
| iter-14 (wider conv) | 18822 | 11851 | **0.63** | 0.64 | 0.98 | 0.09 | 0.27 |
| iter-17 (F1 fix on plain) | 18457 | 12279 | 0.67 | 0.62 | 0.89 | 0.23 | 0.33 |
| iter-18 (wider+F1) | 18274 | 11836 | 0.65 | 0.69 | 0.99 | 0.01 | 0.26 |
| L1 auto+cross REF | 38226 | 34004 | **0.89** | 0.13 | 0.20 | 0.17 | 0.17 |

## Reading

1. **L1's pooled/mos = 0.89** sets the scale of "honest" mode drift on this dataset. CNN kept iters are all in the 0.63–0.82 band — substantially more drift than the reference.

2. **The autoresearch +30% MoS gain since iter-0 (14295 → 18568) is only +10% in pooled (11700 → 12894).** Two thirds of the headline improvement is mode-drift compression masquerading as posterior tightening.

3. **σ_8 is the dominant drift axis**, especially in wider-conv arms (R=0.98–0.99: centroid scatter equals posterior width). Seeds disagree most strongly on σ_8 localization; the plain arms (R=0.74–0.89) are only marginally better.

4. **iter-14 wider-conv has WORSE pooled (11851) than iter-5 (12894) despite higher MoS (18822).** This is the iter-18 transfer inversion (F1 fix from plain didn't transfer to wider-conv) explained: wider conv *tightens per-seed posteriors* (raises MoS) but *worsens mode drift* (drops pooled). Same for iter-18.

5. **The keep-rule has been selecting moves that make per-seed compressor outputs less interchangeable across seeds.** This is consistent with the VMIM compressor learning seed-dependent summaries (the per-seed dataloader split and noise realization differ across seeds, and a tight compressor amplifies those differences).

## Implications for the EV queue

Reframe the search:

- **The CNN auto-only ceiling on pooled FoM3 is ~13k**, not 40k. The constitution's success target (40k MoS) translates to ~33k pooled — and the current best is 13k. We are at 38% of the pooled target. The MoS metric has been hiding the gap.

- **All Tier-1/2 hypotheses should now be evaluated on pooled FoM3.** Specifically:
  - Q2 (compressor-steps > 60k): test will discriminate ONLY if longer training reduces seed-dependence, not if it tightens per-seed posteriors. iter-16 (in flight) will tell us.
  - Q7 (cosine schedule): same caveat.
  - Q4 (VMIM aux width): [[cnn-auto-bug-vmim-aux-may-bias-compressor]] — increasing VMIM aux capacity might reduce the *kind* of seed-dependence the bound rewards.

- **Better candidate hypotheses surfaced by this audit**:
  - **Seed-disjoint compressor training**: train the compressor on ALL seeds simultaneously and evaluate per-seed at inference. Would remove the train-split seed dependence. Concrete test: train once with `--seed 0` (or pool seeds), then evaluate NDE with seeds 41/42/43. If mode drift collapses, the per-seed compressor training is the bug.
  - **Bigger NDE val set**: per-seed NDE val sets are tiny (n_val=200 or similar — verify). Mode drift in posterior centroids correlates with NDE val-set determination of the best step.
  - **σ_8 as the bottleneck**: the wider-conv arm is *worse* at σ_8 stability. The σ_8-discriminating features may be in low spatial frequencies that wider convs lose. This points at [[cnn-auto-bug-pool-window-collapses-spatial]] — pool window 16/stride 8 over (20,20) features is mostly-global; σ_8 features may need finer spatial pooling.

## Methodological change (proposed)

The autoresearch keep rule should be either:

- **Primary**: pooled FoM3 (clean, what science cares about). Noise floor: estimate from L1 reference's std across 5 seeds (need to extend L1 ref).
- **Secondary**: report MoS alongside pooled, but require pooled to clear noise to "keep".

Equivalent formulation: track `MoS × (pooled/MoS)^k` for some k ≥ 1 — penalize drift.

This is a constitution-level change. Filed here as a question for Andreas. See [[cnn-auto-question-switch-to-pooled-fom3]].

## Cross-references

- [[cnn-auto-bug-fom3-rewards-mode-drift]] — the A3 hypothesis this audit confirms.
- [[cnn-auto-pooled-vs-mos]] — earlier session-1 sub-fiber noting the gap; this elevates it.
- [[cnn-auto-bug-pool-window-collapses-spatial]] — likely related (σ_8 spatial features lost).
- [[cnn-auto-bug-vmim-aux-may-bias-compressor]] — also related (bound looseness amplifies seed dependence).

Artifacts: `audits/2026-05-18_A_mode_drift/{compute_mode_drift.py, mode_drift_summary.md, mode_drift.json}`.
