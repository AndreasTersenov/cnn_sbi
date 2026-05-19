---
name: Variance/drift family additive on drift compression, capped at pooled FoM3 ~14 k (cnn-auto-push, Q9b iter-21)
status: closed
tags:
    - finding
    - cnn-auto-push-18-20-2026
    - variance-drift-family
    - ceiling-evidence
created-at: 2026-05-19T01:45:39.902477667Z
outcome: 'iter-21 (F1 on top of Q9 stack): drift compresses to campaign-best joint_R 0.186 (close to L1 0.17), pooled/MoS 0.777 (campaign best, predicted [0.77, 0.79] HIT), R(w_0) 0.037 (6.2x compression vs iter-5). BUT absolute pooled FoM3 stays at 13 829 (-0.83% vs iter-20 13 944, predicted +3..+8% MISS). F1 tightens MoS numerator AND pooled denominator at similar rates -> ratio improves, absolute pooled flat. Variance/drift family decomposable to 3 levers (cbs, pool, F1) on DRIFT axis but cannot break the ~14 k pooled ceiling. Next: Q9c (120 k compressor stack) tests if ceiling is information-limited or architecture-limited.'
---

# What iter-21 shows

Q9b stacked F1 (Stage B reload from iter-20's best-val ckpt at step 24 000,
test loss −12.5104 vs final −12.3709, gap 0.14 nats) on top of the Q9 stack
([[cnn-auto-cbs256-pool88-partial-stack]]: cbs=256 + pool=8/8). The hypothesis
was: variance/drift levers are additive on **both** drift compression
(joint_R, R(w_0), pooled/MoS ratio) **and** the absolute pooled FoM3 axis.

The data say: only **partially additive**, and **only on the drift axis**.

## Numbers

| axis              | iter-5 | iter-19 | iter-20 | **iter-21** | L1 ref |
|-------------------|------:|-------:|-------:|------------:|-------:|
| MoS FoM3          | 18 568 | 18 213 | 18 673 | **17 805**  | 38 226 |
| pooled FoM3       | 12 894 | 13 087 | **13 944** | 13 829  | 34 004 |
| pooled/MoS ratio  | 0.694 | 0.719 | 0.747 | **0.777** *(campaign best)* | 0.89 |
| joint_R           | 0.309 | 0.251 | 0.220 | **0.186** *(campaign best, ~9% above L1)* | 0.17 |
| R(w_0)            | 0.234 | 0.162 | 0.080 | **0.037** *(campaign best)* | (n/a) |
| R(σ_8)            | 0.779 | 0.752 | 0.772 | **0.684** *(first σ_8 compression)* | 0.20 |
| std-of-MoS        |  1604 |   904 |  2014 | 1504        | (n/a) |
| per_seed_min      | 16 368 | 17 187 | 16 158 | 15 685     | (n/a) |

## What this tells us

- **Drift compression IS decomposable.** Three independent levers — `cbs=256`,
  `pool=8/8`, `F1-best-val-ckpt` — each shave another slice off joint_R
  (0.31 → 0.25 → 0.22 → 0.186). The drift family has at least 3
  independent dimensions, not a single shared knob.
- **The pooled/MoS RATIO axis is now well-calibrated** by predictions. Q9
  predicted 0.78–0.82 if fully-additive, 0.703–0.733 if saturating; actual
  iter-20 landed at 0.747 (intermediate). Q9b predicted 0.77–0.79; actual
  iter-21 landed at 0.7767 (in range). **Two consecutive HITs on the
  pooled/MoS axis.**
- **The absolute pooled FoM3 axis is NOT linearly additive across the
  variance/drift family.** iter-19 +1.5%, iter-20 +6.5%, iter-21 −0.83%.
  The pooled axis has plateaued at ~13 870 – 13 944 since iter-16. The
  variance/drift family alone cannot break ~14 k pooled.
- **Mechanism**: F1 (better-quality compressor summary) tightens BOTH
  the per-seed posterior centroids (numerator term: drift) AND the per-seed
  posterior widths (denominator term: contributes to pooled covariance) at
  similar rates. So the **ratio** improves while the **absolute pooled**
  stays flat. The 0.14-nat better compressor produces more-coherent seeds
  but not collectively tighter ones.

## Suspected CNN ceiling near pooled ~14 k

Three independent variance/drift levers in succession landed pooled at
13 087 → 13 944 → 13 829. The pooled axis is "asymptotic" to ~14 k under
the 60k-compressor plain architecture.

To break the ceiling we need a mechanism that is **NOT a variance/drift
lever** — i.e. one that adds new compressor information rather than
re-allocating existing compressor information. Candidates:

1. **Q9c — 120 k compressor with the Q9 stack.** iter-16 showed +5% MoS /
   **+7.5% pooled** vs iter-5 at 120k vs 60k base, so more compressor steps
   add real information. If this +7.5% pooled multiplier compounds with the
   variance/drift stack, predicted pooled lands near 15 k. **This is the
   direct test of "is the ceiling information-limited or
   architecture-limited."**
2. **Q4 — VMIM aux width 256 or 512.** Orthogonal mechanism: tightens the
   VMIM lower bound itself rather than re-allocating its outputs.
   `cnn_vmim_target_stability` at cdim=10 showed no effect; untested at
   cdim=16 with the Q9 stack.
3. **Q1 — resnet50_gn at 120k+.** Larger architecture, but iter-15 collapsed
   at 60k due to undertraining; the 120k variant might express its capacity
   advantage on auto-only.

## Calibration ledger

- 9 magnitude misses on extreme MoS predictions across the campaign.
- Pooled-axis predictions: iter-20 HIT (+5..+15% predicted, +6.5% actual),
  iter-21 MISS (+3..+8% predicted, −0.83% actual).
- Pooled/MoS-ratio predictions: iter-20 mid-range (predicted [0.78,0.82] if
  additive, actual 0.747 = ~45% additive), iter-21 HIT (predicted [0.77,
  0.79], actual 0.777).
- The **ratio axis is the most-calibrated axis** the campaign has produced.
  This suggests the drift-compression mechanism is well-understood, while
  the absolute pooled FoM3 mechanism has a ceiling we haven't modeled yet.

## Links

- [[cnn-auto-cbs256-pool88-partial-stack]] — iter-20 / Q9 result that this
  iter is built on.
- [[cnn-auto-pool-window-is-drift-not-mean-lever]] — iter-19 / first
  drift-only lever finding.
- [[cnn-auto-compressor-last-not-best-ckpt]] — F1 mechanism citation
  (audit A2).
- [[cnn-auto-bug-fom3-rewards-mode-drift]] — the pooled-vs-MoS mode-drift
  bug filing that A4 confirmed empirically.
- [[cnn-auto-pooled-fom3-confirms-mode-drift]] — A4 audit; pooled/MoS
  baseline data.
- [[cnn-auto-question-switch-to-pooled-fom3]] — open Andreas methodology
  question (whether to switch keep-rule to pooled).

## Next action

Q9c (Tier-1, promoted): plain + cbs=256 + pool=8/8 stack at **120 000
compressor steps**. Predicted pooled +3 to +8% vs iter-20 (14.4 k – 15.1 k).
Falsifier: pooled within ±2% of iter-20 → CNN-arch ceiling at ~14 k
confirmed → close ceiling certification on this architecture.
