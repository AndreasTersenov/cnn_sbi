---
name: cnn-auto-cbs256-pool88-partial-stack
status: closed
tags:
    - finding
    - cnn-auto-push-18-20-2026
    - variance-drift-family
created-at: 2026-05-19T01:27:25.876593762Z
outcome: |-
  iter-20 (cbs=256 + pool=8/8 stack): pooled FoM3 +6.5% vs iter-19 (in predicted +5-15% range — first time iter lands inside a predicted pooled range), pooled/MoS 0.719 -> 0.747 (intermediate between full-additive 0.78-0.82 and saturation 0.703-0.733; ~45% of full additivity). NEW POOLED BEST (13944 vs iter-16 13868). Drift compression concentrates on w_0 (centroid scatter 0.0347 -> 0.0255 -> 0.0124 across iter-5/19/20, 2.8x). Std-of-MoS widens (904 -> 2014, s43 disperses) — levers compete on std but cooperate on pooled. Compressor argmin earlier and cleaner (gap 0.14 nats, best of campaign).
---

# cbs=256 + pool=8/8 stack partially-additively on pooled FoM3

**Context.** Across the campaign we identified three variance/drift-compression
levers — **cbs=256** (iter-4/11), **F1 best-val ckpt on plain** (iter-17), and
**pool=8/8** (iter-19) — all with the same signature: tighter per-seed
posteriors, MoS within noise, modest pooled gain, improving pooled/MoS ratio.
iter-20 tested whether two of them (cbs=256 + pool=8/8) stack on top of each
other and how much of the predicted additive gain materializes.

**Result.**

| Metric              | iter-5 | iter-19 (pool) | iter-20 (cbs+pool) | Pred. iter-20 if additive |
|---------------------|--------|----------------|--------------------|---------------------------|
| MoS FoM3            | 18 568 | 18 213         | **18 673**         | 18 213 ± 5 %              |
| Pooled FoM3         | 12 894 | 13 087         | **13 944**         | 13 740–15 050             |
| pooled / MoS        | 0.694  | 0.719          | **0.747**          | 0.78–0.82                 |
| std (3-seed)        | 1 604  | 904            | 2 014              | ≤ 904                     |
| per_seed_min        | 16 368 | 17 187         | 16 158             | ≥ 17 187                  |
| joint_R             | 0.309  | 0.251          | **0.220**          | < 0.20                    |
| R(Ω_m)              | 0.527  | 0.528          | 0.575              | —                         |
| R(σ_8)              | 0.779  | 0.752          | 0.772              | —                         |
| R(w_0)              | 0.234  | 0.162          | **0.080**          | —                         |
| centroid scatter w_0 | 0.0347 | 0.0255         | **0.0124**         | —                         |

**Three observations.**

1. **Pooled gain materializes in the predicted +5–15 % range** (+6.5 %).
   First iter in 8 to land within a predicted pooled range. Pooled axis is
   calibrated.
2. **Pooled/MoS lands ABOVE saturation but BELOW full additivity** —
   intermediate, ~45 % of the predicted additive gap (0.747 vs the predicted
   0.78–0.82). The drift-compression budgets are shared but not identical.
3. **Std-of-MoS widens** (904 → 2014). One seed (s43) dispersed. The levers
   compete on std-of-MoS but cooperate on pooled-of-3-seed-concatenation.
   **Std-of-MoS and pooled FoM3 are decoupled targets.** Tightening
   per-seed-posterior variance is one mechanism; compressing centroid
   drift is another; cbs=256 trades the former for more of the latter.

**Why w_0 and not σ_8.** Audit A4 identified σ_8 as the dominant drift axis
by *absolute R*. But the *relative compression vs baseline* is happening
dominantly on w_0: R(w_0) compresses 2.9× (0.234 → 0.080) across the iter
chain, vs R(σ_8) compresses only 1.04× (0.779 → 0.772). w_0 was the easier
axis to compress and is now near-resolved (R ≈ 0.08 means centroid scatter is
8 % of posterior width — seeds essentially agree on w_0 centroid).

**Mechanism.** cbs=256 produces a cleaner compressor gradient signal —
argmin val loss reached at step 24 000 (40 % of training) with the smallest
argmin-to-final gap of the campaign (0.14 nats vs iter-5's 0.28 and
iter-19's 0.22). The compressor settles into a fixed point earlier, and the
piecewise schedule's later phases don't perturb it. This in turn produces a
less seed-specific summary: when the compressor is the same fixed point
across seeds (modulo the train_seed=42 weight init), the per-seed posterior
modes can't drift in directions the compressor doesn't see.

**Open questions.**

- Does F1 stack on top (Q9b)? Predicted pooled +3–8 %, pooled/MoS → 0.77–0.79.
- Does this stack survive 120k compressor (Q9c)? iter-16 + Q9 combo.
- Why does std-of-MoS widen rather than tighten? File [[cnn-auto-std-of-mos-vs-pooled-decoupled]]
  if Q9b reveals the mechanism.

**Provenance.** Tested in iter-20 against iter-19 baseline. Metadata at
`/nas/tersenov/claude-notes/runs/cnn-auto-push-18-20-2026/metadata/iter-20_q9_cbs256_pool8s8.json`.
Predicted in playbook Q9; result spawned playbook Q9b (next move) and Q9c.

**Linked**: [[cnn-auto-pool-window-is-drift-not-mean-lever]] (iter-19),
[[cnn-auto-cbs256-stability]] (iter-4/11),
[[cnn-auto-pooled-fom3-confirms-mode-drift]] (audit A4),
[[cnn-auto-question-switch-to-pooled-fom3]] (open methodology question — iter-20
strengthens the case for switching).
