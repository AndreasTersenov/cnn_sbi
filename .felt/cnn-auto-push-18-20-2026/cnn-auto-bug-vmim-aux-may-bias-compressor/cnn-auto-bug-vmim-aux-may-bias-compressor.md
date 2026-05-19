---
name: VMIM aux network (--vmim-nf-hidden 128) may bias compressor toward summaries it can model, not max-info (cnn-auto-push, A3) — REFUTED iter-23
status: closed
tags:
    - bug-shape
    - audit-A3
    - cnn-auto-push-18-20-2026
created-at: 2026-05-18T22:15:37.210482858Z
outcome: 'REFUTED by iter-23 (Q4). Widening --vmim-nf-hidden from 128 to 256 on the iter-20 stack at 60k gave: pooled FoM3 -7.2% (12945 vs iter-20 13944, NULL on the +0%/+5% predicted range and below the constitution''s +5% POSITIVE threshold); MoS +6.4% (19874 vs 18673, MISS upward on [-5%, +5%]); joint_R 0.220 → 0.281 (drift WORSE); amended cross-method check FAIL_AMENDED (dJoint/L1 = 0.61 > 0.25). The wider aux NF makes BOTH the VMIM bound looser (mid-flight: best val -12.248 vs iter-20 -12.510, gap 0.629 nats vs iter-20''s 0.139) AND the downstream pooled FoM3 worse, AND the per-seed posterior drift worse. The A3 framing "wider aux = tighter bound = better compressor" is wrong on every axis. The default --vmim-nf-hidden 128 is at or near the joint-stability sweet spot; widening it destabilizes joint compressor + companion-NF training. Q4 is exhausted at cdim=16; do not retry aux 384/512. The MoS +6.4% with pooled -7.2% is the classic mode-drift signature ([[cnn-auto-pooled-fom3-confirms-mode-drift]]) — wider aux makes each seed''s posterior tighter but more inconsistent across seeds.'
---

## Pre-landing mid-flight observation (Ralph iter-16, 2026-05-19 ~03:30 UTC)

Comparing iter-23 (Q4, --vmim-nf-hidden 256) vs iter-20 (same stack
but --vmim-nf-hidden 128 default) compressor val-loss trajectories at
matched steps, both 60k total:

| step  | iter-20 (aux=128) | iter-23 (aux=256) |
|------:|------------------:|------------------:|
|  3000 |  -10.88           |  -10.78           |
| 18000 |  -12.18           |  -12.23           |
| 24000 |  -12.51 (best)    |  -12.25 (best)    |
| 33000 |  -12.43           |  -12.04           |
| 48000 |  -12.42           |  -11.95           |
| 51000 |  -12.10           |  -11.89 (current) |
| 60000 |  -12.37 (last)    |  pending          |

The wider aux NF makes the compressor's joint training **worse** along
two axes:

1. **Best val loss is 0.26 nats higher** (-12.248 vs -12.510). The bound
   is *looser*, not tighter, with the wider companion NF.
2. **Gap (last − best) is 2.5× larger so far** (0.354 vs 0.139). The
   trajectory after the best oscillates rather than gently plateauing.

This **contradicts the A3 hypothesis surface as originally framed**.
A3's premise was "wider companion NF → tighter bound → cleaner gradient
to the compressor". The val-loss observation says: wider companion NF
**destabilizes** joint optimization. The compressor and the companion
NF appear to co-adapt unstably — the companion's extra capacity gets
used to chase fluctuations in the compressor output rather than to
tighten the bound.

### Falsifiable predictions for iter-23's landing FoM3

Three branches, each with a coherent story:

- **Q4 NULL (pooled ≤ 14 720)**: A3 is FALSE. Wider aux NF is a
  destabilizing knob, not a bound-tightening knob. The "VMIM bound is
  the bottleneck" hypothesis is *refuted*. The aux NF default 128 is
  not too small — it's already at the joint-stability sweet spot.
  Action: close [[cnn-auto-bug-vmim-aux-may-bias-compressor]] as
  refuted. Q4 is exhausted at cdim=16; do not try aux 384/512.
- **Q4 POSITIVE (pooled > 14 720)**: A3 is REFRAMED. The mechanism is
  not "tighter VMIM bound" (the bound is looser). It's something else
  — perhaps the destabilized trajectory accidentally samples a
  better-compressed manifold, or the wider aux's chaotic late-stage
  gradients act as a regularization that the deterministic LR schedule
  doesn't supply. Either way, the *premise* (wider = tighter bound) is
  wrong, but the *prediction* (Q4 helps) is right. Close A3, open a
  follow-up sub-fiber pointing at the new mechanism.
- **Q4 NEAR-NULL (pooled in (14 220, 14 720])**: Q4 is a wash.
  Destabilization and any benefit cancel. Close A3 as exhausted on
  inconclusive evidence; don't sweep further.

In all three branches the **specific framing** "VMIM aux 128 is too
small" is dead. What replaces it depends on the FoM3 verdict.

### Why this matters for ceiling certification

The constitution's ceiling-evidence checklist requires every Tier-2
hypothesis tested or deferred with justification. The iter-16 closing
analysis (already in [[cnn-auto-piecewise-schedule-hurts-at-120k]])
showed a similar pattern at the LR-schedule level: worse last-step
val loss did NOT block iter-16's +5% MoS / +7.5% pooled lift over
iter-5. So the "VMIM bound = FoM3 ceiling" mental model has been
flawed twice now (iter-16 + iter-23). The ceiling-evidence document
should refer to this as a *general* observation: the compressor's
VMIM val loss and the downstream NDE's FoM3 are partially decoupled,
and treating the bound as a proxy for the metric mis-predicts the
direction of training-curve changes.

### Sources

- iter-20 trajectory: `/nas/tersenov/claude-notes/runs/cnn-auto-push-18-20-2026/iter-20/compressor/nobnt/vmim/nbody/sigma_0.26/gal_density_30/bin_4/loss_compressor_test.npy` (20 pts).
- iter-23 trajectory: same path under `iter-23` (17 pts so far, lands ~03:43 UTC).
- iter-23 manifest confirms `vmim_nf_hidden: 256` (only knob change vs iter-20).
- Pre-landing snapshot: this section was written before iter-23 produced its FoM3 number, so the predictions above are falsifiable rather than retrofitted.

---

## Post-landing resolution (Ralph iter-16, 2026-05-19 ~03:50 UTC)

iter-23 produced its FoM3 numbers via `landing_analysis.py`:

| metric         | iter-20 (aux=128) | iter-23 (aux=256) | Δ          | predicted band       |
|----------------|------------------:|------------------:|-----------:|----------------------|
| MoS FoM3       |          18 673   |          19 874   |  +6.43 %   | [-5%, +5%] → **MISS up** |
| pooled FoM3    |          13 944   |          12 945   |  -7.16 %   | [+0%, +5%] → **MISS down (NULL)** |
| joint_R        |          0.220    |          0.281    |  +0.061    | drift worsened       |
| amended check  |          FAIL     |          FAIL     |  worse     | dJoint/L1 0.259 → 0.61 |
| per_seed CoV   |          —        |          9.5 %    |            | within Guard         |
| best val loss  |          -12.510  |          -12.248  |  +0.26     | bound LOOSER         |
| argmin→last gap|          0.139    |          0.629    |  4.5× larger | bound DESTABILIZED |

Three branches were registered before landing. The **Q4 NULL** branch
fired cleanly on pooled (the constitution's classification axis), but
with an unusual MoS *upward* miss that requires a small refinement to
the story. Below.

### Verdict against each pre-landing branch

- **Q4 NULL (predicted: pooled ≤ 14 720)**: ✅ FIRED. pooled = 12 945 < 14 720.
  A3 is FALSE *as framed*. Wider aux NF is a destabilizing knob, not a
  bound-tightening knob — confirmed at every measurable axis (best val,
  pooled, joint_R, dJoint/L1).
- **Q4 POSITIVE (predicted: pooled > 14 720)**: ❌ DID NOT FIRE.
- **Q4 NEAR-NULL (predicted: pooled in (14 220, 14 720])**: ❌ DID NOT FIRE.
  pooled is below 14 220 by a margin.

### Refinement — the MoS upward miss

MoS +6.43 % (19 874 vs 18 673) is the only metric that improved, and
it is **outside the +5% upper edge of the predicted band**. Two
mechanisms could explain it; both are consistent with the campaign's
established lessons:

1. **Mode-drift signature on a 3-seed estimate**. Each seed's marginal
   posterior is *tighter* with the wider aux NF (because the looser
   bound + chaotic late training gradients act as a stochastic
   regularizer that the deterministic LR schedule doesn't supply, OR
   because the wider aux's flexibility absorbs noise rather than passing
   it through). But the per-seed *centroids* drift more, so the
   POOLED contour is *wider*. The mean-of-seeds metric averages the
   tight individual contours and rewards tightness; pooled penalizes
   drift. See [[cnn-auto-pooled-fom3-confirms-mode-drift]] for the
   established mechanism; iter-23 is the cleanest single-knob
   demonstration so far (one knob change, +6.4% MoS, -7.2% pooled —
   the exact decoupling the audit predicted).
2. **3-seed sampling noise**. With CoV 9.5 %, a 3-seed MoS has roughly
   ±5.5 % standard error of the mean. A +6.4 % uplift is ≈ 1.16 σ —
   could be noise. Resolution: would need 5-seed replication to know.
   Not worth the compute since pooled and joint_R both moved the wrong
   way regardless.

The MoS upward miss does NOT rescue A3 — pooled (the campaign's
ceiling axis) moved the wrong way by ~3× the standard error.

### Decision

Close [[cnn-auto-bug-vmim-aux-may-bias-compressor]] as **refuted**.
Outcome edited above with the verdict numbers. No further sweeps of
`vmim_nf_hidden` at cdim=16 are warranted.

### Implication for the ceiling-evidence story

The iter-23 verdict is **decisive** on the ceiling-evidence Tier-2 Q4
checkbox: Q4 is tested and falsified. This flips the Q4 row in
`CEILING_EVIDENCE.md` from OPEN to CLOSED (refuted).

The remaining ceiling-evidence work blocked on iter-22 is the Q9c
verdict on whether the variance/drift family compounds with the Q2
information lever at 120k compressor. Once that lands (~04:40 UTC),
`integrate_landings.py` will produce the combined headline and the
ceiling-evidence sub-fiber can close.
