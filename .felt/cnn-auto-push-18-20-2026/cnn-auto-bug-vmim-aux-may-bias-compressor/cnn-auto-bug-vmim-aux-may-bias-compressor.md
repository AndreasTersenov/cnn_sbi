---
name: VMIM aux network (--vmim-nf-hidden 128) may bias compressor toward summaries it can model, not max-info (cnn-auto-push, A3)
status: open
tags:
    - bug-shape
    - audit-A3
    - cnn-auto-push-18-20-2026
created-at: 2026-05-18T22:15:37.210482858Z
outcome: |-
    VMIM trains compressor by maximizing E[log q(θ|s)] where q is a small RealNVP companion network (hidden 128). If q is underexpressive, the compressor learns summaries q can model — not summaries that maximize true I(s;θ). Inference uses a DIFFERENT, larger NDE (hidden 256). Bound looseness directly suppresses compressor quality. Diagnostic: --vmim-nf-hidden 256 and 512 on iter-5 config; if FoM3 changes >5%, the bound is the bottleneck. File: scripts/sbi/npe_cnn_nbody_tomo.py:train_compressor_vmim (~1736-2270). Already on Tier-2 EV queue as Q4; A3 elevates to bug-shape. Mid-flight diagnostic (Ralph iter-16, iter-23 at step 51000): wider aux NF (128→256) does NOT tighten the bound — best val loss is 0.26 nats WORSE (-12.248 vs iter-20 -12.510, same stack), and trajectory is oscillatory rather than monotonic-descent. The A3 "wider = tighter bound" framing is contradicted at the val-loss level; whatever Q4 does to FoM3 is decoupled from the bound.
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
