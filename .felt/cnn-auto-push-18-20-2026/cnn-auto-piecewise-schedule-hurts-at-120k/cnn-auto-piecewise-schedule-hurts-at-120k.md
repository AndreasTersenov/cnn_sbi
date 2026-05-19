---
name: Piecewise LR schedule diverges late at 120k+ — VMIM-bound regression that FoM3 partly absorbs
status: closed
tags:
    - finding
    - cnn-auto-push-18-20-2026
    - ceiling-evidence
created-at: 2026-05-18T22:19:09.548154068Z
outcome: |-
    The piecewise compressor LR schedule does not scale gracefully with total_steps.
    Confirmed across two 120k runs:

      iter-16 (Q2):     argmin -12.521 @ step 33k (28% of 120k), last-step -11.171, gap 1.35 nats
      iter-22 (Q9c):    argmin -12.247 @ step 18k (15% of 54k seen so far), last-step (54k) -12.074
      vs iter-5 (60k):  argmin -12.722 @ step 48k (80% of 60k),  last-step -12.440, gap 0.28 nats

    Best val loss at 120k is WORSE than at 60k (-12.52 vs -12.72) AND last-step
    is much worse (-11.17 vs -12.44). Surprise: iter-16's FoM3 still beats
    iter-5 by +5% MoS / +7.5% pooled — the auxiliary NF (nvp-hidden=128) is
    likely the binding constraint on the VMIM bound, not the compressor.
    The NDE in Stage B is flexible enough to recover information from the
    "diverged" compressor.

    Two implications:
    (1) The 1.3-nat F1 lever on 120k runs is much bigger than F1 at 60k
        (0.28 nats; see [[cnn-auto-f1-variance-not-mean-lever]]). If F1 lifts
        variance/per_seed_min in proportion to argmin-to-final gap, F1 on
        iter-16 should give ~5x bigger per_seed_min lift than F1 on iter-5.
        This is free upside for the 240k promotion.
    (2) Q4 (iter-23) is the live test of the "aux NF bound-limits, not
        compressor" hypothesis — if vmim-hidden 128→256 lifts FoM3, the
        diverging val loss is mostly NF-tracking noise, not real
        compressor regression. If null, the FoM3 lift at 120k is from a
        different mechanism (compressor schedule diversity).

    Originally premised "Q2 FALSIFIED" but iter-16's actual FoM3 result
    overturned that — Q2 is *partially* true at the FoM3 level while
    *falsified* at the VMIM-bound level. The Tier-2 Q7 (cosine LR
    schedule) promotion is still defensible but lower-EV than the F1
    + Q4 paths.
description: Confirmed iter-16 + iter-22 — at 120k, compressor val loss hits argmin at 15-28% of training then climbs by ≥1 nat; last-step is worse than 60k baseline. FoM3 absorbs the regression via NDE flexibility but a 1.3-nat F1 lever is now visible on these long runs.
---
