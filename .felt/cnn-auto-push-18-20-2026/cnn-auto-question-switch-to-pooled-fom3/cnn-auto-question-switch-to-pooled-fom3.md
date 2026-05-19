---
name: Switch autoresearch keep-rule from mean-of-seeds to pooled FoM3?
status: open
tags:
    - question
    - cnn-auto-push-18-20-2026
created-at: 2026-05-18T22:34:42.667533495Z
outcome: 'Mode-drift audit (A4) shows CNN MoS gain since iter-0 is +30% (14295->18568) but pooled gain is only +10% (11700->12894). pooled/mos = 0.69 vs L1 ref 0.89. Question: change the autoresearch keep-rule to use pooled FoM3 (primary) with MoS as a secondary report? Risk: pooled is dominated by per-seed centroid drift which may settle as compressor trains longer; switching too early may discard improvements that DO compound in pooled at 240k. Proposed compromise: report BOTH; require pooled to clear half-noise OR keep showing MoS gain >0.5*std AND pooled/mos ratio not degrading by >5%.'
---

# Question for Andreas — keep-rule for future Ralph campaigns

Filed Ralph iter-5 audit A4 (2026-05-18); stays open by design. Outcome
above captures the audit-time framing; this body captures the
post-campaign evidence so the decision can rest on the *whole* run, not
on the iter-5 snapshot.

## Why this needs to be answered before the next campaign

The constitution's keep-rule was mean-of-seeds (MoS), with a guard on
per-seed-min. The mode-drift audit (A4) showed pooled FoM3 lagging MoS
across the campaign. The full campaign has now resolved that lag into
something more specific: **MoS and pooled are not just different scalars
on the same axis — they fire in opposite directions on the campaign's
sharpest single-knob experiments.**

## Post-campaign evidence (now that the ceiling is certified)

### 1. Single-knob opposite-direction firings

The campaign produced two clean single-knob iterations where MoS and
pooled disagreed in sign:

| iter | knob change                          | MoS Δ vs ref | pooled Δ vs ref | joint_R Δ | which axis "wins"? |
|-----:|--------------------------------------|-------------:|----------------:|----------:|--------------------|
| 23   | iter-20 stack + `--vmim-nf-hidden 256` | **+6.4%**    | **-7.2%**       | 0.220 → 0.281 (worse) | MoS keeps; pooled discards |
| 22   | iter-20 stack + 120k compressor (Q9c) | +3.4%        | **-10.1%**      | 0.220 → 0.272 (worse) | MoS keeps; pooled discards |

These are the cleanest demonstrations the campaign has produced that the
two axes are not just noisy approximations of each other. A wider VMIM
aux NF makes per-seed posteriors *tighter* (MoS up) but more
*inconsistent across seeds* (drift up, pooled down). MoS rewards the
tightness; pooled penalizes the drift. **Under a MoS-only keep-rule both
iter-22 and iter-23 would have to be marked as "ceiling-falsifier
POSITIVE" candidates** even though both make the campaign's true science
target (information extraction across the parameter space) worse.

### 2. Amended-check selectivity across landed iterations

The Ralph iter-13 sweep (`landing_analysis.py`, recorded in
[[cnn-auto-amended-check-rejects-iter5-iter19-iter20]]) scored all five
historically-landed iters against the constitution's amended
3-component check (pooled ratio ≥ 0.35, |Δjoint_R|/joint_R_L1 ≤ 0.25,
MoS ratio ≥ 0.40):

| iter | MoS rank | pooled rank | amended verdict |
|-----:|---------:|------------:|-----------------|
| 5    | 4        | 5           | FAIL on joint_R drift |
| 16   | 1        | 2           | **PASS**          |
| 19   | 5        | 4           | FAIL on joint_R drift |
| 20   | 3        | 1 (best)    | FAIL on joint_R drift |
| 21   | 6 (last) | 3           | **PASS** (best margin) |

**Pure pooled-keep would crown iter-20** (highest pooled). But iter-20
**FAILS** the amended cross-method check on per-seed drift, so it would
not satisfy the constitution's ceiling-certification box even if kept.
**MoS-keep crowns iter-16** (highest MoS), which is *also* the
amended-check passer and ends up being the certified representative
best. **Neither pure scalar gets the right answer alone**; the amended
3-component check (which combines pooled, joint_R, and MoS ratios) is
what's actually selective on the campaign's data.

### 3. Pooled lag at 240k vs 60k — does it settle?

Open question, not answered by this campaign. The 240k a2 baseline
(22 633 MoS at seeds 41/42/43) was never re-run as pooled. Andreas's
manual 240k promotion of iter-16 will produce the first 240k pooled
number; that will tell us whether pooled/MoS climbs from ~0.71 (iter-16
60k+120k-compressor) toward L1's 0.89 at full training, or whether the
ratio is bounded by architecture rather than steps.

If pooled/MoS rises with steps, then **the autoresearch-time keep-rule
penalty for pooled-discarding a MoS-positive screening iter is real**
(we'd be discarding improvements that compound). If pooled/MoS is
bounded by architecture, then **the penalty is zero** and pooled-keep is
strictly better at screening time.

## Three options for the next campaign

**(O1) MoS-primary + amended cross-method check as gate.** Keep the
historical keep-rule (MoS gain > 0.5 × noise), but gate every kept
iteration on the amended 3-component check passing. The unit-test on
this campaign's data: iter-20 would have been *kept* but blocked from
"current best" because amended-check FAILS; iter-16 + iter-21 pass
through cleanly. Probably the lowest-friction migration path. Cost:
amended-check on each iter requires the cross-method overlay (already
mandatory per PHASE 5.5d).

**(O2) Pooled-primary, MoS as secondary report.** Change the keep-rule
to "pooled clears half-noise". The unit-test on this campaign's data:
iter-22 + iter-23 would have been discarded (correct); iter-16 would
have been kept as cleanest pooled gain (correct); iter-20 would have
been kept as pooled-best (also correct, modulo amended-check). Risk:
discarding a MoS-positive screening iter whose pooled would have
compounded at 240k. Mitigated by always running the 240k promotion on
the amended-check passer before declaring the ceiling.

**(O3) Both-primary AND-gate.** Require pooled to clear half-noise AND
MoS gain > 0.5 × noise AND amended-check pass. The unit-test on this
campaign: iter-16 passes all three (cleanly); iter-21 passes pooled +
amended but its MoS is below iter-5's; iter-20 fails amended. The
strictest. Risk: rejects every iteration in this campaign except
iter-16, which is fine *post hoc* but might be too restrictive
*in-flight* (could starve the loop of "keep" events early).

## Recommendation (Ralph's, for Andreas to accept / reject / amend)

**Default to (O1)** for the next cross-only or 240k-promotion campaign.
The amended-check is already implemented (`landing_analysis.py`), is
selective in practice (rejects 3 of 5 landed iters on this campaign),
and changes the constitution surface less than (O2)/(O3). The pooled
axis becomes a *gate*, not a *primary metric*, so MoS-positive
screening iterations still surface but cannot become "best" unless they
pass amended.

**Re-evaluate after iter-16's 240k promotion lands.** If 240k pooled
clears the half-noise threshold relative to iter-16's 60k pooled (~14 k),
then pooled compounds with steps and (O1)'s gating approach was right.
If 240k pooled stays flat or regresses, then the ceiling is structural
and (O2) or (O3) is warranted for the next architecture-change campaign.

## Related fibers

- [[cnn-auto-pooled-fom3-confirms-mode-drift]] — A4 audit, the
  empirical confirmation.
- [[cnn-auto-pooled-vs-mos]] — the pooled/MoS divergence finding that
  motivated the question.
- [[cnn-auto-pooled-ratio-amendment-rationale]] — amended-check
  proposal (pooled ratio + joint_R + MoS ratio, three thresholds).
- [[cnn-auto-amended-check-rejects-iter5-iter19-iter20]] — selectivity
  evidence (3 of 5 landed iters fail).
- [[cnn-auto-pooled-fom3-ceiling-near-14k]] — the architectural pooled
  ceiling argument.
- [[cnn-auto-bug-fom3-rewards-mode-drift]] — A3 challenge 3, confirmed.
