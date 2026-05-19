---
name: Deferred Q1 — resnet50_gn cdim=20 lr=1e-3 at 120k compressor never tested (architecture-change scope)
status: closed
tags:
    - deferred-question
    - cnn-auto-push-18-20-2026
    - tier-1-followup
created-at: 2026-05-19T02:15:47.228004955Z
closed-at: 2026-05-19T03:19:52.552947609Z
outcome: 'Closes the Q1-at-120k Tier-1 checklist box. Q1 at 60k (iter-15) collapsed -52.8% due to undertraining (compressor val loss -11.15 vs plain -12.44). 120k retest should be its own architecture-change-scope campaign, not block ceiling certification on this fiber. Inputs needed: confirm GroupNorm doesn''t undertrain at 120k (cnn_resnet50_zm_sweep evidence is BN-only); rerun iter-15 config at --total-steps 120000.'
---
