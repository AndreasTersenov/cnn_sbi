---
name: Deferred Q8 — resnet50 (stock BN) at cdim=20 lr=1e-3 60k never tested on auto-only (architecture-change scope; BN contamination prior)
status: closed
tags:
    - deferred-question
    - tier-2-followup
    - cnn-auto-push-18-20-2026
created-at: 2026-05-19T02:25:00.109621798Z
closed-at: 2026-05-19T02:25:11.329816604Z
outcome: 'Closes the Q8 Tier-2 checklist box. Constitution Tier-2 item; sibling of deferred Q1 (resnet50_gn-at-120k). Strong prior for collapse: project_resnet_bn_contamination memory shows stock BN ResNet50 on 10-ch harmonic input gives FoM3~700 (BN running stats average across cosmology-mixed batches; GN restores). On auto-only (4-ch tomographic) the contamination mechanism is unchanged — same cosmology-mixed batches. iter-15''s resnet50_gn collapse at 60k (-52.8%) was undertraining not BN contamination, but a stock-BN variant would compound both pathologies. Out of fiber scope (Tier-3 architecture swap per constitution; requires explicit Andreas authorization). Inputs needed if revisited: --compressor-arch resnet50 (not _gn); same iter-15 config; expect a Guard-fail collapse below 11k floor with high probability.'
---
