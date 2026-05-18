---
name: OOM at compress-dataset step when 5+ jobs run concurrent
status: closed
tags:
    - cnn-auto-push
created-at: 2026-05-18T19:47:43.594379351Z
closed-at: 2026-05-18T19:47:43.601528753Z
outcome: 'iter-12 crashed: ds-batch=500 in compress_dataset allocates ~1 GB intermediate conv (f32[500,64,80,80]). With 5 concurrent jobs at peak Phase-A handoff, cumulative reservations exceed 40 GB. Mitigation: lower --ds-batch-size when running many parallel, or stagger Phase-A starts so the compress steps don''t overlap.'
---
