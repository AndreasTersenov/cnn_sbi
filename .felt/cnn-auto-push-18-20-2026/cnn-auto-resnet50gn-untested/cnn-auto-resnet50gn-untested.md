---
name: resnet50_gn auto-only never run before iter-15
status: closed
tags:
    - cnn-auto-push
created-at: 2026-05-18T19:47:43.689559993Z
closed-at: 2026-05-18T19:47:43.710432762Z
outcome: 'Constitution''s 2nd arm. Prior sweeps tried resnet50 stock BN (cnn_resnet50_zm_sweep) on auto-only: cdim=20 hit 27668 at 120k — well above plain 240k baseline (22633). resnet50_gn arch was developed for harmonic multi-channel where BN contaminates (project_resnet_bn_contamination). On auto-only BN may be fine. iter-15 (resnet50_gn cdim=20 lr=1e-3) in flight tests this.'
---
