---
name: Pool window 8/8 is a drift/variance lever, not a mean lever (iter-19)
status: closed
tags:
    - finding
    - audit-A3-followup
    - cnn-auto-push-18-20-2026
created-at: 2026-05-18T23:51:08.3986725Z
outcome: 'Pool window 16/8 -> 8/8 (non-overlap, same 3x3 output, capacity-controlled) directionally improves mode-drift (R(sigma_8) -3.5%, joint_R -19% relative) and gives the best pooled/MoS ratio of the campaign (0.719 vs iter-5 0.694, iter-16 0.711, L1 0.89), but does NOT improve MoS (-1.9% vs iter-5, within noise). Pool window joins the variance/drift family (cbs=256, F1-fix-on-plain): same signature of std compression + per_seed_min improvement without MoS lift. A3 hypothesis [[cnn-auto-bug-pool-window-collapses-spatial]] direction CONFIRMED, magnitude FALSIFIED. Open follow-up: do variance/drift levers stack (cbs=256 + pool=8/8 + F1)?'
---
