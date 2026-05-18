---
name: Compressor pool window 16/stride 8 over 20x20 features = mostly-global pooling (cnn-auto-push, A3)
status: closed
tags:
    - bug-shape
    - audit-A3
    - cnn-auto-push-18-20-2026
created-at: 2026-05-18T22:15:26.895325607Z
closed-at: 2026-05-18T23:51:16.731124494Z
outcome: TESTED in iter-19 (pool_window 16->8, pool_stride 8->8 SAME). Direction CONFIRMED on mode-drift (R(sigma_8) -3.5%, joint_R -19%, pooled/MoS ratio 0.694 -> 0.719 — best in campaign). Magnitude FALSIFIED on MoS (predicted +3-15%, actual -1.9%, within noise). Pool window is a drift/variance lever, NOT a mean lever. Pattern matches cbs=256 and F1-fix. See [[cnn-auto-pool-window-is-drift-not-mean-lever]] for the resolved-finding writeup.
---
