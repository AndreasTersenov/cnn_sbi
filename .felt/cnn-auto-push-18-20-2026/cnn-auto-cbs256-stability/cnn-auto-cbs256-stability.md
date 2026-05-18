---
name: cbs=256 trades mean for 5x tighter seed scatter
status: closed
tags:
    - cnn-auto-push
created-at: 2026-05-18T19:47:43.467986623Z
closed-at: 2026-05-18T19:47:43.477922427Z
outcome: 'iter-4 (cbs=256, lr=5e-4): mean 15589 vs iter-1 16149 (-3.5%), std 303 vs 1532 (5x tighter). iter-11 (cbs=256, lr=1e-3): mean 17134 vs iter-5 18568 (-7.7%), std 496 vs 1604 (3.2x tighter). The ''cbs=256 + winning LR composes to strict improvement'' hypothesis FAILED — both LR values give same pattern: ~7% mean cost for 3-5x tighter scatter. cbs=256 is a robustness-vs-mean tradeoff, not free stability.'
---
