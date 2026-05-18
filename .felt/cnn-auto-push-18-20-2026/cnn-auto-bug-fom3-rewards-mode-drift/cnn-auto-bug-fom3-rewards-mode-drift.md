---
name: mean-of-seeds FoM3 is biased toward tight-but-drifty estimators; pooled ratio 0.63-0.79 reveals it (cnn-auto-push, A3)
status: closed
tags:
    - bug-shape
    - audit-A3
    - cnn-auto-push-18-20-2026
created-at: 2026-05-18T22:15:47.181282164Z
closed-at: 2026-05-18T22:38:35.361372157Z
outcome: 'CONFIRMED empirically by audit A4 (Ralph iter-5, 2026-05-18 22:35 UTC). See [[cnn-auto-pooled-fom3-confirms-mode-drift]]: CNN kept/tie iters have pooled/mos FoM3 = 0.63-0.69 vs L1 reference 0.89; +30% MoS gain since iter-0 is only +10% in pooled. The bug is real; the loop has been climbing a misleading metric. Open methodology question [[cnn-auto-question-switch-to-pooled-fom3]] surfaces the keep-rule change to Andreas. Diagnostic recipe (compute_mode_drift.py) lives in audits/2026-05-18_A_mode_drift/.'
---
