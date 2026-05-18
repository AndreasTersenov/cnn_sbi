---
name: Plain CNN cdim optimum is 16, not 20
status: closed
tags:
    - cnn-auto-push
created-at: 2026-05-18T19:47:43.379246284Z
closed-at: 2026-05-18T19:47:43.394396455Z
outcome: 'Plain @ 60k: cdim=10->14295, 16->18568 (best), 18->17151, 20->14739. Resnet50 BN @ 120k had cdim=20 as peak — the resnet sweet spot does NOT transfer to plain. Confirmed across lr=5e-4 and lr=1e-3 regimes.'
---
