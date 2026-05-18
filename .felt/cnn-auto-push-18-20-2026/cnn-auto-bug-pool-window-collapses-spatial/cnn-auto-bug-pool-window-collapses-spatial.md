---
name: Compressor pool window 16/stride 8 over 20x20 features = mostly-global pooling (cnn-auto-push, A3)
status: open
tags:
    - bug-shape
    - audit-A3
    - cnn-auto-push-18-20-2026
created-at: 2026-05-18T22:15:26.895325607Z
outcome: 'Plain compressor''s hk.AvgPool(16, 8, ''SAME'') over (20,20,256) feature map produces (3,3,256) — pool window is 80% of spatial extent with 40% stride. Discriminative spatial features after the conv trunk are mostly averaged out. Diagnostic: test pool_window=4 stride=2 or pool_window=20 (true global) to bracket. If FoM3 changes >5%, this is a real bottleneck.'
---
