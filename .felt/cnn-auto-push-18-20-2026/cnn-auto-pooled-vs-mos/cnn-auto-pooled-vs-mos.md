---
name: 'Pooled FoM3 << mean-of-seeds: per-seed posterior modes drift'
status: closed
tags:
    - cnn-auto-push
created-at: 2026-05-18T20:22:36.031184883Z
closed-at: 2026-05-18T20:22:36.04444153Z
outcome: 'Renderer computed both: iter-5 mean-of-seeds=18568 vs pooled=12894 (ratio 0.69); iter-14 mos=18822 vs pooled=11851 (ratio 0.63). L1 auto+cross ref ratio=0.89 (seeds agree). The CNN-iter-best''s 3 seeds find DIFFERENT posterior modes — pooling spreads the covariance. Implications: (a) mean-of-seeds overestimates the actual CNN constraining power; (b) 240k promotion of any current best is at risk because the per-seed drift may persist; (c) the autoresearch metric (mean-of-seeds) is partially gaming itself — a config with high mean and low pooled is partly winning by ''lucky-seed averaging'', not true constraining power.'
---
