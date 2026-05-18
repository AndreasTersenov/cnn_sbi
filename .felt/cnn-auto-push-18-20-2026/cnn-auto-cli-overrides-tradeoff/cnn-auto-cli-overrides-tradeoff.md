---
name: CLI overrides for parallel iterations break atomic-commit autoresearch model
status: closed
tags:
    - cnn-auto-push
created-at: 2026-05-18T19:47:43.789943531Z
closed-at: 2026-05-18T19:47:43.798827249Z
outcome: 'iters 7,8,9,10,11,12,13,14,15 used CLI overrides for hyperparameters; only iter-6 had a corresponding commit (later reverted). Trade: ~3x throughput vs weaker provenance chain. metadata/iter-N_*.json captures the actual params. For Ralph: if you want strict atomicity, commit each change before running; if you want speed, CLI override + JSON provenance is fine.'
---
