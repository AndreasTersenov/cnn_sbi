---
name: mean-of-seeds FoM3 is biased toward tight-but-drifty estimators; pooled ratio 0.63-0.79 reveals it (cnn-auto-push, A3)
tags:
    - bug-shape
    - audit-A3
    - cnn-auto-push-18-20-2026
created-at: 2026-05-18T22:15:47.181282164Z
outcome: 'Autoresearch metric is mean-of-seeds FoM3 = mean over seeds of 1/sqrt(det(C_seed)). With per-seed posterior mode drift > per-seed posterior width, the per-seed covs are small (each seed thinks it knows the answer precisely) but the cov of the pooled samples is large (modes scatter). render_overlay.py reports pooled/mos ratio: CNN at 0.63-0.79 vs L1 at 0.89. The metric rewards confidently-wrong estimators. Diagnostic: for each kept iter, log std-of-per-seed-posterior-means / mean-of-per-seed-posterior-stds in the (Ω_m,σ_8,w_0) subspace. If >>1, mode drift dominates. Implication: pooled FoM3 or a mode-aware metric (e.g. coverage-corrected) may be a better optimization target. Existing sub-fiber [[cnn-auto-pooled-vs-mos]] documents the finding; this one elevates it to a methodological challenge.'
---
