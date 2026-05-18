---
name: Piecewise LR schedule hurts compressor at 120k vs 60k — Q2 falsified pre-iter-16-completion (cnn-auto-push)
tags:
    - finding
    - cnn-auto-push-18-20-2026
created-at: 2026-05-18T22:19:09.548154068Z
outcome: 'iter-16 (plain cdim=16 lr=1e-3 at 120k compressor steps) val loss reached argmin -12.52 at step 33k (39%) then wandered up to -11.86 at step 84k. iter-5 (same config at 60k) had argmin -12.72 at step 48k (80%). At 120k the schedule''s bulk-LR phase is wider in absolute steps, model escapes the optimum and the polish phase (begins at 0.6*total_steps = step 72k for 120k vs step 36k for 60k) doesn''t recover. Q2 (more compressor steps = better) FALSIFIED. The LR schedule does not scale gracefully with total_steps; longer training HURTS unless the schedule is redesigned. Mechanism: piecewise_constant decay points at fractions of (2/3)*total_steps, so each chunk grows linearly with total_steps. Step 33k is the noise-floor of the bulk-LR phase; longer bulk training adds noise, not signal. Implication: Tier-2 Q7 (cosine LR schedule) PROMOTED to Tier-1. Try cosine at 120k OR shorter piecewise that polishes by step 30k regardless of total.'
---
