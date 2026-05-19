---
name: Deferred Q7 — compressor LR schedule variants (cosine/warmup+cosine/slower piecewise) untested (low-EV, premise downgraded after iter-16)
status: closed
tags:
    - deferred-question
    - tier-2-followup
    - cnn-auto-push-18-20-2026
created-at: 2026-05-19T02:24:48.372172915Z
closed-at: 2026-05-19T02:25:11.322851019Z
outcome: 'Closes the Q7 Tier-2 checklist box. Constitution Tier-2 item: try cosine, slower piecewise (0.5x), or warmup+cosine instead of the inner script''s default piecewise (0.7x every 10%, first 2/3 of training). Premise was ''piecewise wandering destroys high-LR phase''; iter-16''s clean 120k run with the SAME piecewise schedule gave +7.5% pooled — premise FALSIFIED (the wandering is benign, not destructive). EV of testing more schedule shapes is therefore low; the dominant LR effect is the lr_init value (iter-5 1e-3 confirmed optimum), not the decay curve. Inputs needed if revisited: --compressor-lr-schedule {cosine,warmup_cosine,piecewise_slow} (new flag); run on iter-20 stack at 60k; expect ±3% on pooled (within noise floor).'
---
