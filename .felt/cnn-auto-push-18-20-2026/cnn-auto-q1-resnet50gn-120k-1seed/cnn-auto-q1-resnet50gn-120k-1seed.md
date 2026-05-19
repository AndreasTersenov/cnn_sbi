---
name: Q1 1-seed pilot — resnet50_gn cdim=20 @120k with best_val ckpt policy
status: closed
tags:
    - experiment
    - cnn-auto-push-18-20-2026
    - tier-1-followup
created-at: 2026-05-19T09:41:22.131004808Z
closed-at: 2026-05-19T14:04:56.752070584Z
outcome: 'Q1 NULL on every axis. resnet50_gn cdim=20 @120k 1-seed (post-bug-fix, best_val policy): single-seed FoM3 = 11820 (-39% vs iter-16 MoS 19502; -15% vs iter-16 pooled 13868). VMIM bound at convergence: best val -11.91 at step 42000 (argmin @35%, earliest in campaign vs iter-22''s 15% / iter-16''s 28%) — that is 0.5 nats LOOSER than plain at 120k (~-12.4). After argmin@42k, val loss DRIFTS UPWARD (98k step at -10.63, gap to argmin 1.28 nats) while train loss keeps descending (99k train -12.87, train-val gap 2.24 nats) — classic overfitting signature. resnet50_gn at 120k is data-limited, not capacity-limited; 24M params over ~70k cosmologies. Adds the third ceiling falsifier alongside Q9c (variance) and Q4 (bound) nulls. The remaining mean lever (240k+stable schedule) is now ~95% likely null given (a) Q7 closure (schedule-shape low EV at iter-16 scale) and (b) Q1''s overfitting signature (more steps would amplify overfitting on plain too). Out-of-fiber remaining: more cosmologies in training distribution.'
---
