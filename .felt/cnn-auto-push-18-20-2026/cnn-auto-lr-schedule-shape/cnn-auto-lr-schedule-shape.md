---
name: Compressor LR schedule decays to 4% by step 2/3·total_steps, then plateau
status: closed
tags:
    - cnn-auto-push
created-at: 2026-05-18T21:18:49.969917358Z
outcome: 'VMIM compressor uses optax.piecewise_constant_schedule with schedule_steps=total_steps*2/3 and boundaries at [0.1,0.2,...,0.9]*schedule_steps, each scaled by 0.7. By step (0.9*0.6667*total_steps)=0.6*total_steps the LR is lr_init*0.7^9=0.04*lr_init. The last 1/3 of training runs at ~4% of init LR. Two implications: (1) increasing total_steps doesn''t just lengthen training, it also extends the high-LR phase proportionally — at step 30k, a 60k-total run is at 8% lr_init but a 120k-total run is at 34% lr_init. (2) The final 1/3 ''polish phase'' is at very low LR — may explain why iter-15 (resnet50_gn, 60k) didn''t converge: its useful-update window (above 30% lr_init) ended at step 18000. Q7 (LR schedule variants) gains importance: cosine decay or longer warmup could plausibly help higher-capacity arms more.'
---
