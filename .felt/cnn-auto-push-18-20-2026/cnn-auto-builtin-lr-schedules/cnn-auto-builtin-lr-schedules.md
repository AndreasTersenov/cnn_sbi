---
name: Inner script already has LR schedules — our --compressor-lr is the initial value
status: closed
tags:
    - cnn-auto-push
created-at: 2026-05-18T19:47:43.50121145Z
closed-at: 2026-05-18T19:47:43.523056835Z
outcome: npe_cnn_nbody_tomo.py uses piecewise-constant for compressor (0.7x at every 10% over first 2/3 of training) and optax.cosine_decay_schedule for NDE flow (--lr-init 1e-3 -> --lr-end 1e-5). Schedule variants (warmup, cosine for compressor, slower/faster decay) untried.
---
