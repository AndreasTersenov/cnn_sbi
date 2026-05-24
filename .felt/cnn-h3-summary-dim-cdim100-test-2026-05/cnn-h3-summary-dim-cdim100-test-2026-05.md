---
name: H3 test — does CNN auto+cross cdim=100 lift past the 24k pooled anchor?
status: closed
tags:
    - experiment
    - sbi
    - cnn
    - weak-lensing
    - summary-dim
created-at: 2026-05-23T14:32:30.102353547Z
closed-at: 2026-05-24T05:31:28.296907117Z
outcome: 'H3 FALSIFIED 2026-05-24, opposite direction to hypothesis. 3-seed pool = 12,151 vs cdim=10 anchor 23,986 — cdim=100 CRATERED FoM3 by ~49%. Per-seed: s41=13.8k, s42=15.2k, s43=11.4k vs anchor s41=25.9k, s42=25.7k, s43=22.0k. Posteriors well-calibrated (|bias|med 0.20σ vs anchor 0.17σ; pool/MoS haircut 0.903 vs 0.978) but ~2× wider in volume. Two signals from logs: (1) compressor VMIM val loss tighter at cdim=100 (~-11.8 vs anchor ~-12) — instance of L67 [[feedback_val_loss_not_reliable_fom3_proxy]]; (2) NDE early-stopped at 13.5k-17k of 50k budget across all 3 seeds with late val-loss spikes — RealNVP (8 layers, hidden=256) underprovisioned for 100-d conditioning at the standard cdim=10-tuned config. Andreas''s prior that summary-dim is not the bottleneck is vindicated. Strict claim: cdim=10 essentially near-optimal at this NDE config; pushing it upward degrades the pipeline. Not ruled out (separately): a co-tuned higher-capacity NDE at cdim=100 might match cdim=10, but the spirit of the question is answered. H3 falsified; H1/H2/global-info picture unchanged. Writeup: scripts/sbi/results/exploratory/h3_cdim_sweep/H3_CDIM100_VERDICT.md . Numerical: h3_cdim100_3seed_verdict.json . Updates CNN_CROSS_MAPS_INFORMATION_NOTE.md §8c.4. Practical lesson: 3-way parallel on a single A100 with 10-ch+cdim=100 instances was slower wall-clock (~7.5h) than sequential would have been (~5h) — update the H1 exit-interview''s parallel-seeds lesson with ''only when each instance doesn''t GPU-saturate''.'
---
