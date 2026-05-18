---
name: resnet50_gn auto-only collapses at 60k — compressor undertrained
status: closed
tags:
    - cnn-auto-push
created-at: 2026-05-18T21:14:11.836246671Z
outcome: iter-15 (resnet50_gn cdim=20 lr=1e-3) hit FoM3=8769 vs iter-5 plain=18568 (-52.8%). Compressor val loss ended at -11.15 vs plain -12.44 (1.3 nats looser VMIM bound). resnet50_gn has ~25x more params than plain CNN — at 60k steps it sees ~1/25 the updates per parameter. Falsifies Q1 *at 60k* but confounds with Q2 (compressor-steps). Resnet50-BN at 120k in cnn_resnet50_zm_sweep hit cdim=20 -> 27668; resnet50_gn likely needs >=120k to show its real auto-only performance.
---
