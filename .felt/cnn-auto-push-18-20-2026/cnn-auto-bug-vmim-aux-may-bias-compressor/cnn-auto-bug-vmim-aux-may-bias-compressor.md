---
name: VMIM aux network (--vmim-nf-hidden 128) may bias compressor toward summaries it can model, not max-info (cnn-auto-push, A3)
tags:
    - bug-shape
    - audit-A3
    - cnn-auto-push-18-20-2026
created-at: 2026-05-18T22:15:37.210482858Z
outcome: 'VMIM trains compressor by maximizing E[log q(θ|s)] where q is a small RealNVP companion network (hidden 128). If q is underexpressive, the compressor learns summaries q can model — not summaries that maximize true I(s;θ). Inference uses a DIFFERENT, larger NDE (hidden 256). Bound looseness directly suppresses compressor quality. Diagnostic: --vmim-nf-hidden 256 and 512 on iter-5 config; if FoM3 changes >5%, the bound is the bottleneck. File: scripts/sbi/npe_cnn_nbody_tomo.py:train_compressor_vmim (~1736-2270). Already on Tier-2 EV queue as Q4; A3 elevates to bug-shape.'
---
