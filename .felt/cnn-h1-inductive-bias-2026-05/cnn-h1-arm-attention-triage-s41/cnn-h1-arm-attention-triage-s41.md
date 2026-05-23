---
name: H1 arm 1 / triage — plain_attn (L=1, H=4) auto-only seed 41
status: closed
tags:
    - experiment
    - cnn-h1-inductive-bias-2026-05
    - inductive-bias
    - triage
created-at: 2026-05-22T20:57:05.66909887Z
closed-at: 2026-05-23T08:46:30.41770114Z
outcome: 'ATTENTION ARM FALSIFIED (3-seed, 2026-05-22). Triage s41 promoted to 3-seed pool. Per-seed FoM3: s41=19389, s42=17251, s43=15527. MoS=17389, **pool=11892** (pool/MoS=0.684 — IDENTICAL to plain-CNN''s 0.685). bias|med=0.50σ (~plain-CNN''s 0.52σ). Pool gap closed vs 24k target: +6%. Plain-CNN anchor pool=11130, haircut=0.685. The decision rule''s <=13k threshold says H1 falsified. The smoking gun is the identical haircut — adding ~700k params of global receptive field did NOT reduce the seed-to-seed mode drift, which is the dominant failure mode in auto-only. Strong evidence H1 is not the load-bearing limit on this dataset; missing-from-auto info likely not learnable from this N (CosmoGridV1 ~70k cosmos) regardless of architecture. Verdict writeup: scripts/sbi/results/exploratory/h1_inductive_bias/H1_ATTENTION_VERDICT.md . Numerical: h1_attention_3seed_verdict.json . Caveat: only the tail-attention arm tested; (a) spectral-block-at-input and (c) MLP-Mixer-trunk arms still standing. Awaiting Andreas''s call on next move.'
---
