---
name: Q8 / Q1-reopen — resnet50 stock-BN cdim=20 @120k seed 41 (3rd seed of May 8 sweep)
status: closed
tags:
    - experiment
    - cnn-auto-push-18-20-2026
    - bn-contamination
created-at: 2026-05-22T12:10:32.467488805Z
closed-at: 2026-05-22T19:52:33.299131799Z
outcome: 'Q1 CLOSED — BN-contamination interpretation upheld (2026-05-22). 3-seed sweep (s41=24462, s42=31684, s43=23652) gives MoS=26.6k but pooled FoM3=18,368 (pooled/MoS=0.69 haircut). Per-seed posteriors are tight but disagree on mode location — pooling collapses joint covariance. Critical: all 3 seeds bias w0 in the SAME direction (means -1.13/-1.12/-1.07 vs truth -1.00), a shared -1σ systematic that is the textbook BN-running-stats-leakage signature. Decision rule (pooled>15k AND |bias|med<0.5σ): pooled PASS (18.4k>15k) but global median |bias|=0.51σ FAIL (just above threshold; qualitative 3/3 w0-biased-same-direction unambiguous). Stock-BN auto-only failure mode is milder cousin of the 10-ch harmonic catastrophe (FoM3~700) — same mechanism, tight-but-biased with seed-dependent mode drift. cnn-auto-push ceiling at GN variant stands; 4th independent ceiling confirmation. Writeup: scripts/sbi/results/exploratory/cnn_resnet50_zm_sweep/Q1_REOPEN_VERDICT.md ; numerical summary: q1_reopen_3seed_analysis.json . project_resnet_bn_contamination memory extended with failure-mode-B (auto-only tight-but-biased).'
---
