---
name: 'Refine Phase C: per-perm-average multi-perm + TARP new arms + write G8 caveat'
status: closed
tags:
    - deferred-question
    - definitive
created-at: 2026-06-01T02:47:18.706965684Z
closed-at: 2026-06-01T15:55:53.295811898Z
outcome: 'DONE 2026-06-01. (a) aggregate_all_arms.py rewritten to per-perm-AVERAGE (group by perm, pool 3 seeds within perm, metric per perm, average across perms + report across-perm spread + n_perms col) instead of perm-POOL -> this surfaced a HEADLINE SHIFT (see finding-perm-averaging-overturns-l1-lead): perm-0 ''L1>=CNN on auto+cross'' does NOT survive; matched 3-perm CNN 28093 (+/-12%) >= L1 25808 (+/-27%) on FoM3/2D, L1 keeps only a modest perm-fragile sigma(w0) edge. (b) TARP: 2 genuinely-new arms dumped (cnn_autocross_rnvp_std, cnn_auto_native_rnvp; 3 seeds each, n_dumps 18->24) + re-plotted; both calibrated (max|ECP-a| dim3 0.077/0.051); multi-perm TARP redundant (same NDE as core RealNVP arms) -> NOT re-dumped. (c) G8 patch-center confound written into SUMMARY_DEFINITIVE.md, auto-computed 1.79x (fair native-TFDS) vs 2.93x (lossy harmonic). (d) METRIC DRIFT resolved (Andreas): FoM3 stays declared primary, reported with per-perm spread; sigma/2D secondary -> no constitution metric-stanza change. Artifacts: PHASE_C_2026_05_31/SUMMARY_DEFINITIVE.md + phase_c.csv + tarp_2026_05_31/. Next: commit session code w/ Andreas OK (pending); within-route L1-vs-CNN auto+cross run would settle the residual route confound (held for Andreas).'
---

The first apples-to-apples task before any further comparison claims. Part of [[definitive-l1-vs-cnn-2026-05]].
