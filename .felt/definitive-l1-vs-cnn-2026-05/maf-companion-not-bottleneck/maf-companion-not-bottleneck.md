---
name: MAF VMIM companion is not the CNN bottleneck
status: closed
tags:
    - experiment
    - definitive
created-at: 2026-06-01T02:47:18.695830793Z
outcome: 'CLOSED (2026-05-31): beefier conditional MAF companion (8 transforms [256,256]) is WORSE than sbi_lens RealNVP across all 5 seed pairings (auto+cross FoM3 ~0.45x; sigma uniformly wider). Companion flow quality does NOT limit CNN. Lower VMIM loss yet worse FoM3 (val-loss != FoM3 proxy). Artifacts: companion_comparison_2026_05_31/; memory project_maf_companion_not_bottleneck; code committed 0d58d5e (default backend stays sbi_lens). Next: no follow-up - RealNVP companion stands.'
---

Tested whether the sbi_lens ConditionalRealNVP VMIM companion limits the CNN compressor by swapping in a hand-rolled conditional MAF. It does not - the MAF is worse. Part of [[definitive-l1-vs-cnn-2026-05]].
