---
name: Patch-center confound (G8) is real and large
status: closed
tags:
    - finding
    - caveat
    - definitive
created-at: 2026-06-01T02:47:18.68938107Z
outcome: 'CONFOUND CONFIRMED (2026-05-31): native-TFDS auto-only FoM3 14969 (sigma_w0 0.148) >> harmonic-cache-sliced auto-only 9125 (0.216). CNN cross-gain is route-sensitive: ~1.8x over a fair auto-only, NOT 2.93x over the lossy harmonic auto-only. Quote cross-gain with this caveat. Artifacts: PHASE_C_2026_05_31/SUMMARY_DEFINITIVE.md; memory project_patch_center_confound_g8. Next: write caveat into the summary; TARP the native-auto arm.'
---

The harmonic-cache route's auto-only baseline is lossy (esp. w0), so the within-route cross-gain (2.93x) overstates cross-map info content. Part of [[definitive-l1-vs-cnn-2026-05]]; this is the G8 confound the comprehensive-experiment-audit flagged.
