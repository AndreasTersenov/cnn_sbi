---
name: 'Full-200 fiducial: L1 is wildly patch-sensitive; campaign patch_idx 0 was atypically LOW for L1 (TENTATIVE)'
status: closed
tags:
    - finding
    - definitive
    - suspicious
created-at: 2026-06-02T05:45:13.247994288Z
closed-at: 2026-06-02T14:39:44.272444502Z
outcome: 'RESOLVED 2026-06-02 (measured, after viewing contours). The high per-patch L1 FoM3 is REAL & calibrated (#2 stratified varied-theta TARP: HIGH-FoM3 L1 tercile max|ECP-a| dim3 0.068 on diagonal; CNN HIGH 0.095). NOT over-tight, NOT OOD (#1; patch-0=polar lat 88.5). BUT the contours temper the ''flips to L1>>CNN 2x'' reading: on robust metrics L1 & CNN auto+cross are COMPARABLE (~15-25% on marginals; sigma(w0) 0.125 vs 0.167, sigma(Om) 0.023 vs 0.027, 2D(Om,s8) 1.6x). The FoM3 ~2x is the cube of a ~20-25% per-dim difference. Which method is nominally ahead FLIPS with obs-patch (CNN at polar patch-0, L1 at typical patches) and metric. L1 = modestly tighter (Om-s8, w0) but MORE patch-variable (centers wander ~ width, hence calibrated); CNN = slightly looser, MORE stable. Campaign''s patch-0=polar modestly disfavored L1 (~20%/dim), real but not verdict-changing. THROUGHLINE: feedback_fom3_fragile vindicated 3x - FoM3 generated CNN>=L1 -> L1>>CNN -> (robust) comparable. My intermediate fixed-theta ''no reversal'' AND my over-excited ''flips to L1>>CNN'' were both FoM3/confound-driven; the robust/contour read = comparable + small L1 edge. Figures: tarp_per_patch/figures/reversal_{A,B,C}*.png + tarp_stratified/figures/tarp_per_arm_dim3.png. Methodology: obs should be a typical/averaged patch (not patch-0); never headline FoM3 here.'
---
