---
name: Understand the per-patch structure — WHY do L1/CNN look the way they do? (large-sample diagnostics)
status: closed
tags:
    - investigation
    - definitive
    - next-phase
created-at: 2026-06-03T00:32:24.859443575Z
closed-at: 2026-06-05T02:44:08.85380591Z
outcome: 'CLOSED 2026-06-04: all 6 diagnostic threads done at 20deg. (1+2) L1 patch-FoM3 spread ~92% REALIZATION noise not geometry; the polar-patch outlier was a single tile from a tiling bug (now fixed for 10deg via max_abs_lat). (3) L1 has an ANTI-shrinkage fiducial offset (pull w0 -0.37sig / Om -0.27sig / s8 +0.19sig), CNN unbiased (<0.06). (4) the offset is CROSS-MAP-SPECIFIC -- it FLIPS sign in auto-only -> L1''s high-gain cross-channel w0 extraction (tighter-but-overshooting) vs CNN regularized. (5) SBC: L1 GLOBALLY calibrated (offset cancels over prior, reconciles the varied-theta TARP); CNN L-C2ST (validated logreg): locally mildly over-confident, small effect. HEADLINE: L1''s w0 edge = precision-with-a-LOCAL-bias, not pure info. Artifacts: fiducial_full200/{geometry_map,calibration}/*FINDINGS.md + [[finding-l1-spread-realization-not-geometry]]. CONTINUES in [[definitive-l1-vs-cnn-10deg-2026-06]].'
---
