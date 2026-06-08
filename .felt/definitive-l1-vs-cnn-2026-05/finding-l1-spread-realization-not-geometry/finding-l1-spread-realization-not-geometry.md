---
name: L1 patch-spread is realization-driven, not geometry; L1 has a fiducial w0/Om bias CNN lacks
tags:
    - finding
    - definitive
    - sbi
    - l1
    - cnn
created-at: 2026-06-03T04:32:35.487377642Z
outcome: 'VERDICT (2026-06-03, Threads 1-3, 48 idx x 200 perm = 9600 obs/arm, a+c): handoff hypothesis ''L1 patch-variability tracks latitude/gnomonic'' LARGELY FALSIFIED. Variance decomp: L1''s per-patch log-FoM3 spread is ~92% REALIZATION (perm) noise, ~8% geometry (eta2 16.6x null) -- and that 8% is ENTIRELY the single polar tile patch-0 (drop it: corr(logFoM3,lat) -0.257->+0.174, eta2 16.6x->1.7x ~= chance). sigma marginals >98% realization. CNN flat at chance everywhere. patch-0 low FoM3 (-45%) but its sigma(w0/Om) NORMAL -> covariance-shape/FoM3-fragility, not precision loss. NEW finding: L1 carries a COHERENT fiducial offset (pull w0 -0.37sigma, Om -0.27sigma, s8 +0.19sigma) on all 48 tiles; CNN unbiased (|pull|<0.06). Bias is geometry-modulated (corr(bias_w0,|lat|)=-0.60 excl p0). => L1''s tighter sigma(w0) is partly bought with a w0 centering bias (precision-accuracy trade). Pointer: fiducial_full200/geometry_map/GEOMETRY_FINDINGS.md + geometry_report.json + figures/. Next: Thread 4/5 -- is the -0.37sigma w0 offset a true accuracy cost (fiducial-local/geometry-stratified coverage) or prior-averaged cancellation (reconcile w/ global TARP calling L1 calibrated)?'
---
