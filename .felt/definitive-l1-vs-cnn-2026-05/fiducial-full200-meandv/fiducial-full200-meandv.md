---
name: 'Full-200 fiducial: mean-datavector posterior + per-patch FoM distribution (L1 vs CNN)'
status: closed
tags:
    - experiment
    - definitive
created-at: 2026-06-02T04:10:26.831418959Z
closed-at: 2026-06-02T05:45:13.257180655Z
outcome: 'DONE 2026-06-02 (5/6 arms; MAF re-running after a benign tol fix). Pipeline VALIDATED end-to-end: all arms passed G3 (reproduce campaign perm0 FoM3 within 20%; CNN a+c exact 0.0%, L1 a+c 4.9%). RESULTS: CNN auto+cross clean & consistent (step1 mean-dv 24387 ~ perm0 26755 ~ step2 24570+/-3609; calibrated, trustworthy). TWO SURPRISES on L1 (both flagged, see [[finding-l1-patch-sensitivity-full200]] + FIDUCIAL_FULL200_FINDINGS.md): (1) L1 mean-datavector OVER-TIGHTENS (step1 49323 >> single-survey ~26-33k) = OOD artifact -> use step2 for L1; (2) L1 wildly patch-sensitive, per-patch median 53069 (17k-122k) and campaign''s patch_idx 0 atypically LOW (~20k vs ~53k pop) -> campaign may understate L1 ~2x (CNN unaffected). TENTATIVE pending TARP on per-patch L1 + patch0-vs-pop robustness. Summaries (reusable) in fiducial_full200/summaries/*.npz.'
---
