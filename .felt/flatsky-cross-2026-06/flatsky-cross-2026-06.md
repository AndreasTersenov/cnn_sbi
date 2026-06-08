---
name: 'Flat-sky cross-maps: de-leaked auto+cross (build, train, contours)'
status: active
tags:
    - experiment
    - sbi
    - cnn
    - l1
    - cross-maps
    - flat-sky
    - paper
created-at: 2026-06-08T16:45:21.09193286Z
outcome: 'OPEN (filed 2026-06-08). Replace the LEAKY full-sphere harmonic cross-maps (every cross-patch is a global functional of the whole sky; auto+cross constraining power partly UNPHYSICAL — see CROSS_MAP_LEAKAGE_FINDING.md) with PATCH-LOCAL flat-sky cross-maps, recompute stats, train L1+CNN, get CALIBRATED cosmological contours. Design+validation DONE (FLATSKY_CROSS_REDESIGN_NOTES.md S1-14): cross map = apodized-circular CONVOLUTION (Zurcher Eq.12 flat-sky analog) AND pointwise PRODUCT (its mean = xi_ij); complementary -> TEST BOTH. NO sim/dataset rebuild (cross computed on-the-fly from auto ch0-3 of TFDS grid_10deg_80px_nonoverlap180; auto-only baseline uses identical autos => clean comparison). PRIMARY METRIC: median over typical patches of sigma(w0) + 2D(Om,s8); FoM3 reported NEVER headlined. NEXT: implement augmentation (L1 --cross-op {conv,product,both}; ADD flat-sky to CNN; per-channel/per-scale noise = THE fix for the old shared-auto-sigma bug) -> GATE A construction (bitmatch, xi_ij recovery) -> GATE B cosmology-dependence (NEW, decisive) -> train matrix (auto-only / +conv / +product / +both x L1,CNN, 3 seeds) -> GATE C calibration (TARP/SBC/L-C2ST) -> contours vs auto-only AND vs full-sphere. Expect MODEST gains (cross info is large-scale, patch samples it poorly = physically correct). See FLATSKY_CROSS_BUILD_PLAN.md + HANDOFF_FLATSKY_CROSS_2026-06-08.md. Continues [[definitive-l1-vs-cnn-10deg-2026-06]].'
---

## Primary metric
median over typical patches of sigma(w0) + 2D(Om,s8) area. FoM3 reported, NEVER headlined (feedback_fom3_fragile_use_2d_areas).

## Done condition
Each arm cross-gain over auto-only is measured AND calibration-validated (TARP/SBC/L-C2ST); conv-vs-product complementarity decided; honest flat-sky gain compared to the inflated full-sphere number. Stop when contours + comparison are produced and written up.

## Guardrails
patch-local cross ONLY (never full-sphere = leakage); per-channel noise (not shared auto-sigma); never PCA L1; GPU 1 only; example-disjoint compressor/NDE split by perm; calibrate BEFORE contours; SAME auto channels across all arms; one apodized-circular convolution definition; do not relitigate the operator choice (notes S8-12).

## Loop status (live)
WAITING to start in a fresh Claude Code session. Unblock = read FLATSKY_CROSS_BUILD_PLAN.md + REDESIGN_NOTES + LEAKAGE_FINDING + HANDOFF, then implement the augmentation and run GATE A.

## Pointers
FLATSKY_CROSS_BUILD_PLAN.md (steps/gates), FLATSKY_CROSS_REDESIGN_NOTES.md (design+validation S1-14), CROSS_MAP_LEAKAGE_FINDING.md (why), HANDOFF_FLATSKY_CROSS_2026-06-08.md (handoff). Memory: project_cross_map_leakage_fullsphere, feedback_l1_cross_must_use_harmonic_route, feedback_never_pca_l1, feedback_fom3_fragile_use_2d_areas, feedback_gpu1_only, reference_jaxili_checkpoint_reload_truncation.
