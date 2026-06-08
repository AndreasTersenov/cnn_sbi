---
name: Definitive L1 vs CNN on 10deg patches (paper run; flat-sky test)
tags:
    - experiment
    - sbi
    - cnn
    - l1
    - definitive
    - 10deg
created-at: 2026-06-05T02:44:08.867404816Z
outcome: 'OPEN (filed 2026-06-04). Redo the definitive L1-vs-CNN comparison (4 arms L1/CNN x auto/auto+cross + the full 20deg diagnostic suite) on 10deg-on-a-side patches for the paper -- better flat-sky validity (gnomonic corner distortion 6.3%->1.5% vs 20deg). DATASET BUILT+VERIFIED overnight: TFDS nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180 @ /home/tersenov/tensorflow_datasets (1,636,740 ex; 180 polar-safe patches |lat|<75; 80px/7.5arcmin; SHT 10-ch 4auto+6cross) + fiducial obs cache full_sphere_cache_fiducial_10deg (200 perm). PRIMARY METRIC: sigma(w0) + 2D(Om,s8) median over typical patches (FoM3 amplifies/fragile -> report but DON''T headline). HEADLINE TEST: does L1''s -0.37sig fiducial w0 offset SHRINK at 10deg (=> flat-sky was the cause, 10deg ''more proper'') or PERSIST (=> intrinsic L1 ell1 statistic bias)? NEXT: (a) validate dataset; (b) L1-reads-TFDS loader [SCIENCE-CRITICAL: channel_empirical_global noise, PCA OFF/never] + CNN read_config retune + clean cosmo_idx compressor/NDE split; (c) 4 arms (jaxili MAF, 3 seeds); (d) diagnostics geometry/spread/bias/error-budget/SBC/L-C2ST; (e) compare to 20deg. See HANDOFF_10DEG_CAMPAIGN.md + PLAN_10DEG_CAMPAIGN.md. Continues [[understand-per-patch-structure-2026-06]] (20deg, closed).'
---
