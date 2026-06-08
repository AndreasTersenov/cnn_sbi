---
name: 'Phase A: 10deg TFDS validated (structure/scales/disjoint/bit-exact)'
status: closed
tags:
    - finding
    - 10deg
    - dataset
created-at: 2026-06-05T03:27:07.668023098Z
closed-at: 2026-06-05T03:27:21.288450557Z
outcome: 'PASS (2026-06-05). 10deg TFDS grid_10deg_80px_nonoverlap180 validated CPU-only, 3 checks: A1 structure/scales (both splits (80,80,10) f32 finite; auto ch0-3 RMS 7.3e-3..1.0e-2, cross ch4-9 RMS 2.3e-7..6.7e-7, auto>>cross ordering, ~3e4x gap => channel-aware noise is load-bearing). A2 full-scan disjointness (train 1,132,740 ex/899 cosmos[1,899]; test 504,000/400 cosmos[900,1299]; perm[0,6] patch[0,179]; train INTER test = EMPTY). A3 independent bit-exact: fresh SHT re-derive of ci=703(=grid row703=cosmo_001814)/perm3/patch118 == TFDS, max_abs_diff 0.0 all 10ch. GOTCHA for Phase B: builder uses imap_unordered (tf_dataset_nbody_tomo_cross.py:158) + global shuffle => shards NOT cosmo-contiguous; the clean compressor/NDE split MUST filter by cosmo_idx, NOT example-slice (that was the 20deg leak). Artifacts: scripts/sbi/results/exploratory/cross_maps_campaign/validate_10deg/report_{scales,disjoint,bitmatch}.json + scripts/sbi/validate_10deg_dataset.py, validate_10deg_bitmatch.py. NEXT: A4 smoke (deferred to Andreas) then Phase B loaders.'
---
