---
name: Sanity check — TFDS auto maps vs harmonic-cache auto channels are not byte-identical (cache has SHT/iSHT roundtrip at lmax=1024)
tags:
    - experiment
    - canonical-anchors-refresh-2026-05
    - methodology-fix
    - ship-blocker
created-at: 2026-05-24T20:35:52.177448438Z
outcome: 'Concern: build_full_sphere_cross_cache.py:283 builds the auto channels of the cache via hp.alm2map(alms, lmax=1024) → SHT/iSHT roundtrip → bandlimited to ell=1024. TFDS auto maps come direct from nside=512 HEALPix patches (ell up to ~1535). So CNN auto-only (TFDS) and the 4 auto channels of CNN auto+cross (cache) are NOT the same maps; the cache bandlimits high-ell info. Patch Nyquist is ell~720 so the bandlimit happens at sub-Nyquist scales and may be cosmetically irrelevant, but we want to confirm. Test: run CNN on the cache''s auto-only slice (--channel-mode=auto_only, 4 channels from cache) with same compressor config as TFDS auto-only canonical. Compare 3-seed pooled FoM3. Decision: if within seed scatter (~10%), bandlimit is negligible and cross/auto ratio is clean. If >20% difference, real protocol asymmetry to flag in writeup. Resource: 1 arm × 3 seeds parallel-3, ~1.5h on GPU 1. Can run in parallel with main canonical-refresh if GPU memory allows.'
---
