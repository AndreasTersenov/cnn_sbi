# 10°×10° campaign — dataset production + analysis plan

Status: PLAN (awaiting Andreas sign-off). Created 2026-06-04.
Goal: redo the definitive L1-vs-CNN comparison (4 arms: L1/CNN × auto/auto+cross, + all
per-patch diagnostics / SBC / L-C2ST) on **10°-on-a-side patches** for the paper — better
flat-sky validity (gnomonic corner distortion 6.3%→1.5%) than the 20° fields.

## Locked decisions (Andreas, 2026-06-04)
1. **Obs unit = one 10° patch** (100 deg²; smaller survey, better flat-sky). Per-obs constraints
   ~2× weaker than 20° — intended trade.
2. **~180 non-overlap patches/realization, |lat| < 75°** (storage-neutral vs 20°'s 48; 4× richer
   training + ~36k fiducial test-obs; no polar patch).
3. **Route-matched SHT** for the unified 10-channel (4 auto + 6 cross) dataset (optional separate
   native-auto benchmark later).
4. **TFDS-direct, drop the permanent .npz cache** (Option A: transient cache → TFDS → delete).
   **L1 reads the TFDS too** (new loader).
5. Same nside=512, lmax=1024, **7.5 arcmin/px** (80px × 10° = 160px × 20° resolution).

## Phase 0 — free disk (RUNNING)
Archive finished 20° data to `/nas/tersenov/archive_20deg/` via rsync (non-destructive):
`full_sphere_cache_grid` (623 GB) + `nbody_cosmogrid_dataset_tomo{,_cross}` TFDS (696 GB).
Keep local: `compressed/` (14 GB), `phaseA_…/` compressor checkpoints (2.1 GB), results, `.felt`.
**Verify** (file count + size) then **delete local only after Andreas OK** → ~2 TB free.
(20° is fully analyzed; pull back from /nas only if a referee needs a 20° rerun.)

## Phase 1 — polar-safe center function
Modify `_build_non_overlapping_centers` (or add `_build_polar_safe_centers`): filter candidate
pixels to **|lat| < 75°** BEFORE the greedy selection (so the `np.arange` fallback can't grab a
pole), `min_separation_deg ≈ 14.2`, `center_nside=64`, target **180** centers. Deterministic.
Self-check: assert max|lat| < 75, count == 180, all pairwise separations ≥ 14.2°.

## Phase 2 — build TRANSIENT 10° cache (parallel SHT)
`build_full_sphere_cross_cache.py --field-size 10 --field-npix 80 --n-centers 180
--center-nside 64 --min-separation-deg 14.2` (+ the polar-safe centers). Parallel SHT (50 workers),
nobnt, 10 channels, shape noise before SHT — identical pipeline to 20°, only geometry changes.
**SMOKE FIRST:** build ~5 cosmologies, verify (a) patch shape (180,80,80,10), (b) a patch matches
a direct reference SHT+gnomonic, (c) channel scales sane. Only then the full grid (~2 h).

## Phase 3 — reserialize → TFDS (local XFS, clean shards)
New config in `tf_dataset_nbody_tomo_cross.py`: `grid_10deg_80px_nonoverlap180` (xsize=80, size=10),
TFRecord, `--data_dir /home/tersenov/tensorflow_datasets` (LOCAL, not /nas). **Ordered output
(`imap`, NOT `imap_unordered`)** so shards are cosmology-contiguous → clean compressor/NDE split.
Verify: (a) bit-exact vs a cache sample, (b) example count, (c) **split disjointness audit** (no
cosmo_idx shared across the compressor/NDE groups).

## Phase 4 — delete transient cache
After TFDS verified, delete the 10° `.npz` cache → only the ~700 GB TFDS remains.

## Phase 5 — loaders
- **CNN:** existing fast `tfds.load + tf.data` path, retune `read_config`
  (interleave_cycle_length/block_length) for the 10° shard count.
- **L1 (new):** read 10-channel patches from the TFDS, compute the wavelet ℓ₁ datavector
  (channel-aware noise model `channel_empirical_global`, pca off). Verify the datavector matches a
  reference computed the 20°-cache way on one patch.

## Phase 6 — clean compressor↔NDE split
Split TRAIN cosmologies into **disjoint groups** (e.g. compressor cosmo 1–630, NDE 631–899) by
`cosmo_idx`. Fiducial = obs (held out). Structurally eliminates the 20° example-slice leakage.

## Phase 7 — run + analyze (the campaign)
4 arms (L1/CNN × auto/auto+cross), jaxili MAF NDE, 3 seeds. Then the full diagnostic suite we built
at 20°: per-patch geometry map, spread decomposition, bias/error-budget, SBC, CNN L-C2ST. Compare
to the (archived) 20° numbers — especially **does L1's −0.37σ fiducial w0 offset shrink at 10°?**
(the flat-sky test). felt constitution `definitive-l1-vs-cnn-10deg-2026-06` to track it.

## Back-pressure / guardrails
- Smoke before scale (Phase 2). Verify bit-exactness + split disjointness before trusting (Phase 3).
- Keep disk < ~80% full throughout (others share it); monitor `df` between phases.
- GPU 0+1 only. Don't delete archived-source local copies until /nas verified + Andreas OK.
- Never commit data; stage by path.
