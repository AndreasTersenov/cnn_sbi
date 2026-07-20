# Plan — Full-200 fiducial: mean-datavector posterior + per-patch FoM distribution (L1 vs CNN)

Status: DRAFT awaiting Andreas sign-off (2026-06-02). Nothing run yet.

## Objective

Replace the flimsy 3-perm fiducial observation with the full **200-realization**
fiducial set (200 perms × 48 patches = **9600 patch-observables**), for each arm:

- **Step 1 (mean datavector):** average all per-patch *summaries* → one de-noised
  fiducial datavector → one posterior. The clean, realization-independent headline
  contour.
- **Step 2 (per-patch distribution):** compute the posterior for ~300 individual
  patches → FoM3 / σ / 2D **distribution** (mean ± scatter). The honest
  single-survey-unit (one 20 deg² patch) "which-sky-we-got" variation. Replaces
  the 3-perm spread.

**Interpretation (locked, do not over-read):** the step-1 width is the
*single-patch* posterior width centered at fiducial — NOT 200× tighter. The NDE
encodes the per-realization noise covariance Σ, so feeding E[s] returns the Fisher
width for that same Σ, cleanly centered. Step-2 scatter is the real run-to-run
variation. (If the actual N-realization constraint were wanted, the NDE would need
retraining at Σ/N — a different experiment, NOT this one.)

## Arms (4 — all harmonic route, nobnt, matched)

| arm | compressor | input | dim |
|---|---|---|---|
| L1 auto+cross | L1 wavelet | 10ch | datavector |
| L1 auto-only | L1 wavelet | 4ch | datavector |
| CNN auto+cross | CNN-VMIM | 10ch | 10 (cdim) |
| CNN auto-only | CNN-VMIM | 4ch (harmonic slice) | 10 (cdim) |

CNN native-TFDS / MAF / std are route/companion side-questions, NOT the headline —
left out (native-TFDS can't use this fiducial cross-cache route anyway). Can add later.

## Required code changes (small, additive, low-risk)

1. **`build_full_sphere_cross_cache.py` — fix `perm_dir` zero-pad.**
   `f"perm_000{perm}"` → `f"perm_{perm:04d}"`. Current form is wrong for perm ≥ 10
   (perm=10 → `perm_00010`, real dir is `perm_0010`). New form reproduces perm 0–6
   exactly and is correct for 0–199. (The cache only ever built ≤ perm 6, so the bug
   was latent.)
2. **`build_full_sphere_cross_cache.py` — add `--cosmo-id` filter.** Keep only
   entries whose `cosmo_id` matches (e.g. `cosmo_fiducial`). Without it,
   `--cosmo-subset fiducial` builds ~17 cosmologies → ~113 GB. With it → ~6.6 GB.
3. **NEW `build_fiducial_summaries.py`.** Load the trained compressor (CNN) / L1
   calibration ONCE; loop perms 0–199 × patches 0–47; compute the per-patch summary;
   write `S` of shape `(9600, dim)` per arm. **Reuse the EXACT obs/summary code paths**
   from `npe_cnn_nbody_tomo.py` / `npe_l1norm_cross_jaxili_nbody_tomo.py`
   (`load_observed_from_harmonic_cache` + compressor apply for CNN; the L1 datavector
   fn with channel_empirical_global SNR calib for L1) so summaries are computed
   identically to training. pca off (L1), zero-mean on, cdim=10.
4. **NEW `fiducial_meandv_and_dist.py`.** Per arm: train the jaxili NDE on the
   arm's compressed/datavector cache (seed 41 — reuse `train_jaxili_from_compressed`),
   then sample at `s_mean` (step 1) and at ~300 individual patch summaries (step 2).
   Save posteriors + FoM jsons + per-patch CSV.

## Build-parameter consistency (critical)

Perms 7–199 MUST be built with the SAME parameters as the existing perms 0–6
(sigma_e, galaxy_density, nside-source, lmax, n-centers, min-separation, center-nside,
noise-seed-base, field-size/npix, reso-arcmin, map-label). Procedure: read
`full_sphere_cache_fiducial/manifest.json` + the `cosmo_fiducial_perm6.npz` metadata,
replicate every flag exactly. The per-(cosmo,perm) shape-noise seed is
`noise_seed_base + 100*cosmo_idx + perm` (deterministic, differs per perm — exactly
the realization variation we want).

## Correctness gates (back-pressure — STOP if any fail)

- **G1 (preprocessing match):** the summary from `build_fiducial_summaries.py` at
  (perm 0, patch 0) reproduces the existing `cnn_obs.npz` / L1 obs datavector
  (≈ bit-for-bit). If not → preprocessing mismatch → STOP.
- **G2 (build match):** `cosmo_fiducial_perm7.npz` metadata matches `perm6` except
  `perm` and `noise_seed`. Patch centers identical across perms.
- **G3 (model match):** the NDE retrained here reproduces the Phase C perm-0 3-seed
  FoM at (perm0,patch0) within noise (confirms same model/space).
- **G4 (OOD check):** step-1 mean-datavector posterior width ≈ typical single-perm
  width (not wildly tighter). Wildly tighter ⟹ OOD/calibration red flag → report,
  don't trust.

## Compute & cost (GPU 1 or 0/1, pinned; check contention first)

Estimates from the build script's resource model + the campaign's L1 timing — will be
**measured on the first batch**, not trusted blind.

1. Build cache perms 7–199 (cosmo_fiducial, nobnt): ~5–15 min CPU `Pool(50)`, **6.6 GB**, no GPU.
2. Summarize 9600 patches/arm: L1 ~30–60 min GPU; CNN ~minutes.
3. NDE train + sample (step 1 + ~300 × step 2)/arm: ~minutes.
4. Plots + summary.

## Disk

6.6 GB maps + <100 MB summaries, on `/mnt` (152 GB free). **Build-and-keep** (reusable).
Streaming-and-discard (peak ~35 MB) available if conserving — say so.

## Outputs

```
results/exploratory/definitive_comparison/fiducial_full200/
  summaries/<arm>_S.npz                 # (9600, dim) per-patch summaries
  <arm>/mean_dv_posterior.npy + .fom.json   # step 1
  <arm>/per_patch_fom.csv                    # step 2 distribution
  overlays/meandv_l1_vs_cnn_autocross.png    # + auto-only
  overlays/fom3_distribution_l1_vs_cnn.png   # violin/hist (step 2)
  FIDUCIAL_FULL200_SUMMARY.md
```

## felt

File `definitive-l1-vs-cnn-2026-05/fiducial-full200-meandv` (experiment) on go-ahead;
close with verdict + numbers when it lands.

## Open questions for Andreas

1. **Arms:** the 4 above (L1/CNN × auto/auto+cross)? Or also CNN std/MAF? (I propose 4.)
2. **Step-2 sample size:** ~300 patches out of 9600 (smooth enough distribution,
   cheap)? Or denser / all 9600?
3. **Disk mode:** build-and-keep 6.6 GB (default, reusable) vs stream-and-discard (~0)?
