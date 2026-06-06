# PLAN — Phase B-2: L1 wavelet-ℓ₁ loader on the unified 10° TFDS

Status: **PLAN (awaiting Andreas sign-off).** Created 2026-06-05.
Campaign: `definitive-l1-vs-cnn-10deg-2026-06`. Prereqs: Phase A (data validated),
Phase B-1 (CNN `tfds_cross` loader + smoke, both passed).

## Goal

Make `npe_l1norm_cross_jaxili_nbody_tomo.py` compute the wavelet-ℓ₁ datavector by
reading the **unified 10-channel cross TFDS** directly (no grid `.npz` cache —
deleted), with the **channel-aware noise model** (`channel_empirical_global`),
**PCA OFF**, obs from the kept fiducial cache, then train the jaxili MAF NDE. This is
the science-critical loader: the known failure mode (`feedback_l1_cross_must_use_harmonic_route`)
is the channel-aware noise silently degrading to the broken `auto_scalar`, cratering FoM ~4×.

## Key structural difference from CNN B-1 (drives the design)

**L1 has no learned compressor.** The ℓ₁ datavector is a *deterministic* function of
(maps, σ_c, SNR ranges, wavelet params). So:
- There is **no compressor↔NDE split** to protect — the only split is NDE-train / NDE-val.
- The expensive step is computing ℓ₁ datavectors (PyTorch `WLStatistics` on GPU), not training
  a compressor. Compute once → cache `.npz` → train the MAF (3 seeds) from the cache.

## How the channel-aware noise model actually works (so we keep it correct)

1. `σ_c` = per-channel **std of the raw maps** (`calibrate_channel_noise...`): auto ~7e-3,
   cross ~2e-7 (the ~3×10⁴ gap). NOT the auto pixel-noise scalar.
2. Maps are **divided by σ_c** (`channel_scale`) → every channel ~unit variance.
3. `compute_l1_batch` runs the wavelet transform with a **single `noise_sigma`** on the
   rescaled maps and builds the ℓ₁ histogram over SNR, with **separate auto vs cross SNR
   ranges** (`l1_min/max_snr` vs `l1_min/max_snr_cross`) calibrated on the rescaled maps.

The `auto_scalar` bug skips step 1–2 (uses the auto pixel-σ for all channels) → cross SNR
collapses to ~0 → 95% of cross ℓ₁ bins zero. The new route must reproduce steps 1–3.

## What already exists / reuses (keeps B-2 small)

- **`compute_l1_batch` (`:988`)** — the ℓ₁ math; **source-agnostic, reused unchanged**.
- **`tfds_cross_tfdata_loader.iter_cross_tfds_batches` (built in B-1)** — yields
  `(maps[B,H,W,C], theta[B,6])` from the TFDS with channel-slice, **channel_scale divide**,
  flip, and θ→h0. It is exactly the map stream the calibrations + dataset builder need
  (pass `channel_scale=None` for the σ_c walk, `channel_scale=σ_c` for SNR + datavector).
- **`load_observed_from_harmonic_cache` (`:469`)** — obs from the **fiducial cache** unchanged
  (applies σ_c + slice like the grid).
- The cache functions (`calibrate_channel_noise_sigma_from_harmonic_cache`,
  `calibrate_snr_range_from_harmonic_cache`, `compute_l1_dataset_from_harmonic_cache`) stay
  **untouched** (20° reproducibility) — we add thin TFDS siblings beside them.

## Design — a `tfds_cross` source for the L1 cross pipeline

### New args (mirror B-1)
- `--cross-maps-route tfds_cross` (extend the current choices `["flat","harmonic"]` → add `tfds_cross`).
- `--cross-tfds-name` (default `nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180`),
  `--cross-tfds-data-dir` (`/home/tersenov/tensorflow_datasets`).
- `--fiducial-obs-cache <dir>` (obs source; the kept fiducial cache).
- `--nde-perm-split "5-6"` (which train perms feed the NDE-train ℓ₁ set; see decision below).
- Reuse: `--cross-noise-model channel_empirical_global`, `--harmonic-obs-perm/-patch-idx`,
  `--harmonic-cache-regime nobnt`. **Assert `--pca-components 0`** (hard-fail otherwise — the
  script default of 50 is wrong; `feedback_never_pca_l1`).

### New functions (thin siblings of the cache ones, sharing `iter_cross_tfds_batches`)
- `calibrate_channel_noise_sigma_from_cross_tfds(name, data_dir, n_sample, channel_slice)`
  → σ_c via std `sqrt(E[x²]−E[x]²)` over a raw-map TFDS sample (matches the cache formula).
- `calibrate_snr_range_from_cross_tfds(stats, name, data_dir, σ_c, ...)` → the same per-channel
  wavelet-SNR / reservoir-percentile loop as the cache version, over the TFDS stream.
- `compute_l1_dataset_from_cross_tfds(name, data_dir, split, perm_lo, perm_hi, stats, σ_c,
  SNR ranges, flip, ...)` → per **example** θ (TFDS batches are cosmology-shuffled), calling
  the shared `compute_l1_batch`. Returns `{theta, x}`; cached to `.npz`.

### `main()` branch
A `cross_maps_route == "tfds_cross"` block: σ_c (TFDS) → SNR ranges (TFDS) → ℓ₁ datasets
[NDE-train = `train`/perms 5-6, NDE-val = `test`/all perms] → obs from `--fiducial-obs-cache`
(same σ_c + SNR ranges) → MAF NDE. No `--full-sphere-cross-cache`. Prints
`cross_noise_model = channel_empirical_global` + the 10-value σ_c table (NOT a warning).

## Decisions to confirm

1. **★ L1 NDE training set (fairness) ★.** The headline comparison is "L1 datavector vs
   CNN-VMIM summary as input to the *same* jaxili MAF." Two options:
   - **(A, recommended) Match CNN exactly:** L1 NDE trains on **perms 5-6** (324k datavectors)
     — the *identical* (cosmo,perm,patch) set the CNN NDE uses. Controls the data-volume
     confound; isolates "which summary is more informative" with the NDE training held fixed.
     (CNN additionally trains its compressor on perms 0-4 — that is its inherent pipeline cost;
     L1's wavelet "compressor" needs none. Both NDEs see the same examples.)
   - **(B) L1 uses all perms:** L1 NDE trains on **perms 0-6** (1.13M) since it reserves no
     compressor set. Gives L1 its max data but reintroduces a data-volume asymmetry (this is the
     kind of confound the 10° run exists to remove). This is closer to what 20° did (and part of
     why 20° was messy).
   I recommend **A**; flagging because it changes the headline. (`--nde-perm-split 5-6` = A;
   `0-6` = B.)
2. **σ_c sample size:** 32 cosmologies-worth (~per the cache default `n_calibration_realizations=32`),
   drawn from the TFDS train sample. SNR-range calibration: 16 (cache default). Both bounded vs the
   fiducial-cache values.

## Verification (science-critical — guards the `auto_scalar` regression)

The grid cache is gone, so I verify against the **fiducial cache** (proven path) + parity on
identical maps:
1. **σ_c parity:** σ_c from the TFDS sample vs `calibrate_channel_noise_sigma_from_harmonic_cache`
   on the fiducial cache — must agree within sampling (~few %) AND be **10 distinct channel-aware
   values** spanning ~1e-2→1e-7 (NOT a single scalar). Directly detects an `auto_scalar` fallback.
2. **SNR-range parity:** auto/cross SNR ranges from the TFDS vs from the fiducial cache — agree
   within a few %.
3. **Datavector parity on identical maps:** take a set of fiducial patches; compute ℓ₁ two ways —
   (ref) the proven cache path, (new) feed the same patches through the new route's compute
   (divide by σ_c, `compute_l1_batch` with the same SNR ranges). Since `compute_l1_batch` is
   shared, equality ⇔ σ_c & SNR ranges match ⇒ assert datavectors equal to float precision.
4. **`pca_applied: False`** in the meta.json (assert).
5. **Smoke** (GPU): a reduced L1 auto+cross NDE run → FoM3/σ in the 20° ballpark
   (20° L1 auto+cross: σ(w0) 0.125, FoM3 ~53k; expect 10° wider). stdout shows
   `cross_noise_model = channel_empirical_global` + σ_c table, **no fallback warning**.

## Smoke command (for sign-off before the GPU launch; GPU 1 or 2)
```bash
cd /mnt/home/tersenov/software/cnn_sbi/scripts/sbi
FID=results/exploratory/cross_maps_campaign/full_sphere_cache_fiducial_10deg
OUT=results/exploratory/definitive_comparison_10deg/smoke_l1_autocross
XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 PYTHONUNBUFFERED=1 \
/home/tersenov/anaconda3/envs/jaxili/bin/python npe_l1norm_cross_jaxili_nbody_tomo.py \
  --cross-maps-route tfds_cross \
  --cross-tfds-name nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180 \
  --cross-tfds-data-dir /home/tersenov/tensorflow_datasets \
  --fiducial-obs-cache "$FID" --harmonic-cache-regime nobnt \
  --cross-noise-model channel_empirical_global --pca-components 0 \
  --nde-perm-split 5-6 --harmonic-obs-perm 0 --harmonic-obs-patch-idx 90 \
  --nbins 4 --tomo-bin-indices 1,2,3,4 --field-size 10 --field-npix 80 \
  --n-scales 5 --l1-nbins 40 --l1-min-snr -13 --l1-max-snr 13 \
  --summary-transform log1p-zscore --clip-value 5 \
  --seed 42 --cuda-visible-devices 1 \
  --save-dir "$OUT" --cache-dir "$OUT/cache" \
  --posterior-out "$OUT/posterior.npy" --figure-out "$OUT/corner.pdf"
  # Wavelet/ℓ₁ flags (n-scales 5, l1-nbins 40, ±13 SNR, log1p-zscore, clip 5) copied from the 20°
  # L1 cross invocation (l1_jaxili_run_commands.txt) so ONLY the data source changes. NOTE that
  # file's --pca-components 50/100 is STALE (pre-2026-05-25); we force --pca-components 0
  # (feedback_never_pca_l1). May also need --cross-snr-percentile to match the 20° cross SNR
  # calibration — I'll lift the exact value from the canonical 20° L1 cross command at impl time.
```
A reduced datavector set (e.g. a cosmo subset) keeps the smoke ~10–15 min; Phase C uses the full set.

## Non-goals (B-2)
- Phase C 4-arm × 3-seed run; the per-patch **fiducial summaries** for diagnostics (Phase D —
  reuses this obs path over patches/perms); auto-only L1 (a later `--channel-mode`-style slice if
  wanted). No touching the cache or flat-sky `--cross-maps` paths. No `git add`/commit without OK.

## After sign-off
Implement B-2 → CPU/parity unit checks (1–4 above) → L1 smoke (GPU, against 20° L1) → then
Phase C (4 arms × 3 seeds across GPU 1+2) and Phase D/E diagnostics.
