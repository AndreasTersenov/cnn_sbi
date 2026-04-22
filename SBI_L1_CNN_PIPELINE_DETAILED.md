# Detailed pipeline walkthrough: L1-norm SBI vs CNN-compressor SBI

This document describes, step by step, the full execution sequence of four inference pipelines:

- `scripts/sbi/npe_l1norm_nbody_tomo.py` (wavelet L1 summary + NPE)
- `scripts/sbi/npe_l1norm_jaxili_nbody_tomo.py` (wavelet L1 summary + jaxili NPE)
- `scripts/sbi/npe_cnn_nbody_tomo.py` (CNN summary + NPE)
- `scripts/sbi/npe_cnn_jaxili_nbody_tomo.py` (CNN summary + jaxili NPE)

The goal is to provide an audit trail so you can inspect exactly where information could be lost or distorted.

---

## 0) Shared context and assumptions

All pipelines:

1. infer the same cosmological parameter vector  
   `theta = [Omega_m, sigma_8, w0, h0, n_s, Omega_b]`.
2. use the same observed fiducial map source and add shape noise.
3. use TFDS data built from `tf_dataset_nbody_tomo.py`:
   - `NbodyCosmogridDatasetTomo/grid` (10 deg, 80 px)
   - `NbodyCosmogridDatasetTomo/grid_20deg_160px` (20 deg, 160 px)
4. apply the same augmentation pattern to TFDS maps:
   - select tomographic channels (`--tomo-bin-indices`)
   - add Gaussian shape noise
   - random flips
   - rescale `H0 -> h0` by dividing theta index 3 by 100.
5. train/load a conditional density estimator (`p(theta | summary)`);
6. save posterior samples and optional contour plot.

---

## 1) L1-norm pipeline (full sequence)

Reference script: `scripts/sbi/npe_l1norm_nbody_tomo.py`

### Step L1-1: Parse CLI and normalize scenario

Main controls:

- Geometry/data: `--field-size`, `--field-npix`, `--tfds-name`
- Tomography: `--tomo-bin-indices`, `--nbins`
- L1 extraction: `--l1-implementation`, `--n-scales`, `--l1-nbins`, `--l1-min-snr`, `--l1-max-snr`, `--auto-calibrate-snr`, `--l1-clamp-overflow`, `--subtract-coarse-mean`
- Flow: `--total-steps`, `--batch-size`, `--save-every`, `--patience`, `--nvp-layers`, `--nvp-hidden`
- Preprocessing: `--summary-transform`, `--clip-value`, `--pca-components`
- Runtime: `--cache-dir`, `--no-train`, `--no-sample`, `--plot`

Action:

1. Parse tomo bin string via `parse_tomo_bin_indices`.
2. If `--nbins` disagrees with selected bins, override `args.nbins = len(tomo_bin_indices)`.

Why this matters:

- Prevents silent channel mismatch between observed map, TFDS maps, and summary dimensionality.

### Step L1-2: Hardware/runtime init

`setup_environment`:

1. sets `CUDA_VISIBLE_DEVICES`;
2. enables TF memory growth;
3. selects PyTorch device;
4. reports JAX backend.

Why this matters:

- L1 extraction runs in PyTorch (`wl_stats_torch`) while NPE flow runs in JAX; device inconsistencies can produce hard-to-debug instability/performance issues.

### Step L1-3: Build observed datum (map + truth)

`load_observed_map`:

1. reads fiducial cosmology from `CosmoGridV1_metainfo.h5`;
2. converts `H0 -> h0` (`/100`);
3. projects selected tomographic lensing maps to square patch;
4. stacks into `(H, W, nbins)`;
5. adds Gaussian shape noise using survey noise model.

Outputs:

- `m_data` (observed noisy map patch)
- `truth` (6-parameter fiducial theta)

Audit points:

- Verify selected bins match expectation (e.g., `1,2,3,4` vs `3`).
- Verify projection geometry `(field-size, field-npix)` matches TFDS config.

### Step L1-4: Build L1 computer and augmentation

1. `build_l1_computer` creates `WLStatistics(n_scales=..., pixel_arcmin=...)`.
2. `build_augmentation` creates TF augmentation function with:
   - `tf.gather(..., tomo indices)`
   - shape-noise injection
   - flips
   - `h0` rescaling

L1 implementation mode:

- `--l1-implementation=cnn_sbi` (default): existing local extraction behavior.
- `--l1-implementation=cosmoford`: mirrors CosmOrford datavector construction details:
  - `WLStatistics` in float32
  - wavelet transform called with default coarse-mean handling
  - `clamp_overflow=False` for L1 histograms.

### Step L1-5: SNR range policy

Branch A (default): fixed range

- use `--l1-min-snr` and `--l1-max-snr` directly.

Branch B (`--auto-calibrate-snr`): calibrated range

1. run pilot scan (`calibrate_snr_range`) over training sample;
2. compute global min/max SNR in wavelet coeffs;
3. add margin;
4. optionally cache calibration in `snr_calibration.npz`.

Why this matters:

- SNR bin edges define the summary basis. Any mismatch between observed/train/val edges corrupts comparability.

### Step L1-6: Compute observed L1 vector

`compute_l1_single_map`:

1. for each selected tomographic bin:
   - wavelet transform
   - per-scale L1 histogram over SNR bins
2. concatenate across scales and bins.

Observed summary dimension:

- `raw_summary_dim = n_scales * l1_nbins * nbins`

### Step L1-7: Build train/val summary datasets

Cache check:

1. if `cache-dir` has `l1_train.npz`, `l1_val.npz`, `l1_cache_meta.npz`,
2. verify metadata keys and values:
   - `l1_min_snr`, `l1_max_snr`, `l1_nbins`, `l1_clamp_overflow`
   - `subtract_coarse_mean`, `n_scales`, `l1_implementation`
   - `tfds_name`, `tomo_bin_indices`
3. reuse cache only if all match.

If cache invalid/missing:

1. `compute_l1_dataset(... split=train ...)`
2. `compute_l1_dataset(... split=test ...)`
3. save train/val npz + metadata.

Why this matters:

- stale cache reuse is a classic source of “wrong-but-stable” posteriors.

### Step L1-8: Pre-flow diagnostics on raw L1

1. `plot_l1_diagnostics`:
   - plots observed vs train mean/std L1 curves per tomo bin and scale.
2. `log_l1_health_diagnostics`:
   - dead feature fraction
   - train std ranges
   - observed inlier fraction within train percentiles
   - clipping fractions (after standardization)

### Step L1-9: Summary preprocessing

`preprocess_summaries`:

1. apply transform chosen by `--summary-transform`:
   - `log1p-zscore` (default)
   - `log10p-zscore` (CosmOrford-style)
   - `zscore`
   - `log1p`
   - `log10p`
   - `none`
2. optionally clip transformed features to `±clip-value` (`0` disables clipping).

Outputs:

- processed train/val summaries
- processed observed summary
- preprocessing stats (`l1_standardization.npz`) including transform + clip metadata.

### Step L1-10: Optional PCA

If `--pca-components > 0`:

1. fit whitened PCA on train summaries;
2. transform train/val/obs summaries;
3. use reduced dimension as flow conditioner.

### Step L1-11: Flow training or loading

Flow architecture:

- conditional RealNVP via `build_flow`.

Training path (`--no-train` not set):

1. initialize flow parameters;
2. train with random mini-batches from train dataset;
3. periodic validation every `save-every`;
4. save:
   - `params_l1norm_flow_batch{step}.pkl`
   - best checkpoint `params_l1norm_flow_best.pkl`
5. early stopping with patience;
6. save loss arrays and `flow_training_summary.json`.

Load path (`--no-train`):

1. load preprocessing stats from `l1_standardization.npz` and reuse saved transform/clip (override conflicting CLI values);
2. if PCA stats are present, apply saved PCA transform and override conflicting `--pca-components`;
3. load best checkpoint, fallback to latest batch checkpoint by parsed step index.

### Step L1-12: Posterior sampling and outputs

If sampling enabled:

1. condition flow on observed standardized summary;
2. sample `--npe-samples`;
3. remove NaN rows;
4. save posterior `.npy`;
5. save metadata `.meta.json` with training/provenance info;
6. optional triangle plot.

---

## 2) L1 + jaxili pipeline (full sequence)

Reference script: `scripts/sbi/npe_l1norm_jaxili_nbody_tomo.py`

### Step JAX-1: Parse CLI and normalize scenario

Main controls follow the L1 pipeline for:

- geometry/data (`--field-size`, `--field-npix`, `--tfds-name`);
- tomography (`--tomo-bin-indices`, `--nbins`);
- L1 extraction (`--l1-implementation`, SNR controls, coarse-mean/clamp options);
- preprocessing (`--summary-transform`, `--clip-value`, `--pca-components`).

Estimator-specific controls:

- `--epochs`, `--batch-size`, `--learning-rate`
- `--npe-warmup-steps`, `--npe-decay-steps`
- `--checkpoint-name`
- `--nan-retries`
- `--min-feature-variance`

`--total-steps` is accepted as a compatibility alias and used as default `--epochs`.

### Step JAX-2: Reuse the exact L1 datavector construction path

The script intentionally reuses the same L1 stages as `npe_l1norm_nbody_tomo.py`:

1. observed map loading/projection/noise;
2. `WLStatistics` setup and optional SNR calibration;
3. observed L1 vector extraction;
4. train/val L1 datavector building via TFDS + augmentation;
5. cache reuse guarded by metadata (`l1_cache_meta.npz`).

This isolates estimator differences from datavector-construction differences.

### Step JAX-3: Preprocessing + optional PCA

As in the in-repo L1 pipeline:

1. apply configurable transform (`log1p-zscore`, `log10p-zscore`, `zscore`, ...);
2. optional clipping;
3. optional PCA fit/apply;
4. in `--no-train` mode, enforce saved preprocessing/PCA for compatibility.

Artifacts:

- `l1_jaxili_standardization.npz`

### Step JAX-4: Feature-stability filtering

Before NPE training:

1. compute train-feature variances;
2. drop features with variance `<= --min-feature-variance`;
3. persist mask and apply it to train/val/observed vectors.

Artifact:

- `l1_jaxili_feature_mask.npz`

Why this matters:

- avoids unstable/degenerate dimensions that can trigger NaN losses.

### Step JAX-5: jaxili NPE training/loading

Training path (default):

1. initialize `NPE()` and `append_simulations(theta_train, x_train)`;
2. train with checkpointing (`checkpoint_path = save_dir/l1norm_jaxili/<map_kind>/<checkpoint_name>`);
3. if train/val losses contain NaN, reinitialize and retry up to `--nan-retries`;
4. save `jaxili_training_summary.json`.

Load path (`--no-train`):

1. load existing checkpoint path from `--checkpoint-name`;
2. reuse saved preprocessing + feature mask artifacts.

### Step JAX-6: Posterior sampling and outputs

1. build posterior via `inference.build_posterior()`;
2. sample `--npe-samples` conditioned on observed processed L1 vector;
3. drop non-finite sample rows;
4. save posterior `.npy`;
5. save `.meta.json` with checkpoint/preprocessing/mask provenance;
6. optional triangle plot.

---

## 3) CNN pipeline (full sequence)

Reference script: `scripts/sbi/npe_cnn_nbody_tomo.py`

### Step CNN-1: Parse CLI and normalize scenario

Main controls include:

- data/tomo controls identical to L1 (`--tfds-name`, `--tomo-bin-indices`, geometry params);
- compressor controls:
  - pretrained paths (`--compressor-params`, `--compressor-state`)
  - or training (`--train-compressor`, steps/lr/batch/save cadence);
- summary controls:
  - `--standardize-summary` (default on)
  - `--summary-clip-value`
- flow controls similar to L1.

As in L1, `nbins` is synchronized with parsed tomo bin list.

### Step CNN-2: Runtime init and observed map

Same logic as L1:

1. hardware setup;
2. observed map loading/projection/noise injection;
3. truth parameter vector extraction.

### Step CNN-3: Build compressor and choose source

Compressor network (`CompressorCNN2D`) maps `(H,W,nbins) -> compressor_dim`.

Branch A: train compressor from scratch (`--train-compressor`)

1. `train_compressor_vmim` trains CNN + companion NF on TFDS;
2. checkpoints are written periodically;
3. optional intermediate contour diagnostics;
4. if cache exists, invalidate stale compressed caches.

Branch B: load pretrained compressor (default)

1. load params/state from provided pickle files;
2. log checkpoint provenance (path, file size, SHA256).

Why this matters:

- compressor quality directly controls information entering NPE.

### Step CNN-4: Compute observed compressed summary

Run compressor forward on observed map and get `obs_compressed`.

### Step CNN-5: Build compressed train/val datasets

Cache check (`cnn_train.npz`, `cnn_val.npz`, `cnn_cache_meta.npz`):

metadata includes:

- compressor source and dimension;
- dataset and geometry (`tfds_name`, bins, field size, npix, map kind);
- noise settings;
- compressor checkpoint identity (paths + SHA256 hashes).

Reuse cache only if metadata matches exactly.

If no valid cache:

1. run `compress_dataset(... split=train ...)`
2. run `compress_dataset(... split=test ...)`
3. save compressed datasets + metadata.

### Step CNN-6: Summary diagnostics and optional standardization

Diagnostics:

1. `plot_compressor_diagnostics` (summary-vs-theta scatter);
2. health metrics (std range, dead features, val/train mean shifts, observed inlier fraction).

Standardization branch (`--standardize-summary`, default true):

Training mode:

1. fit z-score stats on train summaries;
2. apply to train/val/obs;
3. optional clipping;
4. save stats in `cnn_summary_standardization.npz`.

`--no-train` mode:

1. try loading existing standardization stats;
2. apply if found;
3. if missing, warn and skip (to avoid mismatch with checkpoint assumptions).

Optional control:

- `--shuffle-theta-train` shuffles labels intentionally for sanity tests.

### Step CNN-7: Flow training/loading

Same pattern as L1:

1. build conditional RealNVP;
2. train with validation checkpoints and early stopping, or load existing checkpoint;
3. save `flow_training_summary.json`.

### Step CNN-8: Posterior sampling and outputs

1. sample posterior conditioned on observed compressed summary;
2. remove NaN samples;
3. save posterior `.npy`;
4. save metadata `.meta.json` (includes summary standardization provenance);
5. optional contour plot.

---

## 4) CNN + jaxili pipeline (full sequence)

Reference script: `scripts/sbi/npe_cnn_jaxili_nbody_tomo.py`

### Step CJAX-1: Parse CLI and normalize scenario

The script mirrors the CNN map/compressor controls:

- data/tomography/geometry (`--tfds-name`, `--field-size`, `--field-npix`, `--tomo-bin-indices`, `--nbins`);
- compressor checkpoint paths (`--compressor-params`, `--compressor-state`, `--compressor-dim`);
- summary preprocessing (`--standardize-summary`, `--summary-clip-value`);
- cache controls (`--cache-dir`) and run modes (`--no-train`, `--no-sample`, `--plot`).

Estimator controls are jaxili-specific:

- `--epochs` (with compatibility alias `--total-steps`);
- `--batch-size`, `--learning-rate`;
- `--npe-warmup-steps`, `--npe-decay-steps`;
- `--checkpoint-name`, `--nan-retries`;
- `--min-feature-variance`.

### Step CJAX-2: Reuse the exact CNN compression path

The script follows the same compressor stages as `npe_cnn_nbody_tomo.py`:

1. load observed map and add shape noise;
2. build CNN compressor architecture (`CompressorCNN2D`);
3. load pretrained compressor params/state;
4. compress observed map to `obs_compressed`;
5. compress train/test TFDS maps with identical augmentation;
6. guard cache reuse via `cnn_cache_meta.npz` metadata checks.

This keeps summary extraction fixed so differences are estimator-driven.

### Step CJAX-3: Summary preprocessing + feature filtering

1. optional summary standardization (fit/apply in train mode; load/apply in `--no-train`);
2. persist standardization stats to `cnn_jaxili_summary_standardization.npz`;
3. apply train-derived feature variance mask (`> --min-feature-variance`);
4. persist mask in `cnn_jaxili_feature_mask.npz` and reuse in `--no-train`.

### Step CJAX-4: jaxili training/loading

Training path:

1. initialize `NPE().append_simulations(theta_train, x_train)`;
2. train at absolute checkpoint path (`save_dir/cnn_jaxili/<map_kind>/<checkpoint-name>`);
3. on NaN-loss detection or train exceptions, retry by reinitializing NPE up to `--nan-retries`;
4. save `jaxili_training_summary.json`.

`--no-train` path:

1. load the existing jaxili checkpoint;
2. enforce saved preprocessing + feature mask consistency.

### Step CJAX-5: Posterior sampling and outputs

1. `build_posterior()` and sample conditioned on observed compressed vector;
2. drop non-finite rows;
3. save posterior `.npy`;
4. save `.meta.json` with compressor/checkpoint/preprocessing provenance;
5. optional contour plot.

---

## 5) Orchestrator sequence (systematic multi-run execution)

Reference: `scripts/sbi/run_cnn_l1_systematic_sweep.py`

High-level sequence:

1. parse matrix settings (methods, variants, seeds, flow/compressor hyperparameters);
2. validate GPU set and constraints;
3. define variants (`tomo4_10deg80`, `tomo4_20deg160`, `bin3_10deg80`, `bin3_20deg160`);
4. optionally train compressor once per variant (for CNN);
5. collect compressor checkpoint paths and verify existence;
6. schedule eval jobs for CNN/CNN+jaxili/L1/L1+jaxili across seeds and variants;
7. pass all critical flags explicitly (`tfds-name`, bins, SNR range, plotting flags, etc.);
8. collect return codes/log paths;
9. write run manifest and summaries.

---

## 6) Exact artifact chain to audit (all methods)

### L1 artifacts

- Cache:
  - `l1_train.npz`, `l1_val.npz`, `l1_cache_meta.npz`
- Flow:
  - `params_l1norm_flow_best.pkl`
  - `params_l1norm_flow_batch*.pkl`
  - `loss_train_l1norm.npy`, `loss_val_l1norm.npy`, `loss_val_steps.npy`
  - `flow_training_summary.json`
- Preprocess:
  - `l1_standardization.npz` (and optional PCA entries)
- Posterior:
  - `*.npy`
  - `*.meta.json`
  - optional contour `*.png`

### L1 + jaxili artifacts

- Cache:
  - `l1_train.npz`, `l1_val.npz`, `l1_cache_meta.npz`
- Preprocess:
  - `l1_jaxili_standardization.npz` (and optional PCA entries)
- Feature filtering:
  - `l1_jaxili_feature_mask.npz`
- jaxili estimator:
  - checkpoint path from `--checkpoint-name`
  - `jaxili_training_summary.json`
- Posterior:
  - `*.npy`
  - `*.meta.json`
  - optional contour `*.png`

### CNN artifacts

- Cache:
  - `cnn_train.npz`, `cnn_val.npz`, `cnn_cache_meta.npz`
- Compressor:
  - `params_nd_compressor_batch*.pkl`
  - `opt_state_resnet_batch*.pkl`
  - compressor loss arrays
- Summary preprocess:
  - `cnn_summary_standardization.npz`
- Flow:
  - `params_cnn_flow_best.pkl`
  - `params_cnn_flow_batch*.pkl`
  - `loss_train_cnn.npy`, `loss_val_cnn.npy`, `loss_val_steps.npy`
  - `flow_training_summary.json`
- Posterior:
  - `*.npy`
  - `*.meta.json`
  - optional contour `*.png`

### CNN + jaxili artifacts

- Cache:
  - `cnn_train.npz`, `cnn_val.npz`, `cnn_cache_meta.npz`
- Summary preprocess:
  - `cnn_jaxili_summary_standardization.npz`
- Feature filtering:
  - `cnn_jaxili_feature_mask.npz`
- jaxili estimator:
  - checkpoint path from `--checkpoint-name`
  - `jaxili_training_summary.json`
- Posterior:
  - `*.npy`
  - `*.meta.json`
  - optional contour `*.png`

---

## 7) Most likely failure points (ordered for debugging)

1. **Scenario mismatch**  
   geometry / TFDS config / tomo bin indices not aligned across observed map and train/val datasets.

2. **Stale cache reuse**  
   wrong cache reused despite changed settings (now guarded by metadata; still verify manually).

3. **Summary preprocessing mismatch**  
   flow checkpoint loaded with incompatible summary preprocessing. L1 now enforces saved preprocessing in `--no-train`; verify for older/legacy runs.

4. **Underconverged flow**  
   best val at final step (`flow_training_summary.json: best_at_final_step = true`).

5. **Compressor quality (CNN only)**  
   weak or undertrained compressor can flatten posterior geometry even if flow is fine.

6. **L1 extraction mode mismatch**  
   mixing datavectors built with different `l1_implementation`/coarse-mean/clamp behavior can shift contours.

7. **Observed-data out-of-distribution**  
   observed summary far outside train summary support (check inlier diagnostics).

8. **Historical checkpoint/preprocess mixing**  
   loading old flow checkpoints with newer preprocessing files can produce misleading contours.

9. **Zero-variance feature leakage (jaxili path)**  
   if train-time feature mask is not reused at inference, posterior conditioning becomes inconsistent.

---

## 8) Minimal end-to-end checklist for each run

For any run you want to trust:

1. Confirm run metadata: `tfds_name`, `tomo_bin_indices`, geometry, seed.
2. Confirm cache metadata match or explicit recomputation.
3. Confirm preprocessing file used by `--no-train` corresponds to same training lineage.
4. Confirm `flow_training_summary.json`:
   - best val not trivially at final step (or intentionally long-trained),
   - no obvious instability.
5. Confirm posterior metadata (`.meta.json`) points to expected flow source.
6. Compare observed summary diagnostics against train distribution diagnostics.

---

If you want, I can also generate a second version of this document as a strict “debug playbook” with explicit commands to run for each checkpoint above.
