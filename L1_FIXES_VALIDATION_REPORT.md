# L1-Norm Pipeline Fixes: Implementation and Validation Report

## Scope
This report summarizes the **successful** fixes applied to `scripts/sbi/npe_l1norm_nbody_tomo.py` and the tests run to verify they work.

## What was changed successfully

1. **Aligned default SNR range with CosmOrford-style behavior**
   - `--l1-min-snr`: changed to `-7.0`
   - `--l1-max-snr`: changed to `7.0`

2. **Made SNR calibration explicit**
   - Added `--auto-calibrate-snr` (off by default).
   - Default mode now uses fixed SNR bounds unless calibration is explicitly requested.

3. **Made overflow handling explicit and reference-compatible**
   - Added `--l1-clamp-overflow` (off by default).
   - L1 computation now forwards `clamp_overflow` explicitly in:
     - `compute_l1_single_map`
     - `compute_l1_batch`
     - `compute_l1_dataset`

4. **Fixed coarse-mean CLI behavior**
   - Replaced ambiguous flag setup with:
     - `--subtract-coarse-mean`
     - `--no-subtract-coarse-mean`
   - Default remains `subtract_coarse_mean=True`, but users can now reliably disable it.

5. **Hardened cache validity checks**
   - Cache metadata now includes and validates:
     - `l1_min_snr`, `l1_max_snr`, `l1_nbins`
     - `l1_clamp_overflow`
     - `subtract_coarse_mean`
     - `n_scales`
   - Prevents reusing stale summaries generated under incompatible settings.

6. **Added L1 feature health diagnostics**
   - Added `log_l1_health_diagnostics(...)` and logging for:
     - raw feature std min/median/max
     - dead-feature fraction
     - zero-entry fraction
     - observed-in-train percentile consistency
     - standardized abs-max and clipping fractions

## Tests run and evidence of success

### A) Syntax and CLI checks
Command:
```bash
python -m py_compile scripts/sbi/npe_l1norm_nbody_tomo.py
python scripts/sbi/npe_l1norm_nbody_tomo.py --help
```
Result: both completed successfully (exit code 0).

### B) End-to-end no-train/no-sample sanity run
Command:
```bash
python scripts/sbi/npe_l1norm_nbody_tomo.py \
  --no-wandb --cache-dir scripts/sbi/cache_l1 --save-dir scripts/sbi/save_params \
  --map-kind nbody --no-train --no-sample
```
Key successful log evidence:
- `Using fixed SNR range: [-7.0, 7.0]`
- `Loading cached L1-norm datasets (metadata matches) ...`
- `PCA: 800 → 50 components (56.4% variance explained)`
- `Loaded flow params .../params_l1norm_flow_best.pkl`
- `Done.`

### C) End-to-end no-train + posterior sampling run
Command:
```bash
python scripts/sbi/npe_l1norm_nbody_tomo.py \
  --no-wandb --cache-dir scripts/sbi/cache_l1 --save-dir scripts/sbi/save_params \
  --map-kind nbody --no-train --npe-samples 20000 \
  --posterior-out scripts/sbi/posterior_l1norm_postfix.npy \
  --figure-out scripts/sbi/posterior_l1norm_postfix.png --plot
```
Key successful log evidence:
- `Loading cached L1-norm datasets (metadata matches) ...`
- `Saved posterior samples → .../scripts/sbi/posterior_l1norm_postfix.npy`
- `Done.`

Generated outputs:
- `scripts/sbi/posterior_l1norm_postfix.npy`
- `scripts/sbi/posterior_l1norm_postfix.png`

### D) Fresh short training run (isolated output dir)
Command:
```bash
python scripts/sbi/npe_l1norm_nbody_tomo.py \
  --no-wandb --cache-dir scripts/sbi/cache_l1 \
  --save-dir scripts/sbi/save_params_postfixtest --map-kind nbody \
  --total-steps 300 --save-every 150 --no-sample
```
Key successful log evidence:
- `Saved @ step 150. Val loss = -6.3217`
- `Saved @ step 300. Val loss = -7.0430`
- `Done.`

This shows training runs stably and validation loss improves over the short sanity run.

### E) Sampling from the freshly trained checkpoint
Command:
```bash
python scripts/sbi/npe_l1norm_nbody_tomo.py \
  --no-wandb --cache-dir scripts/sbi/cache_l1 \
  --save-dir scripts/sbi/save_params_postfixtest --map-kind nbody \
  --no-train --npe-samples 20000 \
  --posterior-out scripts/sbi/posterior_l1norm_postfixtest.npy
```
Result: successful sample generation with output shape `(20000, 6)`.

Generated output:
- `scripts/sbi/posterior_l1norm_postfixtest.npy`

## Conclusion
The L1 pipeline fixes were implemented and validated successfully. The script now uses clearer, reference-aligned defaults, explicit behavior controls, safer cache reuse, and improved diagnostics, and it runs successfully across no-train, training, and sampling workflows.
