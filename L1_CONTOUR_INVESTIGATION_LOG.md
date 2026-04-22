# L1 contour investigation log (readable summary)

## Goal

Understand why recent L1 posteriors look nearly circular in the `(Omega_m, sigma_8)` plane, while older artifacts looked banana-shaped.

---

## Step 1 — Compare old vs new outputs directly

I compared the legacy reference posterior and the latest rerun:

- Legacy reference: `scripts/sbi/posterior_l1norm_tomo.npy`
- New rerun: `scripts/sbi/systematic_runs_l1_snr10_rerun/posteriors/l1_tomo4_10deg80_s41.npy`

Key metrics:

- Legacy: `corr(Omega_m, sigma_8) = -0.714`, axis-ratio `~3.08` (banana-like).
- New: `corr(Omega_m, sigma_8) = -0.013`, axis-ratio `~1.41` (almost circular).

---

## Step 2 — Run targeted ablations to isolate causes

I reran the same variant (`tomo4_10deg80`, seed 41) with controlled changes:

1. **PCA on/off**
   - `pca=50` and `pca=0` both stayed near-circular (`corr ~ -0.01`).

2. **Old-like flow/training settings**
   - Larger flow (`nvp_layers=6`, `hidden=256`), `lr_end=1e-6`, no clipping, `total_steps=50000`.
   - Still near-circular (`corr ~ -0.017`).

3. **`n_scales=6`**
   - Still near-circular (`corr ~ -0.015`).

4. **No coarse-mean subtraction**
   - Still near-circular (`corr ~ -0.012`).

Conclusion from ablations: the banana did **not** come back by toggling these expected knobs.

---

## Step 3 — Verify implementation consistency

I checked the L1 extraction path for hidden numerical issues:

- Batch vs single-map L1 computations match very closely (`max abs diff ~1.17e-4`, numerically consistent).
- Coarse scale is included in summaries.
- `SNR=[-10,10]` is correctly wired and used.

So this does **not** look like a straightforward bug in current L1 feature extraction.

---

## Step 4 — Check whether old banana could be a prior imprint

I measured `(Omega_m, sigma_8)` correlation in training cosmologies:

- Training theta correlation from cache: `~ -0.691`.
- Legacy posterior correlation: `~ -0.714`.

These are very close, which suggests the old banana shape may have been strongly influenced by prior/training correlation structure.

---

## Step 5 — Reproducibility audit of old “good” runs

I audited the old W&B runs that wrote `posterior_l1norm_tomo.npy`:

- W&B metadata shows git commit `08b8e0a`.
- But configs/logs include options not present in that committed script at that time (e.g., `clip_value`, `calibration_samples`, special cache checks), implying local uncommitted code was likely used.
- `scripts/sbi/save_params/l1norm/nbody/` contains mixed checkpoint families (very different file sizes), indicating checkpoint/state overwrites across incompatible setups.
- `l1_standardization.npz` was updated later than many checkpoints, so old checkpoint + newer preprocessing can produce invalid posteriors.

This explains why the old behavior is hard to reproduce exactly and why artifacts may look inconsistent.

---

## Current best interpretation

1. Recent code path is stable and reproducible, but yields weak/near-circular `(Omega_m, sigma_8)` for L1 in this setup.
2. Older banana-looking artifacts are likely not a clean, reproducible baseline from one versioned configuration.
3. A substantial part of the old banana signal may reflect prior/training correlation and mixed historical artifacts (dirty code + stale/mixed checkpoints).

---

## Artifacts produced in this investigation

- Metrics summary JSON:
  `scripts/sbi/diagnostics/l1_regression_investigation_metrics.json`

- Controlled ablation outputs:
  - `scripts/sbi/investigate_pca_regression/posteriors/`
  - `scripts/sbi/investigate_nscales6/posteriors/`
  - `scripts/sbi/investigate_no_coarse/posteriors/`
  - `scripts/sbi/investigate_old_script/posteriors/`

---

## Next concrete step (recommended)

Run one controlled experiment with **decorrelated** training prior in `(Omega_m, sigma_8)` (or importance-reweighted training) and compare contour tilt.
This will tell us whether the expected banana is genuinely data-driven in this pipeline, versus inherited from prior geometry.

