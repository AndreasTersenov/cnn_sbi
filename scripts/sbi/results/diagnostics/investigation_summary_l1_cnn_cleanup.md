# SBI L1 vs CNN Investigation Summary (this session)

## Goal
- Diagnose why L1-based SBI contours looked wrong relative to CNN-based contours.

## What we changed in code
- `scripts/sbi/npe_l1norm_nbody_tomo.py`
  - Added configurable summary preprocessing modes:
    - `log1p-zscore`, `log10p-zscore`, `zscore`, `log1p`, `log10p`, `none`
  - Added robust `--no-train` compatibility:
    - load and enforce saved preprocessing stats (`l1_standardization.npz`)
    - load and enforce saved PCA transform when available
  - Added `--l1-implementation {cnn_sbi,cosmoford}` to isolate datavector construction differences:
    - `cosmoford` mode uses float32 WLStatistics, default coarse-mean handling, `clamp_overflow=False`
  - Added safety guards:
    - validate train/val/obs tensor shapes and finiteness before flow training
    - fail fast on non-finite train/val loss
    - fail if all posterior samples are NaN

## Cross-check against CosmOrford
- Inspected `CosmOrford/cosmoford/summaries.py` and `models.py`.
- Confirmed CosmOrford L1 feature construction differs from original cnn_sbi defaults in details:
  - float32 computation path
  - no explicit coarse-mean toggle in call site
  - `clamp_overflow=False`
- Also confirmed CosmOrford default feature normalization in that codepath is `log10(1+x) + z-score` on the batch.

## Experiments run and outcomes
- Baseline L1 (existing/default preprocessing):
  - Stable training, finite losses, contour produced.
- No preprocessing (`summary-transform=none`, no clip, no PCA):
  - Train/val loss became NaN quickly.
  - Posterior values exploded/unphysical.
- CosmOrford-style preprocessing (`log10p-zscore`, no clip, no PCA):
  - Stable, but contour shift did not resolve the core concern.
- CosmOrford-style datavector construction only (`--l1-implementation cosmoford`) with default preprocessing:
  - Stable, contour changed, but still did not clearly explain the original issue.
- Simplified L1 setting request:
  - Requested `n_scales=1` is invalid in `wl_stats_torch` (requires at least 2).
  - Ran nearest valid setup: `n_scales=2`, `SNR [-3,3]`, `10 bins`.
  - 300-step run: final losses finite.
  - 5000-step run: final train loss `-9.1090`, final val loss `-9.0624`, contour produced.

## Key conclusions so far
- The density estimator architecture/training recipe is effectively matched between CNN and L1 scripts.
- Catastrophic failure appears when preprocessing is removed entirely (NaN divergence).
- Neither switching only preprocessing style nor only datavector construction style fully explained the contour discrepancy.
- Remaining likely causes are data-information content / degeneracy / training-convergence quality rather than a single obvious pipeline bug.

## Cleanup performed in this session
- Removed session-generated investigation folders:
  - `scripts/sbi/investigate_preproc_ablation/`
  - `scripts/sbi/investigate_simple_l1/`
  - `scripts/sbi/cache_l1_cosmoford_impl/`
  - `scripts/sbi/cache_l1_simple_ns2_snr3_b10/`
  - `scripts/sbi/cache_l1_simple_ns1_snr3_b10/` (if present)
- Removed `*.pyc` under `scripts/sbi`.

## Remaining ambiguous artifacts (not removed automatically)
- Many pre-existing untracked notebooks, logs, caches, and run directories already existed in repo.
- These were intentionally left untouched to avoid deleting potentially important user data.
