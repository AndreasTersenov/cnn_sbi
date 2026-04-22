# Zero-mean maps parity check — CNN posteriors with mass-sheet-degeneracy-consistent inputs

## Objective

Test whether the CNN-VMIM compressor had been exploiting per-channel spatial means of
training convergence maps — information that is **unphysical** for real weak-lensing
data because each redshift bin's convergence is recoverable only up to an additive
constant (mass-sheet degeneracy). Decision metric: change in posterior width and
3D Figure-of-Merit (FoM3 = exp(−0.5·log det Cov₃)) for both BNT and no-BNT, across
two strong reference compressors, after subtracting each example's per-channel spatial
mean just before the compressor (implemented via new `--zero-mean-maps` flag).

## Configuration fingerprint

- Code: live `scripts/sbi/npe_cnn_nbody_tomo.py` with new `--zero-mean-maps` flag
  injected at 2 sites (`load_observed_map` after noise injection / before BNT;
  `build_augmentation` in both standard and paired branches), propagated at 3
  call-sites, persisted into cache meta and posterior `.meta.json`.
- TFDS: `NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48`, 20 deg × 160 px,
  4 tomographic bins (`--tomo-bin-indices 1,2,3,4`), truth
  `θ = [0.26, 0.84, −1.0, 0.6736, 0.9649, 0.0493]`.
- Mathematical safety: `B(x − m·1) = B x − B m·1` is still zero-mean per channel in
  BNT space, so demean-before-BNT is invariance-preserving for paired training.
- Fresh compressor checkpoints trained on demeaned input distribution (old
  checkpoints are physically incompatible and were not reused).
- Run A (`run_a_resnet18` / label `resnet18_long15k_nostd6k_l8h256_zm`):
  arch=resnet18, cdim=6, compressor_steps=15000, flow_steps=6000,
  nvp l8 h256, no summary standardization, splits `train[:70%]` / `train[70%:]`
  with `--require-disjoint-train-examples`, seeds 41–43.
- Run B (`run_b_advanced_plain` / label `advanced_arch64_dense256_nostd_long_zm`):
  arch=plain (conv 64,128,256; dense 256; pool 16/8), cdim=10,
  compressor_steps=120000, flow_steps=10000, nvp l8 h256, no summary
  standardization, splits `train`/`test`, seeds 41–45.
- Campaign driver:
  `scripts/sbi/results/exploratory/zero_mean_maps_parity_check/run_zero_mean_parity.py`
  (3 GPUs, XLA mem fractions `0:0.40,1:0.40,2:0.50`).
- Overlay + metrics driver: `plot_4way.py` using the new
  `_plot_overlay_4way` added to
  `scripts/sbi/run_cnn_bnt_losslessness_campaign.py` (after `_plot_overlay`
  at line 513), and the existing `_metrics_for_paths`.

## Quantitative outcomes

| config | variant | regime | std_sum_mean | fom3_mean | σ₈ std mean | inflation BNT/noBNT | FoM3 ratio BNT/noBNT | n_seeds |
|---|---|---|---|---|---|---|---|---|
| run_a_resnet18 | old | no-BNT | 0.1910 | 3.60e5 | 0.0148 | 1.056 | 0.781 | 3 |
| run_a_resnet18 | old | BNT    | 0.2016 | 2.81e5 | 0.0180 | 1.056 | 0.781 | 3 |
| run_a_resnet18 | **new** | no-BNT | **0.3441** | **1.49e4** | **0.0417** | 1.077 | 0.630 | 3 |
| run_a_resnet18 | **new** | BNT    | **0.3706** | **9.41e3** | **0.0501** | 1.077 | 0.630 | 3 |
| run_b_advanced | old | no-BNT | 0.1770 | 5.38e5 | 0.0139 | 1.077 | 0.749 | 5 |
| run_b_advanced | old | BNT    | 0.1906 | 4.03e5 | 0.0157 | 1.077 | 0.749 | 5 |
| run_b_advanced | **new** | no-BNT | **0.3456** | **1.69e4** | **0.0382** | 1.033 | 0.867 | 5 |
| run_b_advanced | **new** | BNT    | **0.3571** | **1.46e4** | **0.0401** | 1.033 | 0.867 | 5 |

Key ratios (new / old):

| config | regime | std_sum_mean | σ₈ std | FoM3 | det(Cov₃) |
|---|---|---|---|---|---|
| run_a_resnet18 | no-BNT | **1.80×** | **2.81×** | **0.042×** (24× worse) | ≈576× |
| run_a_resnet18 | BNT    | **1.84×** | **2.78×** | **0.034×** (30× worse) | ≈900× |
| run_b_advanced | no-BNT | **1.95×** | **2.76×** | **0.031×** (32× worse) | ≈1020× |
| run_b_advanced | BNT    | **1.87×** | **2.56×** | **0.036×** (28× worse) | ≈770× |

Observed-map per-channel means that the pipeline was subtracting (Run A,
seed 41, no-BNT): `[0.00777, 0.01676, 0.03667, 0.05491]`. The bin-4 value is
about 4× the per-pixel shape-noise std (0.01266), so there was substantial
unphysical signal available to the compressor.

BNT-vs-no-BNT parity (the original scientific target of `bnt-parity-techniques`):
- Run A: std inflation 1.056 → 1.077 (slightly worse); FoM3 ratio 0.781 → 0.630 (worse).
- Run B: std inflation 1.077 → 1.033 (closer to 1.0); FoM3 ratio 0.749 → 0.867 (closer to 1.0).

Overlays:
- `overlays/run_a_resnet18_4way_overlay.{png,pdf}`
- `overlays/run_b_advanced_plain_4way_overlay.{png,pdf}`

Both show the "new" contours (green = no-BNT, purple = BNT) enveloping the "old"
contours (blue = no-BNT, red = BNT) by roughly a factor of two in each marginal.

## Robustness

- Pipeline integrity verified by a short smoke test on GPU 0 (`smoke/` subtree):
  observed-map assertion `|mean| < 1e−5` fires; first-batch per-channel spatial
  mean in `compress_dataset` drops to abs max 3.8e−7, consistent with single-
  precision rounding.
- Posterior `.meta.json` for every new run records `"zero_mean_maps": true`.
- Results reproduce across seeds: sample standard deviations inside each
  `variant × regime × config` cell are stable (e.g., Run B new-BNT σ₈ std:
  0.0401 mean across 5 seeds; see `metrics/comparison_old_vs_new.json` for
  full per-seed rows).
- Same set of seeds (41–43 for Run A, 41–45 for Run B) as the originals.
- Same TFDS splits, same flow architecture (l8/h256), same `--no-standardize-summary`,
  same truth. The only varied factor is per-example per-channel demeaning of the
  maps (new compressor + new flow trained on demeaned inputs).

## Scientific conclusion

The original CNN-VMIM posteriors were **artificially tight by a factor of roughly
2× on each parameter's marginal, ≈2.7–2.8× on σ₈, and 24–32× on FoM3** because the
compressor was exploiting the per-example per-channel spatial mean of the simulation
maps — an information channel that is unavailable in real data under the mass-sheet
degeneracy. This holds for both the resnet18 and the deep-plain-cdim10 best-performing
references, so it is not architecture-specific.

Demeaning before compression does **not** eliminate the BNT vs no-BNT contour
mismatch: the std-sum inflation ratio remains close to ~1.03–1.08 in both regimes,
and the FoM3 ratio remains below 1.0 in both setups. So mass-sheet-degeneracy
leakage is a separate and larger issue from the BNT-parity issue, and the two must
be fixed independently.

Practical consequence: every CNN-VMIM posterior produced in this repo prior to
this patch over-states its constraining power by a factor of ~2 in marginals and
~25× in FoM3. Any future scientific claim from this pipeline has to pass through a
`--zero-mean-maps` pipeline (or equivalent compressor-level invariance) before it
is reportable.

## Minimal next action

1. Adopt `--zero-mean-maps` as the default for all future paper-track CNN-VMIM
   runs (keep the flag default OFF to stay backward-compatible with existing
   campaign scripts; opt every new run in explicitly).
2. Re-audit the L1 and L1-VMIM pipelines for the same issue: confirm
   (or refute) that their statistics are mean-invariant, and document the
   result. If they depend on map means, repeat this exercise for L1.
3. Re-launch the strongest BNT-parity sweeps (resnet_extended_tuning_v2,
   losslessness_campaign_multipatch_advanced_cdim10_long120k, and the
   noise-curriculum follow-ups) with `--zero-mean-maps` enabled and rank on
   `abs(fom3_ratio_bnt_over_nobnt − 1)` against the new wider no-BNT baseline.
