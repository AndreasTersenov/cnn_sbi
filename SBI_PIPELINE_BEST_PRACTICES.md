# SBI Pipeline Best Practices

Prescriptive guide for running new weak-lensing SBI experiments in this repo. Distilled
from the campaigns documented in `PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md` — **read that
document for the evidence behind each rule here.** This file tells you *what to do*;
the KB tells you *why*.

Scope: CNN-VMIM and wavelet-L1 pipelines on tomographic n-body convergence maps from
CosmoGridV1. Applies to the active branch `bnt-parity-techniques`.

---

## 1. Dataset construction

### 1.1 Always use multi-patch extraction from the sphere

**Rule.** For every new experiment, use the TFDS builder
`NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48`
(`scripts/sbi/tf_dataset_nbody_tomo.py`).

It extracts **48 non-overlapping 20°×20°/160-px patches per CosmoGridV1 spherical
realization**, centered on an `nside=32` grid with ≥28.5° minimum angular separation.
This:

- Multiplies effective sample count per cosmology by ~48.
- Keeps patches independent enough for IID assumptions to hold (the 28.5° cut is what
  makes paired BNT/no-BNT studies trustworthy).
- Runs only through `grid` and `grid_20deg_160px` are **legacy** — do not launch new
  campaigns on them.

### 1.2 Use the cosmology grid (train) and fiducial (test) the way the builder was designed

- `train` split: cosmologies indexed 1–900 (grid parameter sweep). Use for training both
  the compressor and the NDE.
- `test` split: cosmologies 900–1300 (includes the fiducial realizations). Use for
  validation of both the compressor and the flow.

Never mix the two or swap their roles — the fiducial validation realizations are the
*only* unbiased estimate of generalization we have.

### 1.3 Fixed lensing geometry and tomography

For any new 20°/160-px campaign:

```
--field-size 20 --field-npix 160
--nbins 4 --tomo-bin-indices 1,2,3,4
```

If you want 3-bin, 2-bin, or no-tomography ablations, run them as *additional* arms,
not as substitutes for the 4-bin baseline.

---

## 2. Split the training set into two independent parts

### 2.1 Why

Using the same examples to train a neural compressor and then fit the density
estimator on its outputs leaks information about the compressor's per-example output
into the NDE, which narrows posteriors on held-out fiducial data. Evidence: the
`indep_split_advanced_cdim10_long120k` campaign recovered near-parity BNT inflation
(1.0369) and the expected FoM hit (0.85×) where the pooled campaign showed a stronger
signal that did not survive the split.

### 2.2 How

Pass these four splits to `npe_cnn_nbody_tomo.py` (and `npe_l1vmim_nbody_tomo.py`):

```
--compressor-train-split "train[:70%]"
--compressor-val-split   "test"
--nde-train-split        "train[70%:]"
--nde-val-split          "test"
--require-disjoint-train-examples
```

`--require-disjoint-train-examples` asserts, at data-load time, that the compressor and
NDE train partitions contain **disjoint `(cosmology, patch)` pairs**. Without it, the
70%/30% split is only defined at the cosmology-index level and patches from the same
cosmology can appear in both halves, which still leaks.

### 2.3 When you may not need this

Pure L1 / L1-VMIM runs where the "compressor" is a hand-coded wavelet operator (no
trainable parameters seeing the NDE-train data) can use the full `train` split for the
NDE. But if you introduce **any** trainable compression step (L1+VMIM, PCA fitted to
the training set, etc.), go back to the 70/30 split.

### 2.4 Alternative: shared-cosmology (random-example) split

**Status:** untested in this repo. Plausible; consider as an ablation.

The canonical split above inherits *approximate cosmology-disjointness* as a side
effect of how TFDS orders examples (patches are generated cosmology-by-cosmology, so
`train[:70%]` falls mostly between cosmologies, with at most one straddler). The code
only enforces disjoint `(cosmology, patch)` **examples**, not disjoint cosmologies —
`--require-disjoint-train-examples` reports `shared_theta_count` but does not require
it to be zero.

The alternative is to shuffle examples across cosmologies *before* the 70/30 cut, so
that each cosmology contributes ~34 patches to the compressor-train set and ~14 to the
NDE-train set. Example-level disjointness is still trivially preserved. Motivation:

- Both networks see the full Ω_m/σ₈/w₀/h₀/n_s/Ω_b grid, not complementary slices of
  it.
- In principle this reduces the risk that the NDE has to extrapolate into cosmology
  regions the compressor was not trained on.

Tradeoff to be honest about: the NDE then trains on compressor outputs from
cosmologies the compressor has specifically adapted to, so a residual leakage channel
exists (weaker than patch-level overlap, stronger than cosmology-disjoint training).
No empirical evidence in this repo yet on whether that channel meaningfully narrows
posteriors.

Implementation path (not a supported CLI flag today):

- Simplest: use interleaved TFDS subsplit specs (e.g.
  `train[:7%]+train[10%:17%]+train[20%:27%]+...` for compressor,
  complementary slices for NDE) so each ~10% chunk is split 70/30 internally. This is
  cosmology-level stratification and is closer to a random split than the canonical
  70/30 cut.
- Keep `--require-disjoint-train-examples` on — it still guarantees no patch appears
  in both halves.
- Tag results clearly as `shared_cosmo_split` so they do not accidentally get averaged
  with canonical-split numbers.

If you run this, compare head-to-head against §2.2 with the same compressor config and
seeds, and look specifically at `inflation_std_sum_bnt_over_nobnt` and
`fom3_ratio_bnt_over_nobnt` on the fiducial test split.

---

## 3. Always subtract the per-map, per-channel mean (mass-sheet degeneracy)

### 3.1 The rule

```
--zero-mean-maps
```

Pass this to **every** CNN and CNN-VMIM run unless you are deliberately reproducing a
legacy comparison. It subtracts the spatial mean of each tomographic channel of each
example, at load time, before the compressor.

### 3.2 Why

Real weak-lensing convergence κ is reconstructed only up to an unknown additive
constant per redshift bin (the mass-sheet degeneracy). The CNN otherwise discovers that
the simulated per-channel mean is a clean, shot-noise-free function of cosmology — a
feature no real survey will ever supply. Evidence: pre-demean CNN-VMIM marginals on
σ₈/Ω_m were roughly 2× tighter and FoM3 ~25–30× larger than the demeaned equivalents
on identical configurations. Every CNN-VMIM scientific claim in this repo now has to go
through the demeaned pipeline.

### 3.3 Don't reuse old compressor checkpoints after flipping this flag

`--zero-mean-maps` changes the input distribution the compressor sees. Old checkpoints
are physically incompatible. Do **not** use `--no-train-compressor` across the boundary.

### 3.4 Cache safety

`scripts/sbi/npe_cnn_nbody_tomo.py` writes `zero_mean_maps` into the compressor-cache
metadata and into the posterior `.meta.json`, so a mismatched cache is rejected
automatically. You do not need to manually purge caches between demeaned and
non-demeaned runs — but **verify the assertion fires** on the first training step of a
new campaign (`m_data.mean(axis=(0,1))` must be <1e-6 per channel).

---

## 4. Canonical compressor configurations

Two compressors have passed all the sanity checks (paired BNT/no-BNT parity on
noiseless data, independent-split stability, reasonable generalization to the fiducial
`test` split). Start from one of these and vary only the factor you need to study.

### 4.1 ResNet18, 6-dim summary — the default recommendation

Reference run: `resnet18_long15k_nostd6k_l8h256`
(`scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_resnet_split_campaign/resnet_extended_tuning_v2/backbones/resnet18/...`).

```
--compressor-arch resnet18
--compressor-dim 6
--compressor-steps 15000
--compressor-batch-size 128
--compressor-lr 5e-4
--compressor-save-every 3000
--batch-size 256
--patience 35
```

Trains in ~2 hours on a single modern GPU. Use this as the **default compressor for
quick exploratory or diagnostic runs**, noise-curriculum studies, and anything where
training budget matters.

### 4.2 Plain CNN with wider pooling, 10-dim summary — the best parity

Reference run: `advanced_arch64_dense256_nostd_long`
(`cnn_bnt_losslessness_campaign_indep_split_advanced_cdim10_long120k_v1`).

```
--compressor-arch plain
--compressor-conv-channels 64,128,256
--compressor-dense-width 256
--compressor-pool-window 16
--compressor-pool-stride 8
--compressor-dim 10
--compressor-steps 120000
--compressor-batch-size 128
--compressor-lr 5e-4
--compressor-save-every 3000
--batch-size 256
--patience 50
```

Takes ~8× longer to train than the ResNet18 default. Use it when the run is the final
headline number — it gives the best-observed BNT/no-BNT inflation ratio (~1.03) and the
best FoM parity across BNT conditions. Not recommended for turnarounds under a day.

### 4.3 What to avoid

- **ResNet34 / ResNet50**: explored in `resnet_extended_tuning_v2` and do not beat
  ResNet18 at matched budget. The extra depth buys nothing on 160-px maps.
- **Standardize-summary (`--standardize-summary`)**: degrades flow training on small
  summaries. Use `--no-standardize-summary` (see §5) and rely on `--summary-clip-value`
  instead.
- **`--compressor-dim` below 6 or above ~12**: under 6 underfits (esp. for 4-bin
  tomography); over ~12 the flow overfits the compressor's idiosyncrasies before it
  can see the cosmology signal.

---

## 5. Conditional RealNVP flow (NDE)

Matching the compressor references above:

```
--nvp-layers 8
--nvp-hidden 256
--no-standardize-summary
--summary-clip-value 5.0
--total-steps 6000          # resnet18 / cdim=6
# or 10000                   # advanced_plain / cdim=10
--save-every 500
--npe-samples 100000
--ds-batch-size 500
```

Notes:

- `--summary-clip-value 5.0` clips the compressed summary at ±5 per-dim
  (post-compressor, pre-flow) — cheap insurance against compressor outliers without
  the drift that comes from standardization.
- `--nvp-layers 10 --nvp-hidden 320` was tried and does **not** help once
  `--no-standardize-summary` is on. Do not use the deeper flow as a first move.
- If you see the flow diverge on BNT (loss goes NaN, or rapid posterior collapse),
  the most common cause is a broken BNT path (shape noise applied in the wrong
  order — see §6), not the flow hyperparameters.

---

## 6. BNT path (only valid in full 4-bin tomography)

### 6.1 Hard invariants

- `--apply-bnt` requires `--nbins 4 --tomo-bin-indices 1,2,3,4`. The assertion in
  `validate_bnt_configuration` will refuse anything else.
- **Shape noise goes in before BNT**, always. The BNT matrix is applied in
  `scripts/sbi/bnt_utils.py:apply_bnt_tf` / `apply_bnt`, after noise injection. Do not
  reorder.

### 6.2 Paired BNT / no-BNT training

Use `--compressor-paired-bnt-nobnt-consistency` when the aim is a clean BNT-vs-no-BNT
comparison from the **same compressor**. It returns dict features
`{maps_nobnt, maps_bnt}`. Downstream, `compress_dataset(..., paired_map_view=...)`
must select one view or it will `KeyError: 'maps'`. This plumbing is a known fix
point — do not refactor around it.

### 6.3 Zero-mean and BNT

`--zero-mean-maps` is applied to the pre-BNT map. Because the BNT matrix is linear and
row-sum-zero in practice, a pre-BNT demean is still a zero-mean map post-BNT. Do not
demean twice.

---

## 7. Parameter and preprocessing conventions (non-negotiable)

- **Parameter order is fixed**: `theta = [Omega_m, sigma_8, w0, h0, n_s, Omega_b]`.
- `h0 = H0 / 100` is applied in preprocessing (`theta[3] /= 100`). Do not pass H0 in
  km/s/Mpc to the flow.
- Fiducial truth in the pipeline:
  `[0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493]`.
- Log1p / z-score only on L1 datavectors; maps go into the CNN raw (after
  shape-noise and demean). **No PCA by default** — L1-VMIM is the preferred
  compression path.

---

## 8. Evaluation metrics to report

Every new campaign's `SUMMARY.md` must include, per configuration:

- `std_sum_mean`: sum of marginal stds on [Omega_m, sigma_8, w0], averaged across
  seeds.
- `sigma8_std_mean`, `omega_m_std_mean`: individual marginals.
- `FoM3 = exp(-0.5 * log det Cov3)` on the (Omega_m, sigma_8, w0) block, averaged
  across seeds.
- `inflation_std_sum_bnt_over_nobnt`: the BNT contour inflation ratio, with a target
  window of **0.95–1.10** for any claim of BNT parity.
- `fom3_ratio_bnt_over_nobnt`: expected near 0.85 for well-trained, demeaned,
  independent-split compressors. Values much above 1.0 are a warning sign of a
  train/eval mismatch.
- Number of seeds (minimum 3 for exploratory, minimum 5 for any final claim).

Include a corner plot PDF and the raw posterior `.npy` + `.meta.json` per seed. No
`.pkl` checkpoints in final result trees.

---

## 9. Operational hygiene

- Conda env is `jaxili`; prefix every command with `conda run -n jaxili python ...`.
- W&B logging is expected on any non-dry run. Use `--no-wandb` only for smoke tests.
- Results go under `scripts/sbi/results/{dryruns,exploratory,diagnostics,final}/` —
  in that order of promotion. A result only moves to `final/` when there is a
  SUMMARY.md and a reproducer script committed next to it.
- Drivers accept `--gpus 0,1,2,3` and
  `--xla-mem-fraction-by-gpu 0:0.45,1:0.45,2:0.65` for heterogeneous placement.
- Never `git add .` or `git add -A`; stage by path. Do not commit caches,
  `__pycache__`, or `.pyc` files (already gitignored — don't re-add them).

---

## 10. Anti-patterns (short checklist before launching)

Before hitting `enter` on a big run, confirm:

- [ ] `--zero-mean-maps` is on. If not, you need an explicit reason.
- [ ] You are using `grid_20deg_160px_nonoverlap48`, not `grid` or `grid_20deg_160px`.
- [ ] The 70/30 split with `--require-disjoint-train-examples` is on (unless this is
      a pure-L1 ablation).
- [ ] Compressor config matches §4.1 or §4.2, not a hybrid.
- [ ] Flow config matches §5. No `--standardize-summary`.
- [ ] If `--apply-bnt`: full 4-bin tomography, shape noise before BNT, paired-view
      plumbing intact.
- [ ] You are not reusing a compressor checkpoint across a `--zero-mean-maps` flip, a
      cdim change, an arch change, or a dataset change.
- [ ] At least 3 seeds for exploratory, 5 for any final claim.
- [ ] W&B on; output tree under the right `results/` subtree.

Only vary one factor at a time relative to the closest reference in
`PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md`. Every additional knob twisted in the same run
halves the diagnostic value of the result.
