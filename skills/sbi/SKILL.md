---
name: sbi
description: Run and analyze simulation-based inference experiments for cosmological contours in this repository with strict reproducibility, minimal clutter, and clear setup isolation. Use this skill whenever a user asks to run, compare, debug, or summarize CNN/L1/L1-VMIM/jaxili SBI pipelines, including tomography-vs-single-bin studies, estimator swaps, contour-quality investigations, or baryon-bias checks.
---

# SBI experiment operator

Use this skill to execute SBI studies in `cnn_sbi` in a way that is reproducible, efficient, and scientifically publishable.

## Pipeline mental model

Always reason in three distinct layers and keep them separated:

1. **Summary statistic extraction**
   - CNN VMIM compressor (`npe_cnn*_nbody_tomo.py`)
   - Wavelet L1 datavector (`npe_l1norm*_nbody_tomo.py`)
2. **Optional summary compression/preprocessing**
   - log transforms, z-score, clipping
   - PCA (allowed only when explicitly requested)
   - learned compression (VMIM) when needed
3. **Density estimator** `p(theta | summary)`
   - in-repo conditional RealNVP
   - jaxili NPE

Never blame or credit "the pipeline" globally without identifying which layer changed.

## Hard invariants

- Keep parameter order fixed: `[Omega_m, sigma_8, w0, h0, n_s, Omega_b]`.
- Keep `h0 = H0 / 100` convention consistent everywhere.
- For cross-method comparisons, lock scenario values: TFDS config, geometry, map kind, tomo bins, noise model, seeds, and observed map source.
- Reuse cache only when metadata is fully compatible with current settings.
- Do not mix checkpoints with incompatible preprocessing/PCA/feature-mask artifacts.
- If the objective is observational mismatch (e.g., baryonified observation with no-bary training), keep model training fixed and use `--no-train`.

## L1 defaults for this project

Unless the user explicitly asks otherwise:

- Use L1 **without PCA**.
- If compression is needed, prefer **L1-VMIM compression** over PCA.
- Use L1 extraction defaults:
  - `--n-scales 5`
  - `--l1-min-snr -13`
  - `--l1-max-snr 13`
  - `--l1-nbins 40`

Coarse scale is included in `n_scales`.

## Compressor reuse safety rules

Be conservative. Reuse a trained compressor only if all of the following match:

1. map family (`nbody`, `nbody_with_baryon_ia`, gaussian/BNT context),
2. tomography definition (`tomo-bin-indices`, `nbins`),
3. geometry (`field-size`, `field-npix`, projection assumptions),
4. noise setup (`sigma-e`, galaxy density),
5. training objective and architecture (VMIM settings, compressor dim),
6. dataset config (`tfds-name`) and augmentation policy.

If any item differs, retrain the compressor and mark the old one as incompatible for this run.

## Mandatory logging and outputs

- Enable and keep **W&B logging** for all non-dry runs.
- Save these outputs per run:
  1. posterior samples (`.npy`),
  2. posterior metadata (`.meta.json`),
  3. FoM/metrics summary (JSON or CSV, including at least FoM),
  4. corner plot in **PDF** format,
  5. run manifest/config fingerprint.

Do not conclude from runs that do not produce this minimum artifact set.

## Anti-clutter policy

- Reuse existing scripts/orchestrators before adding new ones.
- Save outputs under `scripts/sbi/results/`:
  - `final/` for conclusion-grade outputs
  - `exploratory/` for iterative scans
  - `dryruns/` for smoke tests
  - `diagnostics/` for debugging artifacts
- Keep only artifacts needed for reproducibility and scientific conclusions.
- Avoid duplicate quicklooks, temporary caches, and redundant logs once key outputs are preserved.

## Mandatory workflow

### 1) Lock the objective before running

Define one explicit objective with:
- target statistic(s),
- comparison axis (e.g., tomo vs single-bin, estimator swap, baryon bias),
- quantitative decision metric(s).

### 2) Prevent duplicate work

Before launching any run, check:
- `scripts/sbi/results/INDEX.txt`
- `scripts/sbi/results/final`
- `scripts/sbi/results/exploratory`

If an equivalent run already answers the question, summarize it instead of rerunning.

### 3) Freeze a config fingerprint

Record complete run settings:
- script path(s),
- dataset/config (`tfds-name`), geometry, map kind,
- `tomo-bin-indices`, `nbins`,
- summary settings (L1/CNN extraction, preprocessing, compression/PCA),
- estimator settings (flow/jaxili hyperparameters and checkpoint names),
- seeds and posterior sample count,
- observed map path,
- training mode (`train` vs `--no-train`).

### 4) Run with isolated changes

For diagnosis/optimization, vary one factor at a time.  
If multiple factors change together, label the result exploratory and non-isolated.

### 5) Validate trustworthiness before conclusions

At minimum verify:
1. cache metadata compatibility,
2. checkpoint/preprocessing/mask compatibility,
3. finite posterior samples with expected shape,
4. training stability (no NaN collapse, no silent fallback),
5. apples-to-apples comparison frame across methods.

### 6) Report in publishable structure

Use this structure exactly:

```markdown
## Objective
## Configuration fingerprint
## Quantitative outcomes (including FoM)
## Robustness and failure checks
## Scientific conclusion
## Minimal next action
```

Clearly separate established findings from tentative findings.

## Claim acceptance criteria

Treat a claim as established only if all are true:

1. comparison is apples-to-apples,
2. result is reproducible from saved artifacts,
3. metrics are stable across relevant seeds/realizations,
4. major failure modes are ruled out (cache mismatch, preprocess mismatch, underconvergence, compressor mismatch).

## Frequent failure modes to guard against

- Scenario mismatch across methods.
- Stale cache reuse after config changes.
- Old checkpoint with new preprocessing artifacts.
- Reusing a compressor across incompatible tomo/BNT contexts.
- Confusing estimator changes with summary-extractor changes.
- Promoting underconverged exploratory runs to final conclusions.
