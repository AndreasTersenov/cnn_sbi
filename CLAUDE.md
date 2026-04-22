# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Simulation-based inference (SBI) for weak gravitational lensing cosmology. Forked/extended from Justine Zeghal's [Learn2Map](https://github.com/Justinezgh/Learn2Map). It learns low-dimensional summaries from convergence maps (CNN-VMIM compressor or wavelet L1/L1-VMIM statistics) and feeds them to a conditional RealNVP flow (or `jaxili` NPE) to infer `theta = [Omega_m, sigma_8, w0, h0, n_s, Omega_b]`.

The current scientific focus is **BNT contour inflation** in tomographic setups. Read `PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md` first — it's the authoritative synthesis of campaign outcomes and is more up-to-date than `README.md`. `CLAUDE_CODE_HANDOFF.md` is the detailed runbook for the latest (`bnt-parity-techniques`) branch.

## Environment and external dependencies

- **Conda env:** `jaxili` (always prefix commands with `conda run -n jaxili python ...`). The virtualenv directory `learn2map/` in the repo is legacy and gitignored — do not install into it.
- **Local PyTorch extension:** `wl_stats_torch` at `/home/tersenov/software/wl_stats_torch` — hard-coded in the L1 pipeline (`_WL_STATS_PATH`). L1 scripts compute summaries in PyTorch on GPU while the flow runs in JAX; both devices are used in the same process.
- **Datasets:** CosmoGridV1 at `/home/tersenov/CosmoGridV1/` (`CosmoGridV1_metainfo.h5`, `stage3_forecast/fiducial/...`). Hard-coded as defaults in scripts.
- **Pip install:** `pip install -e .` registers `learn2map` and pulls `jax-cosmo`, `numpyro`, `lenstools`, and `sbi_lens` (from git).
- **TFDS builders** live in `scripts/sbi/tf_dataset_nbody_tomo*.py` and register via `import tf_dataset_nbody_tomo as _tomo_builder` — TFDS names used by the pipeline are `NbodyCosmogridDatasetTomo/grid` (10 deg / 80 px) and `NbodyCosmogridDatasetTomo/grid_20deg_160px[_nonoverlap48]` (20 deg / 160 px).

## Code layout (what is actually live)

- `scripts/sbi/` — **all live code** for the current scientific work. This is the only tree to edit.
- `learn2map2/datasets/` — older TFDS dataset builders; largely superseded by `scripts/sbi/tf_dataset_*` but still imported in places.
- `learn2map/` — a Python virtualenv, **not source code**. Never edit.
- `notebooks/sbi/` — exploratory/publication notebooks; mostly read-only for inference runs.
- `skills/sbi/SKILL.md` — the project's own SBI operating protocol (workflow, logging, anti-clutter policy). Treat it as authoritative procedure.
- `.worktrees/` — pre-existing git worktrees for parallel experiments; do not delete.

## Pipeline architecture (mental model)

Always reason in three layers and keep them separated — do not blame "the pipeline" without naming a layer:

1. **Summary extraction**
   - `npe_cnn_nbody_tomo.py` / `npe_cnn_jaxili_nbody_tomo.py` — CNN-VMIM compressor; arch family selectable via `--compressor-arch {plain,resnet_small,resnet18,resnet34,resnet50}`.
   - `npe_l1norm_nbody_tomo.py` / `npe_l1norm_jaxili_nbody_tomo.py` — wavelet L1 datavector (PyTorch `WLStatistics`).
   - `npe_l1vmim_nbody_tomo.py` / `npe_l1vmim_jaxili_nbody_tomo.py` — L1 feeds a VMIM MLP compressor.
2. **Preprocessing:** log1p / z-score / clipping; PCA only when explicitly asked (L1-VMIM is preferred for compression).
3. **Density estimator `p(theta | summary)`:** in-repo conditional RealNVP (via `sbi_lens.normflow`) or `jaxili` NPE (scripts with `_jaxili_` in the name).

Shared pipeline invariants:
- Parameter order is fixed `[Omega_m, sigma_8, w0, h0, n_s, Omega_b]`, with `h0 = H0/100` applied in preprocessing (`theta[3] /= 100`).
- In the CNN BNT path, **shape noise is injected before BNT** (`apply_bnt_*` in `scripts/sbi/bnt_utils.py`). `--apply-bnt` requires full tomography (`--nbins 4 --tomo-bin-indices 1,2,3,4`); `validate_bnt_configuration` enforces this.
- Paired BNT/no-BNT training (`--compressor-paired-bnt-nobnt-consistency`) returns dict features `{maps_nobnt, maps_bnt}`; `compress_dataset(..., paired_map_view=...)` must select one view, otherwise it crashes with `KeyError: 'maps'`. This is a known fix point; preserve the `paired_map_view` plumbing when editing.

## Campaign orchestration

Multi-seed / multi-GPU experiments are driven by `scripts/sbi/run_*.py` scripts, which shell out to the per-run entrypoints above. Key ones:

- `run_cnn_bnt_losslessness_campaign.py`, `run_cnn_noise_curriculum_campaign.py` — BNT recovery campaigns.
- `run_cnn_l1_systematic_sweep.py`, `run_cnn_tomo4_opt_sweep.py`, `run_l1_jaxili_tomo4_opt_sweep.py`, `run_l1vmim_tomo4_opt_sweep.py` — systematic sweeps.
- `run_bnt_tomo4_study.py`, `run_baryon_bias_tomo4_study.py`, `run_nobnt_tomo_bins_crosscorr_study.py`, `run_optimal_nobnt_crosscorr_benchmark.py` — focused studies.

Orchestrators accept `--gpus 0,1,2,3` and an optional `--xla-mem-fraction-by-gpu 0:0.75,1:0.30,...` per-GPU memory cap map. Results go under `scripts/sbi/results/{final,exploratory,dryruns,diagnostics}/` — follow that taxonomy (see `skills/sbi/SKILL.md`).

For example run commands (single-run L1/CNN with typical flags, sweep invocations), see `scripts/sbi/l1_jaxili_run_commands.txt` and `scripts/sbi/l1_vmim_run_commands.txt`.

## Common commands

```bash
# Compile check (fast sanity pass before launching GPU work):
conda run -n jaxili python -m py_compile \
  scripts/sbi/npe_cnn_nbody_tomo.py \
  scripts/sbi/run_cnn_noise_curriculum_campaign.py

# Typical CNN+NPE single run (4 tomo bins, 20 deg / 160 px, no BNT):
conda run -n jaxili python scripts/sbi/npe_cnn_nbody_tomo.py \
  --cuda-visible-devices 0 --map-kind nbody \
  --tfds-name NbodyCosmogridDatasetTomo/grid_20deg_160px \
  --field-size 20 --field-npix 160 \
  --nbins 4 --tomo-bin-indices 1,2,3,4 \
  --seed 42 --plot

# Same, but apply BNT after shape noise:
... --apply-bnt   # only valid with full tomo4

# Reuse a trained flow checkpoint, sample only:
... --no-train --npe-samples 100000
```

There is no test suite and no lint configuration — integrity is verified by `py_compile`, W&B-logged training runs, and the per-campaign comparison reports.

## Working-tree and git discipline

The tree is chronically dirty (notebooks with outputs, caches, `__pycache__`, campaign results under `scripts/sbi/results/`). Rules from `CLAUDE_CODE_HANDOFF.md`:

1. **Never `git add .` or `git add -A`.** Stage files explicitly by path.
2. Do not delete or rewrite pre-existing dirty files unless the user explicitly asks.
3. Do not run destructive cleanup (`git reset --hard`, mass deletes, `git clean`).
4. Do not commit generated artifacts (results, caches, `.pkl` checkpoints, `*.pyc`) unless asked.
5. Main development branch is currently `bnt-parity-techniques`; the "main" PR target is `main`.

## Project protocol (from `skills/sbi/SKILL.md`)

When running SBI experiments, follow the workflow in `skills/sbi/SKILL.md`: lock an objective with a decision metric, freeze a config fingerprint, vary one factor at a time, validate cache/checkpoint/preprocess compatibility before drawing conclusions, and report results in the mandated `Objective / Configuration fingerprint / Quantitative outcomes / Robustness / Scientific conclusion / Minimal next action` structure.

Claim acceptance requires apples-to-apples comparison, reproducibility from saved artifacts, stability across seeds, and having ruled out cache/preprocess/compressor mismatches. Keep per-run artifacts minimal: posterior `.npy`, `.meta.json`, metrics CSV/JSON (including FoM), corner plot PDF, and a manifest. W&B logging is expected on non-dry runs.

## Other reference docs in the tree

- `SBI_L1_CNN_PIPELINE_DETAILED.md` — step-by-step audit trail of all four pipelines.
- `L1_CONTOUR_INVESTIGATION_LOG.md`, `L1_FIXES_VALIDATION_REPORT.md`, `L1_VMIM_FINAL_CONCLUSIONS.md` — L1 diagnosis and fix history.
- `BNT_NO_BNT_CONTOUR_INFLATION_NOTE.md` — conceptual note on BNT inflation mechanisms.
- `scripts/sbi/results/final/paper_sbi_consolidation/FINAL_SCIENTIFIC_REPORT.md` — the top-of-campaign synthesis cited by the knowledge base.
