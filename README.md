# cnn_sbi — SBI for weak-lensing cosmology

> Based on [Learn2Map](https://github.com/Justinezgh/Learn2Map) by [Justine Zeghal](https://github.com/Justinezgh) and [Benjamin Remy](https://github.com/b-remy).

Simulation-based inference (SBI) pipeline for weak gravitational lensing. Learns
low-dimensional summaries of tomographic convergence maps — via a CNN-VMIM compressor
or wavelet L1 / L1-VMIM statistics — and feeds them to a conditional RealNVP flow (or a
`jaxili` NPE backend) to infer the 6-parameter cosmology
`theta = [Omega_m, sigma_8, w0, h0, n_s, Omega_b]`.

Current scientific focus: BNT (Baryon Nuller Transform) contour inflation on 4-bin
tomographic CosmoGridV1 n-body maps.

## Start here

- [`CLAUDE.md`](CLAUDE.md) — repo conventions, environments, commands, working-tree rules.
- [`SBI_PIPELINE_BEST_PRACTICES.md`](SBI_PIPELINE_BEST_PRACTICES.md) — prescriptive guide for running new experiments.
- [`PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md`](PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md) — retrospective synthesis of all campaigns and evidence.
- [`CLAUDE_CODE_HANDOFF.md`](CLAUDE_CODE_HANDOFF.md) — runbook for the active `bnt-parity-techniques` branch.
- [`skills/sbi/SKILL.md`](skills/sbi/SKILL.md) — experiment protocol (workflow, claim acceptance, anti-clutter).

## Layout

- `scripts/sbi/` — **all live code**: per-run entrypoints (`npe_cnn_nbody_tomo.py`, `npe_l1norm_nbody_tomo.py`, `npe_l1vmim_nbody_tomo.py`, plus `_jaxili_` variants), campaign drivers (`run_*.py`), TFDS builders, BNT utilities.
- `notebooks/sbi/` — exploratory and publication notebooks.
- `learn2map2/datasets/` — older TFDS builders (mostly superseded).
- `learn2map/` — legacy Python virtualenv, gitignored. Do not edit.

## Installation

```bash
conda activate jaxili
pip install -e .
```

Pulls `jax-cosmo`, `numpyro`, `lenstools`, and `sbi_lens`. L1 pipelines additionally
require the local PyTorch extension `wl_stats_torch` at `/home/tersenov/software/wl_stats_torch`
(hard-coded in `scripts/sbi/npe_l1norm_nbody_tomo.py`). Data: CosmoGridV1 at
`/home/tersenov/CosmoGridV1/`.

## Run

Single CNN+NPE run, 4-bin tomography, 20° / 160 px, no BNT, canonical settings:

```bash
conda run -n jaxili python scripts/sbi/npe_cnn_nbody_tomo.py \
  --cuda-visible-devices 0 --map-kind nbody \
  --tfds-name NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48 \
  --field-size 20 --field-npix 160 \
  --nbins 4 --tomo-bin-indices 1,2,3,4 \
  --compressor-train-split "train[:70%]" --compressor-val-split "test" \
  --nde-train-split "train[70%:]"        --nde-val-split "test" \
  --require-disjoint-train-examples \
  --zero-mean-maps \
  --seed 42 --plot
```

Add `--apply-bnt` for the BNT path (requires full 4-bin tomography). See
[`SBI_PIPELINE_BEST_PRACTICES.md`](SBI_PIPELINE_BEST_PRACTICES.md) for canonical
compressor and flow hyperparameters.

## License

MIT — see [LICENSE](LICENSE).
