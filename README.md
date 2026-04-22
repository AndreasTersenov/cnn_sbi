# Learn2Map

> **Based on [Learn2Map](https://github.com/Justinezgh/Learn2Map) by [Justine Zeghal](https://github.com/Justinezgh).**

## Overview

Learn2Map is a simulation-based inference (SBI) framework for weak gravitational lensing cosmology. It learns compact summary statistics from convergence maps — produced by either Gaussian or N-body simulations — using a convolutional neural network (CNN) compressor trained with a neural likelihood estimator (NLE) or neural posterior estimator (NPE). The learned summaries are then used to infer posterior distributions over cosmological parameters such as $\Omega_m$ and $\sigma_8$.

The project also explores extensions to tomographic analyses (multiple redshift bins), Baryon Transfer Network (BNT) transforms, and the effect of baryonic physics and intrinsic alignments (IA) on posterior recovery.

## Project Structure

```
.
├── learn2map/          # Core package: VAE and dataset utilities
├── learn2map2/
│   └── datasets/       # TensorFlow dataset builders (Gaussian, N-body, fiducial)
├── notebooks/
│   ├── sbi/            # SBI inference notebooks (NLE, NPE, tomographic, baryon+IA)
│   └── *.ipynb         # Data inspection and power spectrum checks
├── scripts/
│   └── sbi/            # Training scripts for compressor and inference pipelines
├── save_params/        # Saved model parameters
└── setup.py
```

## Key Features

- **CNN compressor training** for dimensionality reduction of weak lensing maps
- **Neural Likelihood Estimation (NLE)** and **Neural Posterior Estimation (NPE)** via [`sbi_lens`](https://github.com/DifferentiableUniverseInitiative/sbi_lens)
- Support for **Gaussian** and **N-body** simulated datasets
- **Tomographic analyses** with multiple redshift bins
- **BNT** transform support
- Studies of **baryonic effects** and **intrinsic alignments** on cosmological posteriors
- Differentiable forward modelling via [`jax-cosmo`](https://github.com/DifferentiableUniverseInitiative/jax_cosmo) and [`numpyro`](https://github.com/pyro-ppl/numpyro)

## Installation

A virtual environment is included under `learn2map/`. To install the package and its dependencies from source:

```bash
pip install -e .
```

**Dependencies:**

- [`jax-cosmo`](https://github.com/DifferentiableUniverseInitiative/jax_cosmo)
- [`numpyro`](https://github.com/pyro-ppl/numpyro)
- [`lenstools`](https://lenstools.readthedocs.io/)
- [`sbi_lens`](https://github.com/DifferentiableUniverseInitiative/sbi_lens)

## Usage

1. **Build datasets** using the scripts in `learn2map2/datasets/` or `scripts/sbi/`.
2. **Train the compressor** with `scripts/sbi/train_compressor.py` (or its tomographic variants).
3. **Run inference** using the notebooks in `notebooks/sbi/` — choose between NLE and NPE, and Gaussian or N-body simulations.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgements

This repository extends the original [Learn2Map](https://github.com/Justinezgh/Learn2Map) codebase developed by [Justine Zeghal](https://github.com/Justinezgh) and [Benjamin Remy](https://github.com/b-remy).

