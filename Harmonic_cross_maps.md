# Harmonic cross maps


# Implementation Guide: Tomographic Cross-Maps in Harmonic Space
**Reference:** Zürcher et al. (2022) - *Towards a full wCDM map-based analysis for weak lensing surveys*

This document provides the exact methodology and implementation instructions for computing synthetic "cross-maps" from two tomographic weak lensing convergence maps. This synthetic map extracts the cross-correlational, non-Gaussian information (such as map peaks or minima) between two different redshift bins.

## 1. Mathematical Concept

Given two real-space weak lensing convergence maps from two different tomographic bins, $\kappa^i(\theta, \phi)$ and $\kappa^j(\theta, \phi)$, the standard approach to cross-correlation is 2-point statistics (cross-power spectrum). To apply map-based topological statistics (like Peak Counts or wavelet $\ell_1$-norm), we must fuse the two maps into a single, real-valued 2D map.

The procedure is:
1.  **Spherical Harmonic Transform:** Convert both real-space maps to harmonic space coefficients.
    $$\kappa^i(\theta, \phi) \xrightarrow{\text{SHT}} a_{\ell m}^{(i)}$$
    $$\kappa^j(\theta, \phi) \xrightarrow{\text{SHT}} a_{\ell m}^{(j)}$$
2.  **Harmonic Space Product:** Multiply the complex coefficients to generate a new set of cross-coefficients. 
    $$a_{\ell m}^{(ij)} = a_{\ell m}^{(i)} \cdot a_{\ell m}^{(j)}$$
3.  **Inverse Transform with Reality Enforcement:** Transform the combined coefficients back into real space. To ensure the physical validity of the map (it must be a purely real field to count peaks), the reality condition $a_{\ell, -m} = (-1)^m a_{\ell, m}^*$ must be strictly enforced during the inverse transform.
    $$a_{\ell m}^{(ij)} \xrightarrow{\text{ISHT}} \kappa^{ij}(\theta, \phi)$$

## 2. Implementation Instructions for AI Agents / Developers

The standard and most efficient tool for this in Python is `healpy`. 

### `healpy` Specifics & The "Reality" Shortcut
When `healpy.map2alm` transforms a strictly real map, it optimizes memory by *only* returning the coefficients for $m \geq 0$. It assumes that $m < 0$ modes can be perfectly reconstructed using the reality condition. 
By multiplying these $m \geq 0$ 1D arrays element-wise and feeding them directly into `healpy.alm2map`, `healpy` will automatically force the resulting inverse-transformed map to be perfectly real, mathematically handling the symmetry for the negative $m$ modes under the hood.

### Implementation Steps

#### Prerequisites:
* `numpy`
* `healpy`

#### Step-by-Step Logic:
1.  **Load Maps:** Load the two HEALPix maps (arrays of pixels) for bin $i$ and bin $j$. Ensure both maps have the same `NSIDE`.
2.  **Determine LMAX:** Set `lmax` for the transform (typically `3 * NSIDE - 1` or based on a specific smoothing scale).
3.  **Forward Transform:** Use `healpy.map2alm(map, lmax=lmax)` on both maps to get `alms_i` and `alms_j`. These will be 1D complex numpy arrays containing modes for $m \geq 0$.
4.  **Coefficient Multiplication:** Perform a direct element-wise multiplication: `alms_cross = alms_i * alms_j`.
5.  **Inverse Transform:** Use `healpy.alm2map(alms_cross, nside=nside)` to project the multiplied coefficients back to a real-space HEALPix grid. Because `alms_cross` is structured exactly like a standard `healpy` coefficient array for a real map, `alm2map` natively enforces the reality condition and returns a real-valued 1D array of pixels.
6.  **(Optional but recommended) Smoothing:** Weak lensing maps are usually smoothed to reduce shape noise. Apply `healpy.smoothing` (Gaussian beam) either on the final cross-map or by multiplying the `alms_cross` by the beam window function before the inverse transform.

### Python Pseudo-Code

```python
import numpy as np
import healpy as hp

def compute_harmonic_cross_map(map_i, map_j, nside, lmax=None):
    \"\"\"
    Computes the synthetic cross-map between two tomographic bins.
    
    Parameters:
    - map_i (np.ndarray): HEALPix map for bin i (1D array).
    - map_j (np.ndarray): HEALPix map for bin j (1D array).
    - nside (int): The HEALPix NSIDE resolution.
    - lmax (int): Maximum multipole (defaults to 3 * nside - 1).
    
    Returns:
    - cross_map (np.ndarray): The purely real synthetic cross-map.
    \"\"\"
    if lmax is None:
        lmax = 3 * nside - 1
        
    # 1. Transform both maps to harmonic space (yields m >= 0 coefficients)
    alms_i = hp.map2alm(map_i, lmax=lmax)
    alms_j = hp.map2alm(map_j, lmax=lmax)
    
    # 2. Harmonic space product
    alms_cross = alms_i * alms_j
    
    # 3. Inverse transform back to real space
    # hp.alm2map inherently enforces the reality condition using the m>=0 array
    cross_map = hp.alm2map(alms_cross, nside=nside)
    
    return cross_map

```

---

## 3. Amplitude Imbalance Between Auto- and Cross-Channels and its Treatment

### 3.1 The problem

When the 4 auto-maps ($\kappa^{ii}$, one per tomographic bin) and 6 cross-maps ($\kappa^{ij}$, $i < j$) are used together as a 10-channel input to a CNN, there is a severe amplitude mismatch between the two types:

- **Auto-channels** $C_{ii}(\ell)$: self-correlations of the lensing field, amplitude $\sim 10^{-2}$ (in map-pixel units after per-patch demeaning).
- **Cross-channels** $C_{ij}(\ell)$, $i \neq j$: correlations between different redshift shells, amplitude $\sim 10^{-7}$–$10^{-5}$ — empirically **~4 orders of magnitude smaller** than auto-channels for our 20-deg / 160-px patch setup.

The physical reason is that $C_{ij}(\ell)$ reflects the overlap of the lensing kernels of bins $i$ and $j$, which is smaller than the self-overlap $C_{ii}(\ell)$ for bins with non-overlapping redshift support.

Without any normalization, the gradient signal reaching the CNN's first convolutional layer from the cross-channels is $\sim 10^4 \times$ weaker than from the auto-channels. In practice the network learns to ignore the cross-channels entirely, and the cross-map information is never extracted — even though it is genuinely present and cosmologically informative (see §3.3).

### 3.2 The fix: dataset-level per-channel RMS normalization

Before the maps enter the CNN, each channel $c$ is divided by its **dataset-level RMS**:

$$\sigma_c = \sqrt{\frac{1}{N_{\text{train}} \cdot H \cdot W} \sum_{\text{train examples}} \sum_{p=1}^{H \times W} x_{c,p}^2}$$

where the sum pools over all pixels of all training examples. Because the patches are already zero-mean (enforced by cache construction), RMS equals std here.

The key distinction is that $\sigma_c$ is a **fixed, dataset-level constant** — not a per-example normalizer. After division, every channel has RMS $\approx 1$ across the training set, while the cosmology-dependent amplitude variation of individual examples is fully preserved.

Two alternatives that were considered and rejected:

- **Per-example, per-channel std normalization** ($x_{c} \to x_{c} / \text{std}(x_c)$ per example): would erase the cosmology-dependent map amplitude, which is a cosmological observable (higher $\sigma_8$ produces maps with larger variance). Rejected.
- **Normalizing only cross-channels**: would equalize cross vs. auto but leave the 4 auto-channels slightly unequal among themselves (~40% range). Dataset-level normalization of all 10 channels is the cleaner and more principled choice.

### 3.3 No cosmological information is lost: information-theoretic argument

The transformation $x_c \mapsto x_c / \sigma_c$ (with $\sigma_c$ a fixed known constant) is an **invertible linear map**. By the data processing inequality, the mutual information $I(\mathbf{X}_{\text{norm}}; \theta) = I(\mathbf{X}; \theta)$ — normalization cannot reduce the Fisher information.

More concretely:

1. **Absolute scale is not a cosmological observable.** The value of $\sigma_c$ is set by the cache construction conventions (patch size, pixelisation, smoothing scale). Dividing by it is equivalent to choosing different units per channel. No physics is encoded in the prior-averaged amplitude of a single channel in isolation.

2. **Cosmology-dependent amplitude variation is preserved.** For cosmology $\theta_i$, the auto-channel variance scales roughly as $\sigma_8^2 \Omega_m^{1.5}$. After normalization, example $i$ still has variance $\sigma_c^2(\theta_i) / \sigma_c^2$, where the numerator depends on $\theta_i$. The CNN can still learn to use per-channel amplitude as a feature.

3. **Cross-channel amplitude ratios $C_{ij} / C_{ii}$ are preserved.** These ratios are cosmology-dependent (they encode the geometric overlap of lensing kernels). Normalizing channel $c_1$ by $\sigma_{c_1}$ and channel $c_2$ by $\sigma_{c_2}$ rescales the ratio by a known constant $\sigma_{c_2}/\sigma_{c_1}$, which is absorbed into the first convolutional layer's weights. No ratio information is lost.

In summary: the only thing removed by dataset-level per-channel RMS normalization is the prior-averaged amplitude of each channel — a unit convention with no cosmological content. All cosmologically informative variance, covariance, and amplitude ratios between channels are unchanged.
