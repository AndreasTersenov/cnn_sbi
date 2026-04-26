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
