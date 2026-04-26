# Implementation Guide: Flat-Sky Tomographic Cross-Maps (2D Patches)
**Adapted from:** Zürcher et al. (2022) methodology for the flat-sky approximation.

This document provides the methodology and implementation instructions for computing synthetic "cross-maps" from two 2D tomographic weak lensing convergence maps (flat sky approximation). This synthetic map extracts the cross-correlational, non-Gaussian information between two different redshift bins using Fast Fourier Transforms (FFTs).

## 1. Mathematical Concept

When working with small patches of the sky (typically < 10x10 degrees), the curvature of the universe is negligible. Instead of Spherical Harmonics, we use 2D Fast Fourier Transforms (FFTs).

Let $\kappa^i(\mathbf{x})$ and $\kappa^j(\mathbf{x})$ be your two 2D real-space convergence maps for tomographic bins $i$ and $j$, where $\mathbf{x} = (x, y)$ represents the pixel coordinates.

The procedure is:
1.  **Forward 2D FFT:** Transform both maps into 2D Fourier space.
    $$\tilde{\kappa}^i(\mathbf{k}) = \mathcal{F}\{ \kappa^i(\mathbf{x}) \}$$
    $$\tilde{\kappa}^j(\mathbf{k}) = \mathcal{F}\{ \kappa^j(\mathbf{x}) \}$$
2.  **Fourier Space Product:** Multiply the complex Fourier coefficients element-by-element.
    $$\tilde{\kappa}^{ij}(\mathbf{k}) = \tilde{\kappa}^i(\mathbf{k}) \cdot \tilde{\kappa}^j(\mathbf{k})$$
3.  **Inverse 2D FFT:** Transform the multiplied coefficients back to real space.
    $$\kappa^{ij}(\mathbf{x}) = \mathcal{F}^{-1}\{ \tilde{\kappa}^{ij}(\mathbf{k}) \}$$

*By the Convolution Theorem, multiplying in Fourier space is mathematically identical to a real-space convolution ($\kappa^i * \kappa^j$).*

## 2. The Edge Effect Pitfall (Circular Convolution)

FFTs inherently assume your 2D grid is **periodic** (the top edge wraps to the bottom edge, left to right). Multiplying two map FFTs together performs a *circular convolution*. Structures on the right edge of your patch will "bleed" over and create fake peaks on the left edge.

**Solution:** Before transforming your maps, you must apply an **apodization mask** (tapering the edges smoothly to zero) or use **zero-padding** (embedding your map in a larger grid of zeros) and crop the result afterwards.

## 3. Implementation Instructions for AI Agents / Developers

The standard tool for this in Python is `numpy.fft` or `scipy.fft`. Because the input convergence maps are purely real numbers, use `rfft2` (Real 2D FFT) instead of `fft2`. This automatically handles Hermitian symmetry ($\tilde{\kappa}(-\mathbf{k}) = \tilde{\kappa}^*(\mathbf{k})$) and ensures the inverse transform (`irfft2`) returns a strictly real 2D array.

### Python Code

```python
import numpy as np

def compute_flat_cross_map(map_i, map_j):
    \"\"\"
    Computes the synthetic cross-map between two 2D flat patches.
    
    Parameters:
    - map_i (np.ndarray): 2D real array of the convergence patch for bin i.
    - map_j (np.ndarray): 2D real array of the convergence patch for bin j.
    
    Returns:
    - cross_map (np.ndarray): The purely real 2D synthetic cross-map.
    \"\"\"
    # 1. Verify dimensions
    if map_i.shape != map_j.shape:
        raise ValueError("Input maps must have the exact same dimensions.")
    
    # Optional but critical: Apodize maps here if not already done.
    # Example: map_i = map_i * apodization_window
    # Example: map_j = map_j * apodization_window
    
    # 2. Forward Transform 
    # Use rfft2 because the input maps are strictly real.
    fft_i = np.fft.rfft2(map_i)
    fft_j = np.fft.rfft2(map_j)
    
    # 3. Fourier Space Product
    fft_cross = fft_i * fft_j
    
    # 4. Inverse Transform back to real space
    # irfft2 requires the original map shape to resolve parity ambiguities 
    # and guarantees a strictly real float array as output.
    cross_map = np.fft.irfft2(fft_cross, s=map_i.shape)
    
    return cross_map