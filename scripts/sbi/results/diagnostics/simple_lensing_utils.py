"""Minimal weak-lensing map utilities (NumPy + Healpy only).

This module intentionally avoids JAX/TensorFlow and keeps only:
1) spherical-map -> patch projection,
2) 2D mass map (kappa) -> shear maps (gamma1, gamma2),
3) shear noise + masking,
4) simple HDF5 simulation-map readers,
5) hp.mollview plotting helpers.
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Sequence, Tuple

import h5py
import healpy as hp
import numpy as np


def read_h5_dataset(
    h5_path: str,
    dataset_path: str,
    dtype: Optional[np.dtype] = np.float32,
) -> np.ndarray:
    """Read one dataset from an HDF5 file.

    Example dataset paths in this repository:
      - "kg/stage3_lensing4"
      - "ia/stage3_lensing4"
    """
    path = dataset_path.strip().lstrip("/")
    if not path:
        raise ValueError("dataset_path must be a non-empty HDF5 dataset path.")

    with h5py.File(h5_path, "r") as f:
        if path not in f:
            raise KeyError(f"Dataset '{path}' not found in '{h5_path}'.")
        obj = f[path]
        if not isinstance(obj, h5py.Dataset):
            raise TypeError(
                f"'{path}' in '{h5_path}' is not a dataset (type={type(obj)})."
            )
        arr = np.asarray(obj)

    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def read_simulation_lensing_map(
    h5_path: str,
    lensing_bin: int = 4,
    probe_group: str = "kg",
    dtype: Optional[np.dtype] = np.float32,
) -> np.ndarray:
    """Read one spherical lensing map from simulation HDF5.

    Default path pattern:
        "{probe_group}/stage3_lensing{lensing_bin}"
    """
    if lensing_bin < 1:
        raise ValueError(f"lensing_bin must be >= 1, got {lensing_bin}.")
    dataset_path = f"{probe_group}/stage3_lensing{lensing_bin}"
    return read_h5_dataset(h5_path=h5_path, dataset_path=dataset_path, dtype=dtype)


def read_simulation_tomographic_maps(
    h5_path: str,
    lensing_bins: Sequence[int] = (1, 2, 3, 4),
    probe_group: str = "kg",
    dtype: Optional[np.dtype] = np.float32,
) -> np.ndarray:
    """Read multiple spherical lensing bins and stack to shape (nbins, npix)."""
    bins = tuple(int(b) for b in lensing_bins)
    if not bins:
        raise ValueError("lensing_bins must contain at least one bin.")

    maps = [
        read_simulation_lensing_map(
            h5_path=h5_path,
            lensing_bin=b,
            probe_group=probe_group,
            dtype=dtype,
        )
        for b in bins
    ]

    first_size = maps[0].size
    for i, m in enumerate(maps):
        if m.ndim != 1:
            raise ValueError(
                f"Bin {bins[i]} map must be 1D HEALPix array, got shape {m.shape}."
            )
        if m.size != first_size:
            raise ValueError(
                f"All tomographic maps must have same size; "
                f"bin {bins[0]} has {first_size}, bin {bins[i]} has {m.size}."
            )
    return np.stack(maps, axis=0)


def plot_mollview(
    spherical_map: np.ndarray,
    title: str = "",
    unit: str = "",
    nest: bool = False,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    cmap: str = "viridis",
    output_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Plot one spherical map with hp.mollview.

    If output_path is provided, saves the figure.
    """
    import matplotlib.pyplot as plt

    m = np.asarray(spherical_map)
    if m.ndim != 1:
        raise ValueError(
            f"spherical_map for mollview must be 1D HEALPix array, got {m.shape}."
        )

    hp.mollview(
        m,
        title=title,
        unit=unit,
        nest=nest,
        min=min_val,
        max=max_val,
        cmap=cmap,
    )
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_h5_dataset_mollview(
    h5_path: str,
    dataset_path: str,
    title: str = "",
    unit: str = "",
    nest: bool = False,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    cmap: str = "viridis",
    output_path: Optional[str] = None,
    show: bool = False,
    dtype: Optional[np.dtype] = np.float32,
) -> np.ndarray:
    """Read one HDF5 dataset and plot it with hp.mollview.

    Returns the loaded map array.
    """
    m = read_h5_dataset(h5_path=h5_path, dataset_path=dataset_path, dtype=dtype)
    plot_mollview(
        spherical_map=m,
        title=title or dataset_path,
        unit=unit,
        nest=nest,
        min_val=min_val,
        max_val=max_val,
        cmap=cmap,
        output_path=output_path,
        show=show,
    )
    return m


def make_patch_from_spherical_map(
    spherical_map: np.ndarray,
    nside: int,
    field_size_deg: float,
    field_npix: int,
    lon_deg: float = 0.0,
    lat_deg: float = 0.0,
) -> np.ndarray:
    """Project one HEALPix spherical map to a square gnomonic patch.

    Args:
        spherical_map: 1D HEALPix map (length = 12 * nside^2).
        nside: HEALPix NSIDE.
        field_size_deg: Patch width in degrees.
        field_npix: Patch width in pixels.
        lon_deg: Patch center longitude in degrees.
        lat_deg: Patch center latitude in degrees.

    Returns:
        2D patch of shape (field_npix, field_npix), dtype float32.
    """
    spherical_map = np.asarray(spherical_map)
    if spherical_map.ndim != 1:
        raise ValueError(
            f"spherical_map must be 1D HEALPix array, got shape {spherical_map.shape}."
        )

    expected_npix = hp.nside2npix(nside)
    if spherical_map.size != expected_npix:
        raise ValueError(
            f"Map size {spherical_map.size} does not match nside={nside} "
            f"(expected {expected_npix})."
        )

    if field_npix <= 0:
        raise ValueError(f"field_npix must be positive, got {field_npix}.")
    if field_size_deg <= 0:
        raise ValueError(f"field_size_deg must be positive, got {field_size_deg}.")

    reso_arcmin = field_size_deg * 60.0 / float(field_npix)
    proj = hp.projector.GnomonicProj(
        rot=[lon_deg, lat_deg, 0.0],
        xsize=field_npix,
        ysize=field_npix,
        reso=reso_arcmin,
    )
    patch = proj.projmap(spherical_map, vec2pix_func=partial(hp.vec2pix, nside))
    return np.asarray(patch, dtype=np.float32)


def make_tomographic_patch(
    spherical_maps: np.ndarray,
    nside: int,
    field_size_deg: float,
    field_npix: int,
    lon_deg: float = 0.0,
    lat_deg: float = 0.0,
) -> np.ndarray:
    """Project multiple HEALPix maps and stack them as tomographic channels.

    Args:
        spherical_maps: Array-like with shape (nbins, npix).
        nside: HEALPix NSIDE.
        field_size_deg: Patch width in degrees.
        field_npix: Patch width in pixels.
        lon_deg: Patch center longitude in degrees.
        lat_deg: Patch center latitude in degrees.

    Returns:
        3D patch array with shape (field_npix, field_npix, nbins), dtype float32.
    """
    maps = np.asarray(spherical_maps)
    if maps.ndim != 2:
        raise ValueError(
            f"spherical_maps must have shape (nbins, npix), got {maps.shape}."
        )

    projected = [
        make_patch_from_spherical_map(
            spherical_map=maps[i],
            nside=nside,
            field_size_deg=field_size_deg,
            field_npix=field_npix,
            lon_deg=lon_deg,
            lat_deg=lat_deg,
        )
        for i in range(maps.shape[0])
    ]
    return np.stack(projected, axis=-1).astype(np.float32)


def kappa_to_shear(kappa: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a 2D convergence map kappa into shear maps (gamma1, gamma2).

    Uses the standard flat-sky Fourier relation:
        gamma_hat = ((kx^2 - ky^2) + i*2*kx*ky) / (kx^2 + ky^2) * kappa_hat
    with zero mode set to 0.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    if kappa.ndim != 2:
        raise ValueError(f"kappa must be 2D, got shape {kappa.shape}.")

    ny, nx = kappa.shape
    kappa_hat = np.fft.fft2(kappa)

    kx = 2.0 * np.pi * np.fft.fftfreq(nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k2 = kx_grid**2 + ky_grid**2

    with np.errstate(divide="ignore", invalid="ignore"):
        pref1 = (kx_grid**2 - ky_grid**2) / k2
        pref2 = (2.0 * kx_grid * ky_grid) / k2

    pref1[k2 == 0.0] = 0.0
    pref2[k2 == 0.0] = 0.0

    gamma1 = np.fft.ifft2(pref1 * kappa_hat).real
    gamma2 = np.fft.ifft2(pref2 * kappa_hat).real
    return gamma1.astype(np.float32), gamma2.astype(np.float32)


def shear_noise_sigma(
    sigma_e: float,
    galaxy_density_arcmin2: float,
    field_size_deg: float,
    field_npix: int,
) -> float:
    """Per-pixel Gaussian shear noise standard deviation."""
    if sigma_e <= 0:
        raise ValueError(f"sigma_e must be > 0, got {sigma_e}.")
    if galaxy_density_arcmin2 <= 0:
        raise ValueError(
            f"galaxy_density_arcmin2 must be > 0, got {galaxy_density_arcmin2}."
        )
    if field_size_deg <= 0 or field_npix <= 0:
        raise ValueError(
            f"field_size_deg and field_npix must be positive, got "
            f"{field_size_deg}, {field_npix}."
        )

    pixel_size_arcmin = field_size_deg * 60.0 / float(field_npix)
    return float(
        sigma_e / np.sqrt(galaxy_density_arcmin2 * pixel_size_arcmin * pixel_size_arcmin)
    )


def add_noise_and_mask_to_shear(
    gamma1: np.ndarray,
    gamma2: np.ndarray,
    sigma_e: float,
    galaxy_density_arcmin2: float,
    field_size_deg: float,
    mask: Optional[np.ndarray] = None,
    random_mask_fraction: float = 0.0,
    masked_value: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Add Gaussian noise and apply a mask to shear maps.

    Args:
        gamma1: First shear component (2D).
        gamma2: Second shear component (2D), same shape as gamma1.
        sigma_e: Shape noise parameter.
        galaxy_density_arcmin2: Galaxy density in arcmin^-2.
        field_size_deg: Patch width in degrees (for pixel noise conversion).
        mask: Optional boolean mask (True=keep pixel, False=masked).
        random_mask_fraction: If mask is None, randomly masks this fraction.
        masked_value: Value assigned to masked pixels.
        rng: Optional NumPy random generator.

    Returns:
        gamma1_noisy_masked, gamma2_noisy_masked, mask_used, noise_std
    """
    g1 = np.asarray(gamma1, dtype=np.float64)
    g2 = np.asarray(gamma2, dtype=np.float64)
    if g1.shape != g2.shape:
        raise ValueError(f"gamma1 and gamma2 shape mismatch: {g1.shape} vs {g2.shape}.")
    if g1.ndim != 2:
        raise ValueError(f"gamma maps must be 2D, got shape {g1.shape}.")
    if g1.shape[0] != g1.shape[1]:
        raise ValueError(
            "gamma maps must be square for this helper "
            f"(got shape {g1.shape})."
        )

    if rng is None:
        rng = np.random.default_rng()

    if mask is None:
        if random_mask_fraction < 0.0 or random_mask_fraction >= 1.0:
            raise ValueError(
                "random_mask_fraction must satisfy 0 <= fraction < 1."
            )
        if random_mask_fraction > 0.0:
            mask_used = rng.random(g1.shape) >= random_mask_fraction
        else:
            mask_used = np.ones(g1.shape, dtype=bool)
    else:
        mask_used = np.asarray(mask, dtype=bool)
        if mask_used.shape != g1.shape:
            raise ValueError(
                f"mask shape {mask_used.shape} must match shear shape {g1.shape}."
            )

    noise_std = shear_noise_sigma(
        sigma_e=sigma_e,
        galaxy_density_arcmin2=galaxy_density_arcmin2,
        field_size_deg=field_size_deg,
        field_npix=g1.shape[0],
    )
    n1 = rng.normal(loc=0.0, scale=noise_std, size=g1.shape)
    n2 = rng.normal(loc=0.0, scale=noise_std, size=g2.shape)

    g1_noisy = g1 + n1
    g2_noisy = g2 + n2
    g1_out = np.where(mask_used, g1_noisy, masked_value).astype(np.float32)
    g2_out = np.where(mask_used, g2_noisy, masked_value).astype(np.float32)
    return g1_out, g2_out, mask_used, noise_std
