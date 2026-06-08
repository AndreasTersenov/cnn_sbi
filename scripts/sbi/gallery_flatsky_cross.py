#!/usr/bin/env python3
"""Corrected flat-sky cross-map gallery: show the two operators we'll actually test
(apodized circular convolution, pointwise product), plus the zero-pad variant to
show why we DON'T crop it. CPU-only."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

z = np.load("results/exploratory/cross_maps_campaign/full_sphere_cache_fiducial_10deg/"
            "nobnt/obs/cosmo_fiducial_perm0.npz")
auto = z["patches"][..., :4]; npix = int(z["field_npix"])
OUT = "results/exploratory/definitive_comparison_10deg/phase_c/analysis/figs_flatsky_validation"


def apod(n, roll=0.12):
    r = np.ones(n, np.float32); k = max(1, int(roll * n))
    c = 0.5 * (1 - np.cos(np.pi * np.arange(k) / k)); r[:k] = c; r[-k:] = c[::-1]
    return np.outer(r, r).astype(np.float32)


W = apod(npix)
i, j, k0 = 2, 3, 5
a, b = auto[k0, :, :, i], auto[k0, :, :, j]

conv_circ = np.fft.irfft2(np.fft.rfft2(a * W) * np.fft.rfft2(b * W), s=a.shape)
# zero-padded centered ('same') — shown only to illustrate the taper we avoid
m = 2 * npix; A = np.zeros((m, m), np.float32); B = np.zeros((m, m), np.float32)
A[:npix, :npix] = a * W; B[:npix, :npix] = b * W
full = np.fft.irfft2(np.fft.rfft2(A) * np.fft.rfft2(B), s=(m, m))
o = (npix - 1) // 2; conv_pad = full[o:o + npix, o:o + npix]
prod = a * b

panels = [(auto[k0, :, :, j], "auto κ4 (input)"),
          (conv_circ, "CONVOLUTION cross 3×4\n(apodized circular — use this)"),
          (conv_pad, "convolution, zero-pad+crop\n(corner taper — we DON'T use)"),
          (prod, "PRODUCT cross 3×4\n(pointwise κ3·κ4 — use this)")]
fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
for a_, (mp, t) in zip(ax, panels):
    v = np.percentile(np.abs(mp), 99)
    a_.imshow(mp, cmap="RdBu_r", vmin=-v, vmax=v); a_.set_title(t, fontsize=10); a_.axis("off")
fig.suptitle("Flat-sky cross-map operators (patch 5, fiducial) — corrected", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/gallery_corrected.png", dpi=130, bbox_inches="tight")
print(f"wrote {OUT}/gallery_corrected.png")
