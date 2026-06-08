#!/usr/bin/env python3
"""Cheap construction-validation for a PROPER flat-sky cross-map (no training).

Checks, on cached fiducial auto patches (10deg/80px):
  V1 wrap     : zero-padded LINEAR conv vs unpadded CIRCULAR conv differ only at edges.
  V2 signal   : the construction is sensitive to REAL cross-correlation
                (true pair vs bin-i × independent-patch bin-j); product-mean separates them,
                convolution-variance does not.
  V3 noise    : cross-map noise (n_i conv n_j) is colored and != white auto pixel-σ.
  gallery     : auto / unpadded-cross / padded-cross / pointwise-product for one patch.
CPU-only. Emits numbers + figs_flatsky_validation/*.png.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE = ("results/exploratory/cross_maps_campaign/full_sphere_cache_fiducial_10deg/"
         "nobnt/obs/cosmo_fiducial_perm0.npz")
OUT = "results/exploratory/definitive_comparison_10deg/phase_c/analysis/figs_flatsky_validation"
os.makedirs(OUT, exist_ok=True)
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def apod(n, roll=0.12):  # LOCKED APOD_ROLL_FRAC=0.12 (BUILD_PLAN §1)
    r = np.ones(n, np.float32); k = max(1, int(roll * n))
    c = 0.5 * (1 - np.cos(np.pi * np.arange(k) / k)); r[:k] = c; r[-k:] = c[::-1]
    return np.outer(r, r).astype(np.float32)


def conv_circular(a, b, w):
    return np.fft.irfft2(np.fft.rfft2(a * w) * np.fft.rfft2(b * w), s=a.shape)


def conv_linear(a, b, w):
    """Zero-padded -> linear convolution; return the first NxN block, which is
    index-aligned with the circular convolution (circ[n] = lin[n] + lin[n+N])."""
    n = a.shape[0]; m = 2 * n
    A = np.zeros((m, m), np.float32); B = np.zeros((m, m), np.float32)
    A[:n, :n] = a * w; B[:n, :n] = b * w
    full = np.fft.irfft2(np.fft.rfft2(A) * np.fft.rfft2(B), s=(m, m))
    return full[:n, :n]


def main():
    z = np.load(CACHE); P = z["patches"]; auto = P[..., :4]
    sigma_e = float(z["sigma_e"]); ngal = float(z["galaxy_density"])
    fs = float(z["field_size"]); npix = int(z["field_npix"])
    pix_arcmin = fs * 60 / npix
    sig_pix = sigma_e / np.sqrt(ngal * pix_arcmin ** 2)
    W = apod(npix)
    print(f"patch 10deg/{npix}px  auto σ_pix(noise)={sig_pix:.4g}  auto map std={auto.std():.4g}")

    # ---------- V1: wrap (padded vs unpadded) differ only at edges ----------
    i, j = 2, 3
    circ = np.stack([conv_circular(auto[k, :, :, i], auto[k, :, :, j], W) for k in range(40)])
    lin = np.stack([conv_linear(auto[k, :, :, i], auto[k, :, :, j], W) for k in range(40)])
    rmap = np.array([np.corrcoef(circ[k].ravel(), lin[k].ravel())[0, 1] for k in range(40)])
    diff = np.abs(circ - lin).mean(0)
    yy, xx = np.indices((npix, npix))
    edist = np.minimum.reduce([xx, yy, npix - 1 - xx, npix - 1 - yy])  # px from nearest edge
    prof = np.array([diff[edist == d].mean() for d in range(npix // 2)])
    prof_n = prof / np.abs(lin).mean()
    # NOTE (RETRACTED): this circular-vs-padded r is a REGISTRATION artifact (convolution
    # outputs are lag-indexed; circular vs linear put lag-0 in different places). It does NOT
    # show the convolution is ill-posed. The apodized circular convolution is a valid patch
    # operator. See FLATSKY_CROSS_REDESIGN_NOTES.md §12. Kept for the record only.
    print(f"\n[V1 RETRACTED — registration artifact, not boundary-dependence]:")
    print(f"   map-level Pearson r(circular, padded-first-block) = {rmap.mean():.3f}  "
          f"(shift/crop convention, NOT a defect — see notes §12)")
    print(f"   |circular - padded| / |padded| by distance from edge (px):")
    for d in [0, 2, 4, 8, 16, 32]:
        if d < len(prof_n):
            print(f"     {d:2d}px: {prof_n[d]*100:5.1f}%")

    # ---------- V2: sensitivity to real cross-correlation ----------
    print("\n[V2 signal] cross-correlation sensitivity (bins 3x4):")
    K = 120
    a_i = auto[:K, :, :, i]; a_j = auto[:K, :, :, j]
    a_j_shuf = auto[np.roll(np.arange(K), 37), :, :, j]  # bin j from a DIFFERENT patch
    pm_true = np.mean([np.mean(a_i[k] * a_j[k]) for k in range(K)])
    pm_ind = np.mean([np.mean(a_i[k] * a_j_shuf[k]) for k in range(K)])
    pm_true_s = np.std([np.mean(a_i[k] * a_j[k]) for k in range(K)])
    pm_ind_s = np.std([np.mean(a_i[k] * a_j_shuf[k]) for k in range(K)])
    cv_true = np.mean([conv_linear(a_i[k], a_j[k], W).var() for k in range(K)])
    cv_ind = np.mean([conv_linear(a_i[k], a_j_shuf[k], W).var() for k in range(K)])
    print(f"   POINTWISE-PRODUCT mean : true pair = {pm_true:+.3e} ± {pm_true_s:.1e} | "
          f"independent = {pm_ind:+.3e} ± {pm_ind_s:.1e}  -> ratio {pm_true/max(abs(pm_ind),1e-30):.1f}×")
    print(f"   CONVOLUTION variance   : true pair = {cv_true:.3e} | independent = {cv_ind:.3e}  "
          f"-> ratio {cv_true/cv_ind:.2f}×  (≈1 ⇒ blind to correlation via amplitude)")

    # ---------- V3: cross-map noise is colored and != auto σ ----------
    rng = np.random.default_rng(0)
    n1 = rng.normal(0, sig_pix, (40, npix, npix)).astype(np.float32)
    n2 = rng.normal(0, sig_pix, (40, npix, npix)).astype(np.float32)
    noise_cross = np.stack([conv_linear(n1[k], n2[k], W) for k in range(40)])
    def kmean(f):
        F = np.fft.fftshift(np.abs(np.fft.fft2(f)) ** 2); cy, cx = np.indices(f.shape) - npix // 2
        kr = np.sqrt(cx ** 2 + cy ** 2); F[npix // 2, npix // 2] = 0; return (kr * F).sum() / F.sum()
    print("\n[V3 noise] cross-map of pure noise vs white auto noise:")
    print(f"   auto white noise: std={sig_pix:.4g}")
    print(f"   noise cross map : std={noise_cross.std():.4g}  -> /σ_auto = {noise_cross.std()/sig_pix:.3g}")
    print(f"   ⇒ cross-map noise AMPLITUDE != auto pixel-σ; a shared auto-σ is the wrong SNR "
          f"denominator — need per-channel (and per-scale, on real coloured noise) estimate")

    # ---------- gallery ----------
    k0 = 5
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    for a, m, t in zip(ax,
                       [auto[k0, :, :, j], conv_circular(auto[k0, :, :, i], auto[k0, :, :, j], W),
                        conv_linear(auto[k0, :, :, i], auto[k0, :, :, j], W),
                        (auto[k0, :, :, i] * auto[k0, :, :, j])],
                       ["auto κ4", "cross 3×4 unpadded (circular)",
                        "cross 3×4 padded (linear)", "pointwise product 3×4"]):
        v = np.percentile(np.abs(m), 99)
        im = a.imshow(m, cmap="RdBu_r", vmin=-v, vmax=v); a.set_title(t, fontsize=10); a.axis("off")
    fig.suptitle("Flat-sky cross-map construction — gallery (patch 5, fiducial)", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{OUT}/gallery.png", dpi=130, bbox_inches="tight")

    fig, a = plt.subplots(figsize=(6, 4))
    a.plot(np.arange(len(prof_n)) * pix_arcmin, prof_n * 100, lw=2)
    a.set_xlabel("distance from nearest edge [arcmin]"); a.set_ylabel("|circular−linear| / |linear|  [%]")
    a.set_title("V1: circular-wrap error is an EDGE artifact (padding removes it)")
    a.grid(alpha=0.3); fig.tight_layout(); fig.savefig(f"{OUT}/V1_wrap_profile.png", dpi=130, bbox_inches="tight")
    print(f"\nwrote {OUT}/gallery.png, V1_wrap_profile.png")


if __name__ == "__main__":
    main()
