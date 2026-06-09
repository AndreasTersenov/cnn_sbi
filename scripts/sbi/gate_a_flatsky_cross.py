#!/usr/bin/env python3
"""GATE A1 — flat-sky cross-map OPERATOR correctness (standalone, pre-wiring).

Validates scripts/sbi/flatsky_cross.py before it is wired into the L1/CNN pipelines.
No training, no wavelet here (noise-sigma freezing is GATE A1b). Runs on GPU 1 so
the torch/jax bit-match reflects the production device.

Checks:
  A1.1 channel-order : cache ch 0-3 are autos in BIN ORDER (variance grows with bin),
                       and >> the harmonic-cross channels 4-9 in amplitude.
  A1.2 bit-match     : numpy(CPU,double-FFT) vs torch(GPU) vs jax(GPU) agree for
                       conv & product & both (float32 FFT roundoff tolerance).
  A1.3 xi_ij recovery: product-map spatial mean reproduces the tomographic cross-cov
                       structure (REDESIGN_NOTES §14): diagonal grows bin1->4, all
                       off-diagonals positive, r(3,4) > r(2,4) > r(1,4).
  A1.4 conv mean ~ 0 : UNapodized convolution mean ~ 0 (math sanity: (Σκ)(Σκ)/N=0 for
                       demeaned autos); apodized conv mean is a (nonzero) window artifact.

Env: CUDA_VISIBLE_DEVICES=1 (set by the launcher). Reads the fiducial obs cache.
"""
import os
import glob
import numpy as np

CACHE = ("scripts/sbi/results/exploratory/cross_maps_campaign/"
         "full_sphere_cache_fiducial_10deg/nobnt/obs")
HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, HERE)
import flatsky_cross as fx


def _load_autos(perm_glob, max_perms=None, max_patches=None):
    fs = sorted(glob.glob(perm_glob))
    if max_perms:
        fs = fs[:max_perms]
    autos = []
    for f in fs:
        P = np.load(f)["patches"][..., :4].astype(np.float32)  # (n_centers, H, W, 4)
        if max_patches:
            P = P[:max_patches]
        autos.append(P)
    return np.concatenate(autos, axis=0), fs


def _relnorm(a, b):
    """max abs diff and max-abs-diff normalized by RMS of reference `a`."""
    d = np.abs(a - b)
    rms = np.sqrt(np.mean(a.astype(np.float64) ** 2)) + 1e-30
    return float(d.max()), float(d.max() / rms)


def check_channel_order():
    print("\n========== A1.1 channel order (cache ch 0-3 = autos in bin order) ==========")
    z = np.load(sorted(glob.glob(CACHE + "/cosmo_fiducial_perm*.npz"))[0])
    P = z["patches"]  # (n, H, W, 10)
    stds = [float(P[..., c].std()) for c in range(10)]
    print("  per-channel std:", " ".join(f"ch{c}={stds[c]:.3g}" for c in range(10)))
    auto_std = stds[:4]
    cross_std = stds[4:]
    mono = all(auto_std[i] < auto_std[i + 1] for i in range(3))
    sep = min(auto_std) > 50 * max(cross_std)
    print(f"  auto std monotonic increasing bin1->4 : {mono}  ({[f'{s:.3g}' for s in auto_std]})")
    print(f"  autos >> harmonic-cross amplitude (>50x): {sep}  "
          f"(min auto {min(auto_std):.3g} vs max cross {max(cross_std):.3g})")
    ok = mono and sep
    print(f"  --> A1.1 {'PASS' if ok else 'FAIL'}")
    return ok


def check_bitmatch():
    print("\n========== A1.2 bit-match np(CPU) vs torch(GPU) vs jax(GPU) ==========")
    autos_np, _ = _load_autos(CACHE + "/cosmo_fiducial_perm0.npz", max_patches=64)
    print(f"  autos batch: {autos_np.shape}")
    import torch
    import jax
    import jax.numpy as jnp
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  torch device={dev}; jax devices={jax.devices()}")
    autos_t = torch.from_numpy(autos_np).to(dev, dtype=torch.float32)
    autos_j = jnp.asarray(autos_np)
    ok_all = True
    for op in ("conv", "product", "both"):
        ref = fx.build_channels_np(autos_np, op)
        out_t = fx.build_channels_torch(autos_t, op).detach().cpu().numpy()
        out_j = np.asarray(fx.build_channels_jax(autos_j, op))
        assert ref.shape == out_t.shape == out_j.shape, (op, ref.shape, out_t.shape, out_j.shape)
        a_tj, r_tj = _relnorm(out_t, out_j)         # torch vs jax (both single)
        a_nt, r_nt = _relnorm(ref, out_t)           # numpy(double) vs torch
        a_nj, r_nj = _relnorm(ref, out_j)           # numpy(double) vs jax
        # product is exact elementwise; conv carries float32 FFT roundoff.
        thresh = 1e-5 if op == "product" else 2e-3
        worst = max(r_tj, r_nt, r_nj)
        ok = worst < thresh
        ok_all &= ok
        print(f"  op={op:8s} shape={ref.shape}  relmax: t-vs-j={r_tj:.2e} "
              f"np-vs-t={r_nt:.2e} np-vs-j={r_nj:.2e}  (thr {thresh:.0e})  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"  --> A1.2 {'PASS' if ok_all else 'FAIL'}")
    return ok_all


def check_xi_recovery():
    print("\n========== A1.3 xi_ij recovery (product mean = tomographic cross-cov) ==========")
    autos, fs = _load_autos(CACHE + "/cosmo_fiducial_perm*.npz", max_perms=40)
    print(f"  loaded {autos.shape[0]} patches from {len(fs)} perms")
    # Maps are demeaned per patch already; re-demean to be safe.
    autos = autos - autos.mean(axis=(1, 2), keepdims=True)
    # 4x4 cross-covariance via spatial mean of pairwise products, averaged over patches.
    C = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            C[i, j] = np.mean(autos[..., i] * autos[..., j])  # mean over patches & pixels
    diag = np.diag(C)
    r = C / np.sqrt(np.outer(diag, diag))
    print("  cross-cov C (x1e4):")
    for i in range(4):
        print("    " + " ".join(f"{C[i,j]*1e4:7.3f}" for j in range(4)))
    diag_grows = all(diag[i] < diag[i + 1] for i in range(3))
    offdiag_pos = all(C[i, j] > 0 for i in range(4) for j in range(i + 1, 4))
    ordered = r[2, 3] > r[1, 3] > r[0, 3]
    print(f"  diagonal grows bin1->4              : {diag_grows}  ({[f'{d*1e4:.3f}' for d in diag]})")
    print(f"  all off-diagonals positive          : {offdiag_pos}")
    print(f"  r(3,4) > r(2,4) > r(1,4)            : {ordered}  "
          f"({r[2,3]:.3f} > {r[1,3]:.3f} > {r[0,3]:.3f})")
    ok = diag_grows and offdiag_pos and ordered
    print(f"  --> A1.3 {'PASS' if ok else 'FAIL'}")
    return ok


def check_conv_mean():
    print("\n========== A1.4 unapodized conv mean ~ 0 (math sanity) ==========")
    autos, _ = _load_autos(CACHE + "/cosmo_fiducial_perm0.npz", max_patches=180)
    autos = autos - autos.mean(axis=(1, 2), keepdims=True)
    npix = autos.shape[1]
    ones = np.ones((npix, npix), dtype=np.float32)
    conv_unapod = fx._conv_np(autos, ones)              # (B, H, W, 6)
    conv_apod = fx._conv_np(autos, fx.apod_window_np(npix, 0.10))
    m_unapod = float(np.abs(conv_unapod.mean(axis=(1, 2))).max())
    m_apod = float(np.abs(conv_apod.mean(axis=(1, 2))).max())
    rms = float(np.sqrt((conv_unapod ** 2).mean()))
    print(f"  |mean| unapodized conv (max over patch,pair): {m_unapod:.3e}  (map rms {rms:.3e})")
    print(f"  |mean| apodized conv   (window artifact, nonzero): {m_apod:.3e}")
    ok = m_unapod < 1e-6 * max(rms, 1e-12) or m_unapod < 1e-9
    print(f"  --> A1.4 {'PASS' if ok else 'FAIL'}  (unapodized mean ~ 0)")
    return ok


def main():
    print("############ GATE A1 — flat-sky cross operator correctness ############")
    print(f"cache: {CACHE}")
    results = {
        "A1.1 channel-order": check_channel_order(),
        "A1.2 bit-match": check_bitmatch(),
        "A1.3 xi_ij recovery": check_xi_recovery(),
        "A1.4 conv mean~0": check_conv_mean(),
    }
    print("\n############ GATE A1 SUMMARY ############")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    allok = all(results.values())
    print(f"\nGATE A1: {'ALL PASS' if allok else 'FAILURES PRESENT'}")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
