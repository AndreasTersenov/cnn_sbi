#!/usr/bin/env python3
"""Freeze per-(Haar-mode, scale) NOISE sigma for the 2D-1D Haar SCATTERING ℓ1 (Phase 2).

The modulus folds the Gaussian wavelet coefficient, so quadrature propagation is invalid — we
measure the noise-induced scatter of the modulus-Haar field J_m EMPIRICALLY, with the SAME estimator
and the SAME fiducial signal/noise generation as freeze_flatsky_cross_noise.py (whose S, N_r helpers
we import verbatim): for R realizations, sigma(m,s) = sqrt( mean_{patch,pix} Var_r[ J_{m,s,r} ] ),
Bessel-corrected, subtracting the across-r mean (removes the fixed signal modulus + finite-R bias).

  --pre-basis none  P=I (no-BNT scattering)      --pre-basis bnt  P=B (scattering in BNT space)

Output: <out_dir>/flatsky_haar_scatter_sigma_<pre-basis>.npz  (key 'sigma' (4, n_scales)).
"""
import argparse
import sys
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import freeze_flatsky_cross_noise as fz     # _load_constants, _signal_patches, _noise_patches_worker
import flatsky_cross as fx
import flatsky_haar_scatter as hs

OUT_DIR = HERE / "results/exploratory/flatsky_cross_2026_06"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-basis", choices=("none", "bnt"), default="none")
    ap.add_argument("--n-real", type=int, default=48)
    ap.add_argument("--n-scales", type=int, default=5)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed-base", type=int, default=987654321)
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    a = ap.parse_args()

    import torch
    import multiprocessing as mp
    from wl_stats_torch import WLStatistics

    pre = (a.pre_basis if a.pre_basis == "bnt" else False)   # 'bnt' -> apply B; else identity
    C = fz._load_constants()
    npix, R, S_ns = C["field_npix"], a.n_real, a.n_scales
    print(f"############ FREEZE Haar-scatter noise sigma (pre_basis={a.pre_basis}) ############", flush=True)
    print(f"  field {C['field_size']}deg/{npix}px reso={C['reso_arcmin']}' R={R} n_scales={S_ns}", flush=True)

    S = fz._signal_patches(C)                                # (180,80,80,4) demeaned signal
    n_patches = S.shape[0]
    print(f"  signal S {S.shape}; generating {R} noise realizations on {a.workers} workers "
          f"(spawn context — avoids the fork+healpy-OpenMP deadlock) ...", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers) as pool:
        N_list = pool.map(partial(fz._noise_patches_worker, C=C, seed_base=a.seed_base), range(R))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stats = WLStatistics(n_scales=S_ns, device=dev, pixel_arcmin=C["reso_arcmin"], dtype=torch.float64)
    H = torch.from_numpy(hs.haar4()).to(dev, dtype=torch.float64)        # (4,4)
    M = H.shape[0]
    acc_sum = torch.zeros((M, S_ns, n_patches, npix, npix), dtype=torch.float64, device=dev)
    acc_sq = torch.zeros_like(acc_sum)
    S_t = torch.from_numpy(S).to(dev, dtype=torch.float64)
    print(f"  modulus-Haar pass over r (device={dev}, acc {tuple(acc_sum.shape)}) ...", flush=True)
    for r in range(R):
        data_r = S_t + torch.from_numpy(N_list[r]).to(dev, dtype=torch.float64)   # (180,80,80,4)
        channels4 = fx.build_channels_torch(data_r, "none", bnt=pre)              # P·(S+N)
        J = hs.modulus_haar_fields(channels4, H, stats, subtract_coarse_mean=True)  # (M,180,S_ns,H,W)
        Jp = J.permute(0, 2, 1, 3, 4)                         # (M, S_ns, 180, H, W)
        acc_sum += Jp; acc_sq += Jp * Jp
        if (r + 1) % 8 == 0:
            print(f"    r={r+1}/{R}", flush=True)

    mean = acc_sum / R
    var_pix = (acc_sq - R * mean * mean) / (R - 1)            # (M, S_ns, 180, H, W)
    sigma = torch.sqrt(var_pix.mean(dim=(2, 3, 4))).cpu().numpy()   # (M, S_ns)
    assert np.all(np.isfinite(sigma)) and np.all(sigma > 0), "bad sigma"

    names = ["scat_deep", "scat_coarse", "scat_d12", "scat_d34"]
    print("\n  Frozen per-(mode, scale) noise sigma:", flush=True)
    for m in range(M):
        print("    " + names[m].ljust(12) + "".join(f"{sigma[m,s]:11.4e}" for s in range(S_ns)), flush=True)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    p = out / f"flatsky_haar_scatter_sigma_{a.pre_basis}.npz"
    np.savez(p, sigma=sigma, mode_names=np.array(names), n_scales=S_ns, pre_basis=a.pre_basis,
             n_real=R, note="2D-1D Haar scattering ℓ1 per-mode noise (modulus folds Gaussian; empirical)")
    print(f"\nwrote {p}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
