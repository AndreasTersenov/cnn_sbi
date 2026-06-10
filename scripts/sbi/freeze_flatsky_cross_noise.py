#!/usr/bin/env python3
"""Freeze per-(channel, scale) NOISE sigma for the flat-sky cross-map L1 SNR.

This is the fix for the old shared-auto-sigma bug. Cross-map noise is colored
(n_i conv/prod n_j + signal x noise terms), so a single white sigma propagated
per-scale is wrong. We measure the per-(channel, scale) noise sigma empirically
and freeze it.

LOCKED design (Andreas, 2026-06-08):
  - FROZEN AT FIDUCIAL, fixed deterministic normalization. Per Andreas: this is a
    change of variables -> it CANNOT bias the SBI. Cross-noise is mildly
    signal-dependent via S(x)N, so the frozen sigma is "most exact near fiducial".
    Do NOT make sigma per-patch or cosmology-dependent.
  - ALL channels uniform (auto + conv + product) -> ONE table reused by every arm.
  - Repo NOISE-based SNR convention: denominator = propagated NOISE sigma (this
    table), NOT the std of the signal+noise filtered map (Zurcher).
  - R = 48 realizations; R/(R-1) (Bessel) correction since we subtract the
    across-r sample mean.

Estimator (per channel c, scale s):
    cross(S + N_r) = S*S + (S*N_r + N_r*S) + N_r*N_r        (* = conv or product)
    Subtracting the across-r mean (per pixel) removes the fixed S*S and the
    finite-R residual of <N*N>, leaving the noise-induced scatter (incl. the
    colored S*N cross-terms). sigma(c,s) = sqrt( mean_{patch,pix} Var_r[coeff] ),
    Var_r Bessel-corrected. Same logic gives the right auto sigma (= W_s[N]).

Faithfulness (verified vs build_full_sphere_cross_cache.py):
  - NSIDE=512 before map2alm(lmax=1024); alm2map back at NSIDE=512.
  - sigma_pix = _per_pixel_noise_std(sigma_e, galaxy_density, nside).
  - INDEPENDENT per-bin noise draw, shape (4, npix).
  Roundtrip is linear => data_auto = roundtrip(signal+noise) = S + N exactly, so
  S + N_r is the same decomposition the cache produced, not an approximation.
  All constants are read from the fiducial cache npz so they cannot drift.

GATE A1b:
  - auto empirical sigma vs analytic WHITE propagation (band-limit departure
    quantified, not assumed);
  - cross sigma per-scale visibly departs from a white scaling.

Output: <out_dir>/flatsky_cross_noise_sigma.{npz,json}
"""
import os
import sys
import json
import glob
import argparse
from pathlib import Path
from functools import partial

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Builder helpers (single source of truth for noise + projection conventions).
import build_full_sphere_cross_cache as bx  # _per_pixel_noise_std, _patch_one_realization, N_AUTO
import flatsky_cross as fx

import healpy as hp
import h5py

FID_CACHE = (HERE / "results/exploratory/cross_maps_campaign/"
             "full_sphere_cache_fiducial_10deg/nobnt/obs")
FID_H5 = Path("/home/tersenov/CosmoGridV1/stage3_forecast/fiducial/"
              "cosmo_fiducial/perm_0000/projected_probes_maps_nobaryons512.h5")
OUT_DIR = HERE / "results/exploratory/flatsky_cross_2026_06"


def _load_constants():
    z = np.load(sorted(glob.glob(str(FID_CACHE / "cosmo_fiducial_perm*.npz")))[0])
    return {
        "nside": int(z["nside"]), "lmax": int(z["lmax"]),
        "sigma_e": float(z["sigma_e"]), "galaxy_density": float(z["galaxy_density"]),
        "field_npix": int(z["field_npix"]), "reso_arcmin": float(z["reso_arcmin"]),
        "field_size": float(z["field_size"]),
        "patch_centers": np.asarray(z["patch_centers"], dtype=np.float64),
    }


def _roundtrip_to_patches(full_maps_sphere, C):
    """map2alm(lmax) -> alm2map(nside,lmax) per bin -> gnomonic patches + per-patch
    demean (exactly the cache builder's auto path). full_maps_sphere: list of 4
    HEALPix maps (npix,). Returns (n_centers, H, W, 4) float32, demeaned."""
    rt = [hp.alm2map(hp.map2alm(m, lmax=C["lmax"], iter=0), nside=C["nside"], lmax=C["lmax"])
          for m in full_maps_sphere]
    return bx._patch_one_realization(rt, C["patch_centers"], C["nside"],
                                     C["field_npix"], C["reso_arcmin"])


def _signal_patches(C):
    with h5py.File(FID_H5, "r") as f:
        kg = f["kg"]
        noiseless = [np.asarray(kg[f"stage3_lensing{b}"], dtype=np.float64) for b in (1, 2, 3, 4)]
    return _roundtrip_to_patches(noiseless, C)  # (180,80,80,4) demeaned signal S


def _noise_patches_worker(r, C, seed_base):
    """One noise realization N_r: independent per-bin sphere shape noise ->
    roundtrip -> patches + demean. Returns (180,80,80,4) float32."""
    npix = hp.nside2npix(C["nside"])
    sig_pix = bx._per_pixel_noise_std(C["sigma_e"], C["galaxy_density"], C["nside"])
    rng = np.random.default_rng(seed_base + r)
    noise_sphere = rng.normal(0.0, sig_pix, size=(bx.N_AUTO, npix))  # independent per bin
    return _roundtrip_to_patches([noise_sphere[b] for b in range(bx.N_AUTO)], C)


def _channel_names(nbins=4):
    pairs = fx.cross_pairs(nbins)
    names = [f"auto_bin{b+1}" for b in range(nbins)]
    names += [f"conv_{i+1}{j+1}" for i, j in pairs]
    names += [f"prod_{i+1}{j+1}" for i, j in pairs]
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=48)
    ap.add_argument("--n-scales", type=int, default=5)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed-base", type=int, default=987654321)
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--bnt", action="store_true",
                    help="Freeze the sigma table in BNT space: S+N_r is BNT'd before the "
                         "channel build (BNT(S+N)=BNT(S)+BNT(N), so the estimator logic is "
                         "unchanged; the table captures the BNT-correlated noise). Output "
                         "filename gains a _bnt suffix and the npz carries bnt=True.")
    args = ap.parse_args()

    import torch
    from multiprocessing import Pool
    from wl_stats_torch import WLStatistics

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    C = _load_constants()
    sig_pix = bx._per_pixel_noise_std(C["sigma_e"], C["galaxy_density"], C["nside"])
    npix = C["field_npix"]
    R, S_ns = args.n_real, args.n_scales
    print("############ FREEZE flat-sky cross noise sigma ############")
    print(f"  nside={C['nside']} lmax={C['lmax']} sigma_e={C['sigma_e']} ngal={C['galaxy_density']}")
    print(f"  field {C['field_size']}deg / {npix}px  reso={C['reso_arcmin']}'  sigma_pix={sig_pix:.6g}")
    print(f"  R={R} realizations, n_scales={S_ns}, {C['patch_centers'].shape[0]} patches")

    print("  building signal patches S (roundtrip of noiseless fiducial sky)...")
    S = _signal_patches(C)                                   # (180,80,80,4) demeaned
    n_patches = S.shape[0]
    print(f"    S shape={S.shape}  std per bin={[f'{S[...,b].std():.4g}' for b in range(4)]}")

    print(f"  generating {R} noise realizations on {args.workers} workers...")
    with Pool(args.workers) as pool:
        N_list = pool.map(partial(_noise_patches_worker, C=C, seed_base=args.seed_base),
                          range(R))
    # Sanity: noise std per bin ~ propagated white std, and inter-bin noise uncorrelated.
    Nstack = np.stack(N_list)                                # (R,180,80,80,4)
    print(f"    N noise std per bin={[f'{Nstack[...,b].std():.4g}' for b in range(4)]} "
          f"(roundtripped; < sigma_pix={sig_pix:.4g} due to lmax band-limit)")
    inter = np.corrcoef(Nstack[..., 0].ravel()[:200000], Nstack[..., 1].ravel()[:200000])[0, 1]
    print(f"    inter-bin noise corr (bin0,bin1) = {inter:+.4f} (must be ~0 => independent)")

    # ---- GPU wavelet accumulation (online sum / sumsq across r) ----
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stats = WLStatistics(n_scales=S_ns, device=dev, pixel_arcmin=C["reso_arcmin"], dtype=torch.float64)
    n_ch = fx.n_output_channels(4, "both")                   # 16
    acc_sum = torch.zeros((n_ch, S_ns, n_patches, npix, npix), dtype=torch.float64, device=dev)
    acc_sq = torch.zeros_like(acc_sum)
    S_t = torch.from_numpy(S).to(dev, dtype=torch.float64)
    print(f"  wavelet pass over r (device={dev}, accumulators {tuple(acc_sum.shape)})...")
    for r in range(R):
        data_r = S_t + torch.from_numpy(N_list[r]).to(dev, dtype=torch.float64)  # demean(S)+demean(N)
        chans = fx.build_channels_torch(data_r, "both", bnt=args.bnt)  # (180,80,80,16)
        for c in range(n_ch):
            stats.compute_wavelet_transform(chans[..., c], 1.0, subtract_coarse_mean=True)
            wc = stats.wavelet_coeffs                         # (180, S_ns, H, W)
            wc = wc.permute(1, 0, 2, 3)                       # (S_ns, 180, H, W)
            acc_sum[c] += wc
            acc_sq[c] += wc * wc
        if (r + 1) % 8 == 0:
            print(f"    r={r+1}/{R}")

    # Bessel-corrected per-pixel variance, then pool over (patch, pixel).
    mean = acc_sum / R
    var_pix = (acc_sq - R * mean * mean) / (R - 1)            # (n_ch, S_ns, 180, H, W)
    sigma = torch.sqrt(var_pix.mean(dim=(2, 3, 4))).cpu().numpy()  # (n_ch, S_ns)

    names = _channel_names(4)
    print("\n  Frozen per-(channel, scale) NOISE sigma:")
    print("    " + "channel".ljust(11) + "".join(f"  scale{s}".rjust(11) for s in range(S_ns)))
    for c in range(n_ch):
        print("    " + names[c].ljust(11) + "".join(f"{sigma[c,s]:11.4e}" for s in range(S_ns)))

    # ---- analytic WHITE per-scale level (pipeline's propagation of sigma_pix) ----
    # Use the public API: feed a dummy image with noise_sigma=sigma_pix and read the
    # per-scale noise_levels the pipeline would assume for a WHITE-noise auto channel.
    dummy = torch.zeros((1, npix, npix), dtype=torch.float64, device=dev)
    stats.compute_wavelet_transform(dummy, sig_pix, subtract_coarse_mean=False)
    white_scale = stats.noise_levels.mean(dim=(0, 2, 3)).cpu().numpy()  # (S_ns,)

    # ---- save FIRST (robust: never lose the expensive compute) ----
    provenance = ("frozen at fiducial; fixed deterministic normalization (does not bias SBI); "
                  "cross-noise is mildly signal-dependent via S(x)N so most exact near fiducial")
    out_dir.mkdir(parents=True, exist_ok=True)
    _stem = "flatsky_cross_noise_sigma" + ("_bnt" if args.bnt else "")
    npz_path = out_dir / f"{_stem}.npz"
    np.savez(npz_path, sigma=sigma, channel_names=np.array(names), n_scales=S_ns,
             white_per_scale=white_scale, sigma_pix=sig_pix, n_real=R,
             nside=C["nside"], lmax=C["lmax"], sigma_e=C["sigma_e"],
             galaxy_density=C["galaxy_density"], reso_arcmin=C["reso_arcmin"],
             field_npix=npix, seed_base=args.seed_base, provenance=provenance,
             bnt=bool(args.bnt))
    print(f"\nwrote {npz_path}")

    # ---- GATE A1b checks ----
    print("\n========== GATE A1b ==========")
    # (1) auto empirical vs analytic WHITE propagation. Under BNT each auto bin is a
    # linear mix of INDEPENDENT white noises -> still spatially white per map, with
    # amplitude sqrt(sum_j B_ij^2); the analytic reference scales per bin accordingly
    # (the cross-BIN correlation BNT creates does not affect this per-map check).
    if args.bnt:
        bnt_amp = np.sqrt((fx.bnt_matrix_np().astype(np.float64) ** 2).sum(axis=1))
        print(f"  BNT per-bin white-amplitude factors sqrt(sum B_ij^2): "
              + " ".join(f"{a:.3f}" for a in bnt_amp))
    else:
        bnt_amp = np.ones(4)
    print("  AUTO empirical sigma / analytic-white per scale (expect <1 at fine scales = band-limit):")
    auto_ok = True
    for b in range(4):
        ratios = sigma[b] / (white_scale * bnt_amp[b])
        print(f"    auto_bin{b+1}: " + " ".join(f"s{s}={ratios[s]:.3f}" for s in range(S_ns)))
        # coarse scales (least band-limit-affected) should be within ~30% of white
        if not (0.5 < ratios[-2] < 1.5):
            auto_ok = False
    print(f"    white per-scale level: " + " ".join(f"{white_scale[s]:.3e}" for s in range(S_ns)))
    print(f"    --> auto vs white: {'PASS (coarse ~ white, fine band-limited)' if auto_ok else 'CHECK'}")

    # (2) cross sigma departs from white per-scale shape
    def norm_profile(v):
        v = np.asarray(v, float); return v / v.sum()
    white_prof = norm_profile(white_scale)
    conv_prof = norm_profile(sigma[4:10].mean(0))   # avg over 6 conv channels
    prod_prof = norm_profile(sigma[10:16].mean(0))  # avg over 6 product channels
    l1dist_conv = float(np.abs(conv_prof - white_prof).sum())
    l1dist_prod = float(np.abs(prod_prof - white_prof).sum())
    print("  per-scale NORMALIZED profile (white vs conv vs product) — cross must DEPART:")
    print(f"    white  : {' '.join(f'{x:.3f}' for x in white_prof)}")
    print(f"    conv   : {' '.join(f'{x:.3f}' for x in conv_prof)}  (L1 dist from white {l1dist_conv:.3f})")
    print(f"    product: {' '.join(f'{x:.3f}' for x in prod_prof)}  (L1 dist from white {l1dist_prod:.3f})")
    cross_ok = (l1dist_conv > 0.1) or (l1dist_prod > 0.1)
    print(f"    --> cross departs from white: {'PASS' if cross_ok else 'FAIL (looks white?!)'}")

    inter_ok = abs(inter) < 0.02
    print(f"  inter-bin noise independence: {'PASS' if inter_ok else 'FAIL'} ({inter:+.4f})")
    gate_ok = auto_ok and cross_ok and inter_ok
    print(f"\nGATE A1b: {'ALL PASS' if gate_ok else 'NEEDS REVIEW'}")

    # ---- save human-readable json (npz already saved above) ----
    json_path = out_dir / f"{_stem}.json"
    json_path.write_text(json.dumps({
        "provenance": provenance,
        "bnt": bool(args.bnt),
        "channel_names": names,
        "n_scales": S_ns, "n_real": R, "sigma_pix": sig_pix,
        "constants": {k: C[k] for k in ("nside", "lmax", "sigma_e", "galaxy_density",
                                        "reso_arcmin", "field_npix", "field_size")},
        "sigma": {names[c]: [float(sigma[c, s]) for s in range(S_ns)] for c in range(n_ch)},
        "white_per_scale": [float(x) for x in white_scale],
        "gate_a1b": {"auto_vs_white": bool(auto_ok), "cross_departs": bool(cross_ok),
                     "inter_bin_indep": bool(inter_ok), "inter_bin_corr": float(inter)},
    }, indent=2))
    print(f"\nwrote {npz_path}\nwrote {json_path}")
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
