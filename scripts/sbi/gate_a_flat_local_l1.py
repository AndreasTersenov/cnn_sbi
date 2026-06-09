#!/usr/bin/env python3
"""GATE A1c — flat_local L1 module (frozen-sigma) correctness, standalone.

Checks flatsky_cross_l1.py before it is wired into npe_l1norm_cross_jaxili_nbody_tomo.py:
  - frozen sigma selection: shapes + channel order per op
  - datavector shape = C * n_scales * l1_nbins; finite; non-degenerate
  - obs single-map path == train batch path for the SAME autos (bit-match)
  - auto-channel L1 is identical whether op=conv or op=both (autos op-independent)
  - conv L1 != product L1 (operators carry different info)
  - frozen-sigma SNR is well-behaved O(1) for BOTH auto and cross (NOT the collapsed
    cross SNR of the old shared-auto-sigma bug)
Runs on GPU 1.
"""
import os
import sys
import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flatsky_cross as fx
import flatsky_cross_l1 as fxl

CACHE = (HERE + "/results/exploratory/cross_maps_campaign/"
         "full_sphere_cache_fiducial_10deg/nobnt/obs/cosmo_fiducial_perm0.npz")
SIGMA = HERE + "/results/exploratory/flatsky_cross_2026_06/flatsky_cross_noise_sigma.npz"
L1_NBINS, NBINS = 40, 4


def main():
    import torch
    from wl_stats_torch import WLStatistics
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(CACHE)
    autos = z["patches"][:, :, :, :4].astype(np.float32)           # (180,80,80,4)
    reso = float(z["reso_arcmin"])
    print(f"autos {autos.shape} reso={reso}'  device={dev}")
    results = {}

    # ---- frozen sigma selection ----
    print("\n== frozen sigma selection ==")
    sel_ok = True
    for op, expect_C in (("conv", 10), ("product", 10), ("both", 16)):
        sig, names, n_scales = fxl.select_frozen_sigma(SIGMA, op, NBINS, dev)
        ok = sig.shape == (expect_C, n_scales) and len(names) == expect_C
        print(f"  op={op:8s} sigma{tuple(sig.shape)} names[0:4]={names[:4]} ... [{names[4]}..]  {'OK' if ok else 'BAD'}")
        sel_ok &= ok
    results["sigma-selection"] = sel_ok

    stats = WLStatistics(n_scales=5, device=dev, pixel_arcmin=reso, dtype=torch.float64)
    batch = autos[:32]

    # ---- datavector shape + finiteness + obs==train bit-match ----
    print("\n== datavector shape, finiteness, obs==train ==")
    dv_ok = True
    dvs = {}
    for op in ("conv", "product", "both"):
        sig, _, n_scales = fxl.select_frozen_sigma(SIGMA, op, NBINS, dev)
        C = fx.n_output_channels(NBINS, op)
        ranges = np.tile([-10.0, 10.0], (C, 1))   # broad per-channel range for shape tests
        x = fxl.build_and_l1(batch, op, sig, stats, L1_NBINS, ranges)
        exp_D = C * n_scales * L1_NBINS
        single = fxl.compute_l1_single_map_flat_local(batch[7], op, sig, stats, L1_NBINS, ranges)
        match = np.allclose(single, x[7], rtol=0, atol=1e-9)
        finite = np.isfinite(x).all()
        nondegen = float(x.std()) > 0
        ok = x.shape == (32, exp_D) and finite and nondegen and match
        print(f"  op={op:8s} x{x.shape} (exp D={exp_D}) finite={finite} std={x.std():.3e} "
              f"obs==train={match}  {'OK' if ok else 'BAD'}")
        dv_ok &= ok
        dvs[op] = x
    results["datavector+obs-match"] = dv_ok

    # ---- autos op-independent; conv != product ----
    print("\n== auto op-independence + conv != product ==")
    auto_D = NBINS * 5 * L1_NBINS
    auto_conv = dvs["conv"][:, :auto_D]
    auto_both = dvs["both"][:, :auto_D]
    auto_indep = np.allclose(auto_conv, auto_both, rtol=0, atol=1e-9)
    conv_cross = dvs["conv"][:, auto_D:]
    prod_cross = dvs["product"][:, auto_D:]
    differ = not np.allclose(conv_cross, prod_cross, rtol=1e-3, atol=1e-12)
    print(f"  auto L1 identical (conv vs both): {auto_indep}")
    print(f"  conv-cross L1 != product-cross L1: {differ}")
    results["auto-indep + conv!=prod"] = auto_indep and differ

    # ---- frozen-sigma SNR not collapsed (the old bug => cross SNR ~ 0) ----
    print("\n== frozen-sigma SNR not collapsed (median O(0.1-2), max > 1) ==")
    sig, names, n_scales = fxl.select_frozen_sigma(SIGMA, "both", NBINS, dev)
    chans = fx.build_channels_torch(torch.from_numpy(batch.astype(np.float64)).to(dev), "both")
    snr_ok = True
    for grp, idxs in (("auto", range(0, 4)), ("conv", range(4, 10)), ("prod", range(10, 16))):
        mx, med = 0.0, []
        for c in idxs:
            stats.compute_wavelet_transform(chans[..., c], 1.0, subtract_coarse_mean=True)
            s = (stats.wavelet_coeffs / sig[c].view(1, n_scales, 1, 1)).abs()
            mx = max(mx, float(s.max())); med.append(float(s.median()))
        med = float(np.mean(med))
        ok = mx > 1.0 and med > 0.05         # NOT collapsed (old bug => max,median ~ 1e-4)
        tail = "heavy-tailed (expected for product)" if mx > 60 else ""
        print(f"  {grp}: median|SNR|={med:.3f} max|SNR|={mx:.2f}  {'OK' if ok else 'COLLAPSED'} {tail}")
        snr_ok &= ok
    results["snr-not-collapsed"] = snr_ok

    # ---- per-channel SNR range calibration returns sane (C,2) ----
    print("\n== per-channel SNR-range calibration (frozen sigma) ==")
    try:
        ranges = fxl.calibrate_snr_range_flat_local(
            "nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180", None,
            "both", sig, stats, NBINS, names, n_calibration_examples=900, perm_lo=0, perm_hi=2)
        calib_ok = (ranges.shape == (16, 2) and np.isfinite(ranges).all()
                    and bool((ranges[:, 1] > ranges[:, 0]).all()))
        # product ranges (rows 10-15) should be MUCH wider than conv (rows 4-9) due to heavy tail
        conv_w = float(np.mean(ranges[4:10, 1] - ranges[4:10, 0]))
        prod_w = float(np.mean(ranges[10:16, 1] - ranges[10:16, 0]))
        print(f"  mean conv range width={conv_w:.1f}, product range width={prod_w:.1f} "
              f"(product wider: {prod_w > conv_w})")
        print(f"  --> {'OK' if calib_ok else 'BAD'}")
    except Exception as e:
        import traceback; traceback.print_exc()
        calib_ok = False
    results["snr-calibration"] = calib_ok

    print("\n############ GATE A1c SUMMARY ############")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    allok = all(results.values())
    print(f"\nGATE A1c: {'ALL PASS' if allok else 'FAILURES'}")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
