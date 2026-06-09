#!/usr/bin/env python3
"""GATE B — do the FLAT-LOCAL cross statistics vary with cosmology? (decisive info test)

The frozen-sigma + xi_ij checks only validate construction physics AT the fiducial. This
gate is the one the fiducial-only checks could not do: confirm the flat-local cross-channel
L1 summaries MOVE with theta across the multi-cosmology TFDS. If they don't, the channels
carry no cosmological information and the campaign is pointless -> stop and debug.

Method: sample train-split patches (wide theta coverage), compute op=both L1 (autos + conv +
product) per patch + theta. Bin by a parameter (sigma8, then Omega_m), take per-bin mean
feature vectors, and correlate each feature's bin-means against the bin's parameter value.
A feature that 'moves with cosmology' has |r| ~ 1 across bins. Report the fraction of CROSS
features (conv, product) that track theta, alongside the AUTO features (sanity: autos must).

Runs on GPU 1.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flatsky_cross as fx
import flatsky_cross_l1 as fxl

SIGMA = HERE + "/results/exploratory/flatsky_cross_2026_06/flatsky_cross_noise_sigma.npz"
TFDS = "nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180"
DDIR = "/home/tersenov/tensorflow_datasets"
RESO, L1_NBINS, NBINS = 7.5, 40, 4
OP = "both"
N_PATCHES = 40000
N_BINS = 8
PARAMS = {"Omega_m": 0, "sigma8": 1, "w0": 2}   # theta index


def main():
    import torch
    from wl_stats_torch import WLStatistics
    from tfds_cross_tfdata_loader import iter_cross_tfds_batches
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stats = WLStatistics(n_scales=5, device=dev, pixel_arcmin=RESO, dtype=torch.float64)
    sig, names, n_scales = fxl.select_frozen_sigma(SIGMA, OP, NBINS, dev)
    C = fx.n_output_channels(NBINS, OP)
    ranges = np.tile([-8.0, 8.0], (C, 1))   # fixed broad range (info test, not the calibrated arm)
    feat_per_ch = n_scales * L1_NBINS

    print(f"############ GATE B — cosmology dependence (op={OP}, {N_PATCHES} patches) ############")
    xs, ths = [], []
    n, t0 = 0, time.time()
    for autos_np, theta in iter_cross_tfds_batches(
        TFDS, DDIR, "train", 512, flip=False, channel_scale=None,
        channel_slice=slice(0, 4), perm_lo=0, perm_hi=3, seed=7,
    ):
        xs.append(fxl.build_and_l1(autos_np, OP, sig, stats, L1_NBINS, ranges, clamp_overflow=True))
        ths.append(theta.copy())
        n += autos_np.shape[0]
        if n >= N_PATCHES:
            break
    X = np.concatenate(xs)[:N_PATCHES]               # (N, C*feat_per_ch)
    TH = np.concatenate(ths)[:N_PATCHES]             # (N, 6)
    print(f"  collected {X.shape[0]} patches ({time.time()-t0:.0f}s); summary dim {X.shape[1]}")

    # channel groups in op=both order: auto(0-3), conv(4-9), product(10-15)
    groups = {"auto": range(0, 4), "conv": range(4, 10), "product": range(10, 16)}

    def feat_slice(ch):
        return slice(ch * feat_per_ch, (ch + 1) * feat_per_ch)

    for pname, pidx in PARAMS.items():
        p = TH[:, pidx]
        order = np.argsort(p)
        edges = np.linspace(0, len(p), N_BINS + 1).astype(int)
        binp = np.zeros(N_BINS)
        binX = np.zeros((N_BINS, X.shape[1]))
        for b in range(N_BINS):
            idx = order[edges[b]:edges[b + 1]]
            binp[b] = p[idx].mean()
            binX[b] = X[idx].mean(0)
        # |corr| of each feature's bin-means vs the bin parameter value
        pc = binp - binp.mean()
        denp = np.sqrt((pc ** 2).sum()) + 1e-30
        Xc = binX - binX.mean(0, keepdims=True)
        denX = np.sqrt((Xc ** 2).sum(0)) + 1e-30
        rabs = np.abs((Xc * pc[:, None]).sum(0) / (denX * denp))   # (n_features,)
        print(f"\n  --- {pname} (range {binp.min():.3f}..{binp.max():.3f}) ---")
        for gname, chs in groups.items():
            r_g = np.concatenate([rabs[feat_slice(c)] for c in chs])
            frac = float((r_g > 0.7).mean())
            print(f"    {gname:8s}: median|r|={np.median(r_g):.3f} max|r|={r_g.max():.3f} "
                  f"frac(|r|>0.7)={frac:.2f}")

    print("\nGATE B verdict: PASS if CROSS (conv & product) features track theta (median|r| well "
          "above 0 and a real frac>0.7), i.e. the cross channels carry cosmological information.")


if __name__ == "__main__":
    main()
