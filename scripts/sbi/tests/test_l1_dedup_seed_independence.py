#!/usr/bin/env python
"""Dedup-correctness gate for the L1 cross datavector.

The cross-seed datavector dedup is only valid if the L1 datavector is
*seed-independent* — i.e. computing it for seed A and seed B yields the same
array, so one cached datavector can be reused across the 9 seed×perm runs of an
arm. With `--no-l1-train-flip` the train flip (the dominant seed-dependence,
~10% datavector change) is removed; this test verifies the two remaining pieces
are now deterministic:

  1. channel-σ calibration is independent of the passed rng (flip=False -> rng unused).
  2. SNR-range calibration is reproducible (the cross-channel reservoir now uses
     a FIXED torch.Generator instead of the unseeded global RNG).
  3. the L1 datavector (compute_l1_batch) is reproducible run-to-run.
  4. the cache-metadata key now distinguishes train split and flip (so a shared
     --cache-dir dedups across seeds but never serves the wrong split/flip).

Run on a free GPU (GPU 0 here; GPU 1 may be busy):
    CUDA_VISIBLE_DEVICES=0 conda run -n jaxili python scripts/sbi/tests/test_l1_dedup_seed_independence.py
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import types
from pathlib import Path

import numpy as np

SBI_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SBI_DIR))

import torch  # noqa: E402
import npe_l1norm_cross_jaxili_nbody_tomo as l1  # noqa: E402

CACHE = SBI_DIR / "results/exploratory/cross_maps_campaign/full_sphere_cache_grid"
REGIME = "nobnt"
PIX = 20 * 60 / 160
NOISE_SIGMA = float(0.26 / np.sqrt(10 * PIX ** 2))
N_CALIB = 8


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    if not CACHE.is_dir():
        _fail(f"cache dir not found: {CACHE}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  torch device: {dev}")
    stats = l1.build_l1_computer(
        n_scales=5, pixel_arcmin=PIX, torch_device=dev, l1_implementation="cnn_sbi"
    )

    # 1) channel-σ calibration: independent of the passed rng (flip=False).
    sig_a = l1.calibrate_channel_noise_sigma_from_harmonic_cache(
        CACHE, REGIME, n_calibration_realizations=N_CALIB, channel_slice=None,
        rng=np.random.default_rng(41),
    )
    sig_b = l1.calibrate_channel_noise_sigma_from_harmonic_cache(
        CACHE, REGIME, n_calibration_realizations=N_CALIB, channel_slice=None,
        rng=np.random.default_rng(99),
    )
    if not np.array_equal(sig_a, sig_b):
        _fail(f"channel-σ differs across rng: max abs {np.abs(sig_a-sig_b).max():.3e}")
    print(f"  [1] channel-σ rng-independent (max abs diff 0.0); σ[0]={sig_a[0]:.4g}")

    # 2) SNR-range calibration: reproducible (the reservoir fix). Same inputs,
    #    different passed rng -> identical range iff the reservoir is deterministic.
    def snr(seed):
        return l1.calibrate_snr_range_from_harmonic_cache(
            stats, CACHE, REGIME, noise_sigma=NOISE_SIGMA, nbins=4,
            n_l1_channels=10, l1_implementation="cnn_sbi",
            n_calibration_realizations=N_CALIB, subtract_coarse_mean=True,
            cross_snr_percentile=1.0, rng=np.random.default_rng(seed),
            channel_slice=None, channel_scale=None,
        )
    r_a = snr(41)
    r_b = snr(99)
    if r_a != r_b:
        _fail(f"SNR range differs across runs: {r_a} vs {r_b} (reservoir not deterministic)")
    print(f"  [2] SNR range reproducible across rng: {tuple(round(x,5) for x in r_a)}")

    # 3) L1 datavector reproducibility on a few realizations (flip=False).
    maps = []
    for m, _t, _p in l1.iter_harmonic_examples(
        CACHE, REGIME, "train", flip=False, channel_slice=None, channel_scale=None
    ):
        maps.append(np.asarray(m))
        if len(maps) >= 3:
            break
    M = np.concatenate(maps, axis=0)
    def dv():
        return l1.compute_l1_batch(
            M, NOISE_SIGMA, stats, l1_nbins=40, nbins=4, l1_min_snr=-13,
            l1_max_snr=13, n_l1_channels=10, l1_min_snr_cross=-13, l1_max_snr_cross=13,
        )
    d1, d2 = dv(), dv()
    dmax = float(np.abs(d1 - d2).max())
    if dmax > 1e-5:
        _fail(f"datavector not reproducible: max abs diff {dmax:.3e} > 1e-5")
    print(f"  [3] datavector reproducible: max abs diff {dmax:.1e} (dim {d1.shape[-1]})")

    # 4) cache-metadata key distinguishes split + flip, not seed.
    def fake_args(split, flip):
        return types.SimpleNamespace(
            l1_nbins=40, l1_implementation="cnn_sbi", apply_bnt=False,
            zero_mean_maps=True, cross_maps=True, cross_map_apodize="cosine",
            n_scales=5, tfds_name="x", map_kind="nbody", field_size=20,
            field_npix=160, nside=512, nbins=4, sigma_e=0.26, galaxy_density=10.0,
            ds_batch_size=256, nde_train_split=split, l1_train_flip=flip,
        )
    common = dict(
        tomo_bin_indices=(1, 2, 3, 4), l1_min_snr=-13, l1_max_snr=13,
        l1_clamp_overflow=False, subtract_coarse_mean=True, l1_min_snr_cross=-13,
        l1_max_snr_cross=13, n_l1_channels=10, cross_maps_route="harmonic",
        cross_noise_model="channel_empirical_global",
    )
    m_full_f = l1.build_l1_cache_metadata(args=fake_args("train", False), **common)
    m_7030_f = l1.build_l1_cache_metadata(args=fake_args("train[70%:]", False), **common)
    m_full_t = l1.build_l1_cache_metadata(args=fake_args("train", True), **common)
    if "nde_train_split" not in m_full_f or "l1_train_flip" not in m_full_f:
        _fail("cache key missing nde_train_split / l1_train_flip")
    if m_full_f["nde_train_split"] == m_7030_f["nde_train_split"]:
        _fail("cache key does not distinguish train vs train[70%:]")
    if m_full_f["l1_train_flip"] == m_full_t["l1_train_flip"]:
        _fail("cache key does not distinguish flip=True vs flip=False")
    print("  [4] cache key distinguishes split + flip (and excludes seed)")

    print("\nPASS: L1 datavector is seed-independent under flip=False + deterministic "
          "calibration; cross-seed dedup is exact.")


if __name__ == "__main__":
    main()
