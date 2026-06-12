#!/usr/bin/env python
"""Post-cut feature masking of an existing flat-local L1 cache (lane B, B0/B3).

A scale cut in the L1 pipeline = dropping the (channel, scale) feature blocks of the
datavector (layout: channel-major, scale-major, l1_nbins per block — col =
c*(n_scales*l1_nbins) + s*l1_nbins + b). This script writes a masked copy of a cache
(train/val) and its fiducial summaries, leaving theta/perm/patch bit-identical.

Keep-schedule string: per-channel ';'-separated kept-scale lists, scale index 0 = FINEST
(sigma tables confirm: noise sigma decreases with index). Schedule M (moderate, BNT
channels shallow->deep):  "3,4;2,3,4;1,2,3,4;0,1,2,3,4"
Schedule U (uniform comparator, noBNT autos):  "3,4;3,4;3,4;3,4"

PLAN_OVERNIGHT_MENU_2.md lane B.
"""
import argparse
import os
import numpy as np

NS_DEFAULT, NB_DEFAULT = 5, 40


def parse_keep(spec: str, n_scales: int) -> list[list[int]]:
    keep = []
    for part in spec.split(";"):
        scales = sorted(int(s) for s in part.split(",") if s.strip() != "")
        assert scales, f"empty kept-scale list in {spec!r}"
        assert all(0 <= s < n_scales for s in scales), (part, n_scales)
        keep.append(scales)
    return keep


def column_mask(keep: list[list[int]], n_scales: int, nbins: int) -> np.ndarray:
    cols = []
    for c, scales in enumerate(keep):
        for s in scales:
            base = c * n_scales * nbins + s * nbins
            cols.extend(range(base, base + nbins))
    return np.asarray(cols, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--fid-npz", required=True)
    ap.add_argument("--keep", required=True)
    ap.add_argument("--out-cache", required=True)
    ap.add_argument("--out-fid", required=True)
    ap.add_argument("--n-scales", type=int, default=NS_DEFAULT)
    ap.add_argument("--l1-nbins", type=int, default=NB_DEFAULT)
    a = ap.parse_args()

    keep = parse_keep(a.keep, a.n_scales)
    feat_per_ch = a.n_scales * a.l1_nbins
    cols = column_mask(keep, a.n_scales, a.l1_nbins)
    print(f"keep schedule: {keep}")
    print(f"kept columns: {cols.size} of {len(keep) * feat_per_ch}")

    os.makedirs(a.out_cache, exist_ok=True)
    for split in ("l1_train", "l1_val"):
        z = np.load(f"{a.cache_dir}/{split}.npz")
        x, theta = z["x"], z["theta"]
        C = x.shape[1] // feat_per_ch
        assert C * feat_per_ch == x.shape[1], (x.shape, feat_per_ch)
        assert C == len(keep), f"cache has {C} channels, keep schedule has {len(keep)}"
        xm = x[:, cols]
        np.savez(f"{a.out_cache}/{split}.npz", theta=theta, x=xm)
        zz = np.load(f"{a.out_cache}/{split}.npz")
        assert np.array_equal(zz["theta"], theta), "theta not bit-equal after write"
        assert zz["x"].shape == (x.shape[0], cols.size)
        print(f"  {split}: {x.shape} -> {xm.shape} (theta bit-equal OK)")

    fz = np.load(a.fid_npz)
    S = fz["S"]
    assert S.shape[1] == len(keep) * feat_per_ch, (S.shape, len(keep) * feat_per_ch)
    out = {"S": S[:, cols].astype(np.float32), "perm": fz["perm"], "patch": fz["patch"]}
    for k in ("truth", "theta"):
        if k in fz.files:
            out[k] = fz[k]
    os.makedirs(os.path.dirname(a.out_fid), exist_ok=True)
    np.savez(a.out_fid, **out)
    print(f"  fid: {S.shape} -> {out['S'].shape}")

    np.savez(f"{a.out_cache}/l1_cache_meta.npz",
             parent=np.array(a.cache_dir), keep=np.array(a.keep),
             n_scales=a.n_scales, l1_nbins=a.l1_nbins, kept_cols=cols,
             note="post-cut feature mask; PLAN_OVERNIGHT_MENU_2.md lane B")
    print("MASK BUILD OK")


if __name__ == "__main__":
    main()
