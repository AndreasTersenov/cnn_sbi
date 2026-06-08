#!/usr/bin/env python3
"""D9 — the harmonic cross-maps are large-scale / non-local (leakage diagnostic).

Cross channels are built as full-sphere alm products (a^i_lm * a^j_lm) -> iSHT on
the WHOLE sphere -> gnomonic patch cutouts (build_full_sphere_cross_cache.py). So
every cross-patch pixel is a global functional of the full-sphere field. This plot
shows the angular scale content: cross power sits at much lower ell than auto power,
with a large fraction BELOW the 10deg patch scale (ell~18) = info larger than the
patch (leaked from the rest of the field). CPU-only.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np, healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SNAP = ("results/exploratory/cross_maps_campaign/full_sphere_cache_fiducial_10deg/"
        "_snapshot/fullsphere_nobnt_cosmo_fiducial_perm0.npz")
OUT = "results/exploratory/definitive_comparison_10deg/phase_c/analysis/figs"
os.makedirs(OUT, exist_ok=True)
PAIRS = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
L_PATCH = 18  # ~180/10deg


def main():
    z = np.load(SNAP); FA = z["full_auto"]; FC = z["full_cross"]; lmax = int(z["lmax"])
    ell = np.arange(lmax + 1); w = 2 * ell + 1

    def varspec(m):
        cl = hp.anafast(m.astype(np.float64), lmax=lmax)
        v = w * cl; v[0] = 0
        return v / v.sum()

    auto_v = np.mean([varspec(FA[b]) for b in range(4)], axis=0)
    cross_v = np.mean([varspec(FC[k]) for k in range(6)], axis=0)

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))
    # left: variance contribution per ell (normalized), log-x
    a = ax[0]
    a.plot(ell[1:], auto_v[1:], color="#d62728", lw=2, label="auto κ (mean of 4 bins)")
    a.plot(ell[1:], cross_v[1:], color="#1f77b4", lw=2, label="cross κ$^{ij}$ (mean of 6 pairs)")
    a.axvline(L_PATCH, color="k", ls="--", lw=1.2)
    a.text(L_PATCH * 1.1, a.get_ylim()[1] * 0.5, "10° patch scale\n(ℓ≈18)", fontsize=8)
    a.axvspan(1, L_PATCH, color="0.85", zorder=0)
    a.set_xscale("log"); a.set_xlabel("multipole ℓ")
    a.set_ylabel("variance fraction per ℓ  [(2ℓ+1)C$_ℓ$, normalized]")
    a.set_title("Cross power peaks at far lower ℓ than auto\n(grey = scales larger than the patch)")
    a.legend(fontsize=9); a.grid(alpha=0.2, which="both")

    # right: cumulative variance fraction
    a = ax[1]
    a.plot(ell, np.cumsum(auto_v), color="#d62728", lw=2, label="auto κ")
    a.plot(ell, np.cumsum(cross_v), color="#1f77b4", lw=2, label="cross κ$^{ij}$")
    a.axvline(L_PATCH, color="k", ls="--", lw=1.2)
    a.axvspan(0, L_PATCH, color="0.85", zorder=0)
    fa = np.cumsum(auto_v)[L_PATCH]; fc = np.cumsum(cross_v)[L_PATCH]
    a.text(L_PATCH * 1.3, 0.12, f"at ℓ<18 (super-patch):\nauto {fa*100:.1f}%  vs  cross {fc*100:.1f}%",
           fontsize=9)
    a.set_xscale("log"); a.set_xlabel("multipole ℓ"); a.set_ylabel("cumulative variance fraction")
    a.set_ylim(0, 1); a.set_title("Cross channels carry 15–30× more super-patch (leaked) variance")
    a.legend(fontsize=9, loc="center right"); a.grid(alpha=0.2, which="both")

    fig.suptitle("D9 — harmonic cross-maps are large-scale / non-local: full-sphere alm-product "
                 "sliced into patches ⇒ each patch carries info from the whole sphere", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/D9_cross_leakage_scales.{ext}", dpi=140, bbox_inches="tight")
    print("wrote D9_cross_leakage_scales")


if __name__ == "__main__":
    main()
