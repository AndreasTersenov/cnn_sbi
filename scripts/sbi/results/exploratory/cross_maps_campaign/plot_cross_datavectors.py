"""Visualize the auto+cross L1 datavectors that the NPE actually sees.

Loads the cached training tensors `l1_train.npz` for the BNT and no-BNT
cross runs, reshapes (n_examples, 2000) -> (n_examples, n_scales=5,
n_l1_bins=40, n_channels=10), computes per-channel mean ± std across
examples, and plots a 10-panel figure per regime — one row per channel
(4 auto bins, then 6 cross pairs), with vertical lines marking the
5 wavelet-scale boundaries.

Output:
  cross_summary/datavectors_bnt.png
  cross_summary/datavectors_nobnt.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/cross_maps_campaign")
OUT = ROOT / "cross_summary"
OUT.mkdir(exist_ok=True)

N_SCALES = 5
N_L1_BINS = 40
N_AUTO = 4
N_CROSS = 6
DV_AUTO_ONLY = N_SCALES * N_L1_BINS * N_AUTO         # 800
DV_AUTO_CROSS = N_SCALES * N_L1_BINS * (N_AUTO + N_CROSS)  # 2000


def channel_names(n_chan: int) -> list[str]:
    if n_chan == N_AUTO:
        return [f"auto_{b+1}" for b in range(N_AUTO)]
    if n_chan == N_AUTO + N_CROSS:
        return (
            [f"auto_{b+1}" for b in range(N_AUTO)]
            + [f"cross_{i+1}{j+1}" for i in range(N_AUTO) for j in range(i + 1, N_AUTO)]
        )
    raise ValueError(f"Unsupported channel count {n_chan}")


def reshape_dv(x: np.ndarray) -> np.ndarray:
    """(N, dv) -> (N, n_chan, n_scales, n_l1_bins). n_chan inferred from dv."""
    dv = x.shape[1]
    if dv == DV_AUTO_ONLY:
        n_chan = N_AUTO
    elif dv == DV_AUTO_CROSS:
        n_chan = N_AUTO + N_CROSS
    else:
        raise ValueError(f"Unexpected datavector dim {dv}")
    return x.reshape(-1, n_chan, N_SCALES, N_L1_BINS)


REGIME_CACHES = {
    "bnt": "jaxili_cross_bnt",
    "nobnt": "jaxili_cross_nobnt",
    "bnt_pct1": "jaxili_cross_bnt_pct1",
    "bnt_pct5": "jaxili_cross_bnt_pct5",
    "nobnt_pct1": "jaxili_cross_nobnt_pct1",
    "auto_zm_bnt": "jaxili_auto_zm_bnt",
    "auto_zm_nobnt": "jaxili_auto_zm_nobnt",
}


def plot_regime(regime: str, n_examples: int = 4096):
    sub = REGIME_CACHES.get(regime, f"jaxili_cross_{regime}")
    cache = ROOT / sub / "cache" / "l1_train.npz"
    print(f"[{regime}] loading {cache} ...")
    d = np.load(cache)
    x = d["x"]  # (N, 2000), float64
    print(f"[{regime}]   x: {x.shape} {x.dtype}")
    rng = np.random.default_rng(0)
    idx = rng.choice(x.shape[0], size=min(n_examples, x.shape[0]), replace=False)
    sub = x[idx]
    cube = reshape_dv(sub)  # (N, n_chan, n_scales, n_l1_bins)
    n_chan = cube.shape[1]
    chan_names = channel_names(n_chan)
    mean = cube.mean(0)
    std = cube.std(0)
    p16 = np.percentile(cube, 16, axis=0)
    p84 = np.percentile(cube, 84, axis=0)

    fig, axes = plt.subplots(n_chan, 1, figsize=(13, 1.6 * n_chan), sharex=True)
    if n_chan == 1:
        axes = [axes]
    xs = np.arange(N_SCALES * N_L1_BINS)
    for c, ax in enumerate(axes):
        m = mean[c].reshape(-1)
        lo = p16[c].reshape(-1)
        hi = p84[c].reshape(-1)
        s = std[c].reshape(-1)
        color = "C0" if c < N_AUTO else "C3"
        ax.plot(xs, m, color=color, lw=1.2, label="mean")
        ax.fill_between(xs, lo, hi, color=color, alpha=0.18, lw=0, label="16/84 pctl")
        ax.fill_between(xs, m - s, m + s, color=color, alpha=0.10, lw=0, label="±1σ")
        for k in range(1, N_SCALES):
            ax.axvline(k * N_L1_BINS, color="grey", lw=0.5, ls="--", alpha=0.6)
        ax.set_ylabel(chan_names[c], fontsize=9)
        ax.tick_params(labelsize=8)
        if c == 0:
            ax.legend(loc="upper right", fontsize=7, ncol=3)
        ax.set_xlim(0, N_SCALES * N_L1_BINS)
    axes[-1].set_xlabel("datavector index (5 scales × 40 SNR bins, scale boundaries dashed)")
    fig.suptitle(
        f"L1 datavectors fed to NPE — {regime} ({n_chan} chan, {sub.shape[0]} examples)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = OUT / f"datavectors_{regime}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[{regime}] wrote {out}")

    # Also print sanity stats for each channel
    print(f"[{regime}] per-channel stats (across {sub.shape[0]} examples, all bins):")
    for c in range(n_chan):
        v = cube[:, c]
        print(f"  {chan_names[c]:>9s}  mean={v.mean(): .3e}  std={v.std(): .3e}  "
              f"min={v.min(): .3e}  max={v.max(): .3e}  frac_zero={(v == 0).mean():.3f}")


def main():
    import sys

    regimes = sys.argv[1:] or ("bnt", "nobnt")
    for regime in regimes:
        plot_regime(regime)


if __name__ == "__main__":
    main()
