#!/usr/bin/env python
"""Quick look: (1) overlay corner of the two CNN smoke posteriors, (2) example
10-channel patches from the 10deg fiducial cache. CPU-only."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "results/exploratory/definitive_comparison_10deg"
FID = "results/exploratory/cross_maps_campaign/full_sphere_cache_fiducial_10deg/nobnt/obs/cosmo_fiducial_perm0.npz"
OUT = f"{BASE}/plots"
os.makedirs(OUT, exist_ok=True)
TRUTH = [0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493]
NAMES = ["Om", "s8", "w0", "h0", "ns", "Ob"]
LABELS = [r"\Omega_m", r"\sigma_8", "w_0", "h_0", "n_s", r"\Omega_b"]


def corner():
    from getdist import MCSamples, plots

    def load(tag):
        p = np.load(f"{BASE}/smoke_{tag}/posterior.npy")
        return np.asarray(p).reshape(-1, 6)

    p_ac, p_ao = load("autocross"), load("autoonly")
    s_ac = MCSamples(samples=p_ac, names=NAMES, labels=LABELS, label="CNN auto+cross")
    s_ao = MCSamples(samples=p_ao, names=NAMES, labels=LABELS, label="CNN auto-only")
    g = plots.get_subplot_plotter(width_inch=9)
    g.settings.alpha_filled_add = 0.6
    g.triangle_plot(
        [s_ac, s_ao], filled=True,
        contour_colors=["#1f77b4", "#d62728"],
        markers={n: TRUTH[i] for i, n in enumerate(NAMES)},
    )
    g.export(f"{OUT}/smoke_corner.pdf")
    g.export(f"{OUT}/smoke_corner.png")
    print(f"wrote {OUT}/smoke_corner.png")


def patches():
    d = np.load(FID, allow_pickle=True)
    P = np.asarray(d["patches"])      # (180,80,80,10)
    C = np.asarray(d["patch_centers"])  # (180,2) lon,lat
    idxs = [0, 60, 90, 150]
    chan = [r"$\kappa_1$", r"$\kappa_2$", r"$\kappa_3$", r"$\kappa_4$",
            r"$\kappa_{1\times2}$", r"$\kappa_{1\times3}$", r"$\kappa_{1\times4}$",
            r"$\kappa_{2\times3}$", r"$\kappa_{2\times4}$", r"$\kappa_{3\times4}$"]
    nr, nc = len(idxs), 10
    fig, ax = plt.subplots(nr, nc, figsize=(1.55 * nc, 1.75 * nr))
    for r, pi in enumerate(idxs):
        for c in range(nc):
            img = P[pi, :, :, c]
            s = img.std() or 1.0
            ax[r, c].imshow(img, cmap="RdBu_r", vmin=-3 * s, vmax=3 * s,
                            origin="lower", interpolation="nearest")
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(chan[c], fontsize=10)
            if c == 0:
                ax[r, c].set_ylabel(f"patch {pi}\nlat {C[pi,1]:+.0f}°\nrms {P[pi,:,:,0].std():.1e}",
                                    fontsize=8)
    fig.suptitle(
        "Example 10° patches (fiducial cosmology) — 4 auto (κ_i) + 6 cross (κ_{i×j}); "
        "each panel scaled to its own ±3σ",
        fontsize=11, y=1.005,
    )
    fig.tight_layout()
    fig.savefig(f"{OUT}/example_patches.png", dpi=130, bbox_inches="tight")
    fig.savefig(f"{OUT}/example_patches.pdf", bbox_inches="tight")
    print(f"wrote {OUT}/example_patches.png")


if __name__ == "__main__":
    patches()
    corner()
