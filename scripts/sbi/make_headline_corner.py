#!/usr/bin/env python3
"""4-arm corner overlay (L1/CNN × auto/auto+cross) at ONE representative (non-polar)
patch — the contour version of the typical-patch headline.

auto+cross posteriors are reused from the per-patch tarp dumps (so the a+c pair
matches reversal_A); the two auto-only arms are trained+sampled fresh at the SAME
(perm, patch). 3-seed pooled, 3 science params.
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
DC = REPO / "results/exploratory/definitive_comparison"
FF = DC / "fiducial_full200"; TPP = FF / "tarp_per_patch"
FID = np.array([0.26, 0.84, -1.0]); NAMES = ["Om", "s8", "w0"]; LABS = [r"\Omega_m", r"\sigma_8", r"w_0"]


def arm_S(name):
    z = np.load(FF / "summaries" / f"{name}_S.npz")
    return z["S"], z["perm"], z["patch"]


def main():
    sys.path.insert(0, str(REPO))
    from train_jaxili_from_compressed import setup_env
    setup_env("1")
    import jax, jax.numpy as jnp
    from jaxili.inference import NPE
    from npe_l1norm_cross_jaxili_nbody_tomo import (
        train_with_nan_retry, preprocess_summaries, filter_zero_variance_bins)
    from getdist import MCSamples, plots

    _, perm, patch = arm_S("l1_autocross")
    # reproduce the tarp_per_patch sel order to recover the representative patch
    rng = np.random.default_rng(7)
    p0 = np.where(patch == 0)[0]; p0t = p0[rng.choice(len(p0), 60, replace=False)]
    rest = np.setdiff1d(np.arange(len(patch)), p0t)
    popt = rest[rng.choice(len(rest), 200, replace=False)]
    sel = np.concatenate([popt, p0t])
    Lf = np.load(TPP / "coverage/l1_autocross/coverage_arrays.npz")["fom3"]
    is_p0 = np.array([patch[g] == 0 for g in sel]); pop = ~is_p0
    i_typ = np.arange(len(sel))[pop][np.argmin(np.abs(Lf[pop] - np.median(Lf[pop])))]
    gi = int(sel[i_typ]); P, J = int(perm[gi]), int(patch[gi])
    print(f"representative patch: perm {P} patch {J} (L1 a+c FoM3 {Lf[i_typ]:.0f})", flush=True)

    def dump_pool(arm):
        fs = sorted(glob.glob(str(TPP / "dumps" / arm / "seed_*" / "*" / "posterior_samples.npz")))
        return np.concatenate([np.load(f)["samples"][i_typ] for f in fs], 0)
    samples = {"L1 auto+cross": dump_pool("l1_autocross"),
               "CNN auto+cross": dump_pool("cnn_autocross")}

    def train_sample(cache, prefix, sname, transform, clip, minvar):
        tr = np.load(cache / f"{prefix}_train.npz")
        ttr = tr["theta"].astype(np.float32); xtr = tr["x"].astype(np.float64)
        Sa, pa, pj = arm_S(sname)
        g = int(np.where((pa == P) & (pj == J))[0][0])
        clipv = clip if clip > 0 else None
        trp, obsp, _, m, s = preprocess_summaries(xtr, xtr[:1], Sa[g:g+1],
                                                  summary_transform=transform, clip_value=clipv)
        mask, _ = filter_zero_variance_bins(trp, min_variance=minvar, verbose=False)
        xt = trp[:, mask].astype(np.float32); obs = obsp[:, mask][0].astype(np.float32)
        out = []
        for seed in (41, 42, 43):
            sk = jax.random.PRNGKey(seed + 1)
            inf = NPE().append_simulations(jnp.asarray(ttr), jnp.asarray(xt), key=sk)
            inf, _, _ = train_with_nan_retry(inf, str((FF / "ckpts_corner" / f"{sname}_s{seed}").resolve()),
                                             50000, 1e-4, 256, 100, 10000, jnp.asarray(ttr), jnp.asarray(xt), sk)
            post = inf.build_posterior()
            out.append(np.asarray(post.sample(x=jnp.asarray(obs), num_samples=2000,
                                              key=jax.random.PRNGKey(seed + 7))))
        return np.concatenate(out, 0)

    samples["L1 auto-only"] = train_sample(DC / "compressed/l1_autoonly_split70_dv", "l1",
                                           "l1_autoonly", "log1p-zscore", 5.0, 1e-5)
    samples["CNN auto-only"] = train_sample(DC / "phaseA_tfdata_2026_05_30/compressed/autoonly_s41", "cnn",
                                            "cnn_autoonly", "none", 0.0, 1e-12)

    order = ["L1 auto+cross", "CNN auto+cross", "L1 auto-only", "CNN auto-only"]
    cols = ["#C0392B", "#2471A3", "#E67E22", "#5DADE2"]
    mcs = []
    for k in order:
        x = samples[k][np.all(np.isfinite(samples[k]), 1)][:, :3]
        mcs.append(MCSamples(samples=x, names=NAMES, labels=LABS, label=k,
                             settings={"smooth_scale_2D": 0.35, "fine_bins_2D": 200}))
    g = plots.get_subplot_plotter(width_inch=8)
    g.settings.legend_fontsize = 13; g.settings.axes_labelsize = 15
    g.triangle_plot(mcs, filled=True, contour_colors=cols, legend_labels=order,
                    markers={NAMES[i]: FID[i] for i in range(3)})
    g.fig.suptitle(f"Definitive headline (contours) — representative patch (perm {P}, patch {J})\n"
                   "L1 a+c tightest; auto-only L1≈CNN; single patch (per-patch scatter applies)",
                   y=1.02, fontsize=12)
    out = FF / "figures" / "headline_corner_4arm_typical_patch.png"
    g.export(str(out)); print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
