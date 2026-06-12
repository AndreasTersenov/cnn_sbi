#!/usr/bin/env python3
"""Corner overlay: joint statistics (pair2d, joint wavelet l1; dequantized noBNT arms) vs
the wavelet l1 auto+product arm, at the TYPICAL fiducial obs (perm 16, patch 23).

Retrains each arm's NDE with the population sweep's exact recipe (3 MAF seeds, pooled) —
checkpoint reload is broken at >1000-dim conditioning (jaxili truncation gotcha), retrain
is ~80 s/seed. Samples 33334/seed -> ~100k pooled. Science-params corner (Om, s8, w0).
"""
import os
import sys
import time
from pathlib import Path
import numpy as np

REPO = "/mnt/home/tersenov/software/cnn_sbi"
SBI = f"{REPO}/scripts/sbi"
sys.path.insert(0, REPO); sys.path.insert(0, SBI)
FC = Path(SBI) / "results/exploratory/flatsky_cross_2026_06"
OM = FC / "overnight_menu"
OUT = OM / "corners"
FIGS = OM / "figures"
TYP_PERM, TYP_PATCH = 16, 23
M_PER_SEED, SEEDS = 33334, (41, 42, 43)


def arm_inputs(variant):
    import flatsky_cross_l1 as fxl
    if variant == "nobnt":
        fb = np.load(FC / "fiducial_both_datavectors.npz")
        cols = fxl.op_feature_columns("product", 4, 200)
        arms = [
            ("wavelet $\\ell_1$ auto+product",
             FC / "l1_matrix/l1_product_cache/flat_local_product",
             fb["x"][:, cols], fb["perm"], fb["patch"], fb["truth"]),
        ]
        joint = (("joint PDF (pairwise 2D)", "pair2dq_nobnt"),
                 ("joint wavelet $\\ell_1$", "jointl1q_nobnt"))
    else:
        bz = np.load(FC / "bnt_campaign/fiducial_summaries/fiducial_summaries_l1_product.npz")
        arms = [
            ("BNT wavelet $\\ell_1$ auto+product",
             FC / "bnt_campaign/l1_matrix/l1_product_cache/flat_local_product_bnt",
             bz["S"], bz["perm"], bz["patch"], bz["truth"]),
        ]
        joint = (("BNT joint PDF (pairwise 2D)", "pair2dq_bnt"),
                 ("BNT joint wavelet $\\ell_1$", "jointl1q_bnt"))
    for label, name in joint:
        fz = np.load(OM / name / "fiducial_summaries.npz")
        arms.append((label, OM / name / "cache", fz["S"], fz["perm"], fz["patch"],
                     fz["truth"]))
    return arms


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("nobnt", "bnt"), default="nobnt")
    a = ap.parse_args()
    from train_jaxili_from_compressed import setup_env
    setup_env(os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
    import jax, jax.numpy as jnp
    from jaxili.inference import NPE
    from npe_l1norm_cross_jaxili_nbody_tomo import (
        train_with_nan_retry, preprocess_summaries, filter_zero_variance_bins)

    OUT.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
    results = {}
    truth = None
    for label, cache, S, perm, patch, tr_truth in arm_inputs(a.variant):
        truth = tr_truth
        idx = np.where((perm == TYP_PERM) & (patch == TYP_PATCH))[0]
        assert idx.size == 1, f"typical obs not found for {label}"
        tr = np.load(Path(cache) / "l1_train.npz")
        va = np.load(Path(cache) / "l1_val.npz")
        x_tr_raw = tr["x"].astype(np.float64)
        theta_tr = tr["theta"].astype(np.float32)
        tr_p, _, _, mean, std = preprocess_summaries(
            x_tr_raw, va["x"][:1].astype(np.float64), va["x"][:1].astype(np.float64),
            summary_transform="log1p-zscore", clip_value=5.0)
        mask, _ = filter_zero_variance_bins(tr_p, min_variance=1e-5, verbose=False)
        x_tr = tr_p[:, mask].astype(np.float32)
        _, _, obs_p, _, _ = preprocess_summaries(
            x_tr_raw, va["x"][:1].astype(np.float64), S[idx].astype(np.float64),
            summary_transform="log1p-zscore", clip_value=5.0, mean=mean, std=std)
        x_obs = jnp.asarray(obs_p[:, mask].astype(np.float32)[0])
        print(f"[{label}] dim {x_tr.shape[1]}, obs row {int(idx[0])}", flush=True)

        params, data = jnp.asarray(theta_tr), jnp.asarray(x_tr)
        pooled = []
        for seed in SEEDS:
            t0 = time.time()
            sk = jax.random.PRNGKey(seed + 1)
            inf = NPE().append_simulations(params, data, key=sk)
            inf, _m, _d = train_with_nan_retry(
                inf, str((OUT / f"ckpt_{label[:8]}_{seed}").resolve()), 50000, 1e-4, 256,
                100, 10000, params, data, sk)
            post = inf.build_posterior()
            k = jax.random.PRNGKey(seed * 100003 + int(idx[0]))
            ps = np.asarray(post.sample(x=x_obs, num_samples=M_PER_SEED, key=k))
            pooled.append(ps[np.all(np.isfinite(ps), 1)])
            print(f"  seed {seed}: {time.time()-t0:.0f}s, {pooled[-1].shape[0]} samples",
                  flush=True)
        ps = np.concatenate(pooled, 0)
        results[label] = ps
        np.save(OUT / f"{label.replace(' ', '_').replace('$', '').replace('\\\\', '')}.npy", ps)
        print(f"[{label}] pooled {ps.shape}", flush=True)

    # ---- GetDist corner (science params) ----
    import matplotlib
    matplotlib.use("Agg")
    from getdist import MCSamples, plots
    names = ["Om", "s8", "w0"]
    labels_g = [r"\Omega_m", r"\sigma_8", r"w_0"]
    colors = ["#D55E00", "#0072B2", "#009E73"]
    mcs = [MCSamples(samples=ps[:, :3], names=names, labels=labels_g, label=lbl,
                     settings={"smooth_scale_2D": 0.35, "smooth_scale_1D": 0.35})
           for lbl, ps in results.items()]
    filled = [True] * len(mcs)
    if a.variant == "bnt":
        ref = OUT / "wavelet_\\ell_1_auto+product.npy"
        if ref.exists():
            mcs.append(MCSamples(samples=np.load(ref)[:, :3], names=names, labels=labels_g,
                                 label="no-BNT $\\ell_1$ auto+product (ref)",
                                 settings={"smooth_scale_2D": 0.35, "smooth_scale_1D": 0.35}))
            colors.append("#999999")
            filled.append(False)
    g = plots.get_subplot_plotter(width_inch=6.0)
    g.settings.legend_fontsize = 10
    g.triangle_plot(mcs, filled=filled, contour_colors=colors, legend_loc="upper right")
    for i, nm in enumerate(names):
        for ax in g.subplots[:, i].ravel():
            if ax is not None:
                ax.axvline(truth[i], color="k", lw=0.8, ls=":")
    stem = "corner_joint_vs_l1product" + ("_bnt" if a.variant == "bnt" else "")
    for ext in ("png", "pdf"):
        g.export(str(FIGS / f"{stem}.{ext}"))
    print(f"wrote {FIGS}/{stem}.png", flush=True)


if __name__ == "__main__":
    main()
