#!/usr/bin/env python3
"""Representative-patch contour overlay: BEST L1-based vs BEST CNN-based posterior.

  CNN auto-only  -> sbi_lens RealNVP 4x128  (the calibrated 3139 arm)
  L1+product     -> jaxili MAF              (the 2875 arm)

Both retrained in-process (3 seeds, pooled) and sampled at the canonical typical obs
(perm16/patch23). Each summary is preprocessed with ITS OWN convention (CNN none/min-var1e-12;
L1 log1p-zscore/clip5/min-var1e-5). Per-patch FoM3 printed + shown in the legend. GetDist filled
triangle over [Omega_m, sigma_8, w_0]. One-off figure for the CNN-optimization writeup.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np

REPO = "/mnt/home/tersenov/software/cnn_sbi"
SBI = f"{REPO}/scripts/sbi"
CNNP = f"{SBI}/results/exploratory/flatsky_cross_2026_06/cnn_phase"
L1P = f"{SBI}/results/exploratory/flatsky_cross_2026_06"

CNN_CACHE = f"{CNNP}/cnn_none_s41/cache"
CNN_FID = f"{CNNP}/fiducial_summaries/fiducial_summaries_none.npz"
L1_CACHE = f"{L1P}/l1_matrix/l1_product_cache/flat_local_product"
L1_FID = f"{L1P}/gate_c/lc2st/fiducial_summaries_product.npz"
TRUTH = np.array([0.26, 0.84, -1.0])
PERM, PATCH = 16, 23


def prep(cache_dir, prefix, fid_npz, transform, clip, min_var):
    """Preprocess train + the rep obs identically to the sweep. Returns theta_tr, x_tr, x_obs."""
    from npe_l1norm_cross_jaxili_nbody_tomo import preprocess_summaries, filter_zero_variance_bins
    tr = np.load(Path(cache_dir) / f"{prefix}_train.npz")
    va = np.load(Path(cache_dir) / f"{prefix}_val.npz")
    theta_tr = tr["theta"].astype(np.float32)
    x_tr_raw = tr["x"].astype(np.float64); x_va_raw = va["x"].astype(np.float64)
    clipv = clip if clip > 0 else None
    tr_p, _, _, mean, std = preprocess_summaries(
        x_tr_raw, x_va_raw[:1], x_va_raw[:1], summary_transform=transform, clip_value=clipv)
    mask, _ = filter_zero_variance_bins(tr_p, min_variance=min_var, verbose=False)
    x_tr = tr_p[:, mask].astype(np.float32)
    fz = np.load(fid_npz)
    row = int(np.where((fz["perm"] == PERM) & (fz["patch"] == PATCH))[0][0])
    S = fz["S"][row:row + 1].astype(np.float64)
    _, _, obs_p, _, _ = preprocess_summaries(
        x_tr_raw, x_va_raw[:1], S, summary_transform=transform, clip_value=clipv, mean=mean, std=std)
    x_obs = obs_p[:, mask].astype(np.float32)[0]
    return theta_tr, x_tr, x_obs


def sample_pooled(samplers, x_obs, jax):
    import numpy as _np
    pooled = []
    for seed, fn in samplers:
        k = jax.random.PRNGKey(seed * 100003 + PERM * 1000 + PATCH)
        pooled.append(_np.asarray(fn(x_obs, k)))
    ps = _np.concatenate(pooled, 0)
    return ps[_np.all(_np.isfinite(ps), 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="41,42,43")
    ap.add_argument("--m-samples", type=int, default=4000)
    ap.add_argument("--out-dir", default=f"{CNNP}/nde_sweep_2026_06_13/figs")
    ap.add_argument("--cuda-visible-devices", default="0")
    a = ap.parse_args()
    sys.path.insert(0, REPO); sys.path.insert(0, SBI)
    from train_jaxili_from_compressed import setup_env, compute_fom3
    setup_env(a.cuda_visible_devices)
    import jax
    from train_nde_from_compressed import train_sbilens_realnvp, train_jaxili_family
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    M = a.m_samples
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    flow_kw = dict(flow_total_steps=50000, flow_batch_size=128, flow_lr_init=1e-3, flow_lr_end=1e-5,
                   flow_save_every=2000, flow_patience=20, flow_grad_clip=1.0, flow_weight_decay=1e-4)
    jax_kw = dict(epochs=50000, batch_size=256, learning_rate=1e-4, warmup_steps=100, decay_steps=10000)

    # CNN: sbi_lens RealNVP 4x128
    print("######## CNN sbi_lens RealNVP 4x128 ########", flush=True)
    th_c, x_c, obs_c = prep(CNN_CACHE, "cnn", CNN_FID, "none", 0.0, 1e-12)
    args_c = argparse.Namespace(**flow_kw, **jax_kw)
    samp_c = train_sbilens_realnvp(4, 128, th_c, x_c, seeds, M, x_c.shape[1], out / "ck_cnn", args_c)
    ps_cnn = sample_pooled(samp_c, obs_c, jax)
    fom_cnn = compute_fom3(ps_cnn)["fom3"]
    print(f"  CNN rep-patch FoM3 = {fom_cnn:.0f}  (n={len(ps_cnn)})", flush=True)

    # L1: jaxili MAF
    print("######## L1+product jaxili MAF ########", flush=True)
    th_l, x_l, obs_l = prep(L1_CACHE, "l1", L1_FID, "log1p-zscore", 5.0, 1e-5)
    args_l = argparse.Namespace(**flow_kw, **jax_kw)
    samp_l = train_jaxili_family("jaxili_maf", 5, 50, th_l, x_l, seeds, M, out / "ck_l1", args_l)
    ps_l1 = sample_pooled(samp_l, obs_l, jax)
    fom_l1 = compute_fom3(ps_l1)["fom3"]
    print(f"  L1 rep-patch FoM3 = {fom_l1:.0f}  (n={len(ps_l1)})", flush=True)

    np.savez(out / "corner_l1_vs_cnn_samples.npz",
             cnn=ps_cnn[:, :3], l1=ps_l1[:, :3], truth=TRUTH,
             fom_cnn=fom_cnn, fom_l1=fom_l1, perm=PERM, patch=PATCH)

    # GetDist filled triangle over the 3 science params
    from getdist import MCSamples, plots
    names = ["Omega_m", "sigma_8", "w_0"]
    labels = [r"\Omega_m", r"\sigma_8", r"w_0"]
    mc_cnn = MCSamples(samples=ps_cnn[:, :3], names=names, labels=labels,
                       label=f"CNN auto-only, RealNVP (FoM3={fom_cnn:.0f})")
    mc_l1 = MCSamples(samples=ps_l1[:, :3], names=names, labels=labels,
                      label=f"L1+product, MAF (FoM3={fom_l1:.0f})")
    g = plots.get_subplot_plotter()
    g.settings.legend_fontsize = 13
    g.triangle_plot([mc_l1, mc_cnn], names, filled=True,
                    colors=["#d62728", "#1f77b4"], legend_loc="upper right",
                    markers={n: float(t) for n, t in zip(names, TRUTH)})
    for ext in ("pdf", "png"):
        g.export(str(out / f"corner_l1_vs_cnn_best_nde.{ext}"))
    print(f"[done] wrote {out}/corner_l1_vs_cnn_best_nde.{{pdf,png}}  "
          f"CNN {fom_cnn:.0f} vs L1 {fom_l1:.0f} @ perm{PERM}/patch{PATCH}", flush=True)


if __name__ == "__main__":
    main()
