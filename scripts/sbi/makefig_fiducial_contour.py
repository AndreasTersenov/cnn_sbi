#!/usr/bin/env python3
"""Contour at the NOISE-AVERAGED fiducial data vector (the Fisher-forecast-analog, deterministic).

obs = mean of the fiducial summaries over all 9000 obs (perm<50 x 180 patches) -> no patch choice.
CNN auto-only sbi_lens RealNVP 4x128 vs L1+product jaxili MAF (pooled 3 seeds). NB: the mean summary
is lower-variance than training draws, so this is a BEST-CASE (tightest) contour, slightly OOD for
the NDE; report the per-patch median FoM3 (3139/2875) as the headline scalar and this as the clean
illustrative corner. Prints the FoM3 so we can flag if it is pathologically tight.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

REPO = "/mnt/home/tersenov/software/cnn_sbi"; SBI = f"{REPO}/scripts/sbi"
CNNP = f"{SBI}/results/exploratory/flatsky_cross_2026_06/cnn_phase"
L1P = f"{SBI}/results/exploratory/flatsky_cross_2026_06"
CNN_CACHE = f"{CNNP}/cnn_none_s41/cache"; CNN_FID = f"{CNNP}/fiducial_summaries/fiducial_summaries_none.npz"
L1_CACHE = f"{L1P}/l1_matrix/l1_product_cache/flat_local_product"
L1_FID = f"{L1P}/gate_c/lc2st/fiducial_summaries_product.npz"
FIGS = f"{CNNP}/nde_sweep_2026_06_13/figs"; TRUTH = np.array([0.26, 0.84, -1.0])


def prep_fid(cache_dir, prefix, fid_npz, transform, clip, min_var, max_perm=50):
    from npe_l1norm_cross_jaxili_nbody_tomo import preprocess_summaries, filter_zero_variance_bins
    tr = np.load(Path(cache_dir) / f"{prefix}_train.npz"); va = np.load(Path(cache_dir) / f"{prefix}_val.npz")
    theta_tr = tr["theta"].astype(np.float32)
    x_tr_raw = tr["x"].astype(np.float64); x_va_raw = va["x"].astype(np.float64)
    clipv = clip if clip > 0 else None
    tr_p, _, _, mean, std = preprocess_summaries(
        x_tr_raw, x_va_raw[:1], x_va_raw[:1], summary_transform=transform, clip_value=clipv)
    mask, _ = filter_zero_variance_bins(tr_p, min_variance=min_var, verbose=False)
    x_tr = tr_p[:, mask].astype(np.float32)
    fz = np.load(fid_npz); sel = fz["perm"] < max_perm
    mean_S = fz["S"][sel].astype(np.float64).mean(0, keepdims=True)   # noise-averaged fiducial vector
    _, _, obs_p, _, _ = preprocess_summaries(
        x_tr_raw, x_va_raw[:1], mean_S, summary_transform=transform, clip_value=clipv, mean=mean, std=std)
    return theta_tr, x_tr, obs_p[:, mask].astype(np.float32)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="41,42,43"); ap.add_argument("--m-samples", type=int, default=4000)
    ap.add_argument("--cuda-visible-devices", default="0"); a = ap.parse_args()
    sys.path.insert(0, REPO); sys.path.insert(0, SBI)
    from train_jaxili_from_compressed import setup_env, compute_fom3
    setup_env(a.cuda_visible_devices)
    import jax
    from train_nde_from_compressed import train_sbilens_realnvp, train_jaxili_family
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]; M = a.m_samples
    out = Path(FIGS); out.mkdir(parents=True, exist_ok=True)
    flow_kw = dict(flow_total_steps=50000, flow_batch_size=128, flow_lr_init=1e-3, flow_lr_end=1e-5,
                   flow_save_every=2000, flow_patience=20, flow_grad_clip=1.0, flow_weight_decay=1e-4)
    jax_kw = dict(epochs=50000, batch_size=256, learning_rate=1e-4, warmup_steps=100, decay_steps=10000)
    ns = argparse.Namespace(**flow_kw, **jax_kw)

    def pooled(samplers, x_obs):
        out_ = []
        for seed, fn in samplers:
            out_.append(np.asarray(fn(x_obs, jax.random.PRNGKey(seed * 100003 + 999999 % 100003))))
        ps = np.concatenate(out_, 0); return ps[np.all(np.isfinite(ps), 1)]

    print("######## CNN sbi_lens RealNVP 4x128 ########", flush=True)
    th, x, obs = prep_fid(CNN_CACHE, "cnn", CNN_FID, "none", 0.0, 1e-12)
    s_cnn = pooled(train_sbilens_realnvp(4, 128, th, x, seeds, M, x.shape[1], out / "ck_cnn", ns), obs)
    print("######## L1+product jaxili MAF ########", flush=True)
    th, x, obs = prep_fid(L1_CACHE, "l1", L1_FID, "log1p-zscore", 5.0, 1e-5)
    s_l1 = pooled(train_jaxili_family("jaxili_maf", 5, 50, th, x, seeds, M, out / "ck_l1", ns), obs)
    f_cnn, f_l1 = compute_fom3(s_cnn)["fom3"], compute_fom3(s_l1)["fom3"]
    print(f"FIDUCIAL-vector FoM3: CNN {f_cnn:.0f}  L1 {f_l1:.0f}  (per-patch medians are 3139/2875)", flush=True)
    np.savez(out / "fiducial_contour_samples.npz", cnn=s_cnn[:, :3], l1=s_l1[:, :3],
             truth=TRUTH, fom_cnn=f_cnn, fom_l1=f_l1)

    from getdist import MCSamples, plots
    names = ["Omega_m", "sigma_8", "w_0"]; labels = [r"\Omega_m", r"\sigma_8", r"w_0"]
    mc_l1 = MCSamples(samples=s_l1[:, :3], names=names, labels=labels,
                      label=f"L1+product, MAF (FoM3={f_l1:.0f})")
    mc_cnn = MCSamples(samples=s_cnn[:, :3], names=names, labels=labels,
                       label=f"CNN auto-only, RealNVP (FoM3={f_cnn:.0f})")
    g = plots.get_subplot_plotter(); g.settings.legend_fontsize = 12
    g.triangle_plot([mc_l1, mc_cnn], names, filled=True, colors=["#d62728", "#1f77b4"],
                    legend_loc="upper right", markers={n: float(t) for n, t in zip(names, TRUTH)})
    for ext in ("pdf", "png"):
        g.export(str(out / f"corner_fiducial_datavector.{ext}"))
    print(f"[done] wrote corner_fiducial_datavector  CNN {f_cnn:.0f} vs L1 {f_l1:.0f}", flush=True)


if __name__ == "__main__":
    main()
