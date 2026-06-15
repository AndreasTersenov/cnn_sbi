#!/usr/bin/env python3
"""BNT contours in the matched best-NDE setup: l1+product and CNN ResNet18, each in no-BNT and BNT
space, all VMIM->10-D->sbi_lens RealNVP, sampled at the noiseless mean observation.

M3 expectation: per-channel wavelet l1 COLLAPSES under BNT (the per-channel l1 discards the
cross-channel info before the VMIM MLP can recover it), while the channel-mixing CNN is ~lossless.

Outputs (mean-obs, GetDist triangles, Om/s8/w0):
  contour_bnt_4way_l1_cnn.png      — all four (l1/CNN x noBNT/BNT)
  contour_bnt_l1_collapse.png      — l1 no-BNT vs l1 BNT (the collapse)
  contour_bnt_l1_vs_cnn.png        — BNT-space: l1-BNT vs CNN-BNT (mirror of the no-BNT headline)
Prints FoM3/sigma + BNT/noBNT ratios.
"""
import argparse, sys
from types import SimpleNamespace
import numpy as np
from pathlib import Path

REPO = "/mnt/home/tersenov/software/cnn_sbi"; SBI = f"{REPO}/scripts/sbi"
ROOT = f"{SBI}/results/exploratory/flatsky_cross_2026_06"
HERE = Path(ROOT) / "analytical_nde_match"

ARMS = {  # key: (label, color, filled, cache, prefix, fid)
    "l1_nobnt":  dict(label="l1+product (no-BNT)", color="#d62728", filled=True,
                      cache=f"{HERE}/l1product_vmim_s41/cache", prefix="l1",
                      fid=f"{HERE}/l1product_vmim_s41/fiducial_summaries.npz"),
    "l1_bnt":    dict(label="l1+product (BNT)", color="#f1948a", filled=False,
                      cache=f"{HERE}/l1product_bnt_vmim_s41/cache", prefix="l1",
                      fid=f"{HERE}/l1product_bnt_vmim_s41/fiducial_summaries.npz"),
    "cnn_nobnt": dict(label="CNN ResNet18 (no-BNT)", color="#1f77b4", filled=True,
                      cache=f"{ROOT}/cnn_phase/arch_sweep_2026_06_13/cnn_resnet18_s41/cache", prefix="cnn",
                      fid=f"{ROOT}/cnn_phase/arch_sweep_2026_06_13/fidsumm_resnet18.npz"),
    "cnn_bnt":   dict(label="CNN ResNet18 (BNT)", color="#85c1e9", filled=False,
                      cache=f"{ROOT}/cnn_phase/bnt_resnet18_2026_06_14/cnn_resnet18_bnt_s41/cache", prefix="cnn",
                      fid=f"{ROOT}/cnn_phase/bnt_resnet18_2026_06_14/fidsumm_bnt_resnet18_s41.npz"),
}
FLOW = SimpleNamespace(flow_total_steps=50000, flow_batch_size=128, flow_lr_init=1e-3, flow_lr_end=1e-5,
                       flow_save_every=2000, flow_patience=20, flow_grad_clip=1.0, flow_weight_decay=1e-4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda-visible-devices", default="2")
    ap.add_argument("--max-perm", type=int, default=50)
    ap.add_argument("--m-samples", type=int, default=20000)
    ap.add_argument("--nde-seed", type=int, default=41)
    a = ap.parse_args()
    sys.path.insert(0, REPO); sys.path.insert(0, SBI)
    from train_jaxili_from_compressed import setup_env, compute_fom3, marginal_stats
    setup_env(a.cuda_visible_devices)
    import jax
    from npe_l1norm_cross_jaxili_nbody_tomo import preprocess_summaries, filter_zero_variance_bins
    from train_nde_from_compressed import train_sbilens_realnvp

    res, truth = {}, None
    for key, cfg in ARMS.items():
        cdir = Path(cfg["cache"]); pre = cfg["prefix"]
        tr = np.load(cdir / f"{pre}_train.npz"); theta_tr = tr["theta"].astype(np.float32)
        x_tr_raw = tr["x"].astype(np.float64); x_va_raw = np.load(cdir / f"{pre}_val.npz")["x"].astype(np.float64)
        tr_p, _, _, mean, std = preprocess_summaries(x_tr_raw, x_va_raw[:1], x_va_raw[:1],
                                                     summary_transform="none", clip_value=None)
        mask, _ = filter_zero_variance_bins(tr_p, min_variance=1e-12, verbose=False)
        x_tr = tr_p[:, mask].astype(np.float32); dim = x_tr.shape[1]
        fz = np.load(cfg["fid"]); S = fz["S"].astype(np.float64); perm = fz["perm"]
        sel = np.where(perm < a.max_perm)[0]; idx0 = int(sel[0])
        if truth is None:
            t = next((fz[k] for k in ("truth", "theta") if k in fz.files), None)
            if t is not None:
                truth = (t[idx0] if np.ndim(t) == 2 else t)[:3]
        _, _, fid_p, _, _ = preprocess_summaries(x_tr_raw, x_va_raw[:1], S, summary_transform="none",
                                                 clip_value=None, mean=mean, std=std)
        x_obs = fid_p[:, mask].astype(np.float32)
        x_single = x_obs[sel].mean(0)
        seed, fn = train_sbilens_realnvp(4, 128, theta_tr, x_tr, [a.nde_seed], a.m_samples, dim,
                                         HERE / f"_tmp_bnt_{key}", FLOW)[0]
        ps = np.asarray(fn(x_single, jax.random.PRNGKey(seed * 100003 + idx0)))
        ps = ps[np.all(np.isfinite(ps), 1)]
        ms = marginal_stats(ps); f3 = compute_fom3(ps)["fom3"]
        res[key] = dict(samples=ps[:, :3], fom3=f3,
                        sig=[ms["sigma"][p] for p in ("Omega_m", "sigma_8", "w_0")], **cfg)
        print(f"[{key}] FoM3 {f3:.0f}  sig {res[key]['sig'][0]:.3f}/{res[key]['sig'][1]:.3f}/{res[key]['sig'][2]:.3f}", flush=True)

    print(f"\nBNT/noBNT FoM3 ratio: l1 {res['l1_bnt']['fom3']/res['l1_nobnt']['fom3']:.2f}x  "
          f"CNN {res['cnn_bnt']['fom3']/res['cnn_nobnt']['fom3']:.2f}x", flush=True)

    from getdist import MCSamples, plots
    names = ["Om", "s8", "w0"]; labels = [r"\Omega_m", r"\sigma_8", "w_0"]
    mk = {n: float(truth[i]) for i, n in enumerate(names)} if truth is not None else None

    def mcs(key):
        r = res[key]
        return MCSamples(samples=r["samples"], names=names, labels=labels, label=r["label"])

    def draw(keys, fname, title):
        g = plots.get_subplot_plotter(width_inch=9)
        g.settings.legend_fontsize = 11; g.settings.axes_fontsize = 11; g.settings.lab_fontsize = 14
        g.triangle_plot([mcs(k) for k in keys], filled=[ARMS[k]["filled"] for k in keys],
                        colors=[ARMS[k]["color"] for k in keys],
                        legend_labels=[res[k]["label"] for k in keys], markers=mk)
        box = "\n".join(f"{res[k]['label']}: FoM3 {res[k]['fom3']:.0f}  "
                        f"sig {res[k]['sig'][0]:.3f}/{res[k]['sig'][1]:.3f}/{res[k]['sig'][2]:.3f}" for k in keys)
        g.fig.text(0.56, 0.96, box, fontsize=9, va="top",
                   bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.92))
        g.fig.suptitle(title, fontsize=12, y=1.0)
        for ext in ("png", "pdf"):
            g.export(str(HERE / f"{fname}.{ext}"))
        print("wrote", fname + ".png", flush=True)

    # smaller (more-informative) contour drawn last/on-top within each pair: noBNT is tighter -> on top
    draw(["l1_bnt", "l1_nobnt"], "contour_bnt_l1_collapse",
         "l1+product: BNT (filled out) vs no-BNT — per-channel l1 COLLAPSES under BNT (mean-obs)")
    draw(["l1_bnt", "cnn_bnt"], "contour_bnt_l1_vs_cnn",
         "BNT space: l1+product (balloons) vs CNN ResNet18 (lossless) — mean-obs")
    draw(["l1_bnt", "l1_nobnt", "cnn_bnt", "cnn_nobnt"], "contour_bnt_4way_l1_cnn",
         "no-BNT vs BNT, l1 vs CNN (mean-obs): l1 collapses under BNT, CNN is lossless")


if __name__ == "__main__":
    main()
