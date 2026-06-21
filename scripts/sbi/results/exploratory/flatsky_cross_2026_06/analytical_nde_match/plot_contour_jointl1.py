#!/usr/bin/env python3
"""Posterior contour overlay: jointl1 (new Q1 winner) vs l1+product (prev best) vs CNN.
Trains each arm's sbi_lens RealNVP 4x128 on its 10-D compressed cache, samples at a matched
observation (default noiseless mean-obs), draws a filled GetDist triangle (Om,s8,w0).
Draw order = loosest UNDER, tightest ON TOP (Andreas's rule): l1+product → CNN → jointl1.
Output: contour_jointl1_vs_l1product_vs_cnn.{png,pdf}
"""
import argparse
import sys
from types import SimpleNamespace
import numpy as np
from pathlib import Path

REPO = "/mnt/home/tersenov/software/cnn_sbi"; SBI = f"{REPO}/scripts/sbi"
ROOT = f"{SBI}/results/exploratory/flatsky_cross_2026_06"
HERE = Path(ROOT) / "analytical_nde_match"

# draw order: first = UNDER (loosest), last = ON TOP (tightest)
ORDER = ["l1product", "cnn", "jointl1"]
ARMS = {
    "l1product": dict(label="l1+product → RealNVP", color="#d62728",
                      cache=f"{HERE}/l1product_vmim_s41/cache", prefix="l1",
                      fid=f"{HERE}/l1product_vmim_s41/fiducial_summaries.npz"),
    "cnn":       dict(label="CNN ResNet18 → RealNVP", color="#1f77b4",
                      cache=f"{ROOT}/cnn_phase/arch_sweep_2026_06_13/cnn_resnet18_s41/cache",
                      prefix="cnn",
                      fid=f"{ROOT}/cnn_phase/arch_sweep_2026_06_13/fidsumm_resnet18.npz"),
    "jointl1":   dict(label="joint l1 → RealNVP", color="#2ca02c",
                      cache=f"{HERE}/jointl1_nobnt/cache", prefix="l1",
                      fid=f"{HERE}/jointl1_nobnt/fiducial_summaries.npz"),
}
FLOW = SimpleNamespace(flow_total_steps=50000, flow_batch_size=128, flow_lr_init=1e-3,
                       flow_lr_end=1e-5, flow_save_every=2000, flow_patience=20,
                       flow_grad_clip=1.0, flow_weight_decay=1e-4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda-visible-devices", default="1")
    ap.add_argument("--m-samples", type=int, default=20000)
    ap.add_argument("--nde-seed", type=int, default=41)
    ap.add_argument("--obs-mode", choices=["patch", "mean"], default="mean")
    ap.add_argument("--max-perm", type=int, default=50)
    a = ap.parse_args()

    sys.path.insert(0, REPO); sys.path.insert(0, SBI)
    from train_jaxili_from_compressed import setup_env, compute_fom3, marginal_stats
    setup_env(a.cuda_visible_devices)
    import jax
    from npe_l1norm_cross_jaxili_nbody_tomo import preprocess_summaries, filter_zero_variance_bins
    from train_nde_from_compressed import train_sbilens_realnvp

    results = {}
    truth = None
    for key in ORDER:
        cfg = ARMS[key]
        cdir = Path(cfg["cache"])
        tr = np.load(cdir / f"{cfg['prefix']}_train.npz")
        theta_tr = tr["theta"].astype(np.float32); x_tr_raw = tr["x"].astype(np.float64)
        x_va_raw = np.load(cdir / f"{cfg['prefix']}_val.npz")["x"].astype(np.float64)
        tr_p, _, _, mean, std = preprocess_summaries(
            x_tr_raw, x_va_raw[:1], x_va_raw[:1], summary_transform="none", clip_value=None)
        mask, _ = filter_zero_variance_bins(tr_p, min_variance=1e-12, verbose=False)
        x_tr = tr_p[:, mask].astype(np.float32); dim = x_tr.shape[1]

        fz = np.load(cfg["fid"])
        S = fz["S"].astype(np.float64); perm = fz["perm"]
        sel = np.where(perm < a.max_perm)[0] if a.obs_mode == "mean" else np.array([0])
        idx = int(sel[0])
        if truth is None:
            t = next((fz[k] for k in ("truth", "theta") if k in fz.files), None)
            if t is not None:
                truth = (t[idx] if np.ndim(t) == 2 else t)[:3]
        _, _, fid_p, _, _ = preprocess_summaries(
            x_tr_raw, x_va_raw[:1], S, summary_transform="none", clip_value=None, mean=mean, std=std)
        x_obs_all = fid_p[:, mask].astype(np.float32)
        x_single = x_obs_all[sel].mean(0) if a.obs_mode == "mean" else x_obs_all[idx]

        out = HERE / f"_tmp_flow_jl1_{key}"; out.mkdir(parents=True, exist_ok=True)
        samplers = train_sbilens_realnvp(4, 128, theta_tr, x_tr, [a.nde_seed], a.m_samples, dim, out, FLOW)
        seed, fn = samplers[0]
        k = jax.random.PRNGKey(seed * 100003 + idx)
        ps = np.asarray(fn(x_single, k)); ps = ps[np.all(np.isfinite(ps), 1)]
        ms = marginal_stats(ps); f3 = compute_fom3(ps)["fom3"]
        results[key] = dict(samples=ps[:, :3], fom3=f3,
                            sig=[ms["sigma"][p] for p in ("Omega_m", "sigma_8", "w_0")],
                            label=cfg["label"], color=cfg["color"])
        print(f"[{key}] FoM3={f3:.0f} sig={results[key]['sig']}", flush=True)

    from getdist import MCSamples, plots
    names = ["Om", "s8", "w0"]; labels = [r"\Omega_m", r"\sigma_8", "w_0"]
    mcs = [MCSamples(samples=results[k]["samples"], names=names, labels=labels,
                     label=results[k]["label"]) for k in ORDER]
    g = plots.get_subplot_plotter(width_inch=9)
    g.settings.legend_fontsize = 12; g.settings.axes_fontsize = 11; g.settings.lab_fontsize = 14
    markers = {n: float(truth[i]) for i, n in enumerate(names)} if truth is not None else None
    g.triangle_plot(mcs, filled=True, colors=[results[k]["color"] for k in ORDER],
                    legend_labels=[results[k]["label"] for k in ORDER], markers=markers)
    lines = ["FoM3 / $\\sigma(\\Omega_m,\\sigma_8,w_0)$:"]
    for k in ("jointl1", "cnn", "l1product"):
        r = results[k]
        lines.append(f"  {r['label'].split(' →')[0]}: {r['fom3']:.0f}  "
                     f"{r['sig'][0]:.3f}/{r['sig'][1]:.3f}/{r['sig'][2]:.3f}")
    g.fig.text(0.58, 0.93, "\n".join(lines), fontsize=10, va="top",
               bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.92))
    obs_desc = ("noiseless mean of ~9000 fiducial patches" if a.obs_mode == "mean"
                else "fiducial patch")
    g.fig.suptitle(f"Posterior overlay — {obs_desc}  (truth = crosshairs)", fontsize=12, y=1.0)
    for ext in ("png", "pdf"):
        g.export(str(HERE / f"contour_jointl1_vs_l1product_vs_cnn.{ext}"))
    print("wrote contour_jointl1_vs_l1product_vs_cnn.png")


if __name__ == "__main__":
    main()
