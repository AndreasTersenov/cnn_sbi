#!/usr/bin/env python3
"""Overlay posterior contours of the new best l1 arm vs the best CNN arm at the SAME fiducial patch.
Trains each arm's sbi_lens RealNVP (the exact gated config) on its compressed 10-D cache, samples
both at a matched (perm,patch), and draws a filled GetDist triangle (Om, s8, w0) with truth markers.

l1  arm = l1+product VMIM(s41) -> sbi_lens RealNVP 4x128   (FoM3 ~3146 @n=1000)
CNN arm = ResNet18  VMIM(s41) -> sbi_lens RealNVP 4x128   (FoM3 3293)
Output: contour_overlay_l1_vs_cnn.{png,pdf}
"""
import argparse
import sys
from types import SimpleNamespace
import numpy as np
from pathlib import Path

REPO = "/mnt/home/tersenov/software/cnn_sbi"; SBI = f"{REPO}/scripts/sbi"
ROOT = f"{SBI}/results/exploratory/flatsky_cross_2026_06"
HERE = Path(ROOT) / "analytical_nde_match"

ARMS = {
    "cnn": dict(label="CNN ResNet18 → RealNVP", color="#1f77b4",
                cache=f"{ROOT}/cnn_phase/arch_sweep_2026_06_13/cnn_resnet18_s41/cache",
                prefix="cnn",
                fid=f"{ROOT}/cnn_phase/arch_sweep_2026_06_13/fidsumm_resnet18.npz"),
    "l1":  dict(label="l1+product → RealNVP", color="#d62728",
                cache=f"{HERE}/l1product_vmim_s41/cache",
                prefix="l1",
                fid=f"{HERE}/l1product_vmim_s41/fiducial_summaries.npz"),
}
FLOW = SimpleNamespace(flow_total_steps=50000, flow_batch_size=128, flow_lr_init=1e-3,
                       flow_lr_end=1e-5, flow_save_every=2000, flow_patience=20,
                       flow_grad_clip=1.0, flow_weight_decay=1e-4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda-visible-devices", default="2")
    ap.add_argument("--perm", type=int, default=16)
    ap.add_argument("--patch", type=int, default=23)
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
    for key, cfg in ARMS.items():
        cdir = Path(cfg["cache"])
        tr = np.load(cdir / f"{cfg['prefix']}_train.npz")
        theta_tr = tr["theta"].astype(np.float32); x_tr_raw = tr["x"].astype(np.float64)
        x_va_raw = np.load(cdir / f"{cfg['prefix']}_val.npz")["x"].astype(np.float64)
        tr_p, _, _, mean, std = preprocess_summaries(
            x_tr_raw, x_va_raw[:1], x_va_raw[:1], summary_transform="none", clip_value=None)
        mask, _ = filter_zero_variance_bins(tr_p, min_variance=1e-12, verbose=False)
        x_tr = tr_p[:, mask].astype(np.float32); dim = x_tr.shape[1]

        fz = np.load(cfg["fid"])
        S = fz["S"].astype(np.float64); perm = fz["perm"]; patch = fz["patch"]
        if a.obs_mode == "mean":
            sel = np.where(perm < a.max_perm)[0]
            obs_label = f"mean of {sel.size} patches (perm<{a.max_perm}), noiseless"
        else:
            sel = np.where((perm == a.perm) & (patch == a.patch))[0]
            if sel.size == 0:
                sel = np.array([0]); print(f"[{key}] (perm,patch)=({a.perm},{a.patch}) not found; row 0")
            obs_label = f"perm{a.perm}, patch{a.patch}"
        idx = int(sel[0])
        if truth is None:
            t = next((fz[k] for k in ("truth", "theta") if k in fz.files), None)
            if t is not None:
                truth = (t[idx] if np.ndim(t) == 2 else t)[:3]
        _, _, fid_p, _, _ = preprocess_summaries(
            x_tr_raw, x_va_raw[:1], S, summary_transform="none", clip_value=None, mean=mean, std=std)
        x_obs_all = fid_p[:, mask].astype(np.float32)
        x_single = x_obs_all[sel].mean(0) if a.obs_mode == "mean" else x_obs_all[idx]

        out = HERE / f"_tmp_flow_{key}"; out.mkdir(parents=True, exist_ok=True)
        samplers = train_sbilens_realnvp(4, 128, theta_tr, x_tr, [a.nde_seed], a.m_samples, dim, out, FLOW)
        seed, fn = samplers[0]
        k = jax.random.PRNGKey(seed * 100003 + idx)
        ps = np.asarray(fn(x_single, k)); ps = ps[np.all(np.isfinite(ps), 1)]
        ms = marginal_stats(ps); f3 = compute_fom3(ps)["fom3"]
        results[key] = dict(samples=ps[:, :3], fom3=f3,
                            sig=[ms["sigma"][p] for p in ("Omega_m", "sigma_8", "w_0")],
                            label=cfg["label"], color=cfg["color"])
        print(f"[{key}] obs [{obs_label}] FoM3={f3:.0f} sig={results[key]['sig']}", flush=True)

    # ---- GetDist overlay ----
    from getdist import MCSamples, plots
    names = ["Om", "s8", "w0"]; labels = [r"\Omega_m", r"\sigma_8", "w_0"]
    mcs = []
    for key in ("l1", "cnn"):                          # l1 first (drawn UNDER, larger); CNN ON TOP (smaller)
        r = results[key]
        mcs.append(MCSamples(samples=r["samples"], names=names, labels=labels, label=r["label"]))
    g = plots.get_subplot_plotter(width_inch=9)
    g.settings.legend_fontsize = 12; g.settings.axes_fontsize = 11; g.settings.lab_fontsize = 14
    markers = {n: float(truth[i]) for i, n in enumerate(names)} if truth is not None else None
    g.triangle_plot(mcs, filled=True, colors=[results["l1"]["color"], results["cnn"]["color"]],
                    legend_labels=[results["l1"]["label"], results["cnn"]["label"]], markers=markers)
    txt = (f"FoM3:  CNN {results['cnn']['fom3']:.0f}   l1+product {results['l1']['fom3']:.0f}\n"
           f"$\\sigma(\\Omega_m,\\sigma_8,w_0)$:\n"
           f"  CNN {results['cnn']['sig'][0]:.3f}/{results['cnn']['sig'][1]:.3f}/{results['cnn']['sig'][2]:.3f}\n"
           f"  l1   {results['l1']['sig'][0]:.3f}/{results['l1']['sig'][1]:.3f}/{results['l1']['sig'][2]:.3f}")
    g.fig.text(0.60, 0.93, txt, fontsize=11, va="top",
               bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.92))
    obs_desc = ("noiseless mean of ~9000 fiducial patches" if a.obs_mode == "mean"
                else f"fiducial patch perm{a.perm}, patch{a.patch}")
    g.fig.suptitle(f"Posterior overlay — {obs_desc}  (truth = crosshairs)", fontsize=12, y=1.0)
    suffix = "meanobs" if a.obs_mode == "mean" else f"perm{a.perm}_patch{a.patch}"
    for ext in ("png", "pdf"):
        g.export(str(HERE / f"contour_overlay_{suffix}_l1_vs_cnn.{ext}"))
    print(f"wrote contour_overlay_{suffix}_l1_vs_cnn.png")


if __name__ == "__main__":
    main()
