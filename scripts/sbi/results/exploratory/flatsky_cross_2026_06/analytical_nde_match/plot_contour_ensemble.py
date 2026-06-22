#!/usr/bin/env python3
"""Posterior contour overlay: l1+product vs joint ℓ1 (compressor-ENSEMBLE, the calibrated arm) vs
CNN, for --basis {nobnt,bnt}. Single arms train one flow; the ensemble joint ℓ1 trains its 3
compressor-seed flows and POOLS samples at the mean-obs (each compressor sees the obs through its
own compression). Filled GetDist triangle (Om,s8,w0), tightest contour on top.
Output: contour_ensemble_{basis}_3arm.{png,pdf}
"""
import argparse
import sys
from types import SimpleNamespace
import numpy as np
from pathlib import Path

REPO = "/mnt/home/tersenov/software/cnn_sbi"; SBI = f"{REPO}/scripts/sbi"
ROOT = f"{SBI}/results/exploratory/flatsky_cross_2026_06"
HERE = Path(ROOT) / "analytical_nde_match"
CNNP = f"{ROOT}/cnn_phase"

def C(d):  # cache + fid pair
    return d
ARMS = {
 "nobnt": [
   dict(label="l1+product → RealNVP", color="#d62728", prefix="l1",
        caches=[f"{HERE}/l1product_vmim_s41/cache"], fids=[f"{HERE}/l1product_vmim_s41/fiducial_summaries.npz"]),
   dict(label="joint ℓ1 (ensemble) → RealNVP", color="#2ca02c", prefix="l1",
        caches=[f"{HERE}/jointl1_nobnt/cache", f"{HERE}/jointl1_nobnt_s42/cache", f"{HERE}/jointl1_nobnt_s43/cache"],
        fids=[f"{HERE}/jointl1_nobnt/fiducial_summaries.npz", f"{HERE}/jointl1_nobnt_s42/fiducial_summaries.npz", f"{HERE}/jointl1_nobnt_s43/fiducial_summaries.npz"]),
   dict(label="CNN ResNet18 → RealNVP", color="#1f77b4", prefix="cnn",
        caches=[f"{CNNP}/arch_sweep_2026_06_13/cnn_resnet18_s41/cache"], fids=[f"{CNNP}/arch_sweep_2026_06_13/fidsumm_resnet18.npz"]),
 ],
 "bnt": [
   dict(label="l1+product BNT → RealNVP", color="#d62728", prefix="l1",
        caches=[f"{HERE}/l1product_bnt_vmim_s41/cache"], fids=[f"{HERE}/l1product_bnt_vmim_s41/fiducial_summaries.npz"]),
   dict(label="joint ℓ1 BNT (ensemble) → RealNVP", color="#2ca02c", prefix="l1",
        caches=[f"{HERE}/jointl1_bnt/cache", f"{HERE}/jointl1_bnt_s42/cache", f"{HERE}/jointl1_bnt_s43/cache"],
        fids=[f"{HERE}/jointl1_bnt/fiducial_summaries.npz", f"{HERE}/jointl1_bnt_s42/fiducial_summaries.npz", f"{HERE}/jointl1_bnt_s43/fiducial_summaries.npz"]),
   dict(label="CNN ResNet18 BNT → RealNVP", color="#1f77b4", prefix="cnn",
        caches=[f"{CNNP}/bnt_resnet18_2026_06_14/cnn_resnet18_bnt_s41/cache"], fids=[f"{CNNP}/bnt_resnet18_2026_06_14/fidsumm_bnt_resnet18_s41.npz"]),
 ],
}
FLOW = SimpleNamespace(flow_total_steps=50000, flow_batch_size=128, flow_lr_init=1e-3,
                       flow_lr_end=1e-5, flow_save_every=2000, flow_patience=20,
                       flow_grad_clip=1.0, flow_weight_decay=1e-4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", choices=["nobnt", "bnt"], required=True)
    ap.add_argument("--cuda-visible-devices", default="2")
    ap.add_argument("--m-samples", type=int, default=20000)
    ap.add_argument("--nde-seed", type=int, default=41)
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
    for cfg in ARMS[a.basis]:
        pooled = []
        for ci, (cache, fid) in enumerate(zip(cfg["caches"], cfg["fids"])):
            cdir = Path(cache)
            tr = np.load(cdir / f"{cfg['prefix']}_train.npz")
            theta_tr = tr["theta"].astype(np.float32); x_tr_raw = tr["x"].astype(np.float64)
            x_va_raw = np.load(cdir / f"{cfg['prefix']}_val.npz")["x"].astype(np.float64)
            tr_p, _, _, mean, std = preprocess_summaries(
                x_tr_raw, x_va_raw[:1], x_va_raw[:1], summary_transform="none", clip_value=None)
            mask, _ = filter_zero_variance_bins(tr_p, min_variance=1e-12, verbose=False)
            x_tr = tr_p[:, mask].astype(np.float32); dim = x_tr.shape[1]
            fz = np.load(fid); S = fz["S"].astype(np.float64); perm = fz["perm"]
            sel = np.where(perm < a.max_perm)[0]
            if truth is None:
                t = next((fz[k] for k in ("truth", "theta") if k in fz.files), None)
                truth = (t[sel[0]] if np.ndim(t) == 2 else t)[:3] if t is not None else None
            _, _, fid_p, _, _ = preprocess_summaries(
                x_tr_raw, x_va_raw[:1], S[sel], summary_transform="none", clip_value=None, mean=mean, std=std)
            x_single = fid_p[:, mask].astype(np.float32).mean(0)
            out = HERE / f"_tmp_ce_{a.basis}_{cfg['prefix']}_{ci}"; out.mkdir(parents=True, exist_ok=True)
            samplers = train_sbilens_realnvp(4, 128, theta_tr, x_tr, [a.nde_seed], a.m_samples, dim, out, FLOW)
            seed, fn = samplers[0]
            k = jax.random.PRNGKey(seed * 100003 + ci)
            ps = np.asarray(fn(x_single, k)); pooled.append(ps[np.all(np.isfinite(ps), 1)])
        ps = np.concatenate(pooled, 0)
        ms = marginal_stats(ps); f3 = compute_fom3(ps)["fom3"]
        results[cfg["label"]] = dict(samples=ps[:, :3], fom3=f3, color=cfg["color"],
                                     sig=[ms["sigma"][q] for q in ("Omega_m", "sigma_8", "w_0")])
        print(f"[{cfg['label']}] FoM3={f3:.0f} sig={results[cfg['label']]['sig']}", flush=True)

    # draw order: lowest FoM3 (largest contour) first/under, highest on top
    order = sorted(results, key=lambda L: results[L]["fom3"])
    from getdist import MCSamples, plots
    names = ["Om", "s8", "w0"]; labels = [r"\Omega_m", r"\sigma_8", "w_0"]
    mcs = [MCSamples(samples=results[L]["samples"], names=names, labels=labels, label=L) for L in order]
    g = plots.get_subplot_plotter(width_inch=9)
    g.settings.legend_fontsize = 11; g.settings.axes_fontsize = 11; g.settings.lab_fontsize = 14
    markers = {n: float(truth[i]) for i, n in enumerate(names)} if truth is not None else None
    g.triangle_plot(mcs, filled=True, colors=[results[L]["color"] for L in order],
                    legend_labels=order, markers=markers)
    lines = ["FoM3 / $\\sigma(\\Omega_m,\\sigma_8,w_0)$:"]
    for L in sorted(results, key=lambda L: -results[L]["fom3"]):
        r = results[L]
        lines.append(f"  {L.split(' →')[0]}: {r['fom3']:.0f}  {r['sig'][0]:.3f}/{r['sig'][1]:.3f}/{r['sig'][2]:.3f}")
    g.fig.text(0.57, 0.95, "\n".join(lines), fontsize=9.5, va="top",
               bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.92))
    tt = "no BNT" if a.basis == "nobnt" else "BNT"
    g.fig.suptitle(f"Posterior overlay ({tt}) — noiseless mean-obs (truth = crosshairs)", fontsize=12, y=1.0)
    for ext in ("png", "pdf"):
        g.export(str(HERE / f"contour_ensemble_{a.basis}_3arm.{ext}"))
    print(f"wrote contour_ensemble_{a.basis}_3arm.png")


if __name__ == "__main__":
    main()
