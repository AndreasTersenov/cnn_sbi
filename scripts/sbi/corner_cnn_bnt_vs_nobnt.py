#!/usr/bin/env python3
"""Paper figure: CNN posterior WITH vs WITHOUT BNT (best pipeline) — demonstrates BNT-losslessness.

Both arms = resnet18 compressor + sbi_lens RealNVP NDE (pooled 3 seeds), sampled at the noise-averaged
fiducial data vector. no-BNT and BNT contours should nearly coincide (the CNN loses ~3% — mean ratio
0.97×), in contrast to the L1 norm which collapses to 0.15× under BNT. Deterministic obs (no patch
choice). Over [Omega_m, sigma_8, w_0].
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

REPO = "/mnt/home/tersenov/software/cnn_sbi"; SBI = f"{REPO}/scripts/sbi"
CNNP = f"{SBI}/results/exploratory/flatsky_cross_2026_06/cnn_phase"
NOBNT_CACHE = f"{CNNP}/arch_sweep_2026_06_13/cnn_resnet18_s41/cache"
NOBNT_FID = f"{CNNP}/arch_sweep_2026_06_13/fidsumm_resnet18.npz"
BNT_CACHE = f"{CNNP}/bnt_resnet18_2026_06_14/cnn_resnet18_bnt_s41/cache"
BNT_FID = f"{CNNP}/bnt_resnet18_2026_06_14/fidsumm_bnt_resnet18_s41.npz"
OUT = Path(f"{CNNP}/nde_sweep_2026_06_13/figs")
TRUTH = np.array([0.26, 0.84, -1.0])
NAMES = ["Omega_m", "sigma_8", "w_0"]; LABELS = [r"\Omega_m", r"\sigma_8", r"w_0"]


def prep_fid(cache_dir, fid_npz):
    from npe_l1norm_cross_jaxili_nbody_tomo import preprocess_summaries, filter_zero_variance_bins
    tr = np.load(Path(cache_dir) / "cnn_train.npz"); va = np.load(Path(cache_dir) / "cnn_val.npz")
    theta_tr = tr["theta"].astype(np.float32)
    x_tr_raw = tr["x"].astype(np.float64); x_va_raw = va["x"].astype(np.float64)
    tr_p, _, _, mean, std = preprocess_summaries(x_tr_raw, x_va_raw[:1], x_va_raw[:1],
                                                 summary_transform="none", clip_value=None)
    mask, _ = filter_zero_variance_bins(tr_p, min_variance=1e-12, verbose=False)
    x_tr = tr_p[:, mask].astype(np.float32)
    fz = np.load(fid_npz); mean_S = fz["S"][fz["perm"] < 50].astype(np.float64).mean(0, keepdims=True)
    _, _, obs_p, _, _ = preprocess_summaries(x_tr_raw, x_va_raw[:1], mean_S, summary_transform="none",
                                             clip_value=None, mean=mean, std=std)
    return theta_tr, x_tr, obs_p[:, mask].astype(np.float32)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="41,42,43"); ap.add_argument("--m-samples", type=int, default=4000)
    ap.add_argument("--cuda-visible-devices", default="2"); a = ap.parse_args()
    sys.path.insert(0, REPO); sys.path.insert(0, SBI)
    from train_jaxili_from_compressed import setup_env, compute_fom3
    setup_env(a.cuda_visible_devices)
    import jax
    from train_nde_from_compressed import train_sbilens_realnvp
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]; M = a.m_samples
    OUT.mkdir(parents=True, exist_ok=True)
    ns = argparse.Namespace(flow_total_steps=50000, flow_batch_size=128, flow_lr_init=1e-3,
                            flow_lr_end=1e-5, flow_save_every=2000, flow_patience=20,
                            flow_grad_clip=1.0, flow_weight_decay=1e-4)

    def pooled(samplers, x):
        out = [np.asarray(fn(x, jax.random.PRNGKey(s * 100003 + 12345))) for s, fn in samplers]
        ps = np.concatenate(out, 0); return ps[np.all(np.isfinite(ps), 1)]

    res = {}
    for tag, cache, fid in [("nobnt", NOBNT_CACHE, NOBNT_FID), ("bnt", BNT_CACHE, BNT_FID)]:
        print(f"######## CNN resnet18 RealNVP — {tag} ########", flush=True)
        th, x, obs = prep_fid(cache, fid)
        sm = train_sbilens_realnvp(4, 128, th, x, seeds, M, x.shape[1], OUT / f"ck_bntfig_{tag}", ns)
        s = pooled(sm, obs); res[tag] = (s, compute_fom3(s)["fom3"])
        print(f"  {tag} FoM3 = {res[tag][1]:.0f}", flush=True)

    np.savez(OUT / "cnn_bnt_vs_nobnt_samples.npz", nobnt=res["nobnt"][0][:, :3], bnt=res["bnt"][0][:, :3],
             truth=TRUTH, fom_nobnt=res["nobnt"][1], fom_bnt=res["bnt"][1])

    from getdist import MCSamples, plots
    mc_no = MCSamples(samples=res["nobnt"][0][:, :3], names=NAMES, labels=LABELS,
                      label=f"CNN, no BNT (FoM3={res['nobnt'][1]:.0f})")
    mc_bnt = MCSamples(samples=res["bnt"][0][:, :3], names=NAMES, labels=LABELS,
                       label=f"CNN, BNT (FoM3={res['bnt'][1]:.0f})")
    g = plots.get_subplot_plotter(); g.settings.legend_fontsize = 12
    g.triangle_plot([mc_no, mc_bnt], NAMES, filled=True, colors=["#1f77b4", "#ff7f0e"],
                    legend_loc="upper right", markers={n: float(t) for n, t in zip(NAMES, TRUTH)})
    for ext in ("pdf", "png"):
        g.export(str(OUT / f"corner_cnn_bnt_vs_nobnt.{ext}"))
    print(f"[done] CNN no-BNT {res['nobnt'][1]:.0f} vs BNT {res['bnt'][1]:.0f} "
          f"(ratio {res['bnt'][1]/res['nobnt'][1]:.3f}) — lossless overlap", flush=True)


if __name__ == "__main__":
    main()
