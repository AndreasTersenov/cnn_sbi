#!/usr/bin/env python3
"""Final figure set for the BEST CNN (resnet18 + sbi_lens RealNVP, FoM3 3326) vs best L1 (MAF, 2875).

Produces, with the resnet18 summaries as the CNN arm:
  corner_resnet18_rep_patch       — representative patch (both arms at their medians)
  corner_resnet18_fiducial        — noise-averaged fiducial data vector (conventional headline)
  corner_resnet18_stacked         — population-stacked (realization-marginalized)
  fom3_distribution_resnet18       — per-patch FoM3 violins (the quantitative claim)
Mirrors the plain-CNN figure set; CNN cache/fid swapped to resnet18, patches re-selected for its
medians. One training pass (CNN RealNVP + L1 MAF, 3 seeds).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/mnt/home/tersenov/software/cnn_sbi"; SBI = f"{REPO}/scripts/sbi"
CNNP = f"{SBI}/results/exploratory/flatsky_cross_2026_06/cnn_phase"
L1P = f"{SBI}/results/exploratory/flatsky_cross_2026_06"
ARCH = f"{CNNP}/arch_sweep_2026_06_13"
CNN_CACHE = f"{ARCH}/cnn_resnet18_s41/cache"
CNN_FID = f"{ARCH}/fidsumm_resnet18.npz"
CNN_PP = f"{ARCH}/cnn_resnet18_s41/readout_full/per_patch_metrics.npz"
L1_CACHE = f"{L1P}/l1_matrix/l1_product_cache/flat_local_product"
L1_FID = f"{L1P}/gate_c/lc2st/fiducial_summaries_product.npz"
L1_PP = f"{L1P}/population_sweep/flat_product/per_patch_metrics.npz"
OUT = Path(f"{CNNP}/nde_sweep_2026_06_13/figs")
TRUTH = np.array([0.26, 0.84, -1.0])
NAMES = ["Omega_m", "sigma_8", "w_0"]; LABELS = [r"\Omega_m", r"\sigma_8", r"w_0"]


def select_patches():
    def D(z):
        zz = np.load(z); return {(int(p), int(q)): float(f) for f, p, q in
                                 zip(zz["fom3"], zz["perm"], zz["patch"]) if np.isfinite(f)}
    C, L = D(CNN_PP), D(L1_PP); common = sorted(set(C) & set(L))
    cm, lm = np.median([C[k] for k in common]), np.median([L[k] for k in common])
    rep = min(common, key=lambda k: (C[k] / cm - 1) ** 2 + (L[k] / lm - 1) ** 2)
    order = sorted(common, key=lambda k: C[k]); stack = [order[i] for i in np.linspace(0, len(order) - 1, 30).astype(int)]
    return rep, stack, cm, lm, C, L


def prep_multi(cache_dir, prefix, fid_npz, transform, clip, min_var, patches):
    from npe_l1norm_cross_jaxili_nbody_tomo import preprocess_summaries, filter_zero_variance_bins
    tr = np.load(Path(cache_dir) / f"{prefix}_train.npz"); va = np.load(Path(cache_dir) / f"{prefix}_val.npz")
    theta_tr = tr["theta"].astype(np.float32)
    x_tr_raw = tr["x"].astype(np.float64); x_va_raw = va["x"].astype(np.float64)
    clipv = clip if clip > 0 else None
    tr_p, _, _, mean, std = preprocess_summaries(x_tr_raw, x_va_raw[:1], x_va_raw[:1],
                                                 summary_transform=transform, clip_value=clipv)
    mask, _ = filter_zero_variance_bins(tr_p, min_variance=min_var, verbose=False)
    x_tr = tr_p[:, mask].astype(np.float32)
    fz = np.load(fid_npz)
    # fiducial obs = patches + a synthetic "mean fiducial vector" sentinel ("FID")
    rows = [int(np.where((fz["perm"] == pm) & (fz["patch"] == pa))[0][0]) for pm, pa in patches]
    S = fz["S"][rows].astype(np.float64)
    sel = fz["perm"] < 50
    mean_S = fz["S"][sel].astype(np.float64).mean(0, keepdims=True)
    Sall = np.concatenate([S, mean_S], 0)
    _, _, obs_p, _, _ = preprocess_summaries(x_tr_raw, x_va_raw[:1], Sall, summary_transform=transform,
                                             clip_value=clipv, mean=mean, std=std)
    obs = obs_p[:, mask].astype(np.float32)
    d = {pp: obs[i] for i, pp in enumerate(patches)}; d["FID"] = obs[-1]
    return theta_tr, x_tr, d


def triangle(samples_by_arm, fom, fname, suffix):
    from getdist import MCSamples, plots
    mcs = []
    for arm, lab in [("l1", "L1+product, MAF"), ("cnn", "CNN auto-only, resnet18+RealNVP")]:
        mcs.append(MCSamples(samples=samples_by_arm[arm][:, :3], names=NAMES, labels=LABELS,
                             label=f"{lab} (FoM3={fom[arm]:.0f}{suffix})"))
    g = plots.get_subplot_plotter(); g.settings.legend_fontsize = 11
    g.triangle_plot(mcs, NAMES, filled=True, colors=["#d62728", "#1f77b4"], legend_loc="upper right",
                    markers={n: float(t) for n, t in zip(NAMES, TRUTH)})
    for ext in ("pdf", "png"):
        g.export(str(OUT / f"{fname}.{ext}"))
    print(f"  wrote {fname}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="41,42,43"); ap.add_argument("--m-samples", type=int, default=3000)
    ap.add_argument("--cuda-visible-devices", default="0"); a = ap.parse_args()
    sys.path.insert(0, REPO); sys.path.insert(0, SBI)
    from train_jaxili_from_compressed import setup_env, compute_fom3
    setup_env(a.cuda_visible_devices)
    import jax
    from train_nde_from_compressed import train_sbilens_realnvp, train_jaxili_family
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]; M = a.m_samples
    OUT.mkdir(parents=True, exist_ok=True)

    rep, stack, cm, lm, C, L = select_patches()
    print(f"rep {rep} CNN {C[rep]:.0f} L1 {L[rep]:.0f} | CNN med {cm:.0f} L1 med {lm:.0f}", flush=True)
    patches = [rep] + [p for p in stack if p != rep]
    flow_kw = dict(flow_total_steps=50000, flow_batch_size=128, flow_lr_init=1e-3, flow_lr_end=1e-5,
                   flow_save_every=2000, flow_patience=20, flow_grad_clip=1.0, flow_weight_decay=1e-4)
    jax_kw = dict(epochs=50000, batch_size=256, learning_rate=1e-4, warmup_steps=100, decay_steps=10000)
    ns = argparse.Namespace(**flow_kw, **jax_kw)

    def pooled(samplers, x, key_base):
        out = [np.asarray(fn(x, jax.random.PRNGKey(seed * 100003 + key_base))) for seed, fn in samplers]
        ps = np.concatenate(out, 0); return ps[np.all(np.isfinite(ps), 1)]

    arms = {}
    print("######## CNN resnet18 RealNVP ########", flush=True)
    th, x, obs = prep_multi(CNN_CACHE, "cnn", CNN_FID, "none", 0.0, 1e-12, patches)
    sc = train_sbilens_realnvp(4, 128, th, x, seeds, M, x.shape[1], OUT / "ck_cnn18", ns)
    arms["cnn"] = (sc, obs)
    print("######## L1 MAF ########", flush=True)
    th, x, obs = prep_multi(L1_CACHE, "l1", L1_FID, "log1p-zscore", 5.0, 1e-5, patches)
    sl = train_jaxili_family("jaxili_maf", 5, 50, th, x, seeds, M, OUT / "ck_l1", ns)
    arms["l1"] = (sl, obs)

    def key(pp):
        return pp[0] * 1000 + pp[1] if isinstance(pp, tuple) else 999999 % 100003
    for tag, fname, suffix, getobs in [
            ("rep", "corner_resnet18_rep_patch", f", perm{rep[0]}/p{rep[1]}", lambda o: o[rep]),
            ("fid", "corner_resnet18_fiducial", ", fiducial", lambda o: o["FID"])]:
        S = {arm: pooled(sm, getobs(ob), key(rep if tag == "rep" else (16, 23))) for arm, (sm, ob) in arms.items()}
        fom = {arm: compute_fom3(S[arm])["fom3"] for arm in S}
        print(f"{tag}: CNN {fom['cnn']:.0f}  L1 {fom['l1']:.0f}", flush=True)
        triangle(S, fom, fname, suffix)
    # stacked
    Sst = {arm: np.concatenate([pooled(sm, ob[p], key(p)) for p in stack], 0) for arm, (sm, ob) in arms.items()}
    fom = {arm: compute_fom3(Sst[arm])["fom3"] for arm in Sst}
    print(f"stacked: CNN {fom['cnn']:.0f}  L1 {fom['l1']:.0f}", flush=True)
    triangle(Sst, fom, "corner_resnet18_stacked", ", stacked")

    # FoM3 distribution (from per-patch, no training)
    cnn_pp = np.load(CNN_PP)["fom3"]; l1_pp = np.load(L1_PP)["fom3"]
    cnn_pp, l1_pp = cnn_pp[np.isfinite(cnn_pp)], l1_pp[np.isfinite(l1_pp)]
    n = min(len(cnn_pp), len(l1_pp)); frac = float(np.mean(cnn_pp[:n] > l1_pp[:n]))
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    parts = ax.violinplot([l1_pp, cnn_pp], positions=[0, 1], showextrema=False, widths=0.8)
    for pc, c in zip(parts["bodies"], ["#d62728", "#1f77b4"]):
        pc.set_facecolor(c); pc.set_alpha(0.35); pc.set_edgecolor(c)
    bp = ax.boxplot([l1_pp, cnn_pp], positions=[0, 1], widths=0.18, patch_artist=True, showfliers=False,
                    medianprops=dict(color="k", lw=1.6))
    for patch, c in zip(bp["boxes"], ["#d62728", "#1f77b4"]):
        patch.set_facecolor(c); patch.set_alpha(0.5)
    ax.set_yscale("log"); ax.set_xticks([0, 1])
    ax.set_xticklabels([f"L1+product\nMAF\n(median {np.median(l1_pp):.0f})",
                        f"CNN resnet18\nRealNVP\n(median {np.median(cnn_pp):.0f})"], fontsize=10)
    ax.set_ylabel(r"per-patch FoM3 $=1/\sqrt{\det C_3}$")
    ax.set_title(f"CNN tighter at {100*frac:.0f}% of patches  |  median ratio {np.median(cnn_pp)/np.median(l1_pp):.2f}×", fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fom3_distribution_resnet18.{ext}", dpi=200, bbox_inches="tight")
    print(f"  wrote fom3_distribution_resnet18 (CNN>L1 {100*frac:.0f}%)", flush=True)
    print("[done] all resnet18 figures written", flush=True)


if __name__ == "__main__":
    main()
