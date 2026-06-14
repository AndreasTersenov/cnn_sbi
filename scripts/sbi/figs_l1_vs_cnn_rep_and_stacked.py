#!/usr/bin/env python3
"""Two population-faithful L1-vs-CNN contour figures from ONE training pass.

  CNN auto-only -> sbi_lens RealNVP 4x128 ;  L1+product -> jaxili MAF  (both pooled 3 seeds).
  fig1 (representative patch): contour at a patch where BOTH arms sit at their population medians
        (perm36/patch118; CNN~3139, L1~2875) -> the honest single-realization figure.
  fig2 (population-stacked): posteriors stacked over 30 patches spanning the population -> the
        realization-marginalized contour (its FoM3 is lower than the per-patch median by design,
        since it folds in between-patch scatter; the SHAPE comparison is the point).
Patches come from patch_selection.npz (built from the full-sweep per-patch FoM3).
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
FIGS = f"{CNNP}/nde_sweep_2026_06_13/figs"
TRUTH = np.array([0.26, 0.84, -1.0])


def prep_multi(cache_dir, prefix, fid_npz, transform, clip, min_var, patches):
    """Preprocess train + the obs summaries for a list of (perm,patch). Returns theta_tr, x_tr, {pp:x_obs}."""
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
    rows = [int(np.where((fz["perm"] == pm) & (fz["patch"] == pa))[0][0]) for pm, pa in patches]
    S = fz["S"][rows].astype(np.float64)
    _, _, obs_p, _, _ = preprocess_summaries(
        x_tr_raw, x_va_raw[:1], S, summary_transform=transform, clip_value=clipv, mean=mean, std=std)
    obs = obs_p[:, mask].astype(np.float32)
    return theta_tr, x_tr, {pp: obs[i] for i, pp in enumerate(patches)}


def pooled(samplers, x_obs, pp, jax):
    out = []
    for seed, fn in samplers:
        k = jax.random.PRNGKey(seed * 100003 + pp[0] * 1000 + pp[1])
        out.append(np.asarray(fn(x_obs, k)))
    ps = np.concatenate(out, 0)
    return ps[np.all(np.isfinite(ps), 1)]


def triangle(samples_by_arm, fom_by_arm, fname, title_suffix):
    from getdist import MCSamples, plots
    names = ["Omega_m", "sigma_8", "w_0"]; labels = [r"\Omega_m", r"\sigma_8", r"w_0"]
    mcs = []
    for arm, color, lab in [("l1", "#d62728", "L1+product, MAF"),
                            ("cnn", "#1f77b4", "CNN auto-only, RealNVP")]:
        s = samples_by_arm[arm][:, :3]
        mcs.append(MCSamples(samples=s, names=names, labels=labels,
                             label=f"{lab} (FoM3={fom_by_arm[arm]:.0f}{title_suffix})"))
    g = plots.get_subplot_plotter(); g.settings.legend_fontsize = 12
    g.triangle_plot(mcs, names, filled=True, colors=["#d62728", "#1f77b4"],
                    legend_loc="upper right", markers={n: float(t) for n, t in zip(names, TRUTH)})
    for ext in ("pdf", "png"):
        g.export(fname + "." + ext)
    print(f"  wrote {fname}.{{pdf,png}}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="41,42,43")
    ap.add_argument("--m-samples", type=int, default=3000)
    ap.add_argument("--cuda-visible-devices", default="0")
    a = ap.parse_args()
    sys.path.insert(0, REPO); sys.path.insert(0, SBI)
    from train_jaxili_from_compressed import setup_env, compute_fom3
    setup_env(a.cuda_visible_devices)
    import jax
    from train_nde_from_compressed import train_sbilens_realnvp, train_jaxili_family
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    M = a.m_samples
    out = Path(FIGS); out.mkdir(parents=True, exist_ok=True)

    sel = np.load(out / "patch_selection.npz")
    rep = (int(sel["rep"][0]), int(sel["rep"][1]))
    stack = [(int(p), int(q)) for p, q in sel["stack"]]
    patches = [rep] + [p for p in stack if p != rep]
    print(f"rep patch {rep} | {len(stack)} stack patches", flush=True)

    flow_kw = dict(flow_total_steps=50000, flow_batch_size=128, flow_lr_init=1e-3, flow_lr_end=1e-5,
                   flow_save_every=2000, flow_patience=20, flow_grad_clip=1.0, flow_weight_decay=1e-4)
    jax_kw = dict(epochs=50000, batch_size=256, learning_rate=1e-4, warmup_steps=100, decay_steps=10000)
    args_ns = argparse.Namespace(**flow_kw, **jax_kw)

    rep_s, stack_s = {}, {}
    # CNN
    print("######## CNN sbi_lens RealNVP 4x128 ########", flush=True)
    th, x, obs = prep_multi(CNN_CACHE, "cnn", CNN_FID, "none", 0.0, 1e-12, patches)
    sc = train_sbilens_realnvp(4, 128, th, x, seeds, M, x.shape[1], out / "ck_cnn", args_ns)
    rep_s["cnn"] = pooled(sc, obs[rep], rep, jax)
    stack_s["cnn"] = np.concatenate([pooled(sc, obs[p], p, jax) for p in stack], 0)
    # L1
    print("######## L1+product jaxili MAF ########", flush=True)
    th, x, obs = prep_multi(L1_CACHE, "l1", L1_FID, "log1p-zscore", 5.0, 1e-5, patches)
    sl = train_jaxili_family("jaxili_maf", 5, 50, th, x, seeds, M, out / "ck_l1", args_ns)
    rep_s["l1"] = pooled(sl, obs[rep], rep, jax)
    stack_s["l1"] = np.concatenate([pooled(sl, obs[p], p, jax) for p in stack], 0)

    fom_rep = {k: compute_fom3(rep_s[k])["fom3"] for k in rep_s}
    fom_stk = {k: compute_fom3(stack_s[k])["fom3"] for k in stack_s}
    print(f"REP  perm{rep[0]}/patch{rep[1]}: CNN {fom_rep['cnn']:.0f}  L1 {fom_rep['l1']:.0f}", flush=True)
    print(f"STACK ({len(stack)} patches): CNN {fom_stk['cnn']:.0f}  L1 {fom_stk['l1']:.0f}", flush=True)
    np.savez(out / "rep_and_stacked_samples.npz",
             rep_cnn=rep_s["cnn"][:, :3], rep_l1=rep_s["l1"][:, :3],
             stack_cnn=stack_s["cnn"][:, :3], stack_l1=stack_s["l1"][:, :3],
             rep=np.array(rep), truth=TRUTH, **{f"fom_{k}": v for k, v in
             {**{f"rep_{k}": fom_rep[k] for k in fom_rep}, **{f"stk_{k}": fom_stk[k] for k in fom_stk}}.items()})

    triangle(rep_s, fom_rep, str(out / "corner_rep_patch_both_median"), f", perm{rep[0]}/p{rep[1]}")
    triangle(stack_s, fom_stk, str(out / "corner_population_stacked"), ", stacked")
    print("[done] both figures written", flush=True)


if __name__ == "__main__":
    main()
