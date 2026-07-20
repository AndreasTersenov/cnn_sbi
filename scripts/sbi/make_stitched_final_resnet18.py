#!/usr/bin/env python3
"""FINAL stitched M1 figure (slide style): best CNN (resnet18 + RealNVP) vs best L1 (+product, MAF).

Follows the layout Andreas likes (cnn_phase/figs/stitched_cnn_vs_l1.pdf): a 3-param corner with a
FoM3 inset + legend in the empty upper triangle. Differences from the original:
  - CURRENT headline data: resnet18 CNN auto-only vs L1+product (not the outdated common-MAF run).
  - 3-bar inset: L1 auto (2405) -> L1+product (2875) -> CNN auto-only (3326), pooled 9000-obs median
    (the point: the CNN beats L1's BEST from autos alone, +15%).
  - SLIDE fonts (much larger than paper); Wong colourblind palette (CNN blue / L1 vermillion),
    L1 solid / CNN dashed contours.

The corner needs a one-time GPU re-sample (the resnet18 corner samples were never saved); this script
retrains the two flows at the fiducial obs, SAVES the samples to an npz, and builds the figure. On a
re-run it loads the npz and rebuilds instantly (no GPU). Pin GPU with --cuda-visible-devices (def 1).
"""
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/mnt/home/tersenov/software/cnn_sbi"; SBI = f"{REPO}/scripts/sbi"
sys.path.insert(0, REPO); sys.path.insert(0, SBI)

OUT = Path(f"{REPO}/talk_figures/_new_figs")
SAMP = OUT / "stitched_final_resnet18_samples.npz"
C_CNN, C_L1 = "#0072B2", "#D55E00"   # Wong: blue / vermillion

# population (pooled 3-seed, 9000-obs) median FoM3 — confirmed from per_patch_metrics
FOM_BARS = [("L1\nauto", C_L1, 2405), ("L1\n+product", C_L1, 2875), ("CNN\nauto-only", C_CNN, 3326)]

# slide font sizes
FS_TICK, FS_AXLABEL = 15, 24
FS_INSET_TITLE, FS_INSET_TICK, FS_INSET_YLAB, FS_INSET_XLAB = 13, 12, 16, 13
FS_LEGEND, LW_CONTOUR, WIDTH_INCH = 16, 2.2, 8.6


def compute_samples(gpu):
    import makefigs_best_cnn_resnet18 as mk
    from train_jaxili_from_compressed import setup_env, compute_fom3
    setup_env(gpu)
    import jax
    from train_nde_from_compressed import train_sbilens_realnvp, train_jaxili_family
    seeds, M = [41, 42, 43], 3000
    rep, stack, cm, lm, C, L = mk.select_patches()
    patches = [rep]   # prep_multi appends the "FID" mean-fiducial obs regardless
    flow_kw = dict(flow_total_steps=50000, flow_batch_size=128, flow_lr_init=1e-3, flow_lr_end=1e-5,
                   flow_save_every=2000, flow_patience=20, flow_grad_clip=1.0, flow_weight_decay=1e-4)
    jax_kw = dict(epochs=50000, batch_size=256, learning_rate=1e-4, warmup_steps=100, decay_steps=10000)
    ns = argparse.Namespace(**flow_kw, **jax_kw)

    def pooled(samplers, xobs, key_base):
        out = [np.asarray(fn(xobs, jax.random.PRNGKey(seed * 100003 + key_base))) for seed, fn in samplers]
        ps = np.concatenate(out, 0); return ps[np.all(np.isfinite(ps), 1)]

    print("######## CNN resnet18 RealNVP ########", flush=True)
    th, x, obs = mk.prep_multi(mk.CNN_CACHE, "cnn", mk.CNN_FID, "none", 0.0, 1e-12, patches)
    sc = train_sbilens_realnvp(4, 128, th, x, seeds, M, x.shape[1], OUT / "ck_cnn18_stitch", ns)
    cnn = pooled(sc, obs["FID"], 16 * 1000 + 23)
    print("######## L1 MAF ########", flush=True)
    th, x, obs = mk.prep_multi(mk.L1_CACHE, "l1", mk.L1_FID, "log1p-zscore", 5.0, 1e-5, patches)
    sl = train_jaxili_family("jaxili_maf", 5, 50, th, x, seeds, M, OUT / "ck_l1_stitch", ns)
    l1 = pooled(sl, obs["FID"], 16 * 1000 + 23)
    fom_cnn = compute_fom3(cnn)["fom3"]; fom_l1 = compute_fom3(l1)["fom3"]
    np.savez(SAMP, cnn=cnn[:, :3], l1=l1[:, :3], fom_cnn=fom_cnn, fom_l1=fom_l1)
    print(f"  saved samples -> {SAMP}  (fid corner FoM: CNN {fom_cnn:.0f}  L1 {fom_l1:.0f})", flush=True)
    return cnn[:, :3], l1[:, :3]


def build_figure(cnn, l1):
    from getdist import MCSamples, plots
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "figure.constrained_layout.use": False, "savefig.bbox": None})
    names = ["Om", "s8", "w0"]; labels = [r"\Omega_m", r"\sigma_8", "w_0"]
    truth = {"Om": 0.26, "s8": 0.84, "w0": -1.0}
    mc_cnn = MCSamples(samples=cnn, names=names, labels=labels)
    mc_l1 = MCSamples(samples=l1, names=names, labels=labels)

    g = plots.get_subplot_plotter(width_inch=WIDTH_INCH)
    g.settings.scaling = False
    g.settings.axes_fontsize = FS_TICK
    g.settings.axes_labelsize = FS_AXLABEL
    g.settings.linewidth_contour = LW_CONTOUR
    # draw the LARGER contour first (underneath) so the SMALLER one sits on top and stays visible —
    # here L1 (larger at the fiducial) underneath, CNN (tighter) on top
    g.triangle_plot([mc_l1, mc_cnn], params=names, filled=True,
                    contour_colors=[C_L1, C_CNN], contour_ls=["-", "--"],
                    contour_lws=[LW_CONTOUR, LW_CONTOUR], markers=truth)
    fig = g.fig
    for _leg in list(fig.legends):
        _leg.remove()
    for _ax in fig.axes:
        if _ax.get_legend() is not None:
            _ax.get_legend().remove()

    # --- 3-bar FoM3 inset in the empty upper-right triangle ---
    ax = fig.add_axes([0.638, 0.742, 0.340, 0.185])
    xpos = np.arange(len(FOM_BARS))
    for i, (lab, col, val) in enumerate(FOM_BARS):
        hatch = "///" if col == C_L1 else None
        ax.bar(xpos[i], val, 0.66, color=col, edgecolor="black", linewidth=0.7, hatch=hatch, zorder=3)
    ax.axhline(2875, color="0.45", ls=":", lw=1.0, zorder=0)          # L1's best (reference)
    ax.annotate("+15%", xy=(2, 3326), xytext=(2, 3450), ha="center", fontsize=12,
                fontweight="bold", color=C_CNN)
    ax.set_xticks(xpos); ax.set_xticklabels([b[0] for b in FOM_BARS], fontsize=FS_INSET_XLAB)
    ax.set_ylabel(r"FoM$_3$", fontsize=FS_INSET_YLAB); ax.set_ylim(0, 4000)
    ax.tick_params(labelsize=FS_INSET_TICK, top=False, right=False)
    ax.set_title("pooled 9000-obs median", fontsize=FS_INSET_TITLE, pad=4)

    # method legend in the (1,2) empty cell, shifted right
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=C_CNN, ls="--", lw=2.4, label="CNN (VMIM)"),
               Line2D([0], [0], color=C_L1, ls="-", lw=2.4, label=r"L1 (wavelet $\ell_1$)")]
    ax.legend(handles=handles, fontsize=FS_LEGEND, frameon=False, loc="upper left",
              bbox_to_anchor=(0.18, -1.05), handlelength=1.8, handletextpad=0.6)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"stitched_final_resnet18.{ext}")
    print(f"wrote {OUT}/stitched_final_resnet18.{{pdf,png}}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda-visible-devices", default="1")
    ap.add_argument("--force-recompute", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if SAMP.exists() and not a.force_recompute:
        d = np.load(SAMP); cnn, l1 = d["cnn"], d["l1"]
        print(f"  loaded cached samples from {SAMP} (no GPU)", flush=True)
    else:
        cnn, l1 = compute_samples(a.cuda_visible_devices)
    build_figure(cnn, l1)


if __name__ == "__main__":
    main()
