#!/usr/bin/env python
"""A1 (VMIM joint PDF) vs the previous best (l1 + product cross map) — comparison.

(1) population bar chart: FoM3 + per-param sigma for A1, l1+product, l1-auto, pair2d-K10.
(2) corner overlay A1 vs l1+product at a MATCHED val truth (same 144000-val selection ->
    same rng(3) idx -> identical theta; pooled 3 seeds), truth crosshairs.
All from existing median jsons + TARP dumps (CPU, no GPU).
Calibration context (printed on the figure): l1+product is GATE-C CLEAN (|dev|<=0.037);
A1's joint-coverage is net-conservative (+0.021) but pending VMIM multi-seed.
"""
import glob
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FC = ("/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/"
      "flatsky_cross_2026_06")
OUT = f"{FC}/overnight_menu_2/lane_a_plots"
PN = ["Om", "s8", "w0"]

# label -> (median_summary.json, color)
POP = {
    "A1 VMIM joint PDF": (f"{FC}/overnight_menu_2/A1_pair2d_vmim/population_sweep_full/median_summary.json", "tab:red"),
    "l1 + product (prev best)": (f"{FC}/population_sweep/flat_product/median_summary.json", "tab:purple"),
    "pair2d K=10 raw": (f"{FC}/overnight_menu/pair2dq_nobnt/population_sweep_full/median_summary.json", "tab:blue"),
    "l1 auto-only": (f"{FC}/population_sweep/flat_none/median_summary.json", "tab:gray"),
}
# corner: (gate dumps root, arm key, color) for the two arms with matched val dumps
CORNER = {
    "A1 VMIM joint PDF (3822)": (f"{FC}/overnight_menu_2/gate_c", "A1_pair2d_vmim", "tab:red"),
    "l1 + product (2875, gate-C clean)": (f"{FC}/gate_c", "flat_product", "tab:purple"),
}
SEEDS = (41, 42, 43)


def population_figure():
    data = {k: json.load(open(v[0])) for k, v in POP.items()}
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    metrics = [("fom3", "FoM3 (↑ better)", False),
               ("sigma_Om", "σ(Ωm) (↓ better)", True),
               ("sigma_s8", "σ(σ8) (↓ better)", True),
               ("sigma_w0", "σ(w0) (↓ better)", True)]
    labels = list(POP.keys()); cols = [POP[k][1] for k in labels]
    for ax, (key, title, _) in zip(axes, metrics):
        vals = [data[k][key] for k in labels]
        ax.bar(range(len(labels)), vals, color=cols)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace(" ", "\n") for l in labels], fontsize=7)
        ax.set_title(title, fontsize=10)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
        ax.margins(y=0.15)
    fig.suptitle("A1 (VMIM joint PDF) vs previous best (l1+product) — population medians, "
                 "9000 obs", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/compare_a1_vs_product_bars.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/compare_a1_vs_product_bars.png/pdf", flush=True)
    return data


def gather(gate_root, arm):
    pts = {}
    for seed in SEEDS:
        for terc in ("HIGH", "MID", "LOW"):
            for f in glob.glob(f"{gate_root}/tarp_drp/dumps/{arm}_{terc}/seed_{seed}/"
                               "n*_m*/posterior_samples.npz"):
                z = np.load(f); s, th = z["samples"], z["theta"]
                for i in range(th.shape[0]):
                    pts.setdefault(tuple(np.round(th[i].astype(np.float64), 8)),
                                   []).append(s[i])
    return pts


def corner_figure():
    from getdist import MCSamples, plots as gdplt
    a1 = gather(*CORNER["A1 VMIM joint PDF (3822)"][:2])
    keys = np.array(list(a1.keys()))
    center = np.array([0.26, 0.84, -1.0]); sca = np.array([0.115, 0.288, 0.462])
    d = np.linalg.norm((keys[:, :3] - center) / sca, axis=1)
    truth = np.array(list(a1.keys())[int(np.argmin(d))])
    print(f"corner truth (Om,s8,w0) = {truth[:3]}", flush=True)
    mcs, colors = [], []
    for label, (groot, arm, col) in CORNER.items():
        pts = gather(groot, arm)
        kk = np.array(list(pts.keys()))
        mk = tuple(kk[int(np.argmin(np.linalg.norm(kk - truth, axis=1)))])
        samp = np.concatenate(pts[mk], axis=0)[:, :3]
        mcs.append(MCSamples(samples=samp, names=PN,
                             labels=[r"\Omega_m", r"\sigma_8", "w_0"], label=label))
        colors.append(col)
    g = gdplt.get_subplot_plotter(width_inch=7.0)
    g.settings.legend_fontsize = 9
    g.triangle_plot(mcs, PN, filled=True, colors=colors, legend_labels=list(CORNER.keys()))
    for a, t in enumerate(truth[:3]):
        for ax in g.subplots[:, a]:
            if ax is not None:
                ax.axvline(t, color="k", ls=":", lw=0.8)
    for ext in ("png", "pdf"):
        g.export(f"{OUT}/compare_a1_vs_product_corner.{ext}")
    print(f"wrote {OUT}/compare_a1_vs_product_corner.png/pdf", flush=True)


def write_md(data):
    L = ["# A1 (VMIM joint PDF) vs previous best (l1 + product cross map)", "",
         "Population medians (9000 obs). Calibration: l1+product GATE-C CLEAN (|dev|≤0.037); "
         "A1 joint-coverage net-conservative (TARP +0.021) but pending VMIM multi-seed.", "",
         "| arm | FoM3 | σ(Ωm) | σ(σ8) | σ(w0) | vs l1+product |",
         "|---|---|---|---|---|---|"]
    p = data["l1 + product (prev best)"]
    for k, v in POP.items():
        d = data[k]
        L.append(f"| {k} | {d['fom3']:.0f} | {d['sigma_Om']:.4f} | {d['sigma_s8']:.4f} | "
                 f"{d['sigma_w0']:.4f} | FoM3 ×{d['fom3']/p['fom3']:.2f} |")
    L += ["", f"A1 is FoM3 ×{data['A1 VMIM joint PDF']['fom3']/p['fom3']:.2f} vs l1+product, "
          f"with every science marginal tighter (σ(σ8) {data['A1 VMIM joint PDF']['sigma_s8']:.3f} "
          f"vs {p['sigma_s8']:.3f}). IF the VMIM multi-seed confirms and joint-coverage holds, "
          "A1 is the new best — a calibrated joint-PDF arm beating the explicit cross map, "
          "from auto maps alone.", "",
          "Caveat (FoM3 fragility, standing): differences are ~10-15%/param in width; "
          "marginals-first reading carried."]
    Path(OUT, "COMPARE_A1_VS_PRODUCT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def main():
    Path(OUT).mkdir(parents=True, exist_ok=True)
    data = population_figure()
    write_md(data)
    corner_figure()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
