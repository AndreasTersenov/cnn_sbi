#!/usr/bin/env python3
"""Phase B headline figures: CNN auto+cross vs auto-only (tf.data route, leakage-flagged)."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from getdist import MCSamples, plots

PB = "results/exploratory/definitive_comparison/phaseB_tfdata_2026_05_30"
FIG = os.path.join(PB, "figures"); os.makedirs(FIG, exist_ok=True)
NAMES = ["Omega_m", "sigma_8", "w_0", "h_0", "n_s", "Omega_b"]
LABELS = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]
TRUTH = [0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493]


def pool(arm, cs=41):
    fs = sorted(glob.glob(f"{PB}/posteriors/{arm}/{arm}_cs{cs}/{arm}_cs{cs}_s*_p0.npy"))
    arrs = [np.load(f) for f in fs]
    return (np.concatenate(arrs, 0) if arrs else None), len(fs)


ao, nao = pool("autoonly"); ac, nac = pool("autocross")
print(f"auto-only pooled {None if ao is None else ao.shape} (n={nao}); "
      f"auto+cross pooled {None if ac is None else ac.shape} (n={nac})")

# ---- Figure 1: 6-param corner overlay ----
s_ao = MCSamples(samples=ao, names=NAMES, labels=LABELS, label=f"CNN auto-only (n={nao})")
s_ac = MCSamples(samples=ac, names=NAMES, labels=LABELS, label=f"CNN auto+cross (n={nac})")
g = plots.get_subplot_plotter(width_inch=10)
g.settings.legend_fontsize = 15
g.settings.axes_fontsize = 11
g.settings.axes_labelsize = 15
g.triangle_plot(
    [s_ao, s_ac], filled=True,
    contour_colors=["#888888", "#c1121f"],
    legend_labels=[f"CNN auto-only (n={nao})", f"CNN auto+cross (n={nac})"],
    markers={n: t for n, t in zip(NAMES, TRUTH)},
)
g.fig.suptitle("CNN auto+cross vs auto-only — jaxili MAF NDE, perm 0, comp-seed 41\n"
               "(tf.data route; ABSOLUTE inflated by ~1.6x leakage — read RELATIVE gain)",
               fontsize=12, y=1.02)
f1 = os.path.join(FIG, "corner_autocross_vs_autoonly.png")
g.export(f1)
print("wrote", f1)

# ---- Figure 2: FoM3 bar + per-run points ----
def foms(arm, cs=41):
    out = []
    for f in sorted(glob.glob(f"{PB}/posteriors/{arm}/{arm}_cs{cs}/*.fom.json")):
        out.append(json.load(open(f))["fom3"])
    return out


ao_f, ac_f = foms("autoonly"), foms("autocross")
fig, ax = plt.subplots(figsize=(6, 5))
xs = [0, 1]
means = [np.mean(ao_f), np.mean(ac_f)]
ax.bar(xs, means, width=0.55, color=["#888888", "#c1121f"], alpha=0.85,
       label="mean over NDE seeds")
for x, vals in zip(xs, [ao_f, ac_f]):
    ax.scatter([x] * len(vals), vals, color="k", zorder=5, s=35)
for x, m in zip(xs, means):
    ax.text(x, m, f"{m:.0f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels(["auto-only", "auto+cross"], fontsize=12)
ax.set_ylabel("FoM3  =  1 / sqrt(det C₃)  on (Ωm, σ₈, w₀)", fontsize=11)
ax.set_title(f"CNN FoM3 — cross/auto = {np.mean(ac_f)/np.mean(ao_f):.2f}×\n"
             "(absolute inflated ~1.6× by leakage; trust the ratio)", fontsize=11)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
f2 = os.path.join(FIG, "fom3_bar_autocross_vs_autoonly.png")
fig.savefig(f2, dpi=130)
print("wrote", f2)
