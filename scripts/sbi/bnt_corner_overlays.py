#!/usr/bin/env python3
"""Corner overlays: BNT vs no-BNT, two contour sets per plot (Andreas, 2026-06-11).

Four figures — {L1, CNN} x {auto-only, auto+product} — each overlaying the no-BNT posterior
(existing representative-corner samples, typical obs perm16/patch23, 3-MAF pooled) with the
BNT posterior at the same obs (bnt_campaign representative corners). 3 science params,
truth markers. CPU-only (getdist; reads saved samples).
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

SBI = Path("/mnt/home/tersenov/software/cnn_sbi/scripts/sbi")
FC = SBI / "results/exploratory/flatsky_cross_2026_06"
BNT = FC / "bnt_campaign"
FIGS = BNT / "figures"

NOBNT = {
    ("l1", "none"): FC / "representative_corner/flat_none/corner_samples.npz",
    ("l1", "product"): FC / "representative_corner/flat_product/corner_samples.npz",
    ("cnn", "none"): FC / "cnn_phase/representative_corner/flat_none/corner_samples.npz",
    ("cnn", "product"): FC / "cnn_phase/representative_corner/flat_product/corner_samples.npz",
}
LAB = {"none": "auto-only", "product": "auto+product"}
C_NOBNT, C_BNT = "#0072B2", "#D55E00"


def main():
    from getdist import MCSamples, plots
    names = ["Om", "s8", "w0"]; labs = [r"\Omega_m", r"\sigma_8", r"w_0"]
    truth = {"Om": 0.26, "s8": 0.84, "w0": -1.0}
    FIGS.mkdir(parents=True, exist_ok=True)
    made = []
    for (probe, op), nobnt_f in NOBNT.items():
        bnt_f = BNT / f"representative_corner/{probe}_{op}/corner_samples.npz"
        if not (nobnt_f.exists() and bnt_f.exists()):
            print(f"[skip {probe} {op}] missing "
                  f"{'no-BNT' if not nobnt_f.exists() else 'BNT'} samples")
            continue
        s_n = np.load(nobnt_f)["typical"][:, :3]
        s_b = np.load(bnt_f)["typical"][:, :3]
        s_n = s_n[np.all(np.isfinite(s_n), 1)]; s_b = s_b[np.all(np.isfinite(s_b), 1)]
        mc_n = MCSamples(samples=s_n, names=names, labels=labs,
                         label=f"{probe.upper()} {LAB[op]} (no BNT)")
        mc_b = MCSamples(samples=s_b, names=names, labels=labs,
                         label=f"{probe.upper()} {LAB[op]} (BNT)")
        g = plots.get_subplot_plotter(width_inch=6.5)
        g.settings.alpha_filled_add = 0.55
        # no-BNT first (underneath); BNT on top so the (wider for L1) BNT set stays visible
        g.triangle_plot([mc_n, mc_b], filled=True, contour_colors=[C_NOBNT, C_BNT],
                        markers=truth,
                        legend_labels=[f"{probe.upper()} {LAB[op]} — no BNT",
                                       f"{probe.upper()} {LAB[op]} — BNT"])
        for ext in ("png", "pdf"):
            g.export(str(FIGS / f"corner_bnt_vs_nobnt_{probe}_{op}.{ext}"))
        made.append(f"corner_bnt_vs_nobnt_{probe}_{op}")
        print(f"  wrote {FIGS}/corner_bnt_vs_nobnt_{probe}_{op}.{{png,pdf}}")
    print(f"done: {len(made)}/4 overlays")
    return 0 if len(made) == 4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
