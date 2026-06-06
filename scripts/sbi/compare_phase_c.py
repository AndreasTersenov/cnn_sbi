#!/usr/bin/env python
"""Phase C comparison: pool the 3 seeds per arm, report σ / 2D(Ωm,σ8) / FoM3 and the
L1-vs-CNN ratios for auto-only and auto+cross. Lead with σ + 2D, NOT FoM3 (fragile).

Writes SUMMARY_PHASE_C.md + a per-probe overlay corner. CPU-only.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import glob
import numpy as np

BASE = "results/exploratory/definitive_comparison_10deg/phase_c"
OUT = f"{BASE}/analysis"
TRUTH = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493])
NAMES = ["Om", "s8", "w0", "h0", "ns", "Ob"]
SEEDS = [41, 42, 43]
ARMS = ["l1_auto_cross", "cnn_auto_cross", "l1_auto_only", "cnn_auto_only"]


def fom3(p):
    return 1.0 / np.sqrt(np.linalg.det(np.cov(p[:, :3], rowvar=False)))


def fom2(p):  # 2D(Ωm,σ8)
    return 1.0 / np.sqrt(np.linalg.det(np.cov(p[:, :2], rowvar=False)))


def pool(arm):
    ps = []
    for s in SEEDS:
        f = f"{BASE}/{arm}_s{s}/posterior.npy"
        if os.path.exists(f):
            ps.append(np.asarray(np.load(f)).reshape(-1, 6))
    if not ps:
        return None, 0
    return np.concatenate(ps, 0), len(ps)


def main():
    os.makedirs(OUT, exist_ok=True)
    stats = {}
    lines = ["# Phase C — 4-arm × 3-seed (10°), 3-seed-pooled @ obs patch 90",
             "",
             "Lead metrics: **σ(w0), 2D(Ωm,σ8)** (FoM3 reported, NOT headlined — fragile).",
             "",
             "| arm | n_seed | σ(Ωm) | σ(σ8) | σ(w0) | 2D(Ωm,σ8) | FoM3 | max\\|pull\\| |",
             "|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        p, n = pool(arm)
        if p is None:
            lines.append(f"| {arm} | 0 | — | — | — | — | — | (missing) |")
            continue
        s = p.std(0); m = p.mean(0); pull = np.max(np.abs((m - TRUTH) / s))
        stats[arm] = dict(sOm=s[0], ss8=s[1], sw0=s[2], f2=fom2(p), f3=fom3(p))
        lines.append(f"| {arm} | {n} | {s[0]:.4f} | {s[1]:.4f} | {s[2]:.4f} | "
                     f"{fom2(p):.0f} | {fom3(p):.0f} | {pull:.2f}σ |")

    lines += ["", "## L1-vs-CNN ratios (CNN/L1 for σ → >1 means L1 tighter)"]
    for probe in ("auto_cross", "auto_only"):
        l1, cnn = stats.get(f"l1_{probe}"), stats.get(f"cnn_{probe}")
        if l1 and cnn:
            lines.append(
                f"- **{probe}**: σ(w0) L1 {l1['sw0']:.3f} vs CNN {cnn['sw0']:.3f} "
                f"(×{cnn['sw0']/l1['sw0']:.2f}); σ(Ωm) ×{cnn['sOm']/l1['sOm']:.2f}; "
                f"2D ×{l1['f2']/cnn['f2']:.2f}; FoM3 ×{l1['f3']/cnn['f3']:.2f}")
    lines += ["", "## vs 20° (typical patch, for Phase E)",
              "- 20° L1 a+c: σ(w0) 0.125, 2D 3343, FoM3 53069 | CNN a+c: σ(w0) 0.167, 2D 2085, FoM3 24453",
              "- 20° L1/CNN a+c: σ(w0) ×1.34, 2D ×1.60, FoM3 ×2.17"]

    with open(f"{OUT}/SUMMARY_PHASE_C.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {OUT}/SUMMARY_PHASE_C.md")

    # Per-probe overlay corners (L1 vs CNN).
    try:
        from getdist import MCSamples, plots
        labels = [r"\Omega_m", r"\sigma_8", "w_0", "h_0", "n_s", r"\Omega_b"]
        for probe in ("auto_cross", "auto_only"):
            samps = []
            for arm, col in ((f"l1_{probe}", "#2ca02c"), (f"cnn_{probe}", "#1f77b4")):
                p, _ = pool(arm)
                if p is not None:
                    samps.append(MCSamples(samples=p, names=NAMES, labels=labels,
                                           label=arm.replace("_", " ")))
            if len(samps) == 2:
                g = plots.get_subplot_plotter(width_inch=8)
                g.triangle_plot(samps, filled=True, contour_colors=["#2ca02c", "#1f77b4"],
                                markers={n: TRUTH[i] for i, n in enumerate(NAMES)})
                g.export(f"{OUT}/corner_{probe}.pdf")
                g.export(f"{OUT}/corner_{probe}.png")
                print(f"wrote {OUT}/corner_{probe}.png")
    except Exception as e:
        print(f"(corner plotting skipped: {e})")


if __name__ == "__main__":
    main()
