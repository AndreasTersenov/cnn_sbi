"""Compare fresh jaxili auto-only BNT / no-BNT runs against
paper_sbi_consolidation/bnt_comparison_tomo4 L1 reference posteriors.

Writes: summary.md, summary.json, overlay_bnt.png, overlay_nobnt.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/cross_maps_campaign")
REF_DIR = Path("/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/final/paper_sbi_consolidation/bnt_comparison_tomo4/posteriors")
OUT = ROOT / "consistency_vs_ref"
OUT.mkdir(exist_ok=True)
SEEDS = [41, 42, 43]
PARAM = [r"$\Omega_m$", r"$\sigma_8$", r"$w_0$", r"$h_0$", r"$n_s$", r"$\Omega_b$"]
FID = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493])


def stats(p: np.ndarray) -> dict:
    s = p.std(axis=0)
    cov3 = np.cov(p[:, :3].T)
    return {
        "mean": p.mean(0).tolist(),
        "std": s.tolist(),
        "std_sum_3par": float(s[:3].sum()),
        "omega_m_std": float(s[0]),
        "sigma8_std": float(s[1]),
        "w0_std": float(s[2]),
        "fom3": float(np.exp(-0.5 * np.linalg.slogdet(cov3)[1])),
    }


def agg(posts):
    ss = [stats(p) for p in posts]
    keys = ["std_sum_3par", "omega_m_std", "sigma8_std", "w0_std", "fom3"]
    out = {k: float(np.mean([s[k] for s in ss])) for k in keys}
    out["per_seed"] = ss
    return out


def overlay(cur, ref, out_png, title):
    import corner
    fig = corner.corner(
        np.concatenate(cur, 0), labels=PARAM, truths=FID, color="C0",
        smooth=1.0, plot_datapoints=False, levels=(0.39, 0.86),
        fill_contours=False, hist_kwargs={"density": True, "color": "C0"},
        label_kwargs={"fontsize": 12},
    )
    corner.corner(
        np.concatenate(ref, 0), fig=fig, color="C3",
        smooth=1.0, plot_datapoints=False, levels=(0.39, 0.86),
        fill_contours=False, hist_kwargs={"density": True, "color": "C3"},
    )
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([0], [0], color="C0", lw=2, label="current (jaxili)"),
        Line2D([0], [0], color="C3", lw=2, label="reference"),
    ], loc="upper right", fontsize=12)
    fig.suptitle(title)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print("wrote", out_png)


def load_current(regime: str):
    return [np.load(ROOT / f"jaxili_auto_{regime}" / "posteriors" /
                    f"l1_tomo4_20deg160_{regime}_s{s}.npy") for s in SEEDS]


def load_ref(regime: str):
    return [np.load(REF_DIR / f"l1_tomo4_20deg160_{regime}_s{s}.npy") for s in SEEDS]


def main():
    report = {}
    md = ["# jaxili auto-only consistency vs reference", "",
          "Reference: `final/paper_sbi_consolidation/bnt_comparison_tomo4/posteriors/l1_tomo4_20deg160_{regime}_s{41,42,43}.npy`.",
          "Current arms: `jaxili_auto_{regime}/posteriors/l1_tomo4_20deg160_{regime}_s{41,42,43}.npy`,",
          "built with the exact reference config (`npe_l1norm_jaxili_nbody_tomo.py`, no PCA, grid_20deg_160px, lr=1e-4,",
          "batch 256, --total-steps 5000, SNR [-13, 13], n_scales=5, l1_nbins=40).",
          ""]
    keys = ["std_sum_3par", "omega_m_std", "sigma8_std", "w0_std", "fom3"]
    for regime in ["bnt", "nobnt"]:
        cur = agg(load_current(regime))
        ref = agg(load_ref(regime))
        report[regime] = {"current": cur, "reference": ref}
        md += [f"## {regime.upper()}", "",
               "| metric | current | reference | ratio (cur/ref) |",
               "|---|---|---|---|"]
        for k in keys:
            c, r = cur[k], ref[k]
            md.append(f"| {k} | {c:.5f} | {r:.5f} | {(c/r):.3f} |")
        md.append("")
        md += ["| param | current mean | ref mean | current std | ref std |",
               "|---|---|---|---|---|"]
        cm = np.mean([s["mean"] for s in cur["per_seed"]], 0)
        rm = np.mean([s["mean"] for s in ref["per_seed"]], 0)
        cs = np.mean([s["std"]  for s in cur["per_seed"]], 0)
        rs = np.mean([s["std"]  for s in ref["per_seed"]], 0)
        for i, name in enumerate(["Omega_m", "sigma_8", "w_0", "h_0", "n_s", "Omega_b"]):
            md.append(f"| {name} | {cm[i]:.4f} | {rm[i]:.4f} | {cs[i]:.4f} | {rs[i]:.4f} |")
        md.append("")

        overlay(load_current(regime), load_ref(regime),
                OUT / f"overlay_{regime}.png",
                f"auto-only {regime.upper()}: current jaxili (blue) vs reference (red)")

    (OUT / "summary.md").write_text("\n".join(md))
    (OUT / "summary.json").write_text(json.dumps(report, indent=2))
    print("wrote", OUT / "summary.md")


if __name__ == "__main__":
    main()
