#!/usr/bin/env python3
"""Offset / error-budget comparison across the geometry-resample arms.

Answers: is L1's anti-shrinkage systematic offset at the fiducial CROSS-specific or a general
L1 (flat-sky) trait? Compares auto+cross vs auto-only for L1 and CNN. Reports, per param:
  - systematic offset = population-mean bias (mean - FIDUCIAL); pull = offset/median-sigma-ish (per-patch pull mean)
  - shrinkage direction (grid_mean - fiducial): is the offset anti-shrinkage (genuine) or a prior pull?
  - full error budget W (median posterior width), B (mean-scatter across patches), |offset|, total=sqrt(W^2+B^2+off^2)
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GM = Path("results/exploratory/definitive_comparison/fiducial_full200/geometry_map")
FIDUCIAL = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493])
FOM3 = ["Omega_m", "sigma_8", "w_0"]
# training-grid means (for shrinkage-direction test) from l1_autocross train cache
GRID_MEAN = {"Omega_m": 0.29853, "sigma_8": 0.81587, "w_0": -0.89402}
ARMS = ["l1_autocross", "cnn_autocross", "l1_autoonly", "cnn_autoonly"]
FID = dict(zip(["Omega_m", "sigma_8", "w_0"], FIDUCIAL[:3]))


def load(arm):
    p = GM / arm / "per_patch_grid.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    d = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    for k in d:
        if k != "valid_fom3":
            d[k] = d[k].astype(float)
    d["patch"] = d["patch"].astype(int)
    return d


def main():
    grids = {a: load(a) for a in ARMS}
    grids = {a: g for a, g in grids.items() if g is not None}
    print(f"arms present: {list(grids)}\n")

    report = {}
    # ---- offset / pull table ----
    print("="*92)
    print("SYSTEMATIC OFFSET (population mean bias) and PULL, per param, per arm")
    print("(+ shrinkage dir = grid_mean - fiducial; if offset has OPPOSITE sign -> anti-shrinkage = genuine)")
    print("="*92)
    for p in FOM3:
        sd = GRID_MEAN[p] - FID[p]
        print(f"\n  {p}   (fiducial {FID[p]:+.4f}, grid_mean {GRID_MEAN[p]:+.4f}, shrink_dir {sd:+.4f})")
        print(f"  {'arm':16s} {'mean_bias':>11s} {'mean_pull':>11s} {'vs shrinkage':>14s}")
        for arm, g in grids.items():
            off = float(np.mean(g[f"bias_{p}"])); pull = float(np.nanmean(g[f"pull_{p}"]))
            anti = "ANTI (genuine)" if np.sign(off) != np.sign(sd) else "with-prior"
            tag = anti if abs(off) > 1e-4 else "~zero"
            print(f"  {arm:16s} {off:+11.4f} {pull:+11.3f} {tag:>14s}")
            report.setdefault(arm, {}).setdefault(p, {})["offset"] = off
            report[arm][p]["pull"] = pull

    # ---- full error budget ----
    print("\n" + "="*92)
    print("ERROR BUDGET  W=median width, B=mean-scatter across patches, |off|=offset, total=sqrt(W^2+B^2+off^2)")
    print("="*92)
    for p in FOM3:
        print(f"\n  {p}")
        print(f"  {'arm':16s} {'W':>8s} {'B':>8s} {'|off|':>8s} {'TOTAL':>8s}")
        for arm, g in grids.items():
            W = float(np.median(g[f"sig_{p}"])); B = float(np.std(g[f"mean_{p}"])); off = abs(float(np.mean(g[f"bias_{p}"])))
            tot = float(np.sqrt(W**2 + B**2 + off**2))
            print(f"  {arm:16s} {W:8.4f} {B:8.4f} {off:8.4f} {tot:8.4f}")
            report[arm][p].update(W=W, B=B, total=tot)

    (GM / "offset_comparison.json").write_text(json.dumps(report, indent=2))

    # ---- figure: pull per param, 4 arms ----
    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = {"l1_autocross": "C3", "cnn_autocross": "C0", "l1_autoonly": "C1", "cnn_autoonly": "C9"}
    x = np.arange(len(FOM3)); w = 0.8 / max(len(grids), 1)
    for k, arm in enumerate(grids):
        pulls = [report[arm][p]["pull"] for p in FOM3]
        ax.bar(x + k*w, pulls, w, label=arm, color=colors.get(arm, "k"), alpha=0.85)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x + w*(len(grids)-1)/2); ax.set_xticklabels(FOM3)
    ax.set_ylabel("population-mean PULL (bias / σ) at fiducial")
    ax.set_title("Systematic offset: is L1's anti-shrinkage bias cross-specific or general?")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(GM / "figures" / "offset_comparison.png", dpi=130); plt.close(fig)
    print(f"\nFigure -> {GM/'figures'/'offset_comparison.png'}  Report -> {GM/'offset_comparison.json'}")


if __name__ == "__main__":
    main()
