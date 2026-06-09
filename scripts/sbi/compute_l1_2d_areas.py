#!/usr/bin/env python3
"""Per-arm constraining power for the flat-local L1 matrix: marginal sigma + per-pair 2D FoM.

Computed per NDE seed then averaged (NOT by pooling samples, which would inflate the posterior).
2D FoM(a,b) = 1/sqrt(det Cov[a,b]); FoM3 = 1/sqrt(det Cov[Om,s8,w0]). Leads with sigma/2D since
FoM3 amplifies tiny 3-D correlation changes (feedback_fom3_fragile). Single-obs (perm0 patch90),
pre-GATE-C. Prints a table ratioed to auto-only.
"""
import glob, numpy as np, os

HERE = os.path.dirname(os.path.abspath(__file__))
D = HERE + "/results/exploratory/flatsky_cross_2026_06/l1_matrix"
ARMS = ["none", "conv", "product", "both"]
SCI = [0, 1, 2]  # Om, s8, w0
PAIRS = [(0, 1, "Om-s8"), (0, 2, "Om-w0"), (1, 2, "s8-w0")]


def per_seed_metrics(arm):
    out = []
    for f in sorted(glob.glob(f"{D}/l1_{arm}_s*/posterior.npy")):
        s = np.load(f)[:, :3]
        C = np.cov(s, rowvar=False)
        m = {"sig": np.sqrt(np.diag(C)),
             "fom3": 1.0 / np.sqrt(max(np.linalg.det(C), 1e-300))}
        for i, j, nm in PAIRS:
            C2 = C[np.ix_([i, j], [i, j])]
            m[nm] = 1.0 / np.sqrt(max(np.linalg.det(C2), 1e-300))
        out.append(m)
    return out


def agg(arm):
    ms = per_seed_metrics(arm)
    return {"sig": np.mean([m["sig"] for m in ms], 0),
            "fom3": np.mean([m["fom3"] for m in ms]),
            **{nm: np.mean([m[nm] for m in ms]) for _, _, nm in PAIRS},
            "n": len(ms)}


def main():
    A = {a: agg(a) for a in ARMS}
    base = A["none"]
    print("Flat-local L1 — constraining power (mean over 3 seeds, single-obs, PRE-GATE-C)\n")
    lbl = ["Om", "s8", "w0"]
    print(f"{'arm':9s} | " + " ".join(f"sig({l})" for l in lbl) +
          " | " + " ".join(f"2DFoM {nm}" for _, _, nm in PAIRS) + " |  FoM3")
    print("-" * 92)
    for a in ARMS:
        m = A[a]
        sig = " ".join(f"{m['sig'][k]:7.3f}" for k in range(3))
        fom2 = " ".join(f"{m[nm]:9.0f}" for _, _, nm in PAIRS)
        print(f"{a:9s} | {sig} | {fom2} | {m['fom3']:6.0f}")
    print("\nRatio vs auto-only (sig: <1 = tighter is BETTER; FoM: >1 = BETTER):")
    print(f"{'arm':9s} | " + " ".join(f"sig({l})" for l in lbl) +
          " | " + " ".join(f"2DFoM {nm}" for _, _, nm in PAIRS) + " |  FoM3")
    print("-" * 92)
    for a in ARMS:
        m = A[a]
        sig = " ".join(f"{m['sig'][k]/base['sig'][k]:7.2f}" for k in range(3))
        fom2 = " ".join(f"{m[nm]/base[nm]:9.2f}" for _, _, nm in PAIRS)
        print(f"{a:9s} | {sig} | {fom2} | {m['fom3']/base['fom3']:6.2f}")


if __name__ == "__main__":
    main()
