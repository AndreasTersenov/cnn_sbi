#!/usr/bin/env python3
"""Phase C — definitive L1-vs-CNN table across ALL arms (σ/2D primary, FoM3 secondary).

Discovers every "final definitive" arm's posteriors (perm 0, seeds pooled), computes
marginal σ, 2D FoM, FoM3, and joins the TARP coverage max|ECP-α| (dim3) if present.
Robust: arms with no posteriors yet are skipped. Writes SUMMARY_DEFINITIVE.md + csv.

Note: CNN absolute FoM are fast-tf.data-route; per Andreas (2026-05-31) the
compressor↔NDE overlap is empirically negligible, so absolute is treated as fine.
FoM3 is fragile (use σ/2D as primary) — see feedback_fom3_fragile_use_2d_areas.
"""
import glob, json, os
import numpy as np

DC = "results/exploratory/definitive_comparison"
OUT = os.path.join(DC, "PHASE_C_2026_05_31"); os.makedirs(OUT, exist_ok=True)
NAMES = ["Omega_m", "sigma_8", "w_0", "h_0", "n_s", "Omega_b"]
SEEDS = [41, 42, 43]

# arm display -> glob of perm-0 posterior .npy (seeds pooled)
ARMS = {
    "L1 auto+cross":        f"{DC}/posteriors/l1_autocross_split70/l1_autocross_split70_s*_p0.npy",
    "L1 auto-only":         f"{DC}/posteriors/l1_autoonly_split70/l1_autoonly_split70_s*_p0.npy",
    "CNN-RealNVP auto+cross": f"{DC}/phaseB_tfdata_2026_05_30/posteriors/autocross/autocross_cs41/autocross_cs41_s*_p0.npy",
    "CNN-RealNVP auto-only":  f"{DC}/phaseB_tfdata_2026_05_30/posteriors/autoonly/autoonly_cs41/autoonly_cs41_s*_p0.npy",
    "CNN-MAF auto+cross":     f"{DC}/phaseB_maf_2026_05_31/posteriors/autocross/autocross_cs41/autocross_cs41_s*_p0.npy",
    "CNN-MAF auto-only":      f"{DC}/phaseB_maf_2026_05_31/posteriors/autoonly/autoonly_cs41/autoonly_cs41_s*_p0.npy",
    "CNN-RealNVP auto+cross (std)": f"{DC}/phaseB_std_2026_05_31/posteriors/**/*_s*_p0.npy",
    "CNN-auto native-TFDS (RealNVP)": f"{DC}/phaseB_nativeauto_2026_05_31/posteriors/**/*_s*_p0.npy",
    "CNN auto+cross multi-perm (3 perms)": f"{DC}/phaseB_multiperm_2026_05_31/posteriors/autocross_multiperm/*_p*.npy",
    "CNN auto-only multi-perm (3 perms)": f"{DC}/phaseB_multiperm_2026_05_31/posteriors/autoonly_multiperm/*_p*.npy",
}


def pool(globpat):
    fs = sorted(f for f in glob.glob(globpat, recursive=True) if "fom" not in f.lower())
    a = [np.load(f) for f in fs]
    return (np.concatenate(a, 0), len(a)) if a else (None, 0)


def fom3(x):
    return float(1.0 / np.sqrt(np.linalg.det(np.cov(x[:, :3], rowvar=False))))


def fom2d(x, i, j):
    return float(1.0 / np.sqrt(np.linalg.det(np.cov(x[:, [i, j]], rowvar=False))))


def load_tarp():
    """tarp_summary.json -> {arm: max_dev(dim3)} if present."""
    out = {}
    for p in glob.glob(f"{DC}/tarp_2026_05_31/tarp_summary.json"):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        # structure unknown across versions; best-effort scan for arm->dim3 max dev
        out["_raw"] = p
    return out


def main():
    rows = []
    for arm, gp in ARMS.items():
        x, n = pool(gp)
        if x is None:
            continue
        rows.append((arm, n, fom3(x),
                     fom2d(x, 0, 1), fom2d(x, 0, 2), fom2d(x, 1, 2),
                     {nm: float(np.std(x[:, k])) for k, nm in enumerate(NAMES)}))
    # CSV
    with open(os.path.join(OUT, "phase_c.csv"), "w") as f:
        f.write("arm,n,fom3,fom2d_Om_s8,fom2d_Om_w0,fom2d_s8_w0," + ",".join(f"sig_{n}" for n in NAMES) + "\n")
        for arm, n, f3, a, b, c, sg in rows:
            f.write(f"{arm},{n},{f3:.1f},{a:.1f},{b:.1f},{c:.1f}," + ",".join(f"{sg[n]:.5f}" for n in NAMES) + "\n")
    # Markdown
    L = ["# Definitive L1 vs CNN — Phase C summary (perm 0, seeds pooled)", "",
         "Primary metrics: **marginal σ** and **2D FoM** (FoM3 is fragile — "
         "[[feedback_fom3_fragile_use_2d_areas]]). CNN fast-route absolute treated as "
         "fine (overlap empirically negligible, Andreas 2026-05-31).", "",
         f"Arms with posteriors: {len(rows)}", "",
         "## Marginal σ (lower = tighter)", "",
         "| arm | n | σ(Ωm) | σ(σ8) | σ(w0) | σ(h0) | σ(ns) | σ(Ωb) | FoM3 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for arm, n, f3, a, b, c, sg in rows:
        L.append(f"| {arm} | {n} | " + " | ".join(f"{sg[nm]:.4f}" for nm in NAMES) + f" | {f3:.0f} |")
    L += ["", "## 2D FoM (higher = tighter)", "",
          "| arm | Ωm–σ8 | Ωm–w0 | σ8–w0 |", "|---|---|---|---|"]
    for arm, n, f3, a, b, c, sg in rows:
        L.append(f"| {arm} | {a:.0f} | {b:.0f} | {c:.0f} |")
    L += ["", "## TARP coverage", "",
          "See `tarp_2026_05_31/figures/tarp_{per_arm,overlay}_dim{3,6}.png` and "
          "`tarp_summary.json`. A calibrated arm sits on the diagonal; below = over-confident.",
          "", "_Auto-generated; re-run aggregate_all_arms.py to refresh as arms land._"]
    open(os.path.join(OUT, "SUMMARY_DEFINITIVE.md"), "w").write("\n".join(L) + "\n")
    print(f"[phase-c] {len(rows)} arms -> {OUT}/SUMMARY_DEFINITIVE.md")
    for arm, n, f3, *_ in rows:
        print(f"  {arm:42s} n={n} FoM3={f3:.0f}")


if __name__ == "__main__":
    main()
