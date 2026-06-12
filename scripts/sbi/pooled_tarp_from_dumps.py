#!/usr/bin/env python
"""A3 (PLAN_OVERNIGHT_MENU_2.md): pooled-estimator TARP from existing GATE-C dumps.

The gate's TARP verdicts are per-seed (worst seed per tercile) but the QUOTED posterior is
the 3-seed POOL. This reprocessor concatenates the three seeds' samples per val point
(N, 3M, 6) and recomputes the TARP curve per (arm, tercile) — CPU only, no retraining.

Registered branches (plan lane A, A3):
  pooled HIGH dev <= -0.05  -> over-confidence survives pooling; FAIL stands for the
                               quoted estimator.
  pooled |dev| < 0.05       -> worst-seed gate over-penalized; downgrade softens to a
                               per-seed caveat.
Output: prints a table + appends '## Addendum 2' to overnight_menu/gate_c/GATE_C_JOINT.md.
"""
import glob
import numpy as np
from pathlib import Path
import tarp

GC = ("/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/"
      "flatsky_cross_2026_06/overnight_menu/gate_c")
ARMS = ["pair2dq_nobnt", "jointl1q_nobnt", "pair2dq_bnt", "jointl1q_bnt"]
TERCILES = ["HIGH", "MID", "LOW"]
DIMS = {"dim3": slice(0, 3), "dim6": slice(0, 6)}


def pooled_signed_dev(arm, terc, dim_sl):
    per_seed = []
    theta_ref = None
    for d in sorted(glob.glob(f"{GC}/tarp_drp/dumps/{arm}_{terc}/seed_*/n*_m*/"
                              "posterior_samples.npz")):
        z = np.load(d)
        s, th = z["samples"], z["theta"]
        if theta_ref is None:
            theta_ref = th
        else:
            assert np.array_equal(theta_ref, th), f"theta mismatch across seeds: {d}"
        per_seed.append(s)
    if not per_seed:
        return None
    pooled = np.concatenate(per_seed, axis=1)          # (N, 3M, 6)
    samples_tarp = np.transpose(pooled, (1, 0, 2))[:, :, dim_sl]   # (3M, N, d)
    theta_d = theta_ref[:, dim_sl]
    # kwargs mirror run_tarp_coverage.py's compute_curve (its --seed default 0, norm=True)
    ecp, alpha = tarp.get_tarp_coverage(samples_tarp, theta_d, references="random",
                                        num_bootstrap=200, bootstrap=True, norm=True,
                                        seed=0)
    e = np.asarray(ecp).mean(axis=0); a = np.asarray(alpha)
    i = int(np.argmax(np.abs(e - a)))
    return float(e[i] - a[i])


def main():
    lines = ["", "## Addendum 2 — pooled-estimator TARP (A3, PLAN_OVERNIGHT_MENU_2.md; "
             "derived from the same dumps)",
             "",
             "The gate verdicts above are worst-seed; the QUOTED posterior pools the 3",
             "seeds. Pooled (N, 3M) TARP per tercile (dim-3, signed max ECP − α):",
             "", "| arm | HIGH | MID | LOW |", "|---|---|---|---|"]
    worst_nobnt_high = {}
    for arm in ARMS:
        row = []
        for terc in TERCILES:
            d = pooled_signed_dev(arm, terc, DIMS["dim3"])
            row.append("—" if d is None else f"{d:+.3f}")
            if terc == "HIGH" and arm.endswith("_nobnt") and d is not None:
                worst_nobnt_high[arm] = d
            print(f"{arm:16s} {terc:4s} pooled signed dev: {row[-1]}", flush=True)
        lines.append(f"| {arm} | " + " | ".join(row) + " |")
    # registered branch resolution (derived)
    lines.append("")
    for arm, d in worst_nobnt_high.items():
        if d <= -0.05:
            lines.append(f"- {arm}: pooled HIGH {d:+.3f} ≤ −0.05 — **over-confidence "
                         "survives pooling; the verdict stands for the quoted estimator.**")
        elif abs(d) < 0.05:
            lines.append(f"- {arm}: pooled HIGH {d:+.3f}, |dev| < 0.05 — **the worst-seed "
                         "gate over-penalized; downgrade softens to a per-seed caveat.**")
        else:
            lines.append(f"- {arm}: pooled HIGH {d:+.3f} (positive/conservative) — outside "
                         "both registered branches; read with the table.")
    with open(Path(GC, "GATE_C_JOINT.md"), "a") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"appended Addendum 2 to {GC}/GATE_C_JOINT.md", flush=True)


if __name__ == "__main__":
    main()
