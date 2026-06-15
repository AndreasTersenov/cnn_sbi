#!/usr/bin/env python3
"""Matrix figure: FoM3 of analytical representations x NDE family, vs the CNN, colored by GATE-C
verdict. Curated numbers (this campaign + known baselines) hardcoded for a clean presentation
artifact. Outputs fom3_matrix.{png,pdf} in the campaign dir.

Verdict colors: PASS green, PASS-with-caveat amber, FAIL red, baseline/known grey-edge.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (representation, NDE) -> (FoM3, verdict). Verdict in {PASS, CAVEAT, FAIL, KNOWN}.
# Seed-banded arms show the mean; band annotated in the writeup.
DATA = {
    "l1-auto (800-d)": {
        "raw→MAF":        (2405, "PASS"),    # prior gate-C clean baseline
        "VMIM→MAF":       (1882, "KNOWN"),   # this campaign (not separately gated)
        "VMIM→RealNVP":   (2448, "CAVEAT"),  # control gate PASS-with-caveat
    },
    "l1+product (2000-d)": {
        "raw→MAF":        (2875, "PASS"),    # prior gate-C clean baseline
        "VMIM→MAF":       (2426, "CAVEAT"),  # this campaign; gate verdict
        "VMIM→RealNVP":   (3270, "CAVEAT"),  # seed band {3146,3399,3265} mean; PASS-w-caveat x2(+)
    },
    "pair2d (joint, autos)": {
        "raw→MAF":        (2794, "FAIL"),    # prior: over-confident
        "VMIM→MAF":       (3557, "FAIL"),    # A1 band {3822,3441,3408} mean; borderline/over-conf
        "VMIM→RealNVP":   (4864, "FAIL"),    # band {4922,5156,4513} mean; gate FAIL (over-conf)
    },
}
CNN = 3293  # ResNet18 + sbi_lens RealNVP 4x128, calibrated (PASS)

COLS = ["raw→MAF", "VMIM→MAF", "VMIM→RealNVP"]
VCOLOR = {"PASS": "#2ca02c", "CAVEAT": "#ff7f0e", "FAIL": "#d62728", "KNOWN": "#7f7f7f"}
HATCH = {"PASS": "", "CAVEAT": "", "FAIL": "//", "KNOWN": ".."}

reps = list(DATA)
nrep, ncol = len(reps), len(COLS)
x = np.arange(nrep)
w = 0.26

fig, ax = plt.subplots(figsize=(10, 5.6))
for j, col in enumerate(COLS):
    vals = [DATA[r][col][0] for r in reps]
    verds = [DATA[r][col][1] for r in reps]
    bars = ax.bar(x + (j - 1) * w, vals, w, label=col,
                  color=[VCOLOR[v] for v in verds], edgecolor="black", linewidth=0.7,
                  hatch=[HATCH[v] for v in verds])
    for b, v, vd in zip(bars, vals, verds):
        ax.text(b.get_x() + b.get_width() / 2, v + 40, f"{v:.0f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, 120, {"PASS": "✓", "CAVEAT": "~", "FAIL": "✗",
                "KNOWN": ""}[vd], ha="center", va="bottom", fontsize=11, color="white",
                fontweight="bold")

ax.axhline(CNN, color="navy", ls="--", lw=1.8)
ax.text(nrep - 0.5, CNN + 50, f"CNN ResNet18+RealNVP = {CNN} (calibrated)", color="navy",
        ha="right", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(reps, fontsize=10)
ax.set_ylabel("FoM3 = 1/√det C₃(Ωm,σ8,w0)  (median over fiducial obs)")
ax.set_title("Analytical statistics × NDE family vs the optimal CNN — colored by GATE-C calibration\n"
             "(✓ PASS · ~ PASS-with-caveat · ✗ FAIL=over-confident · hatched=not-trustworthy)",
             fontsize=10.5)
ax.set_ylim(0, 5500)

handles = [mpatches.Patch(facecolor=VCOLOR[k], edgecolor="black",
           hatch=HATCH[k], label={"PASS": "PASS (calibrated)",
           "CAVEAT": "PASS-with-caveat", "FAIL": "FAIL (over-confident)",
           "KNOWN": "known baseline"}[k]) for k in ["PASS", "CAVEAT", "FAIL", "KNOWN"]]
leg1 = ax.legend(handles=handles, title="GATE-C verdict", loc="upper left", fontsize=8)
ax.add_artist(leg1)
ax.legend(COLS, title="NDE path", loc="upper center", fontsize=8, ncol=3)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fom3_matrix.{ext}", dpi=150, bbox_inches="tight")
print("wrote", HERE / "fom3_matrix.png")
