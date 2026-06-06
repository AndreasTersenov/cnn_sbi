#!/usr/bin/env python
"""3-way overlay corner of the 10deg smoke posteriors (L1 a+c, CNN a+c, CNN auto-only)."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
from getdist import MCSamples, plots

BASE = "results/exploratory/definitive_comparison_10deg"
OUT = f"{BASE}/plots"
os.makedirs(OUT, exist_ok=True)
TRUTH = [0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493]
NAMES = ["Om", "s8", "w0", "h0", "ns", "Ob"]
LABELS = [r"\Omega_m", r"\sigma_8", "w_0", "h_0", "n_s", r"\Omega_b"]

arms = [
    ("smoke_l1_autocross", "L1 auto+cross", "#2ca02c"),
    ("smoke_autocross", "CNN auto+cross", "#1f77b4"),
    ("smoke_autoonly", "CNN auto-only", "#d62728"),
]
samps = []
for tag, label, _c in arms:
    p = np.asarray(np.load(f"{BASE}/{tag}/posterior.npy")).reshape(-1, 6)
    samps.append(MCSamples(samples=p, names=NAMES, labels=LABELS, label=label))

g = plots.get_subplot_plotter(width_inch=9)
g.settings.alpha_filled_add = 0.55
g.triangle_plot(
    samps, filled=True,
    contour_colors=[c for _, _, c in arms],
    markers={n: TRUTH[i] for i, n in enumerate(NAMES)},
)
g.export(f"{OUT}/smoke_corner3.pdf")
g.export(f"{OUT}/smoke_corner3.png")
print(f"wrote {OUT}/smoke_corner3.png")
