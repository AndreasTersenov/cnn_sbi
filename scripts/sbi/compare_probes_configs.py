#!/usr/bin/env python
"""3 probes × 3 input configs — overlay corner plots + FoM3 table.

For each probe (L1, CNN plain, CNN resnet50_gn) overlay three contours:
auto-only, cross-only (v2 channel-aware σ), auto+cross (v2 channel-aware σ
for L1; original normalized cache for CNN since CNN doesn't use σ-based SNR).

Saves to scripts/sbi/results/exploratory/probes_configs_comparison/.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from getdist import MCSamples, plots as gplot

REPO = Path("/mnt/home/tersenov/software/cnn_sbi")
EXPL = REPO / "scripts/sbi/results/exploratory"
OUT = EXPL / "probes_configs_comparison"
OUT.mkdir(parents=True, exist_ok=True)

PARAM_NAMES = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]
FIDUCIAL = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493])
FID_DICT = {n: float(FIDUCIAL[i]) for i, n in enumerate(PARAM_NAMES)}

# (probe, config) -> list of posterior paths
CELLS: Dict[tuple, List[Path]] = {
    # --- L1 ---
    ("L1", "auto-only"): [
        EXPL / f"cross_maps_campaign/jaxili_auto_nobnt/posteriors/l1_tomo4_20deg160_nobnt_s{s}.npy"
        for s in (41, 42, 43)
    ],
    ("L1", "cross-only (v2 chσ)"): [
        EXPL / f"cross_only_campaign_v2_chsigma/l1_cross_only/posteriors/l1_cross_only_s{s}.npy"
        for s in (41, 42, 43, 44, 45)
    ],
    ("L1", "auto+cross (v2 chσ)"): [
        EXPL / f"auto_cross_v2_chsigma/l1_auto_cross/posteriors/l1_auto_cross_s{s}.npy"
        for s in (41, 42, 43)
    ],
    # --- CNN plain ---
    ("CNN plain", "auto-only"): [
        EXPL / f"cnn_extended_train_zm/posteriors/cnn_tomo4_20deg160_nobnt_a2_plain_dense512_step240000_zm_s{s}.npy"
        for s in (41, 42, 43)
    ],
    ("CNN plain", "cross-only (v2)"): [
        EXPL / f"cross_only_campaign_v2_chsigma/cnn_cross_only_plain/dim_10/posteriors/cnn_cross_only_plain_d10_s{s}.npy"
        for s in (41, 42, 43)
    ],
    ("CNN plain", "auto+cross"): [
        EXPL / f"cnn_with_harm_cross_normalized/posteriors/cnn_harm_cross_norm_nobnt_s{s}.npy"
        for s in (41, 42, 43)
    ],
    # --- CNN resnet50 ---  (gn variant for cross-channel cases; stock BN for auto-only)
    ("CNN resnet50", "auto-only (stock BN)"): [
        EXPL / f"cnn_resnet50_zm_sweep/posteriors/cnn_resnet50_zm_nobnt_cdim10_s{s}.npy"
        for s in (42, 43)
    ],
    ("CNN resnet50", "cross-only (v2, gn)"): [
        EXPL / f"cross_only_campaign_v2_chsigma/cnn_cross_only_resnet50_gn/dim_10/posteriors/cnn_cross_only_resnet50_gn_d10_s{s}.npy"
        for s in (41, 42, 43)
    ],
    ("CNN resnet50", "auto+cross (gn)"): [
        EXPL / f"cnn_with_harm_cross_normalized/resnet50_gn/posteriors/cnn_harm_cross_norm_resnet50gn_nobnt_s{s}.npy"
        for s in (41, 42, 43)
    ],
}

def _load_seeds(paths: List[Path]):
    """Return (pooled_samples, list_of_per_seed_samples) or (None, []) if empty."""
    per_seed = []
    for p in paths:
        if not p.exists():
            print(f"  [!] missing: {p}")
            continue
        x = np.load(p, allow_pickle=False)
        if x.shape[0] > 100_000:
            x = x[:100_000]
        per_seed.append(x)
    if not per_seed:
        return None, []
    return np.concatenate(per_seed, axis=0), per_seed

def _fom3(samples: np.ndarray) -> float:
    """3-D FoM3 = 1 / sqrt(det(C_{Omega_m, sigma_8, w_0}))."""
    C = np.cov(samples[:, :3].T)
    return 1.0 / np.sqrt(np.linalg.det(C))

# --- Load all cells ---
print("Loading posteriors...")
loaded: Dict[tuple, np.ndarray] = {}
fom3_pooled: Dict[tuple, float] = {}
fom3_mean: Dict[tuple, float] = {}
fom3_std: Dict[tuple, float] = {}
for key, paths in CELLS.items():
    pooled, per_seed = _load_seeds(paths)
    if pooled is None:
        print(f"  ✗ {key}: no posteriors found, skipping")
        continue
    loaded[key] = pooled
    fom3_pooled[key] = _fom3(pooled)
    per_seed_fom3 = [_fom3(s) for s in per_seed]
    fom3_mean[key] = float(np.mean(per_seed_fom3))
    fom3_std[key] = float(np.std(per_seed_fom3)) if len(per_seed_fom3) > 1 else 0.0
    print(f"  ✓ {key}: shape={pooled.shape}  "
          f"FoM3 pooled={fom3_pooled[key]:.0f}  mean={fom3_mean[key]:.0f} ± {fom3_std[key]:.0f}")
# Use mean-of-seeds as the headline metric (less inflated by inter-seed scatter).
fom3 = fom3_mean

# --- One overlay per probe ---
probes = sorted({k[0] for k in loaded})
cfg_colors = {
    "auto-only": "tab:blue",
    "auto-only (stock BN)": "tab:blue",
    "cross-only (v2 chσ)": "tab:orange",
    "cross-only (v2)": "tab:orange",
    "cross-only (v2, gn)": "tab:orange",
    "auto+cross (v2 chσ)": "tab:green",
    "auto+cross": "tab:green",
    "auto+cross (gn)": "tab:green",
}

for probe in probes:
    cells = [(k, loaded[k]) for k in loaded if k[0] == probe]
    if not cells:
        continue
    mc_list = []
    labels = []
    colors = []
    for (probe_, cfg), s in cells:
        if s.shape[0] > 50000:
            rng = np.random.default_rng(0)
            sel = rng.choice(s.shape[0], 50000, replace=False)
            s = s[sel]
        mc_list.append(MCSamples(
            samples=s, names=PARAM_NAMES, labels=PARAM_NAMES,
            label=f"{cfg}  (FoM3={fom3[(probe_, cfg)]:.0f})",
        ))
        labels.append(f"{cfg}  (FoM3={fom3[(probe_, cfg)]:.0f})")
        colors.append(cfg_colors.get(cfg, "tab:gray"))
    g = gplot.get_subplot_plotter(subplot_size=1.5)
    g.triangle_plot(
        mc_list, filled=True,
        markers=FID_DICT, marker_args={"color": "red", "lw": 1.2},
        contour_colors=colors, legend_labels=labels,
    )
    plt.suptitle(f"Corner — {probe}: auto-only / cross-only / auto+cross", y=1.02)
    name = OUT / f"corner_overlay_{probe.replace(' ', '_').lower()}_configs"
    plt.gcf().savefig(name.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"✓ saved {name}.pdf")

# --- 3-D FoM3 subspace overlay (single overview figure) ---
all_mc, all_labels, all_colors = [], [], []
for (probe, cfg), s in loaded.items():
    if s.shape[0] > 50000:
        rng = np.random.default_rng(0)
        s = s[rng.choice(s.shape[0], 50000, replace=False)]
    all_mc.append(MCSamples(
        samples=s, names=PARAM_NAMES, labels=PARAM_NAMES,
        label=f"{probe} | {cfg}  (FoM3={fom3[(probe, cfg)]:.0f})",
    ))
    all_labels.append(f"{probe} | {cfg}  (FoM3={fom3[(probe, cfg)]:.0f})")
    # color by config (so the 3 colors highlight input mode)
    all_colors.append(cfg_colors.get(cfg, "tab:gray"))

# --- FoM3 table (markdown + json) ---
rows = []
for (probe, cfg), v in sorted(fom3.items()):
    rows.append({"probe": probe, "config": cfg, "fom3": v})
md_lines = [
    "# Probe × Config FoM3 comparison",
    "",
    "FoM3 = 1/sqrt(det C_{Omega_m, sigma_8, w_0}). Per-seed FoM3 mean ± std is the headline metric",
    "(pooled FoM3 is also shown — it can be much lower when seeds disagree on posterior means).",
    "",
    "## Mean-of-seeds FoM3 (headline)",
    "",
    "| probe | auto-only | cross-only | auto+cross |",
    "|---|---:|---:|---:|",
]
probe_table = {}
for k in fom3_mean:
    probe, cfg = k
    probe_table.setdefault(probe, {})[cfg] = (fom3_mean[k], fom3_std[k], fom3_pooled[k])

def _cell_str(triple):
    if triple is None:
        return "—"
    m, s, p = triple
    return f"**{m:.0f}** ± {s:.0f}<br>(pooled {p:.0f})"

for probe, by_cfg in probe_table.items():
    auto = next((v for k, v in by_cfg.items() if k.startswith("auto-only")), None)
    cross = next((v for k, v in by_cfg.items() if k.startswith("cross-only")), None)
    both = next((v for k, v in by_cfg.items() if k.startswith("auto+cross")), None)
    md_lines.append(f"| {probe} | {_cell_str(auto)} | {_cell_str(cross)} | {_cell_str(both)} |")

(OUT / "fom3_probes_configs.md").write_text("\n".join(md_lines))
(OUT / "fom3_probes_configs.json").write_text(json.dumps(
    {f"{k[0]} | {k[1]}": {"fom3_mean_seeds": fom3_mean[k], "fom3_std_seeds": fom3_std[k],
                          "fom3_pooled": fom3_pooled[k]} for k in fom3_mean}, indent=2))
print(f"\nFoM3 table → {OUT}/fom3_probes_configs.{{md,json}}")
print("\n" + "\n".join(md_lines[-len(probe_table)-2:]))
