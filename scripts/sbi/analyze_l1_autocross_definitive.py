#!/usr/bin/env python3
"""Analyze the L1 auto+cross definitive arms (flip=False, dedup path).

Layout consumed (flat per-arm dirs, seed+perm in filename):
    definitive_comparison/posteriors/<arm>/<arm>_s<seed>_p<perm>.npy

Arms analyzed:
    l1_autocross_fulltrain   (arm 1, --nde-train-split train)
    l1_autocross_split70     (arm 2, --nde-train-split train[70%:])
    l1_auto_fulltrain        (baseline, flip=True; pooled FoM3 10,452)
    l1_auto_split70          (baseline, flip=True; pooled FoM3  8,086)

Primary metric (per the fiber): 3-seed POOLED FoM3 on (Ωm, σ8, w0), per perm,
then perm-averaged. Also reports per-run, mean-of-seeds±std, marginal σ (6),
and bias. Writes a markdown summary + CSV, and getdist corner overlays:
    (a) autocross full vs split70
    (b) autocross_fulltrain vs auto_fulltrain   (cross-channel gain, full)
    (c) autocross_split70  vs auto_split70      (cross-channel gain, 70/30)

Robust to missing arms / partial runs so it can be run incrementally.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PARAM_NAMES = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]
PARAM_KEYS = ["Omega_m", "sigma_8", "w_0", "h_0", "n_s", "Omega_b"]
FOM3_IDX = [0, 1, 2]
FIDUCIAL = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493])

OUT = Path("scripts/sbi/results/exploratory/definitive_comparison")
POST = OUT / "posteriors"
FIGDIR = OUT / "figures" / "definitive_l1"
ARMS = [
    "l1_autocross_fulltrain",
    "l1_autocross_split70",
    "l1_autoonly_fulltrain",   # route-matched flip=False auto-only (clean baseline)
    "l1_autoonly_split70",
    "l1_auto_fulltrain",       # original TFDS-route, flip=True baseline (reference)
    "l1_auto_split70",
]
BASELINE_FOM3 = {"l1_auto_fulltrain": 10452, "l1_auto_split70": 8086}

_SP_RX = re.compile(r"_s(\d+)_p(\d+)$")
_S_RX = re.compile(r"_s(\d+)$")


def fom3(samples: np.ndarray) -> float:
    C = np.cov(samples[:, FOM3_IDX], rowvar=False)
    det = np.linalg.det(C)
    return float(1.0 / np.sqrt(det)) if det > 0 else float("nan")


_PAIRS = [(0, 1), (0, 2), (1, 2)]  # (Om,s8),(Om,w0),(s8,w0)


def fom2d(samples: np.ndarray) -> dict:
    """2D FoM = 1/sqrt(det C_2) per pair. More reliable than FoM3 (which amplifies
    small correlation changes; see feedback_fom3_fragile_use_2d_areas)."""
    out = {}
    for i, j in _PAIRS:
        C = np.cov(samples[:, [i, j]], rowvar=False)
        det = np.linalg.det(C)
        out[f"{PARAM_KEYS[i]}_{PARAM_KEYS[j]}"] = (
            float(1.0 / np.sqrt(det)) if det > 0 else float("nan"))
    return out


def load_arm(arm: str) -> Dict[tuple, np.ndarray]:
    """Return {(seed, perm): samples}. perm defaults to 0 if filename has no _p."""
    d: Dict[tuple, np.ndarray] = {}
    arm_dir = POST / arm
    if not arm_dir.is_dir():
        return d
    for npy in sorted(arm_dir.glob(f"{arm}_s*.npy")):
        m = _SP_RX.search(npy.stem)
        if m:
            seed, perm = int(m.group(1)), int(m.group(2))
        else:
            ms = _S_RX.search(npy.stem)
            if not ms:
                continue
            seed, perm = int(ms.group(1)), 0
        try:
            s = np.load(npy)
        except (ValueError, OSError):
            print(f"  [warn] unreadable {npy}")
            continue
        if s.ndim == 2 and s.shape[1] >= 6:
            d[(seed, perm)] = s
    return d


def summarize_arm(arm: str, runs: Dict[tuple, np.ndarray]) -> dict:
    if not runs:
        return {"arm": arm, "n_runs": 0}
    per_run = {f"s{s}_p{p}": fom3(v) for (s, p), v in sorted(runs.items())}
    # All secondary metrics use the SAME pooling as FoM3: pool the seeds within
    # each perm (same obs point), compute the metric, then average across perms.
    # (Pooling across perms mixes different obs realizations -> broadens marginals
    #  and is not comparable to the per-perm FoM3. That was the original bug.)
    by_perm: Dict[int, List[np.ndarray]] = defaultdict(list)
    for (s, p), v in runs.items():
        by_perm[p].append(v)
    perm_pooled_samples = {p: np.concatenate(vs, 0) for p, vs in sorted(by_perm.items())}
    perm_pooled = {p: fom3(s) for p, s in perm_pooled_samples.items()}
    perm_avg = float(np.nanmean(list(perm_pooled.values())))
    # per-perm 2D FoM / σ / bias, then perm-average
    f2d_keys = list(fom2d(next(iter(perm_pooled_samples.values()))).keys())
    f2d_acc = {k: [] for k in f2d_keys}
    sig_acc = {k: [] for k in PARAM_KEYS}
    bias_acc = {k: [] for k in PARAM_KEYS}
    for s in perm_pooled_samples.values():
        for k, v in fom2d(s).items():
            f2d_acc[k].append(v)
        for i, k in enumerate(PARAM_KEYS):
            sig_acc[k].append(float(np.std(s[:, i])))
            bias_acc[k].append(float(np.mean(s[:, i]) - FIDUCIAL[i]))
    per_run_vals = np.array(list(per_run.values()), dtype=float)
    return {
        "arm": arm,
        "n_runs": len(runs),
        "seeds": sorted({s for s, _ in runs}),
        "perms": sorted({p for _, p in runs}),
        "fom3_per_run": per_run,
        "fom3_per_perm_pooled": perm_pooled,
        "fom3_perm_avg_pooled": perm_avg,        # <-- PRIMARY
        "fom3_mean_of_runs": float(np.nanmean(per_run_vals)),
        "fom3_std_of_runs": float(np.nanstd(per_run_vals, ddof=1)) if len(per_run_vals) > 1 else float("nan"),
        "fom2d_perm_avg": {k: float(np.nanmean(v)) for k, v in f2d_acc.items()},
        "marginal_sigma": {k: float(np.mean(v)) for k, v in sig_acc.items()},
        "marginal_bias": {k: float(np.mean(v)) for k, v in bias_acc.items()},
    }


def pool_perm0_over_seeds(runs: Dict[tuple, np.ndarray]) -> Optional[np.ndarray]:
    """Pool the 3 seeds at perm 0 (same obs across arms) for overlay contours."""
    vs = [v for (s, p), v in runs.items() if p == 0]
    if not vs:  # fall back to whatever perm is available, pooled by seeds
        vs = list(runs.values())
    return np.concatenate(vs, 0) if vs else None


def _mc(samples: np.ndarray, label: str):
    from getdist import MCSamples
    return MCSamples(samples=samples[:, :6], names=PARAM_KEYS, labels=PARAM_NAMES,
                     label=label)


def corner_overlay(pairs: Sequence[tuple], title: str, fname: str) -> None:
    """pairs: list of (samples, label, color). Saves PDF+PNG."""
    try:
        from getdist import plots as gplot
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] getdist unavailable, skipping {fname}: {e}")
        return
    mcs, colors = [], []
    for s, lab, col in pairs:
        if s is None:
            continue
        mcs.append(_mc(s, lab))
        colors.append(col)
    if not mcs:
        return
    g = gplot.get_subplot_plotter(width_inch=8)
    g.settings.alpha_filled_add = 0.5
    g.triangle_plot(mcs, PARAM_KEYS, filled=True, contour_colors=colors,
                    markers={k: FIDUCIAL[i] for i, k in enumerate(PARAM_KEYS)},
                    legend_labels=[m.getLabel() for m in mcs])
    g.fig.suptitle(title, y=1.02)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        g.export(str(FIGDIR / f"{fname}.{ext}"))
    print(f"  [fig] {FIGDIR / fname}.pdf")


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    loaded = {arm: load_arm(arm) for arm in ARMS}
    summaries = {arm: summarize_arm(arm, loaded[arm]) for arm in ARMS}

    # ---- markdown + CSV ----
    lines = ["# L1 auto+cross definitive arms — flip=False / dedup path\n"]
    lines.append("Primary metric: **3-seed pooled FoM3 on (Ωm, σ8, w0), perm-averaged.**\n")
    lines.append("| arm | n_runs | FoM3 (perm-avg pooled) | FoM3 mean-of-runs ± std | σ(Ωm) | σ(σ8) | σ(w0) |")
    lines.append("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        s = summaries[arm]
        if s["n_runs"] == 0:
            lines.append(f"| {arm} | 0 | — (no runs yet) | — | — | — | — |")
            continue
        ms = s["marginal_sigma"]
        base = f"  (baseline ref {BASELINE_FOM3[arm]})" if arm in BASELINE_FOM3 else ""
        lines.append(
            f"| {arm} | {s['n_runs']} | **{s['fom3_perm_avg_pooled']:.0f}**{base} | "
            f"{s['fom3_mean_of_runs']:.0f} ± {s['fom3_std_of_runs']:.0f} | "
            f"{ms['Omega_m']:.4f} | {ms['sigma_8']:.4f} | {ms['w_0']:.4f} |"
        )
    # 2D FoM table (more reliable than FoM3; perm-averaged, same pooling)
    lines.append("\n## 2D FoM per pair (1/√detC₂; perm-avg pooled — more reliable than FoM3)\n")
    lines.append("| arm | (Ωm,σ8) | (Ωm,w0) | (σ8,w0) |")
    lines.append("|---|---|---|---|")
    for arm in ARMS:
        s = summaries[arm]
        if s["n_runs"] == 0:
            continue
        f2 = s["fom2d_perm_avg"]
        lines.append(f"| {arm} | {f2['Omega_m_sigma_8']:.0f} | {f2['Omega_m_w_0']:.0f} | "
                     f"{f2['sigma_8_w_0']:.0f} |")
    # cross-channel gain — FoM3 AND 2D FoM (2D is the trustworthy read)
    def gain_line(ac, base, base_desc):
        r = ac["fom3_perm_avg_pooled"] / base["fom3_perm_avg_pooled"]
        f2a, f2u = ac["fom2d_perm_avg"], base["fom2d_perm_avg"]
        r2 = {k: f2a[k] / f2u[k] for k in f2a}
        sa, su = ac["marginal_sigma"], base["marginal_sigma"]
        rs = {k: su[k] / sa[k] for k in ("Omega_m", "sigma_8", "w_0")}  # >1 => cross tighter
        return (f"  - vs {base_desc}: FoM3 **{r:.2f}×**; 2D FoM ×: "
                f"(Ωm,σ8) {r2['Omega_m_sigma_8']:.2f}, (Ωm,w0) {r2['Omega_m_w_0']:.2f}, "
                f"(σ8,w0) {r2['sigma_8_w_0']:.2f}; σ-tighten (>1=cross better): "
                f"Ωm {rs['Omega_m']:.2f}, σ8 {rs['sigma_8']:.2f}, w0 {rs['w_0']:.2f}")
    lines.append("\n## Cross-channel gain (auto+cross vs auto-only)\n")
    lines.append("⚠️ FoM3 amplifies 3D correlation changes; trust the 2D-FoM and σ ratios more.\n")
    lines.append("**CLEAN (route-matched)** = vs `l1_autoonly_*` (same harmonic route + flip=False). "
                 "**ref** = vs `l1_auto_*` (TFDS route, flip=True — flip/route-inconsistent).\n")
    for sfx in ("fulltrain", "split70"):
        ac = summaries[f"l1_autocross_{sfx}"]
        if not ac["n_runs"]:
            continue
        lines.append(f"- **{sfx}**:")
        ao = summaries.get(f"l1_autoonly_{sfx}", {"n_runs": 0})
        if ao["n_runs"]:
            lines.append(gain_line(ac, ao, "auto-only route-matched flip=False **[CLEAN]**"))
        au = summaries[f"l1_auto_{sfx}"]
        if au["n_runs"]:
            lines.append(gain_line(ac, au, "auto-only TFDS flip=True [ref]"))
    # split penalty within auto+cross
    af, a7 = summaries["l1_autocross_fulltrain"], summaries["l1_autocross_split70"]
    if af["n_runs"] and a7["n_runs"]:
        pen = 1 - a7["fom3_perm_avg_pooled"] / af["fom3_perm_avg_pooled"]
        lines.append(f"- **auto+cross split penalty (full→70/30)**: "
                     f"{af['fom3_perm_avg_pooled']:.0f} → {a7['fom3_perm_avg_pooled']:.0f} "
                     f"({pen*100:.0f}%)")
    lines.append("\n## Per-run / per-perm detail\n")
    lines.append("```json")
    lines.append(json.dumps({a: summaries[a] for a in ARMS}, indent=2))
    lines.append("```")
    (OUT / "DEFINITIVE_L1_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[summary] wrote {OUT / 'DEFINITIVE_L1_SUMMARY.md'}")

    with (OUT / "definitive_l1_fom3.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "n_runs", "fom3_perm_avg_pooled", "fom3_mean_of_runs",
                    "fom3_std_of_runs", "sigma_Om", "sigma_s8", "sigma_w0"])
        for arm in ARMS:
            s = summaries[arm]
            if s["n_runs"] == 0:
                continue
            ms = s["marginal_sigma"]
            w.writerow([arm, s["n_runs"], f"{s['fom3_perm_avg_pooled']:.2f}",
                        f"{s['fom3_mean_of_runs']:.2f}", f"{s['fom3_std_of_runs']:.2f}",
                        f"{ms['Omega_m']:.5f}", f"{ms['sigma_8']:.5f}", f"{ms['w_0']:.5f}"])

    # ---- corner overlays ----
    p = {arm: pool_perm0_over_seeds(loaded[arm]) for arm in ARMS}
    corner_overlay(
        [(p["l1_autocross_fulltrain"], "auto+cross full", "#1f77b4"),
         (p["l1_autocross_split70"], "auto+cross 70/30", "#d62728")],
        "L1 auto+cross: full-train vs 70/30 (perm0, 3-seed pooled)",
        "autocross_full_vs_split70",
    )
    corner_overlay(
        [(p["l1_auto_fulltrain"], "auto-only full", "#7f7f7f"),
         (p["l1_autocross_fulltrain"], "auto+cross full", "#1f77b4")],
        "Cross-channel gain (full-train): auto-only vs auto+cross",
        "gain_fulltrain_auto_vs_autocross",
    )
    corner_overlay(
        [(p["l1_auto_split70"], "auto-only 70/30", "#bcbd22"),
         (p["l1_autocross_split70"], "auto+cross 70/30", "#d62728")],
        "Cross-channel gain (70/30): auto-only vs auto+cross",
        "gain_split70_auto_vs_autocross",
    )
    # CLEAN route-matched gain (auto+cross vs auto-only, both harmonic + flip=False)
    corner_overlay(
        [(p.get("l1_autoonly_fulltrain"), "auto-only full (route-matched)", "#2ca02c"),
         (p["l1_autocross_fulltrain"], "auto+cross full", "#1f77b4")],
        "Cross-channel gain CLEAN (full-train, route-matched flip=False)",
        "gain_fulltrain_routematched",
    )
    corner_overlay(
        [(p.get("l1_autoonly_split70"), "auto-only 70/30 (route-matched)", "#2ca02c"),
         (p["l1_autocross_split70"], "auto+cross 70/30", "#d62728")],
        "Cross-channel gain CLEAN (70/30, route-matched flip=False)",
        "gain_split70_routematched",
    )
    print("[done] analysis complete.")


if __name__ == "__main__":
    main()
