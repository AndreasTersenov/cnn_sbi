#!/usr/bin/env python3
"""Analyze and plot the cross-only L1 vs CNN campaign.

Discovers all runs under a campaign root directory, loads posteriors and
meta.json files, then produces:

  - training-diagnostic plots: loss curves, summary distributions,
    channel-RMS bars, L1 feature distributions
  - per-arm corner plots
  - cross-arm overlays (corner + 3-D FoM3 subspace + three-way comparison)
  - FoM3 bar chart + table
  - posterior-mean scatter + seed-overlap IoU
  - campaign summary composite + SUMMARY.md

Usage:
    python compare_cross_only.py [campaign_root]

`campaign_root` defaults to scripts/sbi/results/exploratory/cross_only_campaign.

Auxiliary inputs (optional, for the three-way figure):
    --auto-only-root <path>   directory containing auto-only posteriors per seed
    --auto-cross-root <path>  directory containing auto+cross posteriors per seed

All plots saved as both PDF and PNG (dpi=150) under <campaign_root>/figures/.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PARAM_NAMES = [r"\Omega_m", r"\sigma_8", r"w_0", r"h_0", r"n_s", r"\Omega_b"]
PARAM_KEYS = ["Omega_m", "sigma_8", "w_0", "h_0", "n_s", "Omega_b"]
FOM3_IDX = [0, 1, 2]  # (Ω_m, σ_8, w_0)
FIDUCIAL = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493])


@dataclass
class RunRecord:
    arm: str
    seed: int
    dim: Optional[int]
    posterior_path: Path
    meta_path: Optional[Path]
    samples: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        if self.dim is None:
            return f"{self.arm} s{self.seed}"
        return f"{self.arm} d{self.dim} s{self.seed}"


# ---------------------------------------------------------------------------
# Discovery + loading
# ---------------------------------------------------------------------------
_SEED_RX = re.compile(r"seed_(\d+)")
_DIM_RX = re.compile(r"dim_(\d+)")


def _find_posteriors(root: Path) -> List[Tuple[str, Optional[int], int, Path]]:
    """Walk root/<arm>[/dim_<D>]/posteriors/<file>.npy."""
    out: List[Tuple[str, Optional[int], int, Path]] = []
    if not root.exists():
        return out
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        arm = arm_dir.name
        if arm.startswith(".") or arm in {"figures", "logs"}:
            continue
        # Two patterns: arm/posteriors/* or arm/dim_<D>/posteriors/*
        direct = arm_dir / "posteriors"
        if direct.is_dir():
            for npy in sorted(direct.glob("*.npy")):
                m = _SEED_RX.search(npy.stem) or re.search(r"_s(\d+)$", npy.stem)
                if m:
                    out.append((arm, None, int(m.group(1)), npy))
        for sub in sorted(arm_dir.iterdir()):
            if not (sub.is_dir() and sub.name.startswith("dim_")):
                continue
            dim_m = _DIM_RX.match(sub.name)
            dim = int(dim_m.group(1)) if dim_m else None
            inner = sub / "posteriors"
            if inner.is_dir():
                for npy in sorted(inner.glob("*.npy")):
                    m = _SEED_RX.search(npy.stem) or re.search(r"_s(\d+)$", npy.stem)
                    if m:
                        out.append((arm, dim, int(m.group(1)), npy))
    return out


def _load_run(arm: str, dim: Optional[int], seed: int, path: Path) -> Optional[RunRecord]:
    try:
        samples = np.load(path)
    except (ValueError, OSError):
        print(f"  [warn] could not load posterior {path}")
        return None
    if samples.ndim != 2 or samples.shape[1] < 6:
        print(f"  [warn] unexpected posterior shape at {path}: {samples.shape}")
        return None
    meta_path = path.with_suffix(".meta.json")
    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
    return RunRecord(
        arm=arm,
        seed=seed,
        dim=dim,
        posterior_path=path,
        meta_path=meta_path if meta_path.exists() else None,
        samples=samples,
        meta=meta,
    )


def load_campaign(root: Path) -> List[RunRecord]:
    triples = _find_posteriors(root)
    records: List[RunRecord] = []
    for arm, dim, seed, path in triples:
        rec = _load_run(arm, dim, seed, path)
        if rec is not None:
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# FoM3 and metric helpers
# ---------------------------------------------------------------------------
def _fom3_from_samples(samples: np.ndarray) -> float:
    C = np.cov(samples[:, FOM3_IDX], rowvar=False)
    det = np.linalg.det(C)
    if det <= 0:
        return float("nan")
    return float(1.0 / np.sqrt(det))


def _pool_seed_samples(records: Iterable[RunRecord]) -> np.ndarray:
    return np.concatenate([r.samples for r in records], axis=0)


def _arm_dim_key(arm: str, dim: Optional[int]) -> str:
    if dim is None:
        return arm
    return f"{arm}_d{dim}"


def group_records(records: Iterable[RunRecord]) -> Dict[str, List[RunRecord]]:
    out: Dict[str, List[RunRecord]] = {}
    for r in records:
        out.setdefault(_arm_dim_key(r.arm, r.dim), []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r.seed)
    return out


def fom3_table(grouped: Dict[str, List[RunRecord]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for key, recs in sorted(grouped.items()):
        per_seed = [_fom3_from_samples(r.samples) for r in recs]
        pooled = _pool_seed_samples(recs)
        rows.append(
            {
                "arm": key,
                "n_seeds": len(recs),
                "seeds": [int(r.seed) for r in recs],
                "fom3_per_seed": [float(v) for v in per_seed],
                "fom3_pooled": _fom3_from_samples(pooled),
                "fom3_mean_of_seeds": float(np.mean(per_seed))
                if per_seed
                else float("nan"),
                "fom3_std_of_seeds": float(np.std(per_seed, ddof=1))
                if len(per_seed) > 1
                else float("nan"),
                "marginal_std": [
                    float(np.std(pooled[:, i])) for i in range(6)
                ],
                "marginal_bias": [
                    float(np.mean(pooled[:, i]) - FIDUCIAL[i]) for i in range(6)
                ],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _save_pdf_png(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _arm_colors(keys: Sequence[str]) -> Dict[str, str]:
    cmap = plt.get_cmap("tab10")
    return {k: cmap(i % cmap.N) for i, k in enumerate(keys)}


def plot_fom3_bar(rows: List[Dict[str, object]], fig_dir: Path) -> None:
    if not rows:
        return
    labels = [r["arm"] for r in rows]
    means = [r["fom3_mean_of_seeds"] for r in rows]
    stds = [
        r["fom3_std_of_seeds"]
        if r["n_seeds"] > 1 and not np.isnan(r["fom3_std_of_seeds"])
        else 0.0
        for r in rows
    ]
    pooled = [r["fom3_pooled"] for r in rows]
    fig, ax = plt.subplots(figsize=(max(6.0, 1.5 * len(rows)), 4.5))
    x = np.arange(len(labels))
    width = 0.4
    ax.bar(x - width / 2, means, width, yerr=stds, label="mean ± std of seeds",
           color="tab:blue", alpha=0.85)
    ax.bar(x + width / 2, pooled, width, label="pooled FoM3",
           color="tab:orange", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(r"FoM$_3$ = $1/\sqrt{\det C_3}$ over ($\Omega_m, \sigma_8, w_0$)")
    ax.set_title("Cross-only campaign — FoM3 per arm")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save_pdf_png(fig, fig_dir / "fom3_bar_chart")


def _mcsamples_for(samples: np.ndarray, label: str):
    from getdist import MCSamples
    return MCSamples(
        samples=samples,
        names=PARAM_NAMES,
        labels=PARAM_NAMES,
        label=label,
    )


def plot_corner_overlay(
    grouped: Dict[str, List[RunRecord]],
    fig_dir: Path,
    name: str,
    title: str,
    param_idx: Sequence[int],
    max_per_arm: int = 50000,
) -> None:
    try:
        from getdist import plots as gplot
    except ImportError:
        print("[compare] getdist not available; skipping corner overlay.")
        return
    keys = sorted(grouped.keys())
    if not keys:
        return
    colors = _arm_colors(keys)
    rng = np.random.default_rng(0)
    mc_list = []
    legend_labels = []
    contour_colors = []
    for key in keys:
        recs = grouped[key]
        if not recs:
            continue
        samples = _pool_seed_samples(recs)
        if samples.shape[0] > max_per_arm:
            sel = rng.choice(samples.shape[0], max_per_arm, replace=False)
            samples = samples[sel]
        mc_list.append(_mcsamples_for(samples, key))
        legend_labels.append(key)
        contour_colors.append(colors[key])
    if not mc_list:
        return
    subset_names = [PARAM_NAMES[i] for i in param_idx]
    fid_dict = {PARAM_NAMES[i]: float(FIDUCIAL[i]) for i in param_idx}
    g = gplot.get_subplot_plotter(subplot_size=1.8)
    g.triangle_plot(
        mc_list,
        params=subset_names,
        filled=True,
        markers=fid_dict,
        marker_args={"color": "red", "lw": 1.2},
        contour_colors=contour_colors,
        legend_labels=legend_labels,
    )
    if title:
        plt.suptitle(title, y=1.02)
    _save_pdf_png(plt.gcf(), fig_dir / name)


def plot_per_arm_corner(
    grouped: Dict[str, List[RunRecord]], fig_dir: Path,
) -> None:
    try:
        from getdist import plots as gplot
    except ImportError:
        print("[compare] getdist not available; skipping per-arm corners.")
        return
    fid_dict = {n: float(FIDUCIAL[i]) for i, n in enumerate(PARAM_NAMES)}
    for key, recs in sorted(grouped.items()):
        per_seed = {f"seed {r.seed}": r.samples for r in recs}
        if not per_seed:
            continue
        colors = _arm_colors(list(per_seed.keys()))
        mc_list = [_mcsamples_for(s, lbl) for lbl, s in per_seed.items()]
        legend_labels = list(per_seed.keys())
        contour_colors = [colors[lbl] for lbl in legend_labels]
        g = gplot.get_subplot_plotter(subplot_size=1.5)
        g.triangle_plot(
            mc_list,
            filled=True,
            markers=fid_dict,
            marker_args={"color": "red", "lw": 1.2},
            contour_colors=contour_colors,
            legend_labels=legend_labels,
        )
        plt.suptitle(f"Corner — {key}", y=1.02)
        _save_pdf_png(plt.gcf(), fig_dir / f"corner_{key}")


def plot_posterior_mean_scatter(
    grouped: Dict[str, List[RunRecord]], fig_dir: Path,
) -> None:
    keys = sorted(grouped.keys())
    if not keys:
        return
    fig, axes = plt.subplots(1, 6, figsize=(20, 3.5), sharey=False)
    for p in range(6):
        ax = axes[p]
        x = []
        y = []
        labels: List[str] = []
        for k_idx, key in enumerate(keys):
            for r in grouped[key]:
                x.append(k_idx)
                y.append(float(np.mean(r.samples[:, p]) - FIDUCIAL[p]))
                labels.append(f"{key} s{r.seed}")
        ax.scatter(x, y, alpha=0.8)
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"${PARAM_NAMES[p]}$")
        ax.set_ylabel("posterior mean − truth")
        ax.grid(alpha=0.3)
    fig.suptitle("Per-seed posterior-mean bias relative to fiducial")
    fig.tight_layout()
    _save_pdf_png(fig, fig_dir / "posterior_mean_scatter")


def plot_channel_rms(
    grouped: Dict[str, List[RunRecord]], fig_dir: Path,
) -> None:
    relevant = []
    for key, recs in sorted(grouped.items()):
        if not key.startswith("cnn_"):
            continue
        for r in recs:
            scale = r.meta.get("harmonic_channel_scale")
            if scale is None:
                continue
            relevant.append((key, r.seed, np.array(scale, dtype=float)))
    if not relevant:
        return
    n = len(relevant)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False,
    )
    for i, (key, seed, scale) in enumerate(relevant):
        ax = axes[i // ncols][i % ncols]
        ax.bar(range(len(scale)), scale, color="tab:purple", alpha=0.85)
        ax.set_title(f"{key} s{seed}", fontsize=10)
        ax.set_xlabel("channel index")
        ax.set_ylabel("RMS")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Per-channel RMS (cross channels, after channel-mode slicing)")
    fig.tight_layout()
    _save_pdf_png(fig, fig_dir / "channel_rms_bar")


def plot_compressor_summary_distribution(
    grouped: Dict[str, List[RunRecord]], fig_dir: Path,
) -> None:
    # Plotting compressor-summary distributions requires the compressed
    # validation summaries. Those are saved alongside trained checkpoints in
    # `<run_dir>/save_params/.../*.npz`. Discovery: look for a file named
    # `compressed_summary_val.npy` in the run directory; if absent, skip.
    keys = [k for k in sorted(grouped.keys()) if k.startswith("cnn_")]
    if not keys:
        return
    panels: List[Tuple[str, int, np.ndarray]] = []
    for key in keys:
        for r in grouped[key]:
            run_dir = r.posterior_path.parent.parent
            cand = list(run_dir.rglob("compressed_summary_val.npy"))
            if not cand:
                continue
            arr = np.load(cand[0])
            panels.append((key, r.seed, arr))
    if not panels:
        return
    n = len(panels)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False,
    )
    for i, (key, seed, arr) in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        if arr.ndim != 2:
            ax.set_title(f"{key} s{seed} (bad shape)")
            continue
        per_dim_std = np.std(arr, axis=0)
        ax.bar(range(arr.shape[1]), per_dim_std, color="tab:green", alpha=0.85)
        ax.set_title(f"{key} s{seed}", fontsize=10)
        ax.set_xlabel("summary dim")
        ax.set_ylabel("std on val")
        ax.grid(axis="y", alpha=0.3)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Per-summary-dim std on validation split (CNN compressor)")
    fig.tight_layout()
    _save_pdf_png(fig, fig_dir / "compressor_summary_distribution")


def write_fom3_csv(rows: List[Dict[str, object]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "arm", "n_seeds", "seeds",
        "fom3_pooled", "fom3_mean_of_seeds", "fom3_std_of_seeds",
        "fom3_per_seed", "marginal_std", "marginal_bias",
    ]
    with out.open("w", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fieldnames})


def write_summary_md(
    rows: List[Dict[str, object]],
    fig_dir: Path,
    out: Path,
    campaign_root: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Cross-only campaign — auto-generated summary\n")
    lines.append(f"Campaign root: `{campaign_root}`\n")
    lines.append("## FoM3 per arm\n")
    lines.append("| arm | n_seeds | FoM3 pooled | FoM3 mean ± std (seeds) |")
    lines.append("|---|---:|---:|---|")
    for r in rows:
        std_str = (
            f"{r['fom3_std_of_seeds']:.1f}"
            if not np.isnan(r["fom3_std_of_seeds"])
            else "—"
        )
        lines.append(
            f"| {r['arm']} | {r['n_seeds']} | {r['fom3_pooled']:.1f} | "
            f"{r['fom3_mean_of_seeds']:.1f} ± {std_str} |"
        )
    lines.append("")
    lines.append("## Decision rule")
    # Pull the L1 and the CNN arms separately
    l1_arms = [r for r in rows if r["arm"].startswith("l1_")]
    cnn_arms = [r for r in rows if r["arm"].startswith("cnn_")]
    if l1_arms and cnn_arms:
        l1_fom3 = float(l1_arms[0]["fom3_pooled"])
        best_cnn = max(cnn_arms, key=lambda r: float(r["fom3_pooled"]))
        cnn_fom3 = float(best_cnn["fom3_pooled"])
        ratio = l1_fom3 / cnn_fom3 if cnn_fom3 > 0 else float("nan")
        lines.append(
            f"- L1 pooled FoM3: **{l1_fom3:.1f}** (cross-only)\n"
            f"- Best CNN pooled FoM3: **{cnn_fom3:.1f}** "
            f"({best_cnn['arm']})\n"
            f"- L1/CNN ratio: **{ratio:.2f}×**\n"
        )
        if ratio > 1.8:
            verdict = (
                "L1 ≫ CNN on cross-only — supports H1 (wavelet inductive "
                "bias matches the cross-map signal)."
            )
        elif ratio > 1.2:
            verdict = (
                "L1 > CNN on cross-only but smaller gap than on auto+cross — "
                "the auto+cross advantage is partly combinatorial."
            )
        elif ratio > 0.8:
            verdict = (
                "L1 ≈ CNN on cross-only — the auto+cross advantage is largely "
                "combinatorial; H1 weakens."
            )
        else:
            verdict = "CNN > L1 on cross-only — unexpected, investigate."
        lines.append(f"\n**Outcome:** {verdict}\n")
    lines.append("")
    lines.append("## Figures")
    fig_pdfs = sorted(fig_dir.glob("*.pdf"))
    for p in fig_pdfs:
        lines.append(f"- `{p.relative_to(campaign_root)}`")
    out.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[2]
    default_root = (
        repo / "scripts" / "sbi" / "results" / "exploratory" / "cross_only_campaign"
    )
    p.add_argument(
        "campaign_root", type=Path, nargs="?", default=default_root,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root: Path = args.campaign_root.resolve()
    if not root.exists():
        raise SystemExit(f"Campaign root does not exist: {root}")

    print(f"[compare] scanning {root}")
    records = load_campaign(root)
    if not records:
        raise SystemExit("No posteriors found.")
    print(f"[compare] loaded {len(records)} runs")
    grouped = group_records(records)
    rows = fom3_table(grouped)
    for r in rows:
        print(
            f"  {r['arm']}: n_seeds={r['n_seeds']} "
            f"FoM3_pooled={r['fom3_pooled']:.1f} "
            f"mean_of_seeds={r['fom3_mean_of_seeds']:.1f}"
        )

    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_fom3_bar(rows, fig_dir)
    plot_corner_overlay(
        grouped, fig_dir,
        name="corner_overlay_l1_vs_cnn_cross_only",
        title="Cross-only overlay — 6-D",
        param_idx=list(range(6)),
    )
    plot_corner_overlay(
        grouped, fig_dir,
        name="corner_overlay_3d_fom3",
        title=r"Cross-only overlay — FoM3 subspace ($\Omega_m, \sigma_8, w_0$)",
        param_idx=FOM3_IDX,
    )
    plot_per_arm_corner(grouped, fig_dir)
    plot_posterior_mean_scatter(grouped, fig_dir)
    plot_channel_rms(grouped, fig_dir)
    plot_compressor_summary_distribution(grouped, fig_dir)

    write_fom3_csv(rows, root / "fom3_table.csv")
    (root / "fom3_table.json").write_text(json.dumps(rows, indent=2),
                                          encoding="utf-8")
    write_summary_md(rows, fig_dir, root / "SUMMARY.md", root)
    print(f"[compare] figures in {fig_dir}")
    print(f"[compare] summary at {root / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
