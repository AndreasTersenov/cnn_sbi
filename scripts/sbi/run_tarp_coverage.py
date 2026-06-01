#!/usr/bin/env python3
"""Compute TARP joint-coverage curves from per-(arm, seed) posterior dumps and
plot per-arm and overlay figures (3-D and 6-D).

Discovers arms and seeds automatically from the dumps tree
``<dumps-root>/<arm>/seed_<S>/.../posterior_samples.npz``. Each curve uses
the ``tarp`` package's built-in bootstrap to capture within-arm noise;
seed-to-seed scatter is added as a separate axis when aggregating into the
per-arm band.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import tarp  # noqa: E402


PARAMETER_ORDER = ["Omega_m", "sigma_8", "w0", "h0", "n_s", "Omega_b"]
FOM3_INDICES = (0, 1, 2)

DEFAULT_ARM_ORDER = (
    "cnn_auto_only",
    "cnn_harm_cross_plain",
    "cnn_harm_cross_gn",
    "l1_harm_cross",
)
ARM_DISPLAY = {
    "l1_harm_cross": "L1 auto + harm cross",
    "cnn_auto_only": "CNN auto-only (plain)",
    "cnn_harm_cross_plain": "CNN auto + harm cross (plain)",
    "cnn_harm_cross_gn": "CNN auto + harm cross (resnet50-GN)",
    # definitive comparison arms (2026-05-31)
    "l1_autocross": "L1 auto+cross",
    "l1_autoonly": "L1 auto-only",
    "cnn_autocross_rnvp": "CNN auto+cross (RealNVP)",
    "cnn_autoonly_rnvp": "CNN auto-only (RealNVP)",
    "cnn_autocross_maf": "CNN auto+cross (MAF)",
    "cnn_autoonly_maf": "CNN auto-only (MAF)",
}
ARM_COLOR = {
    "l1_harm_cross": "tab:red",
    "cnn_auto_only": "tab:blue",
    "cnn_harm_cross_plain": "tab:green",
    "cnn_harm_cross_gn": "tab:orange",
    # definitive comparison arms (2026-05-31)
    "l1_autocross": "tab:red",
    "l1_autoonly": "tab:pink",
    "cnn_autocross_rnvp": "tab:blue",
    "cnn_autoonly_rnvp": "tab:cyan",
    "cnn_autocross_maf": "tab:green",
    "cnn_autoonly_maf": "tab:olive",
}


@dataclass(frozen=True)
class DumpEntry:
    arm: str
    seed: int
    path: Path


@dataclass(frozen=True)
class CurveRecord:
    arm: str
    seed: int
    dim: int
    alpha: np.ndarray  # (A,)
    ecp_bootstrap: np.ndarray  # (B, A)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_outdir = (
        repo_root / "scripts" / "sbi" / "results" / "diagnostics" / "tarp_harm_cross"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dumps-root", type=Path, default=default_outdir / "dumps")
    p.add_argument("--outdir", type=Path, default=default_outdir)
    p.add_argument("--dims", type=int, nargs="+", default=[3, 6])
    p.add_argument("--num-bootstrap", type=int, default=200)
    p.add_argument("--num-alpha-bins", type=int, default=0, help="0 → tarp default (n_sims // 10)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-norm", action="store_true", help="Disable tarp norm=True")
    p.add_argument(
        "--arms",
        type=str,
        default="",
        help="Optional comma-separated subset (default: all discovered).",
    )
    p.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute curves even if cached npz files already exist.",
    )
    return p.parse_args()


_SEED_RE = re.compile(r"seed_(\d+)$")


def discover_dumps(dumps_root: Path) -> List[DumpEntry]:
    if not dumps_root.exists():
        raise FileNotFoundError(f"Dumps root not found: {dumps_root}")
    entries: List[DumpEntry] = []
    for arm_dir in sorted(p for p in dumps_root.iterdir() if p.is_dir()):
        for dump_path in sorted(arm_dir.rglob("posterior_samples.npz")):
            seed: Optional[int] = None
            for part in dump_path.parts[::-1]:
                m = _SEED_RE.match(part)
                if m is not None:
                    seed = int(m.group(1))
                    break
            if seed is None:
                print(f"[warn] could not infer seed from {dump_path}; skipping")
                continue
            entries.append(DumpEntry(arm=arm_dir.name, seed=seed, path=dump_path))
    return entries


def _slice_dim(samples: np.ndarray, theta: np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    if dim == 3:
        idx = list(FOM3_INDICES)
    elif dim == 6:
        idx = list(range(6))
    else:
        raise ValueError(f"Unsupported --dim {dim}")
    return samples[:, :, idx], theta[:, idx]


def compute_curve(
    dump: DumpEntry,
    dim: int,
    num_bootstrap: int,
    num_alpha_bins: int,
    seed: int,
    norm: bool,
) -> CurveRecord:
    data = np.load(dump.path)
    samples = np.asarray(data["samples"], dtype=np.float32)  # (N, M, D)
    theta = np.asarray(data["theta"], dtype=np.float32)  # (N, D)
    if samples.ndim != 3 or theta.ndim != 2:
        raise ValueError(
            f"Unexpected shapes in {dump.path}: samples {samples.shape}, theta {theta.shape}"
        )
    samples_d, theta_d = _slice_dim(samples, theta, dim)
    # tarp expects (n_samples, n_sims, n_dims) == (M, N, D).
    samples_tarp = np.transpose(samples_d, (1, 0, 2))
    kwargs = dict(
        references="random",
        num_bootstrap=num_bootstrap,
        norm=norm,
        bootstrap=True,
        seed=seed,
    )
    if num_alpha_bins > 0:
        kwargs["num_alpha_bins"] = int(num_alpha_bins)
    ecp, alpha = tarp.get_tarp_coverage(samples_tarp, theta_d, **kwargs)
    return CurveRecord(
        arm=dump.arm,
        seed=dump.seed,
        dim=dim,
        alpha=np.asarray(alpha, dtype=np.float64),
        ecp_bootstrap=np.asarray(ecp, dtype=np.float64),
    )


def _curve_path(curves_dir: Path, arm: str, seed: int, dim: int) -> Path:
    return curves_dir / f"tarp_curve_{arm}_seed{seed}_dim{dim}.npz"


def _load_or_compute(
    dump: DumpEntry,
    dim: int,
    curves_dir: Path,
    args: argparse.Namespace,
) -> CurveRecord:
    cache_path = _curve_path(curves_dir, dump.arm, dump.seed, dim)
    if cache_path.exists() and not args.recompute:
        z = np.load(cache_path)
        return CurveRecord(
            arm=dump.arm,
            seed=dump.seed,
            dim=dim,
            alpha=np.asarray(z["alpha"], dtype=np.float64),
            ecp_bootstrap=np.asarray(z["ecp_bootstrap"], dtype=np.float64),
        )
    record = compute_curve(
        dump=dump,
        dim=dim,
        num_bootstrap=int(args.num_bootstrap),
        num_alpha_bins=int(args.num_alpha_bins),
        seed=int(args.seed),
        norm=(not args.no_norm),
    )
    curves_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        alpha=record.alpha,
        ecp_bootstrap=record.ecp_bootstrap,
        arm=np.array(record.arm),
        seed=np.int32(record.seed),
        dim=np.int32(record.dim),
    )
    return record


def _stack_arm(
    records: List[CurveRecord],
    common_alpha: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Return (alpha, per_seed_median (S,A), seed_stack_for_band (S,A), seeds).

    Each per-seed curve is interpolated onto ``common_alpha`` because tarp's
    ``references='random'`` mode produces empirical alpha grids that differ
    slightly between seeds.
    """
    records = sorted(records, key=lambda r: r.seed)
    per_seed: List[np.ndarray] = []
    for r in records:
        med = np.median(r.ecp_bootstrap, axis=0)
        order = np.argsort(r.alpha)
        per_seed.append(np.interp(common_alpha, r.alpha[order], med[order]))
    per_seed_median = np.stack(per_seed, axis=0)
    return common_alpha, per_seed_median, per_seed_median, [r.seed for r in records]


def _common_alpha_grid(records_by_arm: Dict[str, List[CurveRecord]]) -> np.ndarray:
    sizes = [r.alpha.shape[0] for records in records_by_arm.values() for r in records]
    n = max(sizes)
    return np.linspace(0.0, 1.0, n)


def _arm_order(arms_present: List[str]) -> List[str]:
    known = [a for a in DEFAULT_ARM_ORDER if a in arms_present]
    extras = sorted(a for a in arms_present if a not in DEFAULT_ARM_ORDER)
    return known + extras


def plot_per_arm(
    records_by_arm: Dict[str, List[CurveRecord]],
    dim: int,
    outpath_pdf: Path,
    outpath_png: Path,
    common_alpha: np.ndarray,
) -> None:
    arms = _arm_order(list(records_by_arm.keys()))
    n = len(arms)
    nrows = 2 if n > 2 else 1
    ncols = int(np.ceil(n / nrows))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4.2 * nrows), constrained_layout=True, squeeze=False
    )
    flat = axes.flat
    for ax, arm in zip(flat, arms):
        records = records_by_arm[arm]
        alpha, per_seed_median, _, seeds = _stack_arm(records, common_alpha)
        color = ARM_COLOR.get(arm, "tab:gray")
        for med in per_seed_median:
            ax.plot(alpha, med, color=color, alpha=0.35, linewidth=1.0)
        arm_median = np.median(per_seed_median, axis=0)
        if per_seed_median.shape[0] >= 2:
            lo, hi = np.percentile(per_seed_median, [16, 84], axis=0)
            ax.fill_between(alpha, lo, hi, color=color, alpha=0.20, linewidth=0)
        ax.plot(alpha, arm_median, color=color, linewidth=2.2, label="seed-median")
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Nominal credibility α")
        ax.set_ylabel("Expected coverage")
        ax.set_title(f"{ARM_DISPLAY.get(arm, arm)}\nN seeds = {len(seeds)}")
        ax.grid(alpha=0.3)
    for ax in flat[n:]:
        ax.set_visible(False)
    fig.suptitle(f"TARP coverage — {dim}-D ({'Ω_m,σ_8,w_0' if dim == 3 else 'all 6 params'})")
    outpath_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath_pdf, dpi=150)
    fig.savefig(outpath_png, dpi=150)
    plt.close(fig)


def plot_overlay(
    records_by_arm: Dict[str, List[CurveRecord]],
    dim: int,
    outpath_pdf: Path,
    outpath_png: Path,
    common_alpha: np.ndarray,
) -> None:
    arms = _arm_order(list(records_by_arm.keys()))
    fig, ax = plt.subplots(figsize=(6.0, 5.4), constrained_layout=True)
    for arm in arms:
        records = records_by_arm[arm]
        alpha, per_seed_median, _, seeds = _stack_arm(records, common_alpha)
        color = ARM_COLOR.get(arm, "tab:gray")
        arm_median = np.median(per_seed_median, axis=0)
        if per_seed_median.shape[0] >= 2:
            lo, hi = np.percentile(per_seed_median, [16, 84], axis=0)
            ax.fill_between(alpha, lo, hi, color=color, alpha=0.18, linewidth=0)
        ax.plot(
            alpha, arm_median, color=color, linewidth=2.2,
            label=f"{ARM_DISPLAY.get(arm, arm)} (n={len(seeds)})",
        )
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Nominal credibility α")
    ax.set_ylabel("Expected coverage")
    ax.set_title(
        f"TARP joint coverage — {dim}-D "
        f"({'Ω_m,σ_8,w_0' if dim == 3 else 'all 6 params'})"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    outpath_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath_pdf, dpi=150)
    fig.savefig(outpath_png, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dumps_root = args.dumps_root.resolve()
    outdir = args.outdir.resolve()
    curves_dir = outdir / "curves"
    figures_dir = outdir / "figures"

    entries = discover_dumps(dumps_root)
    if args.arms.strip():
        keep = {a.strip() for a in args.arms.split(",") if a.strip()}
        entries = [e for e in entries if e.arm in keep]
    if not entries:
        raise SystemExit(f"No posterior dumps found under {dumps_root}")

    print(f"[tarp] Discovered {len(entries)} dumps across arms: "
          f"{sorted({e.arm for e in entries})}")

    summary_index: Dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dumps_root": str(dumps_root),
        "n_dumps": len(entries),
        "dims": list(args.dims),
        "num_bootstrap": int(args.num_bootstrap),
        "num_alpha_bins": int(args.num_alpha_bins) if args.num_alpha_bins > 0 else None,
        "norm": (not args.no_norm),
        "curves": [],
    }

    for dim in args.dims:
        records_by_arm: Dict[str, List[CurveRecord]] = {}
        for entry in entries:
            try:
                record = _load_or_compute(entry, dim, curves_dir, args)
            except Exception as exc:
                print(f"[error] {entry.arm} seed={entry.seed} dim={dim}: {exc}")
                continue
            records_by_arm.setdefault(entry.arm, []).append(record)
            summary_index["curves"].append(
                {
                    "arm": entry.arm,
                    "seed": int(entry.seed),
                    "dim": int(dim),
                    "dump": str(entry.path),
                    "curve": str(_curve_path(curves_dir, entry.arm, entry.seed, dim)),
                }
            )
        if not records_by_arm:
            print(f"[warn] no records for dim={dim}; skipping plots")
            continue
        common_alpha = _common_alpha_grid(records_by_arm)
        plot_per_arm(
            records_by_arm=records_by_arm,
            dim=dim,
            outpath_pdf=figures_dir / f"tarp_per_arm_dim{dim}.pdf",
            outpath_png=figures_dir / f"tarp_per_arm_dim{dim}.png",
            common_alpha=common_alpha,
        )
        plot_overlay(
            records_by_arm=records_by_arm,
            dim=dim,
            outpath_pdf=figures_dir / f"tarp_overlay_dim{dim}.pdf",
            outpath_png=figures_dir / f"tarp_overlay_dim{dim}.png",
            common_alpha=common_alpha,
        )
        print(f"[tarp] dim={dim}: plotted per-arm + overlay (arms: "
              f"{sorted(records_by_arm.keys())})")

    summary_path = outdir / "tarp_summary.json"
    summary_path.write_text(json.dumps(summary_index, indent=2), encoding="utf-8")
    print(f"[tarp] Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
