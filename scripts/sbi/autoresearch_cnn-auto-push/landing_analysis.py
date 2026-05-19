#!/usr/bin/env python3
"""Landing analysis for a cnn-auto-push iteration.

One-shot wrapper that computes everything the next Ralph iter needs after
training lands: MoS / pooled FoM3, joint_R + per-param drift, compressor
health verdict, and the constitution's amended 3-component cross-method
check (pooled ratio >= 0.35 AND |dJoint_R|/joint_R_L1 <= 0.25 AND MoS
ratio >= 0.40). Writes a per-iter JSON to <iter-dir>/landing.json and
prints a one-screen summary.

Usage:
    python landing_analysis.py --iter-dir <NOTES_DIR>/runs/<tag>/iter-<n>
    python landing_analysis.py --iter-dir iter-23 \
        --pred-pooled-pct=0,5 --pred-mos-pct=-5,5 \
        --ref-iter iter-20 --ref-pooled 13944 --ref-mos 18673

Use the `=` form for any range that starts with a negative number
(argparse otherwise treats `-5,5` as an option flag and errors out).

The script does NOT render the overlay PDF; for that, still call
render_overlay.py. landing_analysis.py is the numeric / verdict side.

Reference defaults (L1 auto+cross from the project headline run):
    --l1-glob: scripts/sbi/results/exploratory/auto_cross_v2_chsigma/
               l1_auto_cross/posteriors/l1_auto_cross_s4?.npy
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np


PARAMS_3D = ("Omega_m", "sigma_8", "w_0")

DEFAULT_L1_GLOB = (
    "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/"
    "auto_cross_v2_chsigma/l1_auto_cross/posteriors/l1_auto_cross_s4?.npy"
)

GUARD_FLOOR = 11_000  # 60k working floor per STATUS.md
POOLED_RATIO_MIN = 0.35
JOINT_R_REL_MAX = 0.25
MOS_RATIO_MIN = 0.40


def fom3(samples: np.ndarray) -> float:
    C = np.cov(samples[:, :3].T)
    return float(1.0 / np.sqrt(np.linalg.det(C)))


def analyze_posteriors(glob_pattern: str, cap_per_seed: int = 100_000) -> dict | None:
    paths = sorted(glob.glob(glob_pattern))
    if not paths:
        return None
    per_seed_means: list[np.ndarray] = []
    per_seed_stds: list[np.ndarray] = []
    per_seed_fom3: list[float] = []
    arrays: list[np.ndarray] = []
    seed_keys: list[str] = []
    for p in paths:
        try:
            x = np.load(p, allow_pickle=False)
        except (FileNotFoundError, OSError) as e:
            print(f"  [skip] {p}: {e}", file=sys.stderr)
            continue
        if x.ndim != 2 or x.shape[1] < 3:
            print(f"  [skip] {p}: bad shape {x.shape}", file=sys.stderr)
            continue
        if x.shape[0] > cap_per_seed:
            x = x[:cap_per_seed]
        sub = x[:, :3]
        per_seed_means.append(sub.mean(axis=0))
        per_seed_stds.append(sub.std(axis=0))
        per_seed_fom3.append(fom3(x))
        arrays.append(sub)
        m = re.search(r"_s(\d{2})\.npy$", p)
        seed_keys.append(m.group(1) if m else Path(p).stem)
    if not arrays:
        return None

    means_arr = np.asarray(per_seed_means)
    stds_arr = np.asarray(per_seed_stds)
    drift = means_arr.std(axis=0, ddof=0)
    width = stds_arr.mean(axis=0)
    per_param_R = drift / width
    joint_R = float(np.sqrt(np.sum(drift ** 2)) / np.sqrt(np.sum(width ** 2)))

    pooled = np.concatenate(arrays, axis=0)
    mos_fom3 = float(np.mean(per_seed_fom3))
    pooled_fom3 = fom3(pooled)
    return {
        "n_seeds": len(arrays),
        "seeds": seed_keys,
        "per_seed_fom3": {k: float(v) for k, v in zip(seed_keys, per_seed_fom3)},
        "mos_fom3": mos_fom3,
        "pooled_fom3": pooled_fom3,
        "per_seed_min": float(min(per_seed_fom3)),
        "per_seed_std": float(np.std(per_seed_fom3, ddof=0)),
        "per_seed_cov_pct": float(100.0 * np.std(per_seed_fom3, ddof=0) / mos_fom3),
        "pooled_over_mos": float(pooled_fom3 / mos_fom3),
        "per_param_R": dict(zip(PARAMS_3D, per_param_R.tolist())),
        "joint_R": joint_R,
        "centroid_scatter_per_param": dict(zip(PARAMS_3D, drift.tolist())),
        "avg_width_per_param": dict(zip(PARAMS_3D, width.tolist())),
    }


def compressor_health(iter_dir: Path) -> dict:
    """Eyeball compressor log for nan/inf, completion, best-val step position.

    The compressor prints `Step N | train | test` lines at each validation
    interval. The best-val checkpoint is the argmin of test loss across
    those lines. The compressor returns the LAST-step params, so a large
    argmin-to-final gap is the F1-lever signature
    ([[cnn-auto-compressor-last-not-best-ckpt]]).

    The runner also prints `Saved @ step N. Val loss = X` on the FIRST
    val-improvement only (per the current inner script), so we don't rely
    on it; we re-derive the best-step from the Step-line stream.

    Fallback (Ralph iter-14, 2026-05-19): when the .log file is missing
    the printable step lines (CUDA/XLA noise can swamp them, or
    --skip-compressor runs have no log), reconstruct from
    `loss_compressor_test.npy` + run_manifest.json. The .npy stores one
    test-loss value per `--compressor-save-every` step (3000 by default
    in run_arm.py).
    """
    log = iter_dir / "logs" / "compressor.log"
    if not log.exists():
        # Try .npy fallback before declaring missing.
        npy_health = _compressor_health_from_npy(iter_dir)
        if npy_health is not None:
            npy_health["log_path"] = str(log) + " (absent; npy fallback)"
            return npy_health
        return {"verdict": "MISSING_LOG", "log_path": str(log)}

    try:
        with open(log, "rb") as f:
            text = f.read().decode("utf-8", errors="replace")
    except OSError as e:
        return {"verdict": "READ_ERROR", "error": str(e)}

    nan_inf = False
    last_step = None
    total_steps = None
    best_step: int | None = None
    best_test: float | None = None
    final_test: float | None = None

    step_re = re.compile(
        r"Step\s+(\d+)\s*\|\s*train\s+(-?\d+\.\d+)\s*\|\s*test\s+(-?\d+\.\d+)"
    )

    for line in text.splitlines():
        low = line.lower()
        if any(w in low for w in (" nan ", "nan,", "nan]", "loss=nan",
                                   " inf ", "inf,", "inf]", "loss=inf")):
            nan_inf = True
        m_step_line = step_re.search(line)
        if m_step_line:
            step = int(m_step_line.group(1))
            tst = float(m_step_line.group(3))
            final_test = tst
            if best_test is None or tst < best_test:
                best_test = tst
                best_step = step
        m_total = re.search(r"steps=(\d+)", line)
        if m_total:
            try:
                total_steps = int(m_total.group(1))
            except ValueError:
                pass
        m_prog = re.search(r"(\d+)/(\d+)\s*\[", line)
        if m_prog:
            try:
                last_step = int(m_prog.group(1))
                if total_steps is None:
                    total_steps = int(m_prog.group(2))
            except ValueError:
                pass

    completed = (
        total_steps is not None and last_step is not None
        and last_step >= int(0.99 * total_steps)
    )
    best_frac = (
        (best_step / total_steps) if (best_step and total_steps) else None
    )
    argmin_to_final_gap = (
        None if best_test is None or final_test is None
        else float(final_test - best_test)
    )

    verdict = "PASS"
    notes = []
    if nan_inf:
        verdict = "FAIL_NAN_INF"
        notes.append("nan or inf detected in log")
    if not completed:
        verdict = "FAIL_INCOMPLETE" if verdict == "PASS" else verdict
        notes.append(
            f"training not completed: last_step={last_step}/{total_steps}"
        )
    if best_frac is not None and best_frac < 0.33:
        notes.append(
            f"best-val ckpt in first third (step {best_step} / {total_steps})"
        )
    if argmin_to_final_gap is not None and argmin_to_final_gap > 0.20:
        notes.append(
            f"argmin-to-final test-loss gap = {argmin_to_final_gap:.3f} nats — "
            "F1 lever may help"
        )

    # If the log was unparseable (no step lines + log file exists), try .npy.
    if best_step is None and final_test is None:
        npy_health = _compressor_health_from_npy(iter_dir)
        if npy_health is not None:
            npy_health["log_path"] = str(log) + " (unparseable; npy fallback)"
            return npy_health

    return {
        "verdict": verdict,
        "log_path": str(log),
        "last_step": last_step,
        "total_steps": total_steps,
        "best_val_step": best_step,
        "best_val_test_loss": best_test,
        "final_test_loss": final_test,
        "argmin_to_final_gap_nats": argmin_to_final_gap,
        "best_val_position_pct": (
            None if best_frac is None else round(100 * best_frac, 1)
        ),
        "notes": notes,
    }


def _compressor_health_from_npy(iter_dir: Path) -> dict | None:
    """Fallback: reconstruct compressor_health from loss_compressor_test.npy.

    Used when compressor.log is missing or contains only CUDA/XLA noise.
    The .npy stores one test-loss value per --compressor-save-every step
    (3000 in run_arm.py); total_steps is in run_manifest.json.

    Returns None if neither the npy nor manifest is reachable, so the
    caller can fall through to MISSING_LOG.
    """
    import numpy as np

    npys = list(iter_dir.glob(
        "compressor/**/loss_compressor_test.npy"
    ))
    if not npys:
        return None
    npy = sorted(npys)[0]
    try:
        a = np.load(npy)
    except Exception:
        return None
    if a.size == 0:
        return None

    # run_arm.py hard-codes --compressor-save-every 3000.
    save_every = 3000
    manifest = iter_dir / "run_manifest.json"
    total_steps = None
    if manifest.exists():
        try:
            with open(manifest) as f:
                m = json.load(f)
            total_steps = m.get("compressor_steps")
        except (json.JSONDecodeError, OSError):
            pass

    argmin = int(a.argmin())
    best_step = (argmin + 1) * save_every
    best_test = float(a.min())
    final_test = float(a[-1])
    last_step = len(a) * save_every
    argmin_to_final_gap = final_test - best_test
    best_frac = (best_step / total_steps) if total_steps else None
    completed = (
        total_steps is not None
        and last_step >= int(0.99 * total_steps)
    )

    verdict = "PASS"
    notes = ["reconstructed from loss_compressor_test.npy"]
    if not completed:
        verdict = "FAIL_INCOMPLETE"
        notes.append(
            f"training not completed: last_step={last_step}/{total_steps}"
        )
    if best_frac is not None and best_frac < 0.33:
        notes.append(
            f"best-val ckpt in first third (step {best_step} / {total_steps})"
        )
    if argmin_to_final_gap > 0.20:
        notes.append(
            f"argmin-to-final test-loss gap = {argmin_to_final_gap:.3f} nats — "
            "F1 lever may help"
        )

    return {
        "verdict": verdict,
        "npy_path": str(npy),
        "last_step": last_step,
        "total_steps": total_steps,
        "best_val_step": best_step,
        "best_val_test_loss": best_test,
        "final_test_loss": final_test,
        "argmin_to_final_gap_nats": argmin_to_final_gap,
        "best_val_position_pct": (
            None if best_frac is None else round(100 * best_frac, 1)
        ),
        "notes": notes,
    }


def amended_cross_method_check(cnn: dict, l1: dict) -> dict:
    """Constitution's amended 3-component shape+scale check.

    See [[cnn-auto-pooled-ratio-amendment-rationale]] (Ralph iter-12).
    Replaces the original single pooled CNN/L1 >= 0.5 threshold.
    """
    pooled_ratio = cnn["pooled_fom3"] / l1["pooled_fom3"]
    mos_ratio = cnn["mos_fom3"] / l1["mos_fom3"]
    djoint = abs(cnn["joint_R"] - l1["joint_R"])
    djoint_rel = djoint / l1["joint_R"] if l1["joint_R"] > 0 else float("inf")

    c1 = pooled_ratio >= POOLED_RATIO_MIN
    c2 = djoint_rel <= JOINT_R_REL_MAX
    c3 = mos_ratio >= MOS_RATIO_MIN
    pass_all = c1 and c2 and c3

    return {
        "pooled_ratio": float(pooled_ratio),
        "pooled_ratio_threshold": POOLED_RATIO_MIN,
        "pooled_ratio_pass": bool(c1),
        "djoint_R_rel": float(djoint_rel),
        "djoint_R_rel_threshold": JOINT_R_REL_MAX,
        "djoint_R_rel_pass": bool(c2),
        "mos_ratio": float(mos_ratio),
        "mos_ratio_threshold": MOS_RATIO_MIN,
        "mos_ratio_pass": bool(c3),
        "all_pass": bool(pass_all),
        "verdict": "PASS_AMENDED" if pass_all else "FAIL_AMENDED",
    }


def percent_delta(new: float, ref: float | None) -> float | None:
    if ref is None or ref == 0:
        return None
    return float(100.0 * (new / ref - 1.0))


def parse_pred_range(s: str | None) -> list[float] | None:
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise ValueError(f"expected 'lo,hi', got {s!r}")
    return [float(parts[0]), float(parts[1])]


def calibration(pred: list[float] | None, actual: float | None) -> str:
    if pred is None or actual is None:
        return "N/A (no prediction)"
    lo, hi = pred
    if lo <= actual <= hi:
        return f"HIT (actual {actual:+.2f}% in [{lo:+.2f}, {hi:+.2f}])"
    return f"MISS (actual {actual:+.2f}% outside [{lo:+.2f}, {hi:+.2f}])"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iter-dir", required=True, type=Path)
    ap.add_argument("--l1-glob", default=DEFAULT_L1_GLOB,
                    help="Glob for L1 auto+cross reference posteriors")
    ap.add_argument("--cnn-glob", default=None,
                    help="Override CNN posterior glob (default: "
                         "<iter-dir>/posteriors/*_s4?.npy)")
    ap.add_argument("--ref-iter", default=None,
                    help="Label for the reference iter (e.g. iter-20) — used "
                         "only in printout / metadata")
    ap.add_argument("--ref-mos", type=float, default=None,
                    help="MoS FoM3 of the reference iter")
    ap.add_argument("--ref-pooled", type=float, default=None,
                    help="Pooled FoM3 of the reference iter")
    ap.add_argument("--ref-joint-R", type=float, default=None,
                    help="joint_R of the reference iter")
    ap.add_argument("--pred-mos-pct", default=None,
                    help="Predicted MoS delta range as 'lo,hi' percent")
    ap.add_argument("--pred-pooled-pct", default=None,
                    help="Predicted pooled delta range as 'lo,hi' percent")
    ap.add_argument("--guard-floor", type=float, default=GUARD_FLOOR,
                    help=f"Per-seed-min Guard floor (default {GUARD_FLOOR})")
    ap.add_argument("--out", default=None, type=Path,
                    help="Output JSON path (default: <iter-dir>/landing.json)")
    args = ap.parse_args()

    iter_dir = args.iter_dir.resolve()
    if not iter_dir.exists():
        print(f"ERROR: iter dir not found: {iter_dir}", file=sys.stderr)
        return 1

    cnn_glob = args.cnn_glob or str(iter_dir / "posteriors" / "*_s4?.npy")
    cnn = analyze_posteriors(cnn_glob)
    if cnn is None:
        print(f"ERROR: no CNN posteriors at {cnn_glob}", file=sys.stderr)
        return 1
    l1 = analyze_posteriors(args.l1_glob)
    if l1 is None:
        print(f"ERROR: no L1 posteriors at {args.l1_glob}", file=sys.stderr)
        return 1

    pred_mos = parse_pred_range(args.pred_mos_pct)
    pred_pooled = parse_pred_range(args.pred_pooled_pct)

    actual_mos_delta = percent_delta(cnn["mos_fom3"], args.ref_mos)
    actual_pooled_delta = percent_delta(cnn["pooled_fom3"], args.ref_pooled)

    cmc = amended_cross_method_check(cnn, l1)

    guard_passed = cnn["per_seed_min"] >= args.guard_floor

    health = compressor_health(iter_dir)

    result = {
        "iter_dir": str(iter_dir),
        "iter_name": iter_dir.name,
        "cnn": cnn,
        "l1_ref": {
            "n_seeds": l1["n_seeds"],
            "mos_fom3": l1["mos_fom3"],
            "pooled_fom3": l1["pooled_fom3"],
            "joint_R": l1["joint_R"],
            "per_param_R": l1["per_param_R"],
        },
        "guard": {
            "floor": args.guard_floor,
            "per_seed_min": cnn["per_seed_min"],
            "passed": bool(guard_passed),
        },
        "ref_iter": args.ref_iter,
        "ref_mos": args.ref_mos,
        "ref_pooled": args.ref_pooled,
        "ref_joint_R": args.ref_joint_R,
        "predicted_delta_mos_pct": pred_mos,
        "actual_delta_mos_pct": actual_mos_delta,
        "calibration_mos": calibration(pred_mos, actual_mos_delta),
        "predicted_delta_pooled_pct": pred_pooled,
        "actual_delta_pooled_pct": actual_pooled_delta,
        "calibration_pooled": calibration(pred_pooled, actual_pooled_delta),
        "cross_method_check_amended": cmc,
        "compressor_health": health,
    }

    out = args.out or (iter_dir / "landing.json")
    out.write_text(json.dumps(result, indent=2))

    # ------- one-screen printout -------
    print("=" * 72)
    print(f"Landing analysis: {iter_dir.name}")
    print("=" * 72)
    print(f"  n_seeds         = {cnn['n_seeds']}  (seeds: {','.join(cnn['seeds'])})")
    print(f"  MoS FoM3        = {cnn['mos_fom3']:.1f}")
    print(f"  per_seed_min    = {cnn['per_seed_min']:.1f}  "
          f"(Guard floor {args.guard_floor:.0f} -> "
          f"{'PASS' if guard_passed else 'FAIL'})")
    print(f"  per_seed_std    = {cnn['per_seed_std']:.1f}  "
          f"(CoV {cnn['per_seed_cov_pct']:.1f}%)")
    print(f"  pooled FoM3     = {cnn['pooled_fom3']:.1f}")
    print(f"  pooled/MoS      = {cnn['pooled_over_mos']:.4f}")
    print(f"  joint_R         = {cnn['joint_R']:.4f}")
    print(f"  per_param_R     = "
          f"{ {k: round(v, 4) for k, v in cnn['per_param_R'].items()} }")
    print()
    print(f"  L1 ref (auto+cross, {l1['n_seeds']} seeds):")
    print(f"    MoS={l1['mos_fom3']:.0f}  pooled={l1['pooled_fom3']:.0f}  "
          f"joint_R={l1['joint_R']:.3f}")
    print()
    if args.ref_iter:
        print(f"  vs {args.ref_iter}:")
        if actual_mos_delta is not None:
            print(f"    MoS    {args.ref_mos:.0f} -> {cnn['mos_fom3']:.0f}  "
                  f"({actual_mos_delta:+.2f}%)  [{result['calibration_mos']}]")
        if actual_pooled_delta is not None:
            print(f"    pooled {args.ref_pooled:.0f} -> {cnn['pooled_fom3']:.0f}  "
                  f"({actual_pooled_delta:+.2f}%)  [{result['calibration_pooled']}]")
        if args.ref_joint_R is not None:
            print(f"    joint_R {args.ref_joint_R:.4f} -> {cnn['joint_R']:.4f}")
        print()
    print("  Amended cross-method check (3-component):")
    print(f"    [{'PASS' if cmc['pooled_ratio_pass'] else 'FAIL'}] "
          f"pooled_ratio = {cmc['pooled_ratio']:.4f} "
          f"(>= {cmc['pooled_ratio_threshold']})")
    print(f"    [{'PASS' if cmc['djoint_R_rel_pass'] else 'FAIL'}] "
          f"|djoint_R|/joint_R_L1 = {cmc['djoint_R_rel']:.4f} "
          f"(<= {cmc['djoint_R_rel_threshold']})")
    print(f"    [{'PASS' if cmc['mos_ratio_pass'] else 'FAIL'}] "
          f"mos_ratio = {cmc['mos_ratio']:.4f} "
          f"(>= {cmc['mos_ratio_threshold']})")
    print(f"  OVERALL CROSS-METHOD VERDICT: {cmc['verdict']}")
    print()
    print(f"  Compressor health: {health['verdict']}")
    if health.get("last_step") is not None:
        print(f"    progress      = {health['last_step']}/{health['total_steps']}")
        if health.get("best_val_step") is not None:
            print(f"    best_val      = step {health['best_val_step']} "
                  f"({health.get('best_val_position_pct')}% of training, "
                  f"test loss {health.get('best_val_test_loss'):.4f})")
            print(f"    final         = test loss "
                  f"{health.get('final_test_loss'):.4f}  "
                  f"(argmin-to-final gap "
                  f"{health.get('argmin_to_final_gap_nats'):.3f} nats)")
    for note in health.get("notes", []):
        print(f"    note: {note}")
    print()
    print(f"  Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
