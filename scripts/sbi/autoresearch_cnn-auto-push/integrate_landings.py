#!/usr/bin/env python3
"""Integrate post-landing verdicts from iter-22 (Q9c) and iter-23 (Q4).

After ``landing_analysis.py`` has been run on each iter (producing
``<iter-dir>/landing.json``), this script reads both landings, classifies
the combined 4-branch decision matrix called for by the
``cnn-auto-push-18-20-2026`` constitution, and emits:

1. A markdown "Integration headline" block — ready to paste into the
   "Headline (TO FILL IN)" section of ``CEILING_EVIDENCE.md`` and into
   the body of ``[[cnn-auto-ceiling-evidence]]`` when it closes.
2. A disposition recommendation for the open ceiling-evidence sub-fiber
   (CLOSE / DEFER) and the rationale.
3. Which ``REPLICATION_LAUNCH.md`` section to point at (A / B / C / A_alt).
4. A felt-history-append-able summary paragraph.

Falsifier ranges (derived from constitution + CEILING_EVIDENCE.md):

* **iter-22 (Q9c, --pred-pooled-pct=3,8 vs iter-20 pooled=13944):**
    - NULL  (ceiling confirmed):   pooled <= 14_220   (within +2% of iter-20)
    - INTERMEDIATE (defer):        14_220 < pooled <= 15_000
    - POSITIVE (Q2 lever compounds): pooled > 15_000

* **iter-23 (Q4, --pred-pooled-pct=0,5 vs iter-20 pooled=13944):**
    - NULL  (aux-NF not the binding constraint): pooled <= 14_720 (+5%)
    - POSITIVE (Q4 real at cdim=16):              pooled > 14_720

These thresholds are CLI-configurable so the script can be re-run if the
constitution amends them.

Usage::

    python integrate_landings.py \
        --iter22-dir <run-dir>/iter-22 \
        --iter23-dir <run-dir>/iter-23 \
        [--out <run-dir>/integration_summary.md]

The script is read-only against the rest of the repository — it does
NOT touch CEILING_EVIDENCE.md or felt fibers itself. The next Ralph iter
copies the emitted markdown into the right places.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

# Default thresholds (constitution iter-14 entry; can be amended via CLI).
ITER22_NULL_POOLED_MAX = 14_220       # ceiling confirmed if <=
ITER22_INTERMED_POOLED_MAX = 15_000   # defer if in (NULL_MAX, INTERMED_MAX]
# else POSITIVE
ITER23_NULL_POOLED_MAX = 14_720       # +5% of iter-20 pooled 13944

REF_ITER = "iter-20"
REF_POOLED = 13_944.0
REF_MOS = 18_673.0
REF_JOINT_R = 0.220


def _load_landing(p: Path) -> dict:
    if not p.exists():
        raise FileNotFoundError(f"landing.json not found at {p}")
    return json.loads(p.read_text())


def classify_iter22(pooled: float, null_max: float, intermed_max: float) -> str:
    if pooled <= null_max:
        return "NULL"
    if pooled <= intermed_max:
        return "INTERMEDIATE"
    return "POSITIVE"


def classify_iter23(pooled: float, null_max: float) -> str:
    return "NULL" if pooled <= null_max else "POSITIVE"


def combined_branch(c22: str, c23: str) -> dict:
    """Map (iter-22 class, iter-23 class) -> branch label + actions."""
    # Treat INTERMEDIATE for iter-22 as 'defer'; downstream logic groups
    # it with POSITIVE (any non-NULL ⇒ defer).
    iter22_null = c22 == "NULL"
    iter23_null = c23 == "NULL"

    if iter22_null and iter23_null:
        return {
            "branch": "BOTH_NULL",
            "disposition": "CLOSE_CEILING_EVIDENCE",
            "replication_section": "A_PRIMARY (iter-16)",
            "summary": (
                "Both ceiling falsifiers null. Variance/drift family + Q2 "
                "information lever + VMIM-aux width are all exhausted at "
                "the plain-CNN architecture. Ceiling is at pooled ≈ 14 k "
                "with 3-seed MoS ≈ 18-19 k. Close [[cnn-auto-ceiling-evidence]] "
                "with outcome reflecting the final number; replication target "
                "= iter-16 PRIMARY (highest MoS, passes amended check)."
            ),
        }
    if (not iter22_null) and iter23_null:
        return {
            "branch": "Q9C_POSITIVE_Q4_NULL",
            "disposition": "DEFER_CEILING_EVIDENCE",
            "replication_section": "B (iter-22 5-seed)",
            "summary": (
                "Q2 information lever compounds with variance/drift family "
                "at the 120k compressor step count: iter-22 pooled exceeds "
                "the +2% null band. Ceiling thinking was premature; plan Q9d "
                "(4-lever stack: cbs + pool + F1 + 120k compressor) as the "
                "next campaign step. Replicate iter-22 at 5 seeds first."
            ),
        }
    if iter22_null and (not iter23_null):
        return {
            "branch": "Q4_POSITIVE_Q9C_NULL",
            "disposition": "DEFER_CEILING_EVIDENCE",
            "replication_section": "C (iter-23 5-seed)",
            "summary": (
                "Q4 (VMIM aux NF width) is a real lever at cdim=16 even "
                "though it was null at cdim=10 [[cnn_vmim_target_stability]]. "
                "The aux-NF-bound-limit hypothesis "
                "[[cnn-auto-bug-vmim-aux-may-bias-compressor]] is supported. "
                "Plan a Q4-deep-sweep (aux widths 256 / 384 / 512) at "
                "cdim=16 + iter-21 stack. Replicate iter-23 at 5 seeds first."
            ),
        }
    return {
        "branch": "BOTH_POSITIVE",
        "disposition": "DEFER_CEILING_EVIDENCE",
        "replication_section": "B+C (combined sweep)",
        "summary": (
            "BOTH levers real — surprising and high-EV. Q2 (compressor "
            "steps) AND Q4 (aux-NF width) each independently move the "
            "ceiling beyond the previous estimate. Defer ceiling close; "
            "plan a combined Q4+Q9c iteration (--vmim-nf-hidden 256 + 120k "
            "compressor + cbs/pool/F1 stack) as the immediate next launch."
        ),
    }


def headline_block(
    branch: dict,
    iter22_landing: dict | None,
    iter23_landing: dict | None,
    iter22_class: str | None,
    iter23_class: str | None,
) -> str:
    """Render the markdown block that fills the CEILING_EVIDENCE 'Headline' section."""
    lines: list[str] = []
    lines.append(f"## Headline — integrated post-iter-22 / iter-23\n")

    cnn22 = iter22_landing["cnn"] if iter22_landing else None
    cnn23 = iter23_landing["cnn"] if iter23_landing else None

    if cnn22 is not None:
        cm22 = iter22_landing.get("cross_method_check_amended", {})
        lines.append(
            f"**iter-22 (Q9c, Q9 stack at 120k compressor) → {iter22_class}**:"
            f" pooled FoM3 = {cnn22['pooled_fom3']:.0f}, MoS = "
            f"{cnn22['mos_fom3']:.0f}, joint_R = {cnn22['joint_R']:.3f}; "
            f"amended cross-method check verdict = {cm22.get('verdict','?')}.\n"
        )
    if cnn23 is not None:
        cm23 = iter23_landing.get("cross_method_check_amended", {})
        lines.append(
            f"**iter-23 (Q4, --vmim-nf-hidden 256) → {iter23_class}**:"
            f" pooled FoM3 = {cnn23['pooled_fom3']:.0f}, MoS = "
            f"{cnn23['mos_fom3']:.0f}, joint_R = {cnn23['joint_R']:.3f}; "
            f"amended cross-method check verdict = {cm23.get('verdict','?')}.\n"
        )

    lines.append(f"\n**Branch**: `{branch['branch']}`  ")
    lines.append(f"**Disposition for ceiling-evidence sub-fiber**: "
                 f"{branch['disposition']}  ")
    lines.append(f"**Next-action replication section**: "
                 f"{branch['replication_section']}\n\n")
    lines.append(branch["summary"])
    lines.append("\n")
    return "\n".join(lines)


def felt_history_text(
    branch: dict,
    iter22_landing: dict | None,
    iter23_landing: dict | None,
    iter22_class: str | None,
    iter23_class: str | None,
) -> str:
    """Render a felt-history-append-able paragraph for the parent fiber."""
    parts: list[str] = []
    parts.append(
        "iter-22 (Q9c, Q9 stack at 120k) "
        + (
            f"= {iter22_class}: pooled "
            f"{iter22_landing['cnn']['pooled_fom3']:.0f}, MoS "
            f"{iter22_landing['cnn']['mos_fom3']:.0f}"
            if iter22_landing else "= UNAVAILABLE"
        )
    )
    parts.append(
        "iter-23 (Q4, --vmim-nf-hidden 256) "
        + (
            f"= {iter23_class}: pooled "
            f"{iter23_landing['cnn']['pooled_fom3']:.0f}, MoS "
            f"{iter23_landing['cnn']['mos_fom3']:.0f}"
            if iter23_landing else "= UNAVAILABLE"
        )
    )
    parts.append(f"branch = {branch['branch']}")
    parts.append(f"disposition = {branch['disposition']}")
    parts.append(f"replication section = {branch['replication_section']}")
    parts.append(branch["summary"])
    return ". ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--iter22-dir", type=Path,
                    help="Path to iter-22 dir (containing landing.json)")
    ap.add_argument("--iter23-dir", type=Path,
                    help="Path to iter-23 dir (containing landing.json)")
    ap.add_argument("--iter22-null-max", type=float,
                    default=ITER22_NULL_POOLED_MAX,
                    help="Pooled FoM3 upper bound for iter-22 NULL classification")
    ap.add_argument("--iter22-intermed-max", type=float,
                    default=ITER22_INTERMED_POOLED_MAX,
                    help="Pooled FoM3 upper bound for iter-22 INTERMEDIATE")
    ap.add_argument("--iter23-null-max", type=float,
                    default=ITER23_NULL_POOLED_MAX,
                    help="Pooled FoM3 upper bound for iter-23 NULL classification")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output markdown file (default: print to stdout)")
    args = ap.parse_args()

    if not args.iter22_dir and not args.iter23_dir:
        print("ERROR: at least one of --iter22-dir / --iter23-dir is required",
              file=sys.stderr)
        return 1

    iter22_landing = None
    iter23_landing = None
    iter22_class = None
    iter23_class = None

    if args.iter22_dir is not None:
        lp = args.iter22_dir / "landing.json"
        try:
            iter22_landing = _load_landing(lp)
        except FileNotFoundError:
            print(f"WARNING: iter-22 landing.json not yet at {lp}; "
                  "run landing_analysis.py first.", file=sys.stderr)
        if iter22_landing is not None:
            iter22_class = classify_iter22(
                iter22_landing["cnn"]["pooled_fom3"],
                args.iter22_null_max,
                args.iter22_intermed_max,
            )

    if args.iter23_dir is not None:
        lp = args.iter23_dir / "landing.json"
        try:
            iter23_landing = _load_landing(lp)
        except FileNotFoundError:
            print(f"WARNING: iter-23 landing.json not yet at {lp}; "
                  "run landing_analysis.py first.", file=sys.stderr)
        if iter23_landing is not None:
            iter23_class = classify_iter23(
                iter23_landing["cnn"]["pooled_fom3"],
                args.iter23_null_max,
            )

    if iter22_class is None and iter23_class is None:
        print("ERROR: no landing.json files readable; nothing to integrate.",
              file=sys.stderr)
        return 1

    # If only one landed, we still emit a partial summary — the user runs
    # this once after iter-23 lands (~03:55 UTC) and again after iter-22
    # lands (~04:50 UTC) to update the integration.
    c22 = iter22_class or "PENDING"
    c23 = iter23_class or "PENDING"
    if c22 == "PENDING" or c23 == "PENDING":
        # Provisional branch label — actions deferred until both land.
        branch = {
            "branch": f"PARTIAL ({c22}/{c23})",
            "disposition": "DEFER_UNTIL_BOTH_LAND",
            "replication_section": "n/a (pending second landing)",
            "summary": (
                "One of iter-22 / iter-23 has not yet produced a landing.json. "
                "Re-run integrate_landings.py once both are done."
            ),
        }
    else:
        branch = combined_branch(c22, c23)

    md = []
    md.append(f"# Integration summary — iter-22 / iter-23\n")
    md.append(f"_Rendered {_dt.datetime.now(tz=_dt.timezone.utc).isoformat()}_\n\n")
    md.append(f"Reference iter ({REF_ITER}): pooled = {REF_POOLED:.0f}, "
              f"MoS = {REF_MOS:.0f}, joint_R = {REF_JOINT_R:.3f}\n\n")
    md.append("## Classification thresholds (constitution defaults)\n\n")
    md.append(f"- iter-22 NULL  if pooled ≤ {args.iter22_null_max:.0f}\n")
    md.append(f"- iter-22 INTERMEDIATE if {args.iter22_null_max:.0f} "
              f"< pooled ≤ {args.iter22_intermed_max:.0f}\n")
    md.append(f"- iter-22 POSITIVE if pooled > {args.iter22_intermed_max:.0f}\n")
    md.append(f"- iter-23 NULL if pooled ≤ {args.iter23_null_max:.0f}\n")
    md.append(f"- iter-23 POSITIVE if pooled > {args.iter23_null_max:.0f}\n\n")
    md.append(headline_block(branch, iter22_landing, iter23_landing,
                             iter22_class, iter23_class))
    md.append("\n---\n\n## Felt-history-append text\n\n```\n")
    md.append(felt_history_text(branch, iter22_landing, iter23_landing,
                                iter22_class, iter23_class))
    md.append("\n```\n\n")
    md.append("## Disposition for [[cnn-auto-ceiling-evidence]]\n\n")
    md.append(f"- **Status change**: {branch['disposition']}\n")
    md.append(f"- **Replication target**: {branch['replication_section']}\n\n")
    if branch["disposition"] == "CLOSE_CEILING_EVIDENCE":
        # Render outcome text for the close.
        cnn22 = iter22_landing["cnn"] if iter22_landing else None
        cnn23 = iter23_landing["cnn"] if iter23_landing else None
        pooled_max = max([x["pooled_fom3"] for x in [cnn22, cnn23] if x])
        mos_max = max([x["mos_fom3"] for x in [cnn22, cnn23] if x])
        md.append("**Suggested outcome text** (paste into "
                  "`felt edit cnn-auto-push-18-20-2026/cnn-auto-ceiling-evidence "
                  "-s closed -o ...`):\n\n")
        md.append(f"> FoM3 ceiling on plain-CNN 4-auto-channel architecture: "
                  f"pooled ≈ {pooled_max:.0f}, MoS ≈ {mos_max:.0f}. "
                  f"Closed because Q2 (compressor steps), Q4 (VMIM aux width), "
                  f"and Q9 stack (cbs+pool+F1) levers are all exhausted; "
                  f"further gain requires architecture change (Tier-3 / out-of-fiber).\n\n")

    if iter22_landing is not None:
        md.append("## iter-22 cross-method check\n\n```json\n")
        md.append(json.dumps(
            iter22_landing.get("cross_method_check_amended", {}), indent=2))
        md.append("\n```\n\n")
    if iter23_landing is not None:
        md.append("## iter-23 cross-method check\n\n```json\n")
        md.append(json.dumps(
            iter23_landing.get("cross_method_check_amended", {}), indent=2))
        md.append("\n```\n")

    text = "".join(md)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"Wrote {args.out}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
