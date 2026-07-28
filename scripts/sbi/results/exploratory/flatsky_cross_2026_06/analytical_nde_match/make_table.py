#!/usr/bin/env python3
"""Table-1 error bars in the format NOTE_FOM_ERROR_BARS.md specifies.

Spec (paper repo, NOTE_FOM_ERROR_BARS.md, 2026-07-22):

  §5.3  "Quote per row: mean +/- std over the 3 seeds ... or the min-max band."
  §5.4  "Quote the PRE-ENSEMBLE per-seed band for ALL rows (including the
         ensemble-estimated joint l1) ... Do not mix 'spread of singles' with
         'shift under ensembling' -- the latter is the bias term."
  §5.5  "Run the §4 block bootstrap once per arm (seed 41) and report the median
         SE in the caption as the subdominant term."   (percentile 68% interval)
  §1    the ensemble is the QUOTED ESTIMATOR only where the single failed
         calibration (the BNT rows, and joint l1); the single->ensemble shift IS
         the bias term, reported separately, never as an error bar.

So each row emits three distinct things and never conflates them:
  central   -- the quoted estimator (single or ensemble, per the row's spec)
  +/-       -- mean +/- std over the 3 PRE-ENSEMBLE compressor singles  [the error bar]
  bias      -- (ensemble - single)/single, when an ensemble exists       [not an error bar]
  median SE -- block bootstrap 68% interval at seed 41                   [caption term]

NOTE: the leave-one-out ensemble jackknife computed by final_bars.py is NOT part
of this spec. It is retained there as a diagnostic only and must not be quoted.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from final_bars import load, median_term, population_stats


def row_stats(cfg, nboot):
    singles = {k: load(v) for k, v in sorted(cfg.get("singles", {}).items())}
    singles = {k: v for k, v in singles.items() if v is not None}
    med = {k: float(np.median(v["fom3"][np.isfinite(v["fom3"])])) for k, v in singles.items()}
    out = {"published": cfg.get("published"), "quoted": cfg.get("quoted", "single"),
           "per_seed": med}
    if len(med) >= 2:
        v = np.array(list(med.values()), float)
        out.update(mean=float(v.mean()), std=float(v.std(ddof=1)),
                   lo=float(v.min()), hi=float(v.max()),
                   pct=float(100 * v.std(ddof=1) / v.mean()))
    ens = load(cfg["ensemble"]) if cfg.get("ensemble") else None
    if ens is not None:
        e = float(np.median(ens["fom3"][np.isfinite(ens["fom3"])]))
        out["ensemble"] = e
        s41 = med.get("s41")
        if s41:
            out["bias_pct"] = float(100 * (e - s41) / s41)   # single -> ensemble shift
    out["central"] = out.get("ensemble") if out["quoted"] == "ensemble" else med.get("s41")
    # median term: §5.5 says seed 41, once per arm
    ref = singles.get("s41") or (ens if ens is not None else next(iter(singles.values()), None))
    if ref is not None:
        mt = median_term(ref["fom3"], ref.get("patch"), nboot=nboot)
        out["median_se"] = mt["se"]
        out["median_ci68_half"] = mt.get("ci68_half_width")
        out["population"] = population_stats(ref["fom3"], ref.get("patch"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="JSON: {row: {singles:{...}, ensemble, published, quoted}}")
    ap.add_argument("--nboot", type=int, default=10000)
    ap.add_argument("--out", default="TABLE1_ERRORBARS.json")
    ap.add_argument("--md", default=None, help="also write a markdown table here")
    a = ap.parse_args()

    spec = json.load(open(a.rows))
    res = {}
    for name, cfg in spec.items():
        res[name] = row_stats(cfg, a.nboot)
        r = res[name]
        print(f"\n=== {name} ===")
        print("  per-seed singles: " + ", ".join(f"{k}={v:.1f}" for k, v in r["per_seed"].items()))
        if "std" in r:
            print(f"  ERROR BAR   : {r['mean']:.0f} +/- {r['std']:.0f} ({r['pct']:.1f}%)   "
                  f"band {r['lo']:.0f}-{r['hi']:.0f}   [{len(r['per_seed'])} compressor seeds, pre-ensemble]")
        if "ensemble" in r:
            print(f"  ensemble    : {r['ensemble']:.1f}"
                  + (f"   BIAS (single->ens) = {r['bias_pct']:+.1f}%  [NOT an error bar]"
                     if "bias_pct" in r else ""))
        print(f"  quoted      : {r['quoted']}  ->  central = "
              f"{r['central']:.1f}" if r.get("central") else "  quoted: n/a")
        if r.get("published"):
            print(f"  published   : {r['published']}")
        if "median_se" in r:
            p = r["population"]
            print(f"  median term : +/- {r['median_se']:.2f} "
                  f"(68% half-width {r['median_ci68_half']:.2f})   "
                  f"rho={p.get('rho', float('nan')):.3f} CV={p['CV_pop']:.3f}")
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")

    if a.md:
        L = ["# Table 1 — FoM3 error bars (retrained on Jean-Zay, 2026-07-28)", "",
             "## Headline",  "",
             "| Row | published | **retrained** (quoted) | Δ | **±** (n seeds) | n | band |",
             "|---|--:|--:|--:|--:|--:|---|"]
        for n, r in res.items():
            pm = f"±{r['std']:.0f} ({r['pct']:.1f}%)" if "std" in r else "—"
            bd = f"{r['lo']:.0f}–{r['hi']:.0f}" if "lo" in r else "—"
            ce = f"{r['central']:.1f}" if r.get("central") else "—"
            pub = r.get("published")
            dl = (f"{100*(r['central']-pub)/pub:+.1f}%"
                  if (pub and r.get("central")) else "—")
            L.append(f"| {n} | {pub or '—'} | **{ce}** ({r['quoted']}) | {dl} | **{pm}** | {len(r['per_seed'])} | {bd} |")

        L += ["", "## Full detail", "",
              "| Row | published | retrained quoted | retrained single s41 | retrained ensemble | "
              "singles mean | per-seed values | ± (std) | bias single→ens | median SE (68%) | ρ | CV |",
              "|---|--:|--:|--:|--:|--:|---|--:|--:|--:|--:|--:|"]
        def f1(x):
            return "—" if x is None else f"{x:.1f}"

        for n, r in res.items():
            ps = r["per_seed"]
            pop = r.get("population", {})
            cells = [
                n,
                str(r.get("published", "—")),
                f"{f1(r.get('central'))} ({r['quoted']})",
                f1(ps.get("s41")),
                f1(r.get("ensemble")),
                f1(r.get("mean")),
                " / ".join(f1(ps[k]) for k in sorted(ps) if k in ps) or "—",
                f"±{r['std']:.0f} ({r['pct']:.1f}%)" if "std" in r else "—",
                f"{r['bias_pct']:+.1f}%" if "bias_pct" in r else "—",
                f"±{r['median_se']:.2f}" if "median_se" in r else "—",
                f"{pop['rho']:.3f}" if "rho" in pop else "—",
                f"{pop['CV_pop']:.3f}" if "CV_pop" in pop else "—",
            ]
            L.append("| " + " | ".join(cells) + " |")

        L += ["", "**± = spread over independently trained compressors (pre-ensemble singles)**, "
                  "per `NOTE_FOM_ERROR_BARS.md` §5.3–5.4 — quoted for ALL rows, including the",
              "ensemble-estimated ones, so it is comparable across rows as training stochasticity.",
              "",
              "`retrained quoted` follows each row's published estimator convention: the SINGLE (seed 41) "
              "for ℓ1 auto/+product no-BNT and both CNN rows, the 3-compressor ENSEMBLE for the BNT ℓ1",
              "rows and both joint ℓ1 rows (the ensemble is the quoted estimator only where the single "
              "failed the calibration battery — `RESULT_NOBNT_ENSEMBLE_ROBUSTNESS.md`).",
              "",
              "The **single→ensemble shift is the BIAS term**, reported separately and never summed with "
              "the ± (§1, §5.4). Median SE = block bootstrap over the 180 patches keeping all 50 noise",
              "reps, 10⁴ replicates, 68% percentile interval, at seed 41 (§4, §5.5). ρ = intra-patch "
              "correlation by one-way ANOVA; measured ≈0 everywhere, which is why the median term is",
              "at the bottom of the note's predicted 0.1–3% range."]
        open(a.md, "w").write("\n".join(L) + "\n")
        print(f"wrote {a.md}")


if __name__ == "__main__":
    main()
