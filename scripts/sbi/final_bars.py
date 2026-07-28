#!/usr/bin/env python3
"""FoM3 error bars: seed term, ensemble (jackknife) term, median (block bootstrap) term.

The original final_bars.py survived the Titan failure as a 0-byte file, so the ensemble
term is written here from the NOTE_FOM_ERROR_BARS spec recorded in HANDOFF section 6.

Three terms, never quadrature-summed (a range and an SE are not the same kind of object):

  SEED term (dominant, single-estimator rows)
      3 compressor seeds -> per-seed median FoM3 -> HALF-RANGE. Reported as a range,
      not a Gaussian sigma: n=3 does not support one.

  ENSEMBLE term (ensemble rows)
      Leave-one-out jackknife over the K=3 compressor members. Leaving member l out
      leaves a PAIR, whose pooled posterior is a 1/2-1/2 mixture. For a mixture of two
      distributions the moments are exact:
          m_mix = 0.5 * (m_i + m_j)
          C_mix = 0.5 * (C_i + C_j) + 0.25 * (m_i - m_j)(m_i - m_j)^T
      so pooled FoM3 per observation = 1/sqrt(det C_mix), then the median over the 9000
      mocks gives theta_(l). With K=3:
          var_jack = ((K-1)/K) * sum_l (theta_(l) - theta_bar)^2
      The MEANS are essential -- the (m_i - m_j) term is why storing only per-member
      covariances (or only the pooled ensemble) leaves this uncomputable. That is the
      omission this whole recovery exists to fix.
      NOT the raw single-member spread: the singles are a miscalibrated population.

  MEDIAN term (subdominant, ~0.2%)
      Block bootstrap: resample the 180 patches WITH replacement, keeping each patch's
      50 noise reps intact (they share cosmic structure and are not independent),
      1e4 replicates -> SE of the median.

Usage
  # reproduce the one surviving measured bar, as a check on this code
  final_bars.py --validate
  # then, on retrained arms
  final_bars.py --row-json rows.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

NBOOT = 10_000
SEED = 0

SURVIVING = ("/lustre/fsn1/projects/rech/prk/ulx34io/cnn_sbi/scripts/sbi/results/"
             "exploratory/flatsky_cross_2026_06")
# Validation against surviving data. NOTE: the FOM_ERROR_BARS_STATUS headline bar
# "3045 +/- 177 (5.8%)" is NOT internally consistent -- it takes s41 from the N=9000
# file (perm 0..49) and s42/s43 from N=1000 files (perm 0..5). The obs set alone shifts
# s41 by 3.3% (3044.9 -> 3146.1), so a large part of that "seed" range is obs-set
# mismatch. Two separate, self-consistent checks instead:
#   (a) seed band with all three seeds on the IDENTICAL 1000-obs selection
#   (b) central value + median term on the full 9000-obs population (s41 only)
_AN = f"{SURVIVING}/analytical_nde_match"
VALIDATE_ROWS = {
    "L1 +product no-BNT : seed band (consistent, N=1000, perm 0-5)": {
        "seeds": {"s41": f"{_AN}/l1product_rnvp_s41/per_patch_metrics.npz",
                  "s42": f"{_AN}/l1product_rnvp_s42/per_patch_metrics.npz",
                  "s43": f"{_AN}/l1product_rnvp_s43/per_patch_metrics.npz"},
        "expect": {"fom": 3264.6, "seed_halfrange": 126.5, "seed_pct": 3.87,
                   "median_se": 17.66, "median_pct": 0.54},
    },
    "L1 +product no-BNT : central + median term (N=9000, s41)": {
        "seeds": {"s41": f"{_AN}/l1product_rnvp_s41_n9000/per_patch_metrics.npz"},
        "expect": {"fom": 3044.9, "median_se": 6.0, "median_pct": 0.20},
    },
}


def load(path):
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    out = {"fom3": np.asarray(d["fom3"], float)}
    for k in ("patch", "perm", "mean", "cov", "arm_mean", "arm_cov"):
        if k in d.files:
            out[k] = np.asarray(d[k])
    return out


# --------------------------------------------------------------------- seed term
def seed_term(per_seed_medians: dict):
    """Spread of the per-compressor-seed medians.

    Two presentations, because the two specs disagree and the choice is the user's:
      * half_range / central(median)  -- the older HANDOFF section 6 reading, where the
        quoted row is the ENSEMBLE and this band is secondary.
      * mean +/- std and the min-max band -- PLAN_FOM_ERRORBARS_SWEEP.md (2026-07-22),
        which says the +/- IS this spread over PRE-ensemble singles, and that the
        single->ensemble de-inflation is a separate BIAS term, not an error bar.
    Both are computed from the same three numbers; nothing is re-fit.
    """
    v = np.array([m for m in per_seed_medians.values() if np.isfinite(m)], float)
    if v.size < 2:
        return None
    return {"n_seeds": int(v.size),
            "central": float(np.median(v)),
            "half_range": float((v.max() - v.min()) / 2.0),
            # PLAN_FOM_ERRORBARS_SWEEP.md presentation
            "mean": float(v.mean()),
            "std": float(v.std(ddof=1)),
            "min": float(v.min()),
            "max": float(v.max()),
            "per_seed": {k: float(m) for k, m in per_seed_medians.items()}}


def population_stats(fom3, patch):
    """CV_pop and the intra-patch correlation rho, both requested by the PLAN doc.

    rho via one-way ANOVA over the 180 patch blocks: rho = tau^2 / (tau^2 + sigma_w^2),
    with tau^2 the between-patch variance component and sigma_w^2 the within-patch
    (noise-rep) variance. rho is what sets how much the block bootstrap inflates the
    median SE relative to a naive iid bootstrap, so it explains the median term rather
    than just asserting it.
    """
    g = np.isfinite(fom3)
    f = np.asarray(fom3, float)[g]
    out = {"n": int(f.size), "CV_pop": float(f.std(ddof=1) / f.mean())}
    if patch is None:
        return out
    p = np.asarray(patch).ravel()[g]
    groups = [f[p == u] for u in np.unique(p)]
    groups = [x for x in groups if x.size >= 2]
    k = len(groups)
    if k < 2:
        return out
    n_i = np.array([x.size for x in groups], float)
    means = np.array([x.mean() for x in groups])
    grand = f.mean()
    ss_b = float(np.sum(n_i * (means - grand) ** 2))
    ss_w = float(np.sum([((x - x.mean()) ** 2).sum() for x in groups]))
    N = float(n_i.sum())
    ms_b, ms_w = ss_b / (k - 1), ss_w / (N - k)
    # balanced-ish design: n0 is the effective reps per block
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    tau2 = max((ms_b - ms_w) / n0, 0.0)
    out.update({"rho": float(tau2 / (tau2 + ms_w)) if (tau2 + ms_w) > 0 else float("nan"),
                "n_blocks": int(k), "reps_per_block": float(n0)})
    return out


# ---------------------------------------------------------------- ensemble term
def mixture_fom3(m_i, C_i, m_j, C_j):
    """Per-observation FoM3 of the 1/2-1/2 mixture of two members. Vectorised over obs."""
    dm = m_i - m_j                                        # (N,3)
    C = 0.5 * (C_i + C_j) + 0.25 * dm[:, :, None] * dm[:, None, :]
    det = np.linalg.det(C)
    out = np.full(det.shape, np.nan)
    good = np.isfinite(det) & (det > 0)
    out[good] = 1.0 / np.sqrt(det[good])
    return out


def ensemble_term(arm_mean, arm_cov):
    """Leave-one-out jackknife over K members via exact mixture moments."""
    K = arm_mean.shape[0]
    if K < 3:
        return None
    thetas, left_out = [], []
    for l in range(K):
        keep = [x for x in range(K) if x != l]
        i, j = keep[0], keep[1]
        f = mixture_fom3(arm_mean[i], arm_cov[i], arm_mean[j], arm_cov[j])
        g = np.isfinite(f)
        if g.sum() == 0:
            return None
        thetas.append(float(np.median(f[g])))
        left_out.append(l)
    t = np.array(thetas)
    tbar = t.mean()
    var = ((K - 1) / K) * np.sum((t - tbar) ** 2)
    return {"K": int(K), "theta_leave_one_out": [float(x) for x in t],
            "theta_bar": float(tbar), "se": float(np.sqrt(var))}


def ensemble_term_from_samples(samples):
    """Cross-check: same jackknife by literally pooling members' samples.

    `samples` is (K, N, M, 3). Used only to confirm the mixture-moment algebra above.
    """
    K, N = samples.shape[0], samples.shape[1]
    thetas = []
    for l in range(K):
        keep = [x for x in range(K) if x != l]
        f = np.full(N, np.nan)
        for n in range(N):
            ps = np.concatenate([samples[k, n] for k in keep], 0)
            ps = ps[np.all(np.isfinite(ps), 1)]
            if ps.shape[0] < 100:
                continue
            det = np.linalg.det(np.cov(ps[:, :3].T))
            if np.isfinite(det) and det > 0:
                f[n] = 1.0 / np.sqrt(det)
        g = np.isfinite(f)
        thetas.append(float(np.median(f[g])))
    t = np.array(thetas)
    return {"theta_leave_one_out": [float(x) for x in t],
            "se": float(np.sqrt(((K - 1) / K) * np.sum((t - t.mean()) ** 2)))}


# ------------------------------------------------------------------ median term
def median_term(fom3, patch, nboot=NBOOT, seed=SEED):
    """Block bootstrap over patches; each patch's noise reps move together."""
    g = np.isfinite(fom3)
    f = fom3[g]
    if patch is None:
        rng = np.random.default_rng(seed)
        meds = [np.median(f[rng.integers(0, f.size, f.size)]) for _ in range(nboot)]
        return {"se": float(np.std(meds)), "blocks": "plain (no patch index)"}
    p = np.asarray(patch).ravel()[g]
    uniq = np.unique(p)
    idx_by_patch = [np.where(p == u)[0] for u in uniq]
    rng = np.random.default_rng(seed)
    npatch = uniq.size
    meds = np.empty(nboot)
    for b in range(nboot):
        pick = rng.integers(0, npatch, npatch)
        meds[b] = np.median(f[np.concatenate([idx_by_patch[k] for k in pick])])
    lo, hi = np.percentile(meds, [16.0, 84.0])          # PLAN doc asks for the 68% interval
    return {"se": float(np.std(meds)), "blocks": f"{npatch} patches",
            "p16": float(lo), "p84": float(hi),
            "ci68_half_width": float((hi - lo) / 2.0),
            "reps_per_patch": int(np.median([len(x) for x in idx_by_patch]))}


# ------------------------------------------------------------------------- main
def do_row(name, seed_paths, expect=None, nboot=NBOOT):
    print(f"\n=== {name} ===")
    loaded, medians = {}, {}
    for tag, path in sorted(seed_paths.items()):
        d = load(path)
        if d is None:
            print(f"  {tag}: MISSING {path}")
            medians[tag] = np.nan
            continue
        loaded[tag] = d
        g = np.isfinite(d["fom3"])
        medians[tag] = float(np.median(d["fom3"][g]))
        print(f"  {tag}: n={int(g.sum())} median FoM3 = {medians[tag]:.1f}")

    res = {"row": name, "per_seed_median": medians}
    st = seed_term(medians)
    if st:
        res["seed_term"] = st
        pct = 100 * st["half_range"] / st["central"]
        print(f"  SEED term   : {st['central']:.0f} +/- {st['half_range']:.0f} "
              f"({pct:.2f}%)  [half-range, n={st['n_seeds']}]")
        print(f"    PLAN form : {st['mean']:.0f} +/- {st['std']:.0f} "
              f"({100*st['std']/st['mean']:.2f}%)  [mean +/- std, n={st['n_seeds']}]"
              f"   band {st['min']:.0f}-{st['max']:.0f}")

    ref = next(iter(loaded.values()), None)
    if ref is not None:
        mt = median_term(ref["fom3"], ref.get("patch"), nboot=nboot)
        res["median_term"] = mt
        c = st["central"] if st else float(np.median(ref["fom3"][np.isfinite(ref["fom3"])]))
        print(f"  MEDIAN term : +/- {mt['se']:.2f} ({100*mt['se']/c:.2f}%)  [{mt['blocks']}]")
        if "ci68_half_width" in mt:
            print(f"    68% CI    : [{mt['p16']:.1f}, {mt['p84']:.1f}] "
                  f"(half-width {mt['ci68_half_width']:.2f}, {100*mt['ci68_half_width']/c:.2f}%)")
        ps = population_stats(ref["fom3"], ref.get("patch"))
        res["population_stats"] = ps
        if "rho" in ps:
            print(f"  POPULATION  : CV_pop = {ps['CV_pop']:.3f}   rho(intra-patch) = "
                  f"{ps['rho']:.3f}   [{ps['n_blocks']} blocks x {ps['reps_per_block']:.0f} reps]")

        if "arm_mean" in ref and "arm_cov" in ref:
            et = ensemble_term(ref["arm_mean"], ref["arm_cov"])
            if et:
                res["ensemble_term"] = et
                print(f"  ENSEMBLE term: +/- {et['se']:.2f} "
                      f"({100*et['se']/et['theta_bar']:.2f}%)  [jackknife K={et['K']}]")
                print(f"                 leave-one-out medians: "
                      f"{[round(x,1) for x in et['theta_leave_one_out']]}")
        else:
            print("  ENSEMBLE term: n/a (no per-member arm_mean/arm_cov in this file --")
            print("                 pre-recovery outputs lack them; retrained arms will have them)")

    if expect:
        print("  --- vs known values ---")
        central_obs = (st["central"] if st else
                       float(np.median(ref["fom3"][np.isfinite(ref["fom3"])])))
        ok_c = abs(central_obs - expect["fom"]) / expect["fom"] < 0.02
        print(f"      FoM        {central_obs:8.1f} vs {expect['fom']:8.1f}  "
              f"{'OK' if ok_c else 'MISMATCH'}")
        res.setdefault("validation", {})["fom_ok"] = bool(ok_c)
        if st and "seed_halfrange" in expect:
            ok_f = True
            ok_s = abs(st["half_range"] - expect["seed_halfrange"]) / expect["seed_halfrange"] < 0.05
            print(f"      seed +/-   {st['half_range']:8.1f} vs {expect['seed_halfrange']:8.1f}  "
                  f"{'OK' if ok_s else 'MISMATCH'}")
            res.setdefault("validation", {})["seed_ok"] = bool(ok_s)
        if "median_term" in res and "median_se" in expect:
            ok_m = abs(res["median_term"]["se"] - expect["median_se"]) / expect["median_se"] < 0.25
            print(f"      median +/- {res['median_term']['se']:8.2f} vs "
                  f"{expect['median_se']:8.2f}  {'OK' if ok_m else 'MISMATCH'}")
            res.setdefault("validation", {})["median_ok"] = bool(ok_m)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="reproduce the surviving L1+product bar (3045 +/- 177, 5.8%%)")
    ap.add_argument("--row-json", help="{name: {seeds: {tag: path}}} for retrained rows")
    ap.add_argument("--nboot", type=int, default=NBOOT)
    ap.add_argument("--out", default="final_bars_results.json")
    a = ap.parse_args()

    rows = []
    if a.validate:
        for nm, cfg in VALIDATE_ROWS.items():
            rows.append(do_row(nm, cfg["seeds"], cfg["expect"], nboot=a.nboot))
    if a.row_json:
        spec = json.load(open(a.row_json))
        for name, cfg in spec.items():
            rows.append(do_row(name, cfg["seeds"], cfg.get("expect"), nboot=a.nboot))
    if not rows:
        ap.error("nothing to do: pass --validate and/or --row-json")

    json.dump(rows, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    print("\nReport per row as:  FoM = X  (seed +/- A; median +/- B).")
    print("Do NOT quadrature-sum a range with an SE.")


if __name__ == "__main__":
    main()
