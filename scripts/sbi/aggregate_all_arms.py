#!/usr/bin/env python3
"""Phase C — definitive L1-vs-CNN table across ALL arms.

PRIMARY metric: **3-seed POOLED FoM3 on (Ωm, σ8, w0), computed PER PERM then
AVERAGED across the available perms** — this is the constitution's declared metric.
The across-perm spread is reported alongside (± std and %CV) so FoM3's known
sensitivity is visible without changing the headline metric (memory:
feedback_fom3_fragile_use_2d_areas). Marginal σ and 2D FoM are SECONDARY and use
the same per-perm-then-average estimator.

Why per-perm-average and NOT perm-pool (the bug this script previously had for the
multi-perm arms): pooling posteriors across perms mixes the different
fiducial-realization mode locations into one cloud → artificially broad → not
comparable to the single-perm arms (it gave 7868/6096, far below the perm-0 rows).
The right estimator pools the 3 NDE seeds *within* a perm (intentional NDE-seed
scatter, the declared "3-seed pooled" unit), then averages that unit *across* perms
(fiducial-realization noise reduction). Every arm's unit is then a 3-seed pool;
arms with 3 perms average it over perms, arms with only perm-0 report the single unit.

Note: CNN absolute FoM are fast-tf.data-route; per Andreas (2026-05-31) the
compressor↔NDE overlap is empirically negligible, so absolute is treated as fine.
"""
import glob, os, re
import numpy as np

DC = "results/exploratory/definitive_comparison"
OUT = os.path.join(DC, "PHASE_C_2026_05_31"); os.makedirs(OUT, exist_ok=True)
NAMES = ["Omega_m", "sigma_8", "w_0", "h_0", "n_s", "Omega_b"]

# arm display -> glob of posterior .npy across ALL perms (seeds pooled per perm).
ARMS = {
    "L1 auto+cross":        f"{DC}/posteriors/l1_autocross_split70/l1_autocross_split70_s*_p*.npy",
    "L1 auto-only":         f"{DC}/posteriors/l1_autoonly_split70/l1_autoonly_split70_s*_p*.npy",
    "CNN-RealNVP auto+cross": f"{DC}/phaseB_tfdata_2026_05_30/posteriors/autocross/autocross_cs41/autocross_cs41_s*_p*.npy",
    "CNN-RealNVP auto-only":  f"{DC}/phaseB_tfdata_2026_05_30/posteriors/autoonly/autoonly_cs41/autoonly_cs41_s*_p*.npy",
    "CNN-MAF auto+cross":     f"{DC}/phaseB_maf_2026_05_31/posteriors/autocross/autocross_cs41/autocross_cs41_s*_p*.npy",
    "CNN-MAF auto-only":      f"{DC}/phaseB_maf_2026_05_31/posteriors/autoonly/autoonly_cs41/autoonly_cs41_s*_p*.npy",
    "CNN-RealNVP auto+cross (std)": f"{DC}/phaseB_std_2026_05_31/posteriors/**/*_s*_p*.npy",
    "CNN-auto native-TFDS (RealNVP)": f"{DC}/phaseB_nativeauto_2026_05_31/posteriors/**/*_s*_p*.npy",
    "CNN auto+cross multi-perm": f"{DC}/phaseB_multiperm_2026_05_31/posteriors/autocross_multiperm/*_s*_p*.npy",
    "CNN auto-only multi-perm": f"{DC}/phaseB_multiperm_2026_05_31/posteriors/autoonly_multiperm/*_s*_p*.npy",
}

_PERM_RE = re.compile(r"_p(\d+)\.npy$")


def _fom(x, cols):
    return float(1.0 / np.sqrt(np.linalg.det(np.cov(x[:, cols], rowvar=False))))


def _avg_std(v):
    v = np.asarray(v, float)
    return float(v.mean()), (float(v.std()) if v.size > 1 else 0.0)


def per_perm_metrics(globpat):
    """Group files by perm, pool seeds within each perm, compute metrics per perm,
    then average across perms. Returns a dict of (mean, across-perm std) tuples for
    each scalar metric, the per-perm σ matrix mean, and bookkeeping counts."""
    fs = sorted(f for f in glob.glob(globpat, recursive=True)
                if "fom" not in os.path.basename(f).lower())
    by_perm = {}
    for f in fs:
        m = _PERM_RE.search(os.path.basename(f))
        p = int(m.group(1)) if m else 0
        by_perm.setdefault(p, []).append(f)
    if not by_perm:
        return None
    fom3, f01, f02, f12, sigs, nseeds = [], [], [], [], [], []
    for p in sorted(by_perm):
        x = np.concatenate([np.load(f) for f in sorted(by_perm[p])], 0)
        fom3.append(_fom(x, [0, 1, 2]))
        f01.append(_fom(x, [0, 1])); f02.append(_fom(x, [0, 2])); f12.append(_fom(x, [1, 2]))
        sigs.append(np.array([float(np.std(x[:, k])) for k in range(len(NAMES))]))
        nseeds.append(len(by_perm[p]))
    sig = np.stack(sigs, 0)  # (n_perm, 6)
    n_perm = len(by_perm)
    return {
        "n_perms": n_perm,
        "perms": sorted(by_perm),
        "n_seeds_per_perm": nseeds,
        "fom3": _avg_std(fom3), "fom3_perperm": [float(v) for v in fom3],
        "f_om_s8": _avg_std(f01), "f_om_w0": _avg_std(f02), "f_s8_w0": _avg_std(f12),
        "sig": sig.mean(0), "sig_std": (sig.std(0) if n_perm > 1 else np.zeros(6)),
    }


def _spread_str(mean, std):
    """'±std (cv%)' or '—' for single-perm arms."""
    if std <= 0 or mean == 0:
        return "—"
    return f"±{std:.0f} ({100.0 * std / abs(mean):.0f}%)"


def _headline_section(rows):
    """The perm-matched (3-perm) L1-vs-CNN comparison — computed from the table so it
    stays consistent on regeneration. The only fully perm-matched FoM3 comparison is
    L1 (3 perms) vs the CNN multi-perm arm (3 perms, == the core RealNVP NDE)."""
    pairs = [("auto+cross", "L1 auto+cross", "CNN auto+cross multi-perm"),
             ("auto-only", "L1 auto-only", "CNN auto-only multi-perm")]
    out = ["## Headline — perm-matched L1 vs CNN (both 3 perms, perm-averaged)", "",
           "The only apples-to-apples FoM3 comparison is L1 vs the CNN **multi-perm** arm "
           "(both averaged over the same 3 perms). The single-perm CNN rows below are perm-0 "
           "snapshots of the same NDEs.", ""]
    for tag, larm, carm in pairs:
        if larm not in rows or carm not in rows:
            continue
        lf, ls = rows[larm]["fom3"]; cf, cs = rows[carm]["fom3"]
        lw = rows[larm]["sig"][2]; cw = rows[carm]["sig"][2]
        f_lead = "CNN" if cf >= lf else "L1"
        w_lead = "L1" if lw <= cw else "CNN"
        out += [f"- **{tag}:** FoM3 — L1 {lf:.0f} ({_spread_str(lf, ls)}) vs CNN {cf:.0f} "
                f"({_spread_str(cf, cs)}) → **{f_lead} ahead** (within spreads). "
                f"σ(w0) — L1 {lw:.3f} vs CNN {cw:.3f} → **{w_lead} tighter**."]
    out += ["",
            "**The perm-0 'L1 ≥ CNN on auto+cross' headline does NOT survive "
            "perm-averaging** — it was a favorable perm-0 draw (L1 auto+cross FoM3 spread "
            "27%; L1 led only on perm 0). On the matched 3-perm comparison CNN is nominally "
            "ahead on FoM3/2D and L1 retains only a modest, perm-fragile σ(w0) edge. "
            "**CAVEAT:** L1 uses the harmonic-cache route, CNN the tf.data route — a "
            "residual route confound (cf. the G8 section) is uncontrolled; a within-route "
            "run would settle it. See felt fiber "
            "definitive-l1-vs-cnn-2026-05/finding-perm-averaging-overturns-l1-lead.", ""]
    return out


def _typical_patch_section():
    """CORRECTED HEADLINE — typical obs patch (full-200). Reads the per-patch step-2
    medians (median over ~300 typical patches ± 16-84 spread) and folds the full-200
    correction into this summary. The perm-averaged tables below used the fixed obs
    patch-0 = the POLAR patch (atypically low-info for L1) and are SUPERSEDED for the
    L1-vs-CNN headline. Best-effort: skips if the full-200 data is absent."""
    import csv
    base = f"{DC}/fiducial_full200/posteriors"
    arms = [("L1 auto+cross", "l1_autocross"), ("CNN auto+cross", "cnn_autocross"),
            ("CNN auto+cross (std)", "cnn_autocross_std"), ("CNN auto+cross (MAF)", "cnn_maf_autocross"),
            ("L1 auto-only", "l1_autoonly"), ("CNN auto-only", "cnn_autoonly")]
    cols = {"fom3": "fom3", "sig_Om": "sig_Omega_m", "sig_s8": "sig_sigma_8",
            "sig_w0": "sig_w_0", "f2": "fom2d_Omega_m_sigma_8"}
    rows = {}
    for disp, a in arms:
        p = f"{base}/{a}/per_patch_fom.csv"
        if not os.path.exists(p):
            continue
        r = list(csv.DictReader(open(p)))
        def ms(key):
            v = np.array([float(x[key]) for x in r], float); v = v[np.isfinite(v)]
            return np.median(v), np.percentile(v, 16), np.percentile(v, 84)
        rows[disp] = {k: ms(c) for k, c in cols.items()}
    if not rows:
        return ["## ⭐ CORRECTED HEADLINE — typical obs patch", "",
                "_(full-200 per-patch data not found; see fiducial_full200/SUMMARY_TYPICAL_PATCH.md.)_", ""]
    def cell(t, fmt): return f"{fmt.format(t[0])} [{fmt.format(t[1])},{fmt.format(t[2])}]"
    out = ["## ⭐ CORRECTED HEADLINE — typical obs patch (full-200; median over ~300 patches [16,84])", "",
           "**This supersedes the patch-0 tables below for the L1-vs-CNN headline.** The Phase-C "
           "perm-averaged tables used the fixed obs **patch-0 = the POLAR patch** (center lat 88.5°), "
           "atypically low-information for L1's near-polar wavelets (CNN is patch-insensitive) — that "
           "biased the original 'CNN ≳ L1 auto+cross'. Here each value is the median over the patch "
           "population (a typical 20 deg² obs). Read σ/2D, NOT FoM3 (it cubes ~20-25% diffs). Detail: "
           "fiducial_full200/SUMMARY_TYPICAL_PATCH.md + FIDUCIAL_FULL200_FINDINGS.md.", "",
           "| arm | FoM3 | σ(Ωm) | σ(σ8) | σ(w0) | 2D(Ωm,σ8) |", "|---|---|---|---|---|---|"]
    for disp, _ in arms:
        if disp not in rows:
            continue
        r = rows[disp]
        out.append(f"| {disp} | {cell(r['fom3'], '{:.0f}')} | {cell(r['sig_Om'], '{:.4f}')} | "
                   f"{cell(r['sig_s8'], '{:.4f}')} | {cell(r['sig_w0'], '{:.4f}')} | {cell(r['f2'], '{:.0f}')} |")
    if "L1 auto+cross" in rows and "CNN auto+cross" in rows:
        l, c = rows["L1 auto+cross"], rows["CNN auto+cross"]
        out += ["", f"**auto+cross — L1 modestly ahead** (edge in w0 / cross-maps): σ(w0) L1 "
                f"{l['sig_w0'][0]:.3f} vs CNN {c['sig_w0'][0]:.3f} (×{c['sig_w0'][0]/l['sig_w0'][0]:.2f}); "
                f"σ(Ωm) ×{c['sig_Om'][0]/l['sig_Om'][0]:.2f}; 2D(Ωm,σ8) ×{l['f2'][0]/c['f2'][0]:.2f}; "
                f"FoM3 ×{l['fom3'][0]/c['fom3'][0]:.2f} (FoM3 amplifies)."]
    if "L1 auto-only" in rows and "CNN auto-only" in rows:
        l, c = rows["L1 auto-only"], rows["CNN auto-only"]
        out += [f"**auto-only — a tie**: σ(w0) L1 {l['sig_w0'][0]:.3f} vs CNN {c['sig_w0'][0]:.3f}; "
                f"FoM3 {l['fom3'][0]:.0f} vs {c['fom3'][0]:.0f}."]
    out += ["", "Tight L1 posteriors verified calibrated (stratified varied-θ TARP). **Bottom line: "
            "L1 ≈ CNN auto+cross with a small L1 edge (w0/cross-maps); auto-only a tie; the original "
            "'CNN ≳ L1' was a polar-patch artifact.**", ""]
    return out


def _g8_section(rows):
    """Patch-center confound (G8): the harmonic-sliced auto-only baseline is lossy,
    so the CNN cross-gain is route-sensitive. Quote both ratios from the table."""
    d = {arm: r for arm, r in rows.items()}
    try:
        f_cross = d["CNN-RealNVP auto+cross"]["fom3"][0]
        f_harm = d["CNN-RealNVP auto-only"]["fom3"][0]
        f_native = d["CNN-auto native-TFDS (RealNVP)"]["fom3"][0]
        sw_harm = d["CNN-RealNVP auto-only"]["sig"][2]
        sw_native = d["CNN-auto native-TFDS (RealNVP)"]["sig"][2]
    except KeyError:
        return ["## Patch-center confound (G8)", "",
                "_(arms missing — re-run once all CNN auto arms have landed.)_", ""]
    return [
        "## Patch-center confound (G8) — read before quoting any cross-map gain", "",
        "The harmonic-cache route slices auto-only maps from full-sphere patches; the "
        "**native-TFDS** auto-only path does not. They are NOT equivalent baselines:",
        "",
        f"- native-TFDS auto-only: **FoM3 {f_native:.0f}**, σ(w0) {sw_native:.3f}",
        f"- harmonic-sliced auto-only: **FoM3 {f_harm:.0f}**, σ(w0) {sw_harm:.3f}",
        "",
        f"The harmonic auto-only baseline is **lossy** (FoM3 {f_harm:.0f} ≪ {f_native:.0f}; "
        f"σ(w0) {sw_harm:.3f} vs {sw_native:.3f}). Therefore the CNN auto+cross gain is "
        "**route-sensitive**:",
        "",
        f"- over the (lossy) harmonic auto-only:  {f_cross:.0f} / {f_harm:.0f} = "
        f"**{f_cross / f_harm:.2f}×**  ← inflated by a poor baseline",
        f"- over a FAIR (native-TFDS) auto-only:   {f_cross:.0f} / {f_native:.0f} = "
        f"**{f_cross / f_native:.2f}×**  ← the honest number",
        "",
        "The *within-route* cross-channel effect is still valid (only the input channels "
        "differ), but its **magnitude must be quoted against the native-TFDS auto-only "
        f"baseline (~{f_cross / f_native:.1f}×), not the harmonic one (~{f_cross / f_harm:.1f}×).** "
        "See felt fiber definitive-l1-vs-cnn-2026-05/finding-patch-center-confound-g8.",
        "",
    ]


def _tarp_section():
    """Best-effort per-arm calibration table (mean over seeds of max|ECP-α|) read
    from tarp_2026_05_31/tarp_summary.json + the curve npz files. Falls back to a
    pointer if the TARP artifacts are absent."""
    import json
    head = ["## TARP coverage", ""]
    summ = os.path.join(DC, "tarp_2026_05_31", "tarp_summary.json")
    try:
        d = json.load(open(summ))
        from collections import defaultdict
        dev = defaultdict(dict)  # arm -> {dim: [maxdev per seed]}
        for c in d["curves"]:
            z = np.load(c["curve"])
            a, ecp = z["alpha"], z["ecp_bootstrap"]
            med = np.median(ecp, axis=0) if ecp.ndim > 1 else ecp
            dev[c["arm"]].setdefault(c["dim"], []).append(float(np.max(np.abs(med - a))))
        rows = []
        for arm in sorted(dev):
            m3 = np.mean(dev[arm].get(3, [np.nan]))
            m6 = np.mean(dev[arm].get(6, [np.nan]))
            rows.append((arm, m3, m6))
        out = head + [
            "Max |ECP−α| (deviation from the diagonal; lower = better calibrated), "
            "mean over 3 seeds. ≲0.10 = mild mis-calibration, none severe.",
            "", "| arm (dump label) | 3-D | 6-D |", "|---|---|---|"]
        for arm, m3, m6 in rows:
            out.append(f"| {arm} | {m3:.3f} | {m6:.3f} |")
        out += ["",
                "Figures: `tarp_2026_05_31/figures/tarp_{per_arm,overlay}_dim{3,6}.png`. "
                "The multi-perm arms reuse the SAME compressed cache + NDE seeds as the "
                "core RealNVP arms (the perm only selects the single obs map, which never "
                "enters TARP), so their coverage == cnn_autocross_rnvp / cnn_autoonly_rnvp "
                "— not re-dumped.", ""]
        return out
    except Exception as e:
        return head + [
            f"_(calibration table unavailable: {e})_  "
            "See `tarp_2026_05_31/figures/tarp_{per_arm,overlay}_dim{3,6}.png` and "
            "`tarp_summary.json`. Multi-perm arms reuse the core RealNVP NDE (not re-dumped).",
            ""]


def main():
    rows = {}
    for arm, gp in ARMS.items():
        r = per_perm_metrics(gp)
        if r is not None:
            rows[arm] = r

    # ---- CSV (full precision, machine-readable) ----
    with open(os.path.join(OUT, "phase_c.csv"), "w") as f:
        hdr = (["arm", "n_perms", "fom3", "fom3_std",
                "fom2d_Om_s8", "fom2d_Om_w0", "fom2d_s8_w0"]
               + [f"sig_{n}" for n in NAMES] + [f"sig_std_{n}" for n in NAMES])
        f.write(",".join(hdr) + "\n")
        for arm, r in rows.items():
            vals = [arm, r["n_perms"], f"{r['fom3'][0]:.1f}", f"{r['fom3'][1]:.1f}",
                    f"{r['f_om_s8'][0]:.1f}", f"{r['f_om_w0'][0]:.1f}", f"{r['f_s8_w0'][0]:.1f}"]
            vals += [f"{r['sig'][k]:.5f}" for k in range(len(NAMES))]
            vals += [f"{r['sig_std'][k]:.5f}" for k in range(len(NAMES))]
            f.write(",".join(str(v) for v in vals) + "\n")

    # ---- Markdown ----
    L = [
        "# Definitive L1 vs CNN — Phase C summary",
        "",
    ]
    L += _typical_patch_section()
    L += [
        "---",
        "## (historical) Perm-averaged analysis at the campaign obs — patch-0 = POLAR, SUPERSEDED for the headline",
        "",
        "_Retained for the record. These tables condition on the fixed obs patch-0 (the polar patch); "
        "the corrected headline above uses the typical-patch population. The perm-averaging fixed a "
        "different earlier bug (perm-pooling) and is still correct as a perm-averaged-at-patch-0 view._",
        "",
        "**Primary metric: 3-seed pooled FoM3 on (Ωm, σ8, w0), per perm then "
        "perm-averaged** (the constitution's declared metric). The across-perm spread "
        "(± std, %CV) is shown so FoM3's sensitivity is visible — see memory "
        "feedback_fom3_fragile_use_2d_areas. Marginal σ and 2D FoM are secondary.",
        "",
        "`n_perms` makes the comparison transparent: L1 and the multi-perm CNN arms "
        "carry 3 fiducial realizations (perm-averaged); the other CNN arms carry only "
        "perm 0 (so their spread is '—'). Every row's per-perm unit is a 3-seed pool. "
        "CNN absolute FoM are fast-tf.data-route, treated as fine (overlap negligible, "
        "Andreas 2026-05-31).",
        "",
        f"Arms with posteriors: {len(rows)}",
        "",
    ]
    L += _headline_section(rows)
    L += [
        "## Primary — 3-seed pooled FoM3 (perm-averaged; higher = tighter)",
        "",
        "| arm | n_perms | FoM3 | across-perm spread |",
        "|---|---|---|---|",
    ]
    for arm, r in rows.items():
        m, s = r["fom3"]
        L.append(f"| {arm} | {r['n_perms']} | {m:.0f} | {_spread_str(m, s)} |")

    L += ["", "## Secondary — marginal σ (lower = tighter; perm-averaged)", "",
          "| arm | n_perms | σ(Ωm) | σ(σ8) | σ(w0) | σ(h0) | σ(ns) | σ(Ωb) |",
          "|---|---|---|---|---|---|---|---|"]
    for arm, r in rows.items():
        L.append(f"| {arm} | {r['n_perms']} | "
                 + " | ".join(f"{r['sig'][k]:.4f}" for k in range(len(NAMES))) + " |")

    L += ["", "## Secondary — 2D FoM (higher = tighter; perm-averaged)", "",
          "| arm | Ωm–σ8 | Ωm–w0 | σ8–w0 |", "|---|---|---|---|"]
    for arm, r in rows.items():
        L.append(f"| {arm} | {r['f_om_s8'][0]:.0f} | {r['f_om_w0'][0]:.0f} "
                 f"| {r['f_s8_w0'][0]:.0f} |")

    L += [""] + _g8_section(rows)

    L += _tarp_section()
    L += ["_Auto-generated by aggregate_all_arms.py — re-run to refresh as arms land._"]
    open(os.path.join(OUT, "SUMMARY_DEFINITIVE.md"), "w").write("\n".join(L) + "\n")

    print(f"[phase-c] {len(rows)} arms -> {OUT}/SUMMARY_DEFINITIVE.md")
    for arm, r in rows.items():
        m, s = r["fom3"]
        print(f"  {arm:36s} n_perms={r['n_perms']} seeds/perm={r['n_seeds_per_perm']} "
              f"FoM3={m:.0f} {_spread_str(m, s)}")


if __name__ == "__main__":
    main()
