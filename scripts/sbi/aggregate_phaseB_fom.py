#!/usr/bin/env python3
"""Phase C (lite) — aggregate Phase B jaxili-NDE FoM into a comparison table.

Reads phaseB_tfdata_2026_05_30/posteriors/<arm>/<arm>_cs<CS>_s<NDE>_p<PERM>.fom.json,
writes SUMMARY_PHASEB.md + phaseB_fom.csv. No plotting deps. Idempotent.

⚠️ These FoM carry the Phase A tf.data leakage (README_LEAKAGE.md): ABSOLUTE FoM
inflated; the trustworthy signal is the auto+cross-vs-auto-only RELATIVE gain.
"""
from __future__ import annotations
import glob, json, os, re, statistics as st
from pathlib import Path

PB = Path(__file__).resolve().parent / "results/exploratory/definitive_comparison/phaseB_tfdata_2026_05_30"
PAT = re.compile(r"^(autocross|autoonly)_cs(\d+)_s(\d+)_p(\d+)\.fom\.json$")
PARAMS = ["Omega_m", "sigma_8", "w_0", "h_0", "n_s", "Omega_b"]


def load_rows():
    rows = []
    for f in sorted(glob.glob(str(PB / "posteriors" / "**" / "*.fom.json"), recursive=True)):
        m = PAT.match(os.path.basename(f))
        if not m:
            continue
        arm, cs, nde, perm = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"  skip {f}: {e}")
            continue
        rows.append({
            "arm": arm, "cs": cs, "nde": nde, "perm": perm,
            "fom3": d.get("fom3"), "valid": d.get("valid_fom3"),
            "fom2d_Om_s8": d.get("fom2d_Omega_m_sigma_8"),
            "fom2d_Om_w0": d.get("fom2d_Omega_m_w_0"),
            "fom2d_s8_w0": d.get("fom2d_sigma_8_w_0"),
            **{f"sig_{p}": d.get("sigma", {}).get(p) for p in PARAMS},
        })
    return rows


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return st.mean(xs) if xs else float("nan")


def main():
    rows = load_rows()
    PB.mkdir(parents=True, exist_ok=True)
    # CSV
    cols = ["arm", "cs", "nde", "perm", "fom3", "valid", "fom2d_Om_s8",
            "fom2d_Om_w0", "fom2d_s8_w0"] + [f"sig_{p}" for p in PARAMS]
    with open(PB / "phaseB_fom.csv", "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    # Headline: compressor-seed 41 across NDE seeds (plan-faithful 3-NDE-seed set)
    def headline(arm):
        return [r for r in rows if r["arm"] == arm and r["cs"] == 41]
    ac, ao = headline("autocross"), headline("autoonly")
    out = ["# Phase B FoM — CNN auto+cross vs auto-only (tf.data route, 2026-05-30)",
           "",
           "⚠️ **Leakage flag** (README_LEAKAGE.md): absolute FoM inflated ~1.6×; "
           "trust the auto-vs-cross **relative** gain, not absolute values.", "",
           f"Rows found: {len(rows)}  (autocross cs41: {len(ac)}, autoonly cs41: {len(ao)})", ""]
    if ac and ao:
        ac_fom, ao_fom = mean([r["fom3"] for r in ac]), mean([r["fom3"] for r in ao])
        out += ["## Headline (compressor seed 41, NDE seeds pooled by mean)", "",
                f"- auto+cross FoM3 = **{ac_fom:.0f}**   (n={len(ac)})",
                f"- auto-only  FoM3 = **{ao_fom:.0f}**   (n={len(ao)})",
                f"- **FoM3 cross/auto ratio = {ac_fom/ao_fom:.2f}×**" if ao_fom else "",
                ""]
        out += ["### Marginal σ (mean) and cross-tightening factor", "",
                "| param | auto-only σ | auto+cross σ | tighten (auto/cross) |",
                "|---|---|---|---|"]
        for p in PARAMS:
            so, sc = mean([r[f"sig_{p}"] for r in ao]), mean([r[f"sig_{p}"] for r in ac])
            fac = so / sc if sc else float("nan")
            out.append(f"| {p} | {so:.4f} | {sc:.4f} | {fac:.2f}× |")
        out += ["", "### 2D FoM (mean) and cross/auto ratio", "",
                "| pair | auto-only | auto+cross | ratio |", "|---|---|---|---|"]
        for key, lab in [("fom2d_Om_s8", "Ωm–σ8"), ("fom2d_Om_w0", "Ωm–w0"), ("fom2d_s8_w0", "σ8–w0")]:
            vo, vc = mean([r[key] for r in ao]), mean([r[key] for r in ac])
            out.append(f"| {lab} | {vo:.1f} | {vc:.1f} | {vc/vo:.2f}× |" if vo else f"| {lab} | {vo} | {vc} | n/a |")
    else:
        out += ["_(headline incomplete — waiting on cs41 NDE for one or both arms)_"]
    # compressor-seed variance (bonus)
    out += ["", "## Per-run table", "", "| arm | cs | nde | perm | FoM3 | valid |",
            "|---|---|---|---|---|---|"]
    for r in rows:
        f3 = f"{r['fom3']:.0f}" if isinstance(r["fom3"], (int, float)) else r["fom3"]
        out.append(f"| {r['arm']} | {r['cs']} | {r['nde']} | {r['perm']} | {f3} | {r['valid']} |")
    (PB / "SUMMARY_PHASEB.md").write_text("\n".join(out) + "\n")
    print(f"[aggregate] {len(rows)} rows -> {PB/'SUMMARY_PHASEB.md'} + phaseB_fom.csv")


if __name__ == "__main__":
    main()
