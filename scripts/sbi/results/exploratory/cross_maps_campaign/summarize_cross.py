"""Aggregate the cross-map runs (auto+cross, zero-mean, multipatch) and contrast
against the in-tree auto-only reference. Flags the parity caveat: the auto-only
runs do NOT use --zero-mean-maps and use grid_20deg_160px (single-patch), so
absolute FoM ratios are confounded by dataset size and mass-sheet demeaning.

Writes summary.md and summary.json under cross_summary/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/cross_maps_campaign")
OUT = ROOT / "cross_summary"
OUT.mkdir(exist_ok=True)
SEEDS = [41, 42, 43]
PARAM_NAMES = ["Omega_m", "sigma_8", "w_0", "h_0", "n_s", "Omega_b"]


def stats(p):
    s = p.std(0)
    cov3 = np.cov(p[:, :3].T)
    return {
        "mean": p.mean(0).tolist(),
        "std": s.tolist(),
        "std_sum_3par": float(s[:3].sum()),
        "omega_m_std": float(s[0]),
        "sigma8_std": float(s[1]),
        "w0_std": float(s[2]),
        "fom3": float(np.exp(-0.5 * np.linalg.slogdet(cov3)[1])),
    }


def agg(posts):
    ss = [stats(p) for p in posts]
    keys = ["std_sum_3par", "omega_m_std", "sigma8_std", "w0_std", "fom3"]
    return {k: float(np.mean([s[k] for s in ss])) for k in keys} | {"per_seed": ss}


def load(template):
    return [np.load(template.format(s=s)) for s in SEEDS]


def _all_present(template) -> bool:
    return all(Path(template.format(s=s)).exists() for s in SEEDS)


def main():
    arms = {
        "cross_bnt_pct1":   ROOT / "jaxili_cross_bnt_pct1"   / "posteriors" / "l1cross_tomo4_20deg160mp_bnt_p1_s{s}.npy",
        "cross_nobnt_pct1": ROOT / "jaxili_cross_nobnt_pct1" / "posteriors" / "l1cross_tomo4_20deg160mp_nobnt_p1_s{s}.npy",
        "cross_bnt":        ROOT / "jaxili_cross_bnt"        / "posteriors" / "l1cross_tomo4_20deg160mp_bnt_s{s}.npy",
        "cross_nobnt":      ROOT / "jaxili_cross_nobnt"      / "posteriors" / "l1cross_tomo4_20deg160mp_nobnt_s{s}.npy",
        "auto_zm_bnt":      ROOT / "jaxili_auto_zm_bnt"      / "posteriors" / "l1_tomo4_20deg160mp_zm_bnt_s{s}.npy",
        "auto_zm_nobnt":    ROOT / "jaxili_auto_zm_nobnt"    / "posteriors" / "l1_tomo4_20deg160mp_zm_nobnt_s{s}.npy",
        "auto_bnt":         ROOT / "jaxili_auto_bnt"         / "posteriors" / "l1_tomo4_20deg160_bnt_s{s}.npy",
        "auto_nobnt":       ROOT / "jaxili_auto_nobnt"       / "posteriors" / "l1_tomo4_20deg160_nobnt_s{s}.npy",
        "harm_cross_bnt":   ROOT / "jaxili_harm_cross_bnt"   / "posteriors" / "l1cross_tomo4_20deg160mp_harm_bnt_p1_s{s}.npy",
        "harm_cross_nobnt": ROOT / "jaxili_harm_cross_nobnt" / "posteriors" / "l1cross_tomo4_20deg160mp_harm_nobnt_p1_s{s}.npy",
    }
    skipped = [name for name, t in arms.items() if not _all_present(str(t))]
    if skipped:
        print(f"# skipped (missing posteriors): {skipped}")
    arms = {n: t for n, t in arms.items() if n not in skipped}
    summaries = {name: agg(load(str(t))) for name, t in arms.items()}

    md = [
        "# Cross-maps campaign summary",
        "",
        "**Apples-to-apples comparison: `cross_*` vs `auto_zm_*`** — both arms use "
        "`grid_20deg_160px_nonoverlap48` (multipatch, ~150k maps) with "
        "`--zero-mean-maps`, run via the same script. The only difference is the "
        "channel set (4 auto vs 4 auto + 6 cross).",
        "",
        "**Legacy reference: `auto_bnt` / `auto_nobnt`** — single-patch "
        "`grid_20deg_160px`, no zero-mean. Kept for context only; do NOT use for "
        "the cross-information contrast.",
        "",
        "## Aggregate FoM3 (mean over seeds 41/42/43)",
        "",
        "| arm | std_sum_3par | sigma_omega_m | sigma_sigma_8 | sigma_w_0 | FoM3 |",
        "|---|---|---|---|---|---|",
    ]
    keys = ["std_sum_3par", "omega_m_std", "sigma8_std", "w0_std", "fom3"]
    for name, s in summaries.items():
        md.append(f"| {name} | {s['std_sum_3par']:.5f} | {s['omega_m_std']:.5f} | "
                  f"{s['sigma8_std']:.5f} | {s['w0_std']:.5f} | {s['fom3']:.2f} |")
    md.append("")
    md.append("## Per-seed FoM3")
    md.append("")
    md.append("| arm | s41 | s42 | s43 |")
    md.append("|---|---|---|---|")
    for name, s in summaries.items():
        f = [ps["fom3"] for ps in s["per_seed"]]
        md.append(f"| {name} | {f[0]:.2f} | {f[1]:.2f} | {f[2]:.2f} |")
    md.append("")
    md.append("## Cross-vs-auto-zm ratios (matched comparison)")
    md.append("")
    md.append("| metric | BNT pct1 / auto_zm | BNT min/max / auto_zm | no-BNT pct1 / auto_zm | no-BNT min/max / auto_zm |")
    md.append("|---|---|---|---|---|")
    for k in keys:
        b_p = summaries["cross_bnt_pct1"][k];  b_c = summaries["cross_bnt"][k];  b_a = summaries["auto_zm_bnt"][k]
        n_p = summaries["cross_nobnt_pct1"][k]; n_c = summaries["cross_nobnt"][k]; n_a = summaries["auto_zm_nobnt"][k]
        md.append(f"| {k} | {b_p/b_a:.3f} | {b_c/b_a:.3f} | {n_p/n_a:.3f} | {n_c/n_a:.3f} |")
    md.append("")
    md.append("Interpretation: ratio > 1 for FoM3 means cross channels add information; "
              "ratio < 1 for std means tighter constraints (good).")

    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"wrote {OUT/'summary.md'}")
    print(f"wrote {OUT/'summary.json'}")
    print()
    print("\n".join(md))


if __name__ == "__main__":
    main()
