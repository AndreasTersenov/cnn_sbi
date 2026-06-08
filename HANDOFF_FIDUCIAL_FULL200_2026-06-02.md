# HANDOFF — Full-200 fiducial study (launched overnight 2026-06-02 04:43 UTC)

Running **detached** (survives session) on **GPU 1**, self-documenting. Felt fiber:
`definitive-l1-vs-cnn-2026-05/fiducial-full200-meandv` (active).

## What it does (your request)
Extend the fiducial observation from 3 perms to the **full 200 realizations** (9600
patches = 200×48). Per arm:
- **Step 1** — posterior at the **mean of all 9600 per-patch summaries** (de-noised,
  realization-independent headline contour). ⚠️ This is **single-survey width** centered
  at fiducial, **NOT** a 200×-tighter constraint (the NDE encodes the per-patch noise).
- **Step 2** — FoM3/σ over **~300 individual 20 deg² patches** → the real "which-sky"
  distribution (replaces the flimsy 3-perm spread).

6 arms: L1 {auto+cross, auto-only}, CNN {auto+cross, auto-only}, CNN auto+cross std,
CNN auto+cross MAF.

## Where to look in the morning
- **Results:** `scripts/sbi/results/exploratory/definitive_comparison/fiducial_full200/FIDUCIAL_FULL200_SUMMARY.md`
- **Plots:** `…/fiducial_full200/overlays/` (mean-dv L1-vs-CNN corners; per-patch FoM3 dist)
- **Live progress:** `…/fiducial_full200/STATUS.log`; done marker `.FIDUCIAL_FULL200_DONE`
- **Per-arm logs:** `…/fiducial_full200/logs/{summary,analyze}_<arm>.log`
- **Plan:** `…/fiducial_full200/FIDUCIAL_FULL200_PLAN.md` (or `…/definitive_comparison/FIDUCIAL_FULL200_PLAN.md`)

## Correctness gates (so morning results are trustworthy or clearly-failed, never silent-wrong)
- **Build:** 200 perms verified (G2: perm10 matches perm6 structure, distinct noise seed).
- **Summary extractors validated on perm0:** CNN G1 max|Δ|=4.7e-5 (reproduces `cnn_obs`);
  L1 hook G1 max|Δ|=0.0 **and** `[G1-calib] MATCH` vs the campaign cache meta.
- **G3 per arm (end-to-end):** at perm0/patch0 the 3-seed-pooled FoM3 must reproduce the
  campaign perm-0 value within 20% (L1ac 34607 / L1ao 10560 / CNNac 26748 / CNNao 9125 /
  CNNmaf 11984 / CNNstd 24281). **FAIL → that arm aborts and is skipped (logged); summaries
  still saved.** So check STATUS.log for any `FAIL` lines.

## Code (uncommitted; in working tree)
- `build_full_sphere_cross_cache.py`: 2 fixes — perm zero-pad `perm_{p:04d}` (was `perm_000{p}`,
  broke for perm≥10) + new `--cosmo-id` filter (build only cosmo_fiducial → 6.6 GB not 113 GB).
- NEW: `build_fiducial_summaries_cnn.py`, `fiducial_analyze.py`, `plot_fiducial.py`,
  `run_fiducial_full200.sh`.
- `npe_l1norm_cross_jaxili_nbody_tomo.py`: additive `--fiducial-summaries-out` /
  `--fiducial-perms` / `--fiducial-obs-cache-dir` hook (after obs_l1; early-return; G1 + G1-calib).
- Disk: +6.6 GB fiducial cache (built-and-keep). /mnt had 152 GB free.

## If something's wrong
- All `FAIL` in STATUS.log → read the arm's `analyze_<arm>.log` (likely G3 mismatch = a
  preprocessing diff for that arm). Summaries (`fiducial_full200/summaries/*.npz`) are the
  reusable artifact; analysis can be re-run from them.
- Headline interpretation reminder lives in the fiber + `project_perm_averaging_overturns_l1_autocross_lead` memory.
