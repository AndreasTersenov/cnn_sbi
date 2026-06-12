# HANDOFF — overnight menu run (morning of 2026-06-12)

**TL;DR:** all 8 planned arms + 8 unplanned follow-up arms ran (main run 0.8 h + retries,
GPUs 0/1/2 tenant-polite, zero unresolved failures). Read
`overnight_menu/OVERNIGHT_RESULT.md` — tables + the "Night synthesis" section at the bottom.
Plan + every registered prediction/band: `PLAN_OVERNIGHT_MENU.md` (incl. the two night
addenda registered before their runs). Chronology: `OVERNIGHT_STATUS.md`.

## The five takeaways

1. **The joint statistics WORK** (your joint-PDF and joint-l1 asks): pairwise joint PDF
   2794, joint wavelet l1 2788 (full rigor, noBNT) vs l1-auto 2405 (+16%) — at the
   l1+product level (2875). Plain counts ≈ l1-weighted cells: the information is in joint
   occupancy, not amplitude weighting.
2. **Rescue menu fully closed:** unions6 rescues BNT-L1 at 1.18 (survey practice validated,
   matches the deep/deep2 span story); the P7 covariance block measures the **Gaussian share
   of the l1's BNT loss = 38%** — a new, quotable decomposition (62% of the loss is
   non-Gaussian content).
3. **"Joint PDF is BNT-robust by construction" acquired a sharp qualifier** (the night's
   main theory output): the distribution is basis-covariant (P4b) but a BINNED estimator is
   only as invariant as its grid is transported. Measured ladder: fixed noise-scaled grid
   ratio 0.45 → axis-adapted percentile grid 0.70 → exact transport would need SHEARED
   cells (B-images of cells are parallelepipeds), impossible for any axis-aligned
   histogram. The learned compressor's first layer implements exactly that shear — the
   basis-adaptivity advantage, now visible at the estimator level. (Registered bands ≥0.75 /
   ≤0.55: landed between, toward support.)
4. **Engineering:** count-histogram datavectors need dequantization (+U(0,1), seeded) or the
   MAF NaNs on quasi-discrete sparse cells — hit and fixed three times (full4d K=5, then the
   BNT-side jointl1/pair2d full sweeps). All final numbers are from dequantized rebuilds with
   matched treatment in both bases.
5. **VMIM was NOT run** (your last-if-warranted rule): no dimensionality limitation appeared
   anywhere — 3000–3200-dim datavectors train fine through the jaxili MAF.

## Decisions for this morning

- Whether the joint-PDF/joint-l1 result (beats l1-auto by 16%, ≈ l1+product, with no
  cross-map construction at all) is paper material for §1.7/§4.3 — I'd argue yes: it
  empirically completes the hierarchy P5 (marginals < few projections < joint one-point).
- Whether to quote the 38% Gaussian-share number in the BNT section (it sharpens "the
  collapse lives in the non-Gaussian sector").
- The grid-transport qualifier belongs in deep-dive §1.8/§4.3 and the M3 design sheet
  (an adapted/learned binning is part of the statistic).
- Cheap follow-ups NOT run (diminishing returns at night): pair2da (decompose the pairwise
  0.52 ratio into grid vs incompleteness), K=15 pair2d (resolution scaling), VMIM variants.

## Bookkeeping

All numbers pooled medians from `median_summary.json` under
`overnight_menu/<arm>/population_sweep[_full]/`. Screening = 1 seed/3000 obs; full = 3
seeds/9000 obs. Doc folds (deep-dive, FLATSKY_BNT_RESULT) deliberately NOT made — per your
instruction, tables + this handoff only. Night commits: f527173, 0a4b1a2, 2a18080, 97cd362,
3d69df7, aa6f3a6, 97ddf61 (+ the results commit after this file).
