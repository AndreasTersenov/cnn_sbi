# Stale-figure audit — CNN-vs-L1 comparison figures (2026-06-15)

**Trigger:** the CNN was fixed (ResNet18 + sbi_lens RealNVP → FoM3 ~3293, calibrated) and L1+product
was matched to it through the same NDE (RESULT_ANALYTICAL_NDE_MATCH.md). Any figure built on the OLD
CNN (common-MAF, ~2300–2620) and/or the OLD full-sphere LEAKY auto+cross (FoM3 axis to ~120k) is now
stale for the paper. Verified by viewing the headline figures, not assumed.

## ✅ `flatsky_cross_2026_06/figs/` — CLEAN (no stale CNN comparison figures)
All 10 are current (flat-local) and either **L1-only** or **data** figures — none overlay the stale CNN:
- `representative_corner_{typical,favorable}` — L1 arms only (auto / +conv / +product / +both). KEEP.
- `flatsky_showcase` — L1 de-leaked cross-gain bars (2405/2499/2875/2910). L1-only. KEEP.
- `l1_hist_vs_s8*`, `l1_hist_zoom`, `datavector_full_*`, `l1_matrix_corner_science`, `maps_examples`
  — L1 datavector / map illustrations. KEEP.

## ❌ `definitive_comparison/` — STALE (do NOT use for the paper)
The ENTIRE tree predates both the flat-sky de-leak and the NDE-swap (figures dated 2026-05-28 →
06-04). Every CNN-vs-L1 figure here uses the common-MAF / pre-ResNet18 CNN; most also use the
full-sphere LEAKY auto+cross. Confirmed by inspection:
- `fiducial_full200/figures/headline_typical_patch_violins.png` — **the old "headline" violin**;
  per-patch FoM3 axis to ~120k (full-sphere leaky). DROP. (Andreas already flagged this one.)
- `fiducial_full200/overlays/fom3_distribution.png` — same era FoM3 distribution. DROP.
- `PHASE_C_2026_05_31/overlays/01_headline_l1_vs_cnn_autocross.png` + `01b_..._6param` — old headline
  L1-vs-CNN corner (full-sphere auto+cross, old CNN). DROP.
- `PHASE_C_2026_05_31/overlays/{03_perm_sensitivity_CNN_autocross, 05_auto_only_l1_vs_cnn,
  06_cnn_flavors_autocross}.png` — old CNN. DROP.
- `phaseB_tfdata_2026_05_30/figures/{l1_vs_cnn_autocross, l1_vs_cnn_autoonly,
  fom3_bar_autocross_vs_autoonly}.png` — old route + old CNN. DROP.
- `companion_comparison_2026_05_31/compare_{autoonly,autocross}.png` — old MAF-companion CNN study. DROP.
- `tarp_2026_05_31/figures/*`, `fiducial_full200/tarp_stratified/figures/*`,
  `fiducial_full200/calibration/cnn_*` — old CNN calibration. DROP (use analytical_nde_match gates).
- L1-only figures here (`figures/definitive_l1/*`, `figures/early/l1_auto_*`) are not CNN-stale but use
  full-sphere/old-route L1 numbers ⇒ also not paper-current; use `flatsky_cross_2026_06/` instead.

## ✅ Replacements (current, proper — all gated, flat-local, NDE-matched)
- **L1-vs-CNN comparison:** `analytical_nde_match/fom3_matrix`, `tarp_overlay`/`tarp_pooled`,
  `sbc_overlay`/`sbc_pooled`, `contour_overlay_meanobs`, `violins_l1_vs_cnn`, `violin_fom3_l1_vs_cnn`.
- **CNN side:** `cnn_phase/` (`nde_sweep_2026_06_13`, `arch_sweep_2026_06_13` resnet18, `gate_c`).
- **L1 cross-gain / data:** `flatsky_cross_2026_06/figs/`.

## Recommendation
Pull NO paper comparison figure from `definitive_comparison/` — treat it as the pre-de-leak/pre-NDE-swap
archive. The single highest risk of accidental reuse is `headline_typical_patch_violins.png` (it was
literally named "headline"); its replacement is `analytical_nde_match/violin_fom3_l1_vs_cnn.png` (+ the
4-panel `violins_l1_vs_cnn.png`). No regeneration of the stale figures is warranted — the proper
replacements already exist; the stale ones would each need a full pre-flat-local rerun to "fix," which
is pointless. Leave them on disk as provenance; just don't cite them.
