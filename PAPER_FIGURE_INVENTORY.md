# Paper figure inventory — illustrative keepers + placeholder result figures

**Purpose (Andreas, 2026-06-09).** Until the final results are available, `paper-draft` should lay the
paper out with **placeholder figures from the existing (non-final / superseded / wrong) runs**, so the
manuscript's look is clear early. Separately, the repo already has **correct, reusable illustrative
plots** (maps, data-vector visualizations, calibration diagnostics) that improve the paper and ease the
explanations — those are keepers, not placeholders. This inventory points to both.

**Conventions.** Paths are repo-relative. **Prefer the `.pdf`** when listed; a figure tagged
*(PNG only)* has no vector version. Three status tags:
- **KEEPER** — correct & reusable in the final paper as-is (illustrative / methods / data visualization).
- **NEAR-FINAL** — an actual result figure from the citable 10° campaign; will likely stand, except
  the **auto+cross** content is PROVISIONAL pending the flat-sky cross rebuild (regenerate then).
- **PLACEHOLDER** — from a non-final / superseded / WRONG run; use for layout now, **swap when final**.

Status of the underlying runs is per `PAPER_FILE_TRIAGE.md`. Don't headline the *numbers* in PLACEHOLDER
figures; they're for layout. (FoM3 is fine to show — the "don't headline FoM3" rule is retired.)

---

## A. KEEPER — illustrative / methods / data-visualization figures (reusable in the final paper)

These are not result contours; they explain the data and method and make the paper look good. All are
post-`zero-mean-maps`-era constructions or pure visualizations, independent of the L1-vs-CNN verdict.

### A1. Map & data-vector visualizations (the "nice maps")

| figure (prefer PDF) | what it shows | suggested paper role |
|---|---|---|
| `scripts/sbi/results/diagnostics/cross_maps/cross_maps_gallery.pdf` | A 10°×10° patch gallery: **top row = 4 tomographic auto maps κ₁–κ₄, bottom row = 6 cross channels** (1×2…3×4). The actual input data vector the compressors see. | **Methods hero figure** — "the tomographic auto + cross map channels." |
| `scripts/sbi/results/diagnostics/full_sphere_cross_maps/fiducial/fullsphere_maps_nobnt.png` *(PNG only)* | Mollweide **full-sphere** panel: 4 auto channels (amplitude grows with bin) + 6 cross channels, which are visibly **smooth / large-scale / coherent across the sphere** (esp. cross 3×4). | **The leakage illustration** — visually motivates why full-sphere cross-patches carry non-local info (#6). Pair with D9. |
| `scripts/sbi/results/diagnostics/full_sphere_cross_maps/fiducial/fullsphere_maps_bnt.png` *(PNG only)* | Same, BNT-applied — for the BNT/cross discussion. | Pillar-2 / cross-map methods support. |
| `scripts/sbi/results/diagnostics/cross_maps/cross_maps_l1_datavectors.pdf` | The wavelet-ℓ₁ datavectors computed on the cross maps. | Methods — "the ℓ₁ statistic on cross channels." |
| `scripts/sbi/results/diagnostics/cross_maps/cross_maps_snr_histograms.pdf` | Per-channel SNR histograms on the cross maps. | Methods / pitfall support — ties to the L1 cross noise-model bug (#2). |

### A2. The ℓ₁ wavelet statistic — methods illustrations (`…/diagnostics/fig/l1norm_diagnostics/`, PNG only)

| figure | what it shows | suggested paper role |
|---|---|---|
| `…/l1norm_diagnostics/raw_l1_per_scale_tomobin.png` | Raw ℓ₁ datavector structure, per wavelet scale × tomo bin. | Methods — anatomy of the ℓ₁ summary. |
| `…/l1norm_diagnostics/l1_histograms_examples.png` | Example wavelet-coefficient ℓ₁ histograms. | Methods — what the ℓ₁-norm statistic is. |
| `…/l1norm_diagnostics/l1_vs_cosmology.png` | How the ℓ₁ datavector responds to cosmology. | Methods — "the statistic is cosmology-informative." |
| `…/l1norm_diagnostics/mean_l1_profiles.png` | Mean ℓ₁ profiles. | Methods support. |
| `…/l1norm_diagnostics/standardization_comparison.png` | Effect of the log1p / z-score preprocessing. | Methods / pitfall (preprocessing matters). |
| `…/l1norm_diagnostics/correlation_matrix.png`, `…/training_curves.png` | ℓ₁ feature correlation matrix; NDE training curves. | Appendix / supporting. |

### A3. Degeneracy & posterior-structure diagnostics (`…/diagnostics/degeneracy_v2/`, PNG only)

| figure | what it shows | suggested paper role |
|---|---|---|
| `…/degeneracy_v2/corr_heatmap_tomo4_10deg80.png` (+ `…_20deg160.png`, `…_bin3_*`) | Parameter-correlation heatmaps (the strong Ωm–σ8 ρ≈−0.93). | **Motivates the FoM3-fragility caveat**; methods/discussion. |
| `…/degeneracy_v2/posterior_eigenspectra.png`, `…/posterior_condition_numbers.png` | Posterior covariance eigenspectra / conditioning. | Discussion — why FoM3 amplifies correlation noise. |
| `…/degeneracy_v2/sensitivity_jacobian_singular_values.png`, `…/sensitivity_r2_first3_last3.png`, `…/pair_structure_comparison.png` | Summary→parameter sensitivity (Jacobian SVD, R²), parameter-pair structure. | Appendix — information-content diagnostics. |

### A4. The cross-map leakage scale decomposition (CITE-grade)

| figure (prefer PDF) | what it shows | suggested paper role |
|---|---|---|
| `scripts/sbi/results/exploratory/definitive_comparison_10deg/phase_c/analysis/figs/D9_cross_leakage_scales.pdf` | % of each channel's variance at super-patch scales (ℓ<18): **autos 0.4–1%, cross 12–20%**, with cross ℓ_median crashing to ~60. The quantitative leakage evidence. | **Pillar-1 leakage figure** (Section on cross-map strategy). KEEPER — this is part of the citable `CROSS_MAP_LEAKAGE_FINDING.md`. |

---

## B. NEAR-FINAL — result figures from the citable 10° campaign

From `scripts/sbi/results/exploratory/definitive_comparison_10deg/phase_c/analysis/` (the CITE anchor,
`SUMMARY_PHASE_D.md`). All have **PDF + PNG → use the PDF**. These are the actual result figures; the
**auto-only** content is clean, the **auto+cross** content is PROVISIONAL under the leakage finding
(#6) and should be regenerated after the flat-sky rebuild. Calibration figures are solid either way.

| figure (PDF) | what it shows | status / paper role |
|---|---|---|
| `…/figs/D1_constraining_power.pdf` | σ(Ωm/σ8/w0) + 2D area + FoM3, all 4 arms (L1/CNN × auto/auto+cross). | NEAR-FINAL — **main results figure**; auto+cross provisional. |
| `…/figs/D4_corner_autocross.pdf` | The headline **CNN vs L1 auto+cross corner**. | NEAR-FINAL — **headline corner**; provisional (regen post-flat-sky). |
| `…/figs/D2_w0_offset.pdf` | The w0 fiducial offset shrinking 20°→10° (−0.37σ → −0.10σ), no longer L1-specific. | NEAR-FINAL — the **w0-artifact resolution** figure. |
| `…/figs/D3_tarp_coverage.pdf`, `…/figs/D3b_tarp_{coverage,residual}_dim{3,6}.pdf` | TARP coverage / residuals. | NEAR-FINAL — calibration. |
| `…/tarp_drp/figures/tarp_{overlay,per_arm}_dim{3,6}[_colored].pdf` | Proper varied-θ TARP-DRP, per-arm & overlaid, 3-D & 6-D (the valid TARP). | NEAR-FINAL — **calibration centerpiece** (all arms on the diagonal incl. tight HIGH-FoM3 tercile). |
| `…/figs/D5_sbc_ranks.pdf` | SBC rank histograms (uniform on Ωm/σ8/w0). | NEAR-FINAL — calibration. |
| `…/figs/D6_lc2st.pdf` | L-C2ST local calibration at the fiducial (0/30 reject). | NEAR-FINAL — calibration. |
| `…/figs/D7_local_coverage_vs_latitude.pdf` | Per-patch z-std vs latitude (sharpness≠calibration; CNN mildly conservative). | NEAR-FINAL — discussion (CNN tightness is real). |
| `…/figs/D8_shrinkage.pdf` | Fiducial bias vs (prior_mean−truth) and vs FoM3 — prior-shrinkage = information effect. | NEAR-FINAL — discussion (L1 offset is shrinkage, not pathology). |
| `…/corner_resample/figs_multiobs/corner_patch{0,35,66,123,164}_perm1.pdf` | Per-patch posterior corners (obs-to-obs variation across sky patches). | NEAR-FINAL/appendix — per-patch robustness. |
| `…/corner_resample/figs_multiobs/variation_{cnn,l1}_auto_cross.pdf` | Posterior variation across patches, CNN vs L1. | NEAR-FINAL/appendix — robustness. |

---

## C. PLACEHOLDER — result figures from non-final / superseded / WRONG runs (swap when final)

Use these to fill paper figure slots now; replace after the flat-sky cross rebuild + the clean BNT-CNN
run. The underlying numbers are SUPERSEDED/WRONG (see `PAPER_FILE_TRIAGE.md`) — **layout only.**

### C1. Pillar-1 cross-only / cross-maps placeholders

| figure (prefer PDF) | what it shows | underlying status |
|---|---|---|
| `scripts/sbi/results/exploratory/cross_only_campaign_v2_chsigma/figures/corner_overlay_l1_vs_cnn_cross_only.pdf` | L1 vs CNN corner on cross-only (6 cross channels), 20°. | SUPERSEDED (20°, NDE-mismatch flagged) — good **L1-vs-CNN corner placeholder**. |
| `…/cross_only_campaign_v2_chsigma/figures/corner_{cnn_cross_only_plain_d10,cnn_cross_only_resnet50_gn_d10,l1_cross_only}.pdf` | Per-arm cross-only corners. | SUPERSEDED — placeholders. |
| `…/cross_only_campaign_v2_chsigma/figures/fom3_bar_chart.pdf`, `…/posterior_mean_scatter.pdf`, `…/channel_rms_bar.pdf`, `…/corner_overlay_3d_fom3.pdf` | FoM3 bar chart; posterior-mean scatter; per-channel RMS; 3-D FoM3 overlay. | SUPERSEDED — bar-chart/summary placeholders. |
| `scripts/sbi/results/exploratory/cross_maps_campaign/cross_summary/overlay_harm_vs_flat_vs_auto_nobnt.pdf` (+ `…_bnt.pdf`) | Harmonic-cross vs flat-sky-cross vs auto-only contour overlay. | WRONG numbers (v1 noise) — but a useful **"cross-map construction comparison" placeholder** for the cross-strategy section. |
| `…/cross_summary/overlay_harm_cross_vs_cnn_nobnt.pdf` (+ `…_bnt.pdf`) | Harmonic-cross L1 vs CNN overlay. | WRONG/SUPERSEDED — placeholder. |
| `…/cross_summary/datavectors_nobnt.png` (+ `…_bnt`, `…_pct1`) *(PNG only)* | L1 datavectors (auto vs cross), nobnt/bnt. | Illustrative-ish but tied to the broken-noise era — prefer the A1 `cross_maps_l1_datavectors.pdf`. |

### C2. Pillar-2 (BNT) placeholders — the "BNT inflation" figure

| figure *(PNG only)* | what it shows | underlying status |
|---|---|---|
| `scripts/sbi/bnt_tomo4_study/overlays/overlay_cnn_combined_bnt_vs_nobnt.png` | CNN BNT vs no-BNT contour overlay (inflation visualized). | WRONG numbers (pre-zmm) — but the **canonical "does BNT inflate?" placeholder** for Pillar-2. |
| `…/bnt_tomo4_study/overlays/overlay_l1_combined_bnt_vs_nobnt.png`, `…/overlay_l1vmim_combined_bnt_vs_nobnt.png` | Same for L1 and L1+VMIM (L1 inflates more). | WRONG numbers — Pillar-2 placeholders (the contrast IS the story). |
| `scripts/sbi/results/final/paper_sbi_consolidation/analysis/overlays/bnt_nobnt/overlay_*_combined_bnt_vs_nobnt.png` | Same family, consolidated tree (CNN/L1/L1vmim, per-seed + combined). | WRONG numbers — alternative Pillar-2 placeholders. |
| `scripts/sbi/results/final/paper_sbi_consolidation/analysis/overlays/nobnt_tomo_bins/*` | No-BNT single-bin → tomo4 CNN-vs-L1 overlays (the cross-correlation-gain story). | WRONG numbers (pre-zmm) — placeholder for the **G_corr / inter-bin-gain** figure. |

---

## Suggested paper figure plan (slot → file)

| paper slot | recommended file | kind |
|---|---|---|
| Fig. 1 — the data (auto + cross map channels) | `cross_maps/cross_maps_gallery.pdf` | KEEPER |
| Fig. 2 — the ℓ₁ wavelet statistic | `l1norm_diagnostics/raw_l1_per_scale_tomobin.png` (+ `l1_vs_cosmology.png`) | KEEPER |
| Fig. 3 — main constraining power (4 arms) | `figs/D1_constraining_power.pdf` | NEAR-FINAL (a+c provisional) |
| Fig. 4 — headline corner (CNN vs L1) | `figs/D4_corner_autocross.pdf` | NEAR-FINAL (provisional) |
| Fig. 5 — calibration (TARP-DRP) | `tarp_drp/figures/tarp_overlay_dim3.pdf` | NEAR-FINAL |
| Fig. 6 — SBC + L-C2ST | `figs/D5_sbc_ranks.pdf`, `figs/D6_lc2st.pdf` | NEAR-FINAL |
| Fig. 7 — w0 offset = flat-sky artifact | `figs/D2_w0_offset.pdf` | NEAR-FINAL |
| Fig. 8 — cross-map leakage (scales) | `figs/D9_cross_leakage_scales.pdf` + `full_sphere_cross_maps/fiducial/fullsphere_maps_nobnt.png` | KEEPER |
| Fig. 9 — cross-map construction comparison | `cross_summary/overlay_harm_vs_flat_vs_auto_nobnt.pdf` | PLACEHOLDER |
| Fig. 10 (Pillar-2) — does BNT inflate? | `bnt_tomo4_study/overlays/overlay_{cnn,l1,l1vmim}_combined_bnt_vs_nobnt.png` | PLACEHOLDER |
| Discussion — FoM3 fragility / degeneracy | `degeneracy_v2/corr_heatmap_tomo4_10deg80.png` | KEEPER |
| Discussion — sharpness≠calibration; shrinkage | `figs/D7_local_coverage_vs_latitude.pdf`, `figs/D8_shrinkage.pdf` | NEAR-FINAL |

> When the flat-sky cross rebuild (fiber `flatsky-cross-2026-06`) and the clean BNT-CNN run land,
> regenerate Figs. 3, 4, 9, 10 (and re-confirm the auto+cross panels of D1/D4). Everything tagged
> KEEPER stands. PNG-only keepers (full-sphere maps, ℓ₁ diagnostics, degeneracy heatmaps) may want a
> vector re-render via `/polish` for the final submission.
