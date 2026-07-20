# Talk figure audit — punch-list against the best-practices standard

Per-figure assessment of `talk_figures/` against `TALK_BEST_PRACTICES.md` (§B/§C): **message**
(one clear point, assertion-ready), **fonts** (legible at projector scale, ≤3 sizes), **palette**
(global + colorblind-safe), **crop/clutter**. Verdicts: **KEEP** (talk-ready) / **MINOR** (small
restyle) / **REWORK** (needs real work) / **DROP**.

Reflects Andreas's calls (2026-06-14): full-sphere leakage maps **dropped**; `p2_M3_corner_l1_bnt`
he'll **redo himself** (paths below); `p2_methods_l1_vs_cosmology` **trimmed** (extreme SNR bins
removed — done); `p2_M2_cross_deleaking` **no-full-sphere variant** added.

---

## 0. The global color language — LOCKED (2026-06-14)

Adopted the **Wong colorblind-safe palette** already used by the stitched figure Andreas likes
(better than the generic red first proposed). The convention, applied to every figure:

- **Method = COLOR, always:** **CNN = Wong blue `#0072B2`**, **L1 = Wong vermillion `#D55E00`**.
  Color never does double duty — blue always means CNN, never "no-BNT" (that was the old confusion).
- **Method = secondary encoding for grayscale/colorblind:** corners → **L1 solid / CNN dashed**;
  method-only bars → L1 hatched / CNN solid.
- **Basis = TEXTURE (when a no-BNT/BNT/whitened axis exists):** **no-BNT = solid, BNT = hatched**
  (same color, paler), whitened = mid. Demonstrated in `_new_figs/bnt_palette_example`.
- **Slide fonts:** axis labels ~24 pt, ticks ~15 pt (vs paper's 15/11) — `g.settings.scaling=False`
  in getdist so the sizes stick.

Reference implementations (approved style): `_new_figs/make_stitched_slides_example.py` (corner+bars),
`_new_figs/make_bnt_palette_example.py` (method×basis), `make_stitched_final_resnet18.py` (the real
M1 figure with current data + 3-bar inset).

Was: figures mixed **red/blue (method)** with **blue/orange (basis)**, so blue meant both CNN and
no-BNT. The Wong pair + texture-for-basis removes the collision.

---

## 1. Part 2 (this repo) — result figures

| figure | role | verdict | fixes |
|---|---|---|---|
| `p2_M1_fom3_distribution` | S13 money figure | **MINOR** | Already strong (clear title, red=L1/blue=CNN, violins). Enlarge axis/tick fonts for projector; the title carries the assertion ("CNN tighter at 81% of patches, 1.16×") — keep. Confirm red/blue match the global palette. |
| `p2_M1_corner_cnn_vs_l1` | S14 corner | **REWORK** | (a) **Legend FoM3 shows fiducial 3355/2673** — relabel to the population median (3326/2875) or drop the number from the legend to avoid the two-ratio confusion (content §4 flag). (b) Enlarge axis labels (Ω_m, σ8, w0) and tick fonts a lot. (c) Consider a progressive reveal (ℓ1 first, then CNN) so "they nearly coincide" lands. (d) Verify red=L1/blue=CNN global. |
| `p2_reliability_tarp_sbc` | S16 calibration | **MINOR→REWORK** | Four-panel (TARP + 3 SBC). At projector scale the SBC panels + the long title will be small. Enlarge fonts; shorten the title (the slide's assertion headline carries the message); ensure the 99% band + diagonal are thick enough to read from the back. |
| `p2_M2_cross_deleaking` | S17 (full version, w/ leakage) | **KEEP** | New, already follows the standard. Tiny: the +20% arrowhead grazes the "1.20×" label — optional nudge. |
| `p2_M2_cross_deleaking_nofullsphere` | S17 alt (no leakage bar) | **KEEP** | New variant per request; same styling, ylim trimmed. |
| `p2_M3_bnt_inflation` | S18 payoff | **MINOR** | Excellent and clear (0.15×/0.22× collapse vs 0.93×/0.87×). Recolor to the global scheme: bars are method×basis (L1/CNN × noBNT/BNT) — use red/blue for method, shade for basis, instead of blue/orange. Enlarge ratio labels. |
| `p2_M3_bnt_whitening` | S19 mechanism | **MINOR** | Strong. Five-color basis legend is a lot; once the global palette is set, reduce to shade-coded bars + keep the "recovered 1.06/1.01" callouts (those are the message). Enlarge fonts. |
| `p2_M3_corner_l1_bnt` | S18/19 backup contour | **REDO (Andreas)** | Andreas: BNT looks over-biased at this obs. He'll regenerate (paths in §3). |
| `p2_M3_corner_cnn_bnt` | backup contour | **MINOR** | Same generator/palette as the L1 one (blue=noBNT/orange=BNT). If kept as backup, align palette; lower priority. |
| `p2_methods_flatsky_inputs` | S11 methods | **MINOR** | Good "what the compressor sees" panel (4 autos + conv + product). For a slide, the per-panel titles are tiny — enlarge, or crop to fewer representative channels. Keep the diverging map colormap (fine; it's data, not the method axis). |
| `p2_methods_l1_vs_cosmology` | S12 / paper | **REWORK→done(trim)** | Trimmed version (extreme SNR bins removed) generated for the paper. For *slides* it may be too detailed (Andreas's call) — if used, it needs far bigger fonts and probably fewer panels (it's 3×5). Paper: keep full grid, trimmed. |

## 2. Part 1 (baryonic-feedback paper) — reused figures

These are already publication-styled (vector PDFs from the submitted paper). Main risk is
**projector legibility** (paper fonts are small on a big screen) and **colorblind/линestyle** checks.
None need scientific rework — they're done figures.

| figure | role | verdict | fixes |
|---|---|---|---|
| `p1_bias_vs_survey_area` | S5 headline | **MINOR** | Verified clean vector. Enlarge axis/legend fonts for projector; the three lines (PS / two HOS) should differ by **linestyle+marker**, not color alone (colorblind). |
| `p1_PSvsHOS_safe_scales` | S7 headline | **MINOR** | PNG-only; PS/peaks/ℓ1 contours. Check it's high-res enough for projection; ensure the three are distinguishable without color. |
| `p1_BRIDGE_bnt_inflates_l1` | S9/S21 hinge | **MINOR** | Three contours (gray=BNT, red=safe, blue=all). Gray-vs-color reads well. Enlarge corner axis fonts; this is a load-bearing slide — make the BNT-gray contour unmistakably the largest. |
| `p1_baryon_impact_ps`, `p1_baryon_impact_l1` | S3/S6 | **MINOR** | Fractional-difference panels; enlarge fonts; fine otherwise. |
| `p1_bnt_kernels` | S8 | **MINOR** | Kernel before/after. Enlarge; ensure the "nulling" is visually obvious. |
| `p1_methods_tomo_maps`, `p1_setup_nz_bins`, `p1_bnt_noisy_maps`, `p1_l1_constraints_vs_area` | setup/support | **KEEP/MINOR** | Standard setup figures; projector-font check only. |

## 3. `p2_M3_corner_l1_bnt` — paths for Andreas to redo it himself

- **Figure shown (the one to replace):**
  `scripts/sbi/results/exploratory/flatsky_cross_2026_06/bnt_campaign/figures/corner_bnt_vs_nobnt_l1_none.pdf`
  (copied into `talk_figures/p2_M3_corner_l1_bnt.pdf`).
- **Generator (CPU, getdist, ~instant):** `scripts/sbi/bnt_corner_overlays.py`. It overlays two saved
  sample sets at the **same obs**, currently the **`"typical"` key (perm16 / patch23)**:
  - no-BNT: `…/representative_corner/flat_none/corner_samples.npz`
  - BNT: `…/bnt_campaign/representative_corner/l1_none/corner_samples.npz`
  - colors `C_NOBNT="#0072B2"`, `C_BNT="#D55E00"` (line 26).
- **Two ways to get a less-biased contour:**
  1. **Instant, no GPU:** both npz already contain a second obs, **`"favorable"` (perm0 / patch90)**.
     Change `["typical"]` → `["favorable"]` on lines 41–42 of `bnt_corner_overlays.py` and re-run.
     *Caveat:* perm0/patch90 was historically the favorable draw — it may look better-centered but
     could understate the true BNT inflation; check it's representative, not cherry-picked.
  2. **Proper, GPU (pin to GPU 1):** add your chosen `(perm, patch, "label")` to `OBS` in
     `scripts/sbi/representative_corner_flatsky.py` (line 10, currently
     `[(16,23,"typical"),(0,90,"favorable")]`), regenerate the **no-BNT** samples with that script,
     regenerate the **BNT** samples at the same obs (via the BNT campaign / `run_bnt_gate_c.py`
     representative-corner path), then plot with `bnt_corner_overlays.py`. This lets you pick an obs
     whose no-BNT posterior sits on truth so the BNT *inflation* (not a draw offset) is what shows.
- Note: the scientific message is the **width inflation**, not the centroid of any single draw — a
  representative (not favorable) obs with the no-BNT contour on truth makes that cleanest.

---

## Priority order for the rework pass
1. **Lock the global palette** (§0) — unblocks everything else.
2. `p2_M1_corner_cnn_vs_l1` — relabel FoM3 + big fonts (it's the headline corner).
3. `p2_M3_bnt_inflation` + `p2_M3_bnt_whitening` — recolor to the palette, big ratio labels.
4. `p2_reliability_tarp_sbc` — enlarge / shorten title.
5. `p2_M1_fom3_distribution`, `p2_methods_flatsky_inputs` — font bumps.
6. Part-1 figures — projector-font + colorblind-linestyle pass (low effort, they're done).
