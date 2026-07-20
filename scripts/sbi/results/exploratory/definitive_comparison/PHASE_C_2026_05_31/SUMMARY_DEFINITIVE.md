# Definitive L1 vs CNN — Phase C summary

## ⭐ CORRECTED HEADLINE — typical obs patch (full-200; median over ~300 patches [16,84])

**This supersedes the patch-0 tables below for the L1-vs-CNN headline.** The Phase-C perm-averaged tables used the fixed obs **patch-0 = the POLAR patch** (center lat 88.5°), atypically low-information for L1's near-polar wavelets (CNN is patch-insensitive) — that biased the original 'CNN ≳ L1 auto+cross'. Here each value is the median over the patch population (a typical 20 deg² obs). Read σ/2D, NOT FoM3 (it cubes ~20-25% diffs). Detail: fiducial_full200/SUMMARY_TYPICAL_PATCH.md + FIDUCIAL_FULL200_FINDINGS.md.

| arm | FoM3 | σ(Ωm) | σ(σ8) | σ(w0) | 2D(Ωm,σ8) |
|---|---|---|---|---|---|
| L1 auto+cross | 53069 [40215,71315] | 0.0231 [0.0197,0.0267] | 0.0367 [0.0339,0.0400] | 0.1247 [0.1156,0.1367] | 3343 [2632,4142] |
| CNN auto+cross | 24453 [20557,28306] | 0.0272 [0.0248,0.0314] | 0.0395 [0.0373,0.0427] | 0.1673 [0.1343,0.2017] | 2085 [1871,2336] |
| CNN auto+cross (std) | 22654 [19632,29171] | 0.0255 [0.0227,0.0297] | 0.0377 [0.0360,0.0400] | 0.1612 [0.1177,0.2028] | 2185 [1945,2547] |
| CNN auto+cross (MAF) | 16960 [14058,20941] | 0.0306 [0.0281,0.0341] | 0.0426 [0.0407,0.0455] | 0.1901 [0.1527,0.2285] | 1819 [1649,2056] |
| L1 auto-only | 11489 [7633,16131] | 0.0339 [0.0293,0.0385] | 0.0481 [0.0440,0.0532] | 0.1762 [0.1383,0.2066] | 1190 [912,1491] |
| CNN auto-only | 11233 [9757,13084] | 0.0301 [0.0270,0.0340] | 0.0424 [0.0411,0.0439] | 0.1749 [0.1469,0.1970] | 1255 [1154,1350] |

**auto+cross — L1 modestly ahead** (edge in w0 / cross-maps): σ(w0) L1 0.125 vs CNN 0.167 (×1.34); σ(Ωm) ×1.18; 2D(Ωm,σ8) ×1.60; FoM3 ×2.17 (FoM3 amplifies).
**auto-only — a tie**: σ(w0) L1 0.176 vs CNN 0.175; FoM3 11489 vs 11233.

Tight L1 posteriors verified calibrated (stratified varied-θ TARP). **Bottom line: L1 ≈ CNN auto+cross with a small L1 edge (w0/cross-maps); auto-only a tie; the original 'CNN ≳ L1' was a polar-patch artifact.**

---
## (historical) Perm-averaged analysis at the campaign obs — patch-0 = POLAR, SUPERSEDED for the headline

_Retained for the record. These tables condition on the fixed obs patch-0 (the polar patch); the corrected headline above uses the typical-patch population. The perm-averaging fixed a different earlier bug (perm-pooling) and is still correct as a perm-averaged-at-patch-0 view._

**Primary metric: 3-seed pooled FoM3 on (Ωm, σ8, w0), per perm then perm-averaged** (the constitution's declared metric). The across-perm spread (± std, %CV) is shown so FoM3's sensitivity is visible — see memory feedback_fom3_fragile_use_2d_areas. Marginal σ and 2D FoM are secondary.

`n_perms` makes the comparison transparent: L1 and the multi-perm CNN arms carry 3 fiducial realizations (perm-averaged); the other CNN arms carry only perm 0 (so their spread is '—'). Every row's per-perm unit is a 3-seed pool. CNN absolute FoM are fast-tf.data-route, treated as fine (overlap negligible, Andreas 2026-05-31).

Arms with posteriors: 10

## Headline — perm-matched L1 vs CNN (both 3 perms, perm-averaged)

The only apples-to-apples FoM3 comparison is L1 vs the CNN **multi-perm** arm (both averaged over the same 3 perms). The single-perm CNN rows below are perm-0 snapshots of the same NDEs.

- **auto+cross:** FoM3 — L1 25808 (±6968 (27%)) vs CNN 28093 (±3440 (12%)) → **CNN ahead** (within spreads). σ(w0) — L1 0.128 vs CNN 0.143 → **L1 tighter**.
- **auto-only:** FoM3 — L1 8774 (±1263 (14%)) vs CNN 9804 (±1521 (16%)) → **CNN ahead** (within spreads). σ(w0) — L1 0.183 vs CNN 0.199 → **L1 tighter**.

**The perm-0 'L1 ≥ CNN on auto+cross' headline does NOT survive perm-averaging** — it was a favorable perm-0 draw (L1 auto+cross FoM3 spread 27%; L1 led only on perm 0). On the matched 3-perm comparison CNN is nominally ahead on FoM3/2D and L1 retains only a modest, perm-fragile σ(w0) edge. **CAVEAT:** L1 uses the harmonic-cache route, CNN the tf.data route — a residual route confound (cf. the G8 section) is uncontrolled; a within-route run would settle it. See felt fiber definitive-l1-vs-cnn-2026-05/finding-perm-averaging-overturns-l1-lead.

## Primary — 3-seed pooled FoM3 (perm-averaged; higher = tighter)

| arm | n_perms | FoM3 | across-perm spread |
|---|---|---|---|
| L1 auto+cross | 3 | 25808 | ±6968 (27%) |
| L1 auto-only | 3 | 8774 | ±1263 (14%) |
| CNN-RealNVP auto+cross | 1 | 26748 | — |
| CNN-RealNVP auto-only | 1 | 9125 | — |
| CNN-MAF auto+cross | 1 | 11984 | — |
| CNN-MAF auto-only | 1 | 6679 | — |
| CNN-RealNVP auto+cross (std) | 1 | 24281 | — |
| CNN-auto native-TFDS (RealNVP) | 1 | 14969 | — |
| CNN auto+cross multi-perm | 3 | 28093 | ±3440 (12%) |
| CNN auto-only multi-perm | 3 | 9804 | ±1521 (16%) |

## Secondary — marginal σ (lower = tighter; perm-averaged)

| arm | n_perms | σ(Ωm) | σ(σ8) | σ(w0) | σ(h0) | σ(ns) | σ(Ωb) |
|---|---|---|---|---|---|---|---|
| L1 auto+cross | 3 | 0.0296 | 0.0444 | 0.1284 | 0.0413 | 0.0421 | 0.0055 |
| L1 auto-only | 3 | 0.0386 | 0.0501 | 0.1830 | 0.0501 | 0.0495 | 0.0064 |
| CNN-RealNVP auto+cross | 1 | 0.0268 | 0.0378 | 0.1508 | 0.0416 | 0.0385 | 0.0076 |
| CNN-RealNVP auto-only | 1 | 0.0351 | 0.0420 | 0.2163 | 0.0512 | 0.0545 | 0.0078 |
| CNN-MAF auto+cross | 1 | 0.0346 | 0.0416 | 0.2126 | 0.0469 | 0.0521 | 0.0081 |
| CNN-MAF auto-only | 1 | 0.0433 | 0.0590 | 0.2165 | 0.0496 | 0.0546 | 0.0067 |
| CNN-RealNVP auto+cross (std) | 1 | 0.0257 | 0.0385 | 0.1446 | 0.0404 | 0.0356 | 0.0064 |
| CNN-auto native-TFDS (RealNVP) | 1 | 0.0298 | 0.0397 | 0.1484 | 0.0486 | 0.0499 | 0.0083 |
| CNN auto+cross multi-perm | 3 | 0.0250 | 0.0385 | 0.1429 | 0.0410 | 0.0356 | 0.0071 |
| CNN auto-only multi-perm | 3 | 0.0347 | 0.0422 | 0.1987 | 0.0505 | 0.0528 | 0.0078 |

## Secondary — 2D FoM (higher = tighter; perm-averaged)

| arm | Ωm–σ8 | Ωm–w0 | σ8–w0 |
|---|---|---|---|
| L1 auto+cross | 1931 | 296 | 179 |
| L1 auto-only | 983 | 189 | 118 |
| CNN-RealNVP auto+cross | 2190 | 358 | 197 |
| CNN-RealNVP auto-only | 1140 | 189 | 117 |
| CNN-MAF auto+cross | 1466 | 201 | 129 |
| CNN-MAF auto-only | 855 | 154 | 89 |
| CNN-RealNVP auto+cross (std) | 2187 | 372 | 200 |
| CNN-auto native-TFDS (RealNVP) | 1587 | 283 | 179 |
| CNN auto+cross multi-perm | 2316 | 415 | 210 |
| CNN auto-only multi-perm | 1171 | 211 | 131 |

## Patch-center confound (G8) — read before quoting any cross-map gain

The harmonic-cache route slices auto-only maps from full-sphere patches; the **native-TFDS** auto-only path does not. They are NOT equivalent baselines:

- native-TFDS auto-only: **FoM3 14969**, σ(w0) 0.148
- harmonic-sliced auto-only: **FoM3 9125**, σ(w0) 0.216

The harmonic auto-only baseline is **lossy** (FoM3 9125 ≪ 14969; σ(w0) 0.216 vs 0.148). Therefore the CNN auto+cross gain is **route-sensitive**:

- over the (lossy) harmonic auto-only:  26748 / 9125 = **2.93×**  ← inflated by a poor baseline
- over a FAIR (native-TFDS) auto-only:   26748 / 14969 = **1.79×**  ← the honest number

The *within-route* cross-channel effect is still valid (only the input channels differ), but its **magnitude must be quoted against the native-TFDS auto-only baseline (~1.8×), not the harmonic one (~2.9×).** See felt fiber definitive-l1-vs-cnn-2026-05/finding-patch-center-confound-g8.

## TARP coverage

Max |ECP−α| (deviation from the diagonal; lower = better calibrated), mean over 3 seeds. ≲0.10 = mild mis-calibration, none severe.

| arm (dump label) | 3-D | 6-D |
|---|---|---|
| cnn_auto_native_rnvp | 0.051 | 0.051 |
| cnn_autocross_maf | 0.060 | 0.094 |
| cnn_autocross_rnvp | 0.075 | 0.106 |
| cnn_autocross_rnvp_std | 0.077 | 0.102 |
| cnn_autoonly_maf | 0.059 | 0.072 |
| cnn_autoonly_rnvp | 0.057 | 0.077 |
| l1_autocross | 0.077 | 0.091 |
| l1_autoonly | 0.060 | 0.071 |

Figures: `tarp_2026_05_31/figures/tarp_{per_arm,overlay}_dim{3,6}.png`. The multi-perm arms reuse the SAME compressed cache + NDE seeds as the core RealNVP arms (the perm only selects the single obs map, which never enters TARP), so their coverage == cnn_autocross_rnvp / cnn_autoonly_rnvp — not re-dumped.

_Auto-generated by aggregate_all_arms.py — re-run to refresh as arms land._
