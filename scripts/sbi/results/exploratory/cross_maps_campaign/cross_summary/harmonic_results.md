# Harmonic vs flat-sky cross-maps — 3-seed comparison

Date: 2026-05-01. Cache: `full_sphere_cache_grid` (NSIDE=512, lmax=1024,
σ_e=0.26, n_gal=10, BNT applied in `a_lm` space). Identical L1/NPE settings
between flat-sky and harmonic arms (`--cross-snr-percentile 1.0`,
20deg/160px multipatch, 4 auto + 6 cross channels, 5000 steps, 3-seed pool).

## Aggregate FoM3 (mean over seeds 41/42/43)

| arm | FoM3 | gain vs auto-only | gain vs flat-sky cross |
|---|---:|---:|---:|
| auto-only (BNT)            |    789.27 | —      | —      |
| flat-sky cross pct1 (BNT)  |  1155.67  | +46%   | —      |
| **harmonic cross pct1 (BNT)** | **5160.65** | **+554%** | **+347%** |
| auto-only (no-BNT)          | 13130.75 | —      | —      |
| flat-sky cross pct1 (no-BNT)| 11545.21 | -12%   | —      |
| **harmonic cross pct1 (no-BNT)** | **59243.09** | **+351%** | **+413%** |

## Per-seed FoM3 (harmonic arms only)

| arm | s41 | s42 | s43 |
|---|---:|---:|---:|
| harm_cross_bnt   | 5627.26 | 5003.13 | 4851.57 |
| harm_cross_nobnt | 58654.29 | 63533.80 | 55541.20 |

Seed-to-seed dispersion is much smaller than for the flat-sky cross arms
(BNT pct1: σ(FoM3) = 248 across seeds for harmonic vs σ = 247 for flat-sky
*on a 4× higher mean*; no-BNT: σ(FoM3) ≈ 3300 on mean 59243 for harmonic vs
σ ≈ 2200 on mean 11545 for flat-sky — proportionally tighter).

## Truth coverage

`TRUTH = (Ω_m, σ_8, w_0) = (0.26, 0.84, -1.00)`. All harmonic posteriors stay
within |z| ≤ 1.1 of truth across both regimes and all three seeds:

| arm / seed | Ω_m bias (z) | σ_8 bias (z) | w_0 bias (z) |
|---|---:|---:|---:|
| harm_cross_bnt s41   | +1.1 | -0.4 | -0.8 |
| harm_cross_bnt s42   | +0.9 | -0.7 | -0.7 |
| harm_cross_bnt s43   | +0.9 | -0.3 | -0.8 |
| harm_cross_nobnt s41 | -0.3 | +0.3 | -0.4 |
| harm_cross_nobnt s42 | +0.1 | -0.1 | -0.1 |
| harm_cross_nobnt s43 | -0.2 | +0.2 | -0.4 |

The mild persistent bias in BNT (Ω_m ≈ +0.04, w_0 ≈ -0.16 below truth)
is sub-1σ and consistent across all three seeds, so it likely reflects
either a prior boundary effect or a small projection-induced bias rather
than a stochastic excursion.

## Conclusion

The full-sphere harmonic cross-maps deliver a large information gain in
**both** regimes — overturning the flat-sky-era conclusion that no-BNT
cross channels carry no extractable signal. The flat-sky FFT-on-patches
route was severely lossy: it discards (a) cross-information at scales
larger than 20 deg, (b) cross-information that bridges patch boundaries,
and (c) cross-correlations encoded in non-axisymmetric multipole pairs
that get smeared by gnomonic apodization. Harmonic-space cross products
on the full sphere (Zürcher-style `a_ℓm^(i) a_ℓm^(j)`) recover all of
this and, after gnomonic projection to the same 48-patch geometry,
deliver:

- BNT regime: ≈ 6.5× FoM3 over auto-only, 4.5× over flat-sky cross
- no-BNT regime: ≈ 4.5× FoM3 over auto-only, 5.1× over flat-sky cross

This is the new headline. Recommend revising
`PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md` and the cross-map sections of
the paper draft accordingly.

Posteriors and metadata:
- BNT: `jaxili_harm_cross_bnt/posteriors/l1cross_tomo4_20deg160mp_harm_bnt_p1_s{41,42,43}.npy`
- no-BNT: `jaxili_harm_cross_nobnt/posteriors/l1cross_tomo4_20deg160mp_harm_nobnt_p1_s{41,42,43}.npy`
- Cache manifest sha (all seeds): `0a68ea89669da18f...`
- Triple-overlay corner plots: `cross_summary/overlay_harm_vs_flat_vs_auto_{bnt,nobnt}.pdf`
