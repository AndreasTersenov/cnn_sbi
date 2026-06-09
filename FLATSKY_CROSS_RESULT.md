# Flat-sky (patch-local) cross-maps — definitive result (2026-06-09)

**The physically-defensible auto+cross result for the paper.** Replaces the leaky full-sphere
harmonic cross (every cross-patch a global functional of the whole sky; `CROSS_MAP_LEAKAGE_FINDING.md`)
with patch-local flat-sky cross-maps, recomputes L1, trains, calibrates, and measures constraining
power. Calibrated (TARP+SBC). Method matches the full-sphere `SUMMARY_PHASE_D` (pooled 3-seed,
9000-obs median) so the comparison is apples-to-apples.

## Headline

**~92% of the full-sphere L1 auto+cross gain was LEAKAGE.** The physically-buildable patch-local
cross retains only **+21% FoM3** (1.21×) vs the full-sphere's **+288%** (3.88×).

Operator-resolved: the full-sphere harmonic cross = the alm-product, whose flat-sky analog is the
**convolution** — de-leaked it gives **+4%** (≈99% of *that* operator's gain was leakage). The
surviving physical signal is the **pointwise product** κᵢ·κⱼ (mean = ξᵢⱼ): **+20%**, tightening
Ωm/σ8 ~9%. Leakage lived mostly in **w0** (full-sphere drove σ(w0) 0.246→0.188; patch-local only
0.245→0.232). On σ(Ωm) the physical `both` (0.046) matches the full-sphere (0.046) — the Ωm gain
was largely real.

## Table (pooled 3-seed, 9000-obs median; calibrated TARP+SBC)

| arm | FoM3 | vs auto | σ(Ωm) | σ(σ8) | σ(w0) | 2D(Ωm,σ8) |
|---|---|---|---|---|---|---|
| flat-local auto-only | 2405 | 1.00× | 0.053 | 0.082 | 0.245 | 471 |
| flat-local +conv | 2499 | 1.04× | 0.052 | 0.081 | 0.245 | 484 |
| flat-local +product | 2875 | 1.20× | 0.048 | 0.075 | 0.238 | 522 |
| flat-local +both | 2910 | 1.21× | 0.046 | 0.075 | 0.232 | 528 |
| full-sphere auto-only | 2200 | 1.00× | 0.056 | 0.085 | 0.246 | 441 |
| full-sphere auto+cross (LEAKY) | 8530 | 3.88× | 0.046 | 0.072 | 0.188 | 1045 |

Auto-only consistency (flat-local 2405 ≈ full-sphere 2200) validates the comparison — autos are local.

## Operator ranking flips single-obs → population

Single-obs (perm0/patch90, per-seed-mean) gave conv +32% > product +17%. The **pooled 9000-obs
median overturns this: product +20% ≈ both +21% ≫ conv +4%.** Two reasons the single-obs was
inflated: (1) **aggregation** — pooling 3 seeds (the prior-comparable metric) applies the "pool
haircut" and hits the seed-fragile conv hardest; (2) a mildly favorable patch. The convolution
(smooth/large-scale Zürcher analog) is sample-variance-limited and seed-fragile at 10°; the pointwise
product (local, scale-preserving, = ξᵢⱼ) carries the robust cross-info. **Lead with the population
median, not single-obs.**

## Calibration (GATE C)

- **TARP-DRP ✓** — all arms incl. the tight HIGH-FoM3 tercile on the diagonal (calibrated/mildly
  conservative, never over-confident). `gate_c/tarp_drp/`.
- **SBC ✓** — flat ranks within the 99% binomial band, mean ranks ≈0.5 (no bias), per-seed KS p>0.05.
  `gate_c/sbc/`.
- **L-C2ST N/A** — underpowered at high-dim L1 (self-test ST_H1 fails; logreg can't resolve local
  miscalib on 800-3200-dim x). Andreas accepted TARP+SBC. See `reference_lc2st_underpowered_highdim_l1`.

## Reproduce

- Operators: `flatsky_cross.py` (np/torch/jax conv+product). Frozen per-(channel,scale) noise σ:
  `freeze_flatsky_cross_noise.py` → `flatsky_cross_noise_sigma.npz`.
- L1: `npe_l1norm_cross_jaxili_nbody_tomo.py --cross-maps-route flat_local --cross-op {none,conv,product,both}`
  (+ build-both-once-slice via `--flatsky-both-cache`). Matrix: `run_flatsky_l1_matrix.py`.
- GATE C: `run_flatsky_gate_c_tarp.py`, `compute_sbc_from_tarp_dumps.py`.
- Population sweep: `run_flatsky_population_sweep.py` → `population_sweep/<arm>/median_summary.json`.
- Diagnostics/corners: `plot_flatsky_diagnostics.py`, `plot_l1_matrix_corners.py`, `compute_l1_2d_areas.py`.

## Open / next

- CNN arms (jax flat cross + per-channel RMS) — the L1-vs-CNN comparison on the de-leaked cross.
- (backlog) fixed-[-4,4] binning robustness; scale-matched product; `both` high-dim NDE (L1-VMIM?).
