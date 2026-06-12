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

## Why the convolution operator adds so little — theory account (2026-06-12, paper-bound)

Three stacked reasons, ordered by how much they explain; the literature reconciliation
(Zürcher et al. 2023, MNRAS 525, 761; arXiv:2206.01450 — re-read 2026-06-12) is part 3.

1. **The conv map's one-point content is two-point information.** The flat-sky conv map is
   `ifft(fft(κᵢ)·fft(κⱼ))`: the pixel at lag **p** is Σₓ κᵢ(**x**)κⱼ(**p**−**x**) — a sum
   over the whole patch. Up to a reflection, the conv map IS the lag-space empirical
   cross-correlation map: pixel p ≈ N_pix·ξ̂ᵢⱼ(**p**), so its spatial structure is the
   *shape of ξᵢⱼ* — a Gaussian-sector quantity. One-point statistics (l1, PDF) of this map
   re-encode two-point information (plus estimator noise; trispectrum enters only through
   the estimator covariance). The pointwise product is the local complement: its one-point
   moments are the joint moments ⟨κᵢⁿκⱼⁿ⟩ at zero lag — genuinely non-Gaussian cross
   information. This asymmetry is the measured +4% (conv) vs +20% (product), and is
   consistent with pair2d ≈ l1+product (the product's information is joint one-point
   occupancy; `overnight_menu/OVERNIGHT_RESULT.md`).
2. **CLT compression.** Each conv pixel is a patch-wide sum of ~N_pix products → the
   pixel distribution is strongly Gaussianized and pixels are highly correlated across
   lags; the effective dof ≈ the number of independent cross-spectrum bands the patch
   resolves — a handful at 10°. The autos already pin the Gaussian sector, so the marginal
   value is small. Hence also the conv arm's seed-fragility (few effective numbers).
3. **Reconciliation with Zürcher et al. (2023).** (a) *Footprint*: their statistics are
   computed on contiguous 5,000 deg² (stage 3) / 14,300 deg² (stage 4) footprints — the
   large-scale coherent cross modes that carry conv-type information exist in their data
   vector; our 100 deg² patches excise them, and the full-sphere construction that does
   access them is, for a patch-based forecast, leakage (measured: ≈99% of this operator's
   full-sphere gain). The two results are consistent: conv-style cross information lives
   at scales a 10° patch barely samples. (b) *What their cross-bins actually buy* (their
   §5.5 + Table 3): the gain is dominated by **galaxy-IA self-calibration** — removing
   cross-bins degrades σ(A_IA) by ×2–×5.3 (peaks −104%, C_ℓ −430%) and the cosmology hit
   (FoM −41% to −47%) comes largely through the A_IA–S8 degeneracy. **Our forecast has no
   IA**, so the dominant literature channel for cross-map gains is absent here by
   construction. (c) *Definitional note*: their Eq. 12 builds cross-alms as √âᵢ·√âⱼ
   (geometric mean, units of κ); our harmonic route used the plain product âᵢâⱼ and the
   flat-sky conv is its Fourier-product analog. Same global-support family — the locality
   and CLT arguments apply to both variants — but they are not identical operators.

**Corollary for the BNT basis** (why conv channels can't rescue the BNT-l1 collapse
either): the complete per-scale second-moment sector (cov50: all 10 channel-pair
covariances × 5 scales), appended to BNT-l1, recovers only **0.38** of the loss (overnight
A1) — 62% of what per-channel analysis loses under nulling is non-Gaussian. The 2-pt
sector is exactly frame-invariant (deep-dive P7), so nothing two-point is *more* accessible
in either basis; conv maps are a CLT-compressed lag-space encoding of a subset of that same
sector ⇒ their rescue is bounded by 0.38 and realizes less. Measured indication: the BNT
`both` arm (conv+product) lands at FoM3 751 (single run, never population-swept) vs
product-only 637 pooled — same collapsed ballpark, no rescue.

Registered-not-run corollary test: conv gain should grow with patch size (20° dataset
exists: `grid_20deg_160px`) while the product gain stays ~stable.

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
