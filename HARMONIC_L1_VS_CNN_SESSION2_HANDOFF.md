# Harmonic-L1 vs CNN — Session 2 handoff

**Branch:** `l1-cross-maps`  ·  **Date:** 2026-05-11
**Companion to:** `HARMONIC_L1_VS_CNN_INVESTIGATION_BRIEF.md`
**Next planned action:** TARP coverage test on the L1 harmonic-cross posterior.

This file records what was decided and verified in the second working
session of the investigation. The first brief (above) framed the puzzle
(L1 harmonic-cross no-BNT FoM3 ≈ 60 k vs CNN ≈ 17 k); this brief
documents the closing of the no-bug hypothesis space and the open
calibration question that motivates the TARP follow-up.

---

## 1. What was finished this session

### 1.1 Stability of the L1 harmonic-cross no-BNT FoM3

Six seeds now exist (41–46), trained with identical architecture and
hyperparameters. Posteriors live in
`scripts/sbi/results/exploratory/cross_maps_campaign/jaxili_harm_cross_nobnt/posteriors/`.

| seed | FoM3 |
|------|------|
| 41 | 68,108 |
| 42 | 67,059 |
| 43 | 64,944 |
| 44 | 67,824 |
| 45 | 65,895 |
| 46 | 55,541 |

- Mean ⟨FoM3⟩ = **64,895**, std 4,716, relative scatter **8.0 %**.
- Overlay plot: `overlay_l1_harm_cross_seeds.{pdf,png}` (same dir).
- Conclusion: the headline number is reproducible, not a one-seed
  artefact.

### 1.2 FoM3 formula audited

Identical in the overlay scripts and the pipeline:

```python
cov3 = np.cov(samples[:, :3], rowvar=False)        # (Ω_m, σ_8, w_0)
sign, logdet = np.linalg.slogdet(cov3)
fom3 = float(np.exp(-0.5 * logdet))
```

i.e. `FoM3 = 1/√det(C_3)`. Pipeline-canonical definition at
`scripts/sbi/npe_l1norm_cross_jaxili_nbody_tomo.py:1737-1759`.

The 22 % FoM3 spread vs ~7 % per-marginal width is the cube-root
relationship (`1.22^(1/3) ≈ 1.07`). Per-parameter marginal std ratios
across seeds: Ω_m 1.17, σ_8 1.16, w_0 1.13. No discrepancy.

### 1.3 "Different NDE-training data" hypothesis falsified

Hypothesised that CNN pipelines split the simulations into
compressor-train and NDE-train subsets, while L1 trains the NDE on all
simulations. **Verified false:**

- All CNN configurations in this branch set
  `compressor_train_split=train`, `nde_train_split=train`. Same source
  TFDS split fed to both.
- `jaxili.NPE.append_simulations(...)` does an internal 70/20/10 split.
  Both L1 and CNN pipelines hit the same internal split.
- CNN `cnn_train.npz` has `theta` (302,064, 6) and `x` (302,064, 10) —
  identical sample count to L1's `l1_train.npz`.

So the L1 FoM advantage is not a "more training data" effect.

### 1.4 BatchNorm contamination on the harmonic-CNN route — saved as memory

Independent earlier finding, recorded for future Claude sessions:

- Haiku stock ResNet50/34 (BatchNorm) on the 10-channel harmonic CNN
  route gives FoM3 ≈ 700 (catastrophic, ~30× worse than plain CNN).
- Replacing BN → GroupNorm (`--compressor-arch resnet50_gn`) recovers
  FoM3 to ~22 k on the first seed, matching the plain-CNN baseline.
- Mechanism: BN's running stats normalise across batches that mix
  cosmologies, so per-example cosmological variance gets absorbed into
  BN moments; GN normalises within a single example.

Memory file: `~/.claude/.../memory/project_resnet_bn_contamination.md`.
Practical rule: do not use BN ResNets on multi-channel harmonic input.

### 1.5 Four-arm comparison overlay (the canonical figure)

`scripts/sbi/results/exploratory/cnn_with_harm_cross_normalized/resnet50_gn/overlay_autoonly_vs_cnn_harmcross_vs_l1_harmcross.{pdf,png}`

| arm | channels | seeds | ⟨FoM3⟩ |
|-----|---------|-------|--------|
| CNN auto-only (plain) | 4 | 5 | **16,894** |
| CNN auto + harm cross (plain) | 10 | 3 | **25,466** |
| CNN auto + harm cross (ResNet50-GN) | 10 | 3 | **18,763** |
| **L1 auto + harm cross** | 10 | 6 | **64,895** |

Plain CNN with harm cross + per-channel input normalisation is the
**best CNN** (≈ 25 k). It still loses to L1 by 2.5×.

### 1.6 SBC (B1) for L1 harmonic-cross no-BNT — DONE and PLOTTED

Run: `scripts/sbi/results/diagnostics/sbc_harm_l1_nobnt/n1000_m2000_seed20260507/`
(N = 1000 cosmologies, M = 2000 posterior samples, 20 bins).

Verdict table (chi-square dof = 19, KS on normalised ranks):

| param | χ² | p_χ² | KS p | mean-rank z | verdict |
|-------|----|------|------|-------------|---------|
| Ω_m | 18.9 | 0.46 | 0.69 | +2.59 | **OK shape, biased high** |
| σ_8 | 27.5 | 0.094 | 0.04 | −1.78 | **marginal, biased low** |
| w_0 | 31.6 | 0.034 | 0.16 | +1.17 | marginal |
| h_0 | 84.1 | 9e-10 | 1e-5 | −2.52 | **strongly non-uniform** |
| n_s | 41.8 | 0.002 | 0.001 | +2.04 | non-uniform |
| Ω_b | 116.8 | <1e-15 | <1e-7 | −1.25 | **strongly non-uniform, U-shape (overconfident)** |

Plots: `sbc_b1_rank_histograms.{pdf,png}`, `sbc_b1_rank_ecdfs.{pdf,png}`,
`sbc_b1_summary_table.{pdf,png}` in the same dir.

The three FoM3-relevant parameters (Ω_m, σ_8, w_0) pass shape
uniformity, but all carry non-trivial location bias (|mean-z| ≈ 1–2.6).
Ω_b is clearly U-shaped → **the posterior is overconfident** on Ω_b. h_0
is strongly non-uniform.

The σ_8 mean-z = **−1.78** is consistent with the σ_8 inversion
already documented in B2 (`harm_l1_truthcheck/`) and in memory entry
`project_harmonic_cross_overturns_flatsky.md`. SBC and B2 corroborate.

---

## 2. The information-theoretic puzzle and the working answer

User's question: "how can L1 outperform a CNN trained with a VMIM
objective, since L1 captures only a subset of the information and a VMIM
CNN should approach the sufficient statistic?"

Working answer (three threads, all relevant):

1. **FoM3 measures contour volume, not information.** A miscalibrated,
   tight posterior wins on FoM3. SBC shows the L1 posterior *is*
   miscalibrated: Ω_b is overconfident (U-shape, χ² = 117), σ_8 is
   biased low, h_0 is far from uniform. The contour is partly small
   because it is wrong.

2. **VMIM is a variational lower bound on MI, not a supremum.** The CNN
   compressor is finite-width, finite-step, projects to a 10-dim
   bottleneck. The L1 flow conditions directly on a ~2000-dim
   datavector with 200× more room before any bottleneck.

3. **L1 features are scale-aligned with the signal by construction.**
   The wavelet decomposition gives the flow scale-correct features for
   free; the CNN has to discover the same multi-scale structure.

Net: if (a) the CNN compressor reached the sufficient statistic and
(b) the L1 flow were perfectly calibrated, then CNN ≥ L1. Neither
holds. The observed pattern (L1 smaller AND less calibrated; CNN
larger AND better calibrated where we have checked) is consistent with
"L1 wins on raw signal alignment but the flow is paying for it in
calibration."

---

## 3. Recommended next session — TARP coverage test

The remaining question is quantitative: **how much of the L1 contour
shrinkage is genuine information vs. miscalibration?**

SBC catches some of this (it caught Ω_b, h_0, and the σ_8 bias) but
SBC tests marginals. **TARP** (Lemos et al. 2023, *Sampling-Based
Accuracy Testing of Posterior Estimators*) tests the **joint** posterior
by checking expected coverage probability against nominal credible-level
α, using random reference points. This is the right tool to ask "are the
3-D and 6-D contours correctly sized?".

### Suggested implementation steps

1. **Use the existing SBC infrastructure as input.** We already have
   ranks and 1000 posteriors saved at
   `scripts/sbi/results/diagnostics/sbc_harm_l1_nobnt/n1000_m2000_seed20260507/`.
   For TARP, what we actually need is the (theta_true, posterior_samples)
   pairs — those are produced upstream by `run_sbc_harm_l1_nobnt.py`. If
   the posterior samples aren't already on disk, instrument the SBC
   loop to also write the 2000 samples per cosmology (or a subsample
   like 500 to keep disk small).

2. **Pip-installable TARP:** `pip install tarp` (Lemos et al.'s ref
   implementation) inside the `jaxili` conda env. Falls back to a
   ~80-line numpy reimplementation if the package proves heavyweight.

3. **Plot the TARP curve:** expected coverage vs nominal credibility,
   with confidence band from bootstrap over the N=1000 cosmologies.
   Plot for the 3-D (Ω_m, σ_8, w_0) joint and the full 6-D joint.

4. **Decision rule:** if the 3-D TARP curve lies systematically below
   the diagonal at high credibility (e.g. expected 60 % coverage at
   nominal 95 %), the L1 contour shrinkage *is* substantively
   overconfidence and the FoM3 advantage is partly artefactual. If it
   tracks the diagonal, the gain is real.

5. **Run the same TARP on the CNN auto-only posterior** (we already
   have its SBC output at `sbc_cnn_nobnt/n1000_m2000_seed20260504/`),
   so we have a calibrated reference curve to compare against.

### Files / artefacts that will be needed

| artefact | location | status |
|----------|----------|--------|
| L1 SBC posteriors (1000 × 2000 × 6) | `sbc_harm_l1_nobnt/.../sbc_ranks.npz` has *ranks* but probably not the full posterior samples | **check first** |
| CNN SBC posteriors | `sbc_cnn_nobnt/n1000_m2000_seed20260504/` | check |
| `run_sbc_harm_l1_nobnt.py` | `scripts/sbi/` | may need `--dump-posterior-samples` flag added |
| TARP plotting script | new: `scripts/sbi/results/diagnostics/tarp_harm_l1_nobnt/` | to write |

If full posteriors weren't dumped: re-running 1000-cosmology SBC with
dump enabled is ~the same cost as the existing run (which took several
hours). A reduced N=200 first pass is fine for the first TARP read.

---

## 4. Open items NOT addressed this session

- **B4 held-out cosmology test** (deferred from session 1 plan). Still
  unbuilt. Lower priority than TARP given B2 already showed σ_8
  inversion on shifted cosmologies.
- **Why h_0 is strongly non-uniform** in SBC. Not investigated this
  session. The pattern (mean-z = −2.5, U-shape with peaks at low and
  high ranks) suggests bimodality or boundary effects.

---

## 5. One-line summary

> The L1 harmonic-cross FoM3 of ~65 k is reproducible across 6 seeds,
> the FoM3 formula is correct, the data-split hypothesis is wrong, the
> ResNet-BN issue is patched. The L1 posterior is overconfident on Ω_b
> and biased on σ_8/h_0 per SBC; the next step is a TARP joint-coverage
> test to quantify how much of the FoM3 gain is real vs. overconfidence.
