# Project Scientific Knowledge Base

## Scope and intent

This document consolidates the scientific and technical knowledge generated across the full `cnn_sbi` project history (L1 statistics, L1-VMIM compression, CNN compression, BNT/no-BNT studies, and follow-up campaigns), with quantitative outcomes, interpretation, and publication-oriented synthesis.

Primary scientific target:

- understand and reduce contour inflation when using BNT-transformed tomographic maps;
- compare handcrafted higher-order summaries (starlet L1) against learned neural summaries (CNN/VMIM);
- evaluate whether learned tomographic-channel summaries can recover cross-bin information and preserve constraints under BNT.

---

## Common experimental setup (reused across major campaigns)

- Cosmological parameter order used by SBI scripts: `[Omega_m, sigma_8, w0, h0, n_s, Omega_b]` with `H0 -> h0` rescaling (`/100`) in preprocessing.
- Core tomographic setup for final runs: 4 tomographic bins (`1,2,3,4`) and `tomo4_20deg160`.
- L1 default final settings in consolidation runs: `n_scales=5`, `l1_nbins=40`, `SNR=[-13,13]`, no PCA by default in orchestrators.
- BNT path in CNN pipeline applies shape noise first, then BNT transform.

---

## 1) L1 pipeline diagnosis and stabilization

### 1.1 Why older “banana” L1 contours were not reproducible

From the dedicated investigation log:

- legacy L1 artifact had strong `(Omega_m, sigma_8)` correlation (`-0.714`) and axis ratio `~3.08`;
- rerun artifact was near-circular (correlation `-0.013`, axis ratio `~1.41`);
- targeted ablations (PCA on/off, larger flow, `n_scales=6`, coarse-mean toggles) did **not** restore banana-like shape;
- batch-vs-single L1 feature computation agreed numerically (`max abs diff ~1.17e-4`);
- old W&B metadata and checkpoint/preprocessing history showed mixed, likely non-reproducible artifact lineage.

Interpretation: old “good-looking” L1 contours were not a clean baseline from a single, versioned configuration; modern reproducible path yields weaker L1 tilt in this setup.

### 1.2 L1 fixes that were implemented and validated

Implemented in `npe_l1norm_nbody_tomo.py`:

- explicit SNR default policy and calibration control (`--auto-calibrate-snr`);
- explicit histogram overflow behavior (`--l1-clamp-overflow`);
- robust coarse-mean toggles (`--subtract-coarse-mean`, `--no-subtract-coarse-mean`);
- stronger cache metadata validation;
- feature-health diagnostics (dead features, inlier fractions, clipping fractions).

Validation status:

- syntax, CLI, no-train, train, and posterior-sampling flows all executed successfully;
- short retraining showed improving validation loss and valid posterior output.

---

## 2) L1-VMIM conclusions

Best calibrated near-lossless L1-VMIM run reported:

- `std_ratio = 1.019` (vs no-compression L1 baseline);
- `L2(mean-truth) = 0.0642`;
- Mahalanobis `= 1.303`;
- selected artifact: `l1_vmim_tomo4_20deg160_seed202_flowonly.npy`.

Interpretation:

- major improvement came from compressor/preprocessing robustness plus flow training variability;
- tighter-than-baseline posteriors were achievable but often biased, so calibration-constrained selection was necessary.

---

## 3) Final consolidated matrix (no-BNT, BNT, baryon appendix)

### 3.1 QC pass of final consolidated campaign

From `FINAL_SCIENTIFIC_REPORT.md`:

- no-BNT matrix: 55/55 jobs successful;
- BNT tomo4 comparison: 22/22 successful;
- baryonified appendix: 180/180 successful;
- posterior summary checks showed no empty/non-finite pathologies in consolidated inventories.

### 3.2 No-BNT FoM comparison (mean FoM3)

> ⚠ **Superseded in 2026-04-22 by §13.** The CNN FoM3 numbers in this table are inflated by the mass-sheet-degeneracy information leak identified in the zero-mean-maps parity check. After demeaning (see §13), the CNN advantage over L1 in tomo4 no-BNT largely disappears: CNN FoM3 ≈ 1.5e4 at seeds 41-43 vs L1 FoM3 ≈ 1.1e4.

| Method | bin1 | bin4 | tomo4 | tomo4 / best single-bin |
|---|---:|---:|---:|---:|
| CNN | 6,001.0 | 52,767.9 | 387,474.6 | 7.34x |
| L1 | 568.1 | 4,641.3 | 9,651.4 | 2.08x |
| L1VMIM | 410.0 | 4,686.0 | 10,650.9 | 2.27x |

Interpretation (as reported at the time): CNN summaries preserve much more tomographic information than handcrafted L1-family summaries in this matrix. **Revised interpretation (2026-04-22):** most of the apparent CNN advantage at tomo4 was driven by the compressor exploiting per-channel spatial means of the simulation maps, which carry cosmological information that is not recoverable in real data (mass-sheet degeneracy). See §13.

### 3.3 BNT impact in tomo4 (paired no-BNT controls)

> ⚠ **Superseded in 2026-04-22 by §13.** The CNN rows of this table are also inflated by the mass-sheet leak (both the no-BNT and BNT absolute FoM3 values). The *ratio* BNT/noBNT was similarly degraded under the leak, but the absolute scale is not reliable. In the demeaned pipeline (§13), CNN no-BNT and CNN BNT are both ~25–30x wider than pre-patch, and BNT/noBNT ratios sit near 0.63–0.87.

| Method | noBNT mean FoM3 | BNT mean FoM3 | BNT/noBNT |
|---|---:|---:|---:|
| CNN | 221,867.5 | 21,030.4 | 0.0948 |
| L1 | 11,127.0 | 679.9 | 0.0611 |
| L1VMIM | 9,836.8 | 617.9 | 0.0628 |

Interpretation: naive/final-paper BNT setup severely degraded all methods; this motivated dedicated CNN retraining campaigns.

---

## 4) CNN BNT-losslessness retraining campaign (major recovery stage)

From `cnn_bnt_losslessness_campaign/CNN_BNT_LOSSLESSNESS_RETRAIN_REPORT.md`.

### 4.1 Core recovery table

| Setup | inflation (BNT/noBNT) | FoM ratio (BNT/noBNT) |
|---|---:|---:|
| baseline_final_paper | 1.8049 | 0.0948 |
| stagej_repro | 1.0392 | 0.7935 |
| advanced cdim=6 | 1.0452 | 0.8923 |
| advanced cdim=8 | 1.0423 | 0.7892 |
| **advanced cdim=10** | **1.0297** | **0.9065** |
| larger NDE only (L10/H320) | 1.0516 | 0.7437 |

Key inference:

- compressor-side capacity/training choices were the dominant lever;
- increasing NDE capacity alone did not solve BNT mismatch and degraded FoM retention.

### 4.2 Sigma8 broadening diagnosis

For advanced `cdim=10`, sigma8 broadening improved relative to lower cdim variants (`sigma8 std ratio 1.2647` vs larger values at cdim 6/8), but residual broadening remained.

---

## 5) Multipatch and independent split campaigns

### 5.1 Multipatch (nonoverlap48) outcome

Multipatch variants did not beat the random25 cdim10 reference in global FoM parity:

- reference (`old_best_cdim10_random25`): inflation `1.0297`, FoM ratio `0.9065`;
- multipatch variants had lower FoM ratios despite some sigma8-ratio improvements in certain long/capacity settings.

Interpretation: more independent spatial patches helped some width diagnostics but did not translate to best global parity.

### 5.2 Independent compressor/NDE split outcome

Split policy: compressor `train[:70%]`, NDE `train[70%:]`, validation `test`, disjoint exact `(cosmology, patch)` examples.

Best split tradeoff (`indep_split_advanced_cdim10_long120k`):

- inflation `1.0369`, FoM ratio `0.8462`, sigma8 ratio `1.0966`.

Split-independence audit:

- exact-example overlap between compressor/NDE train subsets: `0`;
- unique theta overlap: full overlap (`899/899`), so independence is example-level, not cosmology-level.

---

## 6) Noise-curriculum campaign

From `cnn_bnt_noise_curriculum_campaign/FINAL_NOISE_CURRICULUM_REPORT.md`.

### 6.1 Consolidated metrics (primary finding)

| config | inflation | FoM ratio | abs(FoM-1) | rank |
|---|---:|---:|---:|---:|
| **plain_ref** | 1.0200 | 0.9137 | 0.0863 | 0.1063 |
| plain_curriculum | 1.1117 | 0.7570 | 0.2430 | 0.3548 |
| resnet18_ref | 1.3329 | 0.4331 | 0.5669 | 0.8999 |
| resnet18_curriculum | 0.9615 | 0.8684 | 0.1316 | 0.1701 |

Conclusions reported:

- curriculum harmed plain-family parity versus plain reference;
- curriculum strongly improved resnet18 versus resnet18 reference;
- follow-up resnet schedules did not beat the primary resnet18 curriculum run.

---

## 7) ResNet campaigns and tradeoffs

### 7.1 Split campaign summary

From `cnn_bnt_resnet_split_campaign/CNN_BNT_RESNET_SPLIT_CAMPAIGN_REPORT.md`:

- `control_plain_split`: inflation `1.0592`, FoM ratio `0.6099`;
- `resnet50_split`: inflation `1.0703`, FoM ratio `0.5791`;
- `resnet_small_split`: inflation `1.1222`, FoM ratio `0.4118`.

Interpretation: tested ResNets did not beat matched split plain-CNN control.

### 7.2 Extended ResNet tuning (v2)

From `resnet_extended_tuning_v2/EXTENDED_RESNET_COMPARISON_REPORT.md`:

- best parity-only ResNet: `resnet18_long15k_std10k_l10h320` with near-perfect FoM ratio (`1.0049`) and rank `0.0397`;
- but retention against strongest plain reference was low (`retention_vs_advanced_plain_long=0.4470`);
- no ResNet beat advanced plain long on both parity and retention jointly.

Interpretation: parity-retention Pareto tension persisted.

---

## 8) Noiseless vs noisy evidence

From `cnn_noiseless_vs_noisy/CNN_NOISELESS_VS_NOISY_REPORT.md`:

- no-BNT noiseless/noisy std ratio: `0.6438`, FoM ratio: `7.7962`;
- BNT noiseless/noisy std ratio: `0.3543`, FoM ratio: `86.5925`.

Interpretation: much of the BNT difficulty is noise-conditioned; noiseless BNT can be very constraining, but noisy BNT is substantially harder for the learned inference stack.

---

## 9) Final parity-techniques campaign (paired consistency + domain adversarial)

Phased campaign executed with resource-capped GPUs and 5-seed confirmations.

### 9.1 Pilot and confirmation metrics

| Variant | Family | Seeds | FoM ratio (BNT/noBNT) | inflation | rank |
|---|---|---|---:|---:|---:|
| plain_baseline_pilot | plain | 41,42,43 | 0.8184 | 1.0249 | 0.2065 |
| plain_consistency_pilot | plain | 41,42,43 | 0.7544 | 1.0585 | 0.3041 |
| plain_consistency_adv_pilot | plain | 41,42,43 | 0.8350 | 1.0545 | 0.2195 |
| plain_consistency_adv_confirm | plain | 41..45 | 0.7491 | 1.0564 | 0.3073 |
| resnet18_baseline_pilot | resnet18 | 41,42,43 | 0.0238 | 1.6816 | 1.6578 |
| resnet18_consistency_pilot | resnet18 | 41,42,43 | 0.6088 | 1.0573 | 0.4485 |
| resnet18_consistency_adv_pilot | resnet18 | 41,42,43 | 0.2076 | 1.3024 | 1.0948 |
| resnet18_consistency_confirm | resnet18 | 41..45 | 1.4329 | 0.8222 | 0.6106 |

Campaign verdict:

- no tested invariance trick generalized stably in confirmation;
- consistency/adversarial objectives improved select pilot points but failed to deliver robust near-lossless parity.

---

## 10) Unified interpretation of the project evidence

### 10.1 What is strongly supported

1. ~~**CNN summaries capture much more tomo4 information** than L1-family summaries in the project’s final matrix.~~ **Revised 2026-04-22 (§13):** this claim must be retracted in the pre-patch form. The CNN-over-L1 advantage in tomo4 was dominated by the compressor exploiting per-channel spatial means of simulation maps (mass-sheet-degeneracy leak). With `--zero-mean-maps`, CNN no-BNT sits at parity with L1 no-BNT on std_sum and within ~30–50% on FoM3 — an advantage, but a much smaller one than previously reported.
2. **BNT inflation is not a fixed law**: it can be dramatically reduced by demanding compressor design/training choices.
3. **BNT extraction is harder in noisy regimes** than no-BNT extraction.
4. **Cross-bin information handling is a central differentiator** between methods: channelized learned summaries are effective for tomography, but the magnitude of this differentiator is smaller than previously reported once the mass-sheet leak is closed (§13).

### 10.2 What remains unresolved

1. Achieving **stable near-lossless parity** under BNT across broad seeds/hyperparameter families remains unsolved.
2. Invariance-style regularization (paired consistency/adversarial) showed **objective conflict and poor robustness** without substantial retuning.
3. ResNet families showed a persistent **parity-retention tradeoff** relative to best plain references.

### 10.3 Practical working hypothesis

Observed BNT inflation in higher-order handcrafted pipelines is largely a consequence of difficult information extraction under transformed noisy tomography and incomplete effective recovery of cross-bin couplings. Learned channelized neural summaries can recover a larger part of this cross-information, but only with sufficiently strong optimization/training protocols; otherwise BNT space remains under-extracted.

---

## 11) Evidence index (core files)

- Project-level synthesis:
  - `scripts/sbi/results/final/paper_sbi_consolidation/FINAL_SCIENTIFIC_REPORT.md`
- L1 diagnosis/fixes:
  - `L1_CONTOUR_INVESTIGATION_LOG.md`
  - `L1_FIXES_VALIDATION_REPORT.md`
- L1-VMIM:
  - `L1_VMIM_FINAL_CONCLUSIONS.md`
- Pipeline audit:
  - `SBI_L1_CNN_PIPELINE_DETAILED.md`
- CNN BNT recovery campaigns:
  - `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign/CNN_BNT_LOSSLESSNESS_RETRAIN_REPORT.md`
  - `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_noise_curriculum_campaign/FINAL_NOISE_CURRICULUM_REPORT.md`
  - `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_resnet_split_campaign/CNN_BNT_RESNET_SPLIT_CAMPAIGN_REPORT.md`
  - `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_resnet_split_campaign/resnet_extended_tuning_v2/EXTENDED_RESNET_COMPARISON_REPORT.md`
- Noise regime diagnostic:
  - `scripts/sbi/results/final/paper_sbi_consolidation/cnn_noiseless_vs_noisy/CNN_NOISELESS_VS_NOISY_REPORT.md`
- BNT conceptual note:
  - `BNT_NO_BNT_CONTOUR_INFLATION_NOTE.md`

---

## 12) Draft structure for an A&A letter (Letter to the Editor)

### 12.1 Working title options

1. **Tomographic cross-information recovery with neural summaries mitigates BNT contour inflation in weak-lensing SBI**
2. **BNT contour inflation in higher-order weak-lensing statistics: evidence for cross-bin information loss and neural recovery**
3. **Simulation-based weak-lensing inference with BNT: why handcrafted higher-order summaries inflate and neural summaries recover**

### 12.2 Central claim to test in the letter

In tomographic weak-lensing SBI, apparent BNT-induced contour inflation in higher-order handcrafted summaries is largely due to incomplete effective use of tomographic cross-information; channelized neural summaries recover substantially more of this information and can approach no-BNT performance, but only under sufficiently demanding training.

### 12.3 Proposed letter outline

1. **Introduction (motivation and problem statement)**
   - BNT motivation and empirical inflation issue.
   - Why this matters for practical cosmological inference.

2. **Methods**
   - Data and splits, tomography setup, noise model.
   - Summary families: L1, L1-VMIM, CNN.
   - SBI/NDE setup and metrics (FoM3, inflation ratios, sigma8 width ratios).

3. **Results I: Baseline comparative evidence**
   - no-BNT matrix (CNN vs L1/L1-VMIM).
   - BNT degradation under baseline settings.

4. **Results II: Recovery attempts and limits**
   - compressor-focused recovery (losslessness campaign).
   - multipatch and independent-split outcomes.
   - curriculum and ResNet tradeoffs.
   - parity-techniques confirmation failures.

5. **Interpretation**
   - Cross-bin information argument.
   - Why learned channelized summaries help.
   - Why BNT/no-BNT parity is optimization-sensitive in noisy maps.

6. **Conclusions**
   - What is established.
   - What remains open.
   - Recommended best-practice protocol for future BNT SBI analyses.

### 12.4 Figure plan (letter-sized, high-impact)

1. **Figure 1:** no-BNT method comparison (CNN vs L1 vs L1-VMIM) in tomo4 and single-bin.
2. **Figure 2:** BNT vs no-BNT overlays for baseline methods (showing inflation).
3. **Figure 3:** CNN recovery progression (baseline -> stagej -> advanced cdim10).
4. **Figure 4:** Failure/sensitivity panel (curriculum, ResNet, parity-technique confirmations).
5. **Appendix figure:** baryonified-observation robustness summary.

### 12.5 Table plan

1. **Table 1:** core setup and metrics definitions.
2. **Table 2:** no-BNT performance matrix across methods.
3. **Table 3:** BNT/no-BNT ratios across key campaigns (losslessness, curriculum, parity).
4. **Table 4:** best-available parity-retention tradeoff points and confirmation outcomes.

### 12.6 Claims-to-evidence map (for writing discipline)

- Claim: CNN outperforms L1-family in tomo4 no-BNT.
  - Evidence: `FINAL_SCIENTIFIC_REPORT.md` FoM table.
- Claim: baseline BNT strongly degrades all methods.
  - Evidence: `FINAL_SCIENTIFIC_REPORT.md` BNT impact table.
- Claim: compressor retraining can dramatically recover parity.
  - Evidence: `CNN_BNT_LOSSLESSNESS_RETRAIN_REPORT.md`.
- Claim: perfect/stable parity is still difficult.
  - Evidence: parity campaign Phase C confirmations.
- Claim: noisy BNT is harder than noiseless BNT.
  - Evidence: `CNN_NOISELESS_VS_NOISY_REPORT.md`.

---

## 13) Zero-mean-maps parity check and mass-sheet-degeneracy correction (2026-04-22)

### 13.1 Finding and why it matters

The CNN-VMIM compressor used throughout the campaigns in §3–§9 of this document was operating on absolute-level tomographic convergence maps. Under the mass-sheet degeneracy, real weak-lensing convergence is recoverable per redshift bin only up to an additive constant, so any reliance by the compressor on the per-example per-channel spatial mean is an unphysical information channel unavailable to a real survey.

A dedicated parity check was run on the two strongest reference configs from the losslessness and resnet-tuning campaigns: `resnet18_long15k_nostd6k_l8h256` (cdim=6, 15k compressor / 6k flow steps) and `advanced_arch64_dense256_nostd_long` (plain-CNN cdim=10, 120k compressor / 10k flow steps). A new `--zero-mean-maps` flag (added 2026-04-21 to `scripts/sbi/npe_cnn_nbody_tomo.py`) subtracts the per-example per-channel spatial mean of the maps before the compressor, on observed and augmented training/eval maps, in both the standard and paired-BNT augmentation branches. The BNT transform being linear across channels, `B(x − m·1) = Bx − Bm·1` remains zero-mean per channel in BNT space, so demean-before-BNT is invariance-preserving for paired training.

All four parity-check compressors (arch × {no-BNT, BNT}) were retrained from scratch — old checkpoints are physically incompatible with the demeaned input distribution and are not reused via `--no-train`.

Observed-map per-channel means that the pipeline was subtracting (resnet18, seed 41, no-BNT): `[0.00777, 0.01676, 0.03667, 0.05491]`. The bin-4 value is ≈ 4× the per-pixel shape-noise std (0.01266), so there was substantial unphysical signal available to the compressor pre-patch.

### 13.2 Measured impact on the two strongest reference configs

Seeds 41–43 (resnet18), 41–45 (advanced-plain cdim=10), 20 deg / 160 px, 4 tomo bins, flow NVP l8/h256, truth `[0.26, 0.84, −1.0, 0.6736, 0.9649, 0.0493]`.

| config | variant | regime | std_sum | σ₈ std | FoM3 |
|---|---|---|---:|---:|---:|
| resnet18 | old (leak) | no-BNT | 0.1910 | 0.0148 | 3.60e5 |
| resnet18 | old (leak) | BNT    | 0.2016 | 0.0180 | 2.81e5 |
| resnet18 | **new (demeaned)** | no-BNT | **0.3441** | **0.0417** | **1.49e4** |
| resnet18 | **new (demeaned)** | BNT    | **0.3706** | **0.0501** | **9.41e3** |
| advanced-plain cdim=10 | old (leak) | no-BNT | 0.1770 | 0.0139 | 5.38e5 |
| advanced-plain cdim=10 | old (leak) | BNT    | 0.1906 | 0.0157 | 4.03e5 |
| advanced-plain cdim=10 | **new (demeaned)** | no-BNT | **0.3456** | **0.0382** | **1.69e4** |
| advanced-plain cdim=10 | **new (demeaned)** | BNT    | **0.3571** | **0.0401** | **1.46e4** |

New / old ratios:

| config | regime | std_sum | σ₈ std | FoM3 (new/old) | det(Cov₃) (new/old) |
|---|---|---:|---:|---:|---:|
| resnet18 | no-BNT | **1.80×** | **2.81×** | **0.042×** (24× worse) | ≈576× |
| resnet18 | BNT    | **1.84×** | **2.78×** | **0.034×** (30× worse) | ≈900× |
| advanced-plain cdim=10 | no-BNT | **1.95×** | **2.76×** | **0.031×** (32× worse) | ≈1020× |
| advanced-plain cdim=10 | BNT    | **1.87×** | **2.56×** | **0.036×** (28× worse) | ≈770× |

BNT-vs-no-BNT parity (original target of the `bnt-parity-techniques` branch):
- resnet18: std inflation 1.056 → 1.077 (slightly worse); FoM3 ratio 0.781 → 0.630 (worse).
- advanced-plain cdim=10: std inflation 1.077 → 1.033 (closer to 1.0); FoM3 ratio 0.749 → 0.867 (closer to 1.0).

Demeaning is therefore **not** the fix for BNT/no-BNT parity; mass-sheet leakage and BNT inflation are independent effects, with mass-sheet leakage being by far the larger factor in pre-patch posterior tightness.

### 13.3 CNN (demeaned) vs canonical L1-norm, seeds 41–43

Head-to-head against the reference L1-norm posteriors from `scripts/sbi/results/final/paper_sbi_consolidation/bnt_comparison_tomo4/posteriors/l1_tomo4_20deg160_{bnt,nobnt}_s{41,42,43}.npy`:

| config | compressor | regime | std_sum | σ₈ std | FoM3 |
|---|---|---|---:|---:|---:|
| resnet18 | CNN (demeaned) | no-BNT | 0.344 | 0.0417 | 1.49e4 |
| resnet18 | L1-norm | no-BNT | 0.346 | 0.0467 | 1.11e4 |
| resnet18 | CNN (demeaned) | BNT    | 0.371 | 0.0501 | 9.41e3 |
| resnet18 | L1-norm | BNT    | 0.609 | 0.1511 | 680 |
| advanced-plain cdim=10 | CNN (demeaned) | no-BNT | 0.344 | 0.0389 | 1.68e4 |
| advanced-plain cdim=10 | L1-norm | no-BNT | 0.346 | 0.0467 | 1.11e4 |
| advanced-plain cdim=10 | CNN (demeaned) | BNT    | 0.352 | 0.0406 | 1.51e4 |
| advanced-plain cdim=10 | L1-norm | BNT    | 0.609 | 0.1511 | 680 |

Readings:

- **No-BNT is at near-parity.** Demeaned CNN and L1-norm agree on std_sum to within 0.6%. CNN is ~10–17% tighter on σ₈ and ~1.3–1.5× higher FoM3. This contradicts the pre-patch picture in §3.2 where CNN tomo4 FoM3 was ~40× L1's.
- **BNT is the regime where CNN still clearly wins.** CNN BNT std_sum is 0.35–0.37 vs L1 BNT 0.609; σ₈ std 0.04–0.05 vs 0.151; FoM3 10⁴ vs 680 (15–22×). The BNT-parity problem from §3.3 is now predominantly an L1 problem — the CNN side is only moderately degraded by BNT (~1.03–1.08× std_sum relative to its own no-BNT), while L1 nearly doubles.

Caveat: CNN uses TFDS variant `grid_20deg_160px_nonoverlap48`, L1 uses `grid_20deg_160px`. That changes only the simulation-side distribution used to train each compressor, not the observed-data inference, so the comparison is fair at inference time but the two chains have different training-patch layouts.

### 13.4 Implications for earlier claims in this knowledge base

| Section | Claim (pre-patch) | Status after §13 |
|---|---|---|
| §3.2 | CNN tomo4 FoM3 ≈ 3.87e5 (≈40× L1). | Retracted. Post-demean CNN tomo4 FoM3 ≈ 1.5e4 at seeds 41–43, only ~1.3–1.5× L1. |
| §3.3 | CNN FoM3 BNT/noBNT ≈ 0.095. | Absolute numbers unreliable; post-demean ratios are 0.63 (resnet18) and 0.87 (advanced-plain). |
| §4.1 | Advanced cdim=10 inflation 1.03, FoM ratio 0.91. | Inflation-ratio result survives (ratios are partially leak-invariant), but absolute FoM level was inflated. Re-running the losslessness campaign under `--zero-mean-maps` is needed for a clean comparison. |
| §7.2 | Best parity-only ResNet: `resnet18_long15k_std10k_l10h320` with FoM ratio 1.0049. | Same caveat as §4.1: parity ratio may survive, FoM absolute values do not. |
| §10.1 claim 1 | CNN >> L1 in tomo4 no-BNT information content. | Retracted in the strong form. The CNN advantage is now small. |
| §10.3 "Practical working hypothesis" | Learned channelized summaries recover much more cross-bin info than handcrafted summaries. | Partially preserved: CNN is still best under BNT. The no-BNT part of the hypothesis is now only weakly supported. |

Every CNN-VMIM posterior produced in this repo prior to 2026-04-21 over-states its constraining power by a factor of ~2 in marginals and ~25–32× in FoM3. Any future scientific claim from this pipeline must go through a `--zero-mean-maps` pipeline (or an equivalent compressor-level mass-sheet invariance) before it is reportable.

### 13.5 Minimal next actions

1. **Adopt `--zero-mean-maps` as default for all paper-track CNN-VMIM runs** (the flag default remains OFF to keep existing campaign scripts backwards-compatible; every new run must opt in explicitly).
2. **Re-audit L1 and L1-VMIM pipelines** for the same issue: confirm (or refute) that `WLStatistics`-based summaries are mean-invariant. If L1 statistics depend on map means, repeat this exercise there too.
3. **Re-launch the strongest parity sweeps** (`resnet_extended_tuning_v2`, `losslessness_campaign_multipatch_advanced_cdim10_long120k`, noise-curriculum follow-ups) under `--zero-mean-maps` and rank on `abs(fom3_ratio_bnt_over_nobnt − 1)` against the new wider no-BNT baseline.
4. **Revise §12 (A&A letter draft)** to lead with BNT as the primary regime of CNN advantage, rather than no-BNT tomo4, since no-BNT parity is now the cleaner L1-vs-CNN story.

### 13.6 Evidence

- `scripts/sbi/results/exploratory/zero_mean_maps_parity_check/SUMMARY.md` — protocol-formatted summary.
- `scripts/sbi/results/exploratory/zero_mean_maps_parity_check/metrics/comparison_old_vs_new.{csv,json}` — per-config, per-seed old-vs-new widths and FoM3.
- `scripts/sbi/results/exploratory/zero_mean_maps_parity_check/metrics/comparison_cnn_vs_l1.{csv,json}` — demeaned CNN vs canonical L1-norm at seeds 41–43.
- `scripts/sbi/results/exploratory/zero_mean_maps_parity_check/overlays/*_4way_overlay.{png,pdf}` — old-vs-new triangle plots.
- `scripts/sbi/results/exploratory/zero_mean_maps_parity_check/overlays/*_vs_l1_overlay.{png,pdf}` — CNN (demeaned) vs L1-norm overlays.
- Commits on branch `bnt-parity-techniques`: `deb5ee0` (flag + paired-BNT compressor), `af60555` (4-way overlay helper), `7ce3105` (drivers + SUMMARY), `0b3e64f` (posteriors + overlays + metrics).

---

## 14) Full-sphere harmonic-space cross-maps for L1 (2026-05-01)

### 14.1 Motivation

Two confounds were carried by the flat-sky `jaxili_cross_*_pct1` cross-maps campaign that produced the prior reading "auto+cross L1 channels improve FoM3 in BNT (+46%) but **hurt** it in no-BNT (−12%)":

1. **Flat-sky vs full-sphere geometry.** Patches were produced by gnomonic-projecting NSIDE=512 HEALPix κ maps to 20°/160 px tiles, then cross-maps were computed by FFT *on the patches*. Cross-information at scales > 20° or that crosses patch boundaries is invisible to that route.
2. **FFT vs SHT cross-product.** Flat-sky uses $\tilde\kappa^{ij}(\vec k)=\tilde\kappa^i(\vec k)\,\tilde\kappa^j(\vec k)$ on apodized patches; the full-sphere construction (Zürcher et al. 2022) uses element-wise $a_{\ell m}^{(ij)}=a_{\ell m}^{(i)}\,a_{\ell m}^{(j)}$ on the sphere.

A new pipeline was built to do the cross-maps on the full sphere first, then patch and run the existing L1 + NPE machinery, holding all other knobs identical to the flat-sky `pct1` arm.

### 14.2 Implementation

- New cache builder `scripts/sbi/build_full_sphere_cross_cache.py` (HEALPix NSIDE=512, lmax=1024, σ_e=0.26, n_gal=10, BNT applied in `a_ℓm` space since SHT is linear and commutes with the bin-mixing matrix). For each (cosmology, perm, regime): noise on the sphere → SHT → BNT (regime=bnt only) → 6 element-wise cross products → ISHT → 4 auto + 6 cross full-sphere maps → gnomonic-project to 48 deterministic 20°/160px patches matching the flat-sky TFDS layout → demean per (patch, channel).
- New L1-script entry path `--full-sphere-cross-cache PATH` in `scripts/sbi/npe_l1norm_cross_jaxili_nbody_tomo.py` bypasses the on-the-fly TFDS load + FFT-cross-map step and streams the precomputed 10-channel patches directly. SNR-percentile calibration, L1 build, and NPE training are unchanged. Train/val splits exactly mirror the flat-sky TFDS (cosmologies 1–899 train, 900–1299 val, fiducial reserved for observation).
- Production cache `scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid` (623 GB, 18 186 .npz files, manifest sha256 `0a68ea89669da18f...`, both regimes) built in 56 min wall on 50 CPU workers.

### 14.3 Headline result (3-seed pool, 41/42/43)

Identical L1/NPE settings to `jaxili_cross_*_pct1` (`--cross-snr-percentile 1.0`, 5 scales, 40 SNR bins, [-13, 13], 5000 steps, 3 seeds), only the cross-map construction differs.

| arm | FoM3 | Δ vs auto-only | Δ vs flat-sky cross |
|---|---:|---:|---:|
| auto_zm_bnt              |    789 | —     | —     |
| cross_bnt_pct1 (flat)    |   1156 | +46%  | —     |
| **harm_cross_bnt**       |  **5161** | **+554%** | **+347%** |
| auto_zm_nobnt            |  13131 | —     | —     |
| cross_nobnt_pct1 (flat)  |  11545 | −12%  | —     |
| **harm_cross_nobnt**     | **59243** | **+351%** | **+413%** |

All six runs verified harmonic route (`cross_maps_route=harmonic`, `n_l1_channels=10`, manifest sha = grid sha) and truth coverage |z| ≤ 1.1 across all (seed × parameter) cells; harm_cross_nobnt is exceptionally well-centered (|z| ≤ 0.4). Per-seed FoM3 dispersion is proportionally tighter than the flat-sky arms (harm_cross_bnt σ(FoM3) = 339 on mean 5161; harm_cross_nobnt σ(FoM3) = 3300 on mean 59243).

### 14.4 Implications for earlier claims

| Section | Claim (pre-patch) | Status after §14 |
|---|---|---|
| flat-sky cross campaign | no-BNT cross channels carry no extractable signal beyond auto channels at any percentile | **Retracted.** The conclusion was an artifact of the FFT-on-patches construction. Harmonic-space cross-maps deliver +351% FoM3 over auto-only in no-BNT. |
| flat-sky cross campaign | BNT-regime cross channels add +46% FoM3 | Survives but is a severe under-statement. Harmonic version delivers +554%. |
| §10.3 "Practical working hypothesis" | Learned channelized summaries recover much more cross-bin info than handcrafted summaries (revised in §13 to "no-BNT part only weakly supported"). | The no-BNT cross-bin recovery story is now strongly supported again — but for **handcrafted L1** on full-sphere cross-maps, not for the CNN. |

The flat-sky FFT cross-maps must now be regarded as a known-lossy approximation: they discard (a) cross-information at scales > 20°, (b) cross-information that bridges patch boundaries, (c) non-axisymmetric multipole pairs that gnomonic apodization smears. The Zürcher-style harmonic construction is the new default for any cross-map L1 result going into the paper.

### 14.5 Caveats and next checks

1. The element-wise $a_{\ell m}^{(i)}\cdot a_{\ell m}^{(j)}$ product is **not** a true spherical convolution (that would require an axisymmetric kernel); it is a heuristic that breaks $m$-axis isotropy. This is acknowledged in Zürcher 2022 and is the intended construction; just be careful not to describe it in the paper as "the spherical analog of the FFT cross-map".
2. The mild persistent BNT-regime bias (Ω_m ≈ +0.04, w_0 ≈ −0.16 below truth, all sub-1σ but consistent across the three seeds) deserves a coverage check — likely a prior-boundary or projection effect, but worth confirming with simulation-based calibration before paper submission.
3. Recommend an independent second-cosmology check (e.g. one of the `cosmo_delta_*` corners as observation) to verify the harmonic-cache posteriors track parameter shifts at the right amplitude.

### 14.6 Evidence

- `scripts/sbi/results/exploratory/cross_maps_campaign/cross_summary/summary.{md,json}` — extended with `harm_cross_{bnt,nobnt}` arms.
- `scripts/sbi/results/exploratory/cross_maps_campaign/cross_summary/harmonic_results.md` — interpretive write-up + per-seed truth coverage table.
- `scripts/sbi/results/exploratory/cross_maps_campaign/cross_summary/overlay_harm_vs_flat_vs_auto_{bnt,nobnt}.pdf` — triple-overlay corner plots (auto vs flat-sky cross vs harmonic cross).
- `scripts/sbi/results/exploratory/cross_maps_campaign/jaxili_harm_cross_{bnt,nobnt}/posteriors/l1cross_tomo4_20deg160mp_harm_{bnt,nobnt}_p1_s{41,42,43}.npy` — 6 raw posteriors with `.meta.json` and `.fom.json` siblings.
- Cache + manifest: `scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid/manifest.json` (args sha `0a68ea89669da18f...`).

---

## 15) Final status

This knowledge base is intended to be the main scientific source document for drafting the A&A letter. It should be updated only when new campaign outputs materially change quantitative conclusions.

Last materially significant update: **2026-05-01, §14** (full-sphere harmonic cross-maps overturn the flat-sky no-BNT null result).
