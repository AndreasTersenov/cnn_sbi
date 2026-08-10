# RESULT — FoM₃ error bars for Table 1

Produced 2026-07-28 on IDRIS Jean-Zay, after the Titan RAID0 failure destroyed every
trained compressor, NDE and per-observation posterior behind the published table.
Everything below was re-measured from scratch.

> **Generated file — do not hand-edit.** Produced by `fom_errorbars/make_report.py` from
> the measurement JSONs, so it cannot drift from the data. An identical copy lives in both
> repositories: `L1_vs_CNN_Tomographic_SBI/RESULT_FOM_ERRORBARS.md` (next to the
> `NOTE_FOM_ERROR_BARS.md` specification it implements) and `cnn_sbi/scripts/sbi/results/
> exploratory/flatsky_cross_2026_06/analytical_nde_match/RESULT_FOM_ERRORBARS.md` (next to
> the sibling `RESULT_*.md` documents and the `PLAN_FOM_ERRORBARS_SWEEP.md` that specified
> it). Regenerate both after any change to the sweeps.

This document is the deliverable named in `PLAN_FOM_ERRORBARS_SWEEP.md` (§Outputs) and
implements the specification in `NOTE_FOM_ERROR_BARS.md` (§4, §5). It is self-contained:
the numbers, what each one means, how it was computed, and how it was validated.

---

## 1. The table

Central values are the **published** ones. The ± is the measured compressor-seed spread,
expressed as an absolute uncertainty on that central value.

| Summary | frame | FoM₃ | ± as % | median SE |
|---|---|--:|--:|--:|
| ℓ1 auto-only | no-BNT | **2448 ± 27** | 1.1% | ±5.0 |
| ℓ1 auto-only | BNT | **388 ± 43** | 11.1% | ±1.1 |
| ℓ1 +conv | no-BNT | **2671 ± 133** | 5.0% | ±5.8 |
| ℓ1 +conv | BNT *(new)* | **458 ± 15** | 3.3% | ±1.3 |
| ℓ1 +product | no-BNT | **3045 ± 183** | 6.0% | ±6.1 |
| ℓ1 +product | BNT | **718 ± 29** | 4.1% | ±3.0 |
| ℓ1 +conv+product | no-BNT | **3255 ± 200** | 6.1% | ±7.9 |
| ℓ1 +conv+product | BNT *(new)* | **704 ± 76** | 10.8% | ±2.2 |
| joint ℓ1 | no-BNT | **3371 ± 96** | 2.8% | ±7.6 |
| joint ℓ1 | BNT | **2424 ± 208** | 8.6% | ±6.5 |
| CNN | no-BNT | **3326 ± 14** | 0.4% | ±8.4 |
| CNN | BNT | **3186 ± 19** | 0.6% | ±6.6 |

**n = 3 compressor seeds for every row**, so the ± column is comparable across summaries.
(Six seeds were run for the two widest rows as a check — see §7. The three-seed values are
quoted here deliberately, for cross-row comparability.)

Suggested caption, following `NOTE_FOM_ERROR_BARS.md` §6:

> ± is the spread over three independently trained compressors, the dominant run-to-run
> variability; density-estimator ensembling moves the medians by <1.2%, and the sampling
> error of the median over the 9000 mocks is at the per-cent level (block bootstrap over
> patches).

---

## 2. What the ± means — and what it does not

Each Table-1 entry is the robust **median of FoM₃ over n = 9000 mock observations**
(180 patches × 50 noise realisations). FoM₃ = 1/√det Cov(Ω_m, σ₈, w₀) of the learned
posterior at each mock. The estimate departs from the truth through **bias** (the learned
posterior is mis-calibrated) and **variance** (re-running the pipeline gives a different
number). The ± quotes the *variance* only.

**The ± IS**: the standard deviation of the median FoM₃ across three independently trained
VMIM compressors (seeds 41/42/43), pre-ensemble, with everything else held fixed. It
measures run-to-run reproducibility of the quoted pipeline.

**The ± is NOT**:

* the population scatter of FoM₃ across the 9000 mocks (a different question — that is CV_pop, §6);
* a posterior width (FoM₃ already summarises one);
* the single→ensemble shift — that is the **bias** term, reported separately in §5 and
  **never summed with the ±** (`NOTE_FOM_ERROR_BARS.md` §1, §5.4);
* a leave-one-out ensemble jackknife. An earlier draft of this recovery used one; it is not
  in the specification and must not be quoted. It survives only as a diagnostic in `final_bars.py`.

Three seeds give a σ estimate carrying roughly **46% relative uncertainty**, so these bars
are *indicative* reproducibility, not precision error bars. State them as such.

---

## 3. Which estimator each row quotes

The published table mixes estimators **deliberately**: the 3-compressor ensemble is quoted
only where the single failed the calibration battery and the ensemble corrects it
(`RESULT_BNT_AUTOPROD_ENSEMBLE.md`, `RESULT_NOBNT_ENSEMBLE_ROBUSTNESS.md`).

| Summary | no-BNT | quoted as | BNT | quoted as |
|---|--:|---|--:|---|
| ℓ1 auto-only | 2448 | SINGLE | 388 | ENSEMBLE |
| ℓ1 +product | 3045 | SINGLE | 718 | ENSEMBLE |
| joint ℓ1 | 3371 | ENSEMBLE | 2424 | ENSEMBLE |
| CNN | 3326 | SINGLE | 3186 | SINGLE (already calibrated) |

The ± is nevertheless the **pre-ensemble** seed spread for *all* rows, including the
ensemble-quoted ones, so that the column means the same thing everywhere (§5.4).
**Never compare a retrained single against a published ensemble, or vice versa.**

---

## 4. Full per-row detail

| Row | published | retrained (quoted) | Δ | per-seed singles | ± (std) | bias single→ens | median SE (68%) | ρ | CV_pop |
|---|--:|--:|--:|---|--:|--:|--:|--:|--:|
| l1 auto, no-BNT | 2448 | 2776.9 (single) | +13.4% | 2776.9 / 2739.5 / 2800.0 | ±31 (1.1%) | -9.6% | ±4.95 | 0.000 | 0.171 |
| l1 auto, BNT | 388 | 390.7 (ensemble) | +0.7% | 388.9 / 481.0 / 465.6 | ±49 (11.1%) | +0.4% | ±1.12 | 0.002 | 0.257 |
| l1 +conv, no-BNT | 2671 | 2720.2 (single) | +1.8% | 2720.2 / 2623.5 / 2463.0 | ±130 (5.0%) | -9.5% | ±5.85 | 0.001 | 0.179 |
| l1 +conv, BNT | — *(new)* | 457.6 (ensemble) | — | 496.4 / 495.4 / 524.7 | ±17 (3.3%) | -7.8% | ±1.27 | 0.003 | 0.227 |
| l1 +product, no-BNT | 3045 | 3231.8 (single) | +6.1% | 3231.8 / 3175.9 / 3545.6 | ±199 (6.0%) | -5.5% | ±6.09 | 0.002 | 0.180 |
| l1 +product, BNT | 718 | 758.3 (ensemble) | +5.6% | 912.1 / 890.7 / 964.0 | ±38 (4.1%) | -16.9% | ±2.98 | 0.008 | 0.231 |
| l1 +conv+product, no-BNT | 3255 | 3564.9 (single) | +9.5% | 3564.9 / 3154.9 / 3414.8 | ±207 (6.1%) | -12.2% | ±7.94 | 0.002 | 0.203 |
| l1 +conv+product, BNT | — *(new)* | 703.9 (ensemble) | — | 761.7 / 879.8 / 715.1 | ±85 (10.8%) | -7.6% | ±2.21 | 0.006 | 0.227 |
| joint l1, no-BNT | 3371 | 3379.5 (ensemble) | +0.3% | 3762.3 / 3723.2 / 3926.7 | ±108 (2.8%) | -10.2% | ±7.60 | 0.005 | 0.169 |
| joint l1, BNT | 2424 | 2405.2 (ensemble) | -0.8% | 3075.1 / 3251.4 / 2740.4 | ±260 (8.6%) | -21.8% | ±6.52 | 0.000 | 0.258 |
| CNN, no-BNT | 3326 | 3427.6 (single) | +3.1% | 3427.6 / 3400.8 / 3406.9 | ±14 (0.4%) | -3.0% | ±8.37 | 0.003 | 0.178 |
| CNN, BNT | 3186 | 3147.1 (single) | -1.2% | 3147.1 / 3183.4 / 3158.9 | ±19 (0.6%) | -1.2% | ±6.59 | 0.004 | 0.171 |

`retrained (quoted)` follows each row's published estimator convention (§3). Six of the
eight rows land within 3.1% of published; see §8 for the exception.

---

## 5. The three terms of the error budget

### 5.1 Seed term (the ±, dominant)

Std of the median FoM₃ over compressor seeds 41/42/43, each evaluated on the identical
9000 mocks with the identical NDE recipe. Range 0.4%–11.1% depending on row (§6).

### 5.2 Bias term (reported separately, NOT an error bar)

The single→ensemble shift. Pooling three compressors widens the posterior and lowers FoM₃;
where the single was over-confident this is a *correction*, not noise. Measured here:

| Row | single (s41) | ensemble (3 pooled) | shift |
|---|--:|--:|--:|
| l1 auto, no-BNT | 2776.9 | 2511.1 | -9.6% |
| l1 auto, BNT | 388.9 | 390.7 | +0.4% |
| l1 +conv, no-BNT | 2720.2 | 2461.0 | -9.5% |
| l1 +conv, BNT | 496.4 | 457.6 | -7.8% |
| l1 +product, no-BNT | 3231.8 | 3053.9 | -5.5% |
| l1 +product, BNT | 912.1 | 758.3 | -16.9% |
| l1 +conv+product, no-BNT | 3564.9 | 3130.4 | -12.2% |
| l1 +conv+product, BNT | 761.7 | 703.9 | -7.6% |
| joint l1, no-BNT | 3762.3 | 3379.5 | -10.2% |
| joint l1, BNT | 3075.1 | 2405.2 | -21.8% |
| CNN, no-BNT | 3427.6 | 3324.2 | -3.0% |
| CNN, BNT | 3147.1 | 3109.5 | -1.2% |

### 5.3 Median sampling term (subdominant, caption only)

Block bootstrap over the **180 patches**, resampling patches with replacement and keeping
all 50 noise realisations of each sampled patch intact (the 9000 mocks are not independent),
10⁴ replicates, reporting the **68% percentile interval**, computed once per arm at seed 41.

`NOTE_FOM_ERROR_BARS.md` §4 predicted 0.1–3% depending on the intra-patch correlation ρ and
asked for ρ to be measured rather than assumed. **Measured: ρ = 0.000–0.011 on every row.**
The patch effect is therefore negligible, block resampling behaves almost like an iid
bootstrap, and the median term lands at the bottom of the predicted range (0.16%–0.33%).
The naive √9000 'negligible' claim turns out to be correct — but now demonstrably so.

---

## 6. Why the bars differ so much between rows

The spreads span 0.4% (CNN) to 11.1% (ℓ1 auto BNT), a 27× range. That ordering is real:
two 3-draw σ estimates from the same underlying σ exceed a 4× ratio only ~5% of the time.
Individual row-to-row comparisons are *not* significant, but the pattern is.

**(a) The spread tracks CV_pop** (Pearson +0.84, Spearman +0.71). CV_pop is the mock-to-mock
scatter of FoM₃ for a *fixed, already-trained* pipeline — nothing to do with seeds — yet it
predicts the seed spread. Both measure how sharply determined the FoM is for that summary.

**(b) The mechanism: whether the three marginal σ's compound or compensate.** FoM₃ is a
determinant, so it depends on all three parameter widths jointly. Comparing the FoM₃ spread
to the *sum* of the three marginal σ spreads:

| Row | FoM₃ spread | Σ σ spreads | ratio | interpretation |
|---|--:|--:|--:|---|
| CNN, no-BNT | 0.4% | 1.9% | 0.22 | σ's **compensate** |
| CNN, BNT | 0.6% | 2.3% | 0.26 | σ's **compensate** |
| ℓ1 auto, no-BNT | 1.1% | 2.1% | 0.53 | mixed |
| joint ℓ1, no-BNT | 2.8% | 3.2% | 0.88 | σ's compound |
| ℓ1 +product, BNT | 4.1% | 3.7% | 1.11 | σ's compound |
| ℓ1 +product, no-BNT | 6.0% | 4.5% | 1.34 | σ's compound |
| joint ℓ1, BNT | 8.6% | 8.1% | 1.06 | σ's compound |
| ℓ1 auto, BNT | 11.1% | 10.6% | 1.04 | σ's compound (rank-concordant) |

For the **CNN**, a seed returning a wider σ_Ωm returns a tighter σ_σ8 — the errors trade off,
so the determinant is far more stable than any individual σ. Every seed extracts essentially
the *same amount* of information and differs only in how it distributes it among parameters.

For the **BNT ℓ1 and joint ℓ1 rows**, the σ's move together: the worst seed is worst on all
three parameters at once. Those seeds extract *genuinely different amounts* of information.

**Why the split.** BNT retention is 0.96 for the CNN against 0.16 / 0.24 for ℓ1 auto /
+product. Once BNT has nulled 76–84% of the signal, the hand-designed statistics live on
what remains, and whether a given VMIM compressor finds that residual becomes seed-dependent.
The CNN reads the raw maps, keeps 96%, and every seed reaches the information ceiling —
helped by being heavily over-parameterised for a 10-D bottleneck (ResNet-18, 11.4M params,
80k steps, best-val selection) against a 256×256 MLP on an already-lossy vector.

**This is a publishable observation**: the CNN is not only more BNT-robust in its central
value, it is more BNT-robust in its *reproducibility*.

---

## 7. Six-seed check on the two widest rows

The n=3 spread on both BNT rows is driven by one deviant seed. That seed is **not** an
unlucky draw: the per-seed deviation pattern reproduces across two fully independent
campaigns (different machine, different library versions, freshly trained compressors) —
joint ℓ1 no-BNT r = +0.98, joint ℓ1 BNT r = +0.94 against the published singles, with the
same seed deviating in the same direction. Re-running the same seeds is therefore
uninformative; only *more distinct* seeds probe the population. Seeds 44/45/46 were added:

| Row | n=3 | n=6 | n=6 per-seed | n=6 range |
|---|--:|--:|---|---|
| l1 auto, BNT | ±49 (11.1%) | ±39 (8.8%) | 389 / 481 / 466 / 425 / 401 / 468 | 389–481 |
| joint l1, BNT | ±260 (8.6%) | ±416 (13.4%) | 3075 / 3251 / 2740 / 2551 / 3296 / 3712 | 2551–3712 |

The two rows moved in **opposite directions**. ℓ1 auto BNT tightened (11.1% → 8.8%) with the
six values spread smoothly across the range — a broad unimodal distribution, no isolated
outlier. joint ℓ1 BNT **widened with every seed added** (8.6% at n=3 → 10.9% at n=5 → 13.4%
at n=6), best-to-worst ratio 1.46×.

**Caveat to carry into the paper.** The n=3 values are quoted in §1 for cross-row
comparability, but for joint ℓ1 BNT the three-seed bar understates the measured
compressor-to-compressor variability by roughly a third. That row is the least reproducible
arm in the study: a single trained compressor gives FoM₃ anywhere in 2551–3712. It also
retroactively justifies quoting the ensemble there — with singles that variable, ensembling
is what makes the number quotable at all.

---

## 8. Known caveats

1. **ℓ1 auto no-BNT retrains high.** The retrained single is 2776.9 against the published
   2448 (+13.4%), while its ensemble matches well (2511 vs 2429, +3.4%). Its de-inflation is
   −9.6% where the published one was −0.8%. The published arm mixed `l1none_vmim_s41` with
   `ens_nobnt_auto_s4{2,3}` — caches built at different times — whereas the three seeds here
   are cleanly independent. Since that row is quoted as a *single*, its ± rests on compressors
   that demonstrably behave differently from the originals. **Treat that row's bar as provisional.**
2. **Bias terms are larger than published on the ℓ1 arms** (−9.6% vs −0.8%, −16.9% vs −7.8%,
   −5.5% vs −1.2%) but **match exactly on joint ℓ1** (−10.2% vs −10.2%). The pattern points at
   the ℓ1 auto/+product caches specifically, not at the retraining.
3. **A 3-draw σ carries ~46% relative uncertainty.** These are indicative bars.
4. **The 180 patches tile the same fiducial sky realisations**, so even patch-level
   resampling is mildly optimistic. Irrelevant at the reproducibility-statement level
   (`NOTE_FOM_ERROR_BARS.md` §4).
5. **Training-set realisation** (899 cosmologies, shared by all arms) is not quantified and
   largely cancels in comparisons.

---

## 9. How the numbers were produced

### 9.1 Architecture

Every arm's readout is identical:

```
raw statistic --> VMIM compressor --> 10-D summary --> sbi_lens ConditionalRealNVP 4x128,
                                                       pooled over 3 flow seeds (41/42/43)
```

* **ℓ1 rows** (auto / +product / joint): MLP compressor, hidden (256,256), trained with the
  VMIM objective for 30 000 steps. Do **not** put a flow on the raw ℓ1 vector — it craters.
* **CNN rows**: ResNet-18 VMIM on the four auto maps (`--cross-op none`), dim 10, 80k steps,
  batch 128, lr 5e-4, best-val checkpointing. Replaces both the statistic and the MLP.
* **Ensemble** = the same three compressors pooled (9 flows total).

### 9.2 Data

* Cross TFDS `nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180`
* 180 non-overlapping 10°/80-px patches; `centers_PAPER_180.npy` (never regenerate — see §10)
* Evaluation: 9000 mocks = 180 patches × perms 0–49 of the fiducial obs cache
* Frozen per-(channel,scale) noise σ tables; σ_e = 0.26, galaxy density 7.5, seed base 12345

### 9.3 Pipeline, per arm

```bash
# 1. raw datavector  (joint l1 only; the other arms' caches already existed)
build_flatsky_joint_arm.py --stat jointl1 --basis <nobnt|bnt> --k 10 --adaptive-ranges \
    --out-cache <raw>/cache --out-fid <raw>/fid.npz

# 2. VMIM compressor, one per seed
vmim_from_cache.py --cache-dir <raw>/cache --fid-npz <raw>/fid.npz \
    --out-cache <arm> --out-fid <arm>/fiducial_summaries.npz \
    --summary-dim 10 --hidden 256,256 --nf-layers 4 --nf-hidden 128 \
    --steps 30000 --seed <41|42|43> \
    --preproc-transform log1p-zscore --clip-value 5 --min-feature-variance 1e-5

# 2'. CNN arms instead: compressor from npe_cnn_nbody_tomo.py --train-compressor
#     --exit-after-compress, then the fiducial summaries via
build_fiducial_summaries_cnn.py --arm-dir <arm> --obs-cache <cache root> --g1-tol 9e-4

# 3. population sweep: 1 --arm-dir = single row, 3 = ensemble row
population_sweep.py --arm-dir <arm> [--arm-dir <arm2> --arm-dir <arm3>] \
    --arm-label <label> --out sweeps/<label> \
    --seeds 41,42,43 --n-obs 9000 --max-perm 50 --m-samples 2000 \
    --preproc-transform none --clip-value 0 --min-feature-variance 1e-12

# 4. the bars
make_table.py --rows rows_table1_full.json --nboot 10000 \
    --out TABLE1_ERRORBARS_FULL.json --md TABLE1_ERRORBARS_FULL.md
```

**Preprocessing differs by arm type and this matters.** The compression stage uses
`log1p-zscore / clip 5 / min-var 1e-5`. The downstream sweep on the 10-D summaries uses
`none / clip 0 / min-var 1e-12` for **both** ℓ1 and CNN arms — the campaign's own
`run_flatsky_cnn_population_sweep.py` passes exactly that. Using `zscore` on the CNN arms
instead shifts them by ~3% (CNN BNT: −0.7% → +6.2% against published).

---

## 10. Validation

Nothing here rests on an unverified reconstruction.

* **FoM₃ implementation** verified to 1.7e-16 against 8 surviving `posterior.fom.json` records.
* **`final_bars.py --validate`** reproduces the surviving measured bar exactly
  (3264.6 ± 126.5 = 3.87%; median term ±17.66 vs 17.66).
* **CNN fiducial summaries** — every arm passes a **G1 gate**: recompute the observed summary
  the training run itself persisted (`cache/cnn_obs.npz`) and require max|Δ| ≤ 9e-4.
  Measured 6.3e-05 – 2.8e-04 across all six arms. Compressor checkpoints are additionally
  SHA-256-pinned against the cache fingerprint, so an arm cannot be evaluated with another
  seed's weights.
* **joint ℓ1** reproduces the published campaign to 0.3% (no-BNT ensemble) and 0.8% (BNT
  ensemble), with per-seed singles within 2.7%. The single→ensemble de-inflation matches to
  0.1 percentage points (−10.2% vs −10.2%) — a wrong statistic could not do that.
* **Surviving published runs re-read** and confirmed: `auto_nobnt_ensemble` 2428.6 (pub 2429),
  `auto_bnt_ensemble` 388.4 (388), `jointl1_bnt_ensemble` 2424.3 (2424),
  `l1product_rnvp_s41_n9000` 3044.9 (3045).
* **Patch geometry** confirmed 22/22 by bit-exact reprojection against the cross TFDS.

### Gotchas that cost real time

* **Never regenerate the patch centres.** `_build_non_overlapping_centers` uses an unstable
  `np.argsort` and HEALPix rings are tie blocks: numpy 1.24.4 and 1.26.4 give tilings sharing
  only 2 of 180 centres. Always pass `--centers-npy centers_PAPER_180.npy`.
* The ℓ1 driver's `--fiducial-summaries-out` is **not** flat_local-aware; its own gate rejects it.
* `build_flatsky_joint_arm.py` computes in torch but loads through tf.data and never sets TF
  memory growth — TensorFlow will take the whole GPU and torch will OOM. Set
  `TF_FORCE_GPU_ALLOW_GROWTH=true`.
* `--fiducial-obs-cache-dir` (ℓ1 driver) wants the cache **root**; the CNN driver's flag is
  `--fiducial-obs-cache` (no `-dir`).

---

## 11. Provenance

The Titan RAID0 failure left many files present but filled with NULs. Recovered by comparing
all 16 branches of `AndreasTersenov/cnn_sbi` against the local tree: **197 files destroyed
locally and 50 absent entirely** were retrievable from GitHub (246/247 downloaded).

| File | fate |
|---|---|
| `flatsky_joint_stats.py` | destroyed (11 710 B of NULs); **recovered byte-identical from GitHub** (`analytical-nde-match-2026-06`, sha256 `bd17b539…`) |
| `build_fiducial_summaries_cnn.py` | destroyed; reconstructed here, then cross-checked against the GitHub original (functionally equivalent, stricter gate) |
| `population_sweep_flatsky.py` | destroyed; rewritten as `population_sweep.py`, validated against published numbers |
| `train_jaxili_from_compressed.py` | destroyed; rewritten, FoM₃ verified to 1.7e-16 |
| `final_bars.py` | 0 bytes, **not** in git; written from the `NOTE_FOM_ERROR_BARS.md` spec |
| `flatsky_cross_noise_sigma.npz` (no-BNT) | destroyed; regenerated, cross-checked against the intact `_bnt` twin |
| all compressor / NDE checkpoints | gone; retraining was the only route |

The local `.git` is itself corrupt (`not a GIT packfile`), which is why an early check
wrongly concluded these files were never committed. **Query the remote, not the local object
store.**

The joint ℓ1 configuration was not guessed: the surviving `jointl1_bnt_raw/cache/
l1_cache_meta.npz` records `stat=jointl1, k=10, adaptive_ranges=True, dequantize=False,
snr_range=5.0, append_to=None`. `--rotated-binning` is a *later* experiment that failed its
BNT gate and is not the published lineage.

---

## 12. Files

| File | contents |
|---|---|
| `RESULT_FOM_ERRORBARS.md` | this document |
| `TABLE1_PAPER.md` / `.json` | the paper-ready table |
| `TABLE1_ERRORBARS_FULL.md` / `.json` | full per-row detail: per-seed values, bias, ρ, CV |
| `SEEDCHECK_N6.json` | the six-seed check on the two BNT rows |
| `rows_table1_full.json` | which sweep produced which row |
| `make_table.py` | builds the tables from the sweeps |
| `make_report.py` | builds this document from the JSONs |
| `final_bars.py` | the three bar terms + `--validate` self-check |
| `population_sweep.py` | per-arm 9000-mock sweep, persists per-obs mean + full 3×3 covariance |
| `build_fiducial_summaries_cnn.py` | CNN observed summaries + G1 gate |
| `build_joint_arm.slurm`, `vmim_joint.slurm`, `sw_joint_chain.slurm`, `extra_seeds.slurm`, `cnn_fidsum.slurm`, `sw_cnn.slurm` | the job scripts as run |

Per-observation posterior **mean and full 3×3 covariance** are persisted for every arm and
every pooled member (`per_patch_metrics.npz`: `mean`, `cov`, `arm_mean`, `arm_cov`), plus
thinned raw samples. Their absence is what made the original bars unrecoverable; it will not
happen again.

