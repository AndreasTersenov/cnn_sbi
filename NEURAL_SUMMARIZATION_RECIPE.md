# Neural summarisation (VMIM-MLP) + NDE: a transferable recipe

A self-contained description of the two-stage compress-then-infer pipeline that worked well here
(learning low-dimensional summaries of high-dimensional wavelet ℓ1 datavectors and inferring
cosmology), written so it can be reused on a *different* statistic (e.g. power spectra) in another
project. It covers the architecture, the objective, the recipe that worked, the lessons we learned by
iterating, and pointers to the exact code.

## 0. The shape of the pipeline (two stages)

```
  data vector x  (high-D summary statistic, e.g. ℓ1 bins, or P(k) bins)
        │  Stage 1: VMIM compressor  c(x) = MLP, trained by variational mutual-information max
        ▼
  summary  y = c(x) ∈ R^d   (low-D, d ≈ 10)
        │  Stage 2: NDE  p(θ | y)   (expressive conditional normalising flow)
        ▼
  posterior on θ   →  gate it (TARP + SBC) before trusting any width
```

The two stages are trained **separately**: first the compressor (Stage 1), then — on the *frozen*
compressed summaries — the density estimator (Stage 2). Keeping them separate is what let us swap the
Stage-2 estimator freely and discover that the estimator choice is a large lever (Sect. 4).

Everything is JAX + Haiku; the flows are `sbi_lens`/`tensorflow_probability`-style
`ConditionalRealNVP`. Stage 1 and Stage 2 each take a `(θ, x)` cache as `.npz` and write a compressed
cache, so the same scaffold transfers to any statistic by swapping the input cache.

## 1. Stage 1 — the VMIM compressor

**What VMIM is.** Variational Mutual-Information Maximisation (Jeffrey & Wandelt; the IMNN/VMIM line):
train a summary network `c(x)` to maximise the mutual information `I(θ; c(x))` between the summary and
the parameters. Since `I` is intractable, maximise its variational lower bound

  I(θ; c(x)) ≥ E[ log q(θ | c(x)) ] + const,

where `q` is a flexible **companion** density `q(θ|y)`. Maximising the bound = minimising the loss
`L = −E[ log q(θ | c(x)) ]` over *both* the compressor `c` and the companion `q` jointly. The summary
that best lets a flexible density predict θ is, by construction, the most informative low-D summary —
better than a regression/MSE summary, which only targets the posterior mean.

**Architecture (what worked):**
- Compressor `c`: a plain MLP — hidden `(256, 256)`, `leaky_relu`, final linear to `summary_dim`
  (`d = 10`). Stateless Haiku (`hk.transform`).
- Companion `q(θ|y)`: a `ConditionalRealNVP` with `n_layers = 4` affine-coupling blocks, each coupling
  MLP `[128, 128]` with `silu`. It exists only to define the VMIM loss; it is discarded after Stage 1.
- Loss: `loss = −mean( q.log_prob(θ | c(x)) )`. Adam, piecewise-constant LR decay (×0.7), batch 512,
  ~30k steps, single seed; keep the **best-validation** checkpoint.

**Preprocessing (matters — fit on train only):** `log1p` then z-score (per feature), clip at ±5σ, and
drop zero-variance features (a min-variance mask). For statistics spanning orders of magnitude (ℓ1, and
power spectra) the log step is important. The preprocessing lives *inside* Stage 1; the compressed
summary is passed *raw* to Stage 2 (downstream preproc = none).

**Why a single compressor seed:** statistical rigour comes from the Stage-2 estimator seeds (Sect. 3),
not from the compressor. Exception: a *deep ensemble over compressor seeds* is the tool for fixing
over-confidence (Sect. 4, calibration).

**Code:** `scripts/sbi/vmim_from_cache.py` — the clean, self-contained reference. It takes
`--cache-dir` (a `{prefix}_train.npz`, `{prefix}_val.npz` with arrays `theta`, `x`) and a fiducial
`--fid-npz` (key `S`), and writes a compressed cache + compressed fiducial. Args: `--summary-dim 10
--hidden 256,256 --nf-layers 4 --nf-hidden 128 --steps 30000 --lr 5e-4 --batch-size 512 --seed 41
--preproc-transform log1p-zscore --clip-value 5 --min-feature-variance 1e-5`. (Integrated originals,
if you want the in-pipeline version: `npe_l1vmim_nbody_tomo.py` / `npe_l1vmim_jaxili_nbody_tomo.py`,
the `CompressorMLP` pattern this was distilled from.)

## 2. Stage 2 — the density estimator (NDE)

On the *frozen* 10-D summaries, fit a conditional normalising flow `p(θ|y)` and read the posterior.
The production estimator here is the **sbi_lens `ConditionalRealNVP`** (`build_flow`/`train_flow`),
capacity `(n_layers=4, hidden=128)`, trained ~50k steps, Adam with cosine-ish LR `1e-3 → 1e-5`,
gradient clip 1.0, weight decay 1e-4, early-stopping patience. Pool **3 estimator seeds** for the
reported posterior (per-obs sample pooling).

**Code:** `scripts/sbi/train_nde_from_compressed.py` — `--nde-family {sbilens_realnvp, jaxili_maf,
jaxili_realnvp, jaxili_mdn}`, `--nde-layers/--nde-hidden`, `--seeds 41,42,43`, `--n-obs`. It reuses the
same metric machinery (`compute_fom3`, `marginal_stats`) and per-obs posterior sampling. The
`sbilens_realnvp` family calls `build_flow`/`train_flow` from `scripts/sbi/npe_cnn_nbody_tomo.py`
(that is where the actual flow is defined).

## 3. The minimal command sequence (any statistic)

```
# 1. compress  (high-D x -> 10-D summary)
python vmim_from_cache.py --cache-dir <RAW_CACHE> --fid-npz <RAW_FID> \
       --out-cache <ARM>/cache --out-fid <ARM>/fid.npz --summary-dim 10 --seed 41
# 2. infer + score  (sbi_lens RealNVP on the 10-D summary, 3 seeds)
python train_nde_from_compressed.py --train-cache-dir <ARM>/cache --cache-prefix l1 \
       --fiducial-summaries-npz <ARM>/fid.npz --output-dir <ARM>/run \
       --nde-family sbilens_realnvp --nde-layers 4 --nde-hidden 128 --seeds 41,42,43 --n-obs 9000
# 3. GATE (mandatory) — TARP-DRP + SBC
python tarp_stratified_val_nde.py ... --dumps-root <G>/dumps ; python run_tarp_coverage.py ... ;
python gate_verdict.py --gate-dir <G> --arms <arm> --json-out <G>/verdict.json
```

## 4. Lessons we paid for by iterating (the advice)

1. **The NDE choice is a large lever — test it explicitly.** On the *identical* 10-D summary, the
   sbi_lens RealNVP gave ~30% higher FoM than a MAF (3146 vs 2426). Do not assume the flow family is
   neutral; sweep at least {RealNVP, MAF} on the same frozen summary and pick by *calibrated*
   performance.
2. **Compress first, then use the expressive flow.** Feeding the raw high-D datavector (~10³ dims)
   straight into the RealNVP collapsed it (FoM ~1100). The VMIM compression to ~10 D is what lets an
   expressive estimator be used at all. (For a *low*-D statistic like a binned power spectrum this is
   less acute — the flow may handle the raw vector — but a VMIM compression to ≈ n_params dimensions is
   still a clean, near-sufficient reduction and removes the high-D failure mode entirely.)
3. **VMIM beats regression.** A summary trained to maximise `E log q(θ|c(x))` with a flexible companion
   captures more than an MSE-to-θ summary (which only fits the mean). Use the flow companion.
4. **`summary_dim` ≈ a little above the number of parameters.** d=10 for 6 parameters worked; the
   summary only needs to carry the parameter-relevant directions. Too small loses info; too large just
   makes Stage 2 harder. For a near-Gaussian statistic, d ≈ n_params is often enough.
5. **Calibration is not optional, and tightness ≠ correctness.** Always gate with **TARP-DRP +
   SBC** before trusting any contour. We repeatedly saw summaries reach a *higher* figure of merit by
   becoming *over-confident* (a dense count-histogram statistic hit a spurious FoM ~4900 and failed
   coverage — rejected). Register the predicted direction, then gate. SBC rank-std should sit near the
   uniform value (≈0.289 for our sample size); pooled TARP net bias near 0.
6. **The completeness/calibration trade-off.** Richer summaries buy raw FoM but can tip into
   over-confidence; the calibrated optimum is usually a *robust* form of the summary (we used ℓ1-
   weighted histograms, not raw counts). Prefer low-variance reductions.
7. **Over-confidence has a clean, non-conformal fix: a compressor deep-ensemble.** If the gated
   posterior is mildly over-confident (SBC std a bit high, marginals too narrow), train the VMIM
   compressor with 2–3 different seeds and **pool their posteriors per observation**. This diversifies
   the summary and washes out the single-compressor amortisation over-confidence, moving SBC back to
   uniform and TARP to the diagonal — without a post-hoc recalibration. (If instead the posterior is
   *conservative*, this is the wrong tool — it widens further; sharpen via flow capacity/training
   instead. Diagnose the direction first.)
8. **Differences of 20–30% in a figure of merit between two summaries are partly the estimator, not
   the physics** (lesson 1). Compare summaries only through *one fixed, calibrated* estimator, and read
   the calibrated marginals/2D areas alongside the headline scalar.

## 5. Transfer notes for power spectra (the new project)

The seam is identical — only the input `(θ, x)` cache changes (P(k) bins, possibly across tomographic
or cross spectra, instead of ℓ1 bins). Specific expectations:
- The power spectrum is **lower-dimensional and more Gaussian** than ℓ1/peaks. So (a) the raw-high-D
  failure mode (lesson 2) is mild — the flow may take the raw P(k) directly — but VMIM compression to
  ≈ n_params dims is still a clean, near-sufficient reduction; (b) a Gaussian-ish posterior means a
  **MAF may already be near-optimal and the RealNVP edge smaller** — still test both (lesson 1).
- **Log-transform P(k)** before z-scoring (orders-of-magnitude dynamic range) — the `log1p-zscore`
  preprocessing already does the right thing.
- A linear/PCA compression is the classical baseline for P(k); VMIM should match or beat it and, unlike
  PCA, targets the parameter information directly. Worth a head-to-head.
- **Gate exactly the same way** (TARP + SBC). The over-confidence pitfalls (lessons 5–7) are statistic-
  agnostic; the compressor-ensemble fix transfers directly.
- If you have an analytic Gaussian likelihood for P(k), use it as a sanity oracle: the VMIM+NDE
  posterior should match it in the Gaussian regime, which validates the whole pipeline before you push
  to the non-Gaussian regime.

## 6. File pointers (cnn_sbi repo)
- Stage 1 (compressor): `scripts/sbi/vmim_from_cache.py` (clean reference) ; originals
  `scripts/sbi/npe_l1vmim_nbody_tomo.py`, `npe_l1vmim_jaxili_nbody_tomo.py`.
- Stage 2 (NDE): `scripts/sbi/train_nde_from_compressed.py` ; flow defined in
  `scripts/sbi/npe_cnn_nbody_tomo.py` (`build_flow`/`train_flow`).
- Gate: `scripts/sbi/tarp_stratified_val_nde.py`, `run_tarp_coverage.py`,
  `…/analytical_nde_match/gate_verdict.py`.
- Worked examples / orchestration + the calibration-ensemble fix:
  `…/analytical_nde_match/ensemble_eval.py`, `run_calib_sweep_jointl1.py`,
  `run_jointl1_bnt_ensemble.py`; results `RESULT_JOINT_MATCHED.md`, `RESULT_JOINTL1_ENSEMBLE.md`.
- Background on why the estimator/calibration matter: `JOINT_L1_DEFINITION_AND_THEORY.md`,
  memory `project_analytical_matches_cnn_via_nde`, `project_joint_l1_matches_cnn`.
