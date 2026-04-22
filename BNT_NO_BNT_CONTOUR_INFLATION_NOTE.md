# Why BNT can still inflate contours (and what to do about it)

## Executive summary

The small BNT/no-BNT contour mismatch is **not** evidence that BNT is fundamentally unusable. It is most likely a finite-training, finite-capacity effect of the learned compressor + NDE pipeline.

In the ideal limit (exactly invertible transform, no lossy bottleneck, infinite data/optimization, exact posterior model), BNT and no-BNT should agree. In practice, we compress to low dimension, train with noisy stochastic objectives, and apply BNT after shape-noise injection; this makes exact parity hard.

## Why an inverse-BNT-like first layer is not guaranteed to emerge

The intuition is good: if \(B\) is the BNT map, a first layer could in principle learn something close to \(B^{-1}\).

However, three practical issues prevent guaranteed exact cancellation:

1. **Lossy bottleneck**: the compressor keeps only a few summary dimensions, so it cannot preserve all transformed/noise-reweighted information.
2. **Optimization/estimation error**: VMIM and NPE training are approximate (finite simulations, finite steps, regularization, early stopping).
3. **Noise conditioning**: because noise is injected before BNT in this pipeline, BNT changes the effective noise covariance across modes; a strict inverse can re-amplify low-SNR directions, which VMIM may avoid in favor of robust summaries.

So yes: this is largely a **practical compressor/inference inefficiency** issue, not a contradiction of SBI principles.

## Can we get perfect agreement?

In finite compute/data settings, expecting mathematically perfect overlap is unrealistic. But we can get much closer by explicitly enforcing BNT/no-BNT invariance rather than hoping it emerges implicitly.

## Bibliography-backed techniques that are most relevant

### 1) Use information-preserving compression objectives

- Increase summary dimension and regularize, then ablate downward.
- Use Fisher-/information-maximizing compression ideas (IMNN-style) as a direct check that summaries are near-sufficient.

Why this helps: if compression is near-lossless for cosmological parameters, transform-choice sensitivity drops.

### 2) Add explicit BNT/no-BNT consistency constraints

Train on paired simulations with the same underlying map/seed and penalize disagreement between summaries or posteriors:

\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{VMIM}} + \lambda \|s_{\text{BNT}} - s_{\text{noBNT}}\|_2^2
\]

Optionally add posterior-level consistency (e.g., KL between predicted posteriors for paired views).

Why this helps: enforces transform-invariance directly, not indirectly.

### 3) Domain-adversarial invariance to transform label

Attach a small head that predicts whether input is BNT/no-BNT; reverse gradients so the summary cannot encode that label while still predicting cosmology.

Why this helps: classic way to remove nuisance-domain information from learned features.

### 4) Architecturally precondition the first layer with known linear operators

Initialize the first linear block with \(B^{-1}\) (or a regularized pseudo-inverse), then either:
- keep it fixed for a warm-up phase, or
- fine-tune with spectral norm / conditioning penalties.

Why this helps: gives the optimizer a physically informed start rather than requiring it to rediscover inversion from scratch.

### 5) Strengthen NDE robustness (not just compressor robustness)

Even with good summaries, posterior mismatch can come from density-estimator misspecification. Use:
- higher-capacity flows / ensembles,
- calibration checks (coverage/SBC),
- paired-view posterior consistency penalties.

## Practical recommendation for this project

For this codebase, the highest-yield next experiment is:

1. Keep current best plain/resnet setup.
2. Add paired BNT/no-BNT training batches.
3. Add a summary consistency term + weak domain-adversarial head.
4. Run parity-focused model selection on `abs(fom3_ratio_bnt_over_nobnt - 1)`.

This directly targets the failure mode (transform-dependent summaries) rather than only changing schedule length or architecture depth.

## References (good starting points for paper bibliography)

1. **Heavens, Jimenez, Lahav (2000)** — *Massive Lossless Data Compression and Multiple Parameter Estimation from Galaxy Spectra* (MOPED).  
   arXiv:astro-ph/9911102, doi:10.1046/j.1365-8711.2000.03692.x

2. **Charnock, Lavaux, Wandelt (2018)** — *Automatic physical inference with information maximising neural networks*.  
   arXiv:1802.03537

3. **Makinen, Charnock, Alsing, Wandelt (2021)** — *Lossless, Scalable Implicit Likelihood Inference for Cosmological Fields*.  
   arXiv:2107.07405, doi:10.1088/1475-7516/2021/11/049

4. **Alsing, Wandelt, Feeney (2018)** — *Massive optimal data compression and density estimation for scalable, likelihood-free inference in cosmology*.  
   arXiv:1801.01497, doi:10.1093/mnras/sty819

5. **Alsing, Charnock, Feeney, Wandelt (2019)** — *Fast likelihood-free cosmology with neural density estimators and active learning*.  
   arXiv:1903.00007, doi:10.1093/mnras/stz1960

6. **Cranmer, Brehmer, Louppe (2019/2020)** — *The frontier of simulation-based inference*.  
   arXiv:1911.01429, doi:10.1073/pnas.1912789117

7. **Bernardeau, Nishimichi, Taruya (2013/2014)** — *Cosmic shear full nulling: sorting out dynamics, geometry and systematics* (BNT nulling context).  
   arXiv:1312.0430, doi:10.1093/mnras/stu1861

8. **Dinh, Sohl-Dickstein, Bengio (2016/2017)** — *Density estimation using Real NVP*.  
   arXiv:1605.08803

9. **Ganin et al. (2015/2016)** — *Domain-Adversarial Training of Neural Networks*.  
   arXiv:1505.07818, JMLR 17.
