# Harmonic Cross-Maps: L1 vs CNN Compressor Comparison — Investigation Notes

These notes document the investigation of the L1-vs-CNN performance gap on
harmonic cross-maps data, including the TARP joint-coverage refinement and a
detailed analysis of preprocessing equivalences between the two pipelines.
They are intended as paper-ready material to be folded into the methods and
discussion sections at the end of the project.

## 1. Setup and the empirical puzzle

We compare two compressor families on tomographic convergence maps drawn from
CosmoGridV1 (no-BNT regime, 4 redshift bins, 20°/160 px patches):

- **L1**: starlet wavelet decomposition followed by SNR-binned L1 norms
  (`wl_stats_torch.WLStatistics`). Deterministic featurization producing ~860
  features per realization after zero-variance filtering (raw dim 2000).
- **CNN-VMIM**: ResNet50-GN (or "plain" CNN) trained jointly with a
  normalizing flow under the VMIM objective, with a 10-dimensional summary
  output.

Both summaries are fed to the same `jaxili` NPE flow to infer
`θ = [Ω_m, σ_8, w_0, h_0, n_s, Ω_b]`. The metric of interest is
`FoM3 = 1/√det(C_3)` over the marginal covariance of `(Ω_m, σ_8, w_0)`.

Harmonic cross channels are built à la Zürcher 2022: element-wise alm products
`c_{ij}(ℓ,m) = a_i(ℓ,m) · a_j(ℓ,m)` for the six bin pairs `(i,j) ∈ {(1,2),
(1,3), (1,4), (2,3), (2,4), (3,4)}`, inverse SHT at NSIDE=512, lmax=1024,
gnomonic projection to 48 non-overlapping patches per realization, per-patch
demeaning. Shape noise is injected before the SHT and is shared bit-for-bit
between the two pipelines (both read the same cached `.npz` files written by
`build_full_sphere_cross_cache.py`).

Two configurations are compared:

| input channels | L1 FoM3 (median over seeds) | CNN FoM3 (median over seeds) |
|---|---:|---:|
| 4 tomographic autos | ~13.1 k | ~13.1 k |
| 4 autos + 6 harm-crosses | **~65 k** | ~17–25 k |

The auto-only contours agree in shape and size to within seed scatter; the
auto+cross result has L1 producing contours ~2.5× tighter than the best CNN
configuration. This is anomalous because the CNN-VMIM is, in principle, a
strictly more general compressor than the L1 wavelet histogram — a sufficiently
expressive CNN with sufficient training data should reproduce any fixed
featurization L1 computes.

## 2. TARP joint-coverage refinement

A natural first concern is that L1's tighter contours may be artefactual —
posterior overconfidence rather than real information gain. Marginal-rank
simulation-based calibration (SBC) on the L1 harm-cross arm (N=1000, M=2000)
reported the σ_8 marginal at z ≈ −1.78 and a U-shaped Ω_b distribution with
χ² ≈ 117 (1-D), consistent with marginal miscalibration. Marginal SBC, however,
does not test joint contour size.

We therefore ran a TARP (Lemos et al. 2023) joint-coverage campaign across 17
(arm, seed) combinations — L1 harm_cross (6 seeds), CNN auto-only (5 seeds),
CNN harm_cross plain (3 seeds), CNN harm_cross resnet50_gn (3 seeds) — with
N=200 cosmologies, M=2000 posterior samples each, bootstrap with 200 iterations.

Expected coverage probability (ECP) in the 3-D `(Ω_m, σ_8, w_0)` subspace, the
same subspace as FoM3:

| arm | α=0.50 | α=0.68 | α=0.90 | α=0.95 |
|---|---:|---:|---:|---:|
| cnn_auto_only       | 0.519 | 0.700 | 0.910 | 0.955 |
| cnn_harm_cross_gn   | 0.514 | 0.674 | 0.894 | 0.958 |
| cnn_harm_cross_plain| 0.518 | 0.684 | 0.915 | 0.955 |
| **l1_harm_cross**   | 0.497 | 0.684 | **0.884** | **0.940** |

L1 harm_cross is under-covered in 3-D by only 1–2 percentage points at the
90% and 95% credibility levels. The marginal SBC miscalibrations are real but
cancel almost entirely in the joint covariance determinant: they affect *which*
cosmologies the posterior locks onto more than *how large* the joint contour is.
We therefore interpret the ~2.5× FoM3 advantage as **mostly real signal**, with
a coverage caveat at the few-percent level. The σ_8 marginal still requires
care for any single-parameter scientific claim, but the joint contour volume
is defensible to within ~5%.

## 3. Pipeline asymmetries: what is and is not different

Given that L1's gain is mostly real, the question becomes: what does L1's
featurization extract from harm-cross maps that the CNN-VMIM does not? We
audited every step of the data path on both pipelines.

### 3.1 Per-channel RMS rescaling on the CNN side

The CNN harm-cross route divides each of the 10 input channels by a *global*
training-set RMS computed once over the train split
(`compute_harmonic_channel_rms` at `scripts/sbi/npe_cnn_nbody_tomo.py:783`).
The docstring records the design intent: "Using RMS rather than per-example std
ensures inter-example cosmological amplitude variation is preserved."

This is a constant linear divisor per channel, applied uniformly across all
cosmologies. It is invertible by a single rescaling of the first-layer
convolution weights and is therefore information-preserving with respect to
any compressor that includes a learnable first layer. It is not the source of
the gap.

### 3.2 Summary dimensionality bottleneck

L1 feeds the flow with ~800 features (auto-only) or 860 features (auto+cross,
after zero-variance filter from raw 2000); the CNN-VMIM output dim is 10 in
both regimes. The ratio is ~80× in both cases. If the dimensional bottleneck
were the differential explanation, the CNN would already underperform on
auto-only — it does not. The bottleneck may contribute as a secondary factor
in conjunction with a harder optimization landscape, but it cannot account
for the auto-only/auto+cross asymmetry on its own.

### 3.3 Coarse-scale mean subtraction in the L1 wavelet transform

The L1 pipeline calls `WLStatistics.compute_wavelet_transform(...,
subtract_coarse_mean=True)`. By name this might appear to be an extra
preprocessing the CNN lacks. The algebra resolves it cleanly.

The starlet decomposition (à-trous, B3-spline) writes any input image as

```
x = w_0 + w_1 + ... + w_{J-1} + c_J,
```

where `w_j` are detail bands at scale `j` and `c_J` is the coarsest
approximation. The B3-spline low-pass filter `h` is mass-preserving (∑h = 1),
which implies two identities for arbitrary input `x`:

```
mean(c_J) = mean(c_{J-1}) = ... = mean(x)        (DC passes through the low-pass unchanged)
mean(w_j) = 0   for every detail band j           (details are band-pass; no DC content)
```

The flag implementation is `c_J ← c_J − mean(c_J)` (only the coarsest band is
modified; the detail bands are untouched). By the first identity, this is
arithmetically equivalent to subtracting `mean(x)` from the input *before*
the wavelet transform. Because of the second identity, the detail bands are
*bit-identical* with or without the flag — the L1 statistics computed
afterwards see the same numbers.

In the harm-cross runs, both pipelines consume cached patches with
`mean(x) = 0` enforced at cache-build time
(`build_full_sphere_cross_cache.py:198`). The CNN's harmonic loader asserts
this via `_assert_zero_mean_patches`. So `mean(c_J) ≈ 0` to float32 precision,
and the L1 subtraction reduces to subtracting a ~1e-7 residual — effectively
a no-op. The CNN's `--zero-mean-maps` augmentation flag is similarly a no-op
when feeding from this cache.

**Conclusion**: the L1 coarse-mean step and the CNN map-mean centering are
algebraically equivalent operations — both remove exactly one Fourier mode
(the DC) — and both are no-ops on the harm-cross cache, which already has DC
removed per (patch, channel). This is therefore not an asymmetry source
between the two compressors.

### 3.4 Cross-map–specific note on the DC mode

The harm-cross channels carry a subtlety on which it is worth being explicit,
even though it does not generate L1–CNN asymmetry. For an auto convergence
map `κ_i`, the global spherical mean is zero by construction (`κ` has no
monopole on the sphere), so the per-patch mean is a noise/sample-variance
realization of zero — cosmologically empty. Discarding it via per-patch
demeaning costs nothing.

For a harm-cross channel `c_{ij}(x) = ifft(a_i · a_j)`, the global spherical
mean is *also* zero (`a_i(0,0) = a_j(0,0) = 0`), but the *per-patch* mean is
not. Integrating `c_{ij}` against a patch window picks up low-ℓ modes of the
alm-product field at ℓ ≲ ℓ_patch (roughly ℓ ≲ 9 for 20° patches). This low-ℓ
tail does carry cosmology, tied to the inter-bin cross-power at large scales.
Per-patch demeaning of the harm-cross cache discards it.

The cost is symmetric: both L1 (via cache demean and the redundant
`subtract_coarse_mean`) and CNN (via cache demean) pay it equally. It can
bias the absolute harm-cross FoM3 downward but cannot generate asymmetry
between the two compressors. A separate, future experiment dropping the
per-patch demean for the cross channels would test whether this large-scale
cross-correlation information is in principle recoverable; it is orthogonal to
the present comparison.

### 3.5 Other asymmetries that were considered and ruled out

- **Cache-level apodization**: verified absent. The only operation between
  gnomonic projection and writing the npz is per-patch demean.
- **Different noise realizations**: shape noise is injected before SHT in the
  cache builder with a seed derived from `(cosmo_idx, perm)`; both pipelines
  consume identical noisy bytes.
- **Different number of training examples**: both pipelines iterate the same
  cached files and produce one summary per patch (48 patches per realization).
- **TFDS vs harmonic cache discrepancy**: the harm-cross runs of both
  pipelines read from `full_sphere_cache_grid/`, not from TFDS. The L1 cross
  runner does *also* contain a TFDS-based flat-sky cross route
  (`_compute_cross_maps_tf` in `build_augmentation`), but it is unused by the
  campaign we are comparing.

## 4. Surviving hypotheses for the L1-vs-CNN gap

With the preprocessing asymmetries ruled out, the gap must originate in the
compressor itself. Three non-exclusive hypotheses survive:

**H1. Inductive-bias mismatch.** Starlet + SNR-binned L1 norms are
purpose-built to histogram heavy-tailed, sparse, multi-scale signal.
Harm-cross maps `c_{ij}(x) = ifft(a_i · a_j)` are products of near-Gaussian
fields and therefore have non-Gaussian heavy-tailed pixel distributions across
scales — exactly the regime in which wavelet L1 thresholding is most
informative. The CNN-VMIM must discover this structure from data, without an
architectural prior; with finite training data it may converge to a
representation that captures the Gaussian (mid-scale variance) content
cleanly but under-extracts the sparse non-Gaussian content.

**H2. VMIM optimization non-convexity.** Under the same architecture and
training budget, VMIM converges close to a sufficient statistic on the
auto-only landscape (matching L1). On the 10-channel cross input the loss
landscape changes and the optimizer may dead-end at a local minimum that
prioritizes the easy auto-channel signal and fails to extract cross-channel
signal at the available training budget.

**H3. Receptive-field / scale mismatch.** The CNN's strided-conv stack
determines what spatial scales it can integrate over. The cross-power between
bins peaks at specific ℓ values that may not align with the CNN's effective
scale support, whereas starlets cover decades of scale by construction.

These hypotheses predict different mitigations: H1 requires giving the CNN
explicit multi-scale or wavelet-style inductive bias (e.g., feeding wavelet
bands as input channels or replacing early conv stages with a fixed
multi-scale stem); H2 benefits from longer training, a larger compressor-dim,
better initialization, or a different objective; H3 benefits from
architectural changes (dilated convs, multi-scale skips, attention over
scales).

## 5. The cross-only test

To disambiguate H1–H3 from the alternative "L1 better squeezes the auto
channels when cross channels are present", we run a controlled experiment in
which both compressors are trained on the **6 cross channels only**, with the
4 auto channels dropped, holding everything else fixed (same cache, same
training budget, same seeds, same flow).

Decision rule:

- If L1 cross-only ≫ CNN cross-only in FoM3, the cross channels carry
  information that the CNN-VMIM systematically fails to extract; the gap is
  architecture- or objective-driven (consistent with some combination of
  H1–H3).
- If L1 cross-only ≈ CNN cross-only, the auto+cross gap is not about the
  cross channels per se but about L1 leveraging the auto channels more
  effectively when paired with the cross channels — a different question.
- If both pipelines yield weak cross-only contours, the cross channels add
  little standalone information; the harm-cross gain is a joint-information
  effect with the autos.

The detailed plan for this experiment is tracked separately.

## Appendix: code references

- L1 cross runner — `scripts/sbi/npe_l1norm_cross_jaxili_nbody_tomo.py`
- CNN runner — `scripts/sbi/npe_cnn_nbody_tomo.py`
- Harmonic cache builder — `scripts/sbi/build_full_sphere_cross_cache.py`
- WLStatistics source — `wl_stats_torch/wl_stats_torch/statistics.py`
- TARP infrastructure — `scripts/sbi/run_tarp_dumps_campaign.py`,
  `scripts/sbi/run_tarp_coverage.py`
- TARP campaign output — `scripts/sbi/results/diagnostics/tarp_harm_cross/`
  (per-seed dumps, curves, figures `tarp_per_arm_dim{3,6}.pdf`,
  `tarp_overlay_dim{3,6}.pdf`)
