# Why do harmonic cross-maps help the CNN compressor on tomographic shear?

A working scientific note. Started 2026-05-22 by Andreas + Claude.

## 0. The question

We observe that feeding our CNN-VMIM compressor the 4 tomographic auto-maps
alone gives pooled FoM3 ≈ 11k on the fiducial cosmology, while feeding it the
same auto maps plus 6 harmonic cross-maps (constructed from the cross-power
spectra of the underlying shear fields) gives pooled FoM3 ≈ 24k — a factor
~2.2× improvement.

The intuition that prompts this note: a 2-D CNN should, in principle, be able
to extract cross-bin correlations directly from the stack of auto maps. The
cross-bin information lives in the joint statistics of the 4-channel image,
and a sufficiently expressive convolutional + dense architecture is a
universal function approximator over its input. So the natural prior is:
adding the cross-maps explicitly should add nothing — the CNN should be able
to "see" them already.

That prior is partially correct and mostly wrong. This note explains why.

## 1. Empirical baseline

Pooled FoM3 = 1/√det Cov₃(Ωₘ, σ₈, w₀), 3-seed pool of 100k posterior samples
each.

| pipeline                          | auto-only | auto+cross | gain |
|:----------------------------------|----------:|-----------:|-----:|
| **L1** (wavelet ℓ₁ histogram)     |     9,015 |     33,820 | **3.75×** |
| **CNN-VMIM** (plain CNN, cdim=10) |    11,130 |     23,986 | **2.16×** |

L1 numbers post-noise-fix (see `memory/project_l1_noise_model_correction.md`).
CNN auto-only number from `scripts/sbi/results/exploratory/apples_v_iter108_autoonly/`
(2026-05-22). CNN auto+cross from `iter-108-Q6ON-60k` of the
`cnn-auto-cross-push-18-20-2026` campaign.

The disparity in gain ratios (3.75× vs 2.16×) is the central diagnostic
quantity of this note. L1, by construction, cannot access cross-bin
correlations from the auto maps — it computes a 1-point histogram of wavelet
coefficients channel-by-channel. Cross-bin information is structurally
invisible to it without the cross-maps. The CNN, in contrast, takes the
4-channel image as a single tensor and is *free* to learn any function of it,
including cross-bin moments.

The fact that the CNN's gain ratio is *smaller* than L1's gain ratio is
evidence that the CNN is already recovering some cross-bin information from
the auto maps on its own. The fact that it is non-trivially smaller than 1
(i.e., that the gain is positive) is evidence that the CNN is **not**
recovering all of it.

## 2. The harmonic cross-maps are a globally-constructed object

Before asking why the CNN underperforms, we need to be specific about what
the cross-maps actually are. The construction is documented in
`Harmonic_cross_maps.md`; in summary, for each pair (i, j) of tomographic
bins with i ≤ j:

1. Take the shear field in bin i, project to spherical harmonic coefficients
   $a_{\ell m}^{(i)}$ (full-sky operation).
2. Compute the cross-power spectrum coefficients
   $C_\ell^{ij} = (2\ell + 1)^{-1} \sum_m a_{\ell m}^{(i)} a_{\ell m}^{(j)*}$.
3. Construct a flat-sky cross-map by an inverse-Fourier-style reconstruction
   from the cross-spectrum, with a noise-channel-aware normalization (the
   `channel_empirical_global` model after the 2026-05-15 fix).

This is an **O(N log N) global operation**. Every pixel of the resulting
cross-map depends, in principle, on every pixel of the two input auto-maps.
There is no local kernel of finite support that can reproduce it.

Our plain-CNN trunk has 4 convolutional blocks with kernel size 3 and stride
2 (or pool stride 2), so the receptive field grows roughly as $3 \cdot 2^k$
pixels at depth k — i.e., ~24 pixels at the final conv layer for the 4-block
trunk, on a 160-pixel input. The final dense layer sees the global summary,
but the spatial structure is already aggressively pooled at that point.

This is the architectural mismatch: **the cross-maps are constructed by a
global spectral operation; the CNN trunk processes only local
neighborhoods**. Anything the cross-maps encode that depends on long-range
spatial coupling is, in practice, inaccessible to our CNN.

## 3. Three hypotheses for the residual gap

We propose three mechanistic explanations for why the CNN auto-only result
sits below the auto+cross result.

### H1. Inductive-bias mismatch (architecture, not data)

The CNN's local-convolutional inductive bias means that even with arbitrary
training data, the network cannot economically learn the global spectral
operator that the harmonic cross-maps embody. Adding a global module —
attention over the spatial grid, a learned Fourier transform, or an explicit
FFT layer — should close the gap, even at the current dataset size.

**Falsifier**: a CNN compressor with an explicit global module (FFT layer,
attention, or spectral mixer) trained on auto maps alone should approach or
exceed the auto+cross plain-CNN result.

### H2. Data-limited (capacity, not architecture)

The current dataset (~70k cosmologies in CosmoGridV1) is insufficient for any
CNN family to learn the cross-bin information de novo. Even with the right
architecture, the variance across cosmologies in the cross-bin correlations
is too low-amplitude-per-example to be lifted out by gradient descent on a
finite training set. The harmonic cross-maps work because they short-circuit
the learning: we hand the network the answer to a sub-problem, and the
network doesn't have to extract that signal from data scarcity.

**Falsifier**: train the same plain-CNN on a much larger N-body suite (e.g.,
the CosmoGridV2 successor or a Quijote-scale set) and observe whether
auto-only FoM3 closes the gap to current auto+cross levels.

**Evidence in favor**: the Q1 retest from this week
(`memory/project_resnet50gn_120k_overfits.md`) showed that resnet50_gn
@120k compressor steps on the 4-channel auto-only input overfit by step ~42k
(val-loss argmin), with a 2.24-nat train-val gap by step 120k. This means
the plain-CNN trunk is **not capacity-bottlenecked** — adding parameters
overfits. The bottleneck at our current depth is something else (data,
inductive bias, or compressor).

### H3. Compressor bottleneck (summary dimension, not architecture or data)

The CNN compresses the 4-channel image to a $d$-dimensional summary
(typically d=10 in the cnn-auto-push campaign, d=20 in the May 8 sweeps).
Even if the CNN trunk *did* compute a faithful cross-bin signal, the
compression layer must choose what to encode in d dimensions. With auto maps
alone, the compressor must allocate dimensions to within-bin variance,
between-bin correlation, *and* higher-order statistics. With auto+cross, the
between-bin correlations arrive pre-extracted, freeing summary dimensions
for the rest.

**Falsifier**: a CNN with much larger summary dimension (d=50, d=100) on
auto-only should approach the auto+cross result, *if* the trunk is computing
the cross signal but the bottleneck is throwing it away.

**Evidence against**: the cnn-auto-push campaign swept cdim ∈ {10, 16, 20}
without significant improvement past d=10. This argues against H3 being the
dominant lever — though larger d (d=50, d=100) on a deeper-but-data-limited
architecture has not been tested cleanly.

### Summary table

| hypothesis | dominant effect          | falsifier                                       | evidence so far |
|:-----------|:-------------------------|:------------------------------------------------|:----------------|
| H1         | inductive bias           | global-module CNN on auto-only closes the gap   | **untested**    |
| H2         | dataset size             | bigger N-body suite lifts auto-only             | **untested directly**; deeper CNN overfits → consistent |
| H3         | summary-dim bottleneck   | larger d on auto-only closes the gap            | cdim sweep ∈ {10,16,20} flat → H3 weak           |

The honest summary is that **H1 is the most likely dominant effect, H2 is a
plausible amplifier, and H3 is probably not the load-bearing one**. The
direct test of H1 is what we have not yet run.

## 4. What does this mean about the auto-only Fisher ceiling?

The cnn-auto-push campaign confirmed a ceiling at pooled FoM3 ≈ 25k for the
plain-CNN trunk on auto maps alone (4 independent confirmations: iter-16,
iter-78 cap-match, iter-88 cap-match, Q8 stock-BN). What that ceiling
actually represents is **ambiguous**:

- **Interpretation A**: the ceiling is the cosmological Fisher-information
  limit on (Ωₘ, σ₈, w₀) from the 4 tomographic auto maps at 20 deg² /
  160 px / fiducial noise. In this reading, no architecture can do better on
  auto-only because the information just isn't there.
- **Interpretation B**: the ceiling is the *learnable* ceiling of the
  plain-CNN family at our current dataset size. In this reading, an
  architecture with the right inductive bias (or trained on a larger
  dataset) could close the gap to the auto+cross result.

These two interpretations have very different scientific consequences:

- Under A, the cross-maps add **new** information — they are not derivable
  from the auto maps even by an oracle. Cosmologically, this would mean the
  cross-spectra encode something that the autos do not, which is *not
  generally true at the Fisher level* (the full information of the field is
  in the field itself).
- Under B, the cross-maps **re-package** existing information into a form
  that the CNN can use. Cosmologically, the information is in the autos; we
  just can't get to it with our current architecture.

The 2-point structure of the harmonic cross-maps strongly favors
interpretation B: at the Fisher level, 2-point cross-correlations between
tomographic bins *are* a function of the joint statistics of the auto maps
(in the Gaussian limit, exactly so; in the mildly non-Gaussian regime of weak
lensing at these scales, approximately so). So in principle the auto maps
already contain that information.

The disparity between the L1 gain ratio (3.75×) and the CNN gain ratio
(2.16×) is consistent with interpretation B: the CNN is partially recovering
the cross information from autos, L1 cannot recover any of it. If A were
true, both pipelines would gain the same ratio (the new information is the
same for both, just sitting in the channels they can now see).

## 5. A methodological caveat

The harmonic cross-maps we construct are **2-point statistics** by design:
they encode $C_\ell^{ij}$ between bin pairs. Higher-order cross-information
between bins — cross-bispectra, cross-peak statistics, cross-Minkowski
functionals — is **not** in our cross-maps. So even if a future CNN-on-autos
matched the auto+cross FoM3, that would only mean the CNN extracted the
*2-point* cross-bin information, not the full cross-information content of
the underlying shear field.

This is worth flagging because the natural follow-on question — "what is the
ceiling of a CNN on all the cross-information in the auto maps?" — is
ill-posed without specifying which cross-information. The cleanest
formulation is:

> *Can a CNN with appropriate inductive bias, trained on tomographic auto
> maps alone, recover the 2-point cross-spectrum information that we
> currently expose by handing it the harmonic cross-maps?*

That is a well-posed and falsifiable question.

## 6. Proposed experimental program (test of H1)

Three architectural interventions, each tested on auto-only input, with
auto+cross plain-CNN as the target performance:

1. **Explicit spectral block at input.** Prepend a learned 2D-Fourier-then-
   spectral-mix-then-iFFT block to the CNN trunk. This gives the trunk
   direct access to spatial-frequency content of the input. Lightweight; the
   spectral mix can be a 1×1 conv in Fourier domain. Closest in spirit to
   "give the network an FFT".

2. **Cross-channel attention block.** Replace the first dense layer of the
   trunk (or insert before it) with multi-head attention over the spatial
   grid, allowing every pixel to attend to every other pixel. Standard
   transformer-style. Receptive field becomes global from the first attention
   layer.

3. **Spectral-mixer compressor.** Replace the CNN trunk entirely with an
   MLP-Mixer or FNet variant, where the global mixing operations are baked
   in as architectural primitives.

Primary metric: 3-seed pooled FoM3 on (Ωₘ, σ₈, w₀), auto-only input,
fiducial cosmology, 20 deg / 160 px CosmoGridV1 tomo-4.

Decision rules:
- **If any arm reaches pooled FoM3 ≥ 20k on auto-only** (i.e., ≥ 85% of the
  auto+cross plain-CNN), H1 is confirmed as dominant: inductive bias was the
  load-bearing limit, and architectures *can* close the gap on this dataset.
- **If all arms remain at ≤ 13k on auto-only** (i.e., within ~20% of the
  plain-CNN ceiling), H1 is falsified or H2 dominates: the limit is data,
  not architecture, and we should pivot to a bigger N-body suite test
  (which is more expensive and slower).
- **Intermediate outcomes** (any arm between 13k and 20k) signal partial
  contribution of H1 with H2 plausibly amplifying. Document and stop.

### Diagnostic outputs

Each arm should produce, in addition to the headline FoM3:

- **Per-parameter marginal contours** (truth-bias in σ-units; cf.
  `project_resnet_bn_contamination.md` for the bias-vs-truth diagnostic).
- **Saliency / channel-importance map**: where in the input is the network
  attending? If H1 is correct, a global-aware architecture should show
  long-range spatial dependence.
- **Validation-loss curves** to confirm we are not data-limited (no
  overfitting blow-up at our compressor-step budget).

### What this experiment is NOT

- It is **not** an attempt to push the SBI pipeline's accuracy beyond what
  the cross-maps already deliver. The auto+cross plain-CNN at FoM3 ≈ 24k is
  the *target*, not a baseline to exceed.
- It is **not** a Ralph-loop-style hyperparameter sweep. Each arm should be
  a small, deliberate architectural change with a single seed at first to
  triage, then 3 seeds for the primary metric of any arm that triages
  positive.
- It is **not** a campaign to find the One True Compressor. It is a
  hypothesis test: H1 vs not-H1.

## 7. Bridge to the wider research story

This investigation is interesting beyond cosmology. The pattern "we can
externalize a sub-problem the network is structurally bad at, by computing
that sub-problem explicitly and feeding the result as a channel" is a
recurring move in scientific ML (cf. graph networks for molecular systems,
group-equivariant networks for symmetric domains, etc.). The question of
*when* this is necessary versus *when* the network would have learned it
anyway is itself a research question — and it's exactly the kind of
inductive-bias question that the broader frontier-AI community cares about
under headings like "scaling laws on simulators", "data-efficient
foundations", and "structured priors".

If H1 is confirmed here, the cleanest write-up framing is: *cosmological
inference shows a concrete case where a finite-data CNN has a 2.2× FoM3 gap
that closes with the right inductive bias, not more data*. That has
generality beyond cosmology.

## 8. Open caveats

- We have not run a controlled cdim sweep on the auto-only side; the
  H3-against argument relies on swept results in the auto+cross arm. A clean
  cdim ∈ {10, 20, 50, 100} sweep on auto-only would tighten the bound on H3.
- The plain-CNN trunk's receptive field is not measured exactly — only
  estimated from layer geometry. A direct measurement (e.g., input-gradient
  sensitivity) would sharpen the inductive-bias-mismatch argument.
- The cross-maps' construction uses a `channel_empirical_global` noise
  model. The auto-only baseline does not require any cross-channel noise
  modelling, so the comparison is fair, but we should keep in mind that
  any improvement of the cross-noise model would shift the target.
- "2-point cross-bin information" assumes the shear field is close to
  Gaussian; deviations at small scales mean the harmonic cross-maps don't
  fully capture even the 2-point cross-content non-perturbatively. This is
  a small effect at 20 deg² but worth noting.

## 8b. First experimental result on H1 (added 2026-05-22)

The cross-channel attention arm of the H1 experimental program (§6, arm 2)
ran on auto-only input with config matching iter-108-Q6ON-60k except for the
architecture switch (plain CNN trunk + L=1 transformer block at trunk tail).
**Result: H1 (attention variant) falsified.**

| arm | 3-seed pool FoM3 | pool/MoS haircut | \|bias\| med |
|:---|---:|---:|---:|
| attention (plain trunk + L=1 tail) | 11,892 | 0.684 | 0.50σ |
| plain CNN (anchor)                 | 11,130 | 0.685 | 0.52σ |

The pool/MoS haircut is **identical to 3 decimal places**. Adding ~700k
parameters of global receptive field did not move the dominant failure mode
(seed-to-seed mode drift). Gap closed vs the 24k auto+cross target: +6% —
essentially zero.

**What this updates about the §3 hypothesis ranking:**

- H1 was "most likely dominant"; the attention test is strong Bayesian
  evidence against H1 being the load-bearing limit. *Strict* family-level
  falsification would require testing the spectral-block (§6 arm 1) and
  MLP-Mixer-trunk (§6 arm 3) arms as well, but the haircut-identical
  signal makes those unlikely to flip the verdict.
- H2 (data-limited) is now the most likely standing hypothesis: if no
  architectural intervention changes how seeds disagree on the posterior
  mode, the bottleneck is probably the variance per cosmology in the
  CosmoGridV1 training set, not anything the CNN family can fix.
- H3 (compressor bottleneck) is unchanged: the cdim sweep already weakly
  argued against it; the attention result doesn't bear on cdim directly.

**The Interpretation-A-vs-B framing in §4 should be revisited.** With H1
weakened, Interpretation A (the ceiling IS the auto-only Fisher limit at
this dataset size) becomes more credible. The cross-maps trick may not be
"re-packaging existing information" but rather "providing access to
information that requires more data to learn from autos than we have".

Full writeup of the attention arm:
`scripts/sbi/results/exploratory/h1_inductive_bias/H1_ATTENTION_VERDICT.md`.
Felt campaign: `.felt/cnn-h1-inductive-bias-2026-05/`.

## 8c. Honest revision after Andreas's pushback (2026-05-23)

This section revises §2, §3, and the §8b interpretation after deeper
stress-testing of the two stories we'd been telling. **Both stories were
overstated as I'd framed them.** The corrected picture is fuzzier but more
honest.

### 8c.1. The "multiplicative cross-channel inductive bias" story is weaker than §3.H1 claims

The §3 claim was that a local CNN cannot extract cross-bin information
because the architecture lacks multiplicative cross-channel interactions in
its inductive bias. That overstates the case. Specifically:

- A ReLU/LeakyReLU network *can* approximate the product $xy$. Using the
  polarization identity $xy = \tfrac{1}{4}[(x+y)^2 - (x-y)^2]$ together
  with the fact that ReLU stacks can approximate $x^2$ to arbitrary
  precision, Yarotsky (2017) showed that approximating $xy$ to accuracy
  $\epsilon$ requires $O((\log 1/\epsilon)^2)$ depth. Our 3-conv-layer
  plain trunk is shallow but not zero.
- More accurately, products of input channels are *not in the CNN's
  first-order inductive bias*, so they're harder to learn — but they're
  not architecturally inaccessible. The question is whether the
  optimizer finds these representations under realistic training.
- The H1 attention falsification is consistent with both (i) "the problem
  is expressiveness, but we tested attention in the wrong place" and
  (ii) "the problem is gradient signal / data limit, and attention
  doesn't fix optimization difficulty". The two are not cleanly
  separable from one experiment.

So the corrected H1 story is: **the inductive bias makes cross-bin
features harder to learn than they'd be for an architecture with native
multiplicative interactions, but they're not impossible to learn**. The
gap between "harder to learn" and "actually unlearned in practice" is
filled by the data + optimization regime, which is §3.H2's domain.

This means **H1 and H2 are not cleanly distinct in our setting.** They
overlap heavily. The honest re-ranking: H2 (data + optimization limit
modulated by the H1-style inductive-bias difficulty) is dominant; pure
H1 (architecture is the load-bearing limit) is unlikely.

### 8c.2. The patch-boundary story is stronger than §8b claims (the spherical case is structurally not local)

I'd estimated the patch-boundary effect at "10–20% at the patch edges,
not dominant" assuming the cross-maps behave like a flat-sky convolution
that's effectively local at the cross-correlation scale (~1–2 deg). That
estimate is the **flat-sky** answer. The actual procedure in this project
is the **spherical** one (§2 in `Harmonic_cross_maps.md`), and it is
structurally different.

#### What the flat-sky procedure does (Flat-Sky_Tomographic_Cross_Maps.md)

By the convolution theorem on a flat 2D grid, pointwise multiplication of
FFT coefficients is exactly equivalent to a real-space convolution:

$$\kappa^{ij}(\mathbf{x}) = \mathcal{F}^{-1}\{\tilde{\kappa}^i \cdot \tilde{\kappa}^j\} = (\kappa^i \ast \kappa^j)(\mathbf{x})$$

This convolution kernel is $\kappa^j$ itself, which falls off with $\kappa^j$'s spatial
correlation length (a few degrees for tomographic shear). So the
flat-sky cross-map at a patch interior pixel is mostly determined by
$\kappa^i$ and $\kappa^j$ values within a few degrees of that pixel — *inside* the
patch, except for an edge zone of ~2 deg width.

#### What the spherical procedure actually does (Harmonic_cross_maps.md)

The construction is: take spherical-harmonic transforms of both maps,
multiply the coefficients $a_{\ell m}^i$ and $a_{\ell m}^j$ **element-wise**, then
inverse-transform back. The resulting cross-map value at a pixel $\mathbf{x}$ is

$$\kappa^{ij}(\mathbf{x}) = \sum_{\ell m} a^i_{\ell m}\, a^j_{\ell m}\, Y_{\ell m}(\mathbf{x})$$

Substituting $a_{\ell m}^i = \int Y_{\ell m}^*(\mathbf{y})\,\kappa^i(\mathbf{y})\,d\mathbf{y}$
and similarly for $a_{\ell m}^j$:

$$\kappa^{ij}(\mathbf{x}) = \int\!\!\!\int K(\mathbf{x}, \mathbf{y}, \mathbf{z})\, \kappa^i(\mathbf{y})\, \kappa^j(\mathbf{z})\, d\mathbf{y}\,d\mathbf{z}$$

where the **three-point kernel** is
$K(\mathbf{x}, \mathbf{y}, \mathbf{z}) = \sum_{\ell m} Y_{\ell m}(\mathbf{x})\, Y_{\ell m}^*(\mathbf{y})\, Y_{\ell m}^*(\mathbf{z})$.

This is *not* a spherical convolution by a fixed kernel — the spherical
convolution theorem applies only to convolutions with **rotationally-
invariant** kernels. The cross-map operation is a more general bilinear
operator on $(\kappa^i, \kappa^j)$, and its kernel $K(\mathbf{x}, \mathbf{y}, \mathbf{z})$ is the
Gaunt-type 3-point function on the sphere. It has support on the
**whole sphere × whole sphere**, weighted by the angular geometry of the
triangle $(\mathbf{x}, \mathbf{y}, \mathbf{z})$.

#### What this means for the CNN

For *every* patch pixel $\mathbf{x}$ — interior or edge — the cross-map value
$\kappa^{ij}(\mathbf{x})$ is a quadratic functional of $\kappa^i$ and $\kappa^j$
**evaluated across the whole sphere**, not just locally. The CNN seeing
only auto-patches has the values of $\kappa^i$ inside the patch only, and
therefore *cannot in principle reconstruct* $\kappa^{ij}(\mathbf{x})$ — even at
interior pixels far from any patch boundary. The information about
$\kappa^i$ and $\kappa^j$ *outside* the patch is genuinely missing.

The size of this effect depends on how concentrated the Gaunt kernel
$K(\mathbf{x}, \mathbf{y}, \mathbf{z})$ is around small triangles (i.e., $\mathbf{y}, \mathbf{z}$ near $\mathbf{x}$).
For low-$\ell$ modes the kernel tends to favor compact triangles, so the
dominant contribution to $\kappa^{ij}(\mathbf{x})$ does come from "nearby" $\kappa^i, \kappa^j$
values. But the long tails of the kernel mean there is *always* a
non-trivial contribution from far-away points — and these are exactly
the ones the CNN cannot see.

**This is bigger than I claimed in §8b.** I'd estimated 10–20% of the
gap. Honestly, without an empirical test, I don't know the magnitude.
It could be 20%; it could be 50% or more. The structural argument
("every patch pixel encodes some global information that the auto patches
can't") is robust; the quantitative estimate is not.

### 8c.3. The cleanest empirical test for §8c.2

Run the same CNN with cross-maps constructed by the **flat-sky procedure**
(`Flat-Sky_Tomographic_Cross_Maps.md`) instead of the spherical procedure.
The flat-sky procedure produces cross-maps that are *true real-space
convolutions* of the auto-patches with each other, so they encode only
information from within the patch (plus small apodization-window edge
effects).

If flat-sky cross-maps lift CNN auto-only FoM3 by roughly the same factor
as the spherical cross-maps, the global-information contribution to the
gap is small — the dominant effect is the optimization/data story (H2)
and the cross-maps just help by pre-extracting locally-derivable
information.

If flat-sky cross-maps lift FoM3 *significantly less* than the spherical
ones, the global-information contribution is substantial — the
spherical cross-maps are genuinely adding Fisher information that the
auto-patches do not contain.

This is almost free: no architecture changes, no new training of an
unfamiliar model — just a different cross-cache construction. The
cross-cache builder already supports flat-sky variants via
`compute_flat_cross_map` (see `Flat-Sky_Tomographic_Cross_Maps.md`).
If we ever want to actually disambiguate the "is the spherical procedure
adding genuinely new info" question, this is the experiment.

### 8c.4. Revised hypothesis ranking (supersedes §3 and §8b)

| hypothesis | status after 8c | dominance |
|:---|:---|:---|
| H1 (inductive bias, strict)              | weakened — products are learnable, just harder; H1 attention falsification is consistent with both architecture-limit and data-limit readings | **likely not dominant** |
| H2 (data + optimization limit)           | likely dominant under the corrected reading; H1's "harder to learn" plus VMIM gradient-signal weakness for cross-bin features explains the gap | **likely dominant** |
| H3 (compressor bottleneck)               | **falsified 2026-05-24** at the standard NDE config: cdim=10 → cdim=100 cratered pooled FoM3 by ~49% (24.0k → 12.2k), with well-calibrated but flatter posteriors. NDE underprovisioned for 100-d conditioning. See `scripts/sbi/results/exploratory/h3_cdim_sweep/H3_CDIM100_VERDICT.md`. | **not dominant** |
| **Global-info via spherical procedure** *(new, was §8b's "small effect")* | strengthened — every patch pixel encodes global info the CNN cannot reconstruct; magnitude unclear without flat-sky-vs-spherical comparison | **plausibly load-bearing** |

So the corrected leading hypotheses are H2 and the global-info-via-
spherical-procedure effect, possibly in combination. The "the cross-
maps are mostly re-packaging existing information into a more
accessible form" (Interpretation B from §4) is still on the table, but
Interpretation A ("the cross-maps add genuinely new information") is
substantially more credible than it was in §8b's reading.

**Update 2026-05-24 (H3 falsification)**: pushing summary dimensionality
from 10 to 100 at the standard NDE config cratered pooled FoM3 by ~49%
(24.0k → 12.2k). The compressor's VMIM val-loss *was* tighter at
cdim=100, but the RealNVP NDE was underprovisioned for the higher-dim
conditioning and produced flatter (well-calibrated but uninformative)
posteriors. This decisively closes the "we just needed more summary
dimensions" line — at the pipeline's current configuration, cdim=10
is essentially near-optimal. Strict claim: a separately co-tuned
higher-capacity NDE at cdim=100 might recover the cdim=10 performance,
but the spirit of the question is answered. Removes H3 from the
running entirely.

### 8c.5. Confidence ledger

- **High confidence**: my original "global Fourier ops are out of CNN's
  reach" framing (§3.H1 first paragraph) was wrong as stated.
- **High confidence**: the spherical cross-map procedure is structurally
  non-local — every patch pixel depends on $\kappa^i, \kappa^j$ values across the
  whole sphere via a Gaunt-type kernel.
- **Medium confidence**: H2 (data + optimization, modulated by inductive-
  bias difficulty) is the dominant explanation for the gap.
- **Low-to-medium confidence**: the global-info-via-spherical-procedure
  effect is large in magnitude — possibly 20–50%, possibly more,
  possibly less. Needs the flat-sky-vs-spherical empirical comparison
  to bound.
- **Low confidence**: any single-line summary of "the reason the gap
  exists" without acknowledging it's a combination of effects.

## 9. Pointers

- L1 noise-model correction: `memory/project_l1_noise_model_correction.md`.
- Plain-CNN ceiling (4-confirmation): `memory/project_resnet50gn_120k_overfits.md`,
  `memory/project_resnet_bn_contamination.md`, and the closed fibers under
  `.felt/cnn-auto-push-18-20-2026/`.
- Cross-maps construction: `Harmonic_cross_maps.md`,
  `Flat-Sky_Tomographic_Cross_Maps.md`.
- Pipeline architecture: `SBI_L1_CNN_PIPELINE_DETAILED.md`, `CLAUDE.md`
  §"Pipeline architecture".
- This note's empirical baseline:
  `scripts/sbi/results/exploratory/apples_v_iter108_autoonly/` (auto-only CNN),
  `/nas/tersenov/claude-notes/runs/cnn-auto-cross-push-18-20-2026/iter-108-Q6ON-60k/`
  (auto+cross CNN), and L1 numbers per the noise-correction memory.
