---
name: Test H1 — does inductive bias close the CNN auto-only / auto+cross gap?
status: closed
tags:
    - autoresearch
    - sbi
    - cnn
    - weak-lensing
    - inductive-bias
created-at: 2026-05-22T20:28:46.976675571Z
closed-at: 2026-05-23T08:49:25.359538122Z
outcome: 'H1 (inductive-bias-via-attention) FALSIFIED at the chosen implementation (2026-05-22). Cross-channel attention block (L=1, H=4, MLP×4, tail insertion at 400-token grid) trained on auto-only matches plain-CNN: 3-seed pool = 11,892 vs anchor 11,130, pool/MoS haircut IDENTICAL to 3 decimal places (0.684 vs 0.685). Adding ~700k params of global receptive field did NOT move the dominant failure mode (seed-to-seed mode drift). Gap closed vs 24k auto+cross target: +6% — essentially zero. Strong Bayesian evidence H1 not load-bearing on this dataset. Caveats: only tail-attention tested; spectral-block and MLP-Mixer-trunk arms not run (but unlikely to flip the verdict given haircut signal). Interpretation A from CNN_CROSS_MAPS_INFORMATION_NOTE gains weight: the auto-only ceiling may be close to the Fisher limit AT THIS DATASET SIZE. Pivoting to H2 (data-limited test) — see [[cnn-h2-data-limit-scoping-2026-05]] for scoping. Exit interview in constitution body. Writeup: scripts/sbi/results/exploratory/h1_inductive_bias/H1_ATTENTION_VERDICT.md . Updated reference note: CNN_CROSS_MAPS_INFORMATION_NOTE.md §8b.'
---

## Objective

Test H1 from CNN_CROSS_MAPS_INFORMATION_NOTE: the plain-CNN's 2.16× gap
between auto-only (FoM3 ≈ 11.1k) and auto+cross (FoM3 ≈ 24.0k) is dominantly
an *inductive-bias* limit of the local-convolutional trunk, not a Fisher
limit of the auto maps. Falsifier: a compressor with explicit global mixing
operators, trained on auto-only input, should close the gap.

Three architectural arms originally proposed: (a) explicit spectral block at
input, (b) cross-channel attention block at the trunk's tail, (c) MLP-Mixer /
FNet trunk replacement. **Andreas selected arm (b) as the first triage**
(2026-05-22).

## Primary metric (declared per CLAUDE.md §"Felt / Ralph operating conventions" #1)

**3-seed pooled FoM3 = 1/√det Cov₃(Ωₘ, σ₈, w₀)** on the fiducial cosmology,
auto-only 4-channel input, 20 deg / 160 px CosmoGridV1
(`NbodyCosmogridDatasetTomo/grid_20deg_160px`), tomographic bins {1,2,3,4}.

Anchors for comparison (same metric):
- Plain-CNN auto-only:    **11,130** (apples-to-apples, this week)
- Plain-CNN auto+cross:   **23,986** (iter-108-Q6ON-60k cross-push best)
- L1 auto-only:            9,015
- L1 auto+cross:          33,820 (post-noise-fix v2_chsigma)

## Decision rule

| auto-only pooled FoM3 | verdict on H1                        | next move |
|----------------------:|:-------------------------------------|:----------|
| **≥ 20,000**          | H1 confirmed (≥85% of auto+cross)    | document, write up, optionally extend to other arms |
| **13,000 – 20,000**   | partial — H1 contributes, but other limits also matter | document, stop, consider H2 follow-up |
| **≤ 13,000**          | H1 falsified at this dataset size    | close campaign; open H2 workstream (bigger sim suite) |

For single-seed triage, intermediate threshold for promoting to 3-seed:
**single-seed FoM3 ≥ 13k** (above the ~30% per-seed dispersion observed in
plain-CNN auto-only).

## Budget + plateau-stop (CLAUDE.md convention #2)

- **Budget M**: 12 iterations across all arms (each iteration = one
  single-seed run OR one 3-seed batch).
- **Plateau-stop N=2, X=5%**: if 2 consecutive iters land within ±5% of the
  current best primary metric, auto-close.
- **Per-arm triage**: max 2 single-seed pilots per arm before deciding to
  commit to 3-seed.

## Loop Status (live)

**Arm 1 (attention) FALSIFIED 2026-05-22.** 3-seed pool = 11,892 vs plain-CNN
anchor 11,130 (Δ = +6%, decision rule says ≤ 13k → falsified). Pool/MoS
haircut identical to plain-CNN at 0.684 vs 0.685. Strong evidence H1
(inductive bias) is not the load-bearing limit on this dataset.

**Currently waiting on**:
- Andreas's call: (i) close the campaign as H1 falsified, (ii) escalate to
  H2 (data-limited test on a bigger sim suite), or (iii) test one more H1
  arm before declaring the family dead (cheapest: explicit spectral block at
  input).

No autonomous-loop iterations should fire until Andreas decides.

## First arm: cross-channel attention block — design proposal

### Insertion point

Insert after the plain-CNN trunk's last conv stage, before the dense
compressor head ("Option A" in the menu — minimal architectural delta,
maximally diagnostic).

```
Input (B, 160, 160, 4)
  → ConvBlock1 (160→80, C=64,  3×3, stride 2, leaky_relu)
  → ConvBlock2 (80→40,  C=128, 3×3, stride 2, leaky_relu)
  → ConvBlock3 (40→20,  C=256, 3×3, stride 2, leaky_relu)
  → [NEW] flatten to (B, 400 tokens, 256)
  → [NEW] + learned positional embedding (400, 256)
  → [NEW] L × Transformer block (pre-LN):
       - LayerNorm
       - Multi-Head Self-Attention (H=4, d_k=64)
       - residual
       - LayerNorm
       - MLP (256 → 1024 → 256, GeLU)
       - residual
  → mean-pool over 400 tokens → (B, 256)
  → Dense(256, leaky_relu) → Dense(cdim=10) → summary
```

Matches iter-108-Q6ON-60k conv trunk exactly (3 stages, 64/128/256
channels). Attention block REPLACES the existing AvgPool + Flatten, so
spatial mixing goes through attention rather than averaging. The dense
head is unchanged.

### Hyperparameter defaults (first pilot)

| param            | default | rationale |
|:-----------------|:--------|:----------|
| L (attn layers)  | 1       | minimum to be diagnostic; cheap |
| H (heads)        | 4       | standard; d_k = 64 with C=256 |
| MLP multiplier   | 4×      | standard transformer |
| Positional emb   | learned | required (attn is permutation-invariant) |
| Norm placement   | pre-LN  | stability standard |
| Dropout          | 0.0     | match plain-CNN trunk |
| Compressor steps | 60,000  | match iter-108-Q6ON-60k (auto-only and auto+cross anchors both used this) |
| LR               | 5×10⁻⁴  | match iter-108-Q6ON-60k |
| Compressor dim   | 10      | match iter-108-Q6ON-60k |
| Batch size       | 128     | match iter-108-Q6ON-60k compressor batch |
| NDE batch        | 256     | match iter-108-Q6ON-60k |
| `--zero-mean-maps` | on    | mandatory (project rule, mass-sheet leak) |
| `--standardize-summary` | on | match iter-108-Q6ON-60k |

### Parameter count budget (revised for 400-token grid)

Approximate added parameters for L=1:
- Pos embed: 400 × 256 ≈ 1.0 × 10⁵
- MHA (Q/K/V/O projections): 4 × 256² ≈ 2.6 × 10⁵
- MLP (256→1024→256): 2 × 256 × 1024 ≈ 5.2 × 10⁵
- LayerNorm + biases: negligible

Total new parameters ≈ **8.8 × 10⁵**. The plain-CNN trunk + dense head is
roughly 1.2 × 10⁶ parameters at this config. Combined ≈ 2.1 × 10⁶, still
well below the ~25 × 10⁶ of resnet50 where data-limit overfitting bit us
(project_resnet50gn_120k_overfits).

Attention compute for L=1, 400 tokens, d=256: O(N² d) ≈ 4 × 10⁷ per head
per layer ≈ 1.6 × 10⁸ FLOPs total. Cheap; should add < 5% to per-step
runtime relative to the plain-CNN trunk.

### What this is not

- Not a replacement of the trunk — the conv stages are preserved verbatim
  for direct comparability.
- Not a hyperparameter sweep — single seed first, then promote.
- Not a Ralph-loop campaign — shell-orchestrated triage per CLAUDE.md
  convention #7 with one-off bash launchers.

## Driver and orchestration (CLAUDE.md convention #7)

- **Driver**: shell-orchestrated; no autoresearch driver for the triage
  phase. Each arm gets a `scripts/sbi/results/exploratory/h1_inductive_bias/launch_<arm>_s<seed>.sh`.
- **Checkpoint policy**: `--compressor-checkpoint-policy best_val` (new
  campaign, no historical baseline to match).
- **GPU policy**: GPU 1 sole tenant (CLAUDE.md project rule).
- **Resource scaling**: ~3-4h per run on GPU 1.

## Diagnostic outputs (per run)

In addition to the primary metric:

1. **Per-parameter marginal bias-vs-truth in σ-units** on (Ωₘ, σ₈, w₀).
2. **Validation-loss curves** (compressor + NDE) to detect overfitting.
3. **Attention map** — visualization of attention weights on a fiducial
   image batch, to confirm the model uses global structure.
4. **Best-val checkpoint step** vs total-steps, to detect early-stopping.

## Sub-fibers

- `cnn-h1-arm-attention-triage-s41` — **CLOSED 2026-05-22, FALSIFIED.**
  3-seed pool = 11,892; haircut 0.684 (identical to plain-CNN 0.685).
  Verdict: `scripts/sbi/results/exploratory/h1_inductive_bias/H1_ATTENTION_VERDICT.md`.
- *(pending decision)* `cnn-h1-arm-spectral-block` — cheapest remaining test.
- *(pending decision)* `cnn-h1-arm-mlp-mixer-trunk`.

## Connections

- Builds on the cnn-auto-push ceiling (see closed campaign at
  `[[cnn-auto-push-18-20-2026]]`) and the cross-push best (see
  `[[cnn-auto-cross-push-18-20-2026]]`).
- Implements the experimental program from §6 of `CNN_CROSS_MAPS_INFORMATION_NOTE`.
- Stock-BN contamination memo `project_resnet_bn_contamination` warns
  against any new arm using stock BN; we will use the existing
  GroupNorm conv blocks (no BN in the trunk we are extending).

## Exit interview (2026-05-22)

**Result**: H1 (inductive-bias-via-attention) falsified at the chosen
implementation (L=1, tail insertion, 4 heads). Pool FoM3 = 11,892 vs
plain-CNN anchor 11,130. Pool/MoS haircut **identical to 3 decimal places**
(0.684 vs 0.685). Gap closed vs the 24k auto+cross target: +6%.

**What worked**:
- The seven Felt/Ralph operating conventions (CLAUDE.md) held up. Primary
  metric was declared and never drifted. The single-seed promotion rule
  fired correctly. The campaign produced a clean verdict in ~4h wall-clock
  with no procedural waste.
- Scoping the constitution body BEFORE writing any code (per the
  "plan-before-implementation" rule in `~/.claude/CLAUDE.md`) caught two
  spec issues that would have wasted runs: (a) 4 conv stages → 3 conv
  stages (matching production), (b) 120k → 60k compressor steps (matching
  the iter-108-Q6ON-60k anchor). The constitution body served as the
  contract; both fixes landed before any compute.
- Shell-orchestrated triage (vs building an autoresearch driver up front)
  was the right call. Total engineering cost: ~30 min for 2 launchers + 1
  analysis script.
- AskUserQuestion checkpoints between scope decisions kept Andreas in the
  loop on the architectural choices without slowing things down.

**What was wasted effort**:
- Initial single-seed-threshold of 13k. The threshold was set against the
  3-seed *pooled* anchor of 11.1k but should have been set against the
  matched-seed plain-CNN single-seed of 18.6k. As a result, s41's +4.2%
  "promotion" was lukewarm rather than meaningful. The promotion still
  worked out (the 3-seed pool was the real metric anyway) but the
  triage signal was unnecessarily noisy. Lesson: thresholds for single-seed
  triage should be expressed as *deltas vs the matched-seed baseline*, not
  absolute numbers.

**What we'd do differently**:
- For future architectural-variant arms: set the single-seed promotion
  threshold as **"per-seed FoM3 ≥ matched-seed plain-CNN baseline × 1.15"**
  rather than an absolute number. This avoids the lukewarm-promotion
  noise we saw on s41.
- Run all 3 seeds in parallel (mem-fraction 0.45 each on GPU 1) rather than
  sequentially. Smoke-tested architecture is 1.33M params; comfortably fits
  3 in parallel. Would have cut wall-clock 3× without engineering cost.

**Convention updates proposed for CLAUDE.md "Felt / Ralph operating
conventions"**:
- *(new #8)*: For multi-seed architectural-variant arms, single-seed
  triage thresholds must be expressed as **relative deltas vs the matched-
  seed baseline**, not absolute values. This prevents lukewarm promotions
  when the per-seed anchor sits far above the pooled anchor.

**Status**: closed. Next workstream:
`[[cnn-h2-data-limit-scoping-2026-05]]` (H2 = is the auto-only ceiling a
data-limit at CosmoGridV1's ~70k cosmologies?).
