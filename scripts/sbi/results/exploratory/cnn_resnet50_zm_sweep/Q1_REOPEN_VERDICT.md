# Q1 reopen — verdict (2026-05-22)

**Question:** When `[[cnn-auto-q1-resnet50gn-120k-1seed]]` returned NULL, did we
miss that **stock-BN** resnet50 (not the GN variant we tested) is the actual
auto-only lever? The `overlay_baseline_vs_resnet50_cdim20.pdf` figure showed a
single stock-BN seed (s42) hitting FoM3 31,684 — well above the cnn-auto-push
plain-CNN ceiling at ~25k pooled.

**Test:** add a third seed (s41) to the existing May 8 stock-BN cdim=20 sweep
(s42, s43) and apply the decision rule:

> pooled FoM3 > 15,000 **AND** median |bias| < 0.5σ → reopen Q1

## 3-seed numbers

| seed       | FoM3   | Ωₘ mean | σ₈ mean | w₀ mean |
|:-----------|-------:|--------:|--------:|--------:|
| s41 (today)        | 24,462 | 0.2570 | 0.8355 | **−1.1323** |
| s42 (May 7)        | 31,684 | 0.2578 | 0.8453 | **−1.1214** |
| s43 (May 8)        | 23,652 | 0.2735 | 0.7884 | **−1.1067** |
| **truth**          |   —    | 0.2600 | 0.8400 | −1.0000 |

| aggregate                       | value  |
|:--------------------------------|-------:|
| Mean of seeds (MoS) FoM3        | 26,599 |
| **Pooled (3-seed) FoM3**        | **18,368** |
| Pooled / MoS (joint_R haircut)  | 0.69   |

| bias σ-units | Ωₘ | σ₈ | w₀ |
|:--|--:|--:|--:|
| s41 | −0.13 | −0.13 | **−1.12** |
| s42 | −0.09 | +0.15 | **−1.03** |
| s43 | +0.51 | −1.51 | **−0.79** |
| `\|bias\|` median | 0.13 | 0.15 | **1.03** |

**Global median `\|bias\|` across (Ωₘ, σ₈, w₀) × 3 seeds: 0.51σ**

## Decision

| rule | required | observed | result |
|:-----|:--------|:---------|:-------|
| pooled FoM3 > 15,000   | > 15k | 18,368 | **PASS** |
| median `\|bias\|` < 0.5σ | < 0.5σ | 0.51σ  | **FAIL** (by 0.01σ) |

**Verdict: Q1 CLOSED — BN-contamination interpretation upheld; Q1-at-GN holds.**

## What the numbers actually say

The bias rule fails *by 0.01σ on the global median* — but the qualitative
pattern is the textbook BN-running-stats-leakage signature, not a marginal
miss:

1. **All three seeds bias w₀ in the same direction.** w₀ means cluster at
   −1.13, −1.12, −1.07 with truth at −1.00. That's a systematic, not seed
   noise — three independent draws don't agree like that by chance.
2. **Mode drift on the pool.** Pooled/MoS = 0.69 (31% haircut) means the
   per-seed posteriors are sitting at different mode locations: pooling fattens
   the joint covariance and the FoM3 collapses from MoS=26.6k to pool=18.4k.
   The GN cdim=20 iter-16 pool was 25.5k (no haircut). That gap *is* the
   BN-contamination signal.
3. **Tight-but-disagreeing.** Per-seed FoM3 (24–32k) IS above the GN ceiling
   (11.8k single-seed in Q1 retest, ~25k iter-16 pool). Stock BN does
   produce a tighter posterior. It just produces a tighter *wrong* posterior
   that drifts seed-to-seed.

This is what the existing `project_resnet_bn_contamination.md` memory predicts
on a different input (10-channel harmonic cross): on cross input BN catastrophed
to FoM3~700; on 4-channel auto-only input BN doesn't catastrophe — it produces
tight-but-biased posteriors with seed-dependent mode drift. Same mechanism
(running-stats leakage across cosmology-mixed batches), milder failure mode
because auto-only doesn't have the cross-channel amplitude problem.

## Implication for the cnn-auto-push ceiling

The ceiling **`pooled ≈ 25k`** at the plain-CNN trunk on auto-only stands. The
apparent stock-BN gain at FoM3=31,684 on s42 was a contamination artifact, not
a real lever. We can confirm this off-line by:

- Reading off the pooled stock-BN number (18,368) against the GN-cdim=20
  iter-16 pool (25,466) and the cnn-auto-push iter-108-Q6ON pool (23,986).
  Stock-BN actually *underperforms* GN once you pool and account for bias.
- The w₀ systematic at −1.1 means any "tight 31k" stock-BN result is
  catastrophically miscalibrated for the *actual* parameter of cosmological
  interest in this dataset (truth w₀ = −1.0, well-tracked by GN and L1).

**No follow-up sweeps. Cnn-auto-push ceiling at the GN variant is the
scientific answer.**

## Memory updates needed

- Extend `project_resnet_bn_contamination.md`: add the 3-seed auto-only data
  point showing tight-but-biased mode-drift as the milder cousin of the
  10-channel-cross FoM3-collapse failure mode.
- Close fiber `[[cnn-auto-push-18-20-2026/cnn-auto-q8-resnet50-stockbn-cdim20-seed41]]`
  with this verdict.
- Parent fiber `[[cnn-auto-push-18-20-2026]]` already at CEILING TRIPLY-CONFIRMED
  — append a fourth confirmation (stock-BN cdim=20 also non-lever, via
  BN-contamination interpretation).
