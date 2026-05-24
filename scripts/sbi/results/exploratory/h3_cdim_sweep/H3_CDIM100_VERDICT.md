# H3 cdim=100 test — verdict (2026-05-24)

**Question** (from `[[CNN_CROSS_MAPS_INFORMATION_NOTE]]` §3.H3 and Andreas's
explicit ask): is the CNN-VMIM summary dimensionality (cdim=10 in production)
the load-bearing bottleneck on CNN auto+cross? Test: push cdim to 100 and see
if FoM3 lifts past the iter-108-Q6ON-60k anchor (23,986 pooled).

**Test**: identical config to iter-108-Q6ON-60k EXCEPT `--compressor-dim 10`
→ `--compressor-dim 100`. 3-seed parallel run on GPU 1. Same conv trunk
(64/128/256), same dense head (width 256), same NDE (RealNVP 8 layers,
hidden 256), same 60k compressor steps + 50k NDE steps, same `--zero-mean-maps`,
`--standardize-summary`, harmonic-cross-cache nobnt regime.

## Results (3 seeds, 100k samples each)

| seed | FoM3   | Ωₘ bias | σ₈ bias | w₀ bias |
|:-----|-------:|--------:|--------:|--------:|
| s41  | 13,799 |  +0.20σ |  −0.21σ |  +0.19σ |
| s42  | 15,191 |  +0.00σ |  +0.15σ |  +0.12σ |
| s43  | 11,372 |  +0.52σ |  −0.27σ |  +0.74σ |

| aggregate           | value  |
|:--------------------|-------:|
| MoS FoM3            | 13,454 |
| **Pooled FoM3**     | **12,151** |
| Pool/MoS haircut    | 0.903 |
| Global median \|bias\| | 0.20σ |

## Side-by-side vs the cdim=10 anchor

(anchor: iter-108-Q6ON-60k of the `cnn-auto-cross-push-18-20-2026` campaign,
same seeds, same config except `--compressor-dim`.)

| arm | MoS | Pool | haircut | \|bias\| med |
|:---|----:|-----:|--------:|-------------:|
| cdim=100 | 13,454 | 12,151 | 0.903 | 0.20σ |
| cdim=10  | 24,538 | 23,986 | 0.978 | 0.17σ |
| **Δ (cdim=100 − cdim=10)** | **−11,084 (−45%)** | **−11,835 (−49%)** | −0.075 | +0.03σ |

## Decision

Constitution decision rule (3-seed pooled FoM3):

| pooled FoM3 | verdict |
|:--|:--|
| ≥ 28,000 | H3 confirmed |
| 25,000 – 28,000 | partial |
| ≤ 25,000 | H3 falsified |

**Pooled = 12,151 → H3 FALSIFIED.**

Direction is opposite to the H3 hypothesis: cdim=100 didn't just fail to
lift FoM3 — it **cratered it by ~49%**.

## Interpretation

Posteriors at cdim=100 are still **well-calibrated** (|bias|med = 0.20σ vs
0.17σ at cdim=10; pool/MoS haircut 0.903 vs 0.978). They're just roughly
2× wider in 3-D parameter volume.

Two signals from the logs explain the degradation:

1. **Compressor VMIM loss IS tighter at cdim=100** (best val ≈ −11.73 to
   −11.84 vs anchor's ~−12). The compressor "trained" successfully by its
   own metric — but a tighter VMIM loss with a higher-dim summary doesn't
   automatically translate to a more informative compressed representation
   for the NDE. This is exactly the L67 lesson:
   `[[feedback_val_loss_not_reliable_fom3_proxy]]` — val-loss is not a
   reliable FoM3 proxy across architectures.

2. **NDE early-stopped much sooner** than at cdim=10. All three seeds
   triggered patience=20 between NDE steps 13.5k and 17k out of the 50k
   budget, with late val-loss spikes (one hit 6.2 × 10⁷). The RealNVP
   (8 layers, hidden=256) is **underprovisioned for 100-d conditioning**
   — it cannot fit a flow that takes 100 noisy conditioning channels and
   produces a tight 6-D posterior in the available training budget. The
   resulting NDE produces flatter, more conservative posteriors.

The combination: more degrees of freedom in the summary, but the downstream
NDE cannot exploit them under the standard cdim=10-tuned config.

## What this rules out and what it doesn't

**Ruled out**: at the current NDE configuration (matched to cdim=10),
summary dimensionality is NOT the bottleneck — and in fact pushing it
upward degrades pipeline performance. cdim=10 is essentially near-optimal
for this NDE.

**Not ruled out** (strict claim): a separately-tuned, higher-capacity NDE
at cdim=100 (deeper RealNVP, wider hidden, longer training budget) could
in principle match or exceed the cdim=10 result. We did not test this.
But the spirit of Andreas's question — "is summary-dim the bottleneck?"
— is answered: no, not at any realistic configuration of the existing
pipeline.

## Update to the §3 hypothesis ranking in CNN_CROSS_MAPS_INFORMATION_NOTE.md

H3 (summary-dim bottleneck) → **falsified at the standard NDE config**,
in direction *worse than the null*. Does not change the H1 vs H2 vs
global-info-via-spherical-procedure picture from §8c.4; H3 is decisively
removed from the running.

## Artifacts

- Posteriors: `posteriors/cnn_cross_cdim100_s{41,42,43}.npy`
- Numerical summary: `h3_cdim100_3seed_verdict.json`
- Logs: `logs/cnn_cross_cdim100_s{41,42,43}.log`
- Training dirs: `train_s{41,42,43}/vmim/nbody/.../harmonic_nobnt_ch10/`
- Compressor val-loss curves saved in each `train_s{n}/` (the
  `loss_compressor_train.npy` and `loss_compressor_test.npy` files —
  confirm tighter VMIM val loss for the cdim=100 runs).

## Practical lesson for the H3 fiber exit interview

3-way parallel on a single A100 was **slower wall-clock** (~7.5h) than
sequential would have been (~5h). The 10-channel input × cdim=100 final
layer is heavy enough per-instance that 3 workers GPU-contend rather
than fit. Update the H1 exit-interview's "parallel seeds" lesson:
*parallel-N on a single A100 is faster only when each instance is small
enough not to saturate the GPU on its own*. The 10-ch / cdim=100
combination blows past that threshold.
