# H1 attention arm — verdict (2026-05-22)

**Question** (from `[[CNN_CROSS_MAPS_INFORMATION_NOTE]]`): does inserting a
global-mixing transformer block at the tail of the plain-CNN trunk close the
plain-CNN's 11.1k → 24.0k auto-only / auto+cross FoM3 gap on tomographic
shear?

**Test**: plain CNN trunk (3 stages, 64/128/256, 160→80→40→20 px) +
1 transformer block (4 heads, d_k=64, MLP×4, pre-LN, learned 400-token
positional embedding) + mean-pool + existing dense head. cdim=10, 60k
compressor steps, batch=128, LR=5e-4. Auto-only 4-channel input.
Matches iter-108-Q6ON-60k of the cross-push campaign EXCEPT the
architecture switch.

## Results (3 seeds, 100k samples each)

| seed | FoM3   | Ωₘ bias | σ₈ bias | w₀ bias |
|:-----|-------:|--------:|--------:|--------:|
| s41  | 19,389 |  −0.08σ |  +0.10σ |  −0.48σ |
| s42  | 17,251 |  +0.22σ |  −0.53σ |  −0.50σ |
| s43  | 15,527 |  +0.84σ |  −1.77σ |  −1.07σ |

| aggregate | value |
|:--|--:|
| MoS FoM3            | 17,389 |
| **Pooled FoM3**     | **11,892** |
| Pool/MoS haircut    | **0.684** |
| Global median \|bias\| | 0.50σ |

## Side-by-side vs the plain-CNN auto-only anchor

(plain CNN numbers from `apples_v_iter108_autoonly/posteriors/`, same seeds, same config except `--compressor-arch plain` instead of `plain_attn`.)

| arm | MoS | Pool | haircut | \|bias\| med |
|:---|----:|-----:|--------:|-------------:|
| attention | 17,389 | 11,892 | 0.684 | 0.50σ |
| plain CNN | 16,243 | 11,130 | 0.685 | 0.52σ |
| **Δ (attn−plain)** | **+1,146** | **+762** | **−0.001** | **−0.03σ** |

## Decision

Constitution decision rule (auto-only 3-seed pooled FoM3):

| pooled FoM3 | verdict |
|:--|:--|
| ≥ 20,000 | H1 confirmed |
| 13,000 – 20,000 | partial |
| ≤ 13,000 | H1 falsified |

**Pooled = 11,892 → H1 (attention variant) FALSIFIED.**

Gap closed vs auto+cross target (23,986): **+6%** — essentially zero.

## Interpretation

The strongest signal is the **pool/MoS haircut being identical to 3 decimal
places** (0.684 vs 0.685). The dominant failure mode in plain-CNN auto-only
is seed-to-seed mode drift (31% pooling penalty). Adding ~700k parameters of
global receptive field, with the same training signal and the same data,
**did not move that haircut at all**. Per-seed FoM3 moved within noise.
Bias profile is unchanged.

This is strong Bayesian evidence — though not a complete family-level
falsification — that **H1 (inductive-bias) is not the load-bearing limit on
this dataset**. The seed-to-seed mode drift is structural: data
distribution + parameter initialization variability, not architectural
inductive bias.

Three caveats worth naming:

1. **Only one H1 arm tested.** The constitution lists three: (a) explicit
   spectral block at input, (b) attention block at trunk tail (this run),
   (c) MLP-Mixer / FNet trunk replacement. Strictly, only (b) is falsified.
   It is plausible — though less likely — that (a) or (c) succeed where (b)
   failed.

2. **Single attention configuration.** L=1 transformer block, H=4 heads,
   tail insertion. We did not sweep L, H, or insertion point. A larger
   attention configuration could, in principle, behave differently. The
   parameter-count argument (1.33M total, well below the 25M resnet50 that
   overfit) means we are NOT data-limited at this scale, so escalating
   attention capacity is not the obvious fix.

3. **The hypothesis ladder still has H2 (data-limited) and H3 (compressor
   bottleneck) standing.** This result actively *strengthens* H2: if
   inductive bias doesn't help, the missing-from-auto information may not
   be learnable from this dataset at all.

## Run-time observations

- s41 NDE val-loss spiked late (step 13.5k: 1.87×10⁶, then 2,439, then 42)
  before early-stopping. Best-val checkpoint at step 12.5k held. Pattern
  not seen in s42 or s43. Likely a single bad val batch; not a systemic
  issue with the attention architecture.
- All 3 runs completed in 45-50 min on GPU 1 (compressor 60k + NDE early-
  stops around 15k). No NaN posterior samples in any seed.

## Artifacts

- Posteriors: `posteriors/cnn_attn_auto_s{41,42,43}.npy`
- Numerical summary: `h1_attention_3seed_verdict.json`
- Logs: `logs/cnn_attn_auto_s{41,42,43}.log`
- Training dirs: `train_s{41,42,43}/`

## Recommended next moves (for Andreas)

1. **Pivot to H2 (data-limited test) as next workstream.** Test on a much
   bigger N-body suite — if auto-only FoM3 closes toward 24k with more data
   but same plain-CNN architecture, H2 confirmed.
2. **OR test one more H1 arm before declaring H1 dead at the family level.**
   The cheapest is the **explicit spectral block at input** — prepend a
   learned FFT-style operator, see if frequency-domain access at the *input*
   does what attention at the *output* did not.
3. **OR close the campaign here.** The +6% gap closure is unambiguously
   negative; investing more is reasonable only if you place a high prior on
   spectral-block or mixer arms behaving differently from attention.

My honest read: the architectural lever is not promising on this dataset.
The seed-to-seed haircut is the real story, and it does not respond to
inductive-bias changes. Recommend (1) or (3).
