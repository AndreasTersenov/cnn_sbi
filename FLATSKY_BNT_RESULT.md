# BNT on the flat-sky (patch-local) maps — L1 inflates, the CNN is lossless (2026-06-10)

**Paper pillar 2.** BNT (the nulling transform, `tomo4_bnt_v1`) is linear and invertible, so it
cannot destroy cosmological information — yet higher-order per-channel statistics computed on
BNT'd maps are known to inflate contours (lower per-map S/N + the originally-independent noise
becomes correlated across bins). The hypothesis: the inflation is an *analysis* artifact —
per-channel statistics cannot see the cross-bin structure BNT makes essential — so a
channel-mixing compressor (CNN-VMIM) should be BNT-invariant. All three predictions **hold**.

Setup: BNT applied on-device to the 4 noisy, demeaned auto patches (noise → demean → BNT →
cross-build → whiten/L1), recipe-matched to the no-BNT campaign (plain CNN/80k/val-batches-1;
L1 with the BNT-refrozen per-(channel,scale) noise σ — GATE A1b ALL PASS, the empirical BNT
noise matches the analytic mixing prediction √(Σⱼ B²ᵢⱼ) to 3 digits). Pooled 3-MAF-seed
9000-obs medians; CNN over 3 compressor seeds.

## Inflation ratios (FoM3_BNT / FoM3_noBNT)

| arm | no-BNT FoM3 | BNT FoM3 | inflation |
|---|---|---|---|
| L1 auto-only | 2405 | 364 | **0.15×** |
| L1 auto+product | 2875 | 637 | **0.22×** |
| CNN auto-only (3 comp. seeds) | 2325/2170/2480 | 2174/2172/2137 | 0.94/1.00/0.86 → **0.93×** |
| CNN auto+product (3 comp. seeds) | 2181/2393/2433 | 2054/2004/2072 | 0.94/0.84/0.85 → **0.88×** |

## Marginals confirm (not a FoM3 artifact)

| arm | σ(Ωm) noBNT→BNT | σ(σ8) noBNT→BNT | σ(w0) noBNT→BNT | 2D(Ωm,σ8) |
|---|---|---|---|---|
| L1 auto | 0.053 → 0.090 (+69%) | 0.082 → 0.176 (+114%) | 0.245 → 0.323 (+32%) | 471 → 111 |
| L1 +product | 0.048 → 0.078 | 0.075 → 0.139 | 0.238 → 0.308 | 522 → 183 |
| CNN auto (s41) | 0.051 → 0.052 | 0.077 → 0.079 | 0.244 → 0.250 | 447 → 417 |
| CNN +product (s41) | 0.053 → 0.054 | 0.085 → 0.084 | 0.244 → 0.252 | 421 → 399 |

## Prediction ladder (all derived from the medians)

1. **L1 auto-only inflates dramatically** (0.15× FoM3; σ(σ8) doubles): a strictly per-channel
   statistic is blind to the cross-bin structure that BNT moves the information into. ✓
2. **L1 auto+product inflates less** (0.22× vs 0.15×; every marginal better than L1-auto-BNT):
   the explicit cross channel restores part of the cross-bin information — the cross-map trick
   is exactly a device for per-channel statistics. ✓
3. **The CNN is (near-)lossless** (auto 0.93×, product 0.88×, both within/near the established
   ±8% compressor-seed scatter; marginals move ≤3%): with the bins as input channels, VMIM
   extracts the cross-bin information implicitly, so the invertible BNT costs it nothing —
   **empirically demonstrating BNT losslessness** and locating the L1 inflation in the analysis,
   not the transform. ✓

Notably, the **plain** CNN achieved this at the standard recipe — the 20° campaign's "advanced
architecture" contingency was not needed at 10°/80px.

## Whitening decomposition (2026-06-11): FULL recovery — the collapse is a frame artifact

Per-channel L1 in the noise-whitened BNT basis Q = (BBᵀ)^(−1/2)B — an orthogonal rotation of
the original basis (signal fully mixed, noise back to iid equal-variance). Pooled 3-MAF
9000-obs medians (`whiten_campaign/WHITEN_RESULT.md`):

| arm | no-BNT | whitened | BNT | recovered (whiten−BNT)/(noBNT−BNT) |
|---|---|---|---|---|
| L1 auto | 2405 | 2524 | 364 | **1.06** |
| L1 +product | 2875 | 2897 | 637 | **1.01** |

Recovery is complete marginal-by-marginal (whiten σ(σ8) 0.080/0.075 vs noBNT 0.082/0.075;
σ(w0) 0.239/0.233 vs 0.245/0.238). Reading (full chain in `BNT_THEORY_DEEP_DIVE.md` §5):
the irreducibly-joint share of the L1's BNT loss is ≈ 0 — everything the per-channel
statistic extracts in the original basis is available through single-channel marginals in
ONE FIXED rotation of the nulled maps. The surviving account: BNT trades four deep,
mutually-redundant lensing kernels for one shallow map + three thin lens-z slices — a frame
with NO deep direction anywhere; each slice alone is signal-starved and carries little of the
deep non-Gaussian structure (which nulling removes from every channel by design). Q recovers
because its first row ≈ the deep common mode (70% outside the nulled span). Not noise
correlation (invisible to marginals), not noise amplification (absorbed by the SNR
normalization), not mixing per se. Diagnostic only: Q remixes the nulled kernels — the
practical statement stays "cleaning basis ≠ statistics basis." Honesty trail: the
pre-registered prediction (LOW-to-MID) was falsified AND the first post-mortem (sign
structure) failed its own geometry check — both kept in the deep-dive §5 as journey material.
§5.4 ladder (both rungs measured 2026-06-11; `bntdeep_campaign/` + `bntdeep2_campaign/`):
ONE appended deep channel (bin average; 4 nulled maps untouched) recovers **0.730** (FoM3
364 → 1854 of 2405; below the registered ≥0.8 ⇒ single-direction strong form refuted); TWO
depth-distinct deep channels (average + deepest bin κ₄) recover **1.082** — the registered
SPANNING branch: FoM3 2573, σ(σ8) 0.079 (vs noBNT 0.082), σ(w0) 0.241 (vs 0.245), every
marginal at or better than noBNT. ⇒ The 1-deep residual was among-deep-kernel tomographic
structure; per-channel-accessible information saturates at ~2 depth-distinct deep
directions. Span curve: 0 → 0.00, 1 → 0.730, 2 → 1.082, orthonormal-4 (whiten) → 1.06.
Values above 1 recur across two independent frames and read as "complete; the standard
per-bin frame is itself a mildly suboptimal one-point direction sampling" — not over-read.

## Caveats / validation (GATE C ran 2026-06-11 — pass WITH caveats; `bnt_campaign/gate_c/GATE_C_BNT.md`)

- **Headline-safe:** all calibration deviations are at the ~5–10% credible-width level vs the
  ~90% (L1) and ≤10% (CNN) measured width effects. Directionally: L1-BNT is mildly
  over-confident (SBC std 0.295–0.304) ⇒ the true L1 inflation is at least as severe as quoted
  ⇒ predictions 1–2 conservative as stated.
- **CNN arms are harder to calibrate in BNT space than in the original basis** (no-BNT arms
  were clean): cnn-auto = mild (TARP tightest-tercile −0.068; L-C2ST 13%; SBC conservative);
  cnn-product = locally miscalibrated at the fiducial (L-C2ST 40% reject, self-test powered).
  Quote losslessness from the AUTO arm; carry the product caveat explicitly.
- CNN mean ratios (0.93×/0.88×) sit at the edge of the compressor-seed band: a mild ≲10%
  residual loss is not excluded; "lossless within seed scatter" is the precise claim.
- L1 arms have no compressor-seed dimension (deterministic datavector); 3-MAF pooling only.
- FoM3 fragility is irrelevant at these effect sizes (6.6× and 4.5×), and the marginals agree.

## Figures (`bnt_campaign/figures/`)

- `fom3_bnt_inflation` — headline log-scale bars (0.15×/0.22× collapse vs 0.93×/0.87× lossless).
- `corner_bnt_vs_nobnt_{l1,cnn}_{none,product}` — BNT vs no-BNT contours at the typical obs.
- `sigma_bnt_dumbbell` — per-parameter marginal widths (σ8 hit hardest, w0 mildest).
- `datavectors_bnt_vs_nobnt_s8[_relative]` — σ8-coded L1 'both' datavectors in both bases. The
  RELATIVE version is the mechanism made visible: under BNT the auto/conv blocks lose almost
  all σ8 response while the PRODUCT block retains the most — the per-channel collapse and the
  partial cross-channel rescue in one figure.
- GATE C: `tarp_bnt_colored_dim{3,6}` (campaign colors, 16–84% bands), `sbc_bnt_ranks`,
  `lc2st_bnt_cnn`.
- `../whiten_campaign/figures/fom3_whiten_decomposition` — noBNT/whitened/BNT bars with
  recovered fractions (`whiten_campaign_figure.py`).

## Reproduce

`run_flatsky_bnt_campaign.py` (one driver: sigma freeze → L1 both-BNT build → arm slices +
6 CNN BNT compressors → fiducial summaries → 8 jitted sweeps → derived report).
Artifacts: `scripts/sbi/results/exploratory/flatsky_cross_2026_06/bnt_campaign/`
(BNT_CAMPAIGN_RESULT.md, population_sweep/*/median_summary.json);
BNT noise table `…/flatsky_cross_noise_sigma_bnt.npz` (GATE A1b in the sigma log).
BNT operators: `flatsky_cross.apply_bnt_*` + `bnt=` on `build_channels_*`; CNN `--flatsky-bnt`;
L1 `--apply-bnt` on the flat_local route. Figures: `bnt_campaign_figures.py`,
`bnt_corner_overlays.py`, `plot_tarp_bnt_colored.py`, `plot_bnt_datavectors.py`.
Whitening test: `run_flatsky_whiten_campaign.py` (sigma freeze w/ GATE A1b → both-whiten build
→ arm slices → fiducial precompute → jit sweeps → derived WHITEN_RESULT.md);
`--flatsky-channel-mix whiten` on the L1 flat_local route; figure `whiten_campaign_figure.py`.
Theory: `BNT_THEORY_DEEP_DIVE.md` (same dir as the campaign) — the canonical derivation layer.
