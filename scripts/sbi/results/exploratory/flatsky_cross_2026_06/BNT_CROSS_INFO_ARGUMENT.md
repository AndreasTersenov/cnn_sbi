# Why the CNN is BNT-lossless and per-channel L1 is not — draft argument (2026-06-11)

Paper-discussion draft. Grounded in: FLATSKY_BNT_RESULT.md (L1-auto 0.15x, L1+product 0.22x,
CNN 0.93x/0.88x) and FLATSKY_CNN_RESULT.md (no-BNT: CNN/L1 on product 0.83-0.85x, seed- and
recipe-robust).

## The core argument: order of operations, not deep-learning magic

Every per-channel statistic commits to a channel basis BEFORE its nonlinear reduction. The
wavelet l1 reduces each channel to the marginal amplitude distribution of its wavelet
coefficients, and that reduction does not commute with channel mixing: l1(aX+bY) is not a
function of l1(X), l1(Y). Once per-channel histograms are taken, no downstream combination can
reconstruct statistics of a linear combination of channels. BNT is exactly a linear channel
mixing: it moves information from per-channel marginals into cross-channel joint structure —
the one place a per-channel statistic cannot follow.

The CNN performs a linear channel mix as its FIRST operation: an invertible 4x4 channel
transform is trivially inside its hypothesis class, so it can undo BNT (or rotate to any
preferred basis) at ~zero capacity cost. The measured 0.93x is the empirical statement of this
invariance. The argument predicts BNT-losslessness for ANY channel-mixing learned compressor.

## Why the explicit product channel recovers only part (0.15x -> 0.22x)

1. Basis-fixedness: k'_i k'_j are quadratics in the BNT basis; the l1's nonlinearity means
   feeding a few fixed quadratic projections through per-channel histograms is not equivalent
   to recombination freedom before the reduction.
2. Scale blending: W_s[k_i k_j] mixes all scale PAIRS (Fourier convolution); the discriminating
   post-BNT structure is resolved by scale pair and by PHASE alignment between W_s[k_i] and
   W_s[k_j] — amplitude-only, scale-blended channels under-resolve it.
3. Noise treatment: per-(channel,scale) sigma standardizes amplitudes but cannot represent the
   inter-channel noise covariance BNT creates.

## Intrinsic power vs suboptimal cross-maps — the two-pillar verdict

The CNN is BASIS-ADAPTIVE, not STATISTIC-OPTIMAL. In the friendly (no-BNT) basis it could not
out-extract the hand-crafted l1 on the explicit product channel (0.83-0.85x, robust to
compressor seed and to a 2x-steps + de-noised-best-val recipe). Its demonstrated advantage is
the adaptive linear front-end before the nonlinear reduction (basis robustness), not sharper
cross-moment machinery. L1: statistic-strong, basis-fragile. CNN: basis-robust, not sharper.

## Candidate manual upgrades for L1 under BNT (ranked)

1. NOISE-WHITENING ROTATION (cheap; reuses today's bnt= plumbing with a different matrix).
   N' ~ B B^T; (B B^T)^(-1/2) B is ORTHOGONAL, so whitening the BNT maps = undoing BNT up to a
   rotation Q. Fixed, linear, survey-applicable; restores independent equal-variance noise.
   PREDICTION: recovers most of the 0.15x collapse; the remaining gap to no-BNT measures the
   genuinely-joint (Q-rotated) information. ~1 day incl. sigma re-freeze.
2. PER-SCALE WAVELET-DOMAIN CROSS MAPS: X_{s,ij} = W_s[k_i] * W_s[k_j] (scale-resolved local
   cross-coherence) — fixes the scale-blending loss. 6 pairs x n_scales channels; moderate.
3. PHASE-AWARE CROSS STATISTICS: wavelet phase harmonics / scattering covariances across
   channels (Cheng & Menard; Regaldo-Saint Blancard WPH cross moments) — the principled
   endpoint; bigger build.
4. DEGENERATE ANCHOR: apply B^-1 first -> restores no-BNT identically; one sentence in the
   paper proving the inflation is pure basis choice (zero information loss).
5. Cheap hack: l1 in several fixed bases (original + BNT + rotations), concatenated.

## Falsification ladder

(4) exact by construction; (1) isolates the noise-correlation mechanism; (2) isolates scale
blending. If whitening alone lifts L1-auto-BNT from 0.15x to >~0.8x, the inflation was mostly
noise-basis — a clean quantitative decomposition for the paper.
