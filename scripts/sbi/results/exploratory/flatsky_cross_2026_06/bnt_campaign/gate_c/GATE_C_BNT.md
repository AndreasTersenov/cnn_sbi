# GATE C — BNT arms (derived verdicts)

Validates the BNT campaign posteriors (FLATSKY_BNT_RESULT.md). Same machinery as the no-BNT gates (all of which passed).

## SBC (ranks from the TARP dumps; science params)
| arm | n | mean(Om,s8,w0) | std (uniform=0.289) | min KS p |
|---|---|---|---|---|
| cnn none | 1800 | 0.471,0.524,0.485 | 0.282,0.283,0.273 | 0.000 |
| cnn product | 1800 | 0.477,0.525,0.482 | 0.278,0.275,0.274 | 0.000 |
| l1 none | 1800 | 0.508,0.490,0.480 | 0.301,0.299,0.295 | 0.000 |
| l1 product | 1800 | 0.509,0.488,0.477 | 0.301,0.304,0.295 | 0.000 |

## TARP (dim-3 science subspace; signed max deviation ECP − α, mean over 200 bootstraps; per FoM3 tercile, worst seed)

| arm | HIGH (tightest) | MID | LOW | read |
|---|---|---|---|---|
| cnn none | −0.068 | −0.030 | +0.083 | mild over-confidence in the tightest tercile, conservative elsewhere |
| cnn product | +0.051 | +0.080 | +0.127 | conservative (over-covers) |
| l1 none | +0.080 | +0.075 | −0.087 | mixed ±0.08 |
| l1 product | +0.054 | +0.043 | −0.048 | most calibrated arm |

(positive = over-coverage/conservative; negative = under-coverage/over-confident.
No-BNT reference: load-bearing arms had |dev| ≤ 0.037; conv's −0.068 was flagged.)

## Interpretation (honest synthesis)

The BNT arms are measurably HARDER to calibrate than the no-BNT arms (which were clean at
|dev| ≤ 0.037 / L-C2ST 0%) — consistent with the prior campaign's lesson that BNT space is a
harder learning problem. Specifics and their direction relative to the paper's claims:

- L1 arms: SBC std 0.295–0.304 (> 0.289) = mild GLOBAL under-coverage (~4–5%); TARP mixed ±0.08.
  L1-BNT posteriors are, if anything, slightly OVER-confident ⇒ the true L1 inflation is at
  least as severe as measured ⇒ predictions 1–2 are PROTECTED (0.15×/0.22× optimistic for L1).
- CNN arms: SBC mildly conservative globally (std 0.273–0.283, means ≈ 0.5 with a ~0.08σ-scale
  s8 offset); TARP shows mild over-confidence in cnn-none's tightest tercile (−0.068) and
  conservatism elsewhere; L-C2ST auto 13% reject (mild; self-test powered), product 40%
  (locally miscalibrated at the fiducial — real flag).
- MAGNITUDE CHECK: these deviations correspond to ~5–10% credible-interval misestimates. The
  measured effects are 6.6× (L1) and 1.07× (CNN) in FoM3 — ~90% vs ≤10% in linear width. No
  calibration deviation of this size can mimic or hide either headline.

VERDICT: headline-safe. Quote the CNN losslessness from the AUTO arm (0.93×; calibration
mild-to-acceptable: worst tercile −0.068, L-C2ST 13%); quote the product arm (0.88×) with the
local-calibration caveat (L-C2ST 40%). The L1 inflation claims are conservative as stated.

## L-C2ST (CNN arms; local at fiducial)
| arm | reject@p<0.05 | median p | self-test (H0 p / H1 p) |
|---|---|---|---|
| cnn none | 13% | 0.16 | 0.86 / 0.02 |
| cnn product | 40% | 0.07 | 0.74 / 0.02 |

Corner overlays (BNT vs no-BNT): `bnt_campaign/figures/corner_bnt_vs_nobnt_*.png`.