# RESULT — 2D-1D Haar SCATTERING ℓ1, Phase 2 (Approach B, modulus) + combined verdict

**Date:** 2026-06-14. Same common-MAF / 9000-obs / log1p-zscore path + TARP+SBC gate as every other
arm (apples-to-apples). Run `run_haar_scatter_phase2.py` (GPU2, ~39 min). Pre-validated: transform
mechanically correct (`validate_haar_scatter.py`: einsum exact, deep-mode ≥0 invariant, diffs
zero-mean, shape/NaN clean), empirical R=48 noise freeze (after fixing a fork/healpy `Pool` deadlock
→ `spawn` context), and a cosmology-sensitivity proxy that looked promising (σ8-corr ~0.49 in BNT
space). **The proxy oversold it — the gated FoM3 is the arbiter, and it says the modulus underperforms.**

## Headline (honest)
**Inserting a modulus (the scattering-structured Approach B) UNDERPERFORMS the linear Approach A on
both goals.** It is worse than auto-only on goal 1 and collapses under BNT on goal 2. Combined with
Phase 1, the conclusion is clean: **neither reading of the 2D-1D Haar ℓ1-norm advances either goal** —
the linear form ties the existing cross-map ceiling, and the modulus form is strictly less informative.

## Numbers
| arm | FoM3 | σ(Ωm,σ8,w0) | gate | TARP LOW/MID/HIGH · net-bias · SBC |
|---|---|---|---|---|
| auto-only | 2405 | 0.053, 0.082, 0.245 | clean | — |
| L1+product (bar) | 2875 | 0.048, 0.075, 0.238 | clean | — |
| linear Haar (P1) | 2676 | 0.049, 0.078, 0.235 | FAIL (mild tail) | — |
| linear Haar BNT (P1) | 885 | 0.082, 0.128, 0.303 | PASS-caveat | — |
| **modulus Haar (P2)** | **2234** | 0.055, 0.085, 0.244 | PASS-caveat | +0.091/+0.055/+0.059 · +0.049±0.015 · 0.303,0.303,0.294 |
| **modulus Haar BNT (P2)** | **706** | 0.078, 0.152, 0.293 | PASS-caveat | −0.051/+0.097/−0.033 · +0.014±0.042 · 0.304,0.295,0.291 |

- **Goal 1:** `haarscat_nobnt` 2234 < auto-only 2405 < linear Haar 2676 < product 2875. The modulus
  did NOT beat the linear ceiling — it landed *below auto-only*. It is **calibrated and slightly
  CONSERVATIVE** (net-bias +0.049, SBC 0.303 ≳ 0.289), so 2234 is, if anything, a mild *under*-count
  of its constraining power — not over-confidence. The underperformance is real.
- **Goal 2:** `haarscat_bnt` 706 < linear Haar BNT 885; and 706/2234 = 0.32× — the same ~3× collapse
  the linear form showed. The modulus did NOT survive BNT either. Calibrated (PASS-with-caveat).

## Why the modulus underperforms (the mechanism — and it makes sense in hindsight)
The starlet **ℓ1-norm already uses the absolute value optimally**: it bins coefficients by *signed*
S/N (peaks at +S/N, voids at −S/N) and sums |coeff| within each bin — so the **peak/void asymmetry**,
which is cosmologically informative, is preserved. Inserting an *extra* modulus `|W_b|` BEFORE the
bin-axis Haar **destroys that sign**: a peak and a void in bin b map to the same |W_b|. The cross-bin
power-asymmetry the Haar-of-moduli adds (the difference modes `|W_i|−|W_j|`) does not compensate for
the lost peak/void information — net, the modulus-Haar ℓ1 is a *strictly less informative* use of the
wavelet coefficients than the ordinary signed ℓ1-norm, which is exactly why it falls below auto-only.

This strongly implies **Jean-Luc's "with the absolute values" meant the ℓ1-norm's OWN |·| (Approach A,
linear), not an intermediate modulus (Approach B)** — the data settle the §3 ambiguity: A ties the
ceiling; B throws away sign and underperforms.

## Why the pre-run sensitivity proxy misled (lesson)
The proxy measured corr(mean J_m, σ8) per mode — a single scalar response. It showed the field *does*
respond to σ8 (incl. in BNT space), which correctly ruled out "totally dead" and justified running.
But it could not capture (i) the lost peak/void information, (ii) the w0/joint sensitivity that
dominates FoM3, or (iii) redundancy with auto-only. **Necessary-but-not-sufficient: only the gated
FoM3 is decisive** (the project's recurring val-loss/proxy ≠ FoM3 lesson, here for a sensitivity proxy).

## Combined 2D-1D Haar verdict (Phases 1 + 2)
Across both natural readings of Jean-Luc's 2D-1D-wavelet-ℓ1-norm suggestion:
- **Goal 1 (tighter contours):** linear ties the existing cross-map ceiling (~2900, no gain over
  product/both); modulus underperforms auto-only. **No advance.**
- **Goal 2 (BNT robustness):** linear collapses (885), modulus collapses (706). **No advance** — only
  a *whitening/orthonormalizing* frame recovers BNT info (M3), which neither Haar reading is.
- **Recommendation: do NOT roll with this** ("if it's really good" — it isn't). Keep it as a
  thoroughly-tested, mechanistically-understood negative in the paper's journey, and a concrete answer
  to Jean-Luc.

## What this rules out / open question for Jean-Luc
We have now tested both readings and understand the mechanism. The remaining things that *could* still
help (none cheap, none obviously worth it given the above):
- A whitening pre-rotation before a per-channel ℓ1 (proven to recover BNT in M3) — but that is the
  "whiten" idea Jean-Luc explicitly did not mean, and it doesn't beat the ceiling for goal 1.
- A signed (peak/void-preserving) cross-bin construction instead of the modulus — but the linear
  signed version IS Approach A, which ties the ceiling.
- A genuinely different 1D operation (not Haar) — unlikely to change the ceiling story (§4).

So: ask Jean-Luc whether either implementation matches his intent, and whether he had a specific
construction that escapes "linear ties the ceiling / modulus loses the sign." Backing: this doc,
RESULT_2D1D_PHASE1.md, 2D1D_WAVELET_NOTE.md, validate_haar_scatter.py.
