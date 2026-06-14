# RESULT — 2D-1D Haar wavelet ℓ1-norm, Phase 1 (Approach A, linear)

**Date:** 2026-06-14 (overnight). Implements PLAN_2D1D_PHASE_1_2.md Phase 1. All arms run through the
SAME common-MAF / 9000-obs / log1p-zscore path as the baselines (apples-to-apples), gated by
TARP+SBC. Run: `run_haar_2d1d_phase1.py` (GPU2, ~63 min). Artifacts: `population_sweep/<arm>/`,
`overnight_menu_2/gate_c_2d1d/` (dumps, curves, verdicts.json, GATE_C_2D1D.md).

## Headline (honest)
The faithful **linear** 2D-1D Haar wavelet ℓ1-norm **confirms the linear-recombination ceiling and
does NOT advance either goal**: for goal 1 it ties (does not exceed) the existing cross-map arms, and
for goal 2 it **collapses ~3× under BNT** (calibrated — a real information loss, not over-confidence).
This matches the §4 reduction prediction (the linear form ≡ ordinary ℓ1 on fixed Haar maps) and makes
**Approach B (the modulus version) the only remaining lever** — now motivated by measurement, not hope.

## Numbers (all same common-MAF / 9000-obs path)
| arm | FoM3 | σ(Ωm,σ8,w0) | gate verdict | TARP LOW/MID/HIGH · net-bias · SBC std |
|---|---|---|---|---|
| flat_none (auto-only) | 2405 | 0.053, 0.082, 0.245 | (baseline, calibrated) | — |
| flat_product (L1+ξ_ij) | 2875 | 0.048, 0.075, 0.238 | (bar, calibrated) | — |
| flat_both (L1+conv+prod) | 2910 | 0.046, 0.075, 0.232 | (prior best linear) | — |
| **haar_nobnt** (pure 2D-1D Haar) | **2676** | 0.049, 0.078, 0.235 | **FAIL** | +0.081/+0.037/−0.104 · +0.004±0.061 · 0.303,0.303,0.292 |
| **autohaar_nobnt** (autos ⊕ Haar) | **2954** | 0.046, 0.074, 0.231 | **PASS-with-caveat** | +0.036/+0.066/−0.047 · +0.026±0.024 · 0.303,0.306,0.296 |
| **haar_bnt_uncut** (Haar·B; BNT space) | **885** | 0.082, 0.128, 0.303 | **PASS-with-caveat** | −0.061/−0.045/+0.063 · −0.007±0.036 · 0.298,0.296,0.295 |

## Goal 1 (tighter contours) — modest, and the pure form is mildly over-confident
- **Pure 2D-1D Haar ℓ1 (`haar_nobnt`): FoM3 2676**, +11% over auto-only (2405) but **below the product
  bar (2875)** — and it **FAILS the gate** (HIGH-FoM3 tercile TARP dev −0.104, just over the 0.10 band;
  net-bias +0.004 ≈ unbiased on average, SBC clean). So its 2676 is **partly inflated by HIGH-tercile
  over-confidence** — the calibrated value is lower. The over-confidence appears when we *drop the
  autos* (the well-calibrated per-bin info) and keep only the 4 Haar modes. **Registered prediction
  (2900–3300) MISSED on the low side, and the "clean calibration" assumption was wrong.**
- **Augmented `autohaar_nobnt` (autos ⊕ 4 Haar modes): FoM3 2954**, PASS-with-caveat (worst dev +0.066,
  net-bias +0.026 = mildly conservative, SBC clean). But this is **statistically the same place as the
  existing best linear arms**: flat_both 2910 (σ 0.046/0.075/0.232) and flat_product 2875 — autohaar's
  marginals (0.046/0.074/0.231) are essentially identical to flat_both's. The +3% over the product bar
  is within FoM3 fragility. **So the Haar modes add nothing beyond the product/both cross channels** —
  the deep mode helps, but the autos already carry most of it, and the Haar difference maps duplicate
  what ξ_ij/conv already supply.
- **Reading:** the linear 2D-1D Haar ℓ1 is a *re-parameterization within* the linear-recombination
  family, bounded by the same ~2900 ceiling we already hit. It does not open new ground for goal 1.

## Goal 2 (BNT robustness) — the linear Haar does NOT survive BNT (prediction falsified)
- **`haar_bnt_uncut` (the 2D-1D Haar ℓ1 computed in BNT space): FoM3 885** = **0.33× of `haar_nobnt`
  (2676), 0.37× of auto-only (2405)**; σ(w0) 0.303 vs 0.235 (~30% inflation). **CALIBRATED**
  (PASS-with-caveat, net-bias −0.007, SBC 0.295–0.298) ⇒ this is a **real information loss, not
  over-confidence.** The contours genuinely inflate ~3× moving to BNT space.
- **Registered prediction FALSIFIED:** I predicted "≈ haar_nobnt if BNT-robust." It is **not**
  BNT-robust — it collapses much like per-bin L1.
- **Mechanism (confirmed by the σ table):** `Haar·B` is a fixed linear frame that does **not**
  reconstruct the deep coherent mode (BNT scrambled it across thin channels; Haar of the scrambled
  channels ≠ the deep mode). In BNT space only the `deep_bnt` channel carries appreciable signal and it
  is *weaker* than the no-BNT deep mode (σ 2.9e-3, range [−6.7,7.7] vs no-BNT 5.0e-3, [−12,14]); the
  other three BNT-Haar channels are low-S/N ([−3,3]). This is exactly the M3 lesson: only a
  *whitening/orthonormalizing* frame recovers BNT info — a generic fixed rotation (Haar) does not.

## Scoring vs registered predictions (PLAN_2D1D_PHASE_1_2.md §6.2)
- `haar_nobnt` 2900–3300 → **2676, MISSED low** + FAILED calibration (predicted clean). Over-optimistic.
- `autohaar_nobnt` ≥ haar_nobnt and ≈/> 2875 → **HIT** (2954), but ties flat_both rather than exceeding.
- `haar_bnt_uncut` ≈ haar_nobnt if BNT-robust → **the "if" resolved NO** (885; not BNT-robust).

Net: I was over-optimistic on the linear form for both goals. The honest finding is a clean confirmation
of the §4 linearity-reduction analysis: a linear bin-mix + per-channel ℓ1 cannot exceed the
linear-recombination ceiling and cannot recover BNT info without the *right* (whitening) rotation.

## Implication for Phase 2 (modulus) — now well-motivated, not assumed
The modulus is the **only remaining lever** for BOTH goals:
1. **Goal 1:** to exceed the ~2900 linear ceiling requires the nonlinearity (§4.2) — the linear family
   is now measured to top out there.
2. **Goal 2:** the modulus-Haar *sum* mode is `Σ_b |S_{j1}κ_b|`, a sum of **positive** moduli — it does
   NOT suffer the sign cancellation that just destroyed the deep mode in BNT space. This is the
   mechanism that *could* be BNT-robust where the linear Haar was not. **This is now the key open test.**

**Recommendation:** proceed to Phase 2 (build_flatsky_haar_scatter, code-level spec in
PLAN_2D1D_PHASE_1_2.md §Phase 2 — needs the empirical modulus-field noise freeze, the one careful new
piece given the project's noise-model history). Worth flagging to Jean-Luc: his "absolute values" is
now the crux — the linear ℓ1 underdelivers; the modulus is where any win lives.

## Caveats
- FoM3 differences ≲10–20% here are within the standing fragility caveat (1–2% correlation → ~50% FoM3
  swing); the goal-1 reading is therefore marginals-first (autohaar ≈ flat_both on all three σ). The
  goal-2 collapse (3× FoM3, 30% σ(w0)) is well outside fragility — robust.
- `haar_nobnt`'s FAIL is borderline (single tercile −0.104 vs 0.10 band; net-unbiased) — "mildly
  over-confident in the tight tail," not a catastrophic miscalibration. But enough that 2676 should not
  be quoted as a clean gain.
