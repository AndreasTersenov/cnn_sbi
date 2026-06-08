# Cross-map information leakage — full-sphere construction sliced into patches

**Status:** confirmed (structural + quantitative), 2026-06-08. **Severity:** high — affects the
physical interpretation of every **auto+cross** result in the L1-vs-CNN campaign (auto-only is
unaffected). **Decisive follow-up experiment defined below (flat-sky rebuild).**

This note documents why, at 10°, the **auto-only** posteriors of L1 and CNN essentially match
while the **auto+cross** posteriors differ a lot (CNN ≫ L1), and why that difference is driven by
an *unphysical* information leak in how the cross-maps are built.

---

## 1. The question

- Auto-only: L1 ≈ CNN (FoM3 ≈ 2200 vs 2343; contours match). True at both 10° and 20°.
- Auto+cross (10°): CNN ≫ L1 (FoM3 ≈ 17251 vs 8530; CNN gains ~7.4× over its auto-only, L1 only
  ~3.9×).
- Naïve expectation: a multi-channel CNN fed the tomographic maps κ₁…κ₄ as channels should be
  able to recover any useful cross-bin information *itself* (cross-correlations are just
  combinations of the input channels). So why does adding explicit cross-maps help the CNN so
  much — and why can't it get that information from the auto channels alone?

The answer: the explicit cross-maps carry information that is **not present in the local auto
patches** — it has leaked in from the rest of the sphere.

---

## 2. How the cross-maps are built (the root cause)

`scripts/sbi/build_full_sphere_cross_cache.py` (`cross_patches_from_alms`, ~L230–260) constructs
the 6 cross channels on the **full sphere** and only then cuts patches:

```python
a^i_lm = hp.map2alm(noisy κ^i, lmax=1024)          # full-sphere SHT (nside=512)
cross_alm[ij] = a^i_lm * a^j_lm                      # element-wise product of FULL-SPHERE coeffs
full_cross[ij] = hp.alm2map(cross_alm[ij], nside=512)# inverse SHT on the WHOLE sphere
patches = gnomonic_cutouts(full_auto + full_cross)   # THEN slice into 10° / 80px patches
```

There is **no apodization and no masking**. Because

  κ^{ij}(n) = Σ_{ℓm} a^i_{ℓm} a^j_{ℓm} Y_{ℓm}(n),   and   a_{ℓm} = ∫_{whole sphere} κ Y*_{ℓm},

**every pixel of every cross-patch is a global functional of the full-sky convergence in both
bins.** A 10°×10° cross-patch therefore encodes cross-correlation information from the entire
field, not just its own footprint. This is the leakage.

The **auto** channels do *not* leak: they are a linear SHT→iSHT roundtrip, so an auto-patch is just
the (band-limited) local convergence. (Method: `Harmonic_cross_maps.md`, Zürcher et al. 2022; the
patch-local alternative is `Flat-Sky_Tomographic_Cross_Maps.md`.)

---

## 3. Quantitative confirmation — cross channels are large-scale / non-local

Angular power-spectrum decomposition of the full-sphere fiducial maps
(`…/full_sphere_cache_fiducial_10deg/_snapshot/fullsphere_nobnt_cosmo_fiducial_perm0.npz`,
script `scripts/sbi/plot_cross_leakage.py`, figure `analysis/figs/D9_cross_leakage_scales`).
A 10° patch resolves ℓ ≳ 180/10 ≈ **18**; variance at ℓ<18 is *larger than the patch* and hence
coherent across it (the obviously non-local part).

| channel | % variance at ℓ<18 (super-patch) | ℓ_eff | ℓ_median |
|---|---|---|---|
| auto κ₁ | 0.4% | 655 | 699 |
| auto κ₂ | 0.7% | 620 | 663 |
| auto κ₃ | 1.0% | 568 | 599 |
| auto κ₄ | 0.9% | 542 | 562 |
| cross 1×2 | 12.2% | 454 | 461 |
| cross 1×3 | 13.7% | 386 | 312 |
| cross 1×4 | 12.7% | 367 | 270 |
| cross 2×3 | 19.7% | 266 | 87 |
| cross 2×4 | 18.1% | 252 | 83 |
| cross 3×4 | 17.7% | 186 | 60 |

- Cross channels carry **15–30× more super-patch variance** than autos (≈13% vs ≈0.8% on average).
- The cross field is **large-scale-dominated**: ℓ_median crashes to **60–90** for wide-separation
  pairs (vs ≈600 for autos). Mechanically this is because aⁱ·aʲ ∝ (steeply falling) Cⁱ·Cʲ, so the
  product piles power at low ℓ. This compounds the structural non-locality.

---

## 4. How this resolves the whole puzzle

1. **Auto-only ties.** Auto-patches are honest local measurements (ℓ_eff≈600, <1% super-patch);
   both compressors extract comparable per-patch information ⇒ tie. (Unaffected by the bug.)
2. **The CNN cannot get the cross info from the autos** — your intuition is right for *physical,
   patch-local* cross-correlation (a CNN with κ₁…κ₄ channels can compute local cross-statistics),
   but the cross channels carry **non-local** full-sky modes that are simply not in a 10° auto
   patch. The CNN can't reconstruct full-sphere a_{ℓm} from a cutout, so the cross channels add
   information unavailable from the autos ⇒ the auto→auto+cross jump does not close for the CNN.
3. **CNN ≫ L1 on auto+cross**, two compounding reasons:
   - **L1 is a per-channel statistic.** The wavelet ℓ₁-norm is computed on each map separately and
     concatenated; it never forms cross-channel combinations, so its only access to cross info is
     the standalone cross-map field's ℓ₁ histogram.
   - **Scale mismatch.** The cross field is large-scale (ℓ_median 60–90); ℓ₁-norm is a peak/
     small-scale statistic, and a smooth few-degree field gives few coarse wavelet coefficients per
     10° patch (sample-variance-limited). L1 is poorly matched to where the cross info lives; the
     CNN reads the large-scale patch structure directly with conv filters.
4. **The 20° "reversal" (old: L1 ≫ CNN on auto+cross).** Mostly **artifact**: the original 20°
   L1 lead used the broken cross-channel noise model (auto-σ on cross → inflated L1,
   `memory/project_l1_noise_model_correction`), plus a favorable single-perm draw and FoM3
   fragility (`memory/project_perm_averaging_overturns_l1_autocross_lead`); the corrected/robust
   20° result was already CNN ≳ L1. A plausible secondary (unproven) effect: a 20° patch captures
   more of the leaked large-scale cross field in-footprint and L1's wavelet scales reach larger
   physical scales there, so L1 reads the cross channel better at 20° than 10°.

---

## 5. What it means for the paper

- **Not a calibration bug.** The leakage is present identically in train and test sims, so the
  posteriors are honestly calibrated *with respect to the (leaky) data-generating process* — that
  is why TARP/SBC/L-C2ST all pass. It is a **data-vector realism** problem, not miscalibration.
- **The auto+cross constraining power is partly unphysical.** A real survey observing a 10° patch
  cannot build these cross-maps (no full sphere), so the auto+cross FoM3 — and especially the
  CNN's large gain — overstates what is physically achievable from patch data.
- **The relative ML comparison on this exact data vector is still valid**; what is compromised is
  the physical interpretation of the auto+cross numbers and possibly the CNN-vs-L1 ordering there.
- **Patches are not independent** across a realization (they share global modes), so the
  "9000 obs/arm" undercounts correlations in the cross arms (a secondary caveat for any per-patch
  population statistic on the cross channels).
- **Auto-only results stand clean** (autos are local): the auto-only "tie, both calibrated"
  conclusion is unaffected.

---

## 6. Decisive follow-up experiment — flat-sky (patch-local) cross-maps

Rebuild the 6 cross channels **per patch** with the flat-sky construction
(`Flat-Sky_Tomographic_Cross_Maps.md`): apodize the patch, `rfft2(κ^i)·rfft2(κ^j)`, `irfft2` ⇒
κ^{ij}_flat. This is strictly local (a function of the patch's own two auto maps). At 10°,
flat-sky is a reasonable approximation, so it is a fair construction. Then rerun the auto+cross
arms through the same Phase-C/D pipeline and compare gains over auto-only.

**Predictions / what it disentangles** (flat-sky gain = the *physical* cross information):
- **If leakage was the driver (expected):** the CNN's auto+cross gain largely collapses toward its
  auto-only level — because κ^{ij}_flat = κ^i ⊛ κ^j is a convolution the CNN could already compute
  from the auto channels — so the dramatic CNN≫L1 gap shrinks.
- **L1 may still gain** modestly from flat-sky cross-maps (they hand L1 the cross combination it
  cannot form itself), which would make the *physical* auto+cross story closer to (or even favor)
  L1 — the opposite of the leaky-data headline. This is the key reason the test matters.
- Whatever gain survives the flat-sky rebuild is the physically defensible auto+cross result.

**Scope:** dataset rebuild (cross channels only; autos unchanged) + retrain CNN + L1 auto+cross
arms + rerun Phase C/D. A campaign, not a quick run. Implementation note: the flat-sky cross can be
computed directly from the existing auto patches (no full-sphere maps needed), so it can be layered
on the current cache cheaply for the compute side; the expensive part is retraining + the per-patch
geometry/calibration sweep.

---

## 7. Artifacts & references

- Construction: `scripts/sbi/build_full_sphere_cross_cache.py` (`cross_patches_from_alms` L230–260).
- Scale evidence: `scripts/sbi/plot_cross_leakage.py` → `analysis/figs/D9_cross_leakage_scales.{png,pdf}`.
- Source maps: `results/exploratory/cross_maps_campaign/full_sphere_cache_fiducial_10deg/_snapshot/`.
- Method docs: `Harmonic_cross_maps.md` (full-sphere, used), `Flat-Sky_Tomographic_Cross_Maps.md`
  (patch-local, the proposed test).
- Related: `SUMMARY_PHASE_D.md` §6, `memory/project_cross_map_leakage_fullsphere.md`,
  `memory/project_patch_center_confound_g8.md` (route-sensitivity of the cross gain),
  `memory/project_10deg_definitive_cnn_geq_l1.md`.
