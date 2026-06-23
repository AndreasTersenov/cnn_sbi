# The wavelet joint ℓ1: definition, relation to the full joint PDF, and BNT behaviour

Definitional + interpretive note for the "joint ℓ1" statistic (the analytical summary that ties the
CNN, FoM3 3371 ≈ 3326, and is far more BNT-robust than ℓ1+product). Companion to
`BNT_THEORY_DEEP_DIVE.md` (the BNT frame/shadow picture) and `RESULT_JOINTL1_ENSEMBLE.md` (the numbers).

## 1. Definition

Setup. We have `C=4` tomographic convergence maps. At each starlet wavelet scale `s`, the per-pixel
across-bin coefficient vector in S/N units is

  u(p) = ( u₁, …, u_C )(p),   uᵢ(p) = (W_s κᵢ)(p) / σᵢ,s

where `W_s` is the wavelet filter at scale `s` and `σᵢ,s` the frozen per-(bin,scale) noise level.

**Standard wavelet ℓ1 (per channel).** For each bin `i` and scale `s`, bin the pixels by their S/N
`uᵢ` into K 1-D bins and, in each bin, sum `|uᵢ|`. The datavector is these K numbers per (bin,scale).
It is the |·|-weighted 1-D histogram of the coefficients — a robust summary of the *one-dimensional
marginal* of `u` along axis `i`.

**Joint ℓ1 (this statistic).** For each *bin pair* `(i,j)` and scale `s`, lay a fixed K×K grid over the
joint `(uᵢ, uⱼ)` plane (signal-adapted percentile ranges) and, in each cell, **sum the ℓ1 weight
½(|uᵢ|+|uⱼ|)** over the pixels that fall in it (NOT the pixel count). The datavector is these K² cell
values over all `C(C,2)=6` pairs × S scales, then VMIM-compressed to 10-D and fed to the same RealNVP.
Code: `flatsky_joint_stats.py` (`pair2d_features(..., weighted=True)`, line ~161); `stat="jointl1"`.

So the joint ℓ1 is the **2-D pairwise generalisation of the wavelet ℓ1**: the per-channel ℓ1 is the
|·|-weighted 1-D histogram of each axis; the joint ℓ1 is the |·|-weighted 2-D histogram of each pair.

**Two siblings, same grid, different cell content** (the distinction that controls calibration):
- `pair2d` — cells hold the **count** → the joint PDF estimate (the raw 2-D histogram).
- `jointl1` — cells hold the **ℓ1 sum** → the ℓ1-weighted 2-D histogram.
The count version over-fits (it hit FoM3 ~4900 and FAILED coverage); the ℓ1-weighted version stays
calibratable (it emphasises the tails/peaks the ℓ1 lives on and is lower-variance than dense counts).

## 2. How it approximates the full joint PDF

All the one-point information about θ at scale `s` lives in the **full joint PDF** `p_s(u₁,…,u_C)` of
the coefficient vector — the shape of the C-dimensional point cloud (all bins, jointly, all moments).
A summary statistic captures a *projection* of this cloud:

| statistic | what it captures of `p_s(u)` | "dimension" of the projection |
|---|---|---|
| per-channel ℓ1 | the C one-dimensional **marginals** (|·|-weighted) | 1-D |
| + products κᵢκⱼ | adds the pairwise **2nd cross-moments** ⟨uᵢuⱼ⟩ (= ξᵢⱼ) | a single number/pair |
| **joint ℓ1** | the 6 **two-dimensional pairwise marginals** `p_s(uᵢ,uⱼ)` (full 2-D shapes) | 2-D |
| full 4-D joint PDF (`full4d`) / CNN | the **entire C-dimensional cloud** | C-D |

The joint ℓ1 is therefore the **pairwise projection** of the full joint PDF. It contains *strictly
more* than ℓ1+product: the per-channel marginals are its axis-projections, the products are its 2nd
cross-moments, and on top of those it carries every *higher* pairwise joint moment — the cross-
skewness, cross-kurtosis, the full non-Gaussian 2-D shape of each pair. What it does **not** capture is
the genuinely *≥3-way* joint structure (connected moments like ⟨uᵢuⱼuₖ⟩ and the full 4-D cloud shape
that cannot be rebuilt from pairwise marginals). For a mildly non-Gaussian field most of the joint
information sits in the marginals and the pairwise structure, with the 3-/4-way connected part smaller,
which is why the pairwise projection already captures most of what the full PDF holds. The ℓ1 weighting
is a robust, low-variance reduction of each 2-D histogram (the joint analogue of the 1-D wavelet ℓ1),
which is what makes the projection *estimable and calibratable* rather than a sparse count grid.

## 3. Why it is much closer to lossless under BNT than ℓ1 + product

BNT is a fixed, invertible linear map `B` on the maps: `κ̃ = Bκ`, so by wavelet linearity the
coefficient vector rotates, `ũ = Bu`, at every pixel. The full joint PDF is **basis-covariant**:
`p̃_s(ũ) = p_s(B⁻¹ũ)` — the cloud is rotated/sheared but its shape, and all the information, is
preserved (this is the `P1`/`P4b` content of `BNT_THEORY_DEEP_DIVE.md`). The cosmological information
is conserved; what changes is *how much of it a given projection can see in the rotated frame.*

This is the whole story, and it is a statement about **projection dimension under rotation**:
- **1-D marginals are the most frame-fragile.** BNT rotates the axes onto its "no deep direction"
  frame — one shallow map and three thin, signal-starved slices — so *every* single-axis shadow goes
  nearly blank at once. The per-channel ℓ1 reads only these shadows, so it collapses (×0.15–0.26).
- **The 2nd moment (products) buys back only the Gaussian sector.** Adding ξᵢⱼ restores the pairwise
  *covariance* — one number per pair — in the BNT frame. With the blanked marginals this reconstructs
  essentially the two-point (Gaussian) content, which `BNT_THEORY_DEEP_DIVE.md` measures as ≈0.38 of
  the per-channel loss. Hence ℓ1+product still retains only ×0.26: marginals collapse, products add
  back the Gaussian part, the non-Gaussian joint structure is still unseen.
- **2-D pairwise marginals are far more rotation-robust.** A 2-D projection of a rotating cloud retains
  the joint shape much better than a 1-D projection: even when each axis is signal-poor, the 2-D plane
  still shows the correlations *and* the non-Gaussian joint shape. The joint ℓ1 reads these full 2-D
  shapes, so it restores the entire pairwise non-Gaussian sector, not just the covariance → ×0.72.
- **The full C-D PDF is exactly frame-invariant** (any rotation is undone by relabelling the C-D
  cells) → lossless. The CNN approximates this by *learning the 4-channel mixing* in its first layer,
  so it reads the full joint and is near-lossless (×0.96).

So the retention ladder **0.26 → 0.72 → 0.96** is simply the **dimension of the joint projection** the
summary captures: 1-D marginals + 2nd moment → 2-D pairwise → full C-D. Each step up in projection
dimension is more robust to the BNT rotation, because more of the cloud's structure is carried in the
higher-dimensional, frame-covariant projection rather than in the fragile 1-D shadows. The products
help little under BNT (one number per pair) while the joint ℓ1 helps a lot (the whole 2-D shape per
pair) for exactly this reason.

## 4. What it would take to make it completely BNT-lossless

The gap `0.72 → 1.0` is, by the picture above, precisely the **≥3-way joint structure** the pairwise
projection misses: the connected 3- and 4-way moments / the full 4-D cloud shape. To close it one must
climb the projection-dimension ladder, and the only real obstacle is doing so *calibratedly*:

1. **Triple-wise joint ℓ1 (3-D) and ultimately the full 4-D joint.** The full 4-D joint (`full4d`) is
   *exactly* basis-covariant → completely lossless in principle. The obstacle is the curse of
   dimensionality: a Kᶜ grid over ~6400 pixels is sparse, and the dense count version over-fits (it
   FAILED the gate). Routes that keep it estimable: a **coarse K**, ℓ1-weighting rather than counts,
   and adding only the 3-way term (`triple` joint ℓ1) for the leading connected structure rather than
   the full 4-D grid at once. Expectation: 3-way restores a further chunk (→ ~0.8–0.9), with
   diminishing returns.
2. **A compact set of higher joint cumulants instead of dense histograms.** The 3rd/4th-order cross-
   cumulants ⟨uᵢuⱼuₖ⟩_c, ⟨uᵢuⱼuₖuₗ⟩_c (and their |·|-weighted analogues) are *single robust numbers*,
   basis-covariant by construction (they transform as tensors under `B`), and carry exactly the
   connected ≥3-way information the pairwise misses — without the sparsity that sinks the dense 4-D
   histogram. Appending a modest set of these to the joint ℓ1 is the most promising *calibratable*
   route to higher retention. Gate each (the calibration bar, not raw FoM3, is what decides).
3. **A fixed full-4-channel front-end before the ℓ1 (the CNN's trick, made explicit).** The CNN is
   lossless because it mixes all four channels *before* the per-pixel non-linearity. The analytical
   analogue is a fixed 4-channel rotation/whitening applied before the joint ℓ1. We tested the pairwise
   (2-D) version of this — per-pair whitened binning — and it did **not** help calibration
   (`RESULT_JOINTL1_ROTATED.md`); the full 4-D fixed front-end is either the trivial rotate-back (which
   undoes the BNT nulling, so it is not cut-compatible) or must be *learned*, at which point it is the
   CNN. This is the wall: a *fixed* analytical statistic cannot supply the learned 4-D mixing.

**Honest ceiling.** Routes 1–2 should push retention from 0.72 toward ~0.85–0.9 while staying
calibrated, but full BNT-losslessness (×0.96+) likely belongs to the CNN: the last sliver is the
learned 4-channel mixing that no *fixed* low-variance analytical statistic supplies. The scientifically
clean message is the ladder itself — *how much inter-bin information survives BNT for a given summary
tracks how jointly it reads the bins* — with the joint ℓ1 as the best fixed analytical point on it.

## 5. Pointers
- Code: `flatsky_joint_stats.py` (definition: `pair2d_features` weighted; `full4d_features`;
  `calibrate_joint_rotation` = the rotated/whitened variant of route 3).
- Numbers/calibration: `analytical_nde_match/RESULT_JOINTL1_ENSEMBLE.md`, `RESULT_JOINT_MATCHED.md`,
  `RESULT_JOINTL1_ROTATED.md`; spine `PAPER_MESSAGES.md` M1/M3; memory `project_joint_l1_matches_cnn`.
- BNT frame/shadow theory (the rotation picture, the 0.38 Gaussian share, P1/P4b): `BNT_THEORY_DEEP_DIVE.md`.
