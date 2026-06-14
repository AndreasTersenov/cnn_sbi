# PLAN — 2D-1D wavelet ℓ1-norm, Phase 1 (Approach A) + Phase 2 (Approach B)

**Date:** 2026-06-13 (overnight). Implements TOMO_2D1D_WAVELET_RESEARCH.md §8. Andreas signed off:
"write the plan for phase 1 and 2 and start implementing overnight."

## Objective & decision metric
Build the tomographic 2D-1D wavelet ℓ1-norm and measure whether it (1) recovers more cross-bin
information than per-bin L1 (goal 1: FoM3 + marginals), and (2) is robust to BNT (goal 2: does the
FoM3 inflate moving from no-BNT to BNT space?). **Primary metric:** pooled-3-seed / 9000-obs MEDIAN
FoM3 (the campaign primary metric, via `population_sweep_flatsky.py`), reported with σ(Ωm,σ8,w0)
marginals-first. **Apples-to-apples:** every arm runs through the SAME common jaxili MAF + SAME
preproc (log1p-zscore / clip5 / min-var1e-5) as the existing arms. Baselines (verified, same path):
flat_none (auto-only) **2404.6**, flat_product (L1+ξ_ij) **2875.3**. Calibration (TARP+SBC) MANDATORY
— uncalibrated FoM3 gains do not count (LANE_A_CONCLUSION.md).

## Key mechanism (verified from code)
`flatsky_cross_l1.py` + `flatsky_cross.py`: passing `bnt=<ndarray (rows,4)>` applies that matrix as a
mix over the 4 autos (`out[...,i]=Σ_j M[i,j]κ_j`), `op="none"` → channels = M·autos, then wavelet ℓ1
with a per-channel frozen σ. **So Approach A = pass the Haar matrix as the mix.** Because both the
spatial starlet and the bin-mix are linear, "Haar across BNT channels" is the single combined mix
`M = Haar·B` over the autos — no separate BNT cache/σ-table needed for the UNCUT case; the auto-basis
noise propagates by quadrature `σ²_m = Σ_b M[m,b]² σ²_auto,b` (the verified postcut-arm convention).

The orthonormal 4-bin Haar (rows over autos; deep mode included — the highest-value channel):
```
H = [[ 0.5,  0.5,  0.5,  0.5],   # m0 deep mode  ¼Σκ  (×2; high S/N)
     [ 0.5,  0.5, -0.5, -0.5],   # m1 coarse diff (12)-(34)
     [ 1/√2,-1/√2, 0,    0  ],   # m2 fine diff κ1-κ2
     [ 0,    0,    1/√2,-1/√2]]]  # m3 fine diff κ3-κ4
```
Per-row normalization is immaterial to the S/N-binned ℓ1 (σ scales with the row), so orthonormal is
fine; the row STRUCTURE (which combinations) is what matters.

---

## PHASE 1 — Approach A (pure 2D-1D Haar wavelet ℓ1-norm). The cheap, faithful, diagnostic step.

### Arms
| arm | mix M (over autos) | what it tests | registered prediction |
|---|---|---|---|
| `haar_nobnt` | Haar (4×4) | faithful 2D-1D Haar ℓ1, no-BNT (goal-1 baseline) | FoM3 2900–3300 |
| `autohaar_nobnt` | flat_none autos ⊕ Haar channels | augmented (likely best for goal 1) | ≥ haar_nobnt, ≈ or > 2875 |
| `haar_bnt_uncut` | Haar·B (4×4) | 2D-1D Haar ℓ1 in (uncut) BNT space (goal-2 frame test) | ≈ haar_nobnt if BNT-robust |

`autohaar_nobnt` = concatenate the existing `flat_none` cache (autos) with `haar_nobnt`'s channels
(theta is bit-identical by construction — same loader params), no extra build pass.

### Pipeline per arm
1. **Build** (`build_flatsky_haar_arm.py --mix {haar,haar_bnt}`): train (perms 5–6, flip, seed 1001),
   val (test split, perms 0–1, noflip, seed 2001), fiducial (200 perm files) — identical loader
   params to flat_none ⇒ theta bit-aligned. Per-channel quadrature σ from the no-BNT frozen table;
   per-channel SNR ranges via `calibrate_snr_range_flat_local`. Output: `<arm>/cache/l1_{train,val}.npz`
   + `fiducial_summaries_<arm>.npz`.
2. **Sweep** (`population_sweep_flatsky.py`): 3 NDE seeds, 9000 obs, pooled FoM3 median + σ + 2D.
3. **Gate** (`tarp_stratified_val.py` + `run_tarp_coverage.py` + verdict): TARP terciles (3 seeds) + SBC.
   Clone of `run_laneB_gate_c.py` with the Haar ARMS list → `GATE_C_2D1D.md`.

### Comparison & reading
- Goal 1: `haar_nobnt` / `autohaar_nobnt` FoM3 + marginals vs flat_none 2405, flat_product 2875.
- Goal 2: `haar_bnt_uncut` FoM3 vs `haar_nobnt`. Close ⇒ BNT-robust (contours don't inflate);
  large drop ⇒ the Haar-ℓ1 still loses the deep mode under BNT (a real, reportable result).
- All claims gated; a tight-but-miscalibrated arm is downgraded (LANE_A precedent).

### Back-pressure / tripwires
- Per-channel σ table PRINTS (no fallback warning); `pca_applied` N/A (L1 route never PCAs; min-var
  mask only). Verify dim = 4×5×40 = 800 (haar) / 1600 (autohaar).
- theta bit-equality asserted vs flat_none (build aborts otherwise).
- Smoke-test the build on 1 fiducial file before the full 324k pass.

---

## PHASE 2 — Approach B (intermediate modulus; scattering-structured). Drafted; run if Phase 1 done + GPU free.

### Construction
`2D starlet → |·| → Haar across bins → S/N-binned ℓ1`. The modulus sits BETWEEN the spatial starlet
and the bin-mix, so the Haar mixes the *modulus fields* `|S_{j1}κ_b|` — nonlinear in the maps, NOT
reducible to a starlet of a linear combination (escapes the linear ceiling). Gain is measured against
Phase 1's `haar_nobnt` as the known linear baseline (the payoff of doing A first).

### Implementation spec (code-level; locked 2026-06-13 from the WLStatistics API)
Cannot use the `bnt=M` mix path (that mixes the INPUT maps, before the wavelet). The bin-Haar must
act on `|wavelet coeffs|`, in coefficient space. Concrete transform (`flatsky_haar_scatter.build_and_l1`):
```
autos (B,H,W,4) on GPU
for b in 0..3:  stats.compute_wavelet_transform(autos[...,b], 1.0, subtract_coarse_mean=True)
                Wc[b] = stats.wavelet_coeffs            # (B, n_scales, H, W)
aWc = Wc.abs()                                          # (4, B, n_scales, H, W)  ← Jean-Luc's |·|
J = einsum('mb, b B s H W -> m B s H W', H_haar, aWc)   # (4 modes, B, n_scales, H, W)
for m in 0..3:
    stats.wavelet_coeffs = J[m]                         # the field whose |·| the L1 sums
    stats.noise_levels   = sigma_mode[m].view(1,n_scales,1,1).expand_as(J[m])
    stats.snr_coeffs     = J[m] / stats.noise_levels
    _, l1 = stats.compute_wavelet_l1_norms(n_bins=L1N, min_snr=rng[m,0], max_snr=rng[m,1],
                                           clamp_overflow=True)   # mirrors _l1_with_frozen_sigma
    out.append(cat(l1))                                 # (B, n_scales*L1N)
return cat(out, -1)                                     # (B, 4*n_scales*L1N)
```
This reuses the EXACT `compute_wavelet_l1_norms` seam that `_l1_with_frozen_sigma` (flatsky_cross_l1.py:105)
uses — set `wavelet_coeffs`/`noise_levels`/`snr_coeffs`, then call it.

**Noise model (the genuinely new piece): per-(mode,scale) σ MUST be estimated empirically**, not by
quadrature — the modulus folds the Gaussian (|N(0,σ)| is half-normal, mean σ√(2/π)), so `σ²_m = ΣH²σ²`
is WRONG for the modulus-Haar fields. Freeze pass `freeze_haar_scatter_noise.py` (clone of
freeze_flatsky_cross_noise.py): run the modulus-Haar transform on ~48 PURE-NOISE realizations (shape
noise only, no signal — same noise generator the flat-sky pipeline uses), take per-(mode,scale)
`std` of the field J_m over (realization, pixel) → `sigma_mode (4, n_scales)`. Then SNR ranges via
the same `calibrate_snr_range`-style percentile pass on the J_m fields. Tripwire: if you reuse the
quadrature σ, the deep-mode (sum-of-moduli, strictly positive, large mean) SNR will be miscentred and
the histogram will pile in edge bins — verify the printed σ table and the SNR ranges look like the
Phase-1 ones (deep wide, diffs narrow) but shifted positive for the sum modes.

**Arms reuse the Phase-1 build/sweep/gate scaffold** (`build_flatsky_haar_scatter_arm.py` ≈
`build_flatsky_haar_arm.py` with `build_and_l1` swapped for the modulus-Haar one + the empirical σ).
Same loader params ⇒ theta bit-aligned ⇒ same population_sweep + gate path. `--smoke` first.

### Arms
| arm | what it tests | registered prediction |
|---|---|---|
| `haarscat_nobnt` | modulus-Haar ℓ1, no-BNT | ≥ haar_nobnt; vs CNN/L1+product ceiling = open |
| `haarscat_bnt_uncut` | modulus-Haar ℓ1, BNT | plausibly > haar_bnt_uncut (sum-of-moduli no sign cancellation) — the headline goal-2 test |

### Done condition (both phases)
Phase 1: all 3 arms built, swept, gated; comparison table + reading written to a result doc; memory +
felt updated. Phase 2: code written + smoke-passed; run `haarscat_nobnt` + `haarscat_bnt_uncut` if
Phase 1 finishes with GPU/time to spare, else leave built-and-queued with a morning note.

## GPU / runtime
GPUs {0,1,2}; tonight GPU2 is clean (0% util), GPU0 has a foreign tenant (39%, headroom), GPU1 busy
(66% — avoid). Builds use torch (mem bounded ~few GB by the 6144-map cap); sweeps/gates use jaxili
(XLA_PYTHON_CLIENT_MEM_FRACTION ≈ 0.3–0.42 when packing). Pack ≤2–3 jobs/GPU. CPU ≤50 workers
(OMP/MKL=8/job). Run detached (titan has no scheduler). Fresh `nvidia-smi` before each launch wave.

## Morning deliverables
`RESULT_2D1D_PHASE1.md` (FoM3 + marginals + gate verdicts + goal-1/goal-2 reading vs 2405/2875),
`GATE_C_2D1D.md`, updated felt stanza + memory if a durable finding lands, and a clear statement of
where Phase 2 stands. Honest scoring against the registered predictions above.
