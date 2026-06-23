# HANDOFF — perfectly calibrate the CNN posterior (for the CNN-pipeline session)

**For:** the Claude Code session working on the CNN-VMIM pipeline.
**Objective:** remove the CNN posterior's residual mild *conservatism* so it is *exactly* calibrated
(pooled TARP net bias → ~0, SBC rank-std in-band ~0.289), **without tipping into over-confidence.**
This is a fair-comparison fix, not a bug fix — the CNN already PASSES the gate.

## 0. Why this matters (context)
We compare a learned CNN-VMIM compressor against analytical wavelet ℓ1 summaries (matched best-NDE
pipeline: VMIM→10-D→sbi_lens RealNVP, gated with TARP-DRP + SBC, FoM3 = 1/√det C₃ over a 9000-obs
fiducial population). The analytical **joint ℓ1** was just brought to a *calibrated tie* with the CNN
(joint ℓ1 3371 vs CNN 3326; see `analytical_nde_match/RESULT_JOINTL1_ENSEMBLE.md`). But that tie is
slightly flattered by a calibration asymmetry: the **CNN is mildly conservative** (over-covers), so its
FoM3 is mildly *under*-stated. A perfectly-calibrated CNN would tighten to ≥3326 ≥ joint ℓ1, restoring
the physically-expected "the near-optimal learned compressor is ≳ the best analytical statistic."
The goal here is to make the CNN comparison exact and fair.

## 1. The CNN's current calibration signature (diagnose the DIRECTION first)
From the pooled gate (all terciles + seeds), CNN ResNet-18 → sbi_lens RealNVP 4×128:
- **TARP-DRP: net bias ≈ +0.030 to +0.035, worst dev ≈ +0.027** → *positive = the ECP sits ABOVE the
  diagonal = OVER-coverage = conservative* (credible regions slightly too wide). It PASSES (within
  ±0.05) but is not exactly on the diagonal.
- **SBC rank-std: 0.290 / 0.289 / 0.282 (Ωm/σ8/w0)** vs uniform 0.289 → *marginals essentially ideal*
  (w0 a hair < 0.289, i.e. marginally conservative there too).
- **Reading:** the 1-D marginal *widths* are right, but the *joint* credible volume / correlation
  structure is slightly too large (the posterior is a touch too round/wide in the joint sense). This is
  a JOINT-coverage conservatism, not a marginal-width problem — important for choosing the lever.

(Reproduce with `plot_calibration_pooled_jointl1.py` / `plot_calibration_ensemble.py`, or read the
verdicts under `cnn_phase/arch_sweep_2026_06_13/gate_c_resnet18/`.)

## 2. Does the joint-ℓ1 calibration fix (compressor ensemble) translate? **NO — opposite direction.**
The joint ℓ1 was **over-confident** (SBC ~0.31 > 0.289, marginals too narrow). We fixed it with a
**3-compressor deep-ensemble**: pool the posteriors of 3 VMIM compressor seeds per obs, which
*diversifies the summary and WIDENS the posterior*, washing out the single-compressor amortization
over-confidence (→ SBC 0.30, TARP net −0.005, clean PASS). See `RESULT_JOINTL1_ENSEMBLE.md`,
`ensemble_eval.py`, `run_jointl1_bnt_ensemble.py`.

**That tool widens. The CNN is already (mildly) too wide.** Ensembling the CNN would make it *more*
conservative — the wrong way. So do **not** ensemble the CNN to calibrate it. (An ensemble is still
fine for robustness/seed-averaging, but it will not cure conservatism.)

What DOES carry over is the **methodology**, not the trick:
1. Gate first; read the *direction* (TARP net sign + SBC std vs 0.289).
2. Apply the lever that moves THAT direction (here: SHARPEN, not widen).
3. Re-gate; register the predicted direction before looking; stop when net→0 AND SBC in-band.
4. **Never overshoot into over-confidence.** Conservative is the safe miscalibration; over-confident
   (SBC ≫0.305, ECP below diagonal) is fool's gold — the count-histogram joint summary hit FoM3 ~4900
   that way and was rejected. A slightly-conservative CNN is strictly better than an over-confident one.

## 3. Candidate levers to SHARPEN the CNN (ranked; gate each)
The CNN currently pools 3 NDE (flow) seeds for 1 compressor seed. Pooling flows is itself a mild
*widening* (ensemble) — so several levers are the *inverse* of what helped the ℓ1.

1. **NDE-seed pooling (cheapest, most direct, try first).** Test the single best-val flow vs the
   3-seed pool. Pooling 3 flows averages slightly-different posteriors → wider/more conservative. A
   single well-trained flow (or 2) may sit exactly on the diagonal. *Registered expectation:* fewer
   pooled flows → TARP net drops toward 0; watch SBC doesn't cross into over-confidence (>0.305).
2. **Flow training / convergence / capacity.** An under-fit or early-stopped flow is conservative
   (hasn't tightened the joint). More `--flow-total-steps`, a longer patience, or modestly higher
   `--nde-hidden`/`--nde-layers` can sharpen the joint structure. Sweep capacity × steps, gate each
   (mirror `run_calib_sweep_jointl1.py` but read the TARP-net→0 / no-over-confidence target).
3. **Compressor quality.** Residual information the VMIM summary fails to expose widens the posterior.
   A better-trained / higher-`--summary-dim` compressor, or a VMIM companion-flow improvement, can
   tighten it. (Check `HANDOFF_CNN_OPTIMIZATION.md` for the arch/NDE levers already explored.)
4. **Post-hoc recalibration (direction-agnostic fallback).** If the training-level levers leave a
   residual, a learned recalibration can put it exactly on the diagonal. CAUTION: a single global
   temperature T<1 sharpens *all* dims uniformly, but the CNN's marginals are *already* ideal — a
   uniform sharpen would over-tighten them while fixing the joint. Use a **covariance-aware / per-axis
   or conformal** recalibration that targets the joint volume without narrowing the already-correct
   marginals. Validate by re-gating (TARP net→0 AND SBC still in-band). Treat as the last resort if you
   want the CNN reported as "calibrated by construction" rather than "recalibrated."

## 4. Objective, metric, and the bar
- **Target:** pooled TARP net bias |·| ≤ ~0.01 (from +0.03), worst-tercile dev ≤ 0.05, SBC rank-std
  ∈ [0.275, 0.305] on all three params (it already is — keep it there). I.e. move the CNN from
  "calibrated-but-conservative" to "on the diagonal".
- **Primary deliverable:** the perfectly-calibrated CNN's FoM3 + σ over the same 9000-obs population,
  so the CNN-vs-joint-ℓ1 comparison is exact. Expect the CNN FoM3 to *rise* modestly from 3326 (less
  over-coverage → tighter), likely to ≥ the joint ℓ1's 3371.
- **Hard guard:** do NOT trade conservatism for over-confidence. If a lever pushes SBC > 0.305 or TARP
  net negative beyond ~−0.02, back off — an over-confident CNN would spuriously inflate FoM3 and is
  worse than the current conservative one. Register the predicted direction before each gate.
- **Gate everything** (TARP-DRP + SBC), pooled over terciles+seeds; same protocol as the ℓ1 arms.

## 5. Pointers (cnn_sbi repo)
- Pipeline: `scripts/sbi/train_nde_from_compressed.py` (`--nde-family sbilens_realnvp`, `--seeds`,
  `--flow-total-steps`, `--nde-layers/-hidden`), `tarp_stratified_val_nde.py` → `run_tarp_coverage.py`
  → `analytical_nde_match/gate_verdict.py` (DEV_PASS 0.05, SBC band [0.275,0.305]).
- CNN arms: `cnn_phase/arch_sweep_2026_06_13/cnn_resnet18_s41/` (+ `gate_c_resnet18/`), compressor seeds
  s41/s42/s43 exist (useful for the seed-pooling test, lever 1).
- Calibration plots: `analytical_nde_match/plot_calibration_pooled_jointl1.py`,
  `plot_calibration_ensemble.py` (TARP ECP + SBC, pooled).
- Reference implementation of the ℓ1 calibration (the *widening* case, for contrast):
  `RESULT_JOINTL1_ENSEMBLE.md`, `ensemble_eval.py`, `run_calib_sweep_jointl1.py`,
  `run_jointl1_bnt_ensemble.py`. Memory: `project_joint_l1_matches_cnn`.
- Constraints: env `jaxili`; GPUs 0/1/2 only (never 3), check `nvidia-smi` for foreign tenants;
  never PCA the L1; gate everything; commit only when asked.

## 6. Expected outcome / paper impact
A perfectly-calibrated CNN tightens to ≥3326 (likely ≳ 3371), so the manuscript's "calibrated tie"
becomes "the CNN is the marginally tighter, as expected for a near-optimal compressor, and the joint
ℓ1 ties it to within calibration tolerance." That is the physically-sensible and referee-robust
framing. Feed the new CNN FoM3/σ + calibration numbers back to update `RESULT_JOINT_MATCHED.md` /
`RESULT_JOINTL1_ENSEMBLE.md` and the manuscript §5 Table 1 / §6 / abstract (paper repo:
`~/papers/L1_vs_CNN_Tomographic_SBI`, now under git).
