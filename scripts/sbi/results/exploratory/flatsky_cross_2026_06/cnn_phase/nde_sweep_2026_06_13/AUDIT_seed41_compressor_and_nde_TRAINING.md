# Independent training audit — seed-41 auto-only CNN compressor + its NDE

**Auditor stance:** independent, adversarial, read-only. No input from the pipeline authors;
every number below was re-derived from the on-disk artifacts using
`CUDA_VISIBLE_DEVICES="" /home/tersenov/anaconda3/envs/jaxili/bin/python` (CPU-only numpy reads).
**Date:** 2026-06-13. **Subject:** `flatsky_cross_2026_06/cnn_phase/cnn_none_s41` (compressor)
and `nde_sweep_2026_06_13/B1_jaxili_maf_oracle` (NDE on its frozen summaries).

**Status note:** the B1 NDE run is *still live* (PID 1684875, GPU 2). Seeds 41 and 42 finished
training; seed 43 is mid-train and the 9000-obs population sampling phase has not produced its
median yet. Findings on the two completed seeds are firm; the NDE verdict is "PASS so far".

---

## OVERALL VERDICT: PASS (with two referee-flaggable caveats, neither a training defect)

The compressor converged, the **best_val checkpoint genuinely is the test-loss minimum** (the
documented silent-last-step failure mode did NOT occur — verified by byte-identity + sha256 +
stdout), there is no BatchNorm, no PCA, no log-on-negatives bug, train/val cosmologies are
disjoint, and the NDE is converging cleanly with no NaN retries. The two caveats are (a) the
summary carries only ~3 effective dimensions, and (b) by design this "oracle" arm trains the NDE
on the same examples the compressor saw (no example-level holdout) — defensible for an oracle
baseline but a referee will ask about it.

---

## COMPRESSOR CONVERGENCE — PASS

Loss curves: `loss_compressor_{train,test}.npy`, 40 points each, logged every 2000 steps over
80000 steps (cadence confirmed against the stdout log and source line `if step % save_every == 0`).

- Train loss: −8.77 → −10.47 (min −10.71 @ idx 30). Test loss: −8.65 → −10.41 (min **−10.722 @ idx 18 = step 38000**).
- The curve has clearly plateaued: the last ~22 test points oscillate in [−10.72, −10.41] with no
  sustained further descent. Not under-trained, not diverging.
- VMIM loss is a negative-MI surrogate; the ~2-nat drop from init and the flat tail are the
  expected convergence signature.

**Evidence (re-derived):** test argmin = 18 of 39; test min = −10.7224 (matches the stdout
`best-val ... val_loss=-10.7224` to 4 dp).

## CHECKPOINTING — PASS (the headline good news)

This is the dimension most likely to hide a silent failure. It is clean.

- `cnn_cache_meta.compressor_params_path` → `params_nd_compressor_best_val.pkl`.
- On-disk sha256 of that file = `4ff38813…05abf67` = **exactly** the recorded
  `compressor_params_sha256`. Provenance intact.
- `best_val.pkl` is **byte-for-byte identical to `params_nd_compressor_batch38000.pkl`** (sha256
  match), and step 38000 = test-loss-curve index 18 = the test-loss minimum. So best_val is the
  argmin checkpoint, **not** the last-step (step 80000) checkpoint.
- Stdout confirms: `Saved best-val checkpoint @ step 38000 (val_loss=-10.7224)` and
  `Compressor returning policy=best_val step=38000`. The code at npe_cnn_nbody_tomo.py:3179-3183
  tracks a running argmin and dumps those exact params; the effective-policy guard
  (`checkpoint_policy_effective`, line 3267) would have caught a fallback — it did not fire.

The known "believes best_val but silently hands off last_step" failure mode is **absent here**.

## OVERFITTING — PASS

- Final train−test gap (last logged point): −10.475 − (−10.406) = **−0.069 nat** — negligible,
  and *test* is slightly worse than *train* as expected, not a blown-open gap.
- Test loss drifts UP by **+0.316 nat** from its idx-18 minimum (−10.722) to the end (−10.406).
  This is mild late-training test drift — exactly the situation best_val selection exists to
  handle, and it did: the handed-off checkpoint is the pre-drift minimum, so the drift does not
  contaminate downstream. No action needed.

## DATASET / SUMMARIES — PASS (with effective-rank caveat)

Frozen cache `cnn_train.npz` (323640×10), `cnn_val.npz` (504000×10), `cnn_obs.npz` (10,).

- **Health:** 0 NaN, 0 Inf in all three. No dead/near-constant dims (per-dim std 0.07–0.29).
  ~60% of entries negative, range ≈ [−1.75, +1.58] — important for the preprocessing check below.
- **theta coverage:** train spans 899 distinct cosmologies (~360 maps each); val spans 400
  distinct cosmologies (~1260 each). All 6 parameters span the full prior box
  (e.g. Ωm [0.10,0.50], σ8 [0.40,1.40], w0 [−1.93,−0.33]). NOT cosmology-starved.
- **obs datum:** exactly the fiducial [0.26, 0.84, −1.0, 0.6736, 0.9649, 0.0493], **not** in the
  training grid (nearest train cosmology L2 = 0.0785; 0 rows within 1e-3) — a genuine held-out
  inference point. Its summary is in-distribution (all dims inside train range, max |z| = 0.85).
- **CAVEAT — low effective rank:** the 10-dim summary is far from orthogonal. Correlation-matrix
  condition number ≈ 301 (cov ≈ 318); top two corr-eigenvalues (4.79, 3.06) dominate; bottom four
  are 0.016–0.083; **participation-ratio effective dimensionality ≈ 3.0 / 10**. Strongest pairwise
  |corr| = 0.88 (no pair > 0.9, so nothing exactly collinear/dead). This is consistent with a
  3-parameter-dominated (Ωm,σ8,w0) science target and is not a training fault, but a DL referee
  may note the compressor uses ~3 of its 10 output slots. Worth a sentence in the paper.

## LEAKAGE — CONCERN (by design; disclose, don't hide)

- **Train↔Val cosmologies are fully disjoint:** 899 train vs 400 val distinct cosmologies, **0
  overlap**. So the held-out validation/inference is honest at the cosmology level. Good.
- **Compressor↔NDE example-level overlap EXISTS by design.** `cnn_cache_meta` records
  `compressor_train_split = nde_train_split = "train"` and `require_disjoint_train_examples = 0`.
  The disjointness audit (npe_cnn_nbody_tomo.py:4482-4520) only runs when that flag is set — here
  it is 0, so **the NDE is trained on summaries of the exact same maps the compressor was trained
  on.** This is intentional: `PLAN_CNN_NDE_SWEEP_2026-06-13.md:106` names B1 "the ORACLE: must
  reproduce 2325" — an upper-bound baseline, and the plan itself (line 152) lists "check the
  compressor↔NDE split for leakage/overlap" as a to-do.
- **Risk assessment:** for an oracle/ceiling arm this is acceptable and conventional (VMIM
  compressor + NPE on the same sims is the standard sbi_lens setup). The danger is only if the
  oracle FoM3 is later quoted as the *honest* number. It should be reported as an oracle, and any
  headline science number should come from an arm with `require_disjoint_train_examples=1` (or be
  shown to match the oracle, which is the plan's own back-pressure test). Rated CONCERN purely so
  it is not silently carried into the paper as a clean held-out result.

## NDE (B1 jaxili MAF) — PASS (so far; run still live)

Driver: `population_sweep_flatsky.py` (command recovered from the live process table), arch from
`ckpts/s*/…/hparams.json`.

- **Architecture:** `ConditionalMAF`, 5 layers, hidden [50,50], relu, n_in=6, n_cond=10. Adam
  lr=1e-4, warmup 100, gradient_clip=5.0, weight_decay=0. jaxili-internal 70/20/10 split
  (226548 / 64728 / 32364). Standard, matches the L1 arms' NDE for apples-to-apples.
- **Preprocessing (the negatives trap):** launch flag is **`--preproc-transform none --clip-value 0`**.
  `preprocess_summaries` therefore does identity (no log, no external z-score), and jaxili's
  embedding-net `Standardizer` does the z-scoring. **Proof it's not log1p:** the hparams
  Standardizer `mean` = [−0.0848, −0.2166, −0.2340, 0.1196, 0.2337, 0.5608, −0.2807, −0.3534,
  0.0639, −0.0832] equals the **raw** train per-dim mean I computed to 3 dp. A `log1p` would
  additionally have *crashed* (guard at npe_l1norm_cross…:1508 raises on values < −1, and the
  cache min is −1.75). So negatives are handled correctly. The observed datum gets the same
  transform + same train mean/std → no train/NDE/obs preprocessing mismatch.
- **Convergence / stability:** 3 seeds, each "ATTEMPT 1/10" (NaN-retry mechanism never fired; 0
  "NaN" in the log). Best val loss: seed41 **−13.51** (early-stopped @ epoch 788), seed42 **−13.09**
  (early-stopped @ epoch 493), seed43 −12.88 and still descending. Seed spread ≈ 0.4 nat = tight.
  Early stopping (patience=20, min_delta=1e-3) engaged correctly; well under the 50000-epoch budget,
  so the NDE is not budget-starved. Monotone val improvement from −9.69. No divergence.
- **Pending:** seed-43 finish + the 9000-obs pooled-posterior median FoM3 (the actual "reproduce
  2325" oracle check) are not yet in the log. Re-confirm the FoM3 lands near 2325 when it finishes.

## KNOWN-PITFALL SWEEP

| Pitfall | Verdict | Evidence |
|---|---|---|
| BatchNorm on cosmology-mixed batches | **PASS** | arch = `plain` (CompressorCNN2D, npe_cnn…:1982): Conv→leaky_relu×3 + AvgPool + 2×Linear. **No norm layer at all.** The BN-collapse mode (resnet18/34/50) is not on this path. |
| PCA on summaries before NDE | **PASS** | No PCA in `population_sweep_flatsky.py` or `train_jaxili_from_compressed.py`; log shows `train(323640,10) -> dim 10` (10→10, no reduction). |
| log/log1p on negative summaries | **PASS** | `--preproc-transform none`; jaxili Standardizer mean == raw mean (proves no log); and the code guards/raises on <−1 anyway. |
| FoM predicted from val-loss | **PASS (in this arm)** | This arm computes FoM3 directly from sampled posteriors (`compute_fom3` on 3-seed-pooled samples), not from val-loss. (Project memory `L67` documents that val-loss is NOT a cross-arch FoM3 proxy — relevant when *comparing* B1/B2/B3, not for this single arm's internal validity.) |
| train/NDE/obs preprocessing mismatch | **PASS** | obs uses the same `transform=none` + same train-derived mean/std (population_sweep_flatsky.py:76-79); obs summary max|z|=0.85, in-distribution. |
| Checkpoint silent last-step fallback | **PASS** | best_val == batch38000 == test-loss argmin (sha256 + byte-identity + stdout). |

**Additional referee-attackable points not on the supplied list:**
1. **Effective rank ≈ 3 of 10** (DATASET caveat) — pre-empt by reporting it as expected for a
   3-param FoM3 target.
2. **Oracle in-sample NDE training** (LEAKAGE) — must be labelled "oracle / upper bound", not
   "held-out", in any paper table.
3. **VMIM companion-flow quality** is not auditable from these artifacts (the compressor's
   internal NF that defines the VMIM objective). Project memory notes the sbi_lens RealNVP
   companion is the chosen default; the compressor loss looks healthy, so this is low-risk, but a
   referee could ask whether the MI estimate (and hence the summary) is companion-limited. The
   −10.72 plateau and the clean downstream NDE argue no, but it's not provable from disk here.
4. **val > train rows** (504k val vs 324k train) is unusual; confirm this is the intended
   fiducial-heavy validation design and not a swapped split. (Cosmologies are disjoint, so it is
   not a leak — just worth a sanity glance.)

---

## PRIORITIZED RED FLAGS / FOLLOW-UPS

1. **(disclosure, not a bug)** Label B1 as the **oracle** everywhere — NDE trained in-sample
   (compressor↔NDE example overlap, `require_disjoint_train_examples=0`). The honest science
   number must come from a disjoint-split arm or be shown to match the oracle.
2. **(finish the run)** Re-confirm the 9000-obs pooled-median FoM3 lands at ≈2325 when seed-43 +
   sampling complete. That is the run's own back-pressure oracle; until then the NDE verdict is
   provisional.
3. **(paper hygiene)** State the summary's ~3 effective dimensions / condition number ≈ 300 up
   front, so it's framed as expected rather than discovered by a referee.
4. **(low-priority sanity)** Confirm the 504k-val / 324k-train asymmetry is intended.

**Bottom line:** the seed-41 compressor and its NDE were trained soundly. The one thing that
*looks* alarming on a first pass — a "best_val" file written at the end of training, same size as
every step checkpoint — is a false alarm: it is provably the step-38000 test-minimum checkpoint.
The only real "watch item" is the deliberate oracle in-sample design, which is a reporting
question, not a training error.
