# Pipeline audit — flat-sky L1-vs-CNN analysis (2026-06-10)

Four parallel read-only audits of the flat_local campaign analysis chain, run while the
multi-compressor-seed check was in flight. Scope set by Andreas (handoff §6.4): (a) example
disjointness end-to-end, (b) the per-channel RMS-whitening path, (c) population-sweep MAF
retraining / aggregation / silent fallbacks, (d) the ~2 h/arm sampling throughput.
**No fixes applied — findings only, pending sign-off.**

## Bottom line

**The published numbers stand.** No contamination, no leakage, no aggregation artifact was found
that affects `FLATSKY_CNN_RESULT.md`, `FLATSKY_CROSS_RESULT.md`, or any `median_summary.json`:
the perm split is metadata-driven and runtime-audited, the whitening transform is provably
bit-identical across train/compress/obs, all 8 arms have n = 9000/9000 finite obs, `best_val`
checkpoints were genuinely used (steps 38k/40k), no NaN retries fired, and preprocessing/mask
constants are consistent across every consumer of each cache. The real exposure is
**forward-looking**: several auto-written documents whose verdict sentences don't depend on their
own data, a dead NaN guardrail, one latent wrong-obs plotting script, and a sampling loop that
costs ~10× what it should.

---

## (a) Example-disjointness — VERDICT: CLEAN

Split is by an explicit per-example `perm` feature filtered before shuffle
(`tfds_cross_tfdata_loader.py:98-103`; builder stores it at `tf_dataset_nbody_tomo_cross.py:113`),
with a hard runtime raise on overlap (`npe_cnn_nbody_tomo.py:1747-1768`, called at `:4474-4487`).
The old `imap_unordered` positional-slicing hazard does not apply (no percent-slice splits in this
route). Empirically confirmed on production artifacts: `cnn_train.npz` = 323,640 rows = 899 cosmos
× perms {5,6} × 180 patches; `cnn_val.npz` = 504,000 = 400 held-out cosmos × 7 perms × 180; zero
train↔val theta overlap; fiducial θ absent from the grid (min L∞ 0.051); shape-noise seeds
disjoint between eval obs (perms 0–49 → seeds 12345–12394) and all grid maps (≥12445). The G1
anchor obs enters training only via a plot path that is off in production.

Non-leak findings:
- **Noisy `best_val` selection** [low/methodological]: the compressor best-val checkpoint is chosen
  on a *single random 128-example val batch* per evaluation (`npe_cnn_nbody_tomo.py:3107,3138-3143`).
  Held-out cosmologies, so no leakage — but checkpoint selection is a high-variance estimator, and
  part of compressor seed-to-seed scatter is plausibly selection noise. Directly relevant to the
  "principled best-seed by val-loss" thread: the selection criterion itself needs de-noising
  (fixed, larger val subset) before it can rank seeds.
- **Latent noise-seed collision** [low]: seed = `12345 + 100*cosmo_idx + perm`
  (`build_full_sphere_cross_cache.py:207-209`); fiducial perms 100–106 would share seeds with grid
  cosmo-1 perms 0–6. Safe today (obs use perms 0–49); add an assert if the obs set ever grows.
- RMS sample includes NDE perms 5–6 (no perm filter at `npe_cnn_nbody_tomo.py:4261-4269`) — scalar
  divisors, can't transmit example identity; matches the locked train-sample convention.
- Cosmetic: sweep reads `fz["truth"]` but fidsumm writes `theta`
  (`population_sweep_flatsky.py:70` vs `build_fiducial_summaries_cnn.py:211`) → `per_patch_metrics.npz`
  always has empty truth; widths unaffected, future bias analyses must re-read theta.
- Perm-disjoint ≠ independent N-body realizations (CosmoGrid perms are shell-permuted lightcones
  from shared boxes) — the campaign's declared separation level, noted as a limitation.

## (b) RMS-whitening path — VERDICT: CLEAN (frozen by deterministic recomputation)

Within a run, one jitted transform feeds training, both compress passes, and the obs
(`npe_cnn_nbody_tomo.py:4286/4711/4956/4972/4342`). Cross-process, `build_fiducial_summaries_cnn.py:116-122`
*recomputes* the RMS, but the estimator is deterministic (first 8192 train examples, unshuffled,
fixed interleave; no RNG — `tfds_cross_tfdata_loader.py:92-107`, `npe_cnn_nbody_tomo.py:1863-1922`),
and the result is empirically bit-identical: training-log prints match fidsumm `channel_scale`
digit-for-digit for all 4 ops; vectors `np.array_equal` across compressor seeds. G1 (compare
recomputed obs summary vs `cnn_obs.npz`, rtol/atol 1e-3, hard abort — `build_fiducial_summaries_cnn.py:193-202`)
passed on all 8 jobs at max|Δ| ≤ 1.9e-4 and would catch any plausible different-sample RMS error
(≥0.5% → summary shift ≫ tolerance).

Findings:
- **The freeze is not persisted** [med/hardening]: `channel_scale` lives only in stdout logs and the
  fidsumm npz — not in `cnn_cache_meta.npz` (`build_cnn_cache_metadata`, `:2390-2469`) and the arms'
  `.meta.json` is never written under `--exit-after-compress`. Cache-metadata comparison therefore
  cannot detect a scale mismatch; G1 (an optional flag) is the only end-to-end guard. Fix: store
  `channel_scale` (or its sha) in the cache meta.
- **GATE A does not cover the RMS freeze** [low/writeup]: it validates operators + whitening
  linearity with its own n=2000 RMS (`gate_a_flat_cross_cnn.py:101-104`); attribute the freeze
  guarantee to estimator-determinism + G1, not GATE A. No GATE A log is persisted in the tree.
- Zero-mean ordering consistent everywhere (demean is baked into cache bytes at build time,
  `build_full_sphere_cross_cache.py:198-199`; `--zero-mean-maps` is label-only for this route).

## (c) MAF retraining / aggregation / silent fallbacks

Validated: stopping is jaxili-internal early stopping (patience 20, min_delta 1e-3) with
best-val-checkpoint restore, identical across arms and probes; `epochs=50000` is only a cap (L1
stops ~44–50 epochs, CNN ~494–789 — same criterion, data-driven). Obs preprocessing pins train
mean/std + train variance-mask (`population_sweep_flatsky.py:56-69`); per-probe constants
(CNN none/0/1e-12, L1 log1p-zscore/5/1e-5) identical across all consumers. FoM3/2D/σ math correct
(2D lacks the DETF constant — cancels in ratios). All 8 production sweeps: 1 training attempt per
seed, no retries, n=9000.

Findings (ranked):
1. **Dead NaN guard** [HIGH, guardrail]: `train_with_nan_retry` checks `hasattr(metrics,"train_loss")`
   but jaxili returns a dict keyed `"train/loss"` (`npe_l1norm_cross_jaxili_nbody_tomo.py:2366-2378`
   vs `jaxili/train.py:431`) — the NaN branch can never fire; only exceptions retry. A NaN-corrupted
   run would print "Training completed successfully." Didn't fire this campaign (verified), but it's
   a broken safety net for every future run.
2. **`make_headline_corner.py:60-63` conditions on the wrong obs** [HIGH if run / latent]: return-order
   misuse binds the "obs" to processed train row 0 and discards the real representative-patch summary.
   Its output figure does not exist anywhere and is unreferenced — never produced. Fix or delete
   before anyone runs it for the paper.
3. **Hardcoded verdicts in auto-written documents** [MED-HIGH, process]:
   - `run_multiseed_compressor_check.py:146-148` — writes "product/auto ≤ 1 across all compressor
     seeds ⇒ robust" unconditionally (even alongside a ⚠ FAILURES line). The in-flight run will
     stamp this verdict regardless of its own numbers; this session will recompute ratios from the
     `median_summary.json` files and correct the .md.
   - `consolidate_cnn_vs_l1.py:41-42,75-77,96-98,109-111` — "Calibrated ✓", "no-cross-gain survives
     un-pooling", "pass all three tests", "test has power" are asserted, not derived (all currently
     *true* against on-disk numbers — verified — but they would not flip if the data did).
   - `plot_best_seed.py:46` — figure title asserts the conclusion.
4. **`--decay-steps` is a no-op** [MED]: jaxili's `NPE.train` never forwards it
   (`jaxili/inference/npe.py:544-565`); the cosine horizon defaults to ~25,000 epochs → LR is
   effectively constant 1e-4 after warmup. Symmetric across arms (no comparison bias), but the CLI
   flag on every runner is cosmetic and the documented training schedule is wrong.
5. **"3 NDE seeds" vary only the train/val/test split** [MED, interpretation]: flow init uses
   jaxili's default trainer seed 42 and the loader shuffle is global-seed-42 for all "seeds"
   (`npe.py:214-227,560`). The MAF-seed scatter we quote is split-permutation variance only —
   seed-robustness claims are narrower than stated. Also: retries (if ever triggered) reuse the
   identical keys — a deterministic NaN would recur all 10 attempts.
6. **Per-obs failure handling is silent** [MED mechanism / low actual]: obs with <100 finite pooled
   samples are skipped, partial non-finite filtering is unrecorded, and `n` from
   `median_summary.json` is dropped by every downstream consumer (`consolidate_cnn_vs_l1.py:46-68`).
   n=9000 everywhere this campaign, so no effect — but arm-dependent failure rates would silently
   bias a future comparison. Record per-obs finite counts; print n in the result tables.
7. best_val→last_step fallback is stdout-loud but artifact-quiet [LOW-MED]: wandb records the
   *requested* policy; only the checkpoint filename in `compressor_params_path` disambiguates.
   Verified genuine best_val for all 4 arms this campaign.

## (d) Sampling throughput — mechanism found: un-jitted eager sampling

Measured from production logs (all 8 completed sweeps): sampling = 61–72% (CNN) / ~94% (L1) of
train+sample wall time, at 1.6–2.2 obs/s (~160–210 ms per `post.sample` call). Throughput is nearly
independent of conditioning dim (10 → 3200 costs only ~25% more despite ~300× FLOPs) ⇒
**host-dispatch-bound, not FLOPs-bound**. Cause: jaxili's sampling path has no jit anywhere
(`jaxili/posterior/direct_posterior.py:45-73`; eager MAF inversion loops over 5 layers × 6 dims =
~600 tiny dispatches per call, plus Flax `setup()` rebuilding + re-uploading MADE masks on every
call — `jaxili/model.py:708-757,541-621`). Metrics cost ~0.9 ms/obs (negligible).

Fix ladder (hypotheses with mechanisms; **no speedup numbers until measured** —
`feedback_benchmark_dont_assume`):
1. Jit a per-posterior sample closure (~10 lines in `population_sweep_flatsky.py:90-94`); preserves
   the exact per-obs `PRNGKey(seed*100003 + sel[i])` structure and RNG bits; 3 compiles/arm.
2. Chunked `jit(vmap(...))` over obs with stacked per-obs keys (~25 lines); B≈256 for dim-10,
   B≈16–32 for dim-3200 (memory).
3. Free riders: hoist the 3×-repeated per-obs `jnp.asarray(x_obs[i])`; drop per-call `np.asarray` syncs.

Adoption gate: bit-identity check eager-vs-jit-vs-vmap on 10 obs (same keys) + metric-level
equivalence on 50 obs + a 200-obs timed benchmark on GPU 1 when free; validate one full arm before
adopting. Note: jit may change FP summation order — keys, not bits, are the reproducibility
contract; renegotiate explicitly if bit-identity fails.

Correctness smells from this audit: latent PRNG key collision if a fiducial set reaches ≥100,003
rows (safe at 9000; assert it); sampling keys identical across arms (benign — paired noise — but
arm-vs-arm deltas are correlated); `marginal_stats`/`fom2d` consumed via dict insertion order
(`population_sweep_flatsky.py:99-100` — fragile); jaxili upstream: `MAFLayer.backward` returns a
wrong inverse log-det (`model.py:735` — harmless for sampling, landmine for future use);
`MaskedLinear` requests float64 params with x64 disabled → silently float32.

---

## Proposed fix list (awaiting Andreas's sign-off; nothing applied)

**P1 — before these scripts are reused for the paper:**
1. Make every auto-written verdict data-derived: `run_multiseed_compressor_check.py:146-148`,
   `consolidate_cnn_vs_l1.py` (4 sites), `plot_best_seed.py:46`. (This session will hand-correct the
   multiseed .md after the run lands regardless.)
2. Fix the dead NaN guard (`hasattr` → check `metrics["train/loss"]` finiteness) in
   `train_with_nan_retry`.
3. Fix or delete `make_headline_corner.py` (wrong-obs bug).

**P2 — cheap hardening:**
4. Persist `channel_scale` + effective checkpoint policy in `cnn_cache_meta.npz`.
5. Record per-obs finite-sample counts in `per_patch_metrics.npz`; surface `n` in result tables.
6. Asserts: fiducial rows < 100003 (key collision); `sel` monotonic.
7. Fix `truth`/`theta` key mismatch in the sweep.
8. Either pass decay_steps through to jaxili properly or drop the flag; document the real schedule.
9. De-noise compressor best_val selection (fixed val subset, e.g. 2048 examples, instead of one
   random 128-batch) — also a prerequisite for principled best-seed selection by val loss.
10. Optionally vary the jaxili trainer seed per NDE seed (real seed variance) — NOTE: changes
    numbers; methodology decision, not a bug fix.

**P3 — performance (needs GPU 1 free for the benchmark):**
11. Jit + chunked-vmap sampling in `population_sweep_flatsky.py` with the bit-identity gate above.
    Expected to matter a lot for the BNT campaign (every future arm currently pays ~75–95 min of
    pure dispatch overhead per sweep).
