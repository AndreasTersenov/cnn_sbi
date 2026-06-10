# Plan — next threads after the multiseed compressor check (2026-06-10)

Scoping doc for Andreas's sign-off. Threads: (A) principled best-seed comparison, (B) BNT for the
flat-sky cross, (C) remaining audit fixes (batch B + throughput). Priorities and the possible
recipe-level test depend on the multiseed verdict — § "Priority call" is filled in once it lands.

---

## A. Principled best-seed comparison (CNN vs L1, best-to-best)

**Problem.** The current "best seed" (CNN 2620) is post-hoc selection by FoM3 — selection bias —
and is compared against L1-*pooled* (haircut), at a single obs. Two independent fixes are needed
before "best CNN beats L1" is a clean claim:

1. **Selection criterion.** Select each probe's best seed by a *held-out* criterion (val loss),
   never by the evaluation metric. Two distinct selection layers exist:
   - **Compressor seed** (CNN only): select on compressor VMIM val loss. The audit found the
     current per-save-point val loss is a single random 128-example batch (noisy). The new
     `--compressor-val-batches` flag de-noises future runs; for the three *existing* compressor
     seeds we can recompute a clean val loss post-hoc by evaluating the saved best_val checkpoints
     on a fixed 2048-example val slice (cheap, GPU-light, no retraining).
   - **NDE seed**: select on the jaxili early-stopping val loss (already recorded per checkpoint;
     readable from the ckpt metadata in the sweep dirs).
   Caveat to state in any writeup: within-architecture val-loss → FoM3 correlation is imperfect
   (observed: compressor s42 val −10.748 < s41 −10.722 yet FoM3 2170 < 2325), so "best-by-val"
   is the honest selection, not a guarantee of best FoM3.
2. **Matched aggregation.** Compare best-to-best at the same aggregation level: per-seed pooled
   9000-obs median (the multiseed check already produces exactly this for the CNN side).

**The L1 side is the cost.** L1's ~2000-d datavector cannot reload from jaxili checkpoints
(truncation bug), so per-NDE-seed L1 numbers need a retrain. The population-sweep machinery
already retrains 3 MAF seeds per arm — the small extension is dumping *per-seed* per-obs metrics
(currently only the 3-seed pool is recorded) in the same run. Cost: one rerun of the L1 sweeps
for the arms we care about (auto-only + product), ~2 h/arm at current throughput on one GPU —
or substantially less if the jit fix (C) is adopted first. **Recommendation: adopt C first, then
rerun the two L1 arms with per-seed dumps.**

Deliverable: a best-to-best table (selection by val loss, evaluation = pooled-median FoM3/σ/2D)
+ a short section in FLATSKY_CNN_RESULT.md replacing the current caveated best-seed section.

## B. BNT for the flat-sky cross (calibrated BNT L1-vs-CNN) — PAPER PILLAR 2

**Scientific target (Andreas, 2026-06-10).** BNT is a linear, invertible re-weighting across
tomographic bins that decorrelates the lensing-kernel overlap. Prior studies (and our own earlier
campaign) find higher-order stats computed on BNT maps **inflate** the contours — mechanism:
(i) lower per-map S/N in all but the first BNT bin, (ii) the originally independent white noise
becomes **correlated across bins** after the transform. Since BNT is invertible, the information
is still there; the inflation must come from the analysis not capturing the (now-crucial)
cross-bin correlations. The prediction ladder to test, each as an inflation ratio
`FoM3_BNT / FoM3_noBNT` (pooled 9000-obs median; σ/2D alongside) against the existing no-BNT
arms:

1. **L1 auto-only:** inflates significantly (per-channel statistic, blind to the correlated
   structure BNT creates).
2. **L1 auto+product:** inflates less (the explicit cross channel restores some cross-bin
   information) but still inflates.
3. **CNN:** (almost) no inflation — the bins enter as channels and VMIM should extract the
   cross-bin information implicitly ⇒ **BNT is lossless for a channel-mixing compressor**.

**Grounding from the prior campaign** (`results/final/paper_sbi_consolidation/…_advanced_cdim10_
long120k_v1/advanced_arch64_dense256_nostd_long/metrics.json`, 5 seeds, 20°/160px, different
setup with known bugs): BNT/no-BNT FoM3 = **0.85×** (std_sum +3.7%) — near-lossless, but it took
the bigger "advanced" compressor + 120k steps; the plain CNN was not enough. Expect the same
risk here: information in BNT space is less accessible (complex correlated noise), so prediction
3 may fail at the standard recipe **without falsifying losslessness** — the contingency ladder
below is part of the design, and the in-flight 160k recipe check tells us beforehand how much
the heavier recipe moves the plain CNN in no-BNT space.

**Design (proposal — confirm before build):**
1. **Order: noise → BNT → cross-build → whiten.** Shape noise is already injected at cache build
   (the TFDS autos carry it), matching the project invariant "noise before BNT". BNT is a linear
   per-pixel recombination across bins (κ̃ᵢ = Σⱼ Bᵢⱼ κⱼ) that a survey applies to its *observed*
   noisy auto maps; any physically-buildable cross-map is then built from the BNT'd autos. So:
   read autos ch 0–3 → apply B (4×4 matmul over the channel axis, on-device) → build
   conv/product cross channels from the BNT'd autos → per-channel RMS whitening (recomputed for
   the BNT'd channels — amplitudes change). The alternative (cross-build then BNT) is ill-defined
   for the product channels (the product of BNT'd maps is a *linear mix* of un-BNT'd products,
   not a relabeling) and doesn't correspond to anything a survey would do.
2. **Demeaning:** the cache autos are per-(patch,channel) demeaned; BNT of zero-mean maps is
   zero-mean, so no extra demean step is needed.
3. **Implementation surface is small:** one extra linear channel-mix step at the head of
   `make_flat_cross_transform` (JAX) and its torch/np twins in `flatsky_cross.py` (keep the three
   backends bit-matched; extend `gate_a_flat_cross_cnn.py` to cover the BNT path). The BNT matrix
   comes from `bnt_utils.py` (full tomo4 required; `validate_bnt_configuration`). L1 additionally
   needs its per-(channel,scale) noise σ re-frozen on BNT'd channels
   (`freeze_flatsky_cross_noise.py` rerun with the BNT transform).
4. **Implementation surface (concrete):** `bnt_utils.py` already holds the fixed
   `tomo4_bnt_v1` 4×4 matrix with numpy/TF appliers (channel-last; matrix depends only on the
   tomographic bins, so it carries to 10° unchanged). Add a JAX applier (one `tensordot`) at the
   head of `make_flat_cross_transform` behind a `--flatsky-bnt` flag, the numpy twin in
   `flatsky_cross.py` for the GATE-A oracle, and the same channel-mix in the L1 flat_local torch
   path. Extend `gate_a_flat_cross_cnn.py` with BNT cases (np-vs-jax bit-match, BNT'd autos
   preserved, whitening on BNT'd channels). **L1 noise σ must be re-frozen on BNT'd channels**
   (`freeze_flatsky_cross_noise.py` rerun with the BNT step in the pipeline — the noise-only
   realizations must pass through BNT too, since correlated post-BNT noise is the whole point).
   Per-channel RMS whitening needs no special handling (the deterministic estimator runs on the
   built BNT channels).
5. **Campaign shape:** 4 new arms — {L1, CNN} × {auto-only, auto+product}, all BNT — compared
   against the existing no-BNT arms (this campaign) for the inflation ratios. Mirror the no-BNT
   pipeline exactly: 1 compressor seed (41) + 3-MAF pooling + 9000-obs jit sweep + GATE C
   (TARP/SBC both probes; L-C2ST for CNN arms). Derived-verdict reporting throughout.
   **Robustness add-on given the multiseed lesson:** the CNN BNT arms should get the 2 extra
   compressor seeds (42, 43) for the headline inflation ratio — single-compressor-seed
   cross-claims are exactly what bit us this morning. (~+4 chains, cheap post-jit.)
6. **Contingency ladder if CNN-BNT inflates at the plain-CNN recipe** (expected risk — see
   grounding): (a) 160k + val-batches-16 recipe (informed by the in-flight recipe check);
   (b) the prior campaign's "advanced" arch (arch64/dense256 — exists in
   `--compressor-arch` options); (c) only then discuss capacity/objective changes. Each rung is
   a small set of chains at post-jit cost; decide rung-by-rung with Andreas.

**Cost (measured components, post-jit):** CNN chain ≈ 16 min compressor (80k; 32 min at 160k)
+ ~1 min fidsumm + ~30 min sweep ⇒ ~50–65 min/chain; L1 arm ≈ datavector build (~45 min,
loader-bound) + ~10 min sweep (fast NDE train + jit sampling). Minimal 4-arm matrix + gates ≈
one day on 2 GPUs; +4 CNN compressor-seed chains ≈ +4 h. Packing (Tier-1 scheduler) would
roughly halve the wall time once the interference numbers justify defaults.

## C. Remaining audit fixes (batch B — blocked on the multiseed driver exiting) + throughput

1. Dead NaN guard in `train_with_nan_retry` (check `metrics["train/loss"]`/`["val/loss"]`
   finiteness instead of the never-true `hasattr`).
2. `population_sweep_flatsky.py`: read `truth` with `theta` fallback; record per-obs finite-sample
   counts in `per_patch_metrics.npz`; assert `len(sel) < 100003` (PRNG key collision headroom);
   index `marginal_stats`/`fom2d` outputs by key, not dict order; stop passing the inert
   `decay_steps` (or comment it as cosmetic) so the documented schedule is honest.
3. **Jit sampling** (`bench_sample_jit.py` is ready): run the benchmark on GPU 1 when free →
   if the bit-identity gate passes and the speedup is real, wire the jitted closure into
   `population_sweep_flatsky.py` (per-posterior `jax.jit`, identical per-obs keys; chunked vmap
   only if the plain jit win is insufficient). All future sweeps (incl. A and B above) inherit it.

## Priority call (multiseed verdict in, 2026-06-10 ~13:30)

The check landed in the **mixed branch**: product/auto flips sign with the compressor draw
(0.94/1.10/0.98, mean 1.00×) ⇒ the CNN's cross effect is zero-within-seed-noise, NOT a robust
loss; but every CNN product seed stays ≤ 0.85× of L1 product, and VMIM val losses show no extra
MI extracted from the product channel at this recipe. Consequences for the threads:

1. **Throughput first (C)** — jit benchmark + packing measurements + Tier-1 scheduler. Cheap,
   and halves the cost of everything below. → benchmark running now; scheduler diff after the
   packing numbers, with sign-off.
2. **BNT campaign (B)** — the main remaining scientific deliverable; start once C lands
   (design in §B above needs Andreas's confirmation on the open question).
3. **Recipe-level CNN test — LAUNCHED 2026-06-10 ~15:00** (Andreas: "start with 2"):
   `run_recipe_160k_check.py`, {none, product} × seeds {42, 43} at 160k steps +
   `--compressor-val-batches 16`, GPUs 1+2, paired against the 80k multiseed results →
   `cnn_phase/multiseed_160k/RECIPE_160K_CHECK.md`. Doubles as the recipe calibration for the
   BNT contingency ladder. NB the recipe bundles two changes (2× steps + de-noised best_val);
   ablate before attributing any movement.
4. **Principled best-seed (A) — SKIPPED** (Andreas, 2026-06-10): the multiseed check already
   provides matched-aggregation per-compressor-seed numbers; auto-only remains a tie under
   val-loss selection (s43: 2480 vs L1 2405). No L1 per-NDE-seed retrain.
