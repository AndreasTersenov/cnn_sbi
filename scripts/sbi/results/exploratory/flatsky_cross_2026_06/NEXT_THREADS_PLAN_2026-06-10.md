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

## B. BNT for the flat-sky cross (calibrated BNT L1-vs-CNN)

**Design questions settled (proposal — confirm before build):**
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
4. **Campaign shape mirrors the no-BNT one** (so results are directly comparable): per probe
   {auto-only, +product} minimum (add conv/both only if asked — conv was the throwaway arm),
   1 compressor seed (41) + 3-MAF pooling + 9000-obs median sweep + GATE C (TARP/SBC; L-C2ST for
   CNN). GPU 1(+2 if granted); detached; phase-barriered orchestrator in the
   `run_multiseed_compressor_check.py` style with the **derived-verdict pattern** (no hardcoded
   conclusions) and per-phase failure propagation.

**Open question for Andreas:** the scientific target — is the BNT comparison about (i) whether
BNT changes the L1-vs-CNN ordering on the cross, or (ii) whether the cross-gain itself survives
nulling (the product channel is dominated by the same lens planes BNT reorganizes)? The minimal
4-arm matrix (L1/CNN × auto/product, all BNT) answers both at the headline level; (ii) would
additionally want the no-BNT product numbers re-quoted alongside.

**Cost estimate (grounded in measured numbers, this campaign):** per arm ≈ compressor 16 min +
fidsumm ~2 min + sweep ~100 min (current throughput; less with C adopted) for the CNN side; L1
arms have no compressor stage but pay the wavelet pass. 4 arms ≈ a day on GPU 1 including gates,
assuming no NaN surprises. Multi-day only if conv/both arms or extra seeds are added.

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

## Priority call (after the multiseed verdict)

*Pending — filled in when MULTISEED_COMPRESSOR_CHECK.md lands.*
